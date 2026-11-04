"""Move each robot to a taller flight when it has climbed this one by jumping.

The shape is `common/mdp/curriculum.py::TerrainLevels`' -- a class, so the
per-environment counters survive between calls -- with the evidence changed to
what this task is about, and the rule Rudin et al. 2021 use for their terrain
curriculum (promote on progress, demote on the lack of it, both read off the
episode just ended) rather than one that waits for the robot to settle first.

A robot is promoted when its base gained at least `promote_steps` treads
**in flight** (`steps_by_flight`, from the command term) over its last
`window` episodes at this level, and the episode just ended was not a fall.
It is demoted after `demote_falls` falls in a row, or `demote_zeros` episodes
in a row that were long enough to have tried and gained nothing in flight.
Crawled treads count for nothing either way, so a policy that walks up never
earns a taller riser to walk up -- except with `count_crawled`, which
`jumper.stairs_free` passes because there a tread is a tread.

The counters are cleared only when the level really changes. The first
version cleared them on every demotion as well, and a demotion is asked for
by any episode with nothing gained in flight -- also at level 0, where it
cannot take effect. In 10 h on Kaggle (2026-10-27, 2048 envs, T4) the policy
sat at 1.0-1.5 treads an episode, alternating episodes with 0 and 2, so the
six-episode dwell never completed, `settled` stayed at 0 and not one robot
was promoted: the riser never left 2 cm. That run says nothing about the
robot; it measured the gate.

## Across a resume

`TerrainLevels` deliberately does not checkpoint its rows (one per robot, and
the robot count may change). This one does, because here the rows *are* the
curriculum: every resume of the stairs before 2026-11-04 started every robot
again at row 0 -- `model_6750`'s and `model_10650`'s checkpoints carry
`common_step_counter` and nothing else -- and spent the first hour or two
climbing back to where the policy already was. So this is a
`CheckpointedLadder`, found and saved by `common/runner.py::CurriculumRunner`
(`rl_cfg.py` returns it): `state_dict` is every robot's row and the counters
beside it; `load_state_dict` puts them back when the robot count matches and
otherwise draws each robot's row from the saved rows' distribution, which
keeps the mix of risers the policy was training on. A checkpoint that carries
no rows (the two above) starts every robot on a row drawn from
`$MJRL_STAIRS_START_ROWS` ("3-7": rows 3 to 7 inclusive, uniform) when that
is set, and on row 0 otherwise.

Called once per reset batch from `ManagerBasedRlEnv._reset_idx`, **before** the
command manager zeroes its metrics, which is why the episode's numbers are
still there to read. `env.reset_terminated` says which of `env_ids` ended by a
termination that was not a time-out, i.e. `fell_over`, `fell_off`, `off_side`
or `stall`. The counters are per episode, counted here, for the reason
`TerrainLevels` gives at length (a count against `common_step_counter` did not
survive `--resume`).
"""

from __future__ import annotations

import os

import torch

from ...common.mdp.curriculum import CheckpointedLadder

#: Rows a resumed checkpoint without rows starts on, "lo-hi" inclusive.
START_ROWS_ENV = "MJRL_STAIRS_START_ROWS"


def _start_rows(top: int) -> tuple[int, int] | None:
    raw = (os.environ.get(START_ROWS_ENV) or "").strip()
    if not raw:
        return None
    lo, _, hi = raw.partition("-")
    try:
        lo_i, hi_i = int(lo), int(hi or lo)
    except ValueError:
        raise SystemExit(f"[stairs] ${START_ROWS_ENV}={raw!r}: expected lo-hi, e.g. 3-7") from None
    lo_i, hi_i = max(0, min(lo_i, top)), max(0, min(hi_i, top))
    if hi_i < lo_i:
        raise SystemExit(f"[stairs] ${START_ROWS_ENV}={raw!r}: hi below lo")
    return lo_i, hi_i


class StairsLevels(CheckpointedLadder):
    LEVEL_ENV = START_ROWS_ENV
    EVIDENCE = ()

    def __init__(self, cfg, env) -> None:
        del cfg
        self._env = env
        self._episodes_here = None  # episodes since the level last changed
        self._window = None  # treads gained in flight, one column per recent episode
        self._falls = None  # falls in a row
        self._zeros = None  # long, fall-free episodes in a row with nothing gained in flight
        self.level = 0
        self.last_promotion_step = 0

    def _allocate(self, env, window: int) -> None:
        n, dev = env.num_envs, env.device
        self._episodes_here = torch.zeros(n, dtype=torch.long, device=dev)
        self._window = torch.zeros(n, window, device=dev)
        self._falls = torch.zeros(n, dtype=torch.long, device=dev)
        self._zeros = torch.zeros(n, dtype=torch.long, device=dev)

    # ── Across a resume (`common/runner.py`) ───────────────────────────────
    def _terrain(self):
        terrain = getattr(self._env.scene, "terrain", None)
        if terrain is None or getattr(terrain, "terrain_origins", None) is None:
            return None
        return terrain

    def _set_rows(self, rows: torch.Tensor) -> None:
        """Write every robot's row into the terrain, origins included. Robots
        move to the new row at their next reset; until then the command term
        reads the new riser over the old geometry, for one episode at most."""
        terrain = self._terrain()
        if terrain is None:
            return
        rows = rows.to(terrain.terrain_levels.device, dtype=torch.long)
        terrain.terrain_levels.copy_(rows)
        terrain.env_origins.copy_(terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types])
        self.level = int(rows.float().mean().round())

    def state_dict(self) -> dict:
        terrain = self._terrain()
        rows = terrain.terrain_levels.clone() if terrain is not None else torch.zeros(0, dtype=torch.long)
        return {
            "level": int(rows.float().mean().round()) if rows.numel() else 0,
            "rows": rows.cpu(),
            "window": None if self._window is None else self._window.cpu(),
            "falls": None if self._falls is None else self._falls.cpu(),
            "zeros": None if self._zeros is None else self._zeros.cpu(),
            "episodes_here": None if self._episodes_here is None else self._episodes_here.cpu(),
        }

    def load_state_dict(self, state: dict | None, step: int) -> str:
        del step
        terrain = self._terrain()
        if terrain is None:
            return "no terrain generator: nothing to restore"
        n = terrain.terrain_levels.numel()
        top = max(int(terrain.terrain_origins.shape[0]) - 1, 0)
        if state is None or state.get("rows") is None or state["rows"].numel() == 0:
            span = _start_rows(top)
            if span is None:
                return f"rows from the default (row 0); the checkpoint carries none (set ${START_ROWS_ENV})"
            lo, hi = span
            rows = torch.randint(lo, hi + 1, (n,), device=terrain.terrain_levels.device)
            self._set_rows(rows)
            return f"rows {lo}-{hi} from ${START_ROWS_ENV}; the checkpoint carries none"
        saved = state["rows"].to(dtype=torch.long).clamp(0, top)
        if saved.numel() == n:
            self._set_rows(saved)
            if self._window is not None and state.get("window") is not None and state["window"].shape == self._window.shape:
                self._window.copy_(state["window"].to(self._window.device))
                self._falls.copy_(state["falls"].to(self._falls.device))
                self._zeros.copy_(state["zeros"].to(self._zeros.device))
                self._episodes_here.copy_(state["episodes_here"].to(self._episodes_here.device))
            return f"rows from the checkpoint (mean {saved.float().mean():.2f}, max {int(saved.max())})"
        pick = torch.randint(0, saved.numel(), (n,))
        self._set_rows(saved[pick])
        return (f"rows drawn from the checkpoint's {saved.numel()} robots' rows for {n} robots "
                f"(mean {saved.float().mean():.2f}); the counters restart")

    # ── Per reset batch ─────────────────────────────────────────────────────
    def __call__(
        self,
        env,
        env_ids: torch.Tensor,
        command_name: str = "stairs",
        promote_steps: float = 2.0,
        window: int = 4,
        demote_falls: int = 2,
        demote_zeros: int = 4,
        min_steps: int = 200,
        count_crawled: bool = False,
    ) -> dict:
        terrain = getattr(env.scene, "terrain", None)
        if terrain is None or getattr(terrain.cfg, "terrain_generator", None) is None:
            return {}
        if isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
        if self._window is None or self._window.shape[1] != window:
            self._allocate(env, window)

        cmd = env.command_manager.get_term(command_name)
        levels = terrain.terrain_levels.float()
        riser = cmd.stairs.riser_of_level(terrain.terrain_levels, terrain.cfg.terrain_generator)
        # The first reset happens before any stepping; nothing has been shown.
        # A resume's restore passes no environments: nothing to judge either.
        if env.common_step_counter == 0 or len(env_ids) == 0:
            return {"level": float(levels.mean()), "level_max": float(levels.max()),
                    "riser_m": float(riser.mean()), "promoted": 0.0, "demoted": 0.0,
                    "recent": 0.0}

        jumped = cmd.metrics["steps_by_flight"][env_ids]
        if count_crawled:  # `jumper.stairs_free`: a tread is progress however it was gained
            jumped = jumped + cmd.metrics["steps_crawled"][env_ids]
        if hasattr(env, "reset_terminated"):
            fell = env.reset_terminated[env_ids]
        else:
            fell = torch.zeros_like(jumped, dtype=torch.bool)
        long_enough = env.episode_length_buf[env_ids] >= min_steps

        self._episodes_here[env_ids] += 1
        self._window[env_ids] = torch.roll(self._window[env_ids], 1, dims=1)
        self._window[env_ids, 0] = jumped
        self._falls[env_ids] = torch.where(fell, self._falls[env_ids] + 1, torch.zeros_like(self._falls[env_ids]))
        zero = long_enough & ~fell & (jumped <= 0.0)
        self._zeros[env_ids] = torch.where(
            zero, self._zeros[env_ids] + 1,
            torch.where(jumped > 0.0, torch.zeros_like(self._zeros[env_ids]), self._zeros[env_ids]),
        )
        recent = self._window[env_ids].sum(dim=1)

        move_up = ~fell & (recent >= promote_steps)
        move_down = ((self._falls[env_ids] >= demote_falls) | (self._zeros[env_ids] >= demote_zeros)) & ~move_up

        before = terrain.terrain_levels[env_ids].clone()
        terrain.update_env_origins(env_ids, move_up, move_down)
        changed = terrain.terrain_levels[env_ids] != before
        if bool(changed.any()):
            moved = env_ids[changed]
            self._episodes_here[moved] = 0
            self._window[moved] = 0.0
            self._falls[moved] = 0
            self._zeros[moved] = 0

        levels = terrain.terrain_levels.float()
        riser = cmd.stairs.riser_of_level(terrain.terrain_levels, terrain.cfg.terrain_generator)
        self.level = int(levels.mean().round())
        return {
            "level": float(levels.mean()),
            "level_max": float(levels.max()),
            "riser_m": float(riser.mean()),
            "promoted": float(move_up.float().mean()),
            "demoted": float(move_down.float().mean()),
            "recent": float(recent.mean()),
        }


__all__ = ["START_ROWS_ENV", "StairsLevels"]
