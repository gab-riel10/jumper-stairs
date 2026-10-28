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

Called once per reset batch from `ManagerBasedRlEnv._reset_idx`, **before** the
command manager zeroes its metrics, which is why the episode's numbers are
still there to read. `env.reset_terminated` says which of `env_ids` ended by a
termination that was not a time-out, i.e. `fell_over`, `fell_off` or
`off_side`. The counters are per episode, counted here, for the reason
`TerrainLevels` gives at length (a count against `common_step_counter` did not
survive `--resume`).
"""

from __future__ import annotations

import torch


class StairsLevels:
    def __init__(self, cfg, env) -> None:
        del cfg, env
        self._episodes_here = None  # episodes since the level last changed
        self._window = None  # treads gained in flight, one column per recent episode
        self._falls = None  # falls in a row
        self._zeros = None  # long, fall-free episodes in a row with nothing gained in flight

    def _allocate(self, env, window: int) -> None:
        n, dev = env.num_envs, env.device
        self._episodes_here = torch.zeros(n, dtype=torch.long, device=dev)
        self._window = torch.zeros(n, window, device=dev)
        self._falls = torch.zeros(n, dtype=torch.long, device=dev)
        self._zeros = torch.zeros(n, dtype=torch.long, device=dev)

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
        jumped = cmd.metrics["steps_by_flight"][env_ids]
        if count_crawled:  # `jumper.stairs_free`: a tread is progress however it was gained
            jumped = jumped + cmd.metrics["steps_crawled"][env_ids]
        if hasattr(env, "reset_terminated"):
            fell = env.reset_terminated[env_ids]
        else:
            fell = torch.zeros_like(jumped, dtype=torch.bool)
        long_enough = env.episode_length_buf[env_ids] >= min_steps

        # The first reset happens before any stepping; nothing has been shown.
        if env.common_step_counter == 0:
            levels = terrain.terrain_levels.float()
            riser = cmd.stairs.riser_of_level(terrain.terrain_levels, terrain.cfg.terrain_generator)
            return {"level": float(levels.mean()), "level_max": float(levels.max()),
                    "riser_m": float(riser.mean()), "promoted": 0.0, "demoted": 0.0,
                    "recent": 0.0}

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
        return {
            "level": float(levels.mean()),
            "level_max": float(levels.max()),
            "riser_m": float(riser.mean()),
            "promoted": float(move_up.float().mean()),
            "demoted": float(move_down.float().mean()),
            "recent": float(recent.mean()),
        }


__all__ = ["StairsLevels"]
