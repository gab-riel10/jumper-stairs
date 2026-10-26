"""Move each robot to a taller flight when it has climbed this one by jumping.

The shape is `common/mdp/curriculum.py::TerrainLevels`' -- a class, so the
per-environment dwell counter survives between calls; promotion gated on
evidence and on time at the level, demotion on evidence alone -- with the
evidence changed to what this task is about. A robot is promoted when, in the
episode just ended, its rear feet gained at least `promote_steps` treads
**in flight** (`steps_by_flight`, from the command term) and it did not fall;
it is demoted when it fell, or gained nothing in flight in an episode long
enough to have tried. Crawled treads count for nothing either way, so a policy
that walks up never earns a taller riser to walk up.

Called once per reset batch from `ManagerBasedRlEnv._reset_idx`, **before** the
command manager zeroes its metrics, which is why the episode's numbers are
still there to read. `env.reset_terminated` says which of `env_ids` ended by a
termination that was not a time-out, i.e. `fell_over` or `off_side`.

The dwell is in episodes at the level, counted here, for the reason
`TerrainLevels` gives at length (a count against `common_step_counter` did not
survive `--resume`).
"""

from __future__ import annotations

import torch


class StairsLevels:
    def __init__(self, cfg, env) -> None:
        del cfg, env
        self._episodes_here = None

    def __call__(
        self,
        env,
        env_ids: torch.Tensor,
        command_name: str = "stairs",
        promote_steps: float = 2.0,
        dwell_episodes: int = 6,
        min_steps: int = 200,
    ) -> dict:
        terrain = getattr(env.scene, "terrain", None)
        if terrain is None or getattr(terrain.cfg, "terrain_generator", None) is None:
            return {}
        if isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
        if self._episodes_here is None:
            self._episodes_here = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._episodes_here[env_ids] += 1

        cmd = env.command_manager.get_term(command_name)
        jumped = cmd.metrics["steps_by_flight"][env_ids]
        fell = env.reset_terminated[env_ids] if hasattr(env, "reset_terminated") else torch.zeros_like(jumped, dtype=torch.bool)
        long_enough = env.episode_length_buf[env_ids] >= min_steps
        settled = self._episodes_here[env_ids] >= dwell_episodes

        move_up = settled & long_enough & ~fell & (jumped >= promote_steps)
        move_down = fell | (long_enough & (jumped <= 0.0) & ~move_up)

        # The first reset happens before any stepping; nothing has been shown.
        if env.common_step_counter == 0:
            move_up = torch.zeros_like(move_up)
            move_down = torch.zeros_like(move_down)

        changed = move_up | move_down
        if bool(changed.any()):
            self._episodes_here[env_ids[changed]] = 0
        terrain.update_env_origins(env_ids, move_up, move_down)

        levels = terrain.terrain_levels.float()
        riser = cmd.stairs.riser_of_level(terrain.terrain_levels, terrain.cfg.terrain_generator)
        return {
            "level": float(levels.mean()),
            "level_max": float(levels.max()),
            "riser_m": float(riser.mean()),
            "promoted": float(move_up.float().mean()),
            "demoted": float(move_down.float().mean()),
            "settled": float(settled.float().mean()),
        }


__all__ = ["StairsLevels"]
