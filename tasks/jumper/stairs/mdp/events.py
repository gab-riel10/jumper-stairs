"""Where an episode starts: on the landing, or partway up the flight.

`reset_scene_to_default` has already put the robot at HOME on its tile's spawn
origin, the landing's centre. This event moves a share of the robots up the
stairs -- standing on tread `k`, facing up the flight, with a little jitter --
so that the upper treads are practised before the policy can reach them from
the bottom, the way `jumper.ref_free_jump` spawns on the recording to practise
the landing before it can jump. The rest start where the deployment will, on
the landing.

It records the tread in `env._stairs_spawn_step`, which the command term reads
at its own reset (events run before `command_manager.reset()` in
`ManagerBasedRlEnv._reset_idx`, the same ordering `jumper.jump`'s RSI event
relies on) -- the sites are not forwarded until after the reset, so the command
cannot read the tread off the body yet.
"""

from __future__ import annotations

import math

import torch

from mjlab.entity import Entity

from .terrain import Staircase


def reset_on_tread(
    env,
    env_ids: torch.Tensor | None,
    stairs: Staircase,
    elevated_fraction: float = 0.5,
    max_start_step: int | None = None,
    xy_jitter: float = 0.05,
    yaw_jitter: float = math.radians(10.0),
) -> None:
    """Spawn `elevated_fraction` of `env_ids` on a uniformly chosen tread in
    `1..max_start_step`, the rest on the landing (tread 0); all facing +x."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    n = len(env_ids)
    if n == 0:
        return
    robot: Entity = env.scene["robot"]
    dev = env.device
    top = stairs.num_steps - 1 if max_start_step is None else max_start_step
    top = max(0, min(top, stairs.num_steps))

    r = torch.rand(n, device=dev)
    k = torch.randint(1, top + 1, (n,), device=dev) if top >= 1 else torch.zeros(n, dtype=torch.long, device=dev)
    k = torch.where(r < elevated_fraction, k, torch.zeros_like(k))

    if not hasattr(env, "_stairs_spawn_step"):
        env._stairs_spawn_step = torch.zeros(env.num_envs, dtype=torch.long, device=dev)
    env._stairs_spawn_step[env_ids] = k

    # The riser of each environment's flight, from its terrain row.
    terrain = env.scene.terrain
    riser = stairs.riser_of_level(terrain.terrain_levels[env_ids], terrain.cfg.terrain_generator)

    origin = env.scene.env_origins[env_ids]
    default = robot.data.default_root_state[env_ids].clone()
    # x: the landing's centre for k = 0, the middle of tread k otherwise.
    on_tread = stairs.first_riser_from_origin + (k.float() - 0.5) * stairs.tread
    x = torch.where(k > 0, on_tread, torch.zeros_like(on_tread))
    jitter = (torch.rand(n, 2, device=dev) * 2.0 - 1.0) * xy_jitter
    default[:, 0] = origin[:, 0] + x + jitter[:, 0]
    default[:, 1] = origin[:, 1] + jitter[:, 1]
    default[:, 2] = origin[:, 2] + k.float() * riser + robot.data.default_root_state[env_ids, 2]
    yaw = (torch.rand(n, device=dev) * 2.0 - 1.0) * yaw_jitter
    default[:, 3] = torch.cos(0.5 * yaw)
    default[:, 4] = 0.0
    default[:, 5] = 0.0
    default[:, 6] = torch.sin(0.5 * yaw)
    default[:, 7:] = 0.0
    robot.write_root_state_to_sim(default, env_ids=env_ids)


__all__ = ["reset_on_tread"]
