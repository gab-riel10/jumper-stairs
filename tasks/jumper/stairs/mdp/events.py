"""Where an episode starts: on the landing, partway up the flight at HOME, or
partway up the flight in the state a landing leaves.

`reset_scene_to_default` has already put the robot at HOME on its tile's spawn
origin, the landing's centre. This event moves a share of the robots up the
stairs -- standing on tread `k`, facing up the flight, with a little jitter --
so that the upper treads are practised before the policy can reach them from
the bottom, the way `jumper.ref_free_jump` spawns on the recording to practise
the landing before it can jump. The rest start where the deployment will, on
the landing.

A second share starts **as if it had just landed** on tread `k`: the base's
pose relative to the tread, its velocities, and every joint's position and
velocity come from a bank of states recorded from a trained policy 0.3-0.6 s
after a real touchdown (`tools/stairs_landing_bank.py`). Measured on
`model_6750` and `model_10650` (2026-11-03): from a HOME spawn on a tread they
jump at once; after a landing they never jump again. The landing is the state
the second jump has to start from, and until now it was the one state the
policy never started an episode in. The observation history fills with the
first frame at a reset (`posture/mdp/history.py`), so a bank spawn reads as a
robot that has just come to rest in that pose, not as a fresh HOME.

It records the tread in `env._stairs_spawn_step`, which the command term reads
at its own reset (events run before `command_manager.reset()` in
`ManagerBasedRlEnv._reset_idx`, the same ordering `jumper.jump`'s RSI event
relies on) -- the sites are not forwarded until after the reset, so the command
cannot read the tread off the body yet.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from mjlab.entity import Entity

from .terrain import Staircase


def _load_bank(env, path: Path, robot: Entity) -> dict[str, torch.Tensor]:
    """The bank's arrays as tensors on the env's device, loaded once."""
    cache = getattr(env, "_stairs_landing_bank", None)
    if cache is not None and cache["path"] == str(path):
        return cache
    if not path.is_file():
        raise FileNotFoundError(
            f"[stairs] no landing bank at {path}; record one with tools/stairs_landing_bank.py "
            f"or set LANDED_FRACTION to 0 in env_cfg.py"
        )
    with np.load(path) as data:
        names = [str(n) for n in data["joint_names"]]
        if names != list(robot.joint_names):
            raise ValueError(f"[stairs] the landing bank's joint order {names[:3]}... is not this "
                             f"robot's {list(robot.joint_names)[:3]}...")
        cache = {k: torch.as_tensor(data[k], dtype=torch.float32, device=env.device)
                 for k in ("rel_pos", "quat_wxyz", "lin_vel", "ang_vel", "joint_pos", "joint_vel")}
    cache["path"] = str(path)
    env._stairs_landing_bank = cache
    return cache


def reset_on_tread(
    env,
    env_ids: torch.Tensor | None,
    stairs: Staircase,
    elevated_fraction: float = 0.25,
    landed_fraction: float = 0.0,
    bank: str | Path | None = None,
    max_start_step: int | None = None,
    xy_jitter: float = 0.05,
    yaw_jitter: float = math.radians(10.0),
) -> None:
    """Spawn `landed_fraction` of `env_ids` in a recorded post-landing state on
    a uniformly chosen tread in `1..max_start_step`, `elevated_fraction` of
    them at HOME on such a tread, the rest at HOME on the landing (tread 0);
    all facing +x. The two fractions are of the whole batch and add up to at
    most 1."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    n = len(env_ids)
    if n == 0:
        return
    if landed_fraction + elevated_fraction > 1.0 + 1e-9:
        raise ValueError(f"landed_fraction {landed_fraction} + elevated_fraction {elevated_fraction} > 1")
    robot: Entity = env.scene["robot"]
    dev = env.device
    top = stairs.num_steps - 1 if max_start_step is None else max_start_step
    top = max(0, min(top, stairs.num_steps))

    r = torch.rand(n, device=dev)
    landed = (r < landed_fraction) & (top >= 1) & (bank is not None)
    elevated = (r >= landed_fraction) & (r < landed_fraction + elevated_fraction)
    k = torch.randint(1, top + 1, (n,), device=dev) if top >= 1 else torch.zeros(n, dtype=torch.long, device=dev)
    k = torch.where(landed | elevated, k, torch.zeros_like(k))

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

    if bool(landed.any()):
        states = _load_bank(env, Path(bank), robot)
        m = int(landed.sum())
        pick = torch.randint(0, states["rel_pos"].shape[0], (m,), device=dev)
        rel = states["rel_pos"][pick]
        sel = landed.nonzero(as_tuple=False).flatten()
        default[sel, 0] = origin[sel, 0] + on_tread[sel] + rel[:, 0]
        default[sel, 1] = origin[sel, 1] + rel[:, 1]
        default[sel, 2] = origin[sel, 2] + k[sel].float() * riser[sel] + rel[:, 2]
        default[sel, 3:7] = states["quat_wxyz"][pick]
        default[sel, 7:10] = states["lin_vel"][pick]
        default[sel, 10:13] = states["ang_vel"][pick]
        robot.write_joint_state_to_sim(states["joint_pos"][pick], states["joint_vel"][pick],
                                       env_ids=env_ids[sel])
    robot.write_root_state_to_sim(default, env_ids=env_ids)


__all__ = ["reset_on_tread"]
