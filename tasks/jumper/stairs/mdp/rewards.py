"""Rewards for the stairs. Every quantity here is read off the command term
(`mdp/commands.py`), which computed it once, after the previous step; none of
these functions keeps state of its own.

Two groups. **What pays**: treads gained in flight, as an impulse at the
touchdown and as a stream for the rest of the episode; height and forward
speed while in the air; being upright; settling towards HOME on the ground.
**What costs**: treads gained on the ground, which is the crawl this task
exists to rule out; every step, a little, so that standing still is not the
safest policy; any part that is not a foot on the ground; legs on each other;
drifting sideways or turning away from the flight; and the joint-limit and
action-rate terms every task on this robot carries.

The weights are in `env_cfg.py`, with what each one was set against.
"""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg


def _cmd(env, command_name: str):
    return env.command_manager.get_term(command_name)


def step_up_in_flight(env, command_name: str = "stairs") -> torch.Tensor:
    """New treads -- beyond the episode's high-water mark -- the base was over
    at the touchdown of a jump, paid once. Never negative, never twice."""
    return _cmd(env, command_name).jump_gain


def flight_progress(env, command_name: str = "stairs") -> torch.Tensor:
    """Metres of new ground the base landed on at the touchdown of a jump,
    beyond the furthest it has been this episode; paid once. The dense half of
    `step_up_in_flight`: a jump that drifts 2 cm further than any before pays
    for the 2 cm, where the tread credit waits for a riser line. Mount it at
    the tread credit's rate per tread (`PROGRESS_PER_M`)."""
    return _cmd(env, command_name).progress_gain


def landing_centre(env, command_name: str = "stairs") -> torch.Tensor:
    """At the touchdown of a jump that gained a tread, exp(-(dx/sigma)^2), dx
    the base's distance from the tread's centre; paid once. A robot nearly as
    long as the tread fits on it only near the middle, and lands from there
    into the next crouch."""
    return _cmd(env, command_name).landing_score


def feet_near_edge(env, command_name: str = "stairs") -> torch.Tensor:
    """Grounded feet within a few centimetres of a riser line -- a tread's nose
    or the foot of the next riser -- every step. Mount negative: the edge term
    of Extreme Parkour (Cheng et al. 2023, arXiv 2309.14341), along x only
    because every edge here runs across the flight."""
    return _cmd(env, command_name).edge_feet


def climbed_stream(env, command_name: str = "stairs") -> torch.Tensor:
    """Treads earned by jumping so far this episode, paid every step. Monotone:
    nothing walked back down is refunded, nothing crawled is counted."""
    return _cmd(env, command_name).climbed_net


def crawl(env, command_name: str = "stairs") -> torch.Tensor:
    """New treads the base reached with a foot on the ground (or at the
    touchdown of a hop), this step. Charged once and never paid afterwards.
    Mount it with a negative weight -- or, in `jumper.stairs_free`, with the
    jump's weight: there how a tread is gained is the robot's business."""
    return _cmd(env, command_name).crawl_gain


def reached_top(env, command_name: str = "stairs") -> torch.Tensor:
    """1 on the step the robot has stood on the top platform for `settle_steps`
    -- the step the `reached_top` termination ends the episode on, so it is
    paid once. The position-based framing of blind stair climbing (Ascento,
    arXiv 2402.06143): the flight is the goal, not the treads."""
    return _cmd(env, command_name).arrived.float()


def flight_height(env, command_name: str = "stairs", scale: float = 0.10) -> torch.Tensor:
    """How far the lowest point is above the takeoff tread, in units of `scale`,
    while in the air. The dense half of the jump: a higher arc pays before the
    landing does."""
    cmd = _cmd(env, command_name)
    rise = cmd.lowest_rise().clamp(min=0.0) / scale
    return rise * cmd.in_flight.float()


def flight_forward(env, command_name: str = "stairs", scale: float = 1.0) -> torch.Tensor:
    """Forward speed of the base along the flight (+x of the world, which is
    the way every flight runs), in units of `scale` m/s, while in the air."""
    cmd = _cmd(env, command_name)
    robot: Entity = env.scene["robot"]
    vx = robot.data.root_link_lin_vel_w[:, 0] / scale
    return vx * cmd.in_flight.float()


def upright(env, sigma_sq: float = 0.10) -> torch.Tensor:
    """exp(-|g_xy|^2 / sigma_sq), g the gravity direction in the base frame."""
    robot: Entity = env.scene["robot"]
    g = robot.data.projected_gravity_b[:, :2]
    return torch.exp(-(g**2).sum(dim=1) / sigma_sq)


def settle_pose(
    env, asset_cfg: SceneEntityCfg, command_name: str = "stairs", sigma_sq: float = 0.05
) -> torch.Tensor:
    """Closeness to HOME while on the ground: the pose the next jump starts from,
    and the one the deployment hands over at. Nothing is asked of the pose in
    the air. `asset_cfg` has no default so that the manager resolves the joint
    ids (see `jumper.ref_free_jump`'s `home_pose`)."""
    cmd = _cmd(env, command_name)
    robot: Entity = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    home = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    e = ((q - home) ** 2).mean(dim=1)
    return torch.exp(-e / sigma_sq) * (~cmd.in_flight).float()


def time_cost(env) -> torch.Tensor:
    """1 every step. Mount it with a negative weight: the price of doing nothing."""
    return torch.ones(env.num_envs, device=env.device)


def lateral_drift(env, scale: float = 0.20) -> torch.Tensor:
    """(y / scale)^2, y the base's offset from the flight's centreline (the
    spawn origin's y)."""
    robot: Entity = env.scene["robot"]
    d = robot.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return (d / scale) ** 2


def heading(env) -> torch.Tensor:
    """1 - cos(yaw), yaw against +x, the way up the flight. Mount negative."""
    robot: Entity = env.scene["robot"]
    q = robot.data.root_link_quat_w  # (w, x, y, z)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    sin_yaw = 2.0 * (w * z + x * y)
    yaw = torch.atan2(sin_yaw, cos_yaw)
    return 1.0 - torch.cos(yaw)


def illegal_contact(
    env, sensor_name: str = "body_ground_contact", force_threshold: float = 1.0
) -> torch.Tensor:
    """1 while any part that is not a foot is on the ground. Binary, as
    `jumper.ref_free_jump` argues: a knee on a riser is as illegal as the belly."""
    sensor = env.scene[sensor_name]
    f = sensor.data.force
    assert f is not None
    return (f.norm(dim=-1) > force_threshold).any(dim=1).float()


def leg_collision(
    env, sensor_name: str = "self_collision", force_threshold: float = 1.0
) -> torch.Tensor:
    """1 in a step in which two legs pushed on each other in any physics
    substep. The collision masks (`common/constants.py`) leave only different
    legs able to touch."""
    sensor = env.scene[sensor_name]
    h = sensor.data.force_history
    assert h is not None, f"{sensor_name} keeps no force history"
    return (h.norm(dim=-1) > force_threshold).flatten(1).any(dim=1).float()


__all__ = [
    "climbed_stream",
    "crawl",
    "flight_forward",
    "flight_height",
    "flight_progress",
    "heading",
    "illegal_contact",
    "lateral_drift",
    "leg_collision",
    "reached_top",
    "landing_centre",
    "feet_near_edge",
    "settle_pose",
    "step_up_in_flight",
    "time_cost",
    "upright",
]
