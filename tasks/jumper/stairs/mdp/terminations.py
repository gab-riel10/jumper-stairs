"""How an episode on the stairs ends early.

`fell_over` is mjlab's `bad_orientation` at the controller's own 50 degrees,
mounted in `env_cfg.py`. These two are the task's:

- `off_side`: the base has left the flight sideways. The tile is 2 m wide and
  the stairs span all of it, so past this there is a drop to the border.
- `fell_off`: the base is well below the tread it was last over --
  off the cliff behind the top platform (`terrain.Staircase.tail`), or tumbled
  back down the flight. Two risers and a margin, so a landing one tread down,
  which the reward already charges, is not an ending.
- `reached_top`: the base is over the top platform and the robot has stood
  there, grounded, for the command's `settle_steps`. Mounted as a **time-out**
  so PPO bootstraps from the value of standing at the top, where the climbed
  stream goes on paying: a success must not be worth less than loitering on
  the last tread (`jumper.ref_free_jump` ends `back_home` the same way).
- `stall`: `stall_s` seconds since the base last gained a tread (or since the
  spawn) with none gained, short of the top and not in the air. Mounted as a
  **termination**, not a time-out, on purpose: it is the end of the climbed
  stream, and a bootstrap from the value of standing would hand the stream
  back. The measurement behind it is in `commands.py` ("The second jump"):
  with the stream running to the end of a 16 s episode, neither policy ever
  jumped twice, on any riser. Training only; `env_cfg.py` leaves it out of
  replays, so a recording shows what the policy does when nothing ends it.
"""

from __future__ import annotations

import torch

from mjlab.entity import Entity


def off_side(env, limit: float = 0.7) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    d = robot.data.root_link_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return d.abs() > limit


def fell_off(env, command_name: str = "stairs", margin: float = 0.05) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    robot: Entity = env.scene["robot"]
    z = robot.data.root_link_pos_w[:, 2]
    floor = cmd.tread_z(cmd.k_base) - 2.0 * cmd.riser - margin
    return z < floor


def reached_top(env, command_name: str = "stairs") -> torch.Tensor:
    return env.command_manager.get_term(command_name).arrived


def stall(env, command_name: str = "stairs", stall_s: float = 3.0) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    idle = cmd.steps_since_gain * env.step_dt >= stall_s
    return idle & ~cmd.on_top & ~cmd.in_flight


__all__ = ["fell_off", "off_side", "reached_top", "stall"]
