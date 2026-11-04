"""Critic-only observation: what the rewards are computed from.

The actor sees its body and -- in `jumper.stairs_sighted` only -- the ground
scan; this term goes to the critic alone, which is thrown away at export, so
that the value function can see the bookkeeping the stairs rewards are gated
on rather than fit it as noise (`jumper.ref_free_jump`'s `jump_state`, for
the same reason). Every column is left-right symmetric, and the term says so
through `mirror_kind` for `common/mdp/symmetry.py`.

The seventh column is how far along the stall timer is (`terminations.stall`):
the value of standing on a tread falls to nothing at `stall_s`, and a critic
that cannot see the clock fits that cliff as noise. Appended last, so that a
critic trained before it (`model_6750`, `model_10650`: 486 inputs) can be
carried over by `tools/warm_start.py --critic graft` with one fresh column.
"""

from __future__ import annotations

import torch


def stairs_state(
    env,
    command_name: str = "stairs",
    rise_scale: float = 0.10,
    stall_s: float = 3.0,
    mirror_kind: str | None = None,
) -> torch.Tensor:
    """(in_flight, tread / num_steps, riser / riser_max, rise / rise_scale,
    climbed_net / num_steps, distance to the next riser / tread,
    seconds since the last tread gained / stall_s)."""
    del mirror_kind
    cmd = env.command_manager.get_term(command_name)
    stairs = cmd.stairs
    k = cmd.k_base
    rise = cmd.lowest_rise(k).clamp(min=-1.0, max=1.0) / rise_scale
    ahead = (stairs.next_riser_x(k) - cmd.base_x()) / stairs.tread
    idle = (cmd.steps_since_gain * env.step_dt / stall_s).clamp(max=1.0)
    return torch.stack(
        [
            cmd.in_flight.float(),
            k.float() / stairs.num_steps,
            cmd.riser / stairs.riser_range[1],
            rise,
            cmd.climbed_net / stairs.num_steps,
            ahead.clamp(min=-2.0, max=4.0),
            idle,
        ],
        dim=1,
    )


__all__ = ["stairs_state"]
