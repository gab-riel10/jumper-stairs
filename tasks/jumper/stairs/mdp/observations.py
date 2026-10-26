"""Critic-only observation: what the rewards are computed from.

The actor sees its body and -- in `jumper.stairs_sighted` only -- the ground
scan; this term goes to the critic alone, which is thrown away at export, so
that the value function can see the bookkeeping the stairs rewards are gated
on rather than fit it as noise (`jumper.ref_free_jump`'s `jump_state`, for
the same reason). Every column is left-right symmetric, and the term says so
through `mirror_kind` for `common/mdp/symmetry.py`.
"""

from __future__ import annotations

import torch


def stairs_state(
    env,
    command_name: str = "stairs",
    rise_scale: float = 0.10,
    mirror_kind: str | None = None,
) -> torch.Tensor:
    """(in_flight, tread / num_steps, riser / riser_max, rise / rise_scale,
    climbed_net / num_steps, distance to the next riser / tread)."""
    del mirror_kind
    cmd = env.command_manager.get_term(command_name)
    stairs = cmd.stairs
    k = cmd.k_rear
    rise = cmd.lowest_rise(k).clamp(min=-1.0, max=1.0) / rise_scale
    ahead = (stairs.next_riser_x(k) - cmd.rear_x()) / stairs.tread
    return torch.stack(
        [
            cmd.in_flight.float(),
            k.float() / stairs.num_steps,
            cmd.riser / stairs.riser_range[1],
            rise,
            cmd.climbed_net / stairs.num_steps,
            ahead.clamp(min=-2.0, max=4.0),
        ],
        dim=1,
    )


__all__ = ["stairs_state"]
