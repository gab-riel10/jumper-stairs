"""A first-order low-pass on the joint targets, for the stairs tasks.

`SmoothedJointPositionAction` is mjlab's `JointPositionAction` with one change:
the target it hands the actuators is an exponential moving average of the
targets the policy asks for,

    target_t = alpha * asked_t + (1 - alpha) * target_{t-1},

once per control step (200 Hz here), reset to HOME with the episode. `alpha = 1`
is the plain term. At 200 Hz, alpha 0.5 / 0.3 / 0.2 / 0.1 is a time constant of
5 / 12 / 20 / 45 ms, a corner near 32 / 14 / 8 / 3.5 Hz.

What it is for: the policies of the second Kaggle session (2026-10-27,
`jumper.stairs_free`) tremble after a landing, in the replays with and without
the jump succeeding. A tremble at tens of hertz is above anything a jump needs
-- the push lasts 0.10-0.15 s -- so a corner in that gap takes it out and
leaves the jump. Whether it does is measured, not assumed: see
`tools/stairs_tremor.py`.

What it leaves alone: the `actions` observation is the action manager's raw
action, so a policy replayed through the filter sees its own last request,
as it did in training. Trained with the filter on, it learns the lag, which is
the point -- the robot's controller applies the same filter.

On the robot: `deploy/fsm/src/action.rs` already runs a per-decode EMA on the
target, `action_smoothing`, with the weight on the *previous* target, so this
task's `alpha` is `1 - action_smoothing` (alpha 0.5 is `action_smoothing = 0.5`).
Two things the export does not carry, as of 2026-11-03: the coefficient is a
`--machine` setting (`deploy/BUNDLE.md`), one value for every mode of the bundle,
not a field of this mode's contract; and the controller then rate-limits the
target (`joint_pos_rate_limit`, per policy step; the example in
`deploy/fsm/README.md` is 0.20 rad, 40 rad/s at 200 Hz) and clamps it to the
joint limits, neither of which this term does -- see `env_cfg.py` for what the
training clips.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class SmoothedJointPositionActionCfg(JointPositionActionCfg):
    alpha: float = 1.0
    """Weight of the new request, in (0, 1]. 1 is no filter."""

    def build(self, env: ManagerBasedRlEnv) -> SmoothedJointPositionAction:
        return SmoothedJointPositionAction(self, env)


class SmoothedJointPositionAction(JointPositionAction):
    cfg: SmoothedJointPositionActionCfg

    def __init__(self, cfg: SmoothedJointPositionActionCfg, env: ManagerBasedRlEnv):
        if not 0.0 < cfg.alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {cfg.alpha}")
        super().__init__(cfg=cfg, env=env)
        # HOME: what a reset puts the raw action at (zero) after scale and offset.
        self._filtered = (self._raw_actions * self._scale + self._offset).clone()

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        a = self.cfg.alpha
        self._filtered = a * self._processed_actions + (1.0 - a) * self._filtered
        self._processed_actions = self._filtered

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        home = self._offset if self._offset.dim() == 2 else self._offset.expand_as(self._filtered)
        self._filtered[env_ids] = home[env_ids]


__all__ = ["SmoothedJointPositionAction", "SmoothedJointPositionActionCfg"]
