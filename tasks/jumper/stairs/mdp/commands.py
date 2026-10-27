"""The stairs command: no command at all, and all of the bookkeeping.

There is nothing to ask the policy for -- the stairs are in front of it and up
is the only way -- so `command` is a single number, the riser height of the
flight this environment is on, which nothing in the actor reads. What this term
is for is **whose step it is**: the quantities every reward, termination and
metric reads, computed once per step in one place, in the order the step runs
them (`ManagerBasedRlEnv.step`: rewards are computed before this update, on
the state the previous update left; see `jumper.ref_free_jump`'s command for
the same arrangement).

## Which step the robot is on

**The rear feet decide**, not the base. The robot has climbed a step when the
hindmost of its two rear foot sites (`LR`, `RR`) is over the next tread: that is
the last part of it to get up, and a body that leans over the edge with its rear
legs still below has not climbed anything. `Staircase.step_index` turns an x into
a tread number from the flight's own geometry, so there is no ray cast and no
threshold on heights.

## A jump, and what it earns

A **flight** is every foot at or under `force_threshold`; it starts at the first
such step and ends at the first step a foot is loaded again (the touchdown). Its
**rise** is how far the robot's lowest point -- the least of the six foot sites
and the base, each against its own height at HOME (`jumper.ref_free_jump`'s
measure, for the reasons it gives: feet flicked up over a base that stayed put
are not height) -- got above the tread it took off from. A flight whose rise
never passes `min_rise` is a hop: contact chatter reaches 1.3 mm and a push
that unloads the feet without lifting the body about 9 mm (measured in
`jumper.ref_free_jump`), and neither is a jump.

A tread counts once, the first time the rear feet stand on it (`k_best`, the
episode's high-water mark). Reached at the touchdown of a jump, it is the
jump's: `jump_gain` holds it for one step and `rewards.step_up_in_flight` pays
it. Reached with a foot on the ground, or at the touchdown of a hop, it is a
crawl: `crawl_gain` holds it for one step and `rewards.crawl` charges it, and
it is never paid afterwards. Nothing is refunded and nothing is paid twice:
walking back down and jumping the same tread again earns nothing, which is
what closed the loop the first version had (see `_update_command`).

`climbed_net` is the running total of treads earned by jumping (and, with
`free_climb`, by crawling too), per episode, monotone, and
`rewards.climbed_stream` pays it every step. That stream, not
the impulse, is what makes a jump worth its risk: a step gained early pays for
the rest of the episode, as `jumper.ref_free_jump`'s latched apex does, and an
episode that stands still at HOME earns the upright and settle terms alone.

## Metrics

Per episode, zeroed at reset and logged as their mean by the command manager:
`steps_by_flight`, `steps_crawled`, `jumps` (flights past `min_rise`),
`apex_rise` (the episode's highest), `flight_s`, `max_step` (the highest tread
the rear feet reached), `reached_top`, and `crawl_ratio` = crawled / (crawled +
jumped), 0 when nothing was climbed. `crawl_ratio` is the one to watch: a policy
that climbs with it rising is walking up, whatever the reward says.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from ...common.constants import GAIT_JOINTS, LEGS
from .terrain import Staircase

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

#: The two rear legs, in `LEGS` order.
REAR_LEGS = ("LR", "RR")


@dataclass(kw_only=True)
class StairsCommandCfg(CommandTermCfg):
    class_type: type = None  # filled in __post_init__

    # Resampled only at reset: a timer far longer than any episode.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    entity_name: str = "robot"
    stairs: Staircase
    """The flight's geometry, shared with the terrain and the spawn event."""
    contact_sensor_name: str = "feet_ground_contact"
    force_threshold: float = 1.0
    """A foot under this many newtons is in the air."""
    min_rise: float = 0.01
    """Metres the lowest point must rise above the takeoff tread for a flight to
    be a jump rather than a hop."""
    settle_steps: int = 20
    """Consecutive grounded steps on the top platform that count as arrived."""
    free_climb: bool = False
    """Count crawled treads in `climbed_net` too. `jumper.stairs_free` sets it:
    there a tread is a tread however the rear feet got onto it, and the task's
    `crawl` term pays it instead of charging it. The metrics stay as they are,
    so `crawl_ratio` still says how the climbing was done."""

    def __post_init__(self):
        self.class_type = StairsCommand

    def build(self, env: ManagerBasedRlEnv) -> StairsCommand:
        return StairsCommand(self, env)


class StairsCommand(CommandTerm):
    cfg: StairsCommandCfg

    def __init__(self, cfg: StairsCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        self.stairs = cfg.stairs
        n = self.num_envs
        dev = self.device
        zeros = lambda dtype=torch.float32: torch.zeros(n, dtype=dtype, device=dev)  # noqa: E731

        #: Tread under the hindmost rear foot, now.
        self.k_rear = zeros(torch.long)
        #: Tread it took off from, for the flight in progress.
        self.k_takeoff = zeros(torch.long)
        self.in_flight = zeros(torch.bool)
        #: Peak of `lowest_rise` during the flight in progress.
        self.apex = zeros()
        #: The highest tread the rear feet have stood on this episode: the
        #: high-water mark every credit is measured against.
        self.k_best = zeros(torch.long)
        #: Held for exactly one step after the event, for the rewards.
        self.jump_gain = zeros()
        self.crawl_gain = zeros()
        self.touchdown = zeros(torch.bool)
        #: Running total: jumped minus crawled, this episode.
        self.climbed_net = zeros()
        self.top_steps = zeros()
        self.grounded_steps = zeros()
        #: The riser height of this environment's flight, refreshed every step
        #: because the curriculum moves environments between rows at reset.
        self.riser = torch.full((n,), self.stairs.riser_range[0], device=dev)

        self._gait_ids, _ = self.robot.find_joints(list(GAIT_JOINTS), preserve_order=True)
        self._foot_sites, _ = self.robot.find_sites(list(LEGS), preserve_order=True)
        self._rear_sites, _ = self.robot.find_sites(list(REAR_LEGS), preserve_order=True)
        self._home_z = self._heights_at_home()
        self._sensor = None

        self.metrics["steps_by_flight"] = zeros()
        self.metrics["steps_crawled"] = zeros()
        self.metrics["jumps"] = zeros()
        self.metrics["apex_rise"] = zeros()
        self.metrics["flight_s"] = zeros()
        self.metrics["max_step"] = zeros()
        self.metrics["reached_top"] = zeros()
        self.metrics["crawl_ratio"] = zeros()
        self.metrics["riser"] = zeros()

    # ── What the rest of the task reads ────────────────────────────────────
    @property
    def command(self) -> torch.Tensor:
        return self.riser.unsqueeze(1)

    @property
    def generator_cfg(self):
        return self._env.scene.terrain.cfg.terrain_generator

    def _refresh_riser(self) -> None:
        terrain = self._env.scene.terrain
        levels = getattr(terrain, "terrain_levels", None)
        if levels is None:
            return
        self.riser.copy_(self.stairs.riser_of_level(levels, self.generator_cfg))

    def rear_x(self) -> torch.Tensor:
        """x of the hindmost rear foot, from the spawn origin."""
        data = self.robot.data
        x = data.data.site_xpos[:, data.indexing.site_ids][:, self._rear_sites, 0]
        return x.min(dim=1).values - self._env.scene.env_origins[:, 0]

    def tread_z(self, k: torch.Tensor) -> torch.Tensor:
        """World height of tread `k`'s top face."""
        return self._env.scene.env_origins[:, 2] + k.float() * self.riser

    def lowest_rise(self, k: torch.Tensor | None = None) -> torch.Tensor:
        """How far the robot's lowest point is above where it would be standing
        at HOME on tread `k` (the takeoff tread by default): the least, over the
        six foot sites and the base, of each one's height less its own at HOME.
        `site_xpos` directly, as `jumper.ref_free_jump` reads it."""
        data = self.robot.data
        feet = data.data.site_xpos[:, data.indexing.site_ids][:, self._foot_sites, 2]
        z = torch.cat([feet, data.root_link_pos_w[:, 2:3]], dim=1)
        z = z - self.tread_z(self.k_takeoff if k is None else k).unsqueeze(1)
        return (z - self._home_z).min(dim=1).values

    def airborne(self) -> torch.Tensor:
        if self._sensor is None:
            self._sensor = self._env.scene.sensors[self.cfg.contact_sensor_name]
        forces = self._sensor.data.force
        assert forces is not None
        return (forces.norm(dim=-1) <= self.cfg.force_threshold).all(dim=1)

    @property
    def on_top(self) -> torch.Tensor:
        return self.k_rear >= self.stairs.num_steps

    @property
    def arrived(self) -> torch.Tensor:
        """On the top platform, grounded, for `settle_steps` in a row."""
        return self.top_steps >= self.cfg.settle_steps

    # ── CommandTerm interface ──────────────────────────────────────────────
    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # The spawn event ran first and recorded the tread it put each robot on.
        self._refresh_riser()
        spawn = getattr(self._env, "_stairs_spawn_step", None)
        k = spawn[env_ids] if spawn is not None else torch.zeros_like(env_ids)
        self.k_rear[env_ids] = k.long()
        self.k_takeoff[env_ids] = k.long()
        self.in_flight[env_ids] = False
        self.apex[env_ids] = 0.0
        self.k_best[env_ids] = k.long()
        self.jump_gain[env_ids] = 0.0
        self.crawl_gain[env_ids] = 0.0
        self.touchdown[env_ids] = False
        self.climbed_net[env_ids] = 0.0
        self.top_steps[env_ids] = 0.0
        self.grounded_steps[env_ids] = 0.0

    def _update_metrics(self) -> None:
        self.metrics["max_step"].copy_(torch.maximum(self.metrics["max_step"], self.k_rear.float()))
        self.metrics["reached_top"].copy_(self.arrived.float())
        jumped = self.metrics["steps_by_flight"]
        crawled = self.metrics["steps_crawled"]
        total = jumped + crawled
        self.metrics["crawl_ratio"].copy_(torch.where(total > 0, crawled / total.clamp(min=1e-6), torch.zeros_like(total)))
        self.metrics["riser"].copy_(self.riser)

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is not None:
            # Reset path: the spawn was just written; nothing has moved.
            return
        self._refresh_riser()
        dt = self._env.step_dt
        # The one-step flags were read by this step's rewards; clear them.
        self.jump_gain.zero_()
        self.crawl_gain.zero_()
        self.touchdown.zero_()

        airborne = self.airborne()
        k_new = self.stairs.step_index(self.rear_x())

        takeoff = ~self.in_flight & airborne
        touchdown = self.in_flight & ~airborne
        # Takeoff: remember the tread, start the apex from nothing.
        self.k_takeoff.copy_(torch.where(takeoff, self.k_rear, self.k_takeoff))
        self.apex.masked_fill_(takeoff, 0.0)
        self.in_flight |= takeoff

        rise = self.lowest_rise()
        self.apex.copy_(torch.where(self.in_flight, torch.maximum(self.apex, rise), self.apex))

        # Credit is for **new ground only**: treads beyond `k_best`, the highest
        # tread the rear feet have stood on this episode. The first version paid
        # every tread crossed in flight and refunded every tread lost on the
        # ground, and the sighted policy found the loop in it within five
        # thousand iterations (Kaggle, 2026-10-27, `jumper.stairs_sighted` at
        # iteration ~5000: `steps_by_flight` 31-40 per episode on an 8-tread
        # flight, reward 30 -> 300): jump forward over a riser line, walk back
        # over it, repeat. With a high-water mark there is nothing to walk back
        # to: a tread is paid once, when first reached, and only by a jump.
        gain = (k_new - self.k_best).float().clamp(min=0.0)
        in_air = self.in_flight
        jumped = touchdown & (self.apex > self.cfg.min_rise)
        hopped = touchdown & ~jumped
        # Paid at a jump's touchdown; charged on the ground, and at the touchdown
        # of a hop (a flight that did not lift the body is walking with a bounce
        # in it). In the air before the touchdown nothing is settled yet.
        self.jump_gain.copy_(torch.where(jumped, gain, torch.zeros_like(gain)))
        self.crawl_gain.copy_(torch.where(~in_air | hopped, gain, torch.zeros_like(gain)))
        settled = jumped | ~in_air | hopped
        self.k_best.copy_(torch.where(settled, torch.maximum(self.k_best, k_new), self.k_best))
        self.touchdown.copy_(touchdown)
        self.in_flight &= ~touchdown

        # Treads earned by jumping, this episode: monotone, never refunded.
        self.climbed_net += self.jump_gain
        if self.cfg.free_climb:
            self.climbed_net += self.crawl_gain
        self.k_rear.copy_(k_new)

        grounded = ~airborne
        self.grounded_steps.copy_(torch.where(grounded, self.grounded_steps + 1.0, torch.zeros_like(self.grounded_steps)))
        self.top_steps.copy_(torch.where(grounded & self.on_top, self.top_steps + 1.0, torch.zeros_like(self.top_steps)))

        # Episode metrics.
        m = self.metrics
        m["steps_by_flight"] += self.jump_gain.clamp(min=0.0)
        m["steps_crawled"] += self.crawl_gain.clamp(min=0.0)
        m["jumps"] += jumped.float()
        m["apex_rise"].copy_(torch.where(jumped, torch.maximum(m["apex_rise"], self.apex), m["apex_rise"]))
        m["flight_s"] += airborne.float() * dt

    # ── Geometry at HOME ───────────────────────────────────────────────────
    def _heights_at_home(self) -> torch.Tensor:
        """The six foot sites' heights, then the base's, with the robot at its
        default root pose and HOME on flat ground: forward kinematics on a scratch
        copy of the compiled model, as `jumper.ref_free_jump` does."""
        m = self._env.sim.mj_model
        d = mujoco.MjData(m)
        ix = self.robot.indexing
        d.qpos[ix.free_joint_q_adr.cpu().numpy()] = (
            self.robot.data.default_root_state[0, :7].cpu().numpy()
        )
        d.qpos[ix.joint_q_adr.cpu().numpy()] = self.robot.data.default_joint_pos[0].cpu().numpy()
        mujoco.mj_kinematics(m, d)
        sites = ix.site_ids[self._foot_sites].cpu().numpy()
        z = [*d.site_xpos[sites, 2], d.xpos[ix.root_body_id, 2]]
        return torch.as_tensor(z, dtype=torch.float32, device=self.device)


__all__ = ["REAR_LEGS", "StairsCommand", "StairsCommandCfg"]
