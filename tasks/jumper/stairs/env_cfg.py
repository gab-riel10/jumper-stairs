"""jumper.stairs -- up a flight of stairs by jumping, on the robot's own senses.

The robot starts at the foot of a straight staircase (or partway up it) and has
the episode to get to the top. It is paid for treads its rear feet gain **while
every foot is off the ground** and charged for treads they gain any other way,
so the thing it is trained to do is bound: land, settle, crouch, push, fly,
land one tread up. The riser height is a terrain curriculum from 2 cm to the
real step's 15 cm; the tread is the real step's 0.40 m throughout.

## What it is built from

- **The physics of `jumper.ref_free_jump`**: 200 Hz control (a push lasts
  ~50 ms, ten control steps here, two at 50 Hz), kp=20 (the gain the recorded
  jump was made at, and the one its checkpoint knows), the servo curve, this
  robot's contact parameters from `constants.py`, which were measured on box
  terrain (`_CONTACT_SOLREF`, `_CONTACT_SOLIMP`).
- **`jumper.ref_free_jump`'s actor observation, exactly** -- `base_ang_vel`,
  `projected_gravity`, then `joint_pos`, `joint_vel`, `actions` and
  `actuator_force` with five frames four control steps apart -- 406 inputs in
  that order, because `tools/warm_start.py` starts this task's actor from that
  task's `out/example/model_5999.pt`, and an input moved is an input the
  copied weights read wrong. No scan, no command and no clock: every term is
  one `deploy/fsm/src/obs.rs` can build, so the policy exports and goes in a
  bundle. The robot feels the riser with its front feet and reads the tilt off
  its IMU, and nothing else. (`jumper.stairs_sighted` is this task with the
  ground scan added to the actor -- a simulation-only variant, see there.)
- **A staircase of its own** (`mdp/terrain.py`): ten rows of one flight each,
  the terrain curriculum (`mdp/curriculum.py`) moving robots between them on
  treads jumped, never on treads walked.

## What the critic sees that the actor does not

The ground scan ahead (`terrain_scan`, critic-only, the asymmetric arrangement
`common/velocity_env.py` argues for), the base's linear velocity, which foot is
loaded, and the command's bookkeeping (`mdp/observations.py`). All of it is
discarded at export.

## On the numbers

The physics and the hardware facts are measured, or taken from where they are.
**The reward weights, the curriculum bars and the spawn mix are first guesses**,
set against `jumper.ref_free_jump`'s measured terms where there is an analogue
and against the arithmetic in the comments where there is not. The first run
on a GPU is what sets them; the note beside each one says what it was set
against, so that the next person changes the number and not the reason.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from ..common.constants import (
    ACTUATOR_FORCE_NOISE,
    EFFORT_LIMIT,
    FOOT_GEOMS,
    GAIT_JOINTS,
    IMU_ANG_VEL_NOISE,
    IMU_GRAVITY_NOISE,
    JUMPER_FOOT_FRICTION_RANGE,
    PUSH_VELOCITY_RANGE,
    TIP_LINKS,
    get_jumper_robot_cfg,
)
from ..common.mdp.observations import actuator_force
from ..common.tof import tof_sensor
from ..posture.mdp.history import StridedHistory
from . import mdp as stairs_mdp
from .mdp.terrain import Staircase

# ── Physics: jumper.ref_free_jump's ───────────────────────────────────────────
_SIM_TIMESTEP = 0.0025
_DECIMATION = 2
_KP = 20.0

# ── The action: jumper.ref_free_jump's ───────────────────────────────────────
#: Absolute targets around HOME at 0.5 rad per unit. The warm-started actor was
#: trained at this scale; at another one its outputs would mean another target.
ACTION_SCALE = 0.5

# ── The observation's history: jumper.ref_free_jump's ────────────────────────
OBS_HISTORY = 5
OBS_HISTORY_STRIDE = 4
HISTORY_TERMS = frozenset({"joint_pos", "joint_vel", "actions", "actuator_force"})

# ── The staircase ────────────────────────────────────────────────────────────
#: Eight treads of 0.40 m (the real step's run), risers from 2 cm to 15 cm (the
#: real step's rise) across the ten curriculum rows. 15 cm is 1.4 standing
#: heights; the vertical jump the robot has been measured to make lifts its
#: lowest point 0.115-0.123 m (`jumper.ref_free_jump`), so the top rows ask for
#: more than that jump and the curriculum is expected to stall before them.
#: That is the measurement this task is for.
STAIRS = Staircase(num_steps=8, tread=0.40, riser_range=(0.02, 0.15), landing=1.0, top=0.8, tail=0.4, width=2.0)
NUM_ROWS = 10

#: Every robot starts on the flattest row (`max_init_terrain_level`), for the
#: reason `common/velocity_env.py` measured: a policy put on ground it cannot
#: handle unwinds both curricula at once.
MAX_INIT_LEVEL = 0

# ── The scan, critic-only here ───────────────────────────────────────────────
#: 1.2 x 0.4 m at 0.1 m, centred on the base: 13 x 5 = 65 rays, reaching 0.6 m
#: ahead -- one and a half treads, so the next riser is always in it. Coarser
#: than `common/velocity_env.py`'s 0.05 m because a riser is a vertical face: the
#: scan only has to say where the edge is to within a tenth of a tread.
SCAN_SIZE = (1.2, 0.4)
SCAN_RES = 0.1
SCAN_MAX_DIST = 2.0

# ── The spawn mix ────────────────────────────────────────────────────────────
#: Half the episodes start on a tread 1..7 rather than on the landing, so the
#: upper treads and their landings are practised before the policy can reach
#: them from the bottom -- `jumper.ref_free_jump`'s `RSI_FRACTION`, with the
#: same argument. `play` spawns everyone on the landing.
ELEVATED_FRACTION = 0.5
MAX_START_STEP = 7

# ── Rewards ──────────────────────────────────────────────────────────────────
#: Per new tread the rear feet gained in a jump, once, at the touchdown.
#: Raised from 4 after the first 10 h on Kaggle (2026-10-27, both tasks, 2048
#: envs): the blind policy settled at 1.0-1.5 treads per 8 s episode with the
#: stream paying for the first one and nothing pressing for the next. Eight
#: is what the stream pays in eight steps: the next tread has to be worth
#: taking at once, not only over the rest of the episode.
STEP_UP_WEIGHT = 8.0
#: Per tread of net climb, every step for the rest of the episode. This is the
#: term that makes a jump worth its risk: one tread gained with 6 s to go pays
#: 1200 steps of 1.0, against the 1.0 a step `upright` pays for standing still.
#: `jumper.ref_free_jump` pays its latched apex the same way.
CLIMBED_STREAM_WEIGHT = 1.0
#: Per new tread gained with a foot down: what the jump's impulse pays, with
#: the sign turned. A crawled tread also never enters the stream.
CRAWL_WEIGHT = -8.0
#: Once, on the step the robot has settled on the top platform: the flight as a
#: goal (arXiv 2402.06143's position framing), worth two and a half treads.
REACHED_TOP_WEIGHT = 20.0
#: The lowest point's height above the takeoff tread, per 0.10 m, in flight --
#: `jumper.ref_free_jump`'s `flight_height` at half its weight: here the
#: landing pays, the arc only has to clear the riser.
FLIGHT_HEIGHT_WEIGHT = 1.0
FLIGHT_HEIGHT_SCALE = 0.10
#: Forward speed in flight, per 1 m/s. A vertical jump gains no tread; this is
#: the gradient from the warm-started vertical jump towards a bound.
FLIGHT_FORWARD_WEIGHT = 0.5
#: Lowered from 1.0 with `SETTLE_WEIGHT` and `TIME_COST_WEIGHT` after the first
#: 10 h on Kaggle (2026-10-27): standing at HOME paid about 10 an episode
#: (8 s x (1.0 + 0.3 - 0.05)) against about 14 for the 1.2 treads the policy
#: took, and it took no more. Standing now nets about 1.6 an episode; one
#: tread with 6 s to go pays 8 + 6.
UPRIGHT_WEIGHT = 0.3
UPRIGHT_SIGMA_SQ = 0.10
#: Towards HOME on the ground. Lower than `jumper.ref_free_jump`'s 1.0: the
#: crouch before a push is far from HOME, and this must not price it out.
SETTLE_WEIGHT = 0.1
SETTLE_SIGMA_SQ = 0.05
#: Every step. Standing at HOME then nets 0.4 - 0.2; a fall ends the stream.
TIME_COST_WEIGHT = -0.2
LATERAL_DRIFT_WEIGHT = -0.5
LATERAL_DRIFT_SCALE = 0.20
HEADING_WEIGHT = -0.5
#: `jumper.ref_free_jump`'s, and its argument: a knee on a riser is a crawl's
#: best friend, and must cost more than the tread it helps gain.
ILLEGAL_CONTACT_WEIGHT = -5.0
LEG_COLLISION_WEIGHT = -5.0
#: `jumper.ref_free_jump`'s, per squared radian of target moved in one step.
ACTION_RATE_PER_RAD_SQ = -0.2304

# ── Terminations ─────────────────────────────────────────────────────────────
FELL_OVER_RAD = math.radians(50.0)
OFF_SIDE_M = 0.7
SETTLE_STEPS = 20

#: Eight seconds: eight treads at one bound a second, with the landing to spare.
EPISODE_S = 8.0

# ── Curriculum ───────────────────────────────────────────────────────────────
#: Treads gained in flight over the last `WINDOW_EPISODES` episodes at the
#: level, and the robot moves to the next row: half a tread an episode, which
#: the first 10 h on Kaggle (2026-10-27, 2048 envs) showed the blind policy
#: doing from the start (1.0-1.5 an episode) while the gate of that run -- two
#: in one episode, after six quiet episodes in a row -- never opened once
#: (`mdp/curriculum.py`). The rungs are 1.4 cm apart; what the curriculum has
#: to find out is the height, not the count.
PROMOTE_STEPS = 2.0
WINDOW_EPISODES = 4
#: Down a row after this many falls in a row, or this many long episodes in a
#: row with nothing gained in flight. Both read the policy, not one unlucky
#: episode: at 2048 envs a single fall is noise.
DEMOTE_FALLS = 2
DEMOTE_ZEROS = 4
MIN_EPISODE_STEPS = 200

_COLLISION_GEOMS = r".*_meshcol"

# ── The jaws, in `jumper.stairs_free` ────────────────────────────────────────
#: The collision geoms of the two front jaws: palm, finger, finger tip, front
#: pad and the two grip inserts (`TIP_LINKS`' front-leg half; the rear pad is
#: the foot already). In every other stairs task they are body on the terrain,
#: charged by `illegal_contact` and frictionless (condim 1, `constants.py`'s
#: scheme). The free task lets the robot hook or lean on a riser's edge with
#: them: they leave the illegal set and get a tangential contact at
#: `HAND_FRICTION`. The jaw joints stay uncontrolled, held at HOME, as in
#: every FOOT-mode task: the hook is a shape, not a grip.
HAND_GEOMS: tuple[str, ...] = tuple(f"{link}_meshcol" for link in TIP_LINKS if link[:2] in ("LF", "RF"))
#: Hard plastic on wood or stone, against the silicone foot's 1.2. A handbook
#: figure like the foot's, and the one thing `jumper.stairs_free` assumes that
#: its siblings do not.
HAND_FRICTION = (0.6, 0.01, 0.01)


def _stride_history(cfg: ManagerBasedRlEnvCfg) -> None:
    """`jumper.ref_free_jump`'s: five frames four steps apart on the per-joint
    terms of every group, with the term's noise moved into the wrapper and each
    group's term copied before it is wrapped. Its docstring has the three silent
    failures this ordering avoids."""
    p = StridedHistory.PREFIX
    for group_name, group in cfg.observations.items():
        if group.history_length:
            raise ValueError(f"{group_name}: a group-level history_length would stack twice")
        missing = HISTORY_TERMS - set(group.terms)
        if missing:
            raise ValueError(f"{group_name} has no {sorted(missing)} to give history to")
        for name in HISTORY_TERMS:
            term = group.terms[name] = copy.copy(group.terms[name])
            assert not term.history_length, f"{group_name}/{name} is already stacked"
            term.params = {
                **(term.params or {}),
                p + "func": term.func,
                p + "frames": OBS_HISTORY,
                p + "stride": OBS_HISTORY_STRIDE,
                p + "noise": term.noise if group.enable_corruption else None,
            }
            term.func = StridedHistory
            term.noise = None


def _pin_row(generator, row: str):
    """Narrow the ladder to one row, `scenes/rough.py::_pin`'s way: the same
    generator with `num_rows=1` and `difficulty_range` collapsed onto the row's
    difficulty, so the flight is byte-for-byte the one the full grid builds."""
    from dataclasses import replace

    try:
        index = int(row)
    except ValueError:
        raise SystemExit(f"[stairs] --stairs-row {row!r} is not an integer (0..{NUM_ROWS - 1})") from None
    if not 0 <= index < NUM_ROWS:
        raise SystemExit(f"[stairs] --stairs-row {index} is outside 0..{NUM_ROWS - 1}")
    difficulty = index / (NUM_ROWS - 1)
    print(f"[stairs] pinned to row {index}: riser {STAIRS.riser(difficulty):.3f} m, tread {STAIRS.tread} m")
    return replace(generator, num_rows=1, difficulty_range=(difficulty, difficulty))


def env_cfg(
    asset: Path | None = None, play: bool = False, sighted: bool = False, stairs_row: str = "",
    free_climb: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Stairs by jumping.

    Args:
        asset: path to the model XML, from `--model`. None uses the default jumper.xml.
        play: replay mode -- noise and pushes off, every episode from the landing.
        sighted: give the actor the ground scan too. `jumper.stairs_sighted` passes
            this; `jumper.stairs` never does, because the export would refuse it.
        stairs_row: `--stairs-row`, a curriculum row to pin every flight to; empty
            is the ladder. The curriculum then has nowhere to move anyone.
        free_climb: `jumper.stairs_free`: a crawled tread pays what a jumped one
            does, enters the stream and counts for the curriculum, and the jaws
            may touch the terrain with friction (`HAND_GEOMS`). How the robot
            gets up is its own business; only the body on the terrain still costs.
    """
    cfg = make_velocity_env_cfg()

    # ── Sim ───────────────────────────────────────────────────────────────────
    cfg.sim.mujoco.timestep = _SIM_TIMESTEP
    cfg.decimation = _DECIMATION
    # The velocity skeleton's contact budget for this robot. A flight of boxes
    # is cheaper than the rough scene's grid, whose `put_data` check wanted
    # nconmax 131 for the jumper (`scenes/rough.py`); if the build refuses with
    # "nconmax overflow", raise `nconmax` to exactly the number it names and
    # leave `njmax` -- the first is the expensive knob, the second nearly free.
    cfg.sim.njmax = 1024
    cfg.sim.nconmax = 160
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 128

    # ── Scene: the staircase, and the robot at kp=20 ──────────────────────────
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "generator"
    generator = stairs_mdp.terrain.terrain_cfg(STAIRS, NUM_ROWS)
    if stairs_row:
        generator = _pin_row(generator, stairs_row)
    cfg.scene.terrain.terrain_generator = generator
    cfg.scene.terrain.max_init_terrain_level = MAX_INIT_LEVEL
    robot = get_jumper_robot_cfg(effort_limit=EFFORT_LIMIT, asset=asset)
    for actuator in robot.articulation.actuators:
        actuator.stiffness = _KP
    legal_geoms = FOOT_GEOMS
    if free_climb:
        legal_geoms = FOOT_GEOMS + HAND_GEOMS
        hand_re = "|".join(f"^{n}$" for n in HAND_GEOMS)
        collision = robot.collisions[0]
        # `resolve_expr` takes the first matching pattern, so the jaws go in
        # front of the ".*" catch-all that makes everything else condim 1.
        def _with_hands(field, value):
            table = dict(field)
            last = table.pop(".*")
            table[hand_re] = value
            table[".*"] = last
            return table
        robot.collisions = (dataclasses.replace(
            collision,
            condim=_with_hands(collision.condim, 3),
            priority=_with_hands(collision.priority, 1),
            friction={**collision.friction, hand_re: HAND_FRICTION},
        ),)
    cfg.scene.entities = {"robot": robot}

    # The ground scan, kept: critic-only unless `sighted`. The foot scan is
    # read by nothing here and goes.
    sensors = []
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            sensor.frame.name = "base_link"
            sensor.pattern = GridPatternCfg(size=SCAN_SIZE, resolution=SCAN_RES)
            sensor.max_distance = SCAN_MAX_DIST
            sensors.append(sensor)
        elif sensor.name == "foot_height_scan":
            continue
        else:
            sensors.append(sensor)
    cfg.scene.sensors = tuple(sensors) + (
        ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            track_air_time=True,
        ),
        ContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            fields=("found", "force"),
            reduce="none",
            num_slots=1,
            history_length=_DECIMATION,
        ),
        ContactSensorCfg(
            name="body_ground_contact",
            primary=ContactMatch(mode="geom", pattern=_COLLISION_GEOMS, entity="robot",
                                 exclude=legal_geoms),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("force",),
            reduce="netforce",
            num_slots=1,
        ),
    )

    # ── Observations ──────────────────────────────────────────────────────────
    # The skeleton's actor order is base_lin_vel, base_ang_vel, projected_gravity,
    # joint_pos, joint_vel, actions, command, height_scan. Dropping command and
    # height_scan from both groups and base_lin_vel from the actor, then
    # appending actuator_force, is what gives jumper.ref_free_jump's 406-wide
    # actor in its order. Keep that sequence.
    for group in cfg.observations.values():
        for name in ("command", "height_scan", "foot_height", "foot_air_time",
                     "foot_contact", "foot_contact_forces"):
            group.terms.pop(name, None)
    del cfg.observations["actor"].terms["base_lin_vel"]

    gait_cfg = SceneEntityCfg("robot", joint_names=list(GAIT_JOINTS))
    for group in cfg.observations.values():
        for name in ("joint_pos", "joint_vel"):
            term = group.terms.get(name)
            if term is not None:
                term.params = dict(term.params or {}) | {"asset_cfg": gait_cfg}

    for group in cfg.observations.values():
        if "base_ang_vel" in group.terms:
            group.terms["base_ang_vel"].noise = Unoise(n_min=-IMU_ANG_VEL_NOISE, n_max=IMU_ANG_VEL_NOISE)
        if "projected_gravity" in group.terms:
            group.terms["projected_gravity"].noise = Unoise(n_min=-IMU_GRAVITY_NOISE, n_max=IMU_GRAVITY_NOISE)

    force_cfg = SceneEntityCfg("robot", actuator_names=list(GAIT_JOINTS))
    for group in cfg.observations.values():
        group.terms["actuator_force"] = ObservationTermCfg(
            func=actuator_force,
            params={"asset_cfg": force_cfg},
            noise=Unoise(n_min=-ACTUATOR_FORCE_NOISE, n_max=ACTUATOR_FORCE_NOISE),
        )

    # The scan: after the 406, so a warm start's weights line up. The critic
    # always; the actor only when sighted, with 2 cm of noise on a reading the
    # robot does not have a sensor for.
    cfg.observations["critic"].terms["height_scan"] = ObservationTermCfg(
        func=envs_mdp.height_scan, params={"sensor_name": "terrain_scan"}, scale=1.0 / SCAN_MAX_DIST,
    )
    if sighted:
        cfg.observations["actor"].terms["height_scan"] = ObservationTermCfg(
            func=envs_mdp.height_scan, params={"sensor_name": "terrain_scan"},
            noise=Unoise(n_min=-0.02, n_max=0.02), scale=1.0 / SCAN_MAX_DIST,
        )
    cfg.observations["critic"].terms["foot_contact"] = ObservationTermCfg(
        func=mdp.foot_contact, params={"sensor_name": "feet_ground_contact"},
    )
    cfg.observations["critic"].terms["stairs_state"] = ObservationTermCfg(
        func=stairs_mdp.observations.stairs_state,
        params={"command_name": "stairs", "rise_scale": FLIGHT_HEIGHT_SCALE, "mirror_kind": "scalar"},
    )

    # ── Action: absolute targets around HOME ──────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    joint_pos_action.actuator_names = list(GAIT_JOINTS)
    joint_pos_action.scale = {j: ACTION_SCALE for j in GAIT_JOINTS}
    joint_pos_action.use_default_offset = True

    # ── Command: the bookkeeping ──────────────────────────────────────────────
    cfg.commands = {
        "stairs": stairs_mdp.commands.StairsCommandCfg(
            entity_name="robot",
            stairs=STAIRS,
            settle_steps=SETTLE_STEPS,
            free_climb=free_climb,
        )
    }

    # ── Rewards ───────────────────────────────────────────────────────────────
    R = stairs_mdp.rewards
    cfg.rewards = {
        "step_up_in_flight": RewardTermCfg(func=R.step_up_in_flight, weight=STEP_UP_WEIGHT,
                                           params={"command_name": "stairs"}),
        "climbed_stream": RewardTermCfg(func=R.climbed_stream, weight=CLIMBED_STREAM_WEIGHT,
                                        params={"command_name": "stairs"}),
        "crawl": RewardTermCfg(func=R.crawl, weight=STEP_UP_WEIGHT if free_climb else CRAWL_WEIGHT,
                               params={"command_name": "stairs"}),
        "reached_top": RewardTermCfg(func=R.reached_top, weight=REACHED_TOP_WEIGHT,
                                     params={"command_name": "stairs"}),
        "flight_height": RewardTermCfg(func=R.flight_height, weight=FLIGHT_HEIGHT_WEIGHT,
                                       params={"command_name": "stairs", "scale": FLIGHT_HEIGHT_SCALE}),
        "flight_forward": RewardTermCfg(func=R.flight_forward, weight=FLIGHT_FORWARD_WEIGHT,
                                        params={"command_name": "stairs", "scale": 1.0}),
        "upright": RewardTermCfg(func=R.upright, weight=UPRIGHT_WEIGHT, params={"sigma_sq": UPRIGHT_SIGMA_SQ}),
        "settle_pose": RewardTermCfg(func=R.settle_pose, weight=SETTLE_WEIGHT,
                                     params={"asset_cfg": gait_cfg, "command_name": "stairs",
                                             "sigma_sq": SETTLE_SIGMA_SQ}),
        "time_cost": RewardTermCfg(func=R.time_cost, weight=TIME_COST_WEIGHT),
        "lateral_drift": RewardTermCfg(func=R.lateral_drift, weight=LATERAL_DRIFT_WEIGHT,
                                       params={"scale": LATERAL_DRIFT_SCALE}),
        "heading": RewardTermCfg(func=R.heading, weight=HEADING_WEIGHT),
        "illegal_contact": RewardTermCfg(func=R.illegal_contact, weight=ILLEGAL_CONTACT_WEIGHT,
                                         params={"sensor_name": "body_ground_contact", "force_threshold": 1.0}),
        "leg_collision": RewardTermCfg(func=R.leg_collision, weight=LEG_COLLISION_WEIGHT,
                                       params={"sensor_name": "self_collision", "force_threshold": 1.0}),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=ACTION_RATE_PER_RAD_SQ * ACTION_SCALE**2),
    }

    # ── Curriculum: taller risers for treads jumped ───────────────────────────
    cfg.curriculum = {
        "stairs": CurriculumTermCfg(
            func=stairs_mdp.curriculum.StairsLevels,
            params={"command_name": "stairs", "promote_steps": PROMOTE_STEPS,
                    "window": WINDOW_EPISODES, "demote_falls": DEMOTE_FALLS,
                    "demote_zeros": DEMOTE_ZEROS, "min_steps": MIN_EPISODE_STEPS,
                    "count_crawled": free_climb},
        ),
    }

    # ── Terminations ──────────────────────────────────────────────────────────
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "reached_top": TerminationTermCfg(func=stairs_mdp.terminations.reached_top, time_out=True,
                                          params={"command_name": "stairs"}),
        "fell_over": TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": FELL_OVER_RAD}),
        "off_side": TerminationTermCfg(func=stairs_mdp.terminations.off_side, params={"limit": OFF_SIDE_M}),
        "fell_off": TerminationTermCfg(func=stairs_mdp.terminations.fell_off, params={"command_name": "stairs"}),
    }

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["foot_friction"].params["asset_cfg"] = SceneEntityCfg("robot", geom_names=FOOT_GEOMS)
    cfg.events["foot_friction"].params["ranges"] = JUMPER_FOOT_FRICTION_RANGE
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
    cfg.events["push_robot"].params["velocity_range"].update(PUSH_VELOCITY_RANGE)
    cfg.events.pop("reset_base", None)
    cfg.events.pop("reset_robot_joints", None)
    cfg.events = {
        "reset_to_default": EventTermCfg(func=envs_mdp.reset_scene_to_default, mode="reset"),
        "reset_on_tread": EventTermCfg(
            func=stairs_mdp.events.reset_on_tread, mode="reset",
            params={"stairs": STAIRS, "elevated_fraction": ELEVATED_FRACTION,
                    "max_start_step": MAX_START_STEP},
        ),
        **cfg.events,
    }

    # ── Episode and viewer ────────────────────────────────────────────────────
    cfg.episode_length_s = EPISODE_S
    cfg.viewer.body_name = "base_link"
    cfg.viewer.distance = 1.2
    cfg.viewer.elevation = -15.0

    if play:
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.events["reset_on_tread"].params["elevated_fraction"] = 0.0
        # Keep the ladder: `--terrain-row` has no meaning here, so a replay
        # shows every row. The curriculum still moves robots on the evidence.
        cfg.scene.sensors = (cfg.scene.sensors or ()) + (tof_sensor(),)

    _stride_history(cfg)
    return cfg


__all__ = ["env_cfg"]
