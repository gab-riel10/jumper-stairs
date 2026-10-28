"""jumper.stairs -- up a flight of stairs by jumping, on the robot's own senses.

The actor observes what `jumper.ref_free_jump`'s does and nothing more, so the
policy exports; the critic sees the ground scan and the stairs' bookkeeping.
See `env_cfg.py`. `jumper.stairs_sighted` is the same task with the scan in
the actor, for simulation only.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.stairs",
    assets=JUMPER_ASSETS,
    description=(
        "jumper hexapod bounding up a straight staircase (riser curriculum 2-15 cm, "
        "0.40 m tread); blind actor, exportable"
    ),
    tags=("locomotion", "jumper", "jump", "stairs"),
)


#: The environment key behind `--stairs-row`, so a pinned row can be set from
#: `.env` like every other default in this repository.
STAIRS_ROW_ENV = "MJRL_STAIRS_ROW"
ACTION_EMA_ENV = "MJRL_STAIRS_ACTION_EMA"
STAIRS_LAYOUT_ENV = "MJRL_STAIRS_LAYOUT"


def cli_args(group):
    """This task's own command-line argument; see `tasks.load_cli_args`.

    `--stairs-row N` pins every flight to one curriculum row, 0 (2 cm risers)
    to 9 (15 cm). Empty is the full ladder. For replay and video: a recording
    of one riser height is a recording of one riser height. Declared here, in
    the one module of a task that is always imported, with nothing but argparse.
    """
    import os

    group.add_argument(
        "--stairs-row", dest="stairs_row", default=os.environ.get(STAIRS_ROW_ENV, ""),
        metavar="N",
        help="pin every flight to one riser-height row, 0 (2 cm) to 9 (15 cm). "
        "Empty is the full ladder, which training uses",
    )
    group.add_argument(
        "--action-ema", dest="action_ema", default=os.environ.get(ACTION_EMA_ENV, ""),
        metavar="ALPHA",
        help="low-pass the joint targets: target = ALPHA * asked + (1 - ALPHA) * "
        "previous target, per control step. Empty is the task's own setting "
        "(ACTION_EMA_ALPHA in env_cfg.py); 1 is no filter",
    )
    group.add_argument(
        "--stairs-layout", dest="stairs_layout", default=os.environ.get(STAIRS_LAYOUT_ENV, ""),
        choices=("", "flight", "curb"), metavar="LAYOUT",
        help="flight (the default, what training uses: eight treads) or curb "
        "(one riser and 1.5 m of flat top; replay and video only)",
    )
    return ("stairs_row", "action_ema", "stairs_layout")
