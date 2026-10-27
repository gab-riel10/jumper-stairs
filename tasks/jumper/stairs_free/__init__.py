"""jumper.stairs_free -- jumper.stairs_sighted without the rule that a tread
must be jumped.

**Simulation only**, for the same reason as `jumper.stairs_sighted`. A tread
gained with a foot on the ground pays what a jumped one does, enters the
stream and counts for the curriculum; the body on the terrain still costs
(`illegal_contact`). The question it answers against `jumper.stairs_sighted`,
on the other GPU of the same session: does a policy left free to walk, mantle
or jump get up a 15 cm riser sooner than one made to jump from the first
2 cm? The geometry says that at 15 cm -- 140 % of the robot's height, with
7.85 cm of reach below the body -- the rear legs must leave the ground either
way; the bet is on what is learnt before that. `Metrics/stairs/crawl_ratio`
says which way it went.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS
from ..stairs import cli_args  # noqa: F401 -- the same `--stairs-row`

register(
    id="jumper.stairs_free",
    assets=JUMPER_ASSETS,
    description=(
        "jumper hexapod up a straight staircase any way it finds, the ground "
        "scan in the actor; simulation only, not exportable"
    ),
    tags=("locomotion", "jumper", "jump", "stairs", "sim-only"),
)
