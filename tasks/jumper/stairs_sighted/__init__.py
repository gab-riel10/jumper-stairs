"""jumper.stairs_sighted -- jumper.stairs with the ground scan in the actor.

**Simulation only.** The actor reads `height_scan`, which the controller
(`deploy/fsm/src/obs.rs`) cannot build and `scripts/export.py` therefore
refuses, so nothing trained here goes in a bundle. It exists to measure what
seeing the stairs is worth against the blind task, and as the fallback for a
video if the blind one does not climb in the sessions available.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS
from ..stairs import cli_args  # noqa: F401 -- the same `--stairs-row`

register(
    id="jumper.stairs_sighted",
    assets=JUMPER_ASSETS,
    description=(
        "jumper hexapod bounding up a straight staircase, the ground scan in the "
        "actor; simulation only, not exportable"
    ),
    tags=("locomotion", "jumper", "jump", "stairs", "sim-only"),
)
