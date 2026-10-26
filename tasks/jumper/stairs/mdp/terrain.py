"""The staircase: its numbers, the boxes that build it, and the arithmetic that
reads it back.

One straight flight of stairs along +x, one per tile: a flat landing the robot
spawns on, `num_steps` treads of one depth and one riser height, and a flat
platform at the top. The riser height is the one thing the terrain curriculum
moves -- row `r` of `num_rows` gets `riser_range[0] + (r / (num_rows - 1)) *
(riser_range[1] - riser_range[0])`, which is exactly how mjlab's generator maps a
row onto `difficulty` -- so the tread is the same on every row and the reward's
"one step" means the same thing everywhere.

Everything that reads the staircase (the command's bookkeeping, the spawn event,
the curriculum, the critic's state) goes through `Staircase`, so there is one
place where "which step is this x on" is defined. The generator only ever sees
`StraightStairsTerrainCfg.function`.

Why a terrain of this task's own rather than `--scene rough`: the rough scene's
`pyramid_stairs` is concentric and its difficulty rows mix seven sub-terrains,
and `tools/colab.py`, which the notebooks train through, passes no `--scene` at
all. A task that needs a particular ground owns it (`scenes/registry.apply`:
"a task legitimately configures the ground it was designed for").

Sizes are in metres. The robot stands 0.107 m tall and its fore-aft stance is
0.331 m (`scenes/rough.py`), so a 0.40 m tread holds the whole robot with 7 cm
to spare and a 1.0 m landing holds three of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Torch is imported lazily inside the two tensor helpers so that the terrain
# function -- which only needs numpy and mujoco -- can be exercised on a machine
# without it. Nothing in `tasks/__init__.py` imports this module.


@dataclass(frozen=True)
class Staircase:
    """The numbers, and the arithmetic. Pure data: no torch, no mjlab."""

    num_steps: int = 8
    """Treads in one flight."""
    tread: float = 0.40
    """Depth of one tread (x), the same on every row."""
    riser_range: tuple[float, float] = (0.02, 0.15)
    """Riser height at difficulty 0 and 1, the curriculum's two ends."""
    landing: float = 1.0
    """Flat run before the first riser, where the robot spawns."""
    top: float = 0.8
    """Flat run after the last riser."""
    tail: float = 0.4
    """Floor-level run after the top platform, closing the tile.

    **Not decoration.** Rows are stacked along x with no gap, so the end of one
    flight is the start of the next, and mjwarp checks its contact budget with
    the robot at the world origin -- which, for an even number of rows, is the
    first x of row `num_rows / 2`. Without this the robot's rear 0.15 m would
    sit inside the previous row's top platform at build time, and `put_data`
    would ask for a contact budget sized for that penetration (the rough
    scene's grid has exactly this cost; see its `REQUIRED_NCON`). With it, the
    origin stands on floor either side. The cliff it leaves behind the top
    platform is what `terminations.fell_off` is for."""
    width: float = 2.0
    """Tile width (y). The stairs span the whole of it."""

    @property
    def size(self) -> tuple[float, float]:
        """The tile the generator must be given, (x, y)."""
        return (self.landing + self.num_steps * self.tread + self.top + self.tail, self.width)

    @property
    def local_origin(self) -> np.ndarray:
        """Spawn origin in the tile's local frame: the middle of the landing."""
        return np.array([0.5 * self.landing, 0.5 * self.width, 0.0])

    @property
    def first_riser_from_origin(self) -> float:
        """x of the first riser, measured from the spawn origin."""
        return 0.5 * self.landing

    def riser(self, difficulty: float) -> float:
        lo, hi = self.riser_range
        return lo + difficulty * (hi - lo)

    def riser_of_level(self, level, generator_cfg):
        """Riser height for a terrain row, as a tensor, from the generator's own
        rule -- `difficulty = lower + (upper - lower) * row / (num_rows - 1)` over
        its `difficulty_range` -- so the two cannot disagree, a pinned row
        (`num_rows=1`, the range collapsed) included."""
        lo, hi = self.riser_range
        lower, upper = generator_cfg.difficulty_range
        t = level.float() / max(generator_cfg.num_rows - 1, 1)
        difficulty = lower + (upper - lower) * t
        return lo + difficulty * (hi - lo)

    def step_index(self, x_from_origin):
        """Which tread a point is over, 0 for the landing and `num_steps` for the
        top platform, as a long tensor."""
        import torch

        k = torch.floor((x_from_origin - self.first_riser_from_origin) / self.tread) + 1.0
        return k.clamp(0.0, float(self.num_steps)).long()

    def next_riser_x(self, k):
        """x of the riser in front of tread `k`, from the spawn origin; on the
        top platform, the platform's end (a long way off)."""
        import torch

        x = self.first_riser_from_origin + k.float() * self.tread
        end = self.first_riser_from_origin + self.num_steps * self.tread + self.top
        return torch.where(k >= self.num_steps, torch.full_like(x, end), x)


def build_staircase(stairs: Staircase, difficulty: float, spec, rng=None):
    """Add one flight of stairs to the spec's `terrain` body and return the
    generator's `TerrainOutput`.

    Pure box geometry, and the boxes stand on a floor box rather than on the
    world plane, as mjlab's own box terrains do: a floor the tile owns is what
    keeps the landing solid when the grid is offset from the world origin.
    """
    import mujoco

    from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput
    from mjlab.terrains.utils import make_plane

    del rng
    body = spec.body("terrain")
    riser = stairs.riser(difficulty)
    size = stairs.size
    geometries: list[TerrainGeometry] = []

    # The floor: the whole tile, 0.5 m thick, top face at z = 0.
    for geom in make_plane(body, size, 0.0, center_zero=False, plane_thickness=0.5):
        geometries.append(TerrainGeometry(geom=geom, color=(0.42, 0.42, 0.44, 1.0)))

    # Each tread is one box from the floor up: step k's top face at k * riser.
    # Boxes reaching the floor rather than stacked slabs keep the riser faces
    # whole, so a foot meets one face per riser rather than a seam.
    for k in range(1, stairs.num_steps + 1):
        half = (0.5 * stairs.tread, 0.5 * size[1], max(0.5 * k * riser, 1e-6))
        pos = (stairs.landing + (k - 0.5) * stairs.tread, 0.5 * size[1], 0.5 * k * riser)
        geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=half, pos=pos)
        shade = 0.55 if k % 2 else 0.68
        geometries.append(TerrainGeometry(geom=geom, color=(shade, shade, shade + 0.02, 1.0)))

    # The top platform, at the last tread's height.
    top_z = stairs.num_steps * riser
    half = (0.5 * stairs.top, 0.5 * size[1], max(0.5 * top_z, 1e-6))
    pos = (stairs.landing + stairs.num_steps * stairs.tread + 0.5 * stairs.top, 0.5 * size[1], 0.5 * top_z)
    geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=half, pos=pos)
    geometries.append(TerrainGeometry(geom=geom, color=(0.48, 0.52, 0.48, 1.0)))

    return TerrainOutput(origin=stairs.local_origin.copy(), geometries=geometries)


def terrain_cfg(stairs: Staircase, num_rows: int, border_width: float = 2.0):
    """The generator config for a ladder of `num_rows` flights, one column.

    `curriculum=True` is what makes a row a difficulty level at all
    (`scenes/rough.py` has the measurement); with one sub-terrain the grid is
    `num_rows x 1`, rows stacked along x. The spawn origin of row r sits on its
    own landing, so `env_origins` is the landing's centre and
    `Staircase.first_riser_from_origin` is measured from there.
    """
    from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg

    class StraightStairsTerrainCfg(SubTerrainCfg):
        def function(self, difficulty, spec, rng):
            return build_staircase(stairs, difficulty, spec, rng)

    return TerrainGeneratorCfg(
        curriculum=True,
        color_scheme="height",
        size=stairs.size,
        border_width=border_width,
        num_rows=num_rows,
        num_cols=1,
        sub_terrains={"stairs": StraightStairsTerrainCfg(proportion=1.0)},
        add_lights=False,
    )


__all__ = ["Staircase", "build_staircase", "terrain_cfg"]
