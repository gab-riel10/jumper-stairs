# Recorded jumps

Simulated jumps of policy **C** (`jumper.stairs_free`, checkpoint `model_6750.pt` of the
2026-10-27 Kaggle session) onto a single **15 cm kerb with 1.5 m of flat top**, recorded
for rendering. They come from `tools/stairs_record.py`, MuJoCo native backend on CPU, the
policy's mean action, observation noise as in training, the joint-target low-pass
(`tasks/jumper/stairs/mdp/actions.py`) at the alpha named in each folder.

## `curb15_alpha0.5/`

Filter alpha 0.5, observation noise on, 32 attempts from the spawn's jitter (seed 0): 32 got
onto the kerb, 2 fell afterwards. Kept: the five stillest after landing (RMS joint speed
from 0.5 s after the touchdown; a robot standing still reads about 0.1 rad/s).

| jump | takeoff | landing | apex (lowest point above the street) | base at landing / at rest, past the riser | stillness |
|---|---|---|---|---|---|
| 01 | 0.28 s | 0.55 s | 20.5 cm | 0.10 m / 0.26 m | 0.40 rad/s |
| 02 | 0.28 s | 0.53 s | 19.6 cm | 0.10 m / 0.34 m | 0.43 rad/s |
| 03 | 0.28 s | 0.50 s | 19.4 cm | 0.05 m / 0.30 m | 0.43 rad/s |
| 04 | 0.27 s | 0.55 s | 20.1 cm | 0.01 m / 0.24 m | 0.43 rad/s |
| 05 | 0.28 s | 0.55 s | 19.9 cm | 0.08 m / 0.24 m | 0.45 rad/s |

Each is 6 s (1201 frames): about 0.25 s walking up to the kerb, the jump, then standing on
the pavement. Why alpha 0.5: on this kerb (16 robots per row, Kaggle CPU, 2026-10-28) it
climbs 100 % with no falls and halves the post-landing tremble (0.56 rad/s against 1.05
unfiltered); 0.3 already breaks the jump on the 15 cm flight.

## What is in a folder

| file | contents |
|---|---|
| `jump_NN.npz` | one jump, every control step (200 Hz, 5 ms): `time` (T), `qpos` (T, 29), `qvel` (T, 28), `body_pos` (T, B, 3), `body_quat` (T, B, 4, w x y z), float32 |
| `jump_NN.json` | that jump's takeoff and landing times, apex, landing distance past the riser, stillness after landing |
| `model.json` | shared by the folder: joint names and their `qpos` addresses, body names (the order of `body_pos`), every visible part as an STL of this repository with its pose in its body, the kerb's geometry, timestep, filter alpha, checkpoint sha256 |

`qpos` is MuJoCo's: the free joint first (x y z, then the quaternion w x y z), then the
22 hinges in `joint_names` order (radians): five per front leg (the claw's included),
three per middle and rear leg. Names carry mjlab's `robot/` prefix. `body_pos` / `body_quat` are the
same motion already posed in the world, for a renderer without MuJoCo.

Coordinates are the simulator's world frame, metres, +x up the kerb, +z up. The riser is
the plane `x = kerb.riser_x`, the street is at `z = kerb.ground_z`; the kerb is
`kerb.height_m` tall and `kerb.top_depth_m` deep, `kerb.width_m` wide around `y = kerb.center_y`.
The robot's centre starts 0.5 m before the riser.

## Importing

**Blender (4.x)**, from the repository root:

```bash
blender --python trajectories/blender_import.py -- trajectories/<folder>/jump_01.npz
```

One Empty per body, keyframed at the scene's fps (nearest 200 Hz sample, nothing
interpolated); each part is the repository's STL parented to its body with the pose from
`model.json`; two boxes the size of the simulated street and kerb (`BUILD_KERB`). The
script moves everything so the riser is `x = 0` and the street `z = 0`. Checked with
`bpy` 4.2 against MuJoCo on `jump_01` (the robot's bounding box at frames 1, 61, 91, 141 and
401, all 41 parts): the same to 1e-8 m.

**Any other tool**: for body `b` at frame `i`, a vertex `v` of part `g` (an STL vertex, metres)
is at `R(body_quat[i, b]) · (R(g.file_quat_wxyz) · (g.scale ∘ v) + g.file_pos) + body_pos[i, b]`.

**MuJoCo**: `qpos` drops into a model compiled from `assets/jumper/jumper.xml` (29 values,
same order, names without the `robot/` prefix); place the kerb from `model.json`.

```python
import json, numpy as np
d = np.load("trajectories/<folder>/jump_01.npz"); m = json.load(open("trajectories/<folder>/model.json"))
d["qpos"].shape, m["control_dt_s"], m["action_ema_alpha"], m["joint_names"][:3]
```

To record again (CPU, a few minutes):

```bash
python tools/stairs_record.py --task jumper.stairs_free --checkpoint <model_6750.pt> \
    --row 9 --layout curb --alpha 0.3 --num-envs 32 --keep 5 --out trajectories/<folder>
```

The checkpoint itself is not in git; it is in the Kaggle backup
`jumper.stairs_free-20261027-175113.zip` (`run/model_6750.pt`).
