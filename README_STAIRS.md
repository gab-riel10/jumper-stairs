# Stairs by jumping — `jumper.stairs`, `jumper.stairs_sighted`, `jumper.stairs_free`

A working copy of [KingKongRobotics/jumper](https://github.com/KingKongRobotics/jumper)
(Apache-2.0, see `LICENSE`) for the Crab Robot Open-source Challenge 2026: teach the
Jumper hexapod (10.7 cm tall) to climb a straight staircase — real step 15 cm rise,
40 cm tread — by **jumping**. Everything added is on branch `stairs-task`, on top of
upstream commit `b2e5866`; the commit history says what changed and why.

Developed with AI assistance (Claude).

## The three tasks

| | task | actor sees | climbing rule | deployable |
|---|---|---|---|---|
| A | `jumper.stairs` | its own body only (blind) | treads count only if gained in flight; walked treads are charged | yes (ONNX export passes) |
| B | `jumper.stairs_sighted` | + 65-ray height scan | same as A | no, sim only |
| C | `jumper.stairs_free` | + 65-ray height scan | any way up: walked treads pay like jumped ones; the front jaws may touch the step (friction 0.6) | no, sim only |

All three share one config (`tasks/jumper/stairs/env_cfg.py`): a 10-row terrain
curriculum (riser 2 → 15 cm), warm start from `jumper.ref_free_jump`'s jump
(`tools/warm_start.py`), body-on-terrain still charged everywhere.

## What had to be fixed (in order found)

1. **Reward loop** — treads were paid in flight and refunded on the ground; B learnt to
   hop back and forth over one riser line (43 "treads" per episode on an 8-tread flight).
   Fixed with a high-water mark `k_best`: a tread pays once, when first reached.
2. **Curriculum gate** — the dwell counter reset on every demotion request, also at
   level 0, so in 10 h not one robot was promoted. Now Rudin et al.'s rule: promote on
   treads gained over the last 4 episodes, demote on 2 falls or 4 empty episodes in a row.
3. **Tread index on the rear feet** — measured posture, not progress: spawns started one
   tread "ahead", a jump on the spot could never pay (0 of 114 touchdowns in a probe),
   and session 1's "1.5 treads per episode" were the same misreading the other way.
   The **base** now decides the tread; the feet only say where a flight took off.
4. **No gradient forward** — an honest tread credit needs ~20 cm of forward flight
   before it says anything. Added `flight_progress` (20 per metre of new ground landed
   on, ratcheted like `k_best`), forward speed in flight ×4, and a time cost so that
   standing still loses.
5. **`joint_acc_l2` tried and removed** — with exploration noise it cost 0.9/s for
   nothing and stopped all learning; `joint_vel_l2` (-1e-4) stays.

## Results (Kaggle, 2× T4, 2048 envs per task, 10 h, notebook version 9, 2026-10-27/28)

**C (`jumper.stairs_free`)** climbs **by jumping**: last 20 iterations — curriculum level
5.15 of 9 on average (mean riser 9.4 cm), **level max 9 (15 cm)**, 0.51 treads gained in
flight per episode vs 0.013 walked (crawl ratio 1 %), 1.65 jumps per episode, lowest point
11.8 cm above the takeoff tread at the apex, 0.26 m of new ground per episode, reward 3.0.
The final replay on the 15 cm row: from a standstill on the landing it jumps the first
riser (airborne at t≈0.45 s, jaws reaching forward), lands upright on tread 1 at t≈0.7 s
and stays there. **It makes one jump per episode**; chaining a second is the next thing to learn.

**B** never moved: same start, the only difference being the charge on walked treads;
reward −1.6, zero treads. **A** was trained only in session 1, before fixes 2–5.

## Running a session

**Kaggle (what is used).** The dataset `jumper-stairs-task` holds this branch's task files as
a zip (`tasks/__init__.py`, `tasks/jumper/stairs*` with `landing_bank.npz`, our `tools/*.py`,
the notebook, and the starting checkpoints under `checkpoints/`); the notebook clones upstream
at `b2e5866` and extracts the zip over it.
1. File → Import Notebook → `notebooks/jumper_kaggle_stairs.ipynb`; GPU T4 x2, Internet on.
2. Add Input: the dataset. Earlier sessions' backups (`jumper/backups/*.zip` in a version's
   output, 1.3 GB each) stay on Kaggle, not in git.
3. Cell 1: `RUNS` (label, task, starting checkpoint file name), `NUM_ENVS` 2048, `START_ROWS`,
   `TRAIN_HOURS`, the stop rules. Save Version → Save & Run All (Commit).
4. Watch cell 4's line per minute per run (`level`, `steps_by_flight`, `chained`, `rejumps`,
   `stall`, `fell`); replays land every 2 h in `session-*/replays/`; cell 5 ends with the
   A-against-B table.

To continue a run from its backup, point a `RUNS` entry's `start` at the `model_N.pt` taken
out of the backup zip (`run/model_N.pt`) and add it to the dataset: the conversion in cell 3
(`tools/warm_start.py --critic graft --keep-iter`) accepts any checkpoint of the same task.
Rows saved in that checkpoint are restored; `START_ROWS` applies only when there are none.

**Colab.** `notebooks/jumper_colab_stairs.ipynb` (one T4, code and checkpoint from Google
Drive) was used once, for `model_10650`; it predates the two-run notebook and the stall.

## Round 2: the second jump (2026-11-04)

**What was measured first** (`tools/stairs_tremor.py`, native:cpu, 16 robots, alpha 0.5,
observation noise on, 2026-11-03; the sha256 of `model_6750.pt` starts `76e4afa0`, of
`model_10650.pt` `fead0a04`):

| | climbs the 15 cm kerb | fell | lowest-point apex | base rise | airborne | push-off vz | joint speed RMS 0.5 s after landing | power above 5 Hz |
|---|---|---|---|---|---|---|---|---|
| `model_6750` (C) | 100 % | 0 % | 19.4 cm | 21.9 cm | 0.42 s | 1.37 m/s | 0.51 rad/s | 99 % |
| `model_10650` | 100 % | 0 % | 21.0 cm | 22.1 cm | 0.45 s | 1.49 m/s | 0.77 rad/s | 100 % |

- **One jump, ever.** On the flight, 16 s, both policies: one flight on the 15 cm row and one
  on the 9 cm row (airborne 0.42-0.47 s in 16 s). The stream `climbed_stream` pays ~3000 for
  standing after the first tread and a fall ends it; the second jump was never worth trying
  (`tasks/jumper/stairs/mdp/commands.py`, "The second jump").
- **The tremble is the policy's**, the same with and without observation noise, and the same
  in MuJoCo Warp and native MuJoCo (4 robots, same seed: 1.11 against 1.15 contacts per loaded
  foot, 0.68 against 0.70 rad/s). Crabcraft's four-against-one contact asymmetry (kubony/crabcraft)
  does not occur with this robot's convex-hull feet on box treads.
- **Targets past the joint limits**: 0 % after the landing; in the push 0.4-0.8 % of (step,
  joint) pairs, the `*_J2` knees, up to 1.0 rad (C) and 1.7 rad (10650) past a stop -- the
  number Crabcraft reports. Replayed with the controller's clamp both still climb 100 %.
- **The curriculum was never saved**: the checkpoints carry `common_step_counter` only, so every
  resume (including C → 10650) restarted at row 0.

**What changed** (all in `tasks/jumper/stairs/env_cfg.py`, each with its measurement beside it):
a `stall` termination (3 s without a new tread, short of the top, a true termination); the
n-th tread gained in flight pays `1 + 0.5 (n - 1)` treads (`CHAIN_BONUS`); a quarter of the
episodes start in a recorded post-landing state (`landing_bank.npz`, from
`tools/stairs_landing_bank.py`; `landing_bank.json` has the provenance); joint targets clipped
to the joint limits as the controller clips them; the terrain rows checkpointed and restored
(`CurriculumRunner`; `MJRL_STAIRS_START_ROWS=3-7` for a checkpoint that carries none); new
metrics `chained` (episodes with two or more treads by jumping) and `rejumps`.

Ideas taken from the README of [kubony/crabcraft](https://github.com/kubony/crabcraft)
(Apache-2.0; no code was read or copied): the stillness measure (joint speed RMS from 0.5 s
after the landing, 16 rollouts, native MuJoCo), the Warp-versus-native contact check, and
training with the targets clamped as the controller clamps them.

**The session** (`notebooks/jumper_kaggle_stairs.ipynb`, Kaggle 2x T4, 2048 envs per GPU): two
runs of `jumper.stairs_free` with the same code, A from `model_10650` and B from `model_6750`
(the control), converted by `tools/warm_start.py --critic graft --keep-iter` (the critic gained
the stall clock as its 487th input), stopped by the clock after 10 h or by the rules after 3 h
(`steps_by_flight` < 0.51, `fell_over` above twice the run's own first 20 iterations, or
curriculum row below 3). The question the session answers is `Metrics/stairs/chained`, per run.

## Next steps

- Distil the winner into a blind student for hardware (`rl/rsl_rl`'s `DistillationRunner`; the
  sighted actor reads the height scan, which the controller cannot build), then
  `scripts/export.py` on the student.
- Retrain A with the fixes if a blind policy is wanted directly.
