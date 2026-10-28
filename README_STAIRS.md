# Stairs by jumping — `jumper.stairs`, `jumper.stairs_sighted`, `jumper.stairs_free`

A private working copy of [KingKongRobotics/jumper](https://github.com/KingKongRobotics/jumper)
(Apache-2.0, see `LICENSE`) for the Crab Robot Open-source Challenge 2026: teach the
Jumper hexapod (10.7 cm tall) to climb a straight staircase — real step 15 cm rise,
40 cm tread — by **jumping**. Everything added is on branch `stairs-task`, on top of
upstream commit `b2e5866`; the commit history says what changed and why.

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

## Resuming

**Kaggle (what was used).** The private dataset `jumper-stairs-task` holds this branch's
task files as a zip; the notebook clones upstream at `b2e5866` and copies them over.
1. File → Import Notebook → `notebooks/jumper_kaggle_stairs.ipynb`; GPU T4 x2, Internet on.
2. Add Input: the dataset (rebuild the zip from `tasks/__init__.py`, `tasks/jumper/stairs*`,
   `tools/warm_start.py`, `notebooks/` if starting on another account) and, to continue,
   the output of the notebook version holding `jumper/backups/jumper.stairs_free-*.zip`
   (1.3 GB; it stays on Kaggle, not in git).
3. Cell 1: `TASK = "jumper.stairs_free"`, `SECOND_TASK = ""`, `WORKFLOW_MODE = "Continue training"`,
   `RESTART_TASKS = []`, `ACCEPT_CODE_CHANGES = True` if the task code changed since the
   backup; `NUM_ENVS` must stay 2048. Save Version → Save & Run All.
4. Watch cell 4's line per minute (`level`, `steps_by_flight`, `crawled`, `progress_m`);
   replays land every 2 h in `session-*/replays/`.

**Colab.** Not tested. `notebooks/jumper_colab.ipynb` checks out the commit recorded in a
backup (`b2e5866`, upstream, no stairs code), so it cannot continue these backups as is.
Either adapt `jumper_kaggle_stairs.ipynb` (cell 1's `WORK`/`OUT` already switch off Kaggle,
but `PATCH_DIR` and `BACKUP_GLOB` must be pointed at where the task zip and the backup
are, e.g. Google Drive), or run a **New training** of `jumper.stairs_free`
from this repository (`REPOSITORY_URL` = this repo, `SOURCE_REVISION = "stairs-task"`; a
private repo needs a read-only token to clone).

## Next steps

- Chain jumps: longer episodes (8 → 16 s) and elevated spawns, so the landing is a new start.
- Distil C into a blind student for hardware (C's actor reads the height scan, which the
  controller cannot build).
- Retrain A with fixes 2–4 if a blind policy is wanted directly.
