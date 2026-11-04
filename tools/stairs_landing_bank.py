"""Record what a stairs policy looks like just after it has landed one tread
up, as a bank of states the task can spawn from (`mdp/events.py`,
`landed_fraction`).

Why: the two policies of 2026-11-03 (`model_6750`, `model_10650`) jump once an
episode and never again, from any riser. Half their episodes started standing
at HOME on a tread, and from there they jump at once; what they never trained
from is the state a landing actually leaves -- the crouch, the slide, the
feet off centre, the joints still moving. This records that state so that a
quarter of the episodes can start in it.

For each row in `--rows`, `--num-envs` robots run the policy's mean action from
the landing, native backend, observation noise as in training, the joint-target
low-pass at `--alpha`, for `--seconds`. At every touchdown of a jump that
gained a tread, the robot is sampled `--delays` seconds later if it is still
up, grounded and on the tread: the base's pose **relative to the tread it is
over** (x from the tread's centre, y from the flight's centre line, z from the
tread's top face), its orientation and velocities in the world frame, and the
22 hinge joints' positions and velocities in the entity's joint order.

Output: `<out>.npz` with `rel_pos` (M, 3), `quat_wxyz` (M, 4), `lin_vel` (M,
3), `ang_vel` (M, 3), `joint_pos` (M, J), `joint_vel` (M, J), `row` (M,),
`riser` (M,), `delay_s` (M,), `joint_names` (J,); and `<out>.json` with where
it came from (checkpoints and their sha256, rows, counts, the command).

Example (CPU, a few minutes per checkpoint)::

    python tools/stairs_landing_bank.py --task jumper.stairs_free \\
        --checkpoint logs/.../model_10650.pt logs/.../model_6750.pt \\
        --rows 2-9 --out tasks/jumper/stairs/landing_bank
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np


def _match_training_noise(env_cfg, train_cfg) -> None:
    """`scripts/play.py`'s `--obs-noise`; see `tools/stairs_tremor.py`."""
    for name, group in env_cfg.observations.items():
        reference = train_cfg.observations.get(name)
        if reference is None:
            continue
        group.enable_corruption = reference.enable_corruption
        for term_name, term in group.terms.items():
            source = reference.terms.get(term_name)
            if source is not None:
                term.noise = source.noise


def _rows(spec: str) -> list[int]:
    lo, _, hi = spec.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def record(task: str, checkpoint: Path, row: int, alpha: float, num_envs: int, seconds: float,
           delays: list[float], seed: int) -> dict:
    import torch

    import tasks
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner

    task_args = {"stairs_row": str(row), "action_ema": str(alpha), "stairs_layout": "flight"}
    cfg = tasks.load_env_cfg(task, play=True, task_args=task_args)
    _match_training_noise(cfg, tasks.load_env_cfg(task, task_args=task_args))
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        agent = tasks.load_agent_cfg(task)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device="cpu")
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True, map_location="cpu")
        policy = runner.get_inference_policy(device="cpu")

        cmd = env.command_manager.get_term("stairs")
        robot = env.scene["robot"]
        dt = env.step_dt
        steps = int(round(seconds / dt))
        n = num_envs
        delay_steps = [int(round(d / dt)) for d in delays]
        origins = env.scene.env_origins

        alive = torch.ones(n, dtype=torch.bool)
        due: list[tuple[int, int, float]] = []  # (step, env, delay_s)
        out = {k: [] for k in ("rel_pos", "quat_wxyz", "lin_vel", "ang_vel", "joint_pos", "joint_vel", "delay_s")}

        obs = wrapped.get_observations()
        with torch.no_grad():
            for i in range(steps):
                obs, _, dones, _ = wrapped.step(policy(obs))
                done = dones.bool().cpu()
                live = alive & ~done
                landed = (cmd.jump_gain > 0) & live
                for e in landed.nonzero(as_tuple=False).flatten().tolist():
                    for d, ds in zip(delays, delay_steps, strict=True):
                        due.append((i + ds, e, d))
                k = cmd.k_base
                ok = live & ~cmd.in_flight & (k >= 1) & (k < cmd.stairs.num_steps)
                for step, e, d in [x for x in due if x[0] == i]:
                    if not bool(ok[e]):
                        continue
                    base = robot.data.root_link_pos_w[e]
                    rel = torch.stack([
                        base[0] - origins[e, 0] - cmd.tread_centre_x(k[e:e + 1])[0],
                        base[1] - origins[e, 1],
                        base[2] - cmd.tread_z(k[e:e + 1])[0],
                    ])
                    out["rel_pos"].append(rel.cpu().numpy())
                    out["quat_wxyz"].append(robot.data.root_link_quat_w[e].cpu().numpy())
                    out["lin_vel"].append(robot.data.root_link_lin_vel_w[e].cpu().numpy())
                    out["ang_vel"].append(robot.data.root_link_ang_vel_w[e].cpu().numpy())
                    out["joint_pos"].append(robot.data.joint_pos[e].cpu().numpy())
                    out["joint_vel"].append(robot.data.joint_vel[e].cpu().numpy())
                    out["delay_s"].append(d)
                due = [x for x in due if x[0] > i]
                alive &= ~done
        riser = float(cmd.riser[0])
        return {
            **{k: np.asarray(v, dtype=np.float32) for k, v in out.items()},
            "row": np.full(len(out["delay_s"]), row, dtype=np.int64),
            "riser": np.full(len(out["delay_s"]), riser, dtype=np.float32),
            "joint_names": list(robot.joint_names),
        }
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--task", default="jumper.stairs_free")
    p.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    p.add_argument("--rows", default="2-9", help="curriculum rows, lo-hi inclusive")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--seconds", type=float, default=4.0)
    p.add_argument("--delays", default="0.3,0.6", help="seconds after the touchdown, comma-separated")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True, help="path without extension")
    p.add_argument("--part", action="store_true", help=argparse.SUPPRESS)  # one row, written by a child process
    a = p.parse_args()

    import tasks  # noqa: F401 -- registers the tasks
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    use_backend(resolve(backend="native", device="cpu", num_envs=a.num_envs))
    delays = [float(x) for x in a.delays.split(",")]
    keys = ("rel_pos", "quat_wxyz", "lin_vel", "ang_vel", "joint_pos", "joint_vel", "delay_s", "row", "riser")
    combos = [(ckpt, row) for ckpt in a.checkpoint for row in _rows(a.rows)]
    if len(combos) == 1:
        # One (checkpoint, row): record in this process and write the part.
        ckpt, row = combos[0]
        r = record(a.task, ckpt, row, a.alpha, a.num_envs, a.seconds, delays, a.seed)
        m = int(r["row"].shape[0])
        print(f"[bank] {ckpt.name} row {row} ({r['riser'][0] if m else float('nan'):.3f} m): {m} states", flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.out.with_suffix(".npz"), joint_names=np.asarray(r["joint_names"]), **{k: r[k] for k in keys})
        if "--part" in sys.argv:
            return
        parts = [dict(np.load(a.out.with_suffix(".npz")))]
    else:
        # One process per (checkpoint, row): an environment built and closed in a
        # loop does not give its memory back (measured 2026-11-04: 6 GB after
        # four rows, then the kernel's OOM killer), so each row runs alone.
        import subprocess
        import tempfile

        parts = []
        with tempfile.TemporaryDirectory(prefix="landing-bank-") as tmp:
            for i, (ckpt, row) in enumerate(combos):
                part = Path(tmp) / f"part-{i:03d}"
                cmd = [sys.executable, __file__, "--task", a.task, "--checkpoint", str(ckpt), "--rows", str(row),
                       "--alpha", str(a.alpha), "--num-envs", str(a.num_envs), "--seconds", str(a.seconds),
                       "--delays", a.delays, "--seed", str(a.seed), "--out", str(part), "--part"]
                r = subprocess.run(cmd, capture_output=True, text=True)
                lines = [ln for ln in r.stdout.splitlines() if ln.startswith("[bank]")]
                print("\n".join(lines) or f"[bank] {ckpt.name} row {row}: no output", flush=True)
                if r.returncode != 0 or not part.with_suffix(".npz").is_file():
                    raise SystemExit(f"[bank] {ckpt.name} row {row} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
                parts.append(dict(np.load(part.with_suffix(".npz"))))
    counts = {}
    joint_names = None
    for p in parts:
        names = [str(n) for n in p["joint_names"]]
        if joint_names is None:
            joint_names = names
        assert names == joint_names, "joint order differs between runs"
        if p["row"].shape[0]:
            counts[f"row{int(p['row'][0])}"] = counts.get(f"row{int(p['row'][0])}", 0) + int(p["row"].shape[0])
    merged = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    total = int(merged["row"].shape[0])
    if total == 0:
        raise SystemExit("[bank] no landing state was recorded; nothing written")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out.with_suffix(".npz"), joint_names=np.asarray(joint_names), **merged)
    meta = {
        "task": a.task,
        "checkpoints": {c.name: hashlib.sha256(c.read_bytes()).hexdigest() for c in a.checkpoint},
        "rows": a.rows, "alpha": a.alpha, "num_envs": a.num_envs, "seconds": a.seconds,
        "delays_s": delays, "seed": a.seed, "states": total, "per_run": counts,
        "date": date.today().isoformat(), "command": " ".join(Path(v).name if v.endswith(".pt") else v for v in sys.argv),
        "rel_pos_mean": [float(v) for v in merged["rel_pos"].mean(axis=0)],
        "rel_pos_std": [float(v) for v in merged["rel_pos"].std(axis=0)],
        "joint_vel_rms": float(np.sqrt((merged["joint_vel"] ** 2).mean())),
    }
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[bank] wrote {a.out.with_suffix('.npz')}: {total} states, "
          f"base {meta['rel_pos_mean'][0]:+.3f} m from the tread centre (sd {meta['rel_pos_std'][0]:.3f}), "
          f"{meta['rel_pos_mean'][2]:.3f} m over it, joint speed RMS {meta['joint_vel_rms']:.2f} rad/s")


if __name__ == "__main__":
    main()
