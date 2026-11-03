"""Evaluate a stairs policy by replay: the jump, the landing, the stillness
after it, and what the policy asks of the joints -- with and without
observation noise and through the joint-target low-pass of
`tasks/jumper/stairs/mdp/actions.py`.

This is the stairs' counterpart of `tools/checks/jump_eval.py`, which is wired
to `jumper.jump` (its reference event, its reference `.npz`, `cuda:0`) and
cannot run a stairs policy. The three numbers it reports are measured here the
same way, so they compare with `jumper.jump`'s table and with
`jumper.ref_free_jump`'s (`docs/jumper_jump.md`): **base rise** is the base's
highest point above where it started, **airborne** the time every foot read
under 1 N, **push-off vz** the base's fastest upward speed while a foot was
loaded. Everything else is the stairs':

Every combination of `--noise` and `--alphas` runs the policy's mean action
(no exploration noise) on `--num-envs` robots, all on one curriculum row, from
the landing, native backend, with the reset randomisation the task trains with
(foot friction, base mass and centre of mass; pushes are off in play). Per
robot, until its first termination:

- **climbed**: the base got over tread 1 (the first riser jumped or walked);
- **apex_cm**: the highest the lowest point rose above the takeoff tread
  (`StairsCommand.lowest_rise`, the stairs' own measure);
- **base_rise_cm**, **airborne_s**, **vz_takeoff**: `jump_eval.py`'s three;
- **fell**: the episode ended by a termination that was not the time-out;
- **tremble**, measured from `--settle-s` after the first landing on tread 1
  (or after `--settle-s` + 1 s from the start when it never got there) to the
  end: `qd_rms` the RMS joint speed over the 20 gait joints (rad/s),
  `dtarget_rms` the RMS change of the joint target per control step (mrad),
  `hf_share` the share of joint-speed power above `--hf-hz`.
  The window is Crabcraft's (kubony/crabcraft, `tools/eval_calm.py`, as its
  README describes it: joint speed RMS from 0.5 s after the landing, 16
  rollouts, native MuJoCo), so the numbers compare with theirs;
- **beyond the joint limits**: the share of (step, joint) pairs in which the
  target handed to the actuators lay outside the joint's range
  (`beyond_tgt_share`, whole episode; `beyond_tgt_share_post`, the tremble
  window), the same for the policy's raw request before the low-pass
  (`beyond_raw_share`), and the largest excursion past a limit in radians
  (`overshoot_max_rad`). Training writes the target as it is; the robot's
  controller clamps it to the limits (`deploy/fsm/src/action.rs`), so a policy
  that leans past them is a policy the robot runs differently.

A robot standing still at HOME reads a few hundredths of a rad/s; a tremble
reads tenths and puts most of its power above a few hertz.

Example (CPU)::

    python tools/stairs_tremor.py --task jumper.stairs_free \\
        --checkpoint logs/.../model_6750.pt --row 9 --num-envs 16 --seconds 6
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path


def _match_training_noise(env_cfg, train_cfg) -> None:
    """`scripts/play.py`'s `--obs-noise`: copy the corruption flags and every
    term's noise from the training build of the same task."""
    for name, group in env_cfg.observations.items():
        reference = train_cfg.observations.get(name)
        if reference is None:
            continue
        group.enable_corruption = reference.enable_corruption
        for term_name, term in group.terms.items():
            source = reference.terms.get(term_name)
            if source is not None:
                term.noise = source.noise


def run_one(task: str, checkpoint: Path, row: str, alpha: float, noise: bool, num_envs: int,
            seconds: float, settle_s: float, hf_hz: float, layout: str = "", seed: int = 0) -> dict:
    import torch

    import tasks
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner

    task_args = {"stairs_row": row, "action_ema": str(alpha), "stairs_layout": layout}
    cfg = tasks.load_env_cfg(task, play=True, task_args=task_args)
    if noise:
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
        feet = env.scene.sensors["feet_ground_contact"]
        term = env.action_manager.get_term("joint_pos")
        gait_ids = cmd._gait_ids
        gait_names = [robot.joint_names[int(i)] for i in gait_ids]
        dt = env.step_dt
        steps = int(round(seconds / dt))
        limits = robot.data.joint_pos_limits[:, gait_ids]  # (N, J, 2)
        lo, hi = limits[..., 0], limits[..., 1]

        alive = torch.ones(num_envs, dtype=torch.bool)
        fell = torch.zeros(num_envs, dtype=torch.bool)
        climbed = torch.zeros(num_envs, dtype=torch.bool)
        t_land = torch.full((num_envs,), float("nan"))
        apex = torch.zeros(num_envs)
        z0 = robot.data.root_link_pos_w[:, 2].clone()
        max_z = z0.clone()
        airborne_s = torch.zeros(num_envs)
        vz_takeoff = torch.zeros(num_envs)
        qd_log, dtar_log, alive_log, beyond_tgt_log, beyond_raw_log, over_log = [], [], [], [], [], []
        prev_target = term._processed_actions.clone()

        obs = wrapped.get_observations()
        with torch.no_grad():
            for i in range(steps):
                obs, _, dones, _ = wrapped.step(policy(obs))
                target = term._processed_actions.clone()  # after the low-pass: what the actuators got
                raw_target = term._raw_actions * term._scale + term._offset  # the policy's request
                qd = robot.data.joint_vel[:, gait_ids].clone()
                dtar = (target - prev_target)
                prev_target = target
                live_now = alive & ~(dones.bool().cpu())
                apex = torch.where(live_now, torch.maximum(apex, cmd.apex), apex)
                z = robot.data.root_link_pos_w[:, 2]
                max_z = torch.where(live_now, torch.maximum(max_z, z), max_z)
                # `jump_eval.py`'s reading of the feet, verbatim: every foot under 1 N.
                airborne = (feet.data.force.norm(dim=-1) <= 1.0).all(dim=1)
                airborne_s += (airborne & live_now).float() * dt
                vz = robot.data.root_link_lin_vel_w[:, 2]
                loaded = ~airborne & live_now
                vz_takeoff = torch.where(loaded, torch.maximum(vz_takeoff, vz), vz_takeoff)
                on1 = (cmd.k_base >= 1) & ~cmd.in_flight & live_now
                newly = on1 & torch.isnan(t_land)
                t_land = torch.where(newly, torch.full_like(t_land, (i + 1) * dt), t_land)
                climbed |= on1
                done = dones.bool().cpu() & alive
                # `reset_terminated` leaves the time-outs out (`reached_top` is one).
                fell |= done & env.reset_terminated.cpu()
                # On a done step the env has already reset: not part of the window.
                alive_log.append(alive & ~done)
                qd_log.append(qd)
                dtar_log.append(dtar)
                beyond_tgt_log.append((target < lo) | (target > hi))
                beyond_raw_log.append((raw_target < lo) | (raw_target > hi))
                over_log.append(torch.maximum(lo - target, target - hi).clamp(min=0.0))
                alive &= ~done

        qd = torch.stack(qd_log)  # (T, N, J)
        dtar = torch.stack(dtar_log)
        beyond_tgt = torch.stack(beyond_tgt_log)  # (T, N, J)
        beyond_raw = torch.stack(beyond_raw_log)
        over = torch.stack(over_log)
        live = torch.stack(alive_log)  # (T, N)
        t = (torch.arange(steps) + 1) * dt
        start = torch.where(torch.isnan(t_land), torch.full_like(t_land, settle_s + 1.0), t_land + settle_s)
        window = (t[:, None] >= start[None, :]) & live  # (T, N)

        per_env = []
        for n in range(num_envs):
            w = window[:, n]
            if int(w.sum()) < 40:
                per_env.append(None)
                continue
            q = qd[w, n]  # (Tw, J)
            d = dtar[w, n]
            qd_rms = float(q.pow(2).mean().sqrt())
            dt_rms = float(d.pow(2).mean().sqrt()) * 1000.0
            spec = torch.fft.rfft(q - q.mean(dim=0), dim=0).abs().pow(2).sum(dim=1)
            freqs = torch.fft.rfftfreq(q.shape[0], d=dt)
            total = float(spec[1:].sum())
            hf = float(spec[freqs > hf_hz].sum()) / total if total > 0 else 0.0
            per_env.append((qd_rms, dt_rms, hf, float(beyond_tgt[w, n].float().mean())))

        ok = [p for p in per_env if p is not None]

        def mean(i):
            return sum(p[i] for p in ok) / len(ok) if ok else float("nan")

        live_pairs = live[..., None].expand_as(beyond_tgt)
        n_live = float(live_pairs.sum().clamp(min=1))
        beyond_tgt_share = float((beyond_tgt & live_pairs).sum()) / n_live
        beyond_raw_share = float((beyond_raw & live_pairs).sum()) / n_live
        overshoot = float(torch.where(live_pairs, over, torch.zeros_like(over)).max())
        per_joint = (beyond_tgt & live_pairs).float().sum(dim=(0, 1)) / float(live.sum().clamp(min=1))
        worst = sorted(zip(per_joint.tolist(), gait_names), reverse=True)[:3]

        return {
            "noise": noise, "alpha": alpha, "envs": num_envs, "seed": seed,
            "climbed": float(climbed.float().mean()), "fell": float(fell.float().mean()),
            "apex_cm": float(apex.mean()) * 100.0,
            "base_rise_cm": float((max_z - z0).mean()) * 100.0,
            "airborne_s": float(airborne_s.mean()),
            "vz_takeoff": float(vz_takeoff.mean()),
            "t_land_s": float(torch.nanmean(t_land)) if bool((~torch.isnan(t_land)).any()) else float("nan"),
            "qd_rms": mean(0), "dtarget_rms_mrad": mean(1), "hf_share": mean(2), "measured_envs": len(ok),
            "beyond_tgt_share": beyond_tgt_share, "beyond_tgt_share_post": mean(3),
            "beyond_raw_share": beyond_raw_share, "overshoot_max_rad": overshoot,
            "beyond_worst_joints": [{"joint": name, "share": share} for share, name in worst],
        }
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--task", default="jumper.stairs_free")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--row", default="9")
    p.add_argument("--layout", default="", choices=("", "flight", "curb"),
                   help="flight (empty, eight treads) or curb (one riser, 1.5 m of top)")
    p.add_argument("--alphas", default="1,0.5,0.3,0.2,0.1")
    p.add_argument("--noise", default="on,off", help="on, off, or on,off")
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--settle-s", type=float, default=0.5)
    p.add_argument("--hf-hz", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="write the rows as JSON")
    a = p.parse_args()

    import tasks  # noqa: F401 -- registers the tasks
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    use_backend(resolve(backend="native", device="cpu", num_envs=a.num_envs))

    rows = []
    for noise in [s.strip() == "on" for s in a.noise.split(",")]:
        for alpha in [float(x) for x in a.alphas.split(",")]:
            r = run_one(a.task, a.checkpoint, a.row, alpha, noise, a.num_envs, a.seconds,
                        a.settle_s, a.hf_hz, a.layout, a.seed)
            rows.append(r)
            worst = ", ".join(f"{j['joint']} {100 * j['share']:.0f}%" for j in r["beyond_worst_joints"])
            print(f"noise {'on ' if noise else 'off'} alpha {alpha:<4g} | climbed {r['climbed']:.2f}"
                  f" fell {r['fell']:.2f} apex {r['apex_cm']:5.1f} cm land {r['t_land_s']:.2f} s"
                  f" | base rise {r['base_rise_cm']:5.1f} cm airborne {r['airborne_s']:.2f} s"
                  f" push-off vz {r['vz_takeoff']:.2f} m/s"
                  f" | qd_rms {r['qd_rms']:.3f} rad/s  dtarget {r['dtarget_rms_mrad']:6.2f} mrad"
                  f"  >{a.hf_hz:g}Hz {r['hf_share']:.2f}  (n={r['measured_envs']})"
                  f" | beyond limits: target {100 * r['beyond_tgt_share']:.1f} %"
                  f" (post-landing {100 * r['beyond_tgt_share_post']:.1f} %), raw"
                  f" {100 * r['beyond_raw_share']:.1f} %, max {r['overshoot_max_rad']:.3f} rad [{worst}]",
                  flush=True)
    if a.out:
        a.out.write_text(json.dumps(rows, indent=2, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x))


if __name__ == "__main__":
    main()
