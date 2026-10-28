"""Record a stairs policy's jumps and keep the best few as trajectories a
renderer can play back without the simulator.

Runs the policy's mean action on `--num-envs` robots, one curriculum row, one
layout (`--layout curb`: a single riser and 1.5 m of flat top), native backend,
CPU, for `--seconds` from the landing. The spawn's own jitter is what makes the
attempts differ. Per robot, until its first termination, every control step:

- `qpos`: the whole model's generalized position (free joint first: x y z,
  then the quaternion w x y z; then every hinge, in `joint_names` order);
- `qvel`, and `body_pos` / `body_quat` (world, w x y z) of every body, so a
  renderer that only has meshes can pose them without MuJoCo.

A jump is kept when the robot got onto the top (base over tread 1, grounded),
never fell, and was measured for at least `--settle-s` + 1 s after landing.
The kept ones are ranked by how still they stand after the landing (RMS joint
speed from `--settle-s` after touchdown, lower first), then by apex, and the
first `--keep` are written to `--out`:

    <out>/jump_01.npz ... jump_NN.npz   arrays (float32), see above, plus `time`
    <out>/jump_01.json ...              per-jump metadata: timings, apex, landing x
    <out>/model.json                    the scene: names, the robot's visible
                                        meshes (an STL of this repository and its
                                        pose in its body), the kerb's geometry,
                                        timestep, filter, policy

Coordinates are the simulator's world frame, metres, +x up the step, +z up.
The kerb's riser is the plane `x = riser_x` in `model.json`.

Example (CPU)::

    python tools/stairs_record.py --task jumper.stairs_free \\
        --checkpoint logs/.../model_6750.pt --row 9 --layout curb --alpha 0.3 \\
        --num-envs 32 --keep 5 --out trajectories/curb15_alpha0.3
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
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


def _names(model, kind, count) -> list[str]:
    import mujoco

    return [mujoco.mj_id2name(model, kind, i) or "" for i in range(count)]


def _scene_description(model, prefix: str, repo: Path) -> dict:
    """The robot's visible geoms, each as a mesh file of this repository and the
    pose of **that file's own frame** in its body's frame.

    `model` is the robot's own MJCF compiled alone, not the simulation's: the
    native backend strips visual meshes from a model of many environments
    (`native_sim._maybe_strip_visuals`), and a recording of 32 robots came out
    with none. Body names get the scene's `prefix` (mjlab's `robot/`).

    MuJoCo re-centres a mesh on compile (`mesh_pos`, `mesh_quat`) and folds the
    shift into `geom_pos` / `geom_quat`, so the compiled geom pose applied to the
    STL as it is on disk puts the part in the wrong place. The pose written here
    undoes that: `v_body = R(file_quat) v_file + file_pos`. Checked against the
    compiled vertices to 3e-9 m (jumper.xml, four meshes, 2026-10-28)."""
    import mujoco

    def mat(q):
        r = np.zeros(9)
        mujoco.mju_quat2Mat(r, np.asarray(q, dtype=float))
        return r.reshape(3, 3)

    bodies = _names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    is_mesh = model.geom_type == mujoco.mjtGeom.mjGEOM_MESH
    visual = [g for g in range(model.ngeom) if is_mesh[g] and model.geom_group[g] == 2]
    geoms = []
    for g in visual:
        i = int(model.geom_dataid[g])
        raw = model.paths[int(model.mesh_pathadr[i]):].split(b"\0")[0].decode()
        rot = mat(model.geom_quat[g]) @ mat(model.mesh_quat[i]).T
        pos = model.geom_pos[g] - rot @ model.mesh_pos[i]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rot.reshape(-1))
        found = sorted(f for f in repo.joinpath("assets").rglob(Path(raw).name)
                       if f.as_posix().endswith(Path(raw).as_posix().lstrip("./")))
        geoms.append({
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "",
            "body": prefix + bodies[int(model.geom_bodyid[g])],
            "file": str(found[0].relative_to(repo)) if found else raw,
            "scale": [float(v) for v in model.mesh_scale[i]],
            "file_pos": [float(v) for v in pos],
            "file_quat_wxyz": [float(v) for v in quat],
            "rgba": [float(v) for v in model.geom_rgba[g]],
        })
    return {"units": "metres", "geoms": geoms}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--task", default="jumper.stairs_free")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--row", default="9")
    p.add_argument("--layout", default="curb", choices=("curb", "flight"))
    p.add_argument("--alpha", type=float, default=0.3, help="the joint-target low-pass; 1 is none")
    p.add_argument("--noise", default="on", choices=("on", "off"), help="observation noise as in training")
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--settle-s", type=float, default=0.5)
    p.add_argument("--keep", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    import torch

    import tasks
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from mjrl.backend.resolve import resolve
    from mjrl.backend.select import use_backend

    use_backend(resolve(backend="native", device="cpu", num_envs=a.num_envs))
    task_args = {"stairs_row": a.row, "action_ema": str(a.alpha), "stairs_layout": a.layout}
    cfg = tasks.load_env_cfg(a.task, play=True, task_args=task_args)
    if a.noise == "on":
        _match_training_noise(cfg, tasks.load_env_cfg(a.task, task_args=task_args))
    cfg.scene.num_envs = a.num_envs
    cfg.seed = a.seed
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        agent = tasks.load_agent_cfg(a.task)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = MjlabOnPolicyRunner(wrapped, asdict(agent), device="cpu")
        runner.load(str(a.checkpoint), load_cfg={"actor": True}, strict=True, map_location="cpu")
        policy = runner.get_inference_policy(device="cpu")

        import mujoco

        model = env.sim.mj_model
        cmd = env.command_manager.get_term("stairs")
        robot = env.scene["robot"]
        gait_ids = cmd._gait_ids
        dt = env.step_dt
        steps = int(round(a.seconds / dt))
        n = a.num_envs

        root = int(robot.data.indexing.root_body_id)
        robot_bodies = {b for b in range(model.nbody) if int(model.body_rootid[b]) == int(model.body_rootid[root])}
        scene_body_names = _names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
        from tasks.jumper.common.assets import JUMPER_XML

        root_name = mujoco.mj_id2name(mujoco.MjModel.from_xml_path(str(JUMPER_XML)), mujoco.mjtObj.mjOBJ_BODY, 1)
        body_ids = sorted(robot_bodies)

        alive = torch.ones(n, dtype=torch.bool)
        fell = torch.zeros(n, dtype=torch.bool)
        t_land = torch.full((n,), float("nan"))
        t_takeoff = torch.full((n,), float("nan"))
        x_land = torch.full((n,), float("nan"))
        apex = torch.zeros(n)
        was_flying = torch.zeros(n, dtype=torch.bool)
        qpos, qvel, bpos, bquat, qd, live_log = [], [], [], [], [], []

        def snapshot():
            data = env.sim.data
            qpos.append(data.qpos.clone().float().numpy())
            qvel.append(data.qvel.clone().float().numpy())
            bpos.append(data.xpos[:, body_ids].clone().float().numpy())
            bquat.append(data.xquat[:, body_ids].clone().float().numpy())

        obs = wrapped.get_observations()
        snapshot()
        live_log.append(alive.clone())
        qd.append(torch.zeros(n, len(gait_ids)))
        with torch.no_grad():
            for i in range(steps):
                obs, _, dones, _ = wrapped.step(policy(obs))
                done = dones.bool().cpu() & alive
                live_now = alive & ~done
                flying = cmd.in_flight.cpu()
                # The start of the flight that ends in the first landing on the top:
                # every takeoff before it (the drop from the spawn, a hop) is overwritten.
                newly_up = flying & ~was_flying & torch.isnan(t_land) & live_now
                t_takeoff = torch.where(newly_up, torch.full_like(t_takeoff, (i + 1) * dt), t_takeoff)
                was_flying = flying.clone()
                apex = torch.where(live_now, torch.maximum(apex, cmd.apex.cpu()), apex)
                on1 = (cmd.k_base.cpu() >= 1) & ~flying & live_now
                newly = on1 & torch.isnan(t_land)
                t_land = torch.where(newly, torch.full_like(t_land, (i + 1) * dt), t_land)
                x_land = torch.where(newly, cmd.base_x().cpu().float(), x_land)
                fell |= done & env.reset_terminated.cpu()
                alive = live_now
                snapshot()  # on a done step this is the reset state: masked by `live`
                live_log.append(alive.clone())
                qd.append(robot.data.joint_vel[:, gait_ids].clone().cpu())

        live = torch.stack(live_log).numpy()  # (T+1, N)
        qd_t = torch.stack(qd)  # (T+1, N, J)
        t = np.arange(steps + 1) * dt

        scored = []
        for e in range(n):
            if fell[e] or torch.isnan(t_land[e]):
                continue
            end = int(live[:, e].sum())  # frames before the first termination
            start = int(round((float(t_land[e]) + a.settle_s) / dt))
            if end - start < int(round(1.0 / dt)):
                continue
            rms = float(qd_t[start:end, e].pow(2).mean().sqrt())
            scored.append((rms, -float(apex[e]), e, end))
        scored.sort()
        kept = scored[: a.keep]

        a.out.mkdir(parents=True, exist_ok=True)
        stairs = cmd.stairs
        riser_m = float(cmd.riser[0])
        # All robots share the one pinned tile, so one origin serves every jump.
        origin = env.scene.env_origins[0].cpu().numpy().astype(float)
        riser_x = float(origin[0] + stairs.first_riser_from_origin)
        ckpt_sha = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
        scene = {
            "format": "jumper-stairs-trajectory/1",
            "task": a.task, "checkpoint": a.checkpoint.name, "checkpoint_sha256": ckpt_sha,
            "layout": a.layout, "row": int(a.row), "riser_m": riser_m,
            "action_ema_alpha": a.alpha, "obs_noise": a.noise == "on",
            "control_dt_s": dt, "physics_dt_s": float(model.opt.timestep),
            "frames_per_second": 1.0 / dt,
            "nq": int(model.nq), "nv": int(model.nv),
            "joint_names": _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
            "joint_qpos_adr": [int(v) for v in model.jnt_qposadr],
            "joint_types": [mujoco.mjtJoint(int(v)).name.removeprefix("mjJNT_").lower() for v in model.jnt_type],
            "body_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in body_ids],
            "kerb": {
                "riser_x": riser_x, "height_m": riser_m, "ground_z": float(origin[2]),
                "top_depth_m": stairs.num_steps * stairs.tread + stairs.top,
                "landing_depth_m": stairs.landing, "width_m": stairs.width,
                "center_y": float(origin[1]),
            },
            "robot": _scene_description(mujoco.MjModel.from_xml_path(str(JUMPER_XML)),
                                        scene_body_names[root].removesuffix(root_name),
                                        Path(__file__).resolve().parents[1]),
            "jumps": [],
        }
        unknown = {g["body"] for g in scene["robot"]["geoms"]} - set(scene["body_names"])
        if unknown or not scene["robot"]["geoms"]:
            raise RuntimeError(f"visible meshes on bodies the recording does not have: {sorted(unknown)[:5]}")
        for rank, (rms, neg_apex, e, end) in enumerate(kept, start=1):
            name = f"jump_{rank:02d}"
            np.savez_compressed(
                a.out / f"{name}.npz",
                time=t[:end].astype(np.float32),
                qpos=np.stack([q[e] for q in qpos[:end]]).astype(np.float32),
                qvel=np.stack([q[e] for q in qvel[:end]]).astype(np.float32),
                body_pos=np.stack([q[e] for q in bpos[:end]]).astype(np.float32),
                body_quat=np.stack([q[e] for q in bquat[:end]]).astype(np.float32),
            )
            x_rest = float(qpos[end - 1][e][0]) - riser_x
            meta = {
                "name": name, "env": e, "frames": end, "duration_s": end * dt,
                "takeoff_s": float(t_takeoff[e]), "landing_s": float(t_land[e]),
                "apex_cm": -neg_apex * 100.0,
                "landing_past_riser_m": float(x_land[e]) - float(stairs.first_riser_from_origin),
                "rest_past_riser_m": x_rest,
                "post_landing_qd_rms": rms,
            }
            (a.out / f"{name}.json").write_text(json.dumps(meta, indent=2))
            scene["jumps"].append(meta)
        attempts = {
            "robots": n, "on_top": int((~torch.isnan(t_land)).sum()), "fell": int(fell.sum()),
            "kept": len(kept),
        }
        scene["attempts"] = attempts
        (a.out / "model.json").write_text(json.dumps(scene, indent=2))
        print(f"[record] {a.task} row {a.row} ({riser_m * 100:.1f} cm) {a.layout} alpha {a.alpha:g} "
              f"noise {a.noise}: {attempts['on_top']}/{n} on top, {attempts['fell']} fell; kept {len(kept)}")
        for j in scene["jumps"]:
            print(f"  {j['name']}: env {j['env']:2d} takeoff {j['takeoff_s']:.2f} s land {j['landing_s']:.2f} s "
                  f"apex {j['apex_cm']:.1f} cm landed {j['landing_past_riser_m']:.2f} m past the riser, "
                  f"rests at {j['rest_past_riser_m']:.2f} m "
                  f"qd_rms {j['post_landing_qd_rms']:.3f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
