#!/usr/bin/env python3
"""Start one task's actor from another task's checkpoint.

    python tools/warm_start.py --task jumper.stairs \\
        --from tasks/jumper/ref_free_jump/out/example/model_5999.pt \\
        --out logs/warm/jumper.stairs/model_0.pt
    python scripts/train.py --task jumper.stairs --checkpoint logs/warm/jumper.stairs/model_0.pt ...

`scripts/train.py --checkpoint` loads **strictly**: every tensor of the actor,
the critic and the optimizer must exist with its exact shape, and the iteration
counter and the environment's step counter come along. A checkpoint from
another task fails all of that -- its critic saw another observation, its
counters belong to its own schedule -- and the failure is a wall of mismatched
keys that reads as a bug (`scripts/train.py::_resume`). This writes a
checkpoint the target task *can* load strictly:

- the target's own networks, freshly built, so every shape is the target's;
- the source actor's weights copied in wherever a tensor matches, **by name and
  by overlap**: a layer whose input is wider in the target (an observation term
  appended after the source's) gets the source's columns in its first
  `source_width` inputs and keeps its fresh initialisation for the rest, which
  is why `tasks/jumper/stairs/env_cfg.py` appends and never reorders;
- the source actor's observation normaliser the same way (mean, variance,
  standard deviation and sample count), so the copied weights read inputs
  scaled as they were trained to;
- a fresh critic and a fresh optimizer, `iter = 0`, and no environment state,
  so the target's schedules start at the beginning.

The target environment is built once on the CPU with a handful of robots, which
is the only way to know the target's observation widths. The native backend is
enough for that: nothing is stepped.

A warm start is a hypothesis, not a result: the first hundred iterations say
whether the copied jump survives the new reward. `tasks/jumper/stairs/rl_cfg.py`
lowers the learning rate for that window.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _graft(target: dict, source: dict, label: str) -> list[str]:
    """Copy `source` tensors into `target` by name, over the overlapping slice of
    every dimension. Returns one line per tensor saying what happened."""

    lines = []
    for name, dst in target.items():
        src = source.get(name)
        if src is None:
            lines.append(f"  {label}.{name}: not in source, kept fresh")
            continue
        if dst.shape == src.shape:
            dst.copy_(src)
            lines.append(f"  {label}.{name}: copied {tuple(dst.shape)}")
            continue
        if dst.dim() != src.dim():
            lines.append(f"  {label}.{name}: rank differs {tuple(src.shape)} -> {tuple(dst.shape)}, kept fresh")
            continue
        overlap = tuple(slice(0, min(a, b)) for a, b in zip(dst.shape, src.shape, strict=True))
        dst[overlap] = src[overlap]
        lines.append(f"  {label}.{name}: source {tuple(src.shape)} into target {tuple(dst.shape)}, overlap copied")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="the target task id, e.g. jumper.stairs")
    parser.add_argument("--from", dest="source", type=Path, required=True, help="source model_*.pt")
    parser.add_argument("--out", type=Path, required=True, help="where to write model_0.pt")
    parser.add_argument("--model", default="jumper")
    parser.add_argument("--num-envs", type=int, default=4, help="robots in the throwaway build")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not args.source.is_file():
        raise SystemExit(f"[warm_start] no such checkpoint: {args.source}")

    import torch

    import tasks
    from mjrl.backend import resolve
    from mjrl.backend.select import use_backend

    res = resolve(backend="native" if args.device == "cpu" else "warp", device=args.device,
                  num_envs=args.num_envs, training=False)
    use_backend(res)

    from dataclasses import asdict

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner

    spec = tasks.get(args.task)
    asset = spec.resolve_asset(args.model)
    env_cfg = tasks.load_env_cfg(spec.id, asset=asset)
    env_cfg.scene.num_envs = args.num_envs
    agent_cfg = tasks.load_agent_cfg(spec.id)
    agent_cfg.logger = "tensorboard"
    runner_cls = tasks.load_runner_cls(spec.id) or MjlabOnPolicyRunner

    scratch = args.out.parent / "_warm_start_build"
    scratch.mkdir(parents=True, exist_ok=True)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=res.device)
    try:
        runner = runner_cls(RslRlVecEnvWrapper(env), asdict(agent_cfg), str(scratch), res.device)
        fresh = runner.alg.save()
    finally:
        env.close()

    loaded = torch.load(args.source, map_location="cpu", weights_only=False)
    source_actor = loaded.get("actor_state_dict")
    if source_actor is None:
        raise SystemExit("[warm_start] the source has no actor_state_dict; is it an rsl_rl 5.x checkpoint?")
    # rsl_rl 4.x named the std differently; the runner migrates on load, so do the same here.
    if "std" in source_actor:
        source_actor["distribution.std_param"] = source_actor.pop("std")
    if "log_std" in source_actor:
        source_actor["distribution.log_std_param"] = source_actor.pop("log_std")

    actor = {k: v.clone() for k, v in fresh["actor_state_dict"].items()}
    report = _graft(actor, {k: v.to("cpu") for k, v in source_actor.items()}, "actor")

    out = {
        "actor_state_dict": actor,
        "critic_state_dict": fresh["critic_state_dict"],
        "optimizer_state_dict": fresh["optimizer_state_dict"],
        "iter": 0,
        "infos": {"warm_start": {"task": spec.id, "source": str(args.source),
                                 "source_iter": loaded.get("iter")}},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)

    widths = {k: tuple(v.shape) for k, v in actor.items() if k.endswith("mlp.0.weight") or k == "obs_normalizer._mean"}
    print("[warm_start] target actor input widths:", widths)
    print("\n".join(report))
    print(f"[warm_start] wrote {args.out} (iter 0, fresh critic and optimizer)")
    print(f"[warm_start] next: python scripts/train.py --task {spec.id} --checkpoint {args.out} ...")


if __name__ == "__main__":
    # No sys.path work: `pip install -e .` is what makes `tasks` and `mjrl`
    # importable here, as everywhere else in this repository (CLAUDE.md).
    main()
