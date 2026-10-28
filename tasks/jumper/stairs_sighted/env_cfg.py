"""jumper.stairs_sighted -- `jumper.stairs` with `sighted=True`. See there."""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg

from ..stairs.env_cfg import env_cfg as stairs_env_cfg


def env_cfg(asset: Path | None = None, play: bool = False, stairs_row: str = "",
            action_ema: str = "") -> ManagerBasedRlEnvCfg:
    return stairs_env_cfg(asset=asset, play=play, sighted=True, stairs_row=stairs_row, action_ema=action_ema)


__all__ = ["env_cfg"]
