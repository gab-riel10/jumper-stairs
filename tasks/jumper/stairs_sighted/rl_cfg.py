"""PPO hyperparameters for jumper.stairs_sighted: jumper.stairs', under this
task's own experiment name. The two tasks are each other's control, so they
must train alike; `..stairs.rl_cfg` has the reasons for each number."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline
from ..stairs.rl_cfg import NUM_STEPS_PER_ENV


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    return jumper_ppo_baseline(
        experiment_name="jumper.stairs_sighted",
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        init_std=0.3,
        entropy_coef=0.001,
        gamma=0.99**0.25,
        lam=0.95**0.25,
        num_steps_per_env=NUM_STEPS_PER_ENV,
        learning_rate=3.0e-4,
        max_iterations=20_000,
    )
