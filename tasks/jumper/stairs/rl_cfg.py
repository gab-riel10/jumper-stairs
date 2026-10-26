"""PPO hyperparameters for jumper.stairs: jumper.ref_free_jump's, which the
warm-started actor was trained under, with two changes named below."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline

#: Environment steps per policy update. A rollout is a duration, and this task
#: runs at 200 Hz: 48 steps is 0.24 s, `jumper.ref_free_jump`'s choice for the
#: same reason. A bound -- crouch, push, flight, landing -- is about 0.5 s, so a
#: rollout sees half of one; GAE at `lam` joins the halves.
NUM_STEPS_PER_ENV = 48


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    return jumper_ppo_baseline(
        experiment_name="jumper.stairs",
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        # 0.15 rad of exploration at scale 0.5, as the warm-started actor had.
        init_std=0.3,
        entropy_coef=0.001,
        # Corrected for 200 Hz as jumper.ref_free_jump's are: 2 s of horizon.
        gamma=0.99**0.25,
        lam=0.95**0.25,
        num_steps_per_env=NUM_STEPS_PER_ENV,
        # **Lower than the baseline's 1e-3**: the actor starts from a policy that
        # already jumps, and the critic from nothing. The first updates are driven
        # by a value function that is noise, and at 1e-3 they can take the jump
        # apart before the critic has caught up. The schedule is adaptive, so the
        # rate climbs on its own once the KL allows it.
        learning_rate=3.0e-4,
        # Runs are cut by the clock on a free GPU; this is a ceiling, not a plan.
        max_iterations=20_000,
    )
