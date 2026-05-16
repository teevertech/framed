"""Train a MaskablePPO agent on the panel-sequencing environment.

Usage
-----
Run with defaults::

    python scripts/train.py

Override any ``TrainConfig`` field with ``key=value`` arguments::

    python scripts/train.py learning_rate=1e-4 collision_penalty_multiplier=4.0

All valid keys are the fields of ``framed.config.TrainConfig``.  The run
is fully reproducible from the printed config: every key=value pair that
was passed (or defaulted) is logged to aim as a hyperparameter.

Checkpoints are saved to ``{checkpoint_dir}/{run_name}/`` every
``checkpoint_freq`` timesteps and at the end of training.  The final
model is also written there as ``final_model.zip``.
"""
from __future__ import annotations

import os
import sys

# Allow running from the project root: ``python scripts/train.py``.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from aim import Run
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from framed.callbacks import AimTrainingCallback, EvalCallback
from framed.config import TrainConfig
from framed.env import RandomPanelEnv


# ------------------------------------------------------------------ #
# Environment factory                                                  #
# ------------------------------------------------------------------ #

def _make_env_fn(config: TrainConfig, rank: int):
    """Return a zero-argument factory for one worker env.

    Each worker gets a distinct panel generator seed (config.seed + rank)
    so parallel workers produce independent episode sequences.
    """
    def _fn():
        gen = config.make_panel_generator(seed=config.seed + rank)
        env = RandomPanelEnv(
            gen,
            robot_speed=config.robot_speed,
            collision_penalty_multiplier=config.collision_penalty_multiplier,
        )
        # ActionMasker exposes action_masks() in the form MaskablePPO expects.
        return ActionMasker(env, lambda e: e.action_masks())
    return _fn


# ------------------------------------------------------------------ #
# Core training function                                               #
# ------------------------------------------------------------------ #

def train(config: TrainConfig) -> None:
    """Run one full training job described by *config*.

    Called directly by ``main()`` for single runs, and called repeatedly
    by ``scripts/sweep.py`` for hyperparameter sweeps.
    """
    run_name = config.effective_run_name()
    ckpt_path = os.path.join(config.checkpoint_dir, run_name)
    os.makedirs(ckpt_path, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Seeding                                                              #
    # ------------------------------------------------------------------ #
    np.random.seed(config.seed)

    # ------------------------------------------------------------------ #
    # aim run                                                              #
    # ------------------------------------------------------------------ #
    aim_run = Run(repo=config.aim_repo, experiment=config.experiment_name)
    aim_run.name = run_name
    aim_run["hparams"] = config.as_dict()
    print(f"\n{'='*60}")
    print(f"  run : {run_name}")
    print(f"  hash: {config.run_id()}")
    print(f"  aim : {config.aim_repo}  (experiment: {config.experiment_name})")
    print(f"  ckpt: {ckpt_path}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ #
    # Training envs                                                        #
    # ------------------------------------------------------------------ #
    vec_env = SubprocVecEnv(
        [_make_env_fn(config, rank=i) for i in range(config.n_envs)],
        start_method="spawn",  # spawn = fresh process per worker, no inherited
                               # state from the parent. fork is faster to start
                               # but inherits aim/SB3 thread state and deadlocks
                               # on the second sequential run in sweep.py.
    )

    # ------------------------------------------------------------------ #
    # Eval panels (fixed for the life of the run)                          #
    # ------------------------------------------------------------------ #
    # Use the same generator (with the retry loop) as the training envs so
    # eval panels are guaranteed to have the same n_members. Seed 99_999
    # is well outside the training range [seed, seed+n_envs).
    eval_generator = config.make_panel_generator(seed=99_999)
    eval_panels = [eval_generator() for _ in range(config.n_eval_panels)]

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs=dict(net_arch=list(config.net_arch)),
        seed=config.seed,
        verbose=1,
    )

    # ------------------------------------------------------------------ #
    # Callbacks                                                            #
    # ------------------------------------------------------------------ #
    callbacks = CallbackList([
        AimTrainingCallback(
            aim_run=aim_run,
            n_envs=config.n_envs,
        ),
        EvalCallback(
            config=config,
            eval_panels=eval_panels,
            aim_run=aim_run,
            run_name=run_name,
            eval_freq=config.eval_freq,
            gif_dir=config.gif_dir,
            gif_fps=config.gif_fps,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=config.checkpoint_freq // config.n_envs,
            save_path=ckpt_path,
            name_prefix="model",
            verbose=1,
        ),
    ])

    # ------------------------------------------------------------------ #
    # Train                                                                #
    # ------------------------------------------------------------------ #
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=True,
            progress_bar=True,
        )
    finally:
        # Always save the final model and close workers — even on Ctrl-C.
        final_path = os.path.join(ckpt_path, "final_model")
        model.save(final_path)
        print(f"\nFinal model saved → {final_path}.zip")
        vec_env.close()
        aim_run.close()


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

def _parse_config(argv: list[str]) -> TrainConfig:
    """Parse ``key=value`` CLI arguments into a ``TrainConfig``.

    Unrecognised keys will raise ``TypeError`` from the dataclass
    constructor with a clear error message.
    """
    overrides: dict[str, str] = {}
    for arg in argv[1:]:
        if "=" not in arg:
            print(f"[warn] ignoring argument without '=': {arg!r}", flush=True)
            continue
        key, _, val = arg.partition("=")
        overrides[key.strip()] = val.strip()
    return TrainConfig.from_overrides(overrides)


def main() -> None:
    config = _parse_config(sys.argv)
    train(config)


if __name__ == "__main__":
    main()
