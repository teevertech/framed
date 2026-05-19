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

Output layout
-------------
All run artefacts land under ``models/{run_name}/``::

    models/
    └── {run_name}/
        ├── run_metadata.json
        ├── final_model.zip
        ├── final_model.onnx
        ├── checkpoints/
        │   ├── model_{step}_steps.zip
        │   └── ...
        └── gifs/
            └── step_{N:08d}.gif
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone

# Allow running from the project root: ``python scripts/train.py``.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from aim import Run
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from framed.baselines import greedy_nearest_action, run_episode
from framed.callbacks import AimTrainingCallback, EvalCallback
from framed.config import TrainConfig
from framed.env import MAX_MEMBERS, PanelEnv, RandomPanelEnv
from framed.panel import Panel


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
# Post-training evaluation                                             #
# ------------------------------------------------------------------ #

def _compute_eval_summary(
    model: MaskablePPO,
    config: TrainConfig,
    panels: list[Panel],
) -> dict:
    """Run the trained policy and greedy-nearest on *panels* and summarise.

    Returns a dict matching the ``eval_summary`` field of ``run_metadata.json``.
    Panels are evaluated one at a time (no vectorisation needed here — this
    runs once at the end of training).
    """
    nearest_rewards: list[float] = []
    policy_rewards:  list[float] = []

    for panel in panels:
        env = PanelEnv(
            panel,
            robot_speed=config.robot_speed,
            collision_penalty_multiplier=config.collision_penalty_multiplier,
        )

        # Greedy nearest baseline.
        nearest_rewards.append(run_episode(env, greedy_nearest_action)[0])

        # Trained policy (deterministic).
        obs, _ = env.reset()
        total = 0.0
        for _ in range(env.n_members):
            action, _ = model.predict(
                obs, action_masks=env.action_masks(), deterministic=True
            )
            obs, reward, terminated, _, _ = env.step(int(action))
            total += float(reward)
            if terminated:
                break
        policy_rewards.append(total)

    improvements = [
        (p - n) / abs(n) * 100.0 if n != 0.0 else 0.0
        for p, n in zip(policy_rewards, nearest_rewards)
    ]
    wins = sum(p > n for p, n in zip(policy_rewards, nearest_rewards))

    return {
        "n_panels":             len(panels),
        "win_rate":             wins,
        "mean_improvement_pct": round(float(np.mean(improvements)), 1),
        "min_improvement_pct":  round(float(np.min(improvements)), 1),
        "max_improvement_pct":  round(float(np.max(improvements)), 1),
        "mean_policy_reward":   round(float(np.mean(policy_rewards)), 1),
        "mean_nearest_reward":  round(float(np.mean(nearest_rewards)), 1),
    }


def _collect_checkpoint_entries(ckpt_dir: str) -> list[dict]:
    """Scan *ckpt_dir* for SB3 checkpoint zips and return metadata entries.

    SB3's ``CheckpointCallback`` produces files named
    ``model_{timestep}_steps.zip``.  We parse the timestep from the filename
    and sort chronologically.
    """
    entries = []
    if not os.path.isdir(ckpt_dir):
        return entries
    pattern = re.compile(r"model_(\d+)_steps\.zip$")
    for fname in os.listdir(ckpt_dir):
        m = pattern.match(fname)
        if m:
            timestep = int(m.group(1))
            name = f"step_{timestep:06d}"
            entries.append({"name": name, "timestep": timestep})
    entries.sort(key=lambda e: e["timestep"])
    return entries


# ------------------------------------------------------------------ #
# Core training function                                               #
# ------------------------------------------------------------------ #

def train(config: TrainConfig) -> None:
    """Run one full training job described by *config*.

    Called directly by ``main()`` for single runs, and called repeatedly
    by ``scripts/sweep.py`` for hyperparameter sweeps.
    """
    run_name = config.effective_run_name()
    run_dir  = os.path.join(config.checkpoint_dir, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(run_dir,  exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    created_at = datetime.now(timezone.utc).isoformat()

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
    print(f"  out : {run_dir}")
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
    # Seed 99_999 is well outside the training range [seed, seed+n_envs).
    # Padding handles variable member counts across panels, so no member-
    # count matching is needed here.
    eval_generator = config.make_panel_generator(seed=99_999)
    eval_panels = [eval_generator() for _ in range(config.n_eval_panels)]

    # ------------------------------------------------------------------ #
    # PyTorch threading                                                    #
    # ------------------------------------------------------------------ #
    if config.torch_threads > 0:
        torch.set_num_threads(config.torch_threads)
        print(f"  torch threads: {config.torch_threads}")

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
        device=config.device,
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
            save_path=ckpt_dir,
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
        final_zip_path = os.path.join(run_dir, "final_model")
        model.save(final_zip_path)
        print(f"\nFinal model saved → {final_zip_path}.zip")
        vec_env.close()
        aim_run.close()

    # ------------------------------------------------------------------ #
    # ONNX export                                                          #
    # ------------------------------------------------------------------ #
    # Export to ONNX for future client-side inference.
    # The obs is a flat float32 vector; masking is applied externally.
    try:
        dummy_obs = torch.zeros(1, 2 + 13 * MAX_MEMBERS, dtype=torch.float32)
        onnx_path = os.path.join(run_dir, "final_model.onnx")
        torch.onnx.export(
            model.policy,
            dummy_obs,
            onnx_path,
            input_names=["obs"],
            output_names=["action_logits", "value"],
            opset_version=17,
        )
        print(f"ONNX model saved → {onnx_path}")
    except Exception as e:
        print(f"[onnx] export failed (non-fatal): {e}")

    # ------------------------------------------------------------------ #
    # Post-training eval summary                                           #
    # ------------------------------------------------------------------ #
    print("\nRunning post-training evaluation...")
    # Generate a fresh set of panels distinct from the training/eval sets.
    summary_generator = config.make_panel_generator(seed=999_999)
    summary_panels = [summary_generator() for _ in range(20)]
    eval_summary = _compute_eval_summary(model, config, summary_panels)
    print(
        f"  policy={eval_summary['mean_policy_reward']:.1f}  "
        f"nearest={eval_summary['mean_nearest_reward']:.1f}  "
        f"wins={eval_summary['win_rate']}/{eval_summary['n_panels']}  "
        f"improvement={eval_summary['mean_improvement_pct']:+.1f}%"
    )

    # ------------------------------------------------------------------ #
    # run_metadata.json                                                    #
    # ------------------------------------------------------------------ #
    checkpoint_entries = _collect_checkpoint_entries(ckpt_dir)
    # Append the final model as the last checkpoint entry.
    checkpoint_entries.append({
        "name": "final_model",
        "timestep": config.total_timesteps,
    })

    metadata = {
        "run_name":    run_name,
        "created_at":  created_at,
        "config":      config.as_dict(),
        "obs_dim":     2 + 13 * MAX_MEMBERS,
        "max_members": MAX_MEMBERS,
        "checkpoints": checkpoint_entries,
        "artifacts": {
            "final_model_zip":  "final_model.zip",
            "final_model_onnx": "final_model.onnx",
        },
        "eval_summary": eval_summary,
    }

    metadata_path = os.path.join(run_dir, "run_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Metadata written → {metadata_path}")


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
