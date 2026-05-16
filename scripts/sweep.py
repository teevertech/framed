"""Hyperparameter sweep for the panel-sequencing RL agent.

Runs a grid of ``TrainConfig`` instances **sequentially** — each run uses
all available CPU cores via ``SubprocVecEnv``, so parallel runs would
over-subscribe the machine.  All runs write to the same aim repo; use the
aim UI to compare them::

    aim up               # starts the aim server
    # or
    aim ui               # older versions

Usage
-----
Run the default sweep (learning rate × collision penalty)::

    python scripts/sweep.py

Run a quick smoke sweep (reduced timesteps) to verify the pipeline::

    python scripts/sweep.py smoke

Add a new sweep by defining a function that returns
``list[TrainConfig]`` and adding it to ``SWEEPS``.

Design notes
------------
Each run is launched in-process (calling ``train(config)`` directly)
rather than via subprocess.  This is simpler and aim handles multi-run
aggregation natively.  Process isolation between runs is provided by
SB3's SubprocVecEnv (workers are separate processes that are torn down
after each run).
"""
from __future__ import annotations

import dataclasses
import os
import sys
from itertools import product

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from framed.config import TrainConfig
from train import train   # scripts/train.py is on sys.path when run from scripts/


# ------------------------------------------------------------------ #
# Sweep definitions                                                    #
# ------------------------------------------------------------------ #

def portfolio_sweep() -> list[TrainConfig]:
    """4 runs varying collision penalty — the primary portfolio showcase.

    Each run trains for 300k steps with eval + GIF every 25k steps,
    giving 12 GIF snapshots per run that show the policy evolving.

    k=0.0  baseline: no collision penalty, pure travel-time minimisation
    k=1.0  mild:     collisions cost 2× bare travel time
    k=2.0  default:  collisions cost 3× bare travel time
    k=4.0  strong:   collisions cost 5× bare travel time

    Expect ~15–20 min total on an M2 Pro with 8 workers.
    """
    base = TrainConfig(
        experiment_name="portfolio",
        total_timesteps=300_000,
        n_envs=8,
        eval_freq=25_000,
        checkpoint_freq=75_000,
        gif_dir="gifs",
        gif_fps=20,
    )
    return [
        dataclasses.replace(
            base,
            collision_penalty_multiplier=k,
            run_name=f"portfolio_k{k}",
        )
        for k in [0.0, 1.0, 2.0, 4.0]
    ]


def lr_vs_penalty_sweep() -> list[TrainConfig]:
    """6-run grid: learning rate (3) × collision penalty multiplier (2)."""
    base = TrainConfig(experiment_name="lr_vs_penalty")
    configs = []
    for lr, k in product([1e-4, 3e-4, 1e-3], [2.0, 4.0]):
        configs.append(dataclasses.replace(
            base,
            learning_rate=lr,
            collision_penalty_multiplier=k,
            run_name=f"lr{lr:.0e}_k{k}",
        ))
    return configs


def architecture_sweep() -> list[TrainConfig]:
    """4-run grid over MLP hidden-layer depth and width."""
    base = TrainConfig(experiment_name="architecture")
    archs = [
        (128, 128),
        (256, 256),
        (256, 256, 256),
        (512, 512),
    ]
    return [
        dataclasses.replace(
            base,
            net_arch=arch,
            run_name="x".join(str(n) for n in arch),
        )
        for arch in archs
    ]


def smoke_sweep() -> list[TrainConfig]:
    """2-run sanity check — short runs to verify the pipeline end-to-end."""
    base = TrainConfig(
        experiment_name="smoke",
        total_timesteps=20_000,
        eval_freq=5_000,
        checkpoint_freq=10_000,
        n_envs=2,
    )
    return [
        dataclasses.replace(base, collision_penalty_multiplier=k, run_name=f"smoke_k{k}")
        for k in [1.0, 2.0]
    ]


# Registry: name → sweep factory
SWEEPS: dict[str, callable] = {
    "portfolio":     portfolio_sweep,
    "lr_vs_penalty": lr_vs_penalty_sweep,
    "architecture":  architecture_sweep,
    "smoke":         smoke_sweep,
}

DEFAULT_SWEEP = "portfolio"


# ------------------------------------------------------------------ #
# Runner                                                               #
# ------------------------------------------------------------------ #

def run_sweep(sweep_name: str) -> None:
    if sweep_name not in SWEEPS:
        print(f"Unknown sweep {sweep_name!r}.  Available: {list(SWEEPS)}")
        sys.exit(1)

    configs = SWEEPS[sweep_name]()
    n = len(configs)
    print(f"\nSweep '{sweep_name}': {n} run(s)\n")
    for i, cfg in enumerate(configs, 1):
        print(f"─── Run {i}/{n}: {cfg.effective_run_name()} ───")
        _print_diff(cfg)
        train(cfg)
        print()

    print(f"Sweep complete.  View results:\n  aim ui --repo {configs[0].aim_repo}")


def _print_diff(config: TrainConfig) -> None:
    """Print only the fields that differ from the default config."""
    default = TrainConfig()
    diffs = {
        k: v for k, v in dataclasses.asdict(config).items()
        if v != dataclasses.asdict(default).get(k)
    }
    for k, v in diffs.items():
        print(f"  {k} = {v}")


def main() -> None:
    sweep_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SWEEP
    run_sweep(sweep_name)


if __name__ == "__main__":
    main()
