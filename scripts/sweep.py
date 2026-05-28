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

from framed.config import AIM_REPO, GIFS_DIR, TrainConfig, logger
from train import train


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
        gif_dir=GIFS_DIR,
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


def overnight_sweep() -> list[TrainConfig]:
    """Time-budgeted sweep for ~8-10 hours on an M2 Pro (8 workers).

    Overhead is minimised for sweep runs:
      - eval_freq=50_000  (not 10k — 10× fewer evals, still enough for aim curves)
      - gif_dir=GIFS_DIR     (kept — at 50k eval_freq the overhead is minimal)
      - checkpoint_freq=250_000 (only a couple checkpoints per run, plus final model)
      - n_eval_panels=3   (3 panels is enough to see trends; 5 is for final eval)

    Time budget at ~6 min per 100k steps (observed: 50 min / 500k):
      Phase 1 — exploration (most impactful):  6 × 500k = 3.0M steps  ~5.0 hr
      Phase 2 — rollout buffer:                3 × 500k = 1.5M steps  ~2.5 hr
      Phase 3 — extended training:             1 × 1.0M = 1.0M steps  ~1.7 hr
                                               ─────────────────────  ───────
                                               10 runs,  5.5M steps   ~9.2 hr

    Runs are ordered so the most informative results land first — if you
    wake up and it's still on Phase 3, you already have the exploration
    and rollout results in aim.
    """
    # Shared settings: lean eval, no GIFs, minimal checkpoints.
    lean = dict(
        eval_freq=50_000,
        checkpoint_freq=250_000,
        gif_dir=GIFS_DIR,
        n_eval_panels=3,
    )

    configs: list[TrainConfig] = []

    # ── Phase 1: ent_coef × gamma (6 runs, ~5 hr) ────────────────────
    # The single most impactful pair of knobs given current training stats
    # (entropy already low at 500k, discount horizon affects sequencing).
    explore_base = TrainConfig(
        experiment_name="overnight",
        total_timesteps=500_000,
        **lean,
    )
    for ent, g in product([0.001, 0.01, 0.05], [0.99, 0.999]):
        configs.append(dataclasses.replace(
            explore_base,
            ent_coef=ent,
            gamma=g,
            run_name=f"ent{ent}_g{g}",
        ))

    # ── Phase 2: n_steps / rollout buffer (3 runs, ~2.5 hr) ──────────
    # Tests whether the agent benefits from seeing more complete episodes
    # before each update.  batch_size scales proportionally.
    rollout_base = TrainConfig(
        experiment_name="overnight",
        total_timesteps=500_000,
        **lean,
    )
    for nsteps, bs in [(512, 128), (1024, 256), (2048, 512)]:
        configs.append(dataclasses.replace(
            rollout_base,
            n_steps=nsteps,
            batch_size=bs,
            run_name=f"nsteps_{nsteps}",
        ))

    # ── Phase 3: longer training (1 run, ~1.7 hr) ────────────────────
    # Does the reward curve still have headroom past 500k?
    # LR reduced to 1e-4 for the longer horizon.
    configs.append(TrainConfig(
        experiment_name="overnight",
        total_timesteps=1_000_000,
        learning_rate=1e-4,
        run_name="ext_1M",
        **lean,
    ))

    return configs


def benchmark_sweep() -> list[TrainConfig]:
    """3-run CPU/GPU benchmark — same hyperparameters, different hardware config.

    Run 1: baseline    — PyTorch defaults (all cores), CPU.
    Run 2: pinned      — torch_threads=1, CPU.  Eliminates thread contention
                         with SubprocVecEnv workers.  Often 10-30% faster for
                         small MLPs (256×256).
    Run 3: mps         — Apple Metal GPU + pinned threads.  Offloads inference
                         and backprop to the GPU, freeing CPU for env workers.

    Each run does 50k steps — enough to get a stable FPS reading (~5 min each).
    Compare the ``time/fps`` metric in aim or the terminal output.
    """
    base = TrainConfig(
        experiment_name="benchmark",
        total_timesteps=50_000,
        eval_freq=25_000,
        checkpoint_freq=50_000,
        n_eval_panels=2,
        gif_dir=None,
    )
    return [
        dataclasses.replace(
            base,
            run_name="bench_baseline",
        ),
        dataclasses.replace(
            base,
            torch_threads=1,
            run_name="bench_pinned",
        ),
        dataclasses.replace(
            base,
            torch_threads=1,
            device="mps",
            run_name="bench_mps",
        ),
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
    "overnight":     overnight_sweep,
    "benchmark":     benchmark_sweep,
    "portfolio":     portfolio_sweep,
    "lr_vs_penalty": lr_vs_penalty_sweep,
    "architecture":  architecture_sweep,
    "smoke":         smoke_sweep,
}

DEFAULT_SWEEP = "overnight"


# ------------------------------------------------------------------ #
# Runner                                                               #
# ------------------------------------------------------------------ #

def run_sweep(sweep_name: str) -> None:
    if sweep_name not in SWEEPS:
        logger.error(f"Unknown sweep {sweep_name!r}.  Available: {list(SWEEPS)}")
        sys.exit(1)

    configs = SWEEPS[sweep_name]()
    n = len(configs)
    total_steps = sum(c.total_timesteps for c in configs)
    # Rough estimate: ~6 min per 100k steps on M2 Pro w/ 8 workers.
    est_minutes = total_steps / 100_000 * 6
    est_hours = est_minutes / 60

    logger.info(f"\nSweep '{sweep_name}': {n} run(s)")
    logger.info(f"  total timesteps: {total_steps:,}")
    logger.info(f"  estimated time:  {est_hours:.1f} hr ({est_minutes:.0f} min)")
    logger.info("")

    for i, cfg in enumerate(configs, 1):
        logger.info(f"─── Run {i}/{n}: {cfg.effective_run_name()} ───")
        _print_diff(cfg)
        train(cfg)
        logger.info("")

    logger.info(f"Sweep complete.  View results:\n  aim ui --repo {AIM_REPO}")


def _print_diff(config: TrainConfig) -> None:
    """Print only the fields that differ from the default config."""
    default = TrainConfig()
    diffs = {
        k: v for k, v in dataclasses.asdict(config).items()
        if v != dataclasses.asdict(default).get(k)
    }
    for k, v in diffs.items():
        logger.info(f"  {k} = {v}")


def main() -> None:
    sweep_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SWEEP
    run_sweep(sweep_name)


if __name__ == "__main__":
    main()
