"""Evaluate trained models across many panels.

Produces a summary table and optional per-panel GIFs showing how a
trained policy compares to greedy baselines on panels it has never seen.

Usage
-----
Evaluate on 20 unseen panels (sampled from the training distribution)::

    python scripts/evaluate.py \
        --model checkpoints/portfolio_k4.0/final_model.zip

Save GIFs of the best and worst episodes::

    python scripts/evaluate.py \
        --model checkpoints/portfolio_k4.0/final_model.zip \
        --save-gifs eval_gifs/

Full options::

    python scripts/evaluate.py --help
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from loguru import logger

from framed.baselines import greedy_cost_aware_action, greedy_nearest_action, run_episode
from framed.config import TrainConfig, logger
from framed.env import PanelEnv


# ------------------------------------------------------------------ #
# Panel generation                                                     #
# ------------------------------------------------------------------ #

def _make_same_topology_panels(
    config: TrainConfig, n: int, seed_offset: int = 50_000,
) -> list[tuple[str, PanelEnv]]:
    """Generate *n* evaluation panels from the training distribution.

    Uses ``config.make_panel_generator`` with a seed offset well outside
    the training range so these panels are genuinely unseen.
    """
    gen = config.make_panel_generator(seed=seed_offset)
    panels = []
    for i in range(n):
        panel = gen()
        env = PanelEnv(
            panel,
            robot_speed=config.robot_speed,
            collision_penalty_multiplier=config.collision_penalty_multiplier,
        )
        panels.append((f"panel_{i:03d}", env))
    return panels


# ------------------------------------------------------------------ #
# Evaluation                                                           #
# ------------------------------------------------------------------ #

def evaluate_model(
    model_path: str,
    panels: list[tuple[str, PanelEnv]],
    config: TrainConfig,
) -> list[dict]:
    """Run the model + both baselines on each panel and return results."""
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(model_path)

    results = []
    for name, env in panels:
        # Greedy baselines
        r_nearest, info_nearest     = run_episode(env, greedy_nearest_action)
        r_cost_aware, info_costaware = run_episode(env, greedy_cost_aware_action)

        # Trained policy
        obs, _ = env.reset()
        r_policy = 0.0
        policy_collisions = 0
        for _ in range(env.n_members):
            masks = env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, reward, terminated, _, info = env.step(int(action))
            r_policy += float(reward)
            policy_collisions += int(info.get("collided", False))
            if terminated:
                break

        n_collisions_nearest = sum(
            1 for i in info_nearest if i.get("collided", False)
        )

        improvement = (
            ((r_policy - r_nearest) / abs(r_nearest) * 100)
            if abs(r_nearest) > 1e-6 else 0.0
        )

        results.append({
            "name":                name,
            "n_members":           env.n_members,
            "reward_nearest":      r_nearest,
            "reward_cost_aware":   r_cost_aware,
            "reward_policy":       r_policy,
            "improvement_vs_nearest": improvement,
            "collisions_nearest":  n_collisions_nearest,
            "collisions_policy":   policy_collisions,
        })

    return results


# ------------------------------------------------------------------ #
# Reporting                                                            #
# ------------------------------------------------------------------ #

def print_results(results: list[dict], title: str) -> None:
    if not results:
        print(f"\n{title}: no panels evaluated.\n")
        return

    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")

    # Header
    print(f"  {'Panel':<22} {'Nearest':>9} {'CostAw':>9} {'Policy':>9} {'Improv%':>8} {'Col(N)':>7} {'Col(P)':>7}")
    print(f"  {'-'*22} {'-'*9} {'-'*9} {'-'*9} {'-'*8} {'-'*7} {'-'*7}")

    for r in results:
        print(
            f"  {r['name']:<22} "
            f"{r['reward_nearest']:>9.1f} "
            f"{r['reward_cost_aware']:>9.1f} "
            f"{r['reward_policy']:>9.1f} "
            f"{r['improvement_vs_nearest']:>7.1f}% "
            f"{r['collisions_nearest']:>7d} "
            f"{r['collisions_policy']:>7d}"
        )

    # Summary stats
    improvements = [r["improvement_vs_nearest"] for r in results]
    policy_wins = sum(1 for r in results if r["reward_policy"] > r["reward_nearest"])
    print(f"\n  Summary ({len(results)} panels):")
    print(f"    Policy beats nearest:  {policy_wins}/{len(results)}")
    print(f"    Mean improvement:      {np.mean(improvements):+.1f}%")
    print(f"    Min improvement:       {np.min(improvements):+.1f}%")
    print(f"    Max improvement:       {np.max(improvements):+.1f}%")
    print()


# ------------------------------------------------------------------ #
# GIF saving                                                           #
# ------------------------------------------------------------------ #

def save_eval_gifs(
    model_path: str,
    results: list[dict],
    panels: list[tuple[str, PanelEnv]],
    gif_dir: str,
) -> None:
    """Save side-by-side GIFs for the best and worst panels."""
    from sb3_contrib import MaskablePPO
    from framed.visualize import save_episode_gif

    model = MaskablePPO.load(model_path)
    out = Path(gif_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _model_policy(env: PanelEnv) -> int:
        action, _ = model.predict(
            env.obs, action_masks=env.action_masks(), deterministic=True
        )
        return int(action)

    ranked = sorted(
        zip(results, panels), key=lambda x: x[0]["improvement_vs_nearest"]
    )

    to_save = []
    for r, (name, env) in ranked[:3]:
        to_save.append((f"worst_{name}", env))
    for r, (name, env) in ranked[-3:]:
        to_save.append((f"best_{name}", env))

    for label, env in to_save:
        path = out / f"{label}.gif"
        save_episode_gif(env, _model_policy, str(path),
                         policy_name="Trained Policy", fps=20)
        logger.info(f"saved {path}")


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained model across many panels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True,
                   help="Path to MaskablePPO .zip file")
    p.add_argument("--n-panels", type=int, default=20,
                   help="Number of panels to evaluate (default: 20)")
    p.add_argument("--collision-penalty", type=float, default=None,
                   help="Override collision penalty (default: use training default)")
    p.add_argument("--save-gifs", default=None,
                   help="Directory to save best/worst episode GIFs")
    return p.parse_args()


def main() -> None:
    args = _parse()
    config = TrainConfig()
    if args.collision_penalty is not None:
        config = dataclasses.replace(
            config, collision_penalty_multiplier=args.collision_penalty
        )

    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(args.model)
    obs_dim = model.observation_space.shape[0]
    logger.info(f"model loaded  (obs dim = {obs_dim})")
    del model

    panels = _make_same_topology_panels(config, n=args.n_panels)
    results = evaluate_model(args.model, panels, config)
    print_results(results, f"Evaluation — {args.n_panels} unseen panels")

    if args.save_gifs:
        save_eval_gifs(args.model, results, panels, args.save_gifs)


if __name__ == "__main__":
    main()
