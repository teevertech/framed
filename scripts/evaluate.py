"""Evaluate trained models across many panels and panel types.

Produces a summary table and optional per-panel GIFs showing how a
trained policy compares to greedy baselines on panels it has never seen.

Usage
-----
Evaluate the portfolio_k4.0 model on 20 random panels (same topology)::

    python scripts/evaluate.py \
        --model checkpoints/portfolio_k4.0/final_model.zip

Cross-topology evaluation (window + door + different widths)::

    python scripts/evaluate.py \
        --model checkpoints/portfolio_k4.0/final_model.zip \
        --cross-topology

Save GIFs of the best and worst episodes::

    python scripts/evaluate.py \
        --model checkpoints/portfolio_k4.0/final_model.zip \
        --save-gifs eval_gifs/

Full options::

    python scripts/evaluate.py --help
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from framed.baselines import greedy_cost_aware_action, greedy_nearest_action, run_episode
from framed.config import TrainConfig
from framed.env import PanelEnv
from framed.panel import generate_random_panel
from framed.units import feet, inches


# ------------------------------------------------------------------ #
# Panel generation                                                     #
# ------------------------------------------------------------------ #

def _make_same_topology_panels(
    config: TrainConfig, n: int, seed_offset: int = 50_000,
) -> list[tuple[str, PanelEnv]]:
    """Generate panels with the same topology as training."""
    gen = config.make_panel_generator(seed=seed_offset)
    panels = []
    for i in range(n):
        panel = gen()
        env = PanelEnv(
            panel,
            robot_speed=config.robot_speed,
            collision_penalty_multiplier=config.collision_penalty_multiplier,
        )
        panels.append((f"same_topo_{i:03d}", env))
    return panels


def _make_cross_topology_panels(
    config: TrainConfig, target_n: int, seed_base: int = 70_000,
) -> list[tuple[str, PanelEnv]]:
    """Generate panels with different topologies to test generalisation.

    Only includes variants whose member count matches *target_n* (derived
    from the model's observation space).  Variants that don't match are
    skipped with a note — this is a fundamental limitation of fixed-size
    observation spaces, not a bug.
    """
    panels: list[tuple[str, PanelEnv]] = []
    skipped: list[str] = []

    variants: list[tuple[str, dict]] = [
        # Different opening widths
        ("window_24in",  dict(opening_type="window", opening_width=inches(24),  wall_length=config.wall_length)),
        ("window_30in",  dict(opening_type="window", opening_width=inches(30),  wall_length=config.wall_length)),
        ("window_42in",  dict(opening_type="window", opening_width=inches(42),  wall_length=config.wall_length)),
        ("window_48in",  dict(opening_type="window", opening_width=inches(48),  wall_length=config.wall_length)),
        # Different wall lengths
        ("wall_10ft",    dict(opening_type="window", opening_width=config.opening_width, wall_length=feet(10))),
        ("wall_14ft",    dict(opening_type="window", opening_width=config.opening_width, wall_length=feet(14))),
        ("wall_16ft",    dict(opening_type="window", opening_width=config.opening_width, wall_length=feet(16))),
        ("wall_20ft",    dict(opening_type="window", opening_width=config.opening_width, wall_length=feet(20))),
        # Door instead of window
        ("door_32in",    dict(opening_type="door",   opening_width=inches(32),  wall_length=config.wall_length)),
        ("door_36in",    dict(opening_type="door",   opening_width=inches(36),  wall_length=config.wall_length)),
    ]

    for name, kwargs in variants:
        found = False
        best_n = None
        for seed in range(seed_base, seed_base + 50):
            panel = generate_random_panel(seed=seed, **kwargs)
            best_n = len(panel.members)
            if best_n == target_n:
                env = PanelEnv(
                    panel,
                    robot_speed=config.robot_speed,
                    collision_penalty_multiplier=config.collision_penalty_multiplier,
                )
                panels.append((name, env))
                found = True
                break
        if not found:
            skipped.append(f"{name} ({best_n} members, model expects {target_n})")

    if skipped:
        print(f"\nSkipped {len(skipped)} variant(s) (incompatible member count):")
        for s in skipped:
            print(f"  - {s}")
        print("  (This is expected — the model's observation space is fixed-size.)")

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
    os.makedirs(gif_dir, exist_ok=True)

    def _model_policy(env: PanelEnv) -> int:
        action, _ = model.predict(
            env.obs, action_masks=env.action_masks(), deterministic=True
        )
        return int(action)

    # Sort by improvement — save best 3 and worst 3
    ranked = sorted(
        zip(results, panels), key=lambda x: x[0]["improvement_vs_nearest"]
    )

    to_save = []
    for r, (name, env) in ranked[:3]:
        to_save.append((f"worst_{name}", env))
    for r, (name, env) in ranked[-3:]:
        to_save.append((f"best_{name}", env))

    for label, env in to_save:
        path = os.path.join(gif_dir, f"{label}.gif")
        save_episode_gif(env, _model_policy, path,
                         policy_name="Trained Policy", fps=20)


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
                   help="Number of same-topology panels to test (default: 20)")
    p.add_argument("--cross-topology", action="store_true",
                   help="Also test on different wall lengths, doors, etc.")
    p.add_argument("--collision-penalty", type=float, default=None,
                   help="Override collision penalty (default: use training default)")
    p.add_argument("--save-gifs", default=None,
                   help="Directory to save best/worst episode GIFs")
    return p.parse_args()


def main() -> None:
    args = _parse()
    config = TrainConfig()
    if args.collision_penalty is not None:
        import dataclasses
        config = dataclasses.replace(
            config, collision_penalty_multiplier=args.collision_penalty
        )

    # Derive target member count from the model's observation space.
    # obs shape = 2 + 3*n_members → n_members = (obs_dim - 2) / 3
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(args.model)
    obs_dim = model.observation_space.shape[0]
    target_n = (obs_dim - 2) // 3
    print(f"Model expects {target_n}-member panels (obs dim = {obs_dim})")
    del model  # free memory; evaluate_model reloads it

    # Same-topology evaluation
    same_panels = _make_same_topology_panels(config, n=args.n_panels)
    same_results = evaluate_model(args.model, same_panels, config)
    print_results(same_results, "Same Topology (training distribution)")

    # Cross-topology evaluation
    cross_panels = []
    cross_results = []
    if args.cross_topology:
        cross_panels = _make_cross_topology_panels(config, target_n=target_n)
        if cross_panels:
            cross_results = evaluate_model(args.model, cross_panels, config)
            print_results(cross_results, "Cross Topology (unseen panel types)")
        else:
            print("\nNo cross-topology panels could be generated with matching member count.")

    # GIFs
    if args.save_gifs:
        all_panels  = same_panels  + cross_panels
        all_results = same_results + cross_results
        save_eval_gifs(args.model, all_results, all_panels, args.save_gifs)


if __name__ == "__main__":
    main()
