"""Render and display (or save) a panel-assembly episode animation.

Usage
-----
Show an interactive animation (greedy nearest, random panel)::

    python scripts/visualize_episode.py

Save a GIF comparing both greedy baselines side-by-side::

    python scripts/visualize_episode.py --save comparison.gif

Use a trained model::

    python scripts/visualize_episode.py --model checkpoints/smoke_k1.0/final_model

Pick a specific panel seed::

    python scripts/visualize_episode.py --seed 7

All arguments
-------------
--policy    greedy_nearest | greedy_cost_aware | model  (default: greedy_nearest)
--model     Path to a saved MaskablePPO model (.zip).  Required when --policy model.
--seed      Panel seed for reproducibility.             (default: 0)
--save      Output path for GIF.  If omitted, displays interactively.
--fps       Animation frame rate.                       (default: 20)
--compare   Render nearest vs cost_aware side-by-side.  Ignores --policy.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from framed.baselines import greedy_cost_aware_action, greedy_nearest_action
from framed.config import TrainConfig
from framed.env import PanelEnv
from framed.panel import generate_random_panel
from framed.visualize import animate_episode, save_episode_gif


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _make_env(config: TrainConfig, seed: int) -> PanelEnv:
    """Create a PanelEnv from a fixed panel at the given seed."""
    panel = generate_random_panel(
        wall_length=config.wall_length,
        opening_type=config.opening_type,     # type: ignore[arg-type]
        opening_width=config.opening_width,
        seed=seed,
    )
    return PanelEnv(
        panel,
        robot_speed=config.robot_speed,
        collision_penalty_multiplier=config.collision_penalty_multiplier,
    )


def _load_model_policy(model_path: str):
    """Load a MaskablePPO model and return a policy callable."""
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(model_path)

    def _policy(env: PanelEnv) -> int:
        action, _ = model.predict(
            env.obs, action_masks=env.action_masks(), deterministic=True
        )
        return int(action)

    return _policy


# ------------------------------------------------------------------ #
# Single animation                                                     #
# ------------------------------------------------------------------ #

def run_single(args: argparse.Namespace) -> None:
    config = TrainConfig()
    env = _make_env(config, seed=args.seed)

    if args.policy == "greedy_nearest":
        policy = greedy_nearest_action
        name   = "Greedy Nearest"
    elif args.policy == "greedy_cost_aware":
        policy = greedy_cost_aware_action
        name   = "Greedy Cost-Aware"
    else:
        if not args.model:
            print("Error: --model PATH required when --policy model", file=sys.stderr)
            sys.exit(1)
        policy = _load_model_policy(args.model)
        name   = f"MaskablePPO ({os.path.basename(args.model)})"

    if args.save:
        save_episode_gif(env, policy, args.save,
                         policy_name=name, fps=args.fps)
    else:
        anim = animate_episode(env, policy, policy_name=name, fps=args.fps)
        plt.show()


# ------------------------------------------------------------------ #
# Side-by-side comparison                                             #
# ------------------------------------------------------------------ #

def run_comparison(args: argparse.Namespace) -> None:
    """Render nearest vs cost-aware on the same panel, saving a two-panel GIF."""
    from framed.visualize import _collect_frames, _draw_legend, _figure_size, _robot_size, _setup_axes, _draw_member, _robot_triangle, _PATH_CLEAR, _PATH_COLLIDE, _ROBOT_COLOR
    import matplotlib.patches as mpatches
    from matplotlib.patches import Polygon
    import numpy as np

    config = TrainConfig()

    env_n = _make_env(config, seed=args.seed)
    env_c = _make_env(config, seed=args.seed)
    panel = env_n.panel

    fps = args.fps
    fpm = 12
    pf  = 4

    frames_n = _collect_frames(env_n, greedy_nearest_action,    fpm, pf)
    frames_c = _collect_frames(env_c, greedy_cost_aware_action, fpm, pf)

    # Pad the shorter sequence.
    n = max(len(frames_n), len(frames_c))
    while len(frames_n) < n: frames_n.append(frames_n[-1])
    while len(frames_c) < n: frames_c.append(frames_c[-1])

    fw, fh = _figure_size(panel)
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(fw * 2 + 0.5, fh),
        gridspec_kw={"wspace": 0.35},
    )
    fig.patch.set_facecolor("#FAFAFA")
    rsize = _robot_size(panel)

    def _render_side(ax, frame, title_prefix):
        ax.cla()
        _setup_axes(ax, panel)
        ax.set_title(
            f"{title_prefix}  |  step {frame.step}/{env_n.n_members}"
            f"  |  reward {frame.total_reward:.1f}",
            fontsize=8, pad=4, color="#333333",
        )
        for member in panel.members:
            _draw_member(ax, member,
                         placed=member.id in frame.placed_ids,
                         is_target=member.id == frame.target_id)
        for from_pos, to_pos, collided in frame.paths:
            color = _PATH_COLLIDE if collided else _PATH_CLEAR
            ax.annotate("", xy=to_pos, xytext=from_pos,
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=1.1, mutation_scale=7), zorder=3)
        if frame.partial_path is not None:
            fp, tp = frame.partial_path
            color = _PATH_COLLIDE if frame.collided_this else _PATH_CLEAR
            ax.plot([fp[0], tp[0]], [fp[1], tp[1]],
                    color=color, linewidth=1.1, alpha=0.7, zorder=3)
        tri = _robot_triangle(frame.robot_xy, frame.direction, size=rsize)
        ax.add_patch(Polygon(tri, closed=True, facecolor=_ROBOT_COLOR,
                             edgecolor="white", linewidth=0.8, zorder=5))

    def _draw(i):
        _render_side(ax_l, frames_n[i], "Greedy Nearest")
        _render_side(ax_r, frames_c[i], "Greedy Cost-Aware")
        _draw_legend(ax_r, panel)

    anim = FuncAnimation(fig, _draw, frames=n,
                         interval=1000 / fps, repeat=True, blit=False)

    if args.save:
        print(f"Saving comparison GIF → {args.save}  ({n} frames @ {fps} fps) ...")
        anim.save(args.save, writer="pillow", fps=fps, dpi=110)
        print("Done.")
        plt.close("all")
    else:
        plt.show()


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize a panel-assembly episode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--policy",  default="greedy_nearest",
                   choices=["greedy_nearest", "greedy_cost_aware", "model"])
    p.add_argument("--model",   default=None,
                   help="Path to MaskablePPO .zip (required for --policy model)")
    p.add_argument("--seed",    type=int, default=0,
                   help="Panel seed (default: 0)")
    p.add_argument("--save",    default=None,
                   help="Output GIF path.  Omit to display interactively.")
    p.add_argument("--fps",     type=int, default=20)
    p.add_argument("--compare", action="store_true",
                   help="Side-by-side nearest vs cost-aware comparison")
    return p.parse_args()


def main() -> None:
    args = _parse()
    if args.compare:
        run_comparison(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
