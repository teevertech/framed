"""Summarise and rank sweep results.

Reads ``run_metadata.json`` from each run directory and (optionally) the
aim repo to produce a ranked table of runs by policy performance.

Usage
-----
Run from the project root after a sweep completes::

    python scripts/summarize.py

Point at a specific models directory or aim repo::

    python scripts/summarize.py --models models --aim-repo .aim

Show only a specific experiment::

    python scripts/summarize.py --experiment overnight
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from framed.config import MODELS_DIR, logger


# ------------------------------------------------------------------ #
# Collect results from run_metadata.json files                         #
# ------------------------------------------------------------------ #

def _collect_metadata(models_dir: str) -> list[dict]:
    """Walk *models_dir* for run_metadata.json files and parse them."""
    results = []
    if not os.path.isdir(models_dir):
        return results

    for run_name in sorted(os.listdir(models_dir)):
        meta_path = os.path.join(models_dir, run_name, "run_metadata.json")
        if not os.path.isfile(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        results.append(meta)
    return results


# ------------------------------------------------------------------ #
# Collect last-logged metrics from aim                                 #
# ------------------------------------------------------------------ #

def _collect_aim_metrics(aim_repo: str, experiment: str | None) -> dict[str, dict]:
    """Query aim for the last value of each metric per run.

    Returns ``{run_name: {metric_name: last_value}}``.
    Silently returns an empty dict if aim is unavailable.
    """
    try:
        from aim import Repo
        from aim.sdk.types import QueryReportMode
    except ImportError:
        return {}

    try:
        repo = Repo(aim_repo, read_only=True)
    except Exception:
        return {}

    query = f'run.experiment == "{experiment}"' if experiment else ""
    aim_data: dict[str, dict] = {}

    try:
        for metric_seq in repo.query_metrics(query, report_mode=QueryReportMode.DISABLED):
            run_name = metric_seq.run.name or metric_seq.run.hash
            name = metric_seq.name
            values = metric_seq.values
            if not len(values):
                continue
            aim_data.setdefault(run_name, {})[name] = values[-1]
    except Exception as e:
        logger.warning(f"aim query failed: {e}")

    return aim_data


# ------------------------------------------------------------------ #
# Formatting                                                           #
# ------------------------------------------------------------------ #

def _fmt(val, width=8):
    """Format a numeric value for table display."""
    if val is None:
        return " " * width
    if isinstance(val, float):
        return f"{val:>{width}.1f}"
    return f"{val:>{width}}"


def _print_table(rows: list[dict], aim_data: dict[str, dict]) -> None:
    """Print a ranked summary table to stdout."""

    # Header
    print()
    print(f"{'#':>3}  {'run_name':<30}  {'policy':>8}  {'nearest':>8}  "
          f"{'improv%':>8}  {'wins':>5}  {'expl_var':>8}  {'entropy':>8}  "
          f"{'fps':>6}  {'steps':>10}")
    print("─" * 130)

    for i, row in enumerate(rows, 1):
        name = row["run_name"]
        ev = row.get("eval_summary", {})
        cfg = row.get("config", {})

        # Pull aim metrics if available
        aim = aim_data.get(name, {})
        # Try common metric name patterns
        expl_var = (aim.get("train/explained_variance")
                    or aim.get("explained_variance"))
        entropy = (aim.get("train/entropy_loss")
                   or aim.get("entropy_loss"))
        fps = aim.get("time/fps") or aim.get("fps")

        print(
            f"{i:>3}  {name:<30}  "
            f"{_fmt(ev.get('mean_policy_reward'))}  "
            f"{_fmt(ev.get('mean_nearest_reward'))}  "
            f"{_fmt(ev.get('mean_improvement_pct'))}  "
            f"{str(ev.get('win_rate', '')):>5}  "
            f"{_fmt(expl_var)}  "
            f"{_fmt(entropy)}  "
            f"{_fmt(fps, 6)}  "
            f"{cfg.get('total_timesteps', ''):>10}"
        )

    print()


def _print_config_diff(rows: list[dict]) -> None:
    """Print a compact table of hyperparameters that vary across runs."""
    if len(rows) < 2:
        return

    configs = [r.get("config", {}) for r in rows]
    all_keys = set()
    for c in configs:
        all_keys.update(c.keys())

    # Find keys where values differ across runs
    varying = {}
    for key in sorted(all_keys):
        vals = [c.get(key) for c in configs]
        if len(set(str(v) for v in vals)) > 1:
            varying[key] = vals

    # Skip bookkeeping fields
    skip = {"run_name", "experiment_name", "aim_repo", "checkpoint_dir",
            "gif_dir", "seed"}
    varying = {k: v for k, v in varying.items() if k not in skip}

    if not varying:
        print("  All runs share identical hyperparameters.\n")
        return

    print("Varying hyperparameters:")
    names = [r["run_name"] for r in rows]
    name_width = max(len(n) for n in names)

    # Print as: run_name | param1 | param2 | ...
    keys = list(varying.keys())
    col_widths = [max(len(k), max(len(str(v)) for v in varying[k])) for k in keys]

    header = f"  {'run':<{name_width}}  " + "  ".join(
        f"{k:>{w}}" for k, w in zip(keys, col_widths)
    )
    print(header)
    print("  " + "─" * (len(header) - 2))

    for i, name in enumerate(names):
        vals = "  ".join(
            f"{str(varying[k][i]):>{w}}" for k, w in zip(keys, col_widths)
        )
        print(f"  {name:<{name_width}}  {vals}")
    print()


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise sweep results.")
    parser.add_argument("--models", default=str(MODELS_DIR),
                        help=f"Root models directory (default: {MODELS_DIR})")
    parser.add_argument("--aim-repo", default=".aim",
                        help="Path to aim repo (default: .aim)")
    parser.add_argument("--experiment", default=None,
                        help="Filter to a specific experiment name")
    parser.add_argument("--sort", default="mean_improvement_pct",
                        choices=["mean_improvement_pct", "mean_policy_reward",
                                 "win_rate"],
                        help="Sort metric (default: mean_improvement_pct)")
    args = parser.parse_args()

    # Collect data
    all_meta = _collect_metadata(args.models)
    if not all_meta:
        print(f"No run_metadata.json files found under {args.models}/")
        sys.exit(1)

    # Filter by experiment if requested
    if args.experiment:
        all_meta = [m for m in all_meta
                    if m.get("config", {}).get("experiment_name") == args.experiment]
        if not all_meta:
            print(f"No runs found for experiment '{args.experiment}'")
            sys.exit(1)

    # Sort by chosen metric (descending = best first)
    all_meta.sort(
        key=lambda m: m.get("eval_summary", {}).get(args.sort, float("-inf")),
        reverse=True,
    )

    # Aim data (optional enrichment)
    aim_data = _collect_aim_metrics(args.aim_repo, args.experiment)

    # Display
    exp_label = args.experiment or "all experiments"
    print(f"\n{'='*60}")
    print(f"  Sweep summary — {exp_label}")
    print(f"  {len(all_meta)} run(s), ranked by {args.sort}")
    print(f"{'='*60}")

    _print_table(all_meta, aim_data)
    _print_config_diff(all_meta)

    # Highlight the winner
    best = all_meta[0]
    ev = best.get("eval_summary", {})
    print(f"  ★ Best: {best['run_name']}")
    print(f"    policy={ev.get('mean_policy_reward')}  "
          f"nearest={ev.get('mean_nearest_reward')}  "
          f"improvement={ev.get('mean_improvement_pct')}%  "
          f"wins={ev.get('win_rate')}/{ev.get('n_panels')}")
    print()


if __name__ == "__main__":
    main()
