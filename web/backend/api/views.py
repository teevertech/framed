"""API views for the framed web UI.

POST /api/panels/generate  — generate a panel from parameters (multi-opening)
POST /api/panels/random    — generate a fully random panel for generalization demos
POST /api/sequence/run     — run all three policies on a panel
GET  /api/models           — list available trained model checkpoints with metadata
"""
from __future__ import annotations

import json
import os
import random
from typing import Any, Callable

from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.request import Request
from rest_framework.response import Response

from framed.baselines import greedy_cost_aware_action, greedy_nearest_action
from framed.env import PanelEnv
from framed.panel import (
    Member,
    MemberKind,
    OpeningSpec,
    Panel,
    generate_panel,
    generate_random_panel,
)
from framed.units import feet, inches

# ------------------------------------------------------------------ #
# Model cache (module-level, survives across requests)                 #
# ------------------------------------------------------------------ #

_model_cache: dict[str, Any] = {}


def _load_model(model_name: str):
    """Load a MaskablePPO model, caching for reuse."""
    if model_name not in _model_cache:
        from sb3_contrib import MaskablePPO

        model_path = os.path.join(settings.CHECKPOINT_DIR, model_name)
        if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
            return None
        _model_cache[model_name] = MaskablePPO.load(model_path)
    return _model_cache[model_name]


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _panel_to_dict(panel: Panel) -> dict:
    """Serialize a Panel to a JSON-friendly dict, including openings."""
    return {
        "wall_length": panel.wall_length,
        "wall_height": panel.wall_height,
        "members": [
            {
                "id": m.id,
                "kind": m.kind.value,
                "position": list(m.position),
                "size": list(m.size),
                "prerequisites": m.prerequisites,
                "bounds": list(m.bounds),
                "center": list(m.center),
            }
            for m in panel.members
        ],
        "openings": [
            {
                "kind": o.kind,
                "center_x": o.center_x,
                "width": o.width,
                "member_ids": o.member_ids,
            }
            for o in panel.openings
        ],
    }


def _panel_from_dict(data: dict) -> Panel:
    """Reconstruct a Panel from the JSON dict, including openings."""
    members = [
        Member(
            id=m["id"],
            kind=MemberKind(m["kind"]),
            position=tuple(m["position"]),
            size=tuple(m["size"]),
            prerequisites=m.get("prerequisites", []),
        )
        for m in data["members"]
    ]
    openings = [
        OpeningSpec(
            kind=o["kind"],
            center_x=o["center_x"],
            width=o["width"],
            member_ids=o["member_ids"],
        )
        for o in data.get("openings", [])
    ]
    return Panel(
        wall_length=data["wall_length"],
        wall_height=data["wall_height"],
        members=members,
        openings=openings,
    )


def _run_sequence(
    env: PanelEnv,
    policy: Callable[[PanelEnv], int],
) -> dict:
    """Run a full episode and return step-by-step data for the frontend."""
    env.reset()
    steps = []
    cumulative = 0.0
    for _ in range(env.n_members):
        from_xy = env.robot_pos
        action = policy(env)
        _, reward, terminated, _, info = env.step(action)
        cumulative += float(reward)
        steps.append({
            "member_id": info["member_id"],
            "member_index": info["member_index"],
            "from_xy": list(from_xy),
            "to_xy": list(info["robot_pos"]),
            "travel_time": round(info["travel_time"], 3),
            "collided": info["collided"],
            "reward": round(float(reward), 3),
            "cumulative_reward": round(cumulative, 3),
        })
        if terminated:
            break
    return {
        "total_reward": round(cumulative, 3),
        "collision_count": sum(1 for s in steps if s["collided"]),
        "steps": steps,
    }


# ------------------------------------------------------------------ #
# Views                                                                #
# ------------------------------------------------------------------ #

@api_view(["POST"])
def generate_panel_view(request: Request) -> Response:
    """Generate a panel from parameters (supports multiple openings).

    Body: {
        wall_length_ft: number,
        openings: [{ type: "window"|"door", width_in: number }, ...],
        seed: number
    }
    """
    data = request.data
    try:
        wall_length = feet(float(data.get("wall_length_ft", 12)))
        seed = int(data.get("seed", 0))

        raw_openings = data.get("openings", [{"type": "window", "width_in": 36}])
        openings = [
            {"type": o["type"], "width": inches(float(o["width_in"]))}
            for o in raw_openings
        ]

        panel = generate_panel(
            wall_length=wall_length,
            openings=openings,
            seed=seed,
        )
        return Response(_panel_to_dict(panel))

    except Exception as e:
        return Response(
            {"error": str(e)},
            status=status.HTTP_400_BAD_REQUEST,
        )


@api_view(["POST"])
def random_panel_view(request: Request) -> Response:
    """Generate a fully random panel for generalization demos.

    Body (all optional): { seed?: number, max_openings?: number }

    Randomizes wall length, number of openings (1..max_openings),
    opening types, and opening widths. Intended to produce panels
    outside the training distribution to demonstrate generalization.
    """
    data = request.data or {}
    try:
        seed = int(data.get("seed", random.randint(0, 2**31 - 1)))
        max_openings = int(data.get("max_openings", 3))

        rng = random.Random(seed)

        wall_length_ft = rng.choice([8, 10, 12, 14, 16])
        wall_length = feet(wall_length_ft)

        n_openings = rng.randint(1, max_openings)
        opening_types = ["window", "door"]
        window_widths = [24, 30, 32, 36, 42, 48]
        door_widths = [32, 36]

        openings = []
        for _ in range(n_openings):
            kind = rng.choice(opening_types)
            if kind == "window":
                width_in = rng.choice(window_widths)
            else:
                width_in = rng.choice(door_widths)
            openings.append({"type": kind, "width": inches(width_in)})

        ep_seed = rng.randint(0, 2**31 - 1)

        # Retry loop: randomly positioned openings may not fit.
        last_error = None
        for _ in range(50):
            try:
                panel = generate_panel(
                    wall_length=wall_length,
                    openings=openings,
                    seed=ep_seed,
                )
                return Response(_panel_to_dict(panel))
            except ValueError as e:
                last_error = e
                ep_seed = rng.randint(0, 2**31 - 1)

        # Fallback: single window on the chosen wall, guaranteed to succeed.
        panel = generate_panel(
            wall_length=wall_length,
            openings=[{"type": "window", "width": inches(36)}],
            seed=ep_seed,
        )
        return Response(_panel_to_dict(panel))

    except Exception as e:
        return Response(
            {"error": str(e)},
            status=status.HTTP_400_BAD_REQUEST,
        )


@api_view(["POST"])
def run_sequence(request: Request) -> Response:
    """Run all three policies on a panel and return step-by-step results.

    Body: { panel: {...}, robot_speed?, collision_penalty_multiplier?,
            model_name? }
    """
    data = request.data
    try:
        panel = _panel_from_dict(data["panel"])
        robot_speed = float(data.get("robot_speed", 10.0))
        k = float(data.get("collision_penalty_multiplier", 2.0))
        model_name = data.get("model_name", None)

        env = PanelEnv(
            panel,
            robot_speed=robot_speed,
            collision_penalty_multiplier=k,
        )

        result = {
            "greedy_nearest": _run_sequence(env, greedy_nearest_action),
            "greedy_cost_aware": _run_sequence(env, greedy_cost_aware_action),
        }

        if model_name:
            model = _load_model(model_name)
            if model is None:
                return Response(
                    {"error": f"Model not found: {model_name}"},
                    status=status.HTTP_404_NOT_FOUND,
                )

            def _model_policy(e: PanelEnv) -> int:
                action, _ = model.predict(
                    e.obs, action_masks=e.action_masks(), deterministic=True
                )
                return int(action)

            result["policy"] = _run_sequence(env, _model_policy)
        else:
            result["policy"] = None

        return Response(result)

    except KeyError as e:
        return Response(
            {"error": f"Missing field: {e}"},
            status=status.HTTP_400_BAD_REQUEST,
        )
    except Exception as e:
        return Response(
            {"error": str(e)},
            status=status.HTTP_400_BAD_REQUEST,
        )


@api_view(["GET"])
def list_models(request: Request) -> Response:
    """List available trained model checkpoints with rich metadata.

    Reads ``run_metadata.json`` sidecars from each run directory.

    Returns: { models: { run_name: RunMetadata } }

    RunMetadata shape:
        {
            run_name: str,
            created_at: str,
            config: { ... },
            obs_dim: int,
            max_members: int,
            checkpoints: [ { name: str, timestep: int }, ... ],
            artifacts: { final_model_zip: str, final_model_onnx: str },
            eval_summary: {
                n_panels: int,
                win_rate: float,
                mean_improvement_pct: float,
                min_improvement_pct: float,
                max_improvement_pct: float,
                mean_policy_reward: float,
                mean_nearest_reward: float,
            }
        }

    Falls back to a minimal entry (checkpoint list only) for runs
    without a metadata sidecar.
    """
    ckpt_dir = settings.CHECKPOINT_DIR
    grouped: dict[str, Any] = {}

    if os.path.isdir(ckpt_dir):
        for entry in sorted(os.listdir(ckpt_dir)):
            run_dir = os.path.join(ckpt_dir, entry)
            if not os.path.isdir(run_dir):
                continue

            meta_path = os.path.join(run_dir, "run_metadata.json")
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    # Ensure checkpoints list is populated even if the
                    # sidecar was written before all checkpoints existed.
                    # Scan for any .zip files not already in the list.
                    existing_names = {
                        c["name"] for c in meta.get("checkpoints", [])
                    }
                    for fname in sorted(os.listdir(run_dir)):
                        if fname.endswith(".zip"):
                            ckpt_name = fname.removesuffix(".zip")
                            if ckpt_name not in existing_names:
                                meta.setdefault("checkpoints", []).append(
                                    {"name": ckpt_name, "timestep": 0}
                                )
                    grouped[entry] = meta
                except (json.JSONDecodeError, OSError):
                    # Corrupted sidecar — fall through to filesystem scan.
                    grouped[entry] = _minimal_run_meta(entry, run_dir)
            else:
                grouped[entry] = _minimal_run_meta(entry, run_dir)

    return Response({"models": grouped})


def _minimal_run_meta(run_name: str, run_dir: str) -> dict:
    """Build a minimal RunMetadata dict from filesystem scan alone."""
    checkpoints = []
    for fname in sorted(os.listdir(run_dir)):
        if fname.endswith(".zip"):
            checkpoints.append({
                "name": fname.removesuffix(".zip"),
                "timestep": 0,
            })
    return {
        "run_name": run_name,
        "created_at": "",
        "config": {},
        "obs_dim": 0,
        "max_members": 0,
        "checkpoints": checkpoints,
        "artifacts": {"final_model_zip": "", "final_model_onnx": ""},
        "eval_summary": None,
    }
