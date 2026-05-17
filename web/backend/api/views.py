"""API views for the framed web UI.

POST /api/panels/generate  — generate a random panel from parameters
POST /api/sequence/run     — run all three policies on a panel
GET  /api/models           — list available trained model checkpoints
"""
from __future__ import annotations

import glob
import os
from typing import Any, Callable

from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.request import Request
from rest_framework.response import Response

from framed.baselines import greedy_cost_aware_action, greedy_nearest_action
from framed.env import PanelEnv
from framed.panel import Member, MemberKind, Panel, generate_random_panel
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
    """Serialize a Panel to a JSON-friendly dict."""
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
    }


def _panel_from_dict(data: dict) -> Panel:
    """Reconstruct a Panel from the JSON dict."""
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
    return Panel(
        wall_length=data["wall_length"],
        wall_height=data["wall_height"],
        members=members,
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
def generate_panel(request: Request) -> Response:
    """Generate a random panel from parameters.

    Body: { wall_length_ft, opening_type, opening_width_in,
            opening_center_x_in?, seed? }
    """
    data = request.data
    try:
        wall_length = feet(float(data.get("wall_length_ft", 12)))
        opening_type = data.get("opening_type", "window")
        opening_width = inches(float(data.get("opening_width_in", 36)))
        seed = int(data.get("seed", 0))

        kwargs: dict[str, Any] = {
            "wall_length": wall_length,
            "opening_type": opening_type,
            "opening_width": opening_width,
            "seed": seed,
        }
        if "opening_center_x_in" in data:
            kwargs["opening_center_x"] = inches(float(data["opening_center_x_in"]))

        panel = generate_random_panel(**kwargs)
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
    """List available trained model checkpoints."""
    ckpt_dir = settings.CHECKPOINT_DIR
    models = []

    if os.path.isdir(ckpt_dir):
        for dirpath, dirnames, filenames in os.walk(ckpt_dir):
            for f in filenames:
                if f.endswith(".zip"):
                    rel = os.path.relpath(
                        os.path.join(dirpath, f), ckpt_dir
                    )
                    name = rel.replace(".zip", "")
                    models.append({
                        "name": name,
                        "path": rel,
                    })

    models.sort(key=lambda m: m["name"])
    return Response({"models": models})
