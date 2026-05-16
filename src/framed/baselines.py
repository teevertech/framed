"""Greedy baseline policies and episode runner for ``PanelEnv``.

Two greedy strategies are provided:

``greedy_nearest_action``
    Pick the valid member whose center is closest (Euclidean) to the
    robot's current position.  Ignores the collision penalty entirely.
    This is the "dumb but fast" baseline — equivalent to a nearest-
    neighbour TSP heuristic applied to the panel graph.

``greedy_cost_aware_action``
    Pick the valid member that minimises the immediate one-step cost,
    including the collision penalty.  Uses the same cost function the
    RL agent optimises.  This is a stronger baseline because it accounts
    for obstruction, but it is still myopic (looks one step ahead only).

Both use lowest-index tiebreaking: when two candidates have equal cost
(or equal distance), the one with the lower index in ``panel.members``
wins.  This makes the baselines fully deterministic for a given panel
and starting position.

``run_episode`` is a convenience runner that executes any
``policy(env) → action`` function for a full episode and returns the
total reward plus per-step info dicts.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from framed.env import PanelEnv
from framed.geometry import path_collides, travel_time


def greedy_nearest_action(env: PanelEnv) -> int:
    """Return the valid action whose member center is closest to the robot.

    Ignores the collision penalty — distance is the only criterion.

    Raises
    ------
    ValueError
        If the env is in a terminal state (no valid actions).
    """
    mask = env.action_masks()
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        raise ValueError("No valid actions — episode is in terminal state")

    robot = np.array(env.robot_pos, dtype=np.float64)
    centers = np.array(
        [env.panel.members[int(i)].center for i in valid], dtype=np.float64
    )
    dists = np.linalg.norm(centers - robot, axis=1)
    # argmin returns the first occurrence on ties → lowest valid index wins.
    return int(valid[int(np.argmin(dists))])


def greedy_cost_aware_action(env: PanelEnv) -> int:
    """Return the valid action that minimises the immediate step cost.

    Cost is computed identically to ``PanelEnv.step()``::

        cost = travel_time * (1 + collision_penalty_multiplier * collided)

    The liftoff rule is replicated: the last-placed member is excluded
    from the obstacle set, just as the env does internally.

    Raises
    ------
    ValueError
        If the env is in a terminal state (no valid actions).
    """
    mask = env.action_masks()
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        raise ValueError("No valid actions — episode is in terminal state")

    robot_pos = env.robot_pos
    placed = env.placed_mask
    last_idx = env.last_placed_idx

    # Build the obstacle list once (same logic as PanelEnv.step).
    obstacles = [
        m
        for i, (m, p) in enumerate(zip(env.panel.members, placed))
        if p and i != last_idx
    ]

    best_cost = float("inf")
    best_action = -1
    for idx in valid:
        idx = int(idx)
        member = env.panel.members[idx]
        target = member.center
        t = travel_time(robot_pos, target, speed=env.robot_speed)
        collided = path_collides(robot_pos, target, obstacles)
        cost = t * (1.0 + env.collision_penalty_multiplier * float(collided))
        if cost < best_cost:
            best_cost = cost
            best_action = idx

    return best_action


# ------------------------------------------------------------------ #
# Episode runner                                                       #
# ------------------------------------------------------------------ #

def run_episode(
    env: PanelEnv,
    policy: Callable[[PanelEnv], int],
) -> tuple[float, list[dict[str, Any]]]:
    """Execute *policy* for a full episode and return ``(total_reward, step_infos)``.

    ``policy`` is any callable that accepts a ``PanelEnv`` and returns
    an action index.  The env is ``reset()`` at the start.

    Returns
    -------
    total_reward:
        Sum of per-step rewards (will be ≤ 0).
    step_infos:
        The ``info`` dict from each ``step()`` call, in order.
    """
    env.reset()
    total_reward = 0.0
    step_infos: list[dict[str, Any]] = []
    for _ in range(env.n_members):
        action = policy(env)
        _, reward, terminated, _, info = env.step(action)
        total_reward += reward
        step_infos.append(info)
        if terminated:
            break
    return total_reward, step_infos
