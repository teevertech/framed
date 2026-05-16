"""Tests for framed.baselines.

Structure
---------
TestGreedyNearestAction     Distance-only greedy: closest valid member.
TestGreedyCostAwareAction   Cost-aware greedy: includes collision penalty.
TestRunEpisode              Episode runner utility.
TestBaselineComparison      Cross-baseline properties.

The ``_collision_panel`` helper creates a 4-member panel whose geometry
makes it easy to reason about which paths collide with the beam obstacle:

      beam:       x=[20, 100], y=[40, 55]   (center 60, 47.5)
      step_stone: x=[ 0,  10], y=[ 0, 10]   (center  5,  5  )
      close_tgt:  x=[55,  65], y=[70, 90]   (center 60, 80  )
      far_tgt:    x=[105,115], y=[ 0, 10]   (center 110, 5  )

After beam(0) → step_stone(1), the robot sits at (5, 5).
  • Path to close_tgt crosses the beam → collision.
  • Path to far_tgt stays below the beam → clear.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from framed.baselines import (
    greedy_cost_aware_action,
    greedy_nearest_action,
    run_episode,
)
from framed.env import PanelEnv
from framed.panel import (
    LUMBER_THICKNESS,
    Member,
    MemberKind,
    Panel,
    generate_random_panel,
)
from framed.units import feet, inches


# ===================================================================== #
# Fixtures and helpers                                                   #
# ===================================================================== #

def _collision_panel() -> Panel:
    """See module docstring for geometry description."""
    return Panel(
        wall_length=feet(10),
        wall_height=feet(8),
        members=[
            Member(
                id="beam",
                kind=MemberKind.HEADER,
                position=(inches(20), inches(40)),
                size=(inches(80), inches(15)),
                prerequisites=[],
            ),
            Member(
                id="step_stone",
                kind=MemberKind.BOTTOM_PLATE,
                position=(0.0, 0.0),
                size=(inches(10), inches(10)),
                prerequisites=[],
            ),
            Member(
                id="close_tgt",
                kind=MemberKind.COMMON_STUD,
                position=(inches(55), inches(70)),
                size=(inches(10), inches(20)),
                prerequisites=[],
            ),
            Member(
                id="far_tgt",
                kind=MemberKind.COMMON_STUD,
                position=(inches(105), 0.0),
                size=(inches(10), inches(10)),
                prerequisites=[],
            ),
        ],
    )


def _equidistant_panel() -> Panel:
    """Two members whose centers are equidistant from (50, 50).

    left  center: (21, 50) — distance 29 from robot.
    right center: (79, 50) — distance 29 from robot.
    """
    return Panel(
        wall_length=feet(10),
        wall_height=feet(8),
        members=[
            Member(
                id="left",
                kind=MemberKind.COMMON_STUD,
                position=(inches(20), inches(40)),
                size=(inches(2), inches(20)),
                prerequisites=[],
            ),
            Member(
                id="right",
                kind=MemberKind.COMMON_STUD,
                position=(inches(78), inches(40)),
                size=(inches(2), inches(20)),
                prerequisites=[],
            ),
        ],
    )


def _masked_panel() -> Panel:
    """Three members where the closest to (0,0) requires the second-closest
    to be placed first.

    a_closest center: ( 6, 10) — distance ≈ 11.7
    b_second  center: (21, 10) — distance ≈ 23.3
    c_far     center: (81, 10) — distance ≈ 81.6
    """
    return Panel(
        wall_length=feet(10),
        wall_height=feet(8),
        members=[
            Member(
                id="a_closest",
                kind=MemberKind.COMMON_STUD,
                position=(inches(5), inches(5)),
                size=(inches(2), inches(10)),
                prerequisites=["b_second"],
            ),
            Member(
                id="b_second",
                kind=MemberKind.COMMON_STUD,
                position=(inches(20), inches(5)),
                size=(inches(2), inches(10)),
                prerequisites=[],
            ),
            Member(
                id="c_far",
                kind=MemberKind.COMMON_STUD,
                position=(inches(80), inches(5)),
                size=(inches(2), inches(10)),
                prerequisites=[],
            ),
        ],
    )


def _simple_panel() -> Panel:
    """3-member panel with strict ordering: bot → stud → top."""
    return Panel(
        wall_length=feet(10),
        wall_height=feet(8),
        members=[
            Member(
                id="bot",
                kind=MemberKind.BOTTOM_PLATE,
                position=(0.0, 0.0),
                size=(feet(10), LUMBER_THICKNESS),
                prerequisites=[],
            ),
            Member(
                id="stud",
                kind=MemberKind.COMMON_STUD,
                position=(inches(48), LUMBER_THICKNESS),
                size=(LUMBER_THICKNESS, inches(93)),
                prerequisites=["bot"],
            ),
            Member(
                id="top",
                kind=MemberKind.TOP_PLATE,
                position=(0.0, inches(94.5)),
                size=(feet(10), LUMBER_THICKNESS),
                prerequisites=["stud"],
            ),
        ],
    )


# ===================================================================== #
# greedy_nearest_action                                                  #
# ===================================================================== #

class TestGreedyNearestAction:

    def test_picks_closest_member(self) -> None:
        """Robot at (0, 0); b_second (dist ≈ 23) is the closest valid
        member (a_closest is masked)."""
        env = PanelEnv(_masked_panel())
        env.reset()
        action = greedy_nearest_action(env)
        assert env.panel.members[action].id == "b_second"

    def test_tiebreak_picks_lowest_index(self) -> None:
        """Both 'left' (index 0) and 'right' (index 1) are equidistant
        from the robot at (50, 50).  Lowest index should win."""
        env = PanelEnv(_equidistant_panel(), initial_robot_pos=(50.0, 50.0))
        env.reset()
        action = greedy_nearest_action(env)
        assert action == 0
        assert env.panel.members[action].id == "left"

    def test_respects_action_mask(self) -> None:
        """a_closest is nearest to (0, 0) but requires b_second first.
        Nearest should pick b_second, not a_closest."""
        env = PanelEnv(_masked_panel())
        env.reset()
        # Confirm a_closest IS actually closer
        a_dist = math.hypot(6.0, 10.0)
        b_dist = math.hypot(21.0, 10.0)
        assert a_dist < b_dist
        # But mask blocks it
        action = greedy_nearest_action(env)
        assert action == 1  # b_second

    def test_raises_on_terminal_state(self) -> None:
        env = PanelEnv(_equidistant_panel())
        env.reset()
        env.step(0)
        env.step(1)
        with pytest.raises(ValueError, match="terminal"):
            greedy_nearest_action(env)

    def test_runs_full_episode_simple(self) -> None:
        """The simple panel has a strict total order — nearest must still
        follow it (only one valid action at each step)."""
        env = PanelEnv(_simple_panel())
        total, infos = run_episode(env, greedy_nearest_action)
        assert len(infos) == 3
        assert math.isfinite(total)
        assert total <= 0.0

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_runs_full_episode_random_panel(self, seed: int) -> None:
        panel = generate_random_panel(seed=seed)
        env = PanelEnv(panel)
        total, infos = run_episode(env, greedy_nearest_action)
        assert len(infos) == env.n_members
        assert total <= 0.0


# ===================================================================== #
# greedy_cost_aware_action                                               #
# ===================================================================== #

class TestGreedyCostAwareAction:

    def test_matches_nearest_on_first_step(self) -> None:
        """With nothing placed, there can be no collisions, so cost-aware
        and nearest should agree (both pick the closest valid member)."""
        for seed in range(5):
            panel = generate_random_panel(seed=seed)
            env_n = PanelEnv(panel)
            env_c = PanelEnv(panel)
            env_n.reset()
            env_c.reset()
            assert greedy_nearest_action(env_n) == greedy_cost_aware_action(env_c)

    def test_avoids_collision_when_detour_cheaper(self) -> None:
        """With k=2, the collision path to close_tgt costs 3× base time,
        making far_tgt (no collision) cheaper even though it's farther.

        After beam(0) → step_stone(1), robot at (5, 5):
          close_tgt center (60, 80): dist ≈ 93, cost = 93 × 3 = 279
          far_tgt center  (110, 5):  dist = 105, cost = 105 × 1 = 105
        """
        env = PanelEnv(
            _collision_panel(),
            robot_speed=1.0,
            collision_penalty_multiplier=2.0,
        )
        env.reset()
        env.step(0)  # beam
        env.step(1)  # step_stone → robot at (5, 5)

        action = greedy_cost_aware_action(env)
        assert env.panel.members[action].id == "far_tgt"

    def test_nearest_picks_closer_colliding_path(self) -> None:
        """In the same setup, nearest ignores collisions and picks
        close_tgt because it's nearer."""
        env = PanelEnv(
            _collision_panel(),
            robot_speed=1.0,
            collision_penalty_multiplier=2.0,
        )
        env.reset()
        env.step(0)  # beam
        env.step(1)  # step_stone → robot at (5, 5)

        action = greedy_nearest_action(env)
        assert env.panel.members[action].id == "close_tgt"

    def test_picks_colliding_when_still_cheapest(self) -> None:
        """With a tiny penalty (k=0.1), close_tgt's collision cost is
        still lower than far_tgt's clear cost.

          close_tgt: dist ≈ 93, cost = 93 × 1.1 ≈ 102.3
          far_tgt:   dist = 105, cost = 105 × 1.0 = 105.0
        Cost-aware should pick close_tgt — it doesn't blindly avoid
        all collisions."""
        env = PanelEnv(
            _collision_panel(),
            robot_speed=1.0,
            collision_penalty_multiplier=0.1,
        )
        env.reset()
        env.step(0)  # beam
        env.step(1)  # step_stone

        action = greedy_cost_aware_action(env)
        assert env.panel.members[action].id == "close_tgt"

    def test_matches_nearest_when_penalty_is_zero(self) -> None:
        """With k=0, the cost function is pure travel time — identical to
        nearest.  Both should agree at every step."""
        panel = generate_random_panel(seed=5)
        env_n = PanelEnv(panel, collision_penalty_multiplier=0.0)
        env_c = PanelEnv(panel, collision_penalty_multiplier=0.0)
        env_n.reset()
        env_c.reset()
        for _ in range(env_n.n_members):
            an = greedy_nearest_action(env_n)
            ac = greedy_cost_aware_action(env_c)
            assert an == ac
            env_n.step(an)
            env_c.step(ac)

    def test_raises_on_terminal_state(self) -> None:
        env = PanelEnv(_equidistant_panel())
        env.reset()
        env.step(0)
        env.step(1)
        with pytest.raises(ValueError, match="terminal"):
            greedy_cost_aware_action(env)

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_runs_full_episode_random_panel(self, seed: int) -> None:
        panel = generate_random_panel(seed=seed)
        env = PanelEnv(panel)
        total, infos = run_episode(env, greedy_cost_aware_action)
        assert len(infos) == env.n_members
        assert total <= 0.0


# ===================================================================== #
# run_episode                                                            #
# ===================================================================== #

class TestRunEpisode:

    def test_returns_total_reward_and_infos(self) -> None:
        env = PanelEnv(_simple_panel())
        total, infos = run_episode(env, greedy_nearest_action)
        assert isinstance(total, float)
        assert isinstance(infos, list)

    def test_total_reward_is_sum_of_step_rewards(self) -> None:
        """Manual sanity: total should equal the sum of individual travel
        times (negated, with penalties)."""
        env = PanelEnv(_simple_panel(), robot_speed=10.0)
        total, infos = run_episode(env, greedy_nearest_action)
        reconstructed = sum(
            -info["travel_time"]
            * (1.0 + env.collision_penalty_multiplier * float(info["collided"]))
            for info in infos
        )
        assert total == pytest.approx(reconstructed, rel=1e-6)

    def test_infos_length_matches_n_members(self) -> None:
        panel = generate_random_panel(seed=0)
        env = PanelEnv(panel)
        _, infos = run_episode(env, greedy_cost_aware_action)
        assert len(infos) == env.n_members

    def test_works_with_lambda_policy(self) -> None:
        """A lambda that always picks the first valid action should work."""
        env = PanelEnv(_simple_panel())
        first_valid = lambda e: int(np.flatnonzero(e.action_masks())[0])  # noqa: E731
        total, infos = run_episode(env, first_valid)
        assert len(infos) == 3
        assert total <= 0.0


# ===================================================================== #
# Baseline comparison                                                    #
# ===================================================================== #

class TestBaselineComparison:

    def test_cost_aware_not_worse_on_collision_panel(self) -> None:
        """On the collision panel with k=2, cost-aware should achieve
        total reward >= nearest (i.e. less negative), because it avoids
        costly collision penalties where nearest does not."""
        panel = _collision_panel()
        env_n = PanelEnv(panel, robot_speed=1.0, collision_penalty_multiplier=2.0)
        env_c = PanelEnv(panel, robot_speed=1.0, collision_penalty_multiplier=2.0)
        total_n, _ = run_episode(env_n, greedy_nearest_action)
        total_c, _ = run_episode(env_c, greedy_cost_aware_action)
        # cost_aware total should be >= nearest total (less negative = better)
        assert total_c >= total_n - 1e-9

    @pytest.mark.parametrize("seed", list(range(20)))
    def test_both_complete_random_episodes(self, seed: int) -> None:
        """Both baselines must always produce valid, complete episodes."""
        panel = generate_random_panel(seed=seed)
        env_n = PanelEnv(panel)
        env_c = PanelEnv(panel)
        total_n, infos_n = run_episode(env_n, greedy_nearest_action)
        total_c, infos_c = run_episode(env_c, greedy_cost_aware_action)
        assert len(infos_n) == env_n.n_members
        assert len(infos_c) == env_c.n_members
        assert math.isfinite(total_n)
        assert math.isfinite(total_c)

    def test_baselines_are_competitive_across_seeds(self) -> None:
        """Neither greedy heuristic strictly dominates the other.

        Nearest stays close to the remaining work but eats collision
        penalties.  Cost-aware avoids collisions but detours can strand
        the robot far from the remaining cluster.  Their aggregate scores
        should be in the same ballpark — neither should be catastrophically
        worse (> 2× the other).

        This gap is exactly what the RL agent should learn to exploit:
        plan multi-step paths that balance proximity and obstruction."""
        total_n_sum = 0.0
        total_c_sum = 0.0
        n_seeds = 50
        for seed in range(n_seeds):
            panel = generate_random_panel(seed=seed)
            env_n = PanelEnv(panel, collision_penalty_multiplier=2.0)
            env_c = PanelEnv(panel, collision_penalty_multiplier=2.0)
            total_n, _ = run_episode(env_n, greedy_nearest_action)
            total_c, _ = run_episode(env_c, greedy_cost_aware_action)
            total_n_sum += total_n
            total_c_sum += total_c
        # Both are negative; neither should be more than 2× worse.
        ratio = total_c_sum / total_n_sum
        assert 0.5 < ratio < 2.0, (
            f"Baselines diverged too far: nearest avg={total_n_sum / n_seeds:.2f}, "
            f"cost_aware avg={total_c_sum / n_seeds:.2f}, ratio={ratio:.3f}"
        )
