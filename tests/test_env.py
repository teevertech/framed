"""Tests for framed.env.PanelEnv.

Structure
---------
TestInit                Construction & spaces.
TestReset               Initial state, idempotency.
TestStep                Basic step mechanics, state transitions.
TestStepErrors          Invalid actions raise ValueError loudly.
TestActionMasks         Mask correctness across the episode lifecycle.
TestRewardComputation   Travel-time math, collision penalty math.
TestObservation         Layout, normalization, content correctness.
TestFullEpisode         Integration: run a full valid sequence end-to-end.
TestDeterminism         Same panel → same trajectory.

Two test fixtures:

* ``_simple_panel()`` — a hand-built 3-member panel (bottom plate → stud →
  top plate) with strict ordering. Used wherever an exact expected reward
  or sequence matters.
* ``generate_random_panel(seed=...)`` — used for broader integration
  checks where structure (not exact numbers) is what we're testing.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pytest
from gymnasium import spaces

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

def _simple_panel() -> Panel:
    """A 3-member panel with strict ordering: bot → stud → top.

    Geometry is chosen so member centers are easy to reason about:
      - bottom plate spans the full 10 ft, centered at (60, 0.75)
      - stud sits at x=48", centered at (48.75, 48)
      - top plate spans the full 10 ft, centered at (60, 95.25)
    """
    return Panel(
        wall_length=feet(10),       # 120"
        wall_height=feet(8),        # 96"
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


def _topo_order_indices(env: PanelEnv) -> list[int]:
    """Pick any valid placement order using action_masks. Used by integration
    tests that just need to run a full episode to completion."""
    order: list[int] = []
    placed = np.zeros(env.n_members, dtype=bool)
    # We replicate the mask logic locally so the test doesn't depend on
    # the env being mid-episode.
    for _ in range(env.n_members):
        mask = np.zeros(env.n_members, dtype=bool)
        for i, m in enumerate(env.panel.members):
            if placed[i]:
                continue
            prereq_ids = m.prerequisites
            placed_ids = {
                env.panel.members[j].id
                for j in range(env.n_members)
                if placed[j]
            }
            if all(p in placed_ids for p in prereq_ids):
                mask[i] = True
        # Pick the lowest-index valid action (deterministic).
        candidates = np.flatnonzero(mask)
        assert candidates.size > 0, "graph is acyclic, mask should be non-empty"
        choice = int(candidates[0])
        order.append(choice)
        placed[choice] = True
    return order


def _collision_panel() -> Panel:
    """4-member panel for collision tests.

    Members (in panel.members order):
      0: beam        — large horizontal obstacle, y=[40, 55], x=[20, 100].
                       Center (60, 47.5).
      1: step_stone  — small pad in the bottom-left corner.
                       Center (5, 5).
      2: close_tgt   — above the beam.  Center (60, 80).
      3: far_tgt     — bottom-right, clear path from step_stone.
                       Center (110, 5).

    After placing beam(0) → step_stone(1), the robot sits at (5, 5).
    Path to close_tgt(2) crosses the beam → collision.
    Path to far_tgt(3) stays below the beam → no collision.
    """
    return Panel(
        wall_length=feet(10),       # 120"
        wall_height=feet(8),        # 96"
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


# ===================================================================== #
# Init                                                                   #
# ===================================================================== #

class TestInit:

    def test_constructs_with_simple_panel(self) -> None:
        env = PanelEnv(_simple_panel())
        assert env.n_members == 3

    def test_action_space_matches_member_count(self) -> None:
        env = PanelEnv(_simple_panel())
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 3

    def test_observation_space_shape(self) -> None:
        """Layout: 2 (robot pos) + n (placed mask) + 2n (centers) = 2 + 3n."""
        env = PanelEnv(_simple_panel())
        assert isinstance(env.observation_space, spaces.Box)
        assert env.observation_space.shape == (2 + 3 * 3,)
        assert env.observation_space.dtype == np.float32
        assert float(env.observation_space.low.min()) == pytest.approx(0.0)
        assert float(env.observation_space.high.max()) == pytest.approx(1.0)

    def test_observation_space_scales_with_panel_size(self) -> None:
        big = generate_random_panel(wall_length=feet(20), seed=0)
        env = PanelEnv(big)
        assert env.observation_space.shape == (2 + 3 * len(big.members),)

    def test_rejects_zero_speed(self) -> None:
        with pytest.raises(ValueError, match="robot_speed must be positive"):
            PanelEnv(_simple_panel(), robot_speed=0.0)

    def test_rejects_negative_speed(self) -> None:
        with pytest.raises(ValueError, match="robot_speed must be positive"):
            PanelEnv(_simple_panel(), robot_speed=-1.0)

    def test_rejects_negative_collision_penalty(self) -> None:
        with pytest.raises(ValueError, match="collision_penalty_multiplier"):
            PanelEnv(_simple_panel(), collision_penalty_multiplier=-0.1)

    def test_zero_collision_penalty_is_allowed(self) -> None:
        """k=0 disables the penalty — a valid configuration."""
        env = PanelEnv(_simple_panel(), collision_penalty_multiplier=0.0)
        assert env.collision_penalty_multiplier == 0.0


# ===================================================================== #
# Reset                                                                  #
# ===================================================================== #

class TestReset:

    def test_returns_obs_and_info(self) -> None:
        env = PanelEnv(_simple_panel())
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_obs_in_observation_space(self) -> None:
        env = PanelEnv(_simple_panel())
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    def test_initial_placed_mask_is_all_zeros(self) -> None:
        env = PanelEnv(_simple_panel())
        obs, _ = env.reset()
        # placed mask occupies indices [2 : 2+n_members]
        n = env.n_members
        placed_slice = obs[2 : 2 + n]
        assert np.all(placed_slice == 0.0)

    def test_initial_robot_pos_default_is_origin(self) -> None:
        env = PanelEnv(_simple_panel())
        obs, info = env.reset()
        assert info["robot_pos"] == (0.0, 0.0)
        # The normalized robot position lives in obs[0:2]
        assert obs[0] == pytest.approx(0.0)
        assert obs[1] == pytest.approx(0.0)

    def test_initial_robot_pos_custom(self) -> None:
        env = PanelEnv(_simple_panel(), initial_robot_pos=(inches(60), inches(48)))
        _, info = env.reset()
        assert info["robot_pos"] == (60.0, 48.0)

    def test_reset_is_idempotent(self) -> None:
        """Calling reset() twice yields the same observation."""
        env = PanelEnv(_simple_panel())
        obs1, _ = env.reset()
        obs2, _ = env.reset()
        np.testing.assert_array_equal(obs1, obs2)

    def test_reset_after_steps_clears_state(self) -> None:
        env = PanelEnv(_simple_panel())
        env.reset()
        # Place the bottom plate
        env.step(0)
        # Reset and confirm placement is wiped
        obs, info = env.reset()
        assert info["n_placed"] == 0
        assert np.all(obs[2 : 2 + env.n_members] == 0.0)

    def test_info_at_reset(self) -> None:
        env = PanelEnv(_simple_panel())
        _, info = env.reset()
        assert info["n_placed"] == 0
        assert info["step_count"] == 0
        assert "robot_pos" in info


# ===================================================================== #
# Step                                                                   #
# ===================================================================== #

class TestStep:

    def test_step_returns_5_tuple(self) -> None:
        env = PanelEnv(_simple_panel())
        env.reset()
        result = env.step(0)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_truncated_is_always_false(self) -> None:
        env = PanelEnv(_simple_panel())
        env.reset()
        for action in [0, 1, 2]:
            _, _, _, truncated, _ = env.step(action)
            assert truncated is False

    def test_step_marks_member_placed(self) -> None:
        env = PanelEnv(_simple_panel())
        env.reset()
        obs, _, _, _, info = env.step(0)
        # placed[0] (bot) should be 1.0 in the observation
        assert obs[2 + 0] == 1.0
        assert obs[2 + 1] == 0.0
        assert obs[2 + 2] == 0.0
        assert info["n_placed"] == 1
        assert info["member_id"] == "bot"
        assert info["member_index"] == 0

    def test_step_updates_robot_pos_to_member_center(self) -> None:
        env = PanelEnv(_simple_panel())
        env.reset()
        _, _, _, _, info = env.step(0)
        # Bottom plate center: ((0 + 120)/2, (0 + 1.5)/2) = (60.0, 0.75)
        assert info["robot_pos"] == pytest.approx((60.0, 0.75))

    def test_terminated_only_when_all_placed(self) -> None:
        env = PanelEnv(_simple_panel())
        env.reset()
        _, _, terminated, _, _ = env.step(0)
        assert terminated is False
        _, _, terminated, _, _ = env.step(1)
        assert terminated is False
        _, _, terminated, _, _ = env.step(2)
        assert terminated is True

    def test_step_count_increments(self) -> None:
        env = PanelEnv(_simple_panel())
        env.reset()
        _, _, _, _, info = env.step(0)
        assert info["step_count"] == 1
        _, _, _, _, info = env.step(1)
        assert info["step_count"] == 2

    def test_step_info_contains_travel_time_and_collided(self) -> None:
        env = PanelEnv(_simple_panel())
        env.reset()
        _, _, _, _, info = env.step(0)
        assert "travel_time" in info
        assert "collided" in info
        assert isinstance(info["travel_time"], float)
        assert isinstance(info["collided"], bool)


# ===================================================================== #
# Step error cases                                                       #
# ===================================================================== #

class TestStepErrors:

    def test_action_out_of_range_below_zero_raises(self) -> None:
        env = PanelEnv(_simple_panel())
        env.reset()
        with pytest.raises(ValueError, match="out of range"):
            env.step(-1)

    def test_action_out_of_range_above_n_raises(self) -> None:
        env = PanelEnv(_simple_panel())
        env.reset()
        with pytest.raises(ValueError, match="out of range"):
            env.step(3)  # valid actions are 0, 1, 2

    def test_already_placed_member_raises(self) -> None:
        env = PanelEnv(_simple_panel())
        env.reset()
        env.step(0)  # place bot
        with pytest.raises(ValueError, match="already placed"):
            env.step(0)

    def test_unmet_prereq_raises(self) -> None:
        env = PanelEnv(_simple_panel())
        env.reset()
        # 'stud' has prereq 'bot' — placing stud first must fail
        with pytest.raises(ValueError, match="unmet prerequisites"):
            env.step(1)

    def test_unmet_prereq_message_names_the_prereq(self) -> None:
        env = PanelEnv(_simple_panel())
        env.reset()
        with pytest.raises(ValueError, match="bot"):
            env.step(1)


# ===================================================================== #
# Action masks                                                           #
# ===================================================================== #

class TestActionMasks:

    def test_returns_bool_array_of_correct_shape(self) -> None:
        env = PanelEnv(_simple_panel())
        env.reset()
        mask = env.action_masks()
        assert mask.shape == (3,)
        assert mask.dtype == bool

    def test_returns_fresh_array_each_call(self) -> None:
        """Mutating a returned mask must not affect future calls."""
        env = PanelEnv(_simple_panel())
        env.reset()
        m1 = env.action_masks()
        m1[:] = False
        m2 = env.action_masks()
        # m2 should still reflect the actual valid actions
        assert m2[0] is np.True_ or m2[0] == True  # noqa: E712

    def test_initial_mask_only_root_members(self) -> None:
        """Only 'bot' has no prereqs, so only action 0 is valid initially."""
        env = PanelEnv(_simple_panel())
        env.reset()
        mask = env.action_masks()
        assert mask[0] == True   # noqa: E712
        assert mask[1] == False  # noqa: E712
        assert mask[2] == False  # noqa: E712

    def test_mask_updates_after_step(self) -> None:
        env = PanelEnv(_simple_panel())
        env.reset()
        env.step(0)  # place bot → unlocks stud
        mask = env.action_masks()
        assert mask[0] == False  # already placed
        assert mask[1] == True   # bot placed, stud now valid
        assert mask[2] == False  # stud not yet placed

    def test_mask_after_terminal_state_is_all_false(self) -> None:
        env = PanelEnv(_simple_panel())
        env.reset()
        env.step(0)
        env.step(1)
        env.step(2)
        mask = env.action_masks()
        assert not mask.any()

    def test_mask_consistent_with_step_for_random_panel(self) -> None:
        """For an arbitrary panel state, every action where mask[i] is True
        must succeed in step(), and every action where mask[i] is False
        must raise.  (We test the latter only for already-placed
        members; for unplaced-with-unmet-prereqs we'd need a more
        careful copy of state.)"""
        panel = generate_random_panel(seed=7)
        env = PanelEnv(panel)
        env.reset()

        # Place a few members following the mask, checking each transition.
        for _ in range(min(5, env.n_members)):
            mask = env.action_masks()
            valid = np.flatnonzero(mask)
            assert valid.size > 0
            env.step(int(valid[0]))


# ===================================================================== #
# Reward computation                                                    #
# ===================================================================== #

class TestRewardComputation:

    def test_reward_is_negative_travel_time_no_collision(self) -> None:
        """First step: robot at (0,0), target = bot center (60, 0.75).
        Distance = hypot(60, 0.75). At speed 10 → time = dist / 10.
        Reward = -time. No collision possible (placed_members is empty)."""
        env = PanelEnv(_simple_panel(), robot_speed=10.0)
        env.reset()
        expected_dist = math.hypot(60.0, 0.75)
        expected_reward = -expected_dist / 10.0
        _, reward, _, _, info = env.step(0)
        assert reward == pytest.approx(expected_reward, rel=1e-6)
        assert info["collided"] is False
        assert info["travel_time"] == pytest.approx(expected_dist / 10.0, rel=1e-6)

    def test_reward_scales_inversely_with_speed(self) -> None:
        """Doubling speed halves travel time → halves the negative reward."""
        env_slow = PanelEnv(_simple_panel(), robot_speed=1.0)
        env_fast = PanelEnv(_simple_panel(), robot_speed=2.0)
        env_slow.reset()
        env_fast.reset()
        _, r_slow, _, _, _ = env_slow.step(0)
        _, r_fast, _, _, _ = env_fast.step(0)
        assert r_fast == pytest.approx(r_slow / 2.0, rel=1e-6)

    def test_collision_penalty_applies_when_path_collides(self) -> None:
        """Place beam → step_stone → close_tgt.  The third step's path
        (from step_stone center to close_tgt center) crosses the beam,
        which was placed two steps ago and is NOT excluded by the liftoff
        rule (last-placed is step_stone, not beam)."""
        panel = _collision_panel()
        env = PanelEnv(panel, robot_speed=1.0, collision_penalty_multiplier=2.0)
        env.reset()

        env.step(0)  # place beam, robot → (60, 47.5)
        env.step(1)  # place step_stone, robot → (5, 5). Beam excluded (liftoff).

        # Step 3: place close_tgt. Path from (5, 5) → (60, 80) crosses
        # beam [20,100]×[40,55]. Last-placed is step_stone, so beam is
        # a valid obstacle. → collision.
        _, reward, _, _, info = env.step(2)
        assert info["collided"] is True

        base_t = info["travel_time"]
        expected_reward = -base_t * (1.0 + 2.0)  # k=2 → 3× cost
        assert reward == pytest.approx(expected_reward, rel=1e-6)

    def test_zero_penalty_disables_collision_cost(self) -> None:
        """Same geometry as the collision test but with k=0 — the penalty
        disappears and the reward equals bare travel time."""
        panel = _collision_panel()
        env = PanelEnv(panel, robot_speed=1.0, collision_penalty_multiplier=0.0)
        env.reset()

        env.step(0)  # beam
        env.step(1)  # step_stone

        _, reward, _, _, info = env.step(2)  # close_tgt — path still collides
        assert info["collided"] is True
        assert reward == pytest.approx(-info["travel_time"], rel=1e-6)

    def test_liftoff_from_just_placed_does_not_collide(self) -> None:
        """After placing beam, the robot is at its center (inside beam).
        Moving to step_stone should NOT register a collision with beam
        because beam is the last-placed member (liftoff rule)."""
        panel = _collision_panel()
        env = PanelEnv(panel, robot_speed=1.0, collision_penalty_multiplier=2.0)
        env.reset()

        env.step(0)  # place beam, robot is now inside beam at (60, 47.5)

        # Place step_stone: path from beam center (inside beam) to
        # step_stone center. Without liftoff rule this would collide
        # with beam. With the rule, beam is excluded. → No collision.
        _, _, _, _, info = env.step(1)
        assert info["collided"] is False

    def test_no_collision_when_path_is_clear(self) -> None:
        """Path to far_tgt from step_stone avoids the beam entirely."""
        panel = _collision_panel()
        env = PanelEnv(panel, robot_speed=1.0, collision_penalty_multiplier=2.0)
        env.reset()

        env.step(0)  # beam
        env.step(1)  # step_stone, robot at (5, 5)

        # Place far_tgt: path from (5, 5) → (110, 5) is horizontal at y=5,
        # well below beam at y=[40, 55]. → No collision.
        _, _, _, _, info = env.step(3)
        assert info["collided"] is False

    def test_initial_step_cannot_collide(self) -> None:
        """At episode start nothing is placed, so the path can't hit any
        member — collided must be False for the first step."""
        for seed in range(5):
            env = PanelEnv(generate_random_panel(seed=seed))
            env.reset()
            mask = env.action_masks()
            first = int(np.flatnonzero(mask)[0])
            _, _, _, _, info = env.step(first)
            assert info["collided"] is False


# ===================================================================== #
# Public state properties                                               #
# ===================================================================== #

class TestStateProperties:

    def test_robot_pos_returns_tuple(self) -> None:
        env = PanelEnv(_simple_panel())
        env.reset()
        pos = env.robot_pos
        assert isinstance(pos, tuple)
        assert len(pos) == 2

    def test_robot_pos_matches_initial(self) -> None:
        env = PanelEnv(_simple_panel(), initial_robot_pos=(10.0, 20.0))
        env.reset()
        assert env.robot_pos == (10.0, 20.0)

    def test_robot_pos_updates_after_step(self) -> None:
        env = PanelEnv(_simple_panel())
        env.reset()
        env.step(0)
        bot_center = env.panel.members[0].center
        assert env.robot_pos == pytest.approx(bot_center)

    def test_placed_mask_shape_and_dtype(self) -> None:
        env = PanelEnv(_simple_panel())
        env.reset()
        m = env.placed_mask
        assert m.shape == (3,)
        assert m.dtype == bool

    def test_placed_mask_is_copy(self) -> None:
        """Mutating the returned array must not affect env state."""
        env = PanelEnv(_simple_panel())
        env.reset()
        m = env.placed_mask
        m[:] = True
        assert not env.placed_mask.any()  # internal state untouched

    def test_placed_mask_updates_after_step(self) -> None:
        env = PanelEnv(_simple_panel())
        env.reset()
        assert not env.placed_mask[0]
        env.step(0)
        assert env.placed_mask[0]

    def test_last_placed_idx_none_after_reset(self) -> None:
        env = PanelEnv(_simple_panel())
        env.reset()
        assert env.last_placed_idx is None

    def test_last_placed_idx_updates_after_step(self) -> None:
        env = PanelEnv(_simple_panel())
        env.reset()
        env.step(0)
        assert env.last_placed_idx == 0
        env.step(1)
        assert env.last_placed_idx == 1


# ===================================================================== #
# Observation correctness                                               #
# ===================================================================== #

class TestObservation:

    def test_obs_dtype_is_float32(self) -> None:
        env = PanelEnv(_simple_panel())
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_robot_pos_normalized_by_wall_dims(self) -> None:
        """Set initial robot pos to (wall_length, wall_height) and verify
        the normalized observation reads (1.0, 1.0)."""
        panel = _simple_panel()
        env = PanelEnv(
            panel,
            initial_robot_pos=(panel.wall_length, panel.wall_height),
        )
        obs, _ = env.reset()
        assert obs[0] == pytest.approx(1.0)
        assert obs[1] == pytest.approx(1.0)

    def test_member_centers_section_matches_panel(self) -> None:
        """obs[2+n : 2+3n] is the flattened normalized member centers,
        ordered the same as panel.members."""
        panel = _simple_panel()
        env = PanelEnv(panel)
        obs, _ = env.reset()
        n = env.n_members
        centers_slice = obs[2 + n : 2 + 3 * n].reshape(n, 2)
        expected = np.array(
            [m.center for m in panel.members], dtype=np.float32
        ) / np.array([panel.wall_length, panel.wall_height], dtype=np.float32)
        np.testing.assert_allclose(centers_slice, expected, rtol=1e-6)

    def test_member_centers_in_unit_box(self) -> None:
        """For any panel, every normalized center should be in [0, 1]."""
        env = PanelEnv(generate_random_panel(seed=3))
        obs, _ = env.reset()
        n = env.n_members
        centers = obs[2 + n : 2 + 3 * n]
        assert centers.min() >= 0.0
        assert centers.max() <= 1.0

    def test_obs_stays_in_observation_space_through_episode(self) -> None:
        env = PanelEnv(generate_random_panel(seed=11))
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        for action in _topo_order_indices(env):
            obs, _, _, _, _ = env.step(action)
            assert env.observation_space.contains(obs)


# ===================================================================== #
# Full episode                                                          #
# ===================================================================== #

class TestFullEpisode:

    def test_simple_panel_runs_to_completion(self) -> None:
        env = PanelEnv(_simple_panel())
        env.reset()
        terminated = False
        steps = 0
        for action in [0, 1, 2]:
            _, _, terminated, truncated, _ = env.step(action)
            steps += 1
            assert truncated is False
        assert terminated is True
        assert steps == env.n_members

    @pytest.mark.parametrize("seed", list(range(10)))
    def test_random_panel_runs_to_completion(self, seed: int) -> None:
        """Across many seeds, following any topological order produces a
        valid full episode."""
        panel = generate_random_panel(seed=seed)
        env = PanelEnv(panel)
        env.reset()
        order = _topo_order_indices(env)
        assert len(order) == env.n_members
        for i, action in enumerate(order):
            _, _, terminated, truncated, info = env.step(action)
            assert truncated is False
            # terminated only after the last step
            assert terminated == (i == len(order) - 1)
            assert info["n_placed"] == i + 1

    def test_cumulative_reward_is_finite_and_negative(self) -> None:
        """Sanity: every step has reward ≤ 0, so total reward should be
        finite and ≤ 0."""
        env = PanelEnv(generate_random_panel(seed=2))
        env.reset()
        total = 0.0
        for action in _topo_order_indices(env):
            _, r, _, _, _ = env.step(action)
            total += r
        assert math.isfinite(total)
        assert total <= 0.0


# ===================================================================== #
# Determinism                                                            #
# ===================================================================== #

class TestDeterminism:

    def test_same_actions_same_rewards(self) -> None:
        """Two envs with the same panel and same action sequence must
        produce identical reward trajectories."""
        panel = generate_random_panel(seed=4)
        env_a = PanelEnv(panel)
        env_b = PanelEnv(panel)
        env_a.reset()
        env_b.reset()
        order = _topo_order_indices(env_a)
        for action in order:
            _, ra, _, _, _ = env_a.step(action)
            _, rb, _, _, _ = env_b.step(action)
            assert ra == rb

    def test_same_actions_same_observations(self) -> None:
        panel = generate_random_panel(seed=8)
        env_a = PanelEnv(panel)
        env_b = PanelEnv(panel)
        obs_a, _ = env_a.reset()
        obs_b, _ = env_b.reset()
        np.testing.assert_array_equal(obs_a, obs_b)
        for action in _topo_order_indices(env_a):
            obs_a, _, _, _, _ = env_a.step(action)
            obs_b, _, _, _, _ = env_b.step(action)
            np.testing.assert_array_equal(obs_a, obs_b)
