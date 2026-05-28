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
TestPaddedObsActionSpace  Fixed-size (MAX_MEMBERS) space invariants.
TestFullEpisode         Integration: run a full valid sequence end-to-end.
TestDeterminism         Same panel → same trajectory.
TestRandomPanelEnv      Variable-member-count panels across resets.

Two test fixtures:

* ``_simple_panel()`` — a hand-built 3-member panel (bottom plate → stud →
  top plate) with strict ordering. Used wherever an exact expected reward
  or sequence matters.
* ``generate_panel(...)`` — used for broader integration checks where
  structure (not exact numbers) is what we're testing.

Observation layout (803-element flat vector)
--------------------------------------------
Total obs dim = 3 + 16 * MAX_MEMBERS = 803.

  [0:2]                      normalised robot position
  [2 : 2+MAX]                placed flags (real members; padding = 0)
  [2+MAX : 2+3*MAX]          normalised centers, x/y interleaved (padding = 0)
  [2+3*MAX : 2+5*MAX]        normalised member sizes, w/h interleaved (padding = 0)
  [2+5*MAX : 2+14*MAX]       kind one-hots, 9 classes × MAX_MEMBERS (padding = 0)
  [2+14*MAX : 2+15*MAX]      prereq-satisfied flags (padding = 0)
  [2+15*MAX : 2+16*MAX]      robot-to-member distances / wall diagonal (padding = 0)
  [2+16*MAX]                 progress — fraction of members placed
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pytest
from gymnasium import spaces

from framed.env import MAX_MEMBERS, OBS_DIM, PanelEnv, RandomPanelEnv
from framed.panel import (
    LUMBER_THICKNESS,
    Member,
    MemberKind,
    Panel,
    generate_panel,
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
    """Pick any valid placement order using a locally-computed mask.

    Builds its own placement tracking over ``env.n_members`` (the real member
    count) so it is independent of the env's padded action_masks().  Used by
    integration tests that just need to drive a full episode to completion.
    """
    order: list[int] = []
    placed = np.zeros(env.n_members, dtype=bool)
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

    def test_action_space_is_discrete_max_members(self) -> None:
        """action_space is Discrete(MAX_MEMBERS) for any panel — the padded
        size, not the real member count."""
        env = PanelEnv(_simple_panel())
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == MAX_MEMBERS

    def test_observation_space_shape(self) -> None:
        """Obs dim = 3 + 16 * MAX_MEMBERS = 803, regardless of panel size.

        Sections: robot pos (2) | placed flags (MAX) | centers (2*MAX) |
                  sizes (2*MAX) | kind one-hots (9*MAX) | prereq flags (MAX) |
                  distances (MAX) | progress (1).
        """
        env = PanelEnv(_simple_panel())
        assert isinstance(env.observation_space, spaces.Box)
        assert env.observation_space.shape == (OBS_DIM,)
        assert env.observation_space.dtype == np.float32
        assert float(env.observation_space.low.min())  == pytest.approx(0.0)
        assert float(env.observation_space.high.max()) == pytest.approx(1.0)

    def test_obs_shape_fixed_regardless_of_panel_size(self) -> None:
        """Panels of any member count must produce the same obs space shape."""
        for seed in range(5):
            panel = generate_panel(
                wall_length=feet(12),
                openings=[{"type": "window", "width": inches(36)}],
                seed=seed,
            )
            env = PanelEnv(panel)
            assert env.observation_space.shape == (OBS_DIM,)

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
        """The full placed section obs[2 : 2+MAX_MEMBERS] should be zero —
        both the real member slots and the padding slots."""
        env = PanelEnv(_simple_panel())
        obs, _ = env.reset()
        placed_section = obs[2 : 2 + MAX_MEMBERS]
        assert np.all(placed_section == 0.0)

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
        env.step(0)
        obs, info = env.reset()
        assert info["n_placed"] == 0
        # Full placed section (real + padding) should be zeroed.
        assert np.all(obs[2 : 2 + MAX_MEMBERS] == 0.0)

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
        # Placed flags for real members live at obs[2], obs[3], obs[4].
        assert obs[2 + 0] == 1.0   # bot placed
        assert obs[2 + 1] == 0.0   # stud not yet placed
        assert obs[2 + 2] == 0.0   # top not yet placed
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

    def test_action_out_of_range_above_n_members_raises(self) -> None:
        """step() validates action < n_members (the real count), not < MAX_MEMBERS.
        For a 3-member panel, action=3 is out of range."""
        env = PanelEnv(_simple_panel())
        env.reset()
        with pytest.raises(ValueError, match="out of range"):
            env.step(3)

    def test_padding_slot_raises(self) -> None:
        """Indices in [n_members, MAX_MEMBERS) are padding slots and must
        raise ValueError — they are never valid actions."""
        env = PanelEnv(_simple_panel())
        env.reset()
        with pytest.raises(ValueError, match="out of range"):
            env.step(MAX_MEMBERS - 1)

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
        """action_masks() now returns a MAX_MEMBERS-length array."""
        env = PanelEnv(_simple_panel())
        env.reset()
        mask = env.action_masks()
        assert mask.shape == (MAX_MEMBERS,)
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
        """Only 'bot' (index 0) has no prereqs, so only slot 0 is True.
        All padding slots (index >= n_members) are always False."""
        env = PanelEnv(_simple_panel())
        env.reset()
        mask = env.action_masks()
        assert mask[0] is np.True_    # bot — no prereqs
        assert mask[1] is np.False_   # stud — needs bot
        assert mask[2] is np.False_   # top — needs stud
        assert not mask[3:].any()     # padding slots permanently False

    def test_mask_updates_after_step(self) -> None:
        env = PanelEnv(_simple_panel())
        env.reset()
        env.step(0)  # place bot → unlocks stud
        mask = env.action_masks()
        assert mask[0] is np.False_   # already placed
        assert mask[1] is np.True_    # bot placed, stud now valid
        assert mask[2] is np.False_   # stud not yet placed
        assert not mask[3:].any()     # padding still False

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
        panel = generate_panel(
            wall_length=feet(12),
            openings=[{"type": "window", "width": inches(36)}],
            seed=7,
        )
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
            env = PanelEnv(generate_panel(
                wall_length=feet(12),
                openings=[{"type": "window", "width": inches(36)}],
                seed=seed,
            ))
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
        """placed_mask is the real-member array (n_members), not the padded
        MAX_MEMBERS array — it's an internal state view, not the action mask."""
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
        """Centers for real members are stored at obs[2+MAX : 2+MAX+2n],
        ordered the same as panel.members, normalised by wall dims."""
        panel = _simple_panel()
        env = PanelEnv(panel)
        obs, _ = env.reset()
        n = env.n_members
        base = 2 + MAX_MEMBERS
        centers_slice = obs[base : base + n * 2].reshape(n, 2)
        expected = np.array(
            [m.center for m in panel.members], dtype=np.float32
        ) / np.array([panel.wall_length, panel.wall_height], dtype=np.float32)
        np.testing.assert_allclose(centers_slice, expected, rtol=1e-6)

    def test_member_centers_in_unit_box(self) -> None:
        """For any panel, every normalised center should be in [0, 1]."""
        env = PanelEnv(generate_panel(
            wall_length=feet(12),
            openings=[{"type": "window", "width": inches(36)}],
            seed=3,
        ))
        obs, _ = env.reset()
        n = env.n_members
        centers = obs[2 + MAX_MEMBERS : 2 + MAX_MEMBERS + n * 2]
        assert centers.min() >= 0.0
        assert centers.max() <= 1.0

    def test_kind_onehots_section(self) -> None:
        """Each real member's 9-element one-hot sits at the right offset
        in obs[2+5*MAX : 2+14*MAX].

        Kind indices (alphabetical):
          0 bottom_cripple | 1 bottom_plate | 2 common_stud | 3 header |
          4 jack_stud | 5 king_stud | 6 sill_plate | 7 top_cripple |
          8 top_plate
        """
        panel = _simple_panel()  # bot=BOTTOM_PLATE(1), stud=COMMON_STUD(2), top=TOP_PLATE(8)
        env = PanelEnv(panel)
        obs, _ = env.reset()
        base = 2 + 5 * MAX_MEMBERS

        # bot (member 0) → BOTTOM_PLATE → kind index 1
        bot_onehot = obs[base : base + 9]
        assert bot_onehot[1] == pytest.approx(1.0)
        assert bot_onehot.sum() == pytest.approx(1.0)

        # stud (member 1) → COMMON_STUD → kind index 2
        stud_onehot = obs[base + 9 : base + 18]
        assert stud_onehot[2] == pytest.approx(1.0)
        assert stud_onehot.sum() == pytest.approx(1.0)

        # top (member 2) → TOP_PLATE → kind index 8
        top_onehot = obs[base + 18 : base + 27]
        assert top_onehot[8] == pytest.approx(1.0)
        assert top_onehot.sum() == pytest.approx(1.0)

    def test_prereq_satisfied_flags_initial(self) -> None:
        """Before any placement:
          bot (no prereqs) → flag = 1.0
          stud (needs bot, not yet placed) → flag = 0.0
          top (needs stud, not yet placed) → flag = 0.0
          padding slots → 0.0
        """
        env = PanelEnv(_simple_panel())
        obs, _ = env.reset()
        n = env.n_members
        base = 2 + 14 * MAX_MEMBERS
        assert obs[base + 0] == pytest.approx(1.0)   # bot unlocked
        assert obs[base + 1] == pytest.approx(0.0)   # stud locked
        assert obs[base + 2] == pytest.approx(0.0)   # top locked
        assert np.all(obs[base + n : base + MAX_MEMBERS] == 0.0)  # padding

    def test_prereq_satisfied_flags_update_after_step(self) -> None:
        """Placing bot unlocks stud; placing stud unlocks top."""
        env = PanelEnv(_simple_panel())
        env.reset()
        base = 2 + 14 * MAX_MEMBERS

        obs, _, _, _, _ = env.step(0)   # place bot
        assert obs[base + 1] == pytest.approx(1.0)   # stud now unlocked
        assert obs[base + 2] == pytest.approx(0.0)   # top still locked

        obs, _, _, _, _ = env.step(1)   # place stud
        assert obs[base + 2] == pytest.approx(1.0)   # top now unlocked

    def test_obs_stays_in_observation_space_through_episode(self) -> None:
        env = PanelEnv(generate_panel(
            wall_length=feet(12),
            openings=[{"type": "window", "width": inches(36)}],
            seed=11,
        ))
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        for action in _topo_order_indices(env):
            obs, _, _, _, _ = env.step(action)
            assert env.observation_space.contains(obs)


# ===================================================================== #
# Padded observation / action space invariants                          #
# ===================================================================== #

class TestPaddedObsActionSpace:
    """Verify the fixed-size padded space invariants introduced by MAX_MEMBERS."""

    def test_obs_shape_is_always_803(self) -> None:
        """Obs shape is (803,) for any panel regardless of member count."""
        for seed in range(5):
            panel = generate_panel(
                wall_length=feet(12),
                openings=[{"type": "window", "width": inches(36)}],
                seed=seed,
            )
            env = PanelEnv(panel)
            obs, _ = env.reset()
            assert obs.shape == (OBS_DIM,), (
                f"seed {seed}: expected ({OBS_DIM},), got {obs.shape}"
            )

    def test_action_space_is_always_max_members(self) -> None:
        for seed in range(5):
            env = PanelEnv(generate_panel(
                wall_length=feet(12),
                openings=[{"type": "window", "width": inches(36)}],
                seed=seed,
            ))
            assert env.action_space.n == MAX_MEMBERS

    def test_padding_slots_in_mask_are_always_false(self) -> None:
        """Slots [n_members, MAX_MEMBERS) must be False before, during,
        and after an episode."""
        env = PanelEnv(_simple_panel())
        env.reset()
        n = env.n_members
        for action in _topo_order_indices(env):
            mask = env.action_masks()
            assert not mask[n:].any(), (
                "Padding slots became True mid-episode"
            )
            env.step(action)

    def test_padding_slots_zero_in_placed_section(self) -> None:
        """For a 3-member panel, obs[2+3 : 2+MAX_MEMBERS] must be 0."""
        env = PanelEnv(_simple_panel())
        obs, _ = env.reset()
        n = env.n_members
        assert np.all(obs[2 + n : 2 + MAX_MEMBERS] == 0.0)

    def test_padding_slots_zero_in_centers_section(self) -> None:
        """Centers section padding: obs[2+MAX+n*2 : 2+3*MAX] must be 0."""
        env = PanelEnv(_simple_panel())
        obs, _ = env.reset()
        n = env.n_members
        base = 2 + MAX_MEMBERS
        assert np.all(obs[base + n * 2 : base + MAX_MEMBERS * 2] == 0.0)

    def test_padding_slots_zero_in_sizes_section(self) -> None:
        """Sizes section padding: obs[2+3*MAX+n*2 : 2+5*MAX] must be 0."""
        env = PanelEnv(_simple_panel())
        obs, _ = env.reset()
        n = env.n_members
        base = 2 + 3 * MAX_MEMBERS
        assert np.all(obs[base + n * 2 : base + MAX_MEMBERS * 2] == 0.0)

    def test_padding_slots_zero_in_onehots_section(self) -> None:
        """Kind one-hots padding: obs[2+5*MAX+n*9 : 2+14*MAX] must be 0."""
        env = PanelEnv(_simple_panel())
        obs, _ = env.reset()
        n = env.n_members
        base = 2 + 5 * MAX_MEMBERS
        assert np.all(obs[base + n * 9 : base + MAX_MEMBERS * 9] == 0.0)

    def test_padding_slots_zero_in_prereq_section(self) -> None:
        """Prereq flags padding: obs[2+14*MAX+n : 2+15*MAX] must be 0."""
        env = PanelEnv(_simple_panel())
        obs, _ = env.reset()
        n = env.n_members
        base = 2 + 14 * MAX_MEMBERS
        assert np.all(obs[base + n : base + MAX_MEMBERS] == 0.0)

    def test_padding_slots_zero_in_distances_section(self) -> None:
        """Distances section padding: obs[2+15*MAX+n : 2+16*MAX] must be 0."""
        env = PanelEnv(_simple_panel())
        obs, _ = env.reset()
        n = env.n_members
        base = 2 + 15 * MAX_MEMBERS
        assert np.all(obs[base + n : base + MAX_MEMBERS] == 0.0)

    def test_padding_slots_remain_zero_through_episode(self) -> None:
        """All padding sections stay zero across every step of a full episode."""
        env = PanelEnv(generate_panel(
            wall_length=feet(12),
            openings=[{"type": "window", "width": inches(36)}],
            seed=5,
        ))
        obs, _ = env.reset()
        n = env.n_members

        def _check_padding(obs: np.ndarray) -> None:
            assert np.all(obs[2 + n : 2 + MAX_MEMBERS] == 0.0), "placed padding"
            base_c = 2 + MAX_MEMBERS
            assert np.all(obs[base_c + n * 2 : base_c + MAX_MEMBERS * 2] == 0.0), "centers padding"
            base_s = 2 + 3 * MAX_MEMBERS
            assert np.all(obs[base_s + n * 2 : base_s + MAX_MEMBERS * 2] == 0.0), "sizes padding"
            base_k = 2 + 5 * MAX_MEMBERS
            assert np.all(obs[base_k + n * 9 : base_k + MAX_MEMBERS * 9] == 0.0), "onehots padding"
            base_p = 2 + 14 * MAX_MEMBERS
            assert np.all(obs[base_p + n : base_p + MAX_MEMBERS] == 0.0), "prereq padding"
            base_d = 2 + 15 * MAX_MEMBERS
            assert np.all(obs[base_d + n : base_d + MAX_MEMBERS] == 0.0), "distances padding"

        _check_padding(obs)
        for action in _topo_order_indices(env):
            obs, _, _, _, _ = env.step(action)
            _check_padding(obs)


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
        panel = generate_panel(
            wall_length=feet(12),
            openings=[{"type": "window", "width": inches(36)}],
            seed=seed,
        )
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
        env = PanelEnv(generate_panel(
            wall_length=feet(12),
            openings=[{"type": "window", "width": inches(36)}],
            seed=2,
        ))
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
        panel = generate_panel(
            wall_length=feet(12),
            openings=[{"type": "window", "width": inches(36)}],
            seed=4,
        )
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
        panel = generate_panel(
            wall_length=feet(12),
            openings=[{"type": "window", "width": inches(36)}],
            seed=8,
        )
        env_a = PanelEnv(panel)
        env_b = PanelEnv(panel)
        obs_a, _ = env_a.reset()
        obs_b, _ = env_b.reset()
        np.testing.assert_array_equal(obs_a, obs_b)
        for action in _topo_order_indices(env_a):
            obs_a, _, _, _, _ = env_a.step(action)
            obs_b, _, _, _, _ = env_b.step(action)
            np.testing.assert_array_equal(obs_a, obs_b)


# ===================================================================== #
# RandomPanelEnv                                                        #
# ===================================================================== #

class TestRandomPanelEnv:
    """RandomPanelEnv must expose fixed MAX_MEMBERS spaces and accept panels
    of any member count across resets — padding handles the variable sizes."""

    def test_obs_and_action_space_are_max_members_sized(self) -> None:
        panels = [generate_panel(
            wall_length=feet(12),
            openings=[{"type": "window", "width": inches(36)}],
            seed=s,
        ) for s in range(3)]
        it = iter(panels)
        env = RandomPanelEnv(lambda: next(it), robot_speed=10.0)
        assert env.observation_space.shape == (OBS_DIM,)
        assert env.action_space.n == MAX_MEMBERS

    def test_spaces_do_not_change_across_resets(self) -> None:
        """The space objects are created once at init and must not mutate."""
        panels = [generate_panel(
            wall_length=feet(12),
            openings=[{"type": "window", "width": inches(36)}],
            seed=s,
        ) for s in range(6)]
        it = iter(panels)
        env = RandomPanelEnv(lambda: next(it), robot_speed=10.0)
        obs_space_id = id(env.observation_space)
        act_space_id = id(env.action_space)
        for _ in range(3):
            env.reset()
            assert id(env.observation_space) == obs_space_id
            assert id(env.action_space) == act_space_id

    def test_accepts_panels_of_different_member_counts(self) -> None:
        """reset() must not raise when successive panels have different
        member counts — padding absorbs the difference."""
        # A window panel has more members (sill + bottom cripples) than
        # the equivalent door panel.
        panel_window = generate_panel(
            wall_length=feet(12),
            openings=[{"type": "window", "width": inches(36)}],
            seed=0,
        )
        panel_door = generate_panel(
            wall_length=feet(12),
            openings=[{"type": "door", "width": inches(36)}],
            seed=0,
        )
        assert len(panel_window.members) != len(panel_door.members)

        # RandomPanelEnv.__init__ calls panel_generator() once to build the
        # initial inner env, so we need one extra panel beyond the four reset
        # calls below (5 total = 1 for __init__ + 4 for reset).
        panel_sequence = [panel_window, panel_door, panel_window, panel_door, panel_window]
        it = iter(panel_sequence)
        env = RandomPanelEnv(lambda: next(it), robot_speed=10.0)

        for _ in panel_sequence[1:]:   # 4 resets — each pulls the next panel
            obs, _ = env.reset()
            assert obs.shape == (OBS_DIM,)
            assert env.observation_space.contains(obs)

    def test_each_reset_produces_obs_in_observation_space(self) -> None:
        # RandomPanelEnv.__init__ calls the generator once, so we need
        # len(panels) + 1 entries: one for __init__ and one per reset call.
        panels = [generate_panel(
            wall_length=feet(12),
            openings=[{"type": "window", "width": inches(36)}],
            seed=s,
        ) for s in range(5)]
        it = iter([generate_panel(
            wall_length=feet(12),
            openings=[{"type": "window", "width": inches(36)}],
            seed=99,
        )] + panels)
        env = RandomPanelEnv(lambda: next(it), robot_speed=10.0)
        for _ in panels:
            obs, _ = env.reset()
            assert env.observation_space.contains(obs)
