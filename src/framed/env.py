"""Gymnasium environment for the panel-sequencing problem.

``PanelEnv`` frames a single wall panel as an MDP: the agent picks which
framing member to place next, the robot moves to it (paying a detour
penalty if the straight-line path passes through any already-placed
member's interior), and the episode ends when every member has been
placed.

Action space
------------
``Discrete(MAX_MEMBERS)``.  Action ``i`` means "place member ``i`` next."
Slots ``>= n_members`` are permanently masked False and will raise
``ValueError`` if passed to ``step()`` anyway (they indicate a masking
bug, not a valid no-op).

Observation space
-----------------
Flat ``Box`` of shape ``(652,)`` — i.e. ``(2 + 13 * MAX_MEMBERS,)`` —
with all values in ``[0, 1]``.  Layout:

    [0:2]                       normalised robot position (x/L, y/H)
    [2 : 2+MAX]                 placed flags — 1.0 if placed, 0.0 otherwise
                                (0.0 for padding slots)
    [2+MAX : 2+3*MAX]           normalised member centers, flattened as
                                (c0.x, c0.y, c1.x, c1.y, …)
                                (0.0 for padding slots)
    [2+3*MAX : 2+12*MAX]        member kind one-hot (9 classes × MAX_MEMBERS)
                                (all zeros for padding slots)
    [2+12*MAX : 2+13*MAX]       prereq-satisfied flag — 1.0 if all prereqs
                                of member i are placed (or it has none),
                                0.0 otherwise
                                (0.0 for padding slots)

``MAX_MEMBERS = 50``, so the obs dim is ``2 + 13 * 50 = 652``.

The 9 ``MemberKind`` classes in one-hot order (alphabetical by enum value):

    0  bottom_cripple
    1  bottom_plate
    2  common_stud
    3  header
    4  jack_stud
    5  king_stud
    6  sill_plate
    7  top_cripple
    8  top_plate

Positions are normalised in-env by ``panel.wall_length`` /
``panel.wall_height``.  See README / design notes on why we picked
baked-in normalisation over ``VecNormalize``.

Reward
------
At each step::

    reward = -travel_time(robot_pos, member.center, speed) *
             (1 + collision_penalty_multiplier * collided)

A non-colliding move costs the bare straight-line time; a colliding move
adds a detour-cost multiplier on top.  The most recently placed member is
excluded from the collision check (the "liftoff" rule): the robot lifts
off the piece it just released, it doesn't crash through it.  Without this
rule, every step after the first would register a collision (the robot's
position is always inside the last-placed member), making the penalty a
uniform per-step tax with no discriminating power for sequencing.

After placing, the robot moves to the placed member's center — this is
what makes the problem an actual sequencing problem rather than
independent placement decisions (the cost of the next move depends on
the previous choice).

Termination
-----------
``terminated = True`` exactly when all members are placed.  Because every
action places exactly one member, every episode is exactly ``n_members``
steps long.  ``truncated`` is never True; there is no time cap.

Episode randomisation
---------------------
A ``PanelEnv`` instance is bound to one fixed ``Panel``.  To train across
many random panels, recreate the env (or use ``RandomPanelEnv``) between
episodes.  The padded observation/action space means panels with different
member counts all share the same fixed Gymnasium spaces, so
``RandomPanelEnv`` no longer requires a constant member count.
"""
from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from framed.geometry import path_collides, travel_time
from framed.panel import MemberKind, Panel

Position = tuple[float, float]

# ------------------------------------------------------------------ #
# Constants                                                            #
# ------------------------------------------------------------------ #

MAX_MEMBERS: int = 50
"""Maximum number of members a panel can have.

All obs/action spaces are sized to this constant so panels with
different member counts share the same fixed Gymnasium spaces.
Panels with more members raise ``ValueError`` at env construction time.
"""

MEMBER_KIND_INDEX: dict[MemberKind, int] = {
    MemberKind.BOTTOM_CRIPPLE: 0,
    MemberKind.BOTTOM_PLATE:   1,
    MemberKind.COMMON_STUD:    2,
    MemberKind.HEADER:         3,
    MemberKind.JACK_STUD:      4,
    MemberKind.KING_STUD:      5,
    MemberKind.SILL_PLATE:     6,
    MemberKind.TOP_CRIPPLE:    7,
    MemberKind.TOP_PLATE:      8,
}
"""One-hot index for each ``MemberKind``.  Alphabetical by enum value."""

N_MEMBER_KINDS: int = 9

# Precomputed obs dimension for reference / ONNX export.
OBS_DIM: int = 2 + 13 * MAX_MEMBERS   # = 652


class PanelEnv(gym.Env):
    """Gymnasium env for panel-member placement sequencing.

    See the module docstring for the full spec.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        panel: Panel,
        *,
        robot_speed: float = 10.0,
        collision_penalty_multiplier: float = 2.0,
        initial_robot_pos: Position = (0.0, 0.0),
    ) -> None:
        """Build an env bound to *panel*.

        Parameters
        ----------
        panel:
            The wall panel to assemble.  Must have ``len(members) <=
            MAX_MEMBERS``.  Held by reference; not mutated.
        robot_speed:
            End-effector speed in canonical units per time unit.  Must be > 0.
            Tunable hyperparameter (default 10.0 in/s).
        collision_penalty_multiplier:
            Multiplicative detour cost when a step's path collides.  0 disables
            the penalty entirely; 2.0 means colliding moves cost 3× the bare
            travel time.  Must be ≥ 0.  Tunable hyperparameter.
        initial_robot_pos:
            Where the robot sits at episode start.  Defaults to (0, 0)
            (bottom-left corner of the wall — a reasonable "home" position).
        """
        super().__init__()
        if robot_speed <= 0.0:
            raise ValueError(f"robot_speed must be positive, got {robot_speed!r}")
        if collision_penalty_multiplier < 0.0:
            raise ValueError(
                f"collision_penalty_multiplier must be >= 0, "
                f"got {collision_penalty_multiplier!r}"
            )
        if len(panel.members) > MAX_MEMBERS:
            raise ValueError(
                f"Panel has {len(panel.members)} members but MAX_MEMBERS={MAX_MEMBERS}. "
                f"Increase MAX_MEMBERS or reduce the panel size."
            )

        self.panel = panel
        self.n_members = len(panel.members)
        self.robot_speed = robot_speed
        self.collision_penalty_multiplier = collision_penalty_multiplier
        self.initial_robot_pos = initial_robot_pos

        # Per-member centers cached as (n_members, 2) for fast obs building.
        self._member_centers = np.array(
            [m.center for m in panel.members], dtype=np.float32
        )

        # Normalisation divisor: (wall_length, wall_height) broadcasts against
        # any (x, y) pair.
        self._norm_xy = np.array(
            [panel.wall_length, panel.wall_height], dtype=np.float32
        )

        # Precompute prereq indices (not ids) for O(1) per-prereq masking.
        id_to_index = {m.id: i for i, m in enumerate(panel.members)}
        self._prereq_indices: list[np.ndarray] = [
            np.array([id_to_index[p] for p in m.prerequisites], dtype=np.intp)
            for m in panel.members
        ]

        # Precompute kind one-hots: shape (n_members, N_MEMBER_KINDS).
        # Only real member slots are populated; padding slots stay zero.
        self._kind_onehots = np.zeros(
            (self.n_members, N_MEMBER_KINDS), dtype=np.float32
        )
        for i, m in enumerate(panel.members):
            self._kind_onehots[i, MEMBER_KIND_INDEX[m.kind]] = 1.0

        # Fixed spaces — sized to MAX_MEMBERS so all panels share the same
        # Gymnasium contract regardless of their actual member count.
        self.action_space = spaces.Discrete(MAX_MEMBERS)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

        # Runtime state populated by reset().
        self._placed: np.ndarray = np.zeros(self.n_members, dtype=bool)
        self._robot_pos: np.ndarray = np.array(initial_robot_pos, dtype=np.float32)
        self._step_count: int = 0
        self._last_placed_idx: int | None = None

    # ------------------------------------------------------------------ #
    # Gym API                                                              #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to an empty panel with the robot at ``initial_robot_pos``.

        The env is deterministic given the fixed panel, so *seed* is accepted
        for API compliance but does not affect dynamics.  *options* is unused.
        """
        super().reset(seed=seed)
        self._placed = np.zeros(self.n_members, dtype=bool)
        self._robot_pos = np.array(self.initial_robot_pos, dtype=np.float32)
        self._step_count = 0
        self._last_placed_idx = None
        return self._get_obs(), self._get_info()

    def step(
        self, action: int | np.integer
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Place member *action*.

        Raises
        ------
        ValueError
            If *action* is out of range for *this panel's* real member count
            (not MAX_MEMBERS), refers to an already-placed member, or has
            unmet prerequisites.  Callers using MaskablePPO should never hit
            these — they indicate a masking bug.
        """
        action = int(action)
        # Validate against n_members, not MAX_MEMBERS — padding slots are
        # never legal actions even though the action space includes them.
        if not 0 <= action < self.n_members:
            raise ValueError(
                f"Action {action} out of range [0, {self.n_members}) "
                f"(panel has {self.n_members} members; "
                f"slots [{self.n_members}, {MAX_MEMBERS}) are padding)."
            )

        member = self.panel.members[action]

        if self._placed[action]:
            raise ValueError(
                f"Member {member.id!r} (index {action}) is already placed. "
                f"Check action_masks() before stepping."
            )
        prereqs = self._prereq_indices[action]
        if prereqs.size > 0 and not self._placed[prereqs].all():
            unmet = [
                self.panel.members[i].id for i in prereqs if not self._placed[i]
            ]
            raise ValueError(
                f"Member {member.id!r} has unmet prerequisites: {unmet}. "
                f"Check action_masks() before stepping."
            )

        # Travel and collision
        robot_pos = self.robot_pos
        target = member.center
        t = travel_time(robot_pos, target, speed=self.robot_speed)

        # Liftoff rule: exclude the last-placed member from obstacle set.
        placed_obstacles = [
            m
            for i, (m, p) in enumerate(zip(self.panel.members, self._placed))
            if p and i != self._last_placed_idx
        ]
        collided = path_collides(robot_pos, target, placed_obstacles)

        # Detour penalty is additive: total cost = t * (1 + k) when colliding.
        reward = -t * (1.0 + self.collision_penalty_multiplier * float(collided))

        # Update state.
        self._placed[action] = True
        self._robot_pos = np.array(target, dtype=np.float32)
        self._step_count += 1
        self._last_placed_idx = action

        terminated = bool(self._placed.all())
        truncated = False

        info = self._get_info()
        info.update(
            travel_time=float(t),
            collided=bool(collided),
            member_id=member.id,
            member_index=action,
        )

        return self._get_obs(), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Action masking (consumed by MaskablePPO)                             #
    # ------------------------------------------------------------------ #

    def action_masks(self) -> np.ndarray:
        """Return a boolean array of length ``MAX_MEMBERS``.

        ``mask[i]`` is True iff member ``i`` is a valid action right now:
        not yet placed, and all its prerequisites are placed.

        Slots ``>= n_members`` are permanently False (padding).  Returned
        as a fresh array each call (no aliasing of internal state).
        """
        mask = np.zeros(MAX_MEMBERS, dtype=bool)
        for i in range(self.n_members):
            if self._placed[i]:
                continue
            prereqs = self._prereq_indices[i]
            if prereqs.size == 0 or self._placed[prereqs].all():
                mask[i] = True
        return mask

    # ------------------------------------------------------------------ #
    # Public state accessors                                               #
    # ------------------------------------------------------------------ #

    @property
    def robot_pos(self) -> Position:
        """Current robot position as an ``(x, y)`` tuple in canonical units."""
        return (float(self._robot_pos[0]), float(self._robot_pos[1]))

    @property
    def placed_mask(self) -> np.ndarray:
        """Boolean array of length ``n_members``.  Returns a copy so callers
        cannot mutate internal state."""
        return self._placed.copy()

    @property
    def last_placed_idx(self) -> int | None:
        """Index of the most recently placed member, or ``None`` if the
        episode has just been reset.  The env excludes this member from
        collision checks (the "liftoff" rule)."""
        return self._last_placed_idx

    @property
    def obs(self) -> np.ndarray:
        """Current observation vector — same array as returned by ``reset()``
        and ``step()``.  Useful for model policies that need to call
        ``model.predict(env.obs, action_masks=env.action_masks())``."""
        return self._get_obs()

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_obs(self) -> np.ndarray:
        """Build the 652-element flat observation vector.

        Layout (all sections zero-padded to MAX_MEMBERS slots):

            [0:2]               normalised robot position
            [2 : 2+MAX]         placed flags (real members only)
            [2+MAX : 2+3*MAX]   normalised centers (real members only, x then y)
            [2+3*MAX : 2+12*MAX] kind one-hots (real members only, 9 classes each)
            [2+12*MAX : 2+13*MAX] prereq-satisfied flags (real members only)
        """
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        # Robot position (always written, even for empty panels)
        obs[0:2] = self._robot_pos / self._norm_xy

        # Placed flags — real member slots only; padding stays 0.
        obs[2 : 2 + self.n_members] = self._placed.astype(np.float32)

        # Normalised centers — two floats per real member slot.
        base_centers = 2 + MAX_MEMBERS
        centers_norm = self._member_centers / self._norm_xy   # (n_members, 2)
        obs[base_centers : base_centers + self.n_members * 2] = centers_norm.reshape(-1)

        # Kind one-hots — 9 floats per real member slot.
        base_kinds = 2 + 3 * MAX_MEMBERS
        obs[base_kinds : base_kinds + self.n_members * N_MEMBER_KINDS] = (
            self._kind_onehots.reshape(-1)
        )

        # Prereq-satisfied flags — one float per real member slot.
        base_prereq = 2 + 12 * MAX_MEMBERS
        for i in range(self.n_members):
            prereqs = self._prereq_indices[i]
            if prereqs.size == 0 or self._placed[prereqs].all():
                obs[base_prereq + i] = 1.0

        return obs

    def _get_info(self) -> dict[str, Any]:
        """Common info fields shared by reset() and step() returns."""
        return {
            "n_placed": int(self._placed.sum()),
            "step_count": self._step_count,
            "robot_pos": (
                float(self._robot_pos[0]),
                float(self._robot_pos[1]),
            ),
        }


class RandomPanelEnv(gym.Env):
    """``PanelEnv`` variant that generates a fresh panel on every ``reset()``.

    Wraps ``PanelEnv`` with a panel generator callable so each episode
    exposes the agent to a different layout.  The padded observation/action
    space (fixed at ``MAX_MEMBERS``) means panels with different member
    counts are fully compatible — no member-count equality check is needed.

    All public properties and methods delegate directly to the inner
    ``PanelEnv``, so this class is a drop-in replacement wherever
    ``PanelEnv`` is expected.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        panel_generator: Callable[[], Panel],
        *,
        robot_speed: float = 10.0,
        collision_penalty_multiplier: float = 2.0,
        initial_robot_pos: Position = (0.0, 0.0),
    ) -> None:
        super().__init__()
        self._panel_generator = panel_generator
        self._robot_speed = robot_speed
        self._collision_penalty_multiplier = collision_penalty_multiplier
        self._initial_robot_pos = initial_robot_pos

        # Fixed spaces — derived from MAX_MEMBERS, not the initial panel,
        # so they remain valid across resets with different panel sizes.
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(MAX_MEMBERS)

        # Build the initial inner env so all properties are available
        # immediately after construction.
        self._env = PanelEnv(
            panel_generator(),
            robot_speed=robot_speed,
            collision_penalty_multiplier=collision_penalty_multiplier,
            initial_robot_pos=initial_robot_pos,
        )

    # ------------------------------------------------------------------ #
    # Gym API                                                              #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Generate a new panel and reset the inner env.

        No member-count check is performed — the padded observation/action
        space handles panels of any size up to ``MAX_MEMBERS``.
        """
        new_panel = self._panel_generator()
        self._env = PanelEnv(
            new_panel,
            robot_speed=self._robot_speed,
            collision_penalty_multiplier=self._collision_penalty_multiplier,
            initial_robot_pos=self._initial_robot_pos,
        )
        return self._env.reset(seed=seed, options=options)

    def step(
        self, action: int | np.integer
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return self._env.step(action)

    def action_masks(self) -> np.ndarray:
        return self._env.action_masks()

    # ------------------------------------------------------------------ #
    # State accessors (delegated)                                          #
    # ------------------------------------------------------------------ #

    @property
    def robot_pos(self) -> Position:
        return self._env.robot_pos

    @property
    def placed_mask(self) -> np.ndarray:
        return self._env.placed_mask

    @property
    def last_placed_idx(self) -> int | None:
        return self._env.last_placed_idx

    @property
    def obs(self) -> np.ndarray:
        return self._env.obs

    @property
    def panel(self) -> Panel:
        """The panel currently being assembled."""
        return self._env.panel

    @property
    def n_members(self) -> int:
        """Member count of the *current* panel (changes each reset)."""
        return self._env.n_members
