"""Gymnasium environment for the panel-sequencing problem.

``PanelEnv`` frames a single wall panel as an MDP: the agent picks which
framing member to place next, the robot moves to it (paying a detour
penalty if the straight-line path passes through any already-placed
member's interior), and the episode ends when every member has been
placed.

Action space
------------
``Discrete(n_members)``.  Action ``i`` means "place member ``i`` next."
Invalid actions — members that are already placed, or whose prerequisites
have not yet been placed — are surfaced via ``action_masks()`` for
MaskablePPO; if an invalid action is passed to ``step()`` anyway, it
raises ``ValueError`` (rather than silently penalising) so masking bugs
fail loudly.

Observation space
-----------------
Flat ``Box`` of shape ``(2 + 3*n_members,)`` with all values in ``[0, 1]``:

    [0:2]           normalized robot position (x, y)
    [2 : 2+n]       placement mask: 1.0 if member i is placed, else 0.0
    [2+n : 2+3n]    normalized member centers, flattened as
                    (c0.x, c0.y, c1.x, c1.y, ...)

Positions are normalized in-env by ``panel.wall_length`` / ``panel.wall_height``.
See README / design notes on why we picked baked-in normalization over
``VecNormalize``.

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

After placing, the robot moves to
the placed member's center — this is what makes the problem an actual
sequencing problem rather than independent placement decisions (the cost
of the next move depends on the previous choice).

Termination
-----------
``terminated = True`` exactly when all members are placed.  Because every
action places exactly one member, every episode is exactly ``n_members``
steps long.  ``truncated`` is never True; there is no time cap.

Episode randomization
---------------------
A ``PanelEnv`` instance is bound to one fixed ``Panel``.  To train across
many random panels, recreate the env (or use a Gym wrapper that does so)
between episodes.  This keeps the observation space's shape constant —
different panels can have different member counts, which would otherwise
break Gym's "fixed observation space" contract.
"""
from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from framed.geometry import path_collides, travel_time
from framed.panel import Panel

Position = tuple[float, float]


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
            The wall panel to assemble.  Held by reference; not mutated.
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

        self.panel = panel
        self.n_members = len(panel.members)
        self.robot_speed = robot_speed
        self.collision_penalty_multiplier = collision_penalty_multiplier
        self.initial_robot_pos = initial_robot_pos

        # Per-member centers cached as (n_members, 2) for fast obs building.
        self._member_centers = np.array(
            [m.center for m in panel.members], dtype=np.float32
        )

        # Normalization divisor (broadcasts against (x, y) pairs).
        self._norm_xy = np.array(
            [panel.wall_length, panel.wall_height], dtype=np.float32
        )

        # Precompute prereq indices (not ids) for O(1) per-prereq masking.
        id_to_index = {m.id: i for i, m in enumerate(panel.members)}
        self._prereq_indices: list[np.ndarray] = [
            np.array([id_to_index[p] for p in m.prerequisites], dtype=np.intp)
            for m in panel.members
        ]

        # Spaces.
        self.action_space = spaces.Discrete(self.n_members)
        obs_dim = 2 + 3 * self.n_members
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
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
            If *action* is out of range, refers to an already-placed member,
            or has unmet prerequisites.  Callers using MaskablePPO should
            never hit these — they indicate a masking bug.
        """
        action = int(action)
        if not 0 <= action < self.n_members:
            raise ValueError(
                f"Action {action} out of range [0, {self.n_members})"
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

        # Compute travel cost from current robot position to the member's center.
        target: Position = member.center
        robot_pos: Position = (float(self._robot_pos[0]), float(self._robot_pos[1]))
        t = travel_time(robot_pos, target, speed=self.robot_speed)

        # Check whether the straight-line path collides with anything
        # already on the table.  The most recently placed member is excluded:
        # the robot is "lifting off" from it, not crashing through it.
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
        """Return a boolean array of length ``n_members``.

        ``mask[i]`` is True iff member ``i`` is a valid action right now:
        not yet placed, and all its prerequisites are placed.  Returned as
        a fresh array each call (no aliasing of internal state).
        """
        mask = np.zeros(self.n_members, dtype=bool)
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
        """Build the flat observation vector (see module docstring for layout)."""
        robot_norm = self._robot_pos / self._norm_xy
        placed_f = self._placed.astype(np.float32)
        centers_norm = (self._member_centers / self._norm_xy).reshape(-1)
        return np.concatenate([robot_norm, placed_f, centers_norm]).astype(np.float32)

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
    exposes the agent to a different layout.  Because the Gymnasium
    contract requires a fixed observation space across all episodes, the
    generator **must always return panels with the same number of members**.
    A mismatch raises ``ValueError`` on the first offending reset — see
    ``framed.config.TrainConfig.make_panel_generator`` for a generator that
    guarantees this by fixing wall length, opening type, and opening width.

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

        # Build the initial inner env to establish fixed spaces.
        self._env = PanelEnv(
            panel_generator(),
            robot_speed=robot_speed,
            collision_penalty_multiplier=collision_penalty_multiplier,
            initial_robot_pos=initial_robot_pos,
        )
        self._n_members: int = len(self._env.panel.members)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    # ------------------------------------------------------------------ #
    # Gym API                                                              #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Generate a new panel and reset the inner env."""
        new_panel = self._panel_generator()
        if len(new_panel.members) != self._n_members:
            raise ValueError(
                f"Panel generator returned {len(new_panel.members)} members "
                f"but the initial panel had {self._n_members}. "
                f"Fix generator parameters (wall_length, opening_type, "
                f"opening_width) so member count stays constant."
            )
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
        return self._n_members
