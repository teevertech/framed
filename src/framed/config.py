"""Training configuration for the panel-sequencing RL agent.

``TrainConfig`` is a frozen dataclass — every field has a typed default
so a full training run needs just::

    from framed.config import TrainConfig
    config = TrainConfig()

Override individual fields with ``dataclasses.replace``::

    tuned = dataclasses.replace(config, learning_rate=1e-4, n_envs=4)

Or build from the CLI key=value format used by ``scripts/train.py``::

    config = TrainConfig.from_overrides({"learning_rate": "1e-4", "n_envs": "4"})

Panel generation and the fixed-topology constraint
---------------------------------------------------
Different panels can have different numbers of members.  Because
``PanelEnv``'s observation space shape is ``2 + 3*n_members``, Gymnasium
requires that every episode uses a panel with the **same** member count.

``make_panel_generator`` achieves this by fixing ``wall_length``,
``opening_type``, and ``opening_width`` (which jointly determine
member count) while randomising only ``opening_center_x`` (which shifts
the opening left or right without adding or removing any members).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Callable, Literal

import numpy as np

from framed.panel import LUMBER_THICKNESS, Panel, generate_random_panel
from framed.units import feet, inches


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # ------------------------------------------------------------------ #
    # Panel generation                                                     #
    # ------------------------------------------------------------------ #

    wall_length: float = feet(12)
    """Wall length in canonical units (inches).
    Fixed across all episodes to keep ``n_members`` — and therefore the
    observation space shape — constant."""

    opening_type: Literal["window", "door"] = "window"
    """Opening type.  Mixing types in one training run would change member
    count (windows have a sill; doors do not)."""

    opening_width: float = inches(36)
    """Rough-opening width in canonical units.  Fixed for the same reason."""

    # ------------------------------------------------------------------ #
    # Environment                                                          #
    # ------------------------------------------------------------------ #

    robot_speed: float = 10.0
    """End-effector speed in canonical units per time unit (in/s)."""

    collision_penalty_multiplier: float = 2.0
    """Detour-cost multiplier when the path collides with a placed member.
    0 disables the penalty; 2.0 means colliding moves cost 3× bare time.
    Primary tuning target for hyperparameter sweeps."""

    # ------------------------------------------------------------------ #
    # MaskablePPO hyperparameters                                          #
    # ------------------------------------------------------------------ #

    total_timesteps: int = 500_000
    n_envs: int = 8
    """Number of parallel rollout workers (SubprocVecEnv)."""

    n_steps: int = 1024
    """Rollout buffer size per worker.  Total rollout = n_steps * n_envs."""

    batch_size: int = 256
    """Mini-batch size for PPO gradient updates."""

    n_epochs: int = 10
    """Number of passes over the rollout buffer per policy update."""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    net_arch: tuple[int, ...] = (256, 256)
    """Hidden layer sizes for both actor and critic MLPs."""

    # ------------------------------------------------------------------ #
    # Logging and checkpointing                                            #
    # ------------------------------------------------------------------ #

    experiment_name: str = "panel_sequencing"
    run_name: str = ""
    """Human-readable label for this run.  Auto-generated from the config
    hash if left empty."""

    aim_repo: str = ".aim"
    """Path to the aim repository (created on first use)."""

    checkpoint_dir: str = "checkpoints"
    """Directory for model checkpoints."""

    checkpoint_freq: int = 50_000
    """Save a checkpoint every N timesteps."""

    eval_freq: int = 10_000
    """Run greedy baseline + policy evaluation every N timesteps."""

    n_eval_panels: int = 5
    """Number of fixed panels used for evaluation (baseline + policy)."""

    gif_dir: str = "gifs"
    """Directory for per-eval GIF snapshots.  Set to '' to disable."""

    gif_fps: int = 20
    """Frame rate for saved GIFs."""

    # ------------------------------------------------------------------ #
    # Reproducibility                                                      #
    # ------------------------------------------------------------------ #

    seed: int = 42

    # ------------------------------------------------------------------ #
    # Derived helpers                                                      #
    # ------------------------------------------------------------------ #

    @property
    def n_members(self) -> int:
        """Member count for panels produced by this config.

        Computed from one sample panel.  Inexpensive, but avoid calling
        in a tight loop.  Useful for validating generators and sizing
        the observation space before training starts.
        """
        sample = generate_random_panel(
            wall_length=self.wall_length,
            opening_type=self.opening_type,  # type: ignore[arg-type]
            opening_width=self.opening_width,
            seed=0,
        )
        return len(sample.members)

    def effective_run_name(self) -> str:
        """Return ``run_name`` if set, otherwise ``{experiment}_{hash[:8]}``."""
        if self.run_name:
            return self.run_name
        return f"{self.experiment_name}_{self.run_id()[:8]}"

    def run_id(self) -> str:
        """8-hex-char SHA-1 digest of the training hyperparameters.

        Book-keeping fields (names, paths, seed) are excluded so that two
        runs that differ only in ``run_name`` share the same id.
        """
        exclude = {
            "experiment_name", "run_name", "aim_repo",
            "checkpoint_dir", "seed",
        }
        d = {k: v for k, v in dataclasses.asdict(self).items()
             if k not in exclude}
        payload = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha1(payload.encode()).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        """Plain dict of all fields (for aim hyperparameter logging)."""
        return dataclasses.asdict(self)

    def make_panel_generator(self, seed: int) -> Callable[[], Panel]:
        """Return a callable that yields a new panel on each call.

        ``opening_center_x`` is randomised; all other parameters are fixed
        so the observation space shape stays constant.  Pass a different
        ``seed`` to each parallel worker to avoid correlated rollouts.

        Why a retry loop
        ----------------
        Studs are placed on a regular grid (every ``DEFAULT_STUD_SPACING``
        inches).  As the opening shifts left/right it can swallow or reveal
        a stud grid position, changing ``n_members`` by ±1.  Rather than
        trying to analytically enumerate safe positions, we sample randomly
        and accept only panels whose member count matches a target derived
        from the wall centre — the one position guaranteed to succeed.
        In practice the loop resolves in 1–3 attempts.

        Parameters
        ----------
        seed:
            Seed for the internal NumPy RNG.  Should be distinct per worker.
        """
        rng = np.random.default_rng(seed)
        wall_length = self.wall_length
        opening_type = self.opening_type
        opening_width = self.opening_width
        margin = opening_width / 2.0 + 4.0 * LUMBER_THICKNESS
        min_cx = margin
        max_cx = wall_length - margin
        centre_cx = (min_cx + max_cx) / 2.0

        # Determine the target member count from the centre position.
        # This is the reference: all generated panels must match it.
        _reference = generate_random_panel(
            wall_length=wall_length,
            opening_type=opening_type,      # type: ignore[arg-type]
            opening_width=opening_width,
            opening_center_x=centre_cx,
            seed=0,
        )
        target_n: int = len(_reference.members)

        def _generate() -> Panel:
            for _ in range(200):
                cx = float(rng.uniform(min_cx, max_cx))
                ep_seed = int(rng.integers(0, 2 ** 31))
                panel = generate_random_panel(
                    wall_length=wall_length,
                    opening_type=opening_type,  # type: ignore[arg-type]
                    opening_width=opening_width,
                    opening_center_x=cx,
                    seed=ep_seed,
                )
                if len(panel.members) == target_n:
                    return panel
            # Fallback: centre position always produces target_n members.
            return generate_random_panel(
                wall_length=wall_length,
                opening_type=opening_type,  # type: ignore[arg-type]
                opening_width=opening_width,
                opening_center_x=centre_cx,
                seed=int(rng.integers(0, 2 ** 31)),
            )

        return _generate

    @classmethod
    def from_overrides(cls, overrides: dict[str, Any]) -> TrainConfig:
        """Construct a config from a dict of string or typed overrides.

        Values that are still strings are coerced to the appropriate
        Python type (int, float, bool, or a comma-separated tuple of ints
        for ``net_arch``).  Unknown keys raise ``TypeError`` from the
        dataclass constructor.

        Used by ``scripts/train.py`` to parse CLI key=value arguments.
        """
        coerced: dict[str, Any] = {}
        for key, val in overrides.items():
            if isinstance(val, str):
                val = _coerce(val)
            # net_arch may arrive as a list from json round-trips.
            if key == "net_arch" and isinstance(val, list):
                val = tuple(val)
            coerced[key] = val
        return cls(**coerced)


# ------------------------------------------------------------------ #
# Internal helpers                                                     #
# ------------------------------------------------------------------ #

def _coerce(s: str) -> Any:
    """Best-effort parse of a CLI string value into a Python scalar.

    Tries: bool → int → float → tuple-of-ints (comma-sep) → str.
    """
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if "," in s:
        try:
            return tuple(int(x.strip()) for x in s.split(","))
        except ValueError:
            pass
    return s
