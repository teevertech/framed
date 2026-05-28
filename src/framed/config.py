"""Project paths, logging, and training configuration.

Paths
-----
All path constants are derived from ``PROJ_ROOT`` (the repository root,
two levels above this file).  Import them wherever a hardcoded path
string would otherwise appear::

    from framed.config import MODELS_DIR, GIFS_DIR, AIM_REPO

Logging
-------
Loguru is configured on import to write through ``tqdm.write`` so that
log messages don't clobber progress bars.  Every module that wants
structured output should import ``logger`` from here rather than using
``print`` or the stdlib ``logging`` module::

    from framed.config import logger

Training configuration
----------------------
``TrainConfig`` is a frozen dataclass — every field has a typed default
so a full training run needs just::

    from framed.config import TrainConfig
    config = TrainConfig()

Override individual fields with ``dataclasses.replace``::

    tuned = dataclasses.replace(config, learning_rate=1e-4, n_envs=4)

Or build from the CLI key=value format used by ``scripts/train.py``::

    config = TrainConfig.from_overrides({"learning_rate": "1e-4", "n_envs": "4"})

Panel generation and the multi-opening sampling distribution
------------------------------------------------------------
Each training episode draws a panel from a *distribution* of configurations
rather than a single fixed topology.  The distribution is defined by:

- ``wall_lengths`` — pool of wall lengths sampled uniformly each episode.
- ``max_openings`` — upper bound on the number of openings per panel (1 is
  always the minimum).
- ``opening_types`` — pool of opening types ("window" or "door").
- ``opening_widths_window`` / ``opening_widths_door`` — per-type width pools.

Because ``PanelEnv``'s observation and action spaces are padded to
``MAX_MEMBERS = 50`` (defined in ``framed.env``), panels of different member
counts can appear across resets without requiring a fixed topology.
``RandomPanelEnv`` no longer enforces a member-count match between episodes.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from dotenv import load_dotenv
from loguru import logger

from framed.env import MAX_MEMBERS
from framed.panel import Panel, generate_panel
from framed.units import feet, inches

# ------------------------------------------------------------------ #
# Environment                                                          #
# ------------------------------------------------------------------ #

load_dotenv()

# ------------------------------------------------------------------ #
# Paths                                                                #
# ------------------------------------------------------------------ #

PROJ_ROOT: Path = Path(__file__).resolve().parents[2]
"""Repository root — the directory that contains pyproject.toml."""

MODELS_DIR: Path = PROJ_ROOT / "models"
"""Root output directory for trained models and run artefacts.

Each training run writes to ``MODELS_DIR / run_name /``."""

GIFS_DIR: Path = PROJ_ROOT / "gifs"
"""Root output directory for per-eval GIF snapshots."""

AIM_REPO: Path = PROJ_ROOT / ".aim"
"""Path to the Aim experiment-tracking repository."""

# ------------------------------------------------------------------ #
# Logging                                                              #
# ------------------------------------------------------------------ #

# Remove the default loguru handler and replace it with one that writes
# through tqdm.write so that log messages don't clobber progress bars.
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm
    if logger._core.handlers:
        logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
except Exception:
    pass

logger.info(f"PROJ_ROOT: {PROJ_ROOT}")


# ------------------------------------------------------------------ #
# TrainConfig                                                          #
# ------------------------------------------------------------------ #

@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # ------------------------------------------------------------------ #
    # Panel generation — sampling distribution                            #
    # ------------------------------------------------------------------ #

    wall_lengths: tuple[float, ...] = (feet(8), feet(12), feet(16))
    """Pool of wall lengths (canonical units / inches) to sample from each
    episode.  All values must be ≤ ``feet(16)``."""

    max_openings: int = 2
    """Maximum number of openings per episode.  The actual count is sampled
    uniformly from 1..max_openings each episode."""

    opening_types: tuple[str, ...] = ("window", "door")
    """Pool of opening types to sample from.  Each opening in an episode
    independently draws from this pool."""

    opening_widths_window: tuple[float, ...] = (inches(24), inches(32), inches(36))
    """Pool of rough-opening widths (canonical units) for window openings."""

    opening_widths_door: tuple[float, ...] = (inches(32), inches(36))
    """Pool of rough-opening widths (canonical units) for door openings."""

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

    device: str = "auto"
    """PyTorch device for policy inference and training.  ``"auto"`` lets
    SB3 choose (usually ``"cpu"``).  Set to ``"mps"`` to use the Apple
    Metal GPU on Apple-Silicon Macs."""

    torch_threads: int = 0
    """Number of PyTorch inter-op / intra-op CPU threads.  ``0`` (default)
    leaves PyTorch's default (all cores).  ``1`` eliminates thread
    contention with SubprocVecEnv workers and often *increases* FPS for
    small MLPs."""

    # ------------------------------------------------------------------ #
    # Logging and checkpointing                                            #
    # ------------------------------------------------------------------ #

    experiment_name: str = "panel_sequencing"
    run_name: str = ""
    """Human-readable label for this run.  Auto-generated from the config
    hash if left empty."""

    checkpoint_freq: int = 50_000
    """Save a checkpoint every N timesteps."""

    eval_freq: int = 10_000
    """Run greedy baseline + policy evaluation every N timesteps."""

    n_eval_panels: int = 5
    """Number of fixed panels used for evaluation (baseline + policy)."""

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
        """Maximum padded member count used by the observation/action spaces.

        Returns ``MAX_MEMBERS`` (currently 50).  Individual panels produced
        by the sampling distribution may have fewer real members; the env
        pads the observation to this size automatically.
        """
        return MAX_MEMBERS

    def run_dir(self) -> Path:
        """Per-run output directory: ``MODELS_DIR / effective_run_name``."""
        return MODELS_DIR / self.effective_run_name()

    def ckpt_dir(self) -> Path:
        """Checkpoint subdirectory inside ``run_dir``."""
        return self.run_dir() / "checkpoints"

    def run_gif_dir(self) -> Path:
        """GIF subdirectory inside ``run_dir``."""
        return self.run_dir() / "gifs"

    def effective_run_name(self) -> str:
        """Return ``run_name`` if set, otherwise ``{experiment}_{hash[:8]}``."""
        if self.run_name:
            return self.run_name
        return f"{self.experiment_name}_{self.run_id()[:8]}"

    def run_id(self) -> str:
        """8-hex-char SHA-1 digest of the training hyperparameters.

        Book-keeping fields (names, paths, seed) and the sampling-distribution
        pool fields (``opening_types``, ``opening_widths_*``, ``wall_lengths``)
        are excluded.  ``max_openings`` is retained because it meaningfully
        changes the problem complexity.  Two runs that differ only in which
        widths are in the pool, or in their run name, share the same id.
        """
        exclude = {
            "experiment_name", "run_name", "seed",
            # Infrastructure fields — don't affect learning dynamics.
            "device", "torch_threads",
            # Distribution pool fields — describe the sampling range, not a
            # single fixed config.
            "opening_types", "opening_widths_window", "opening_widths_door",
            "wall_lengths",
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

        Each call samples a wall length, a number of openings, and a type and
        width for each opening from the configured pools, then delegates to
        ``generate_panel`` with a freshly drawn episode seed.

        Padding (``MAX_MEMBERS = 50``) means panels of different member counts
        are fine across resets — no member-count matching is needed here.

        The inner retry loop handles the rare case where randomly positioned
        openings don't fit on the wall (``generate_panel`` raises ``ValueError``
        in that situation).  After 50 failures it falls back to a single
        36-inch window on the shortest configured wall, which is guaranteed to
        succeed.

        Parameters
        ----------
        seed:
            Seed for the internal NumPy RNG.  Should be distinct per worker
            to avoid correlated rollouts.
        """
        rng = np.random.default_rng(seed)

        # Capture config fields for the closure (avoids repeated self lookups).
        wall_lengths = self.wall_lengths
        max_openings = self.max_openings
        opening_types = self.opening_types
        opening_widths_window = self.opening_widths_window
        opening_widths_door = self.opening_widths_door

        def _generate() -> Panel:
            wall_length = float(rng.choice(wall_lengths))
            n_openings = int(rng.integers(1, max_openings + 1))

            openings = []
            for _ in range(n_openings):
                kind = str(rng.choice(opening_types))
                if kind == "window":
                    width = float(rng.choice(opening_widths_window))
                else:
                    width = float(rng.choice(opening_widths_door))
                openings.append({"type": kind, "width": width})

            ep_seed = int(rng.integers(0, 2 ** 31))

            # Retry loop: handles placement failures (openings don't fit),
            # NOT member-count mismatches (padding renders those irrelevant).
            for _ in range(50):
                try:
                    return generate_panel(
                        wall_length=wall_length,
                        openings=openings,
                        seed=ep_seed,
                    )
                except ValueError:
                    ep_seed = int(rng.integers(0, 2 ** 31))

            # Fallback: guaranteed-to-succeed minimal panel.
            return generate_panel(
                wall_length=min(wall_lengths),
                openings=[{"type": "window", "width": inches(36)}],
                seed=ep_seed,
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
            # net_arch and tuple-of-floats fields may arrive as lists from
            # JSON round-trips.
            if key in ("net_arch", "opening_widths_window",
                        "opening_widths_door", "wall_lengths",
                        "opening_types") and isinstance(val, list):
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
