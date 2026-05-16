# framed

**I've Been Framed** — an RL-based wall panel assembly sequencer.

## What this is

A reinforcement learning project that learns to sequence pick-and-place
operations for a robotic arm assembling wood-framed wall panels. Given a
panel specification (members, positions, and precedence constraints), the
trained policy outputs a placement order that minimizes total assembly time —
accounting for both robot travel distance and collision detour costs — while
respecting structural dependencies.

This is a portfolio project modeled on the problem space at
[Promise Robotics](https://www.promiserobotics.com), an Edmonton-based
startup building robotic manufacturing systems for prefabricated wall frames.

## Results

Trained with MaskablePPO (300k steps, 8 parallel workers, ~10 min on M2 Pro),
the policy **beats the greedy nearest-neighbour baseline on 20/20 unseen
panels** with a mean improvement of **+16.7%** and up to +28.8% on
favorable layouts. The agent learns to sequence placements so the robot
rarely has to detour around its own work — a behavior the myopic baselines
cannot discover.

```
Same Topology (training distribution) — 20 panels, k=4.0

  Policy beats nearest:  20/20
  Mean improvement:      +16.7%
  Min improvement:       +3.4%
  Max improvement:       +28.8%
```

## Quick start

```bash
uv sync
uv run pytest -v          # run the test suite
```

### Train

```bash
# Smoke test (~10s, verifies the pipeline end-to-end)
uv run python scripts/sweep.py smoke

# Portfolio sweep (4 runs × 300k steps, ~15–20 min)
uv run python scripts/sweep.py portfolio

# Single run with custom hyperparameters
uv run python scripts/train.py collision_penalty_multiplier=4.0 total_timesteps=500000
```

### Evaluate

```bash
# Test on 50 unseen panels (same topology as training)
uv run python scripts/evaluate.py \
    --model checkpoints/portfolio_k4.0/final_model.zip \
    --n-panels 50

# Cross-topology generalization (different wall lengths, doors, etc.)
uv run python scripts/evaluate.py \
    --model checkpoints/portfolio_k4.0/final_model.zip \
    --cross-topology \
    --save-gifs eval_gifs/
```

### Visualize

```bash
# Interactive matplotlib animation
uv run python scripts/visualize_episode.py

# Side-by-side greedy vs greedy comparison GIF
uv run python scripts/visualize_episode.py --compare --save comparison.gif

# Trained model vs greedy baseline
uv run python scripts/visualize_episode.py \
    --policy model \
    --model checkpoints/portfolio_k4.0/final_model.zip \
    --save trained.gif
```

## Project layout

```
src/framed/
├── units.py        canonical internal length unit + conversion helpers
├── panel.py        Member, Panel, MemberKind, generate_random_panel
├── geometry.py     Rectangle, travel_time, segment_intersects_rect, path_collides
├── env.py          PanelEnv (Gymnasium) + RandomPanelEnv wrapper
├── baselines.py    greedy_nearest, greedy_cost_aware, run_episode
├── config.py       TrainConfig dataclass + panel generator with retry loop
├── callbacks.py    Aim logging + eval callback with GIF snapshots
└── visualize.py    matplotlib animation (robot triangle, lumber colors, path lines)

scripts/
├── train.py              single-run training entry point (key=value CLI)
├── sweep.py              hyperparameter sweep launcher (portfolio, lr_vs_penalty, etc.)
├── evaluate.py           same- and cross-topology generalization testing
└── visualize_episode.py  interactive or GIF rendering of episodes

tests/
├── test_panel.py         panel validation, generator invariants
├── test_geometry.py      rectangle, collision, travel-time edge cases
├── test_env.py           step mechanics, masking, liftoff rule, determinism
└── test_baselines.py     greedy correctness, baseline comparison
```

## How it works

**The problem.** A wall panel is a set of framing members (studs, plates,
headers, cripples) with positions on a 2D table and structural precedence
constraints (bottom plate before studs, jacks before header, etc.). A robot
arm must place every member exactly once. The cost of each move is the
straight-line travel time from the robot's current position to the next
member's center, plus a detour penalty if the path crosses any already-placed
member's footprint. The goal is to find the placement order that minimizes
total cost.

**The environment.** `PanelEnv` is a Gymnasium env with a `Discrete(n_members)`
action space. Invalid actions (already placed, or prerequisites not met) are
exposed via `action_masks()` for MaskablePPO. Observations are a flat vector
of normalized robot position, per-member placement flags, and member centers.
The "liftoff rule" excludes the most recently placed member from collision
checks — the robot lifts off what it just placed, it doesn't crash through it.

**The baselines.** Two greedy heuristics provide reference scores. Greedy
nearest picks the closest valid member by Euclidean distance. Greedy
cost-aware picks the member that minimizes the immediate step cost including
the collision penalty. Neither dominates the other — nearest stays close to
the work but eats collision penalties; cost-aware avoids penalties but can
strand the robot far from the remaining cluster.

**The agent.** MaskablePPO with a 256×256 MLP policy, trained across random
panel layouts (opening position varies per episode). The agent learns to
sequence placements so the robot rarely needs to detour around its own work —
a planning-ahead behavior that myopic greedy baselines cannot discover.

## Units

All internal lengths are stored as floats in a single canonical unit
(currently inches). To switch to mm, edit `src/framed/units.py` and re-run
the test suite — no other module hardcodes a unit assumption.

## Known limitations

**Fixed panel size.** The MLP policy requires a fixed observation dimension,
so all training and evaluation panels must have the same member count. Panels
with different wall lengths, opening types, or opening widths that produce
different member counts cannot be evaluated without retraining.

**2D collision model.** The robot is modeled as a point moving in the plane
of the panel. In reality, a robot arm operates in 3D and can lift over
obstacles. The collision penalty is a proxy for the real 3D detour cost
(higher lifts, wider arcs, slower speeds when maneuvering around placed
members).

**Single opening.** The panel generator produces walls with exactly one
window or door opening. Multi-opening walls, solid walls, and irregular
layouts are not yet supported.

## Future directions

**Attention-based policy.** Replacing the fixed-size MLP with a
transformer encoder-decoder (in the style of Kool et al., "Attention,
Learn to Solve Routing Problems") would allow a single policy to handle
variable panel sizes. Each member becomes a token with its own feature
vector; self-attention captures spatial relationships regardless of set
size. The env, baselines, and evaluation harness are architecture-agnostic
and would not change.

**3D cost model.** Replacing the 2D collision penalty with queries to a
real robot path planner would make the reward signal physically accurate.
The RL formulation stays the same — only the cost function inside `step()`
changes.
