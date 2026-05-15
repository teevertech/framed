# framed

> **I've Been Framed** — an RL-based wall panel assembly sequencer.

<!-- Logo placeholder: figure standing behind lumber framing as bars -->

## What this is

A reinforcement learning project that learns to sequence pick-and-place
operations for a robotic arm assembling wood-framed wall panels. Given a
panel specification (members + positions + precedence constraints), the
trained policy outputs an order that minimizes total assembly time while
respecting structural dependencies.

This is a portfolio project / industrial-robotics warmup, modeled on the
problem space at [Promise Robotics](https://www.promiserobotics.com).

## Status

In development. Phase 0 + Phase 1 (project scaffold + data model) is the
current checkpoint.

## Setup

```bash
uv sync
uv run pytest -v
```

## Project layout

```
src/framed/
├── units.py     - canonical internal length unit + conversion helpers
├── panel.py     - Member, Panel, MemberKind
├── geometry.py  - rectangles, collision tests, travel-time model
├── env.py       - Gym env
├── baselines.py - random_valid, plates-first heuristic
├── benchmark.py - same shape as Gridworld
├── render.py    - matplotlib stills + GIFs
├── training.py  - PPO + Aim
└── api/         - FastAPI app (later)

web/             - frontend (later)
tests/           - pytest suite
docs/            - design notes (forthcoming)
scripts/         - training & benchmark entry points (forthcoming)
```

## Units

All internal lengths are stored as floats in a single canonical unit
(currently inches). To switch to mm, edit `src/framed/units.py` and re-run
the test suite — no other module hardcodes a unit assumption.
