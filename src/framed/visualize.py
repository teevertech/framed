"""Matplotlib-based visualizer for panel assembly episodes.

Produces an animation showing the robot (filled triangle) traveling
between framing members, placed members filling in with lumber colors,
and path lines coloured green (clear) or red (collision detour).

Public API
----------
``animate_episode(env, policy, ...)``
    Run a full episode and return a ``FuncAnimation``.

``save_episode_gif(env, policy, output_path, ...)``
    Convenience wrapper that saves a GIF directly.

``render_panel_snapshot(panel, placed_ids, robot_pos, ax)``
    Draw a single static frame — useful for debugging.

Coordinate system
-----------------
Matches the env exactly: x = wall-length direction (left → right),
y = wall-height direction (bottom → top).  The matplotlib axes are
set up with equal aspect ratio and a small margin around the wall.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch, Polygon

from framed.env import PanelEnv
from framed.panel import Member, MemberKind, Panel

Position = tuple[float, float]

# ------------------------------------------------------------------ #
# Colour palette                                                       #
# ------------------------------------------------------------------ #

# Lumber colours — warm wood tones that read clearly at small sizes.
_KIND_COLOR: dict[MemberKind, str] = {
    MemberKind.BOTTOM_PLATE:   "#C4A265",   # tan
    MemberKind.TOP_PLATE:      "#C4A265",
    MemberKind.COMMON_STUD:    "#DEB887",   # burlywood
    MemberKind.KING_STUD:      "#CD853F",   # peru — slightly darker
    MemberKind.JACK_STUD:      "#D2691E",   # chocolate — orange-amber
    MemberKind.HEADER:         "#7B5E2A",   # dark engineered-lumber brown
    MemberKind.SILL_PLATE:     "#B8895A",   # medium wood
    MemberKind.TOP_CRIPPLE:    "#E8C99A",   # pale — short off-cut feel
    MemberKind.BOTTOM_CRIPPLE: "#E8C99A",
}

_WALL_BG      = "#F7F4EE"   # off-white wall background
_WALL_OUTLINE = "#BDBDBD"   # wall perimeter
_GHOST_FACE   = "#EDEDED"   # unplaced member fill
_GHOST_EDGE   = "#C0C0C0"   # unplaced member edge
_PATH_CLEAR   = "#43A047"   # green — no collision
_PATH_COLLIDE = "#E53935"   # red — detour required
_ROBOT_COLOR  = "#1565C0"   # deep blue triangle
_TARGET_GLOW  = "#FFD54F"   # amber highlight on the target member


# ------------------------------------------------------------------ #
# Frame data                                                           #
# ------------------------------------------------------------------ #

@dataclasses.dataclass
class _Frame:
    """All state needed to draw one animation frame."""
    robot_xy:      np.ndarray          # shape (2,)
    placed_ids:    frozenset[str]
    target_id:     str | None          # member currently being approached
    paths:         list[tuple[Position, Position, bool]]  # completed paths
    partial_path:  tuple[Position, Position] | None       # growing path
    collided_this: bool                # did the current move collide?
    step:          int
    total_reward:  float
    direction:     np.ndarray          # unit vector robot is facing, shape (2,)


# ------------------------------------------------------------------ #
# Episode collection                                                   #
# ------------------------------------------------------------------ #

def _collect_frames(
    env: PanelEnv,
    policy: Callable[[PanelEnv], int],
    frames_per_move: int,
    pause_frames: int,
) -> list[_Frame]:
    """Run a full episode and build a frame list for animation."""
    obs, _ = env.reset()

    frames: list[_Frame] = []
    completed_paths: list[tuple[Position, Position, bool]] = []
    robot_xy = np.array(env.robot_pos, dtype=float)
    direction = np.array([1.0, 0.0])   # start pointing right
    step = 0
    total_reward = 0.0
    placed_ids: frozenset[str] = frozenset()

    # Pause at episode start so the viewer can see the empty panel.
    for _ in range(pause_frames * 2):
        frames.append(_Frame(
            robot_xy=robot_xy.copy(),
            placed_ids=placed_ids,
            target_id=None,
            paths=list(completed_paths),
            partial_path=None,
            collided_this=False,
            step=0,
            total_reward=0.0,
            direction=direction.copy(),
        ))

    for _ in range(env.n_members):
        action = policy(env)
        obs, reward, terminated, _, info = env.step(action)
        total_reward += float(reward)
        step += 1

        member   = env.panel.members[action]
        from_xy  = robot_xy.copy()
        to_xy    = np.array(member.center, dtype=float)
        collided = bool(info["collided"])
        member_id = member.id

        # Direction of travel.
        delta = to_xy - from_xy
        dist = np.linalg.norm(delta)
        if dist > 1e-6:
            direction = delta / dist

        # Travel frames — robot moves, path line grows.
        for t in range(frames_per_move):
            alpha = t / frames_per_move
            current_xy = from_xy + alpha * (to_xy - from_xy)
            partial_end = (
                float(current_xy[0]),
                float(current_xy[1]),
            )
            frames.append(_Frame(
                robot_xy=current_xy.copy(),
                placed_ids=placed_ids,
                target_id=member_id,
                paths=list(completed_paths),
                partial_path=((float(from_xy[0]), float(from_xy[1])), partial_end),
                collided_this=collided,
                step=step,
                total_reward=total_reward,
                direction=direction.copy(),
            ))

        # Arrival frames — robot at destination, member lights up.
        robot_xy = to_xy.copy()
        placed_ids = placed_ids | {member_id}
        completed_paths.append(
            ((float(from_xy[0]), float(from_xy[1])),
             (float(to_xy[0]),   float(to_xy[1])),
             collided)
        )
        for _ in range(pause_frames):
            frames.append(_Frame(
                robot_xy=robot_xy.copy(),
                placed_ids=placed_ids,
                target_id=None,
                paths=list(completed_paths),
                partial_path=None,
                collided_this=False,
                step=step,
                total_reward=total_reward,
                direction=direction.copy(),
            ))

        if terminated:
            break

    # Hold on the final frame.
    for _ in range(pause_frames * 4):
        frames.append(frames[-1])

    return frames


# ------------------------------------------------------------------ #
# Drawing primitives                                                   #
# ------------------------------------------------------------------ #

def _draw_member(
    ax: plt.Axes,
    member: Member,
    placed: bool,
    is_target: bool,
) -> None:
    x_min, y_min, x_max, y_max = member.bounds
    w, h = x_max - x_min, y_max - y_min

    if placed:
        color = _KIND_COLOR.get(member.kind, "#DEB887")
        edge  = "#5D4037"
        alpha = 1.0
        lw    = 0.8
    elif is_target:
        color = _TARGET_GLOW
        edge  = "#F57F17"
        alpha = 0.85
        lw    = 1.2
    else:
        color = _GHOST_FACE
        edge  = _GHOST_EDGE
        alpha = 1.0
        lw    = 0.5

    rect = mpatches.FancyBboxPatch(
        (x_min, y_min), w, h,
        boxstyle="square,pad=0",
        facecolor=color,
        edgecolor=edge,
        linewidth=lw,
        alpha=alpha,
        zorder=2 if placed else 1,
    )
    ax.add_patch(rect)


def _robot_triangle(center: np.ndarray, direction: np.ndarray, size: float) -> np.ndarray:
    """Return (3, 2) array of triangle vertices pointing in *direction*."""
    angle = np.arctan2(direction[1], direction[0])
    tip   = center + size       * np.array([np.cos(angle),           np.sin(angle)])
    left  = center + size * 0.6 * np.array([np.cos(angle + 2.3),    np.sin(angle + 2.3)])
    right = center + size * 0.6 * np.array([np.cos(angle - 2.3),    np.sin(angle - 2.3)])
    return np.array([tip, left, right])


def render_panel_snapshot(
    panel: Panel,
    placed_ids: frozenset[str] | set[str],
    robot_pos: Position | None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Draw a single static snapshot of the panel — useful for debugging.

    Parameters
    ----------
    panel:
        The panel to render.
    placed_ids:
        Member ids that should be shown as placed (filled with lumber colour).
    robot_pos:
        If provided, draw the robot triangle at this position.
    ax:
        Matplotlib axes to draw on.  Created if not provided.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=_figure_size(panel))

    _setup_axes(ax, panel)

    for member in panel.members:
        _draw_member(ax, member, placed=member.id in placed_ids, is_target=False)

    if robot_pos is not None:
        tri = _robot_triangle(
            np.array(robot_pos), np.array([1.0, 0.0]), size=_robot_size(panel)
        )
        ax.add_patch(Polygon(tri, closed=True, facecolor=_ROBOT_COLOR,
                             edgecolor="white", linewidth=0.8, zorder=5))
    return ax


# ------------------------------------------------------------------ #
# Animation                                                            #
# ------------------------------------------------------------------ #

def animate_episode(
    env: PanelEnv,
    policy: Callable[[PanelEnv], int],
    *,
    policy_name: str = "policy",
    fps: int = 20,
    frames_per_move: int = 12,
    pause_frames: int = 4,
) -> FuncAnimation:
    """Run a full episode and return a ``FuncAnimation``.

    Parameters
    ----------
    env:
        A ``PanelEnv`` (or ``RandomPanelEnv``) — will be ``reset()``
        internally at the start of frame collection.
    policy:
        Callable ``(env) → action`` — same interface as the baselines.
    policy_name:
        Label shown in the title bar.
    fps:
        Frames per second for the animation.
    frames_per_move:
        Interpolation steps between member placements.  Higher = smoother.
    pause_frames:
        Frames to hold after each placement.
    """
    panel = env.panel
    frames = _collect_frames(env, policy, frames_per_move, pause_frames)

    fig, ax = plt.subplots(figsize=_figure_size(panel))
    fig.patch.set_facecolor("#FAFAFA")
    plt.tight_layout(pad=1.2)

    rsize = _robot_size(panel)

    def _draw(frame: _Frame) -> None:
        ax.cla()
        _setup_axes(ax, panel)

        # Title.
        ax.set_title(
            f"{policy_name}  |  step {frame.step}/{env.n_members}"
            f"  |  reward {frame.total_reward:.1f}",
            fontsize=9, pad=4, color="#333333",
        )

        # Draw all members.
        for member in panel.members:
            _draw_member(
                ax, member,
                placed=member.id in frame.placed_ids,
                is_target=member.id == frame.target_id,
            )

        # Completed paths.
        for from_pos, to_pos, collided in frame.paths:
            color = _PATH_COLLIDE if collided else _PATH_CLEAR
            ax.annotate(
                "", xy=to_pos, xytext=from_pos,
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=1.2,
                    mutation_scale=8,
                ),
                zorder=3,
            )

        # Growing path (current move).
        if frame.partial_path is not None:
            from_pos, to_pos = frame.partial_path
            color = _PATH_COLLIDE if frame.collided_this else _PATH_CLEAR
            ax.plot(
                [from_pos[0], to_pos[0]],
                [from_pos[1], to_pos[1]],
                color=color, linewidth=1.2, alpha=0.7, zorder=3,
            )

        # Robot triangle.
        tri = _robot_triangle(frame.robot_xy, frame.direction, size=rsize)
        ax.add_patch(Polygon(
            tri, closed=True,
            facecolor=_ROBOT_COLOR, edgecolor="white",
            linewidth=0.8, zorder=5,
        ))

        # Legend (drawn once — static content).
        _draw_legend(ax, panel)

    anim = FuncAnimation(
        fig,
        func=lambda i: _draw(frames[i]),
        frames=len(frames),
        interval=1000 / fps,
        repeat=True,
        blit=False,
    )
    return anim


# ------------------------------------------------------------------ #
# Convenience savers                                                   #
# ------------------------------------------------------------------ #

def save_episode_gif(
    env: PanelEnv,
    policy: Callable[[PanelEnv], int],
    output_path: str,
    *,
    policy_name: str = "policy",
    fps: int = 20,
    frames_per_move: int = 12,
    pause_frames: int = 4,
    dpi: int = 120,
) -> None:
    """Save a full episode animation as a GIF.

    Requires Pillow (``pip install Pillow``).  The GIF is written to
    ``output_path``; the directory must already exist.

    Parameters mirror ``animate_episode``; ``dpi`` controls output resolution.
    """
    anim = animate_episode(
        env, policy,
        policy_name=policy_name,
        fps=fps,
        frames_per_move=frames_per_move,
        pause_frames=pause_frames,
    )
    anim.save(output_path, writer="pillow", fps=fps, dpi=dpi)
    print(f"Saved → {output_path}")
    plt.close("all")


# ------------------------------------------------------------------ #
# Internal layout helpers                                              #
# ------------------------------------------------------------------ #

def _figure_size(panel: Panel) -> tuple[float, float]:
    """Scale figure so the wall always looks right, ~10" wide."""
    aspect = panel.wall_length / panel.wall_height
    return (10.0, 10.0 / aspect + 0.8)


def _robot_size(panel: Panel) -> float:
    """Robot triangle size proportional to the panel — about 3% of height."""
    return panel.wall_height * 0.035


def _setup_axes(ax: plt.Axes, panel: Panel) -> None:
    """Configure axes limits, background, and wall outline."""
    margin = panel.wall_height * 0.06
    ax.set_xlim(-margin, panel.wall_length + margin)
    ax.set_ylim(-margin, panel.wall_height + margin)
    ax.set_aspect("equal")
    ax.set_facecolor(_WALL_BG)
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Wall perimeter.
    wall_rect = mpatches.Rectangle(
        (0, 0), panel.wall_length, panel.wall_height,
        linewidth=1.5, edgecolor=_WALL_OUTLINE,
        facecolor="none", zorder=0,
    )
    ax.add_patch(wall_rect)


def _draw_legend(ax: plt.Axes, panel: Panel) -> None:
    """Draw a compact member-type legend using the kinds present in the panel."""
    kinds_present = {m.kind for m in panel.members}
    handles = []
    labels  = []
    for kind in MemberKind:
        if kind not in kinds_present:
            continue
        color = _KIND_COLOR.get(kind, "#DEB887")
        handles.append(mpatches.Patch(facecolor=color, edgecolor="#5D4037",
                                      linewidth=0.6))
        labels.append(kind.value.replace("_", " ").title())

    # Path legend entries.
    handles += [
        plt.Line2D([0], [0], color=_PATH_CLEAR,   linewidth=1.5),
        plt.Line2D([0], [0], color=_PATH_COLLIDE, linewidth=1.5),
        mpatches.Patch(facecolor=_ROBOT_COLOR, edgecolor="white"),
    ]
    labels += ["clear path", "collision path", "robot"]

    ax.legend(
        handles, labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=7,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        handlelength=1.4,
    )
