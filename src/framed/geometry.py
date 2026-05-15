"""2-D geometry primitives for robot path planning on the panel table.

Coordinate system
-----------------
Identical to panel.py: x runs along the wall's length, y along the wall's
height (when erected).  (0, 0) is the bottom-left corner of the wall.  All
coordinates are floats in the canonical unit (see ``framed.units``).

Public API
----------
``Rectangle``               Named-field axis-aligned bounding box.
``travel_time``             Euclidean travel time between two points at a speed.
``segment_intersects_rect`` True iff the open interior of a rect is pierced.
``path_collides``           True iff a straight robot path hits any placed member.

Strict-interior convention
--------------------------
``segment_intersects_rect`` (and therefore ``path_collides``) treats the
rectangle as an **open** set: a segment that merely grazes an edge or corner
returns False.  This mirrors the Panel no-overlaps validator, which permits
edge-sharing between adjacent members (e.g. a stud's top face touching the
bottom face of the top plate).

Zero-clearance note
-------------------
``path_collides`` models the robot end-effector as a dimensionless point.
To add physical clearance around members, expand each member's footprint
before passing it in by calling ``Rectangle.from_member(m).grow(clearance)``
and then calling ``segment_intersects_rect`` directly.
"""
from __future__ import annotations

import math
from typing import NamedTuple

from framed.panel import Member

# Mirrors the alias in panel.py; both are just ``tuple[float, float]``.
Position = tuple[float, float]


class Rectangle(NamedTuple):
    """Axis-aligned bounding box with named fields.

    Field order matches ``Member.bounds`` — (x_min, y_min, x_max, y_max) —
    so ``Rectangle(*member.bounds)`` and ``Rectangle.from_member(member)``
    are equivalent.  Because ``Rectangle`` is a NamedTuple it is also a
    plain tuple, so it can be unpacked anywhere a four-float tuple is
    expected.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    # ------------------------------------------------------------------ #
    # Constructors                                                         #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_member(cls, member: Member) -> Rectangle:
        """Construct from a Member's footprint (``member.bounds``)."""
        return cls(*member.bounds)

    @classmethod
    def from_position_and_size(
        cls,
        position: Position,
        size: tuple[float, float],
    ) -> Rectangle:
        """Construct from a bottom-left ``position`` and ``(width, height)``
        size pair — the same representation used by ``Member.position`` and
        ``Member.size``."""
        x, y = position
        w, h = size
        return cls(x, y, x + w, y + h)

    # ------------------------------------------------------------------ #
    # Derived properties                                                   #
    # ------------------------------------------------------------------ #

    @property
    def width(self) -> float:
        """Extent along the x-axis (wall length direction)."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Extent along the y-axis (wall height direction)."""
        return self.y_max - self.y_min

    @property
    def center(self) -> Position:
        """Centroid of the rectangle."""
        return (
            (self.x_min + self.x_max) / 2.0,
            (self.y_min + self.y_max) / 2.0,
        )

    # ------------------------------------------------------------------ #
    # Manipulation                                                         #
    # ------------------------------------------------------------------ #

    def grow(self, amount: float) -> Rectangle:
        """Return a new Rectangle expanded outward by *amount* on all four
        sides.

        Useful for adding a physical clearance envelope around a placed
        member before passing it to ``segment_intersects_rect``::

            rect = Rectangle.from_member(m).grow(clearance_inches)
            hit = segment_intersects_rect(robot_pos, target, rect)

        A negative *amount* shrinks the rectangle; the caller is responsible
        for ensuring the result remains valid (x_min < x_max, y_min < y_max).
        """
        return Rectangle(
            self.x_min - amount,
            self.y_min - amount,
            self.x_max + amount,
            self.y_max + amount,
        )


# ------------------------------------------------------------------ #
# Travel time                                                          #
# ------------------------------------------------------------------ #

def travel_time(a: Position, b: Position, speed: float) -> float:
    """Return the Euclidean travel time from *a* to *b* at *speed*.

    Parameters
    ----------
    a, b:
        Start and end positions in canonical units.
    speed:
        Travel speed in canonical units per time unit.  Must be strictly
        positive.

    Returns
    -------
    float
        ``distance(a, b) / speed``.  Returns 0.0 when *a == b*.

    Raises
    ------
    ValueError
        If *speed* ≤ 0.
    """
    if speed <= 0.0:
        raise ValueError(f"speed must be positive, got {speed!r}")
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.hypot(dx, dy) / speed


# ------------------------------------------------------------------ #
# Segment / rectangle intersection                                     #
# ------------------------------------------------------------------ #

def segment_intersects_rect(
    p1: Position,
    p2: Position,
    rect: Rectangle,
) -> bool:
    """Return True iff segment p1 → p2 pierces the *open interior* of *rect*.

    Segments that only touch a boundary or corner (but do not cross into the
    interior) return False.  This is intentional — see module docstring.

    Algorithm
    ---------
    Liang-Barsky parameterised clipping.  The segment is expressed as::

        P(t) = p1 + t·(p2 − p1),   t ∈ [0, 1]

    and clipped against each of the four half-planes that bound the
    rectangle.  Each clip edge produces a constraint on the ``t``-interval.
    A non-empty surviving interval (``t_enter < t_exit`` strictly) means the
    segment's interior overlaps the rectangle's open interior.

    For the parallel case (``p == 0``): ``q ≤ 0`` means the segment lies on
    or outside the corresponding boundary, so we return False immediately.
    Using ``≤`` (rather than the standard ``<``) ensures a segment sitting
    exactly on an edge is not counted as an interior intersection.

    Floating-point note
    -------------------
    No epsilon guard is applied.  Coordinates in this project are small
    multiples of ``LUMBER_THICKNESS`` (1.5 in) so floating-point noise is
    far below any physically meaningful threshold.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    # Each (p, q) pair encodes one half-plane constraint:
    #
    #   left   (x ≥ x_min):  p = −dx,  q = px − x_min
    #   right  (x ≤ x_max):  p =  dx,  q = x_max − px
    #   bottom (y ≥ y_min):  p = −dy,  q = py − y_min
    #   top    (y ≤ y_max):  p =  dy,  q = y_max − py
    #
    # p < 0 → this is an "entering" boundary as t increases.
    # p > 0 → this is an "exiting"  boundary as t increases.
    # p = 0 → segment is parallel to this boundary.
    clip = (
        (-dx,  p1[0] - rect.x_min),   # left
        ( dx,  rect.x_max - p1[0]),   # right
        (-dy,  p1[1] - rect.y_min),   # bottom
        ( dy,  rect.y_max - p1[1]),   # top
    )

    t_enter, t_exit = 0.0, 1.0

    for p, q in clip:
        if p == 0.0:
            # Parallel to this boundary.
            # q <= 0: segment is outside or exactly on the boundary → no
            # interior intersection along this dimension is possible.
            if q <= 0.0:
                return False
            # q > 0: segment is strictly inside this half-plane; no t-clipping.
        elif p < 0.0:
            # Entering half-plane: raise the lower bound on t.
            t_enter = max(t_enter, q / p)
        else:
            # Exiting half-plane: lower the upper bound on t.
            t_exit = min(t_exit, q / p)

        # Early exit: the surviving interval is already empty.
        if t_enter >= t_exit:
            return False

    # Strict inequality: t_enter == t_exit means the segment only touches a
    # boundary or corner and does not pass through the open interior.
    return t_enter < t_exit


# ------------------------------------------------------------------ #
# Path collision                                                       #
# ------------------------------------------------------------------ #

def path_collides(
    robot_pos: Position,
    target: Position,
    placed_members: list[Member],
) -> bool:
    """Return True iff the straight-line path from *robot_pos* to *target*
    passes through the open interior of any member in *placed_members*.

    Parameters
    ----------
    robot_pos:
        Current robot end-effector (or tool-centre-point) position.
    target:
        Destination position — typically the centre of the member being
        placed, or a pick-up / put-down point.
    placed_members:
        Members already on the assembly table; their footprints are the
        obstacles.

    Returns
    -------
    bool
        True if any placed member's footprint is pierced; False otherwise.

    Notes
    -----
    *target* is a **position**, not a Member.  The caller is responsible for
    translating a target member's centre (``member.center``) or any other
    reference point into a coordinate.  The member being placed should
    **not** appear in *placed_members* yet.

    The function short-circuits on the first collision found; ordering of
    ``placed_members`` does not affect correctness, only performance.
    """
    for member in placed_members:
        if segment_intersects_rect(robot_pos, target, Rectangle.from_member(member)):
            return True
    return False
