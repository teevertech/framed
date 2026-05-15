"""Tests for framed.geometry.

Structure
---------
TestRectangle           Construction, properties, grow().
TestTravelTime          Basic arithmetic, edge cases, bad input.
TestSegmentIntersectsRect
    Clear misses and clear hits (parametrised).
    Boundary and corner edge cases (individual, with explanatory comments).
TestPathCollides        Integration: uses real Member objects.

Coordinate convention: x = wall-length direction, y = wall-height direction,
(0, 0) = bottom-left.  All values are in canonical units (inches) unless a
comment says otherwise; helpers from framed.units are used for readability.

A standard 6×6 test rectangle ``R = Rectangle(2, 2, 8, 8)`` is reused
across many segment tests so the interesting geometry is obvious at a glance.
"""
from __future__ import annotations

import math

import pytest

from framed.geometry import (
    Rectangle,
    path_collides,
    segment_intersects_rect,
    travel_time,
)
from framed.panel import Member, MemberKind
from framed.units import feet, inches


# ===================================================================== #
# Fixtures and helpers                                                   #
# ===================================================================== #

# Standard 6×6 rectangle used throughout segment tests.
R = Rectangle(2.0, 2.0, 8.0, 8.0)


def _member(
    id: str = "m",
    position: tuple[float, float] = (0.0, 0.0),
    size: tuple[float, float] = (10.0, 10.0),
    kind: MemberKind = MemberKind.COMMON_STUD,
) -> Member:
    return Member(id=id, kind=kind, position=position, size=size)


# ===================================================================== #
# Rectangle                                                              #
# ===================================================================== #

class TestRectangle:
    def test_positional_construction(self) -> None:
        r = Rectangle(1.0, 2.0, 5.0, 7.0)
        assert r.x_min == 1.0
        assert r.y_min == 2.0
        assert r.x_max == 5.0
        assert r.y_max == 7.0

    def test_is_a_plain_tuple(self) -> None:
        """NamedTuple must remain tuple-compatible for callers that unpack."""
        r = Rectangle(1.0, 2.0, 5.0, 7.0)
        x_min, y_min, x_max, y_max = r
        assert (x_min, y_min, x_max, y_max) == (1.0, 2.0, 5.0, 7.0)
        assert r[0] == 1.0

    def test_from_member(self) -> None:
        m = _member(position=(3.0, 4.0), size=(10.0, 20.0))
        r = Rectangle.from_member(m)
        assert r == Rectangle(3.0, 4.0, 13.0, 24.0)

    def test_from_member_matches_bounds(self) -> None:
        """from_member must equal Rectangle(*member.bounds) for any member."""
        m = _member(position=(0.0, 1.5), size=(inches(1.5), feet(8)))
        assert Rectangle.from_member(m) == Rectangle(*m.bounds)

    def test_from_position_and_size(self) -> None:
        r = Rectangle.from_position_and_size((3.0, 4.0), (10.0, 20.0))
        assert r == Rectangle(3.0, 4.0, 13.0, 24.0)

    def test_width(self) -> None:
        r = Rectangle(2.0, 5.0, 9.0, 11.0)
        assert r.width == pytest.approx(7.0)

    def test_height(self) -> None:
        r = Rectangle(2.0, 5.0, 9.0, 11.0)
        assert r.height == pytest.approx(6.0)

    def test_center(self) -> None:
        r = Rectangle(0.0, 0.0, 10.0, 4.0)
        cx, cy = r.center
        assert cx == pytest.approx(5.0)
        assert cy == pytest.approx(2.0)

    def test_center_non_square(self) -> None:
        r = Rectangle(1.0, 3.0, 5.0, 9.0)
        assert r.center == pytest.approx((3.0, 6.0))

    def test_grow_expands_all_sides(self) -> None:
        r = Rectangle(2.0, 2.0, 8.0, 8.0)
        g = r.grow(1.0)
        assert g == Rectangle(1.0, 1.0, 9.0, 9.0)

    def test_grow_zero_is_identity(self) -> None:
        r = Rectangle(2.0, 2.0, 8.0, 8.0)
        assert r.grow(0.0) == r

    def test_grow_negative_shrinks(self) -> None:
        r = Rectangle(2.0, 2.0, 8.0, 8.0)
        g = r.grow(-1.0)
        assert g == Rectangle(3.0, 3.0, 7.0, 7.0)


# ===================================================================== #
# travel_time                                                            #
# ===================================================================== #

class TestTravelTime:
    def test_horizontal_segment(self) -> None:
        # 10 units at speed 2 → 5 time units
        assert travel_time((0.0, 0.0), (10.0, 0.0), speed=2.0) == pytest.approx(5.0)

    def test_vertical_segment(self) -> None:
        assert travel_time((0.0, 0.0), (0.0, 10.0), speed=5.0) == pytest.approx(2.0)

    def test_diagonal_3_4_5(self) -> None:
        # Classic 3-4-5 right triangle → distance = 5
        assert travel_time((0.0, 0.0), (3.0, 4.0), speed=1.0) == pytest.approx(5.0)

    def test_same_point_returns_zero(self) -> None:
        assert travel_time((5.0, 7.0), (5.0, 7.0), speed=10.0) == pytest.approx(0.0)

    def test_speed_scales_inversely(self) -> None:
        t1 = travel_time((0.0, 0.0), (6.0, 8.0), speed=1.0)   # distance = 10
        t2 = travel_time((0.0, 0.0), (6.0, 8.0), speed=2.0)
        assert t1 == pytest.approx(10.0)
        assert t2 == pytest.approx(5.0)

    def test_zero_speed_raises(self) -> None:
        with pytest.raises(ValueError, match="speed must be positive"):
            travel_time((0.0, 0.0), (1.0, 0.0), speed=0.0)

    def test_negative_speed_raises(self) -> None:
        with pytest.raises(ValueError, match="speed must be positive"):
            travel_time((0.0, 0.0), (1.0, 0.0), speed=-1.0)

    def test_symmetry(self) -> None:
        """Direction should not matter."""
        a, b = (3.0, 7.0), (11.0, 2.0)
        assert travel_time(a, b, speed=3.0) == pytest.approx(travel_time(b, a, speed=3.0))


# ===================================================================== #
# segment_intersects_rect                                                #
# ===================================================================== #

# R = Rectangle(2, 2, 8, 8) throughout this class.

class TestSegmentIntersectsRect:

    # ---- parametrised: clear misses (all expected False) ---- #

    @pytest.mark.parametrize("p1, p2", [
        # Entirely to the left of R
        ((0.0, 5.0), (1.5, 5.0)),
        # Entirely to the right
        ((8.5, 5.0), (12.0, 5.0)),
        # Entirely below
        ((5.0, 0.0), (5.0, 1.5)),
        # Entirely above
        ((5.0, 8.5), (5.0, 12.0)),
        # Diagonal that passes under the bottom-left corner
        ((0.0, 0.0), (1.9, 1.9)),
        # Passes to the right of R (x always > 8)
        ((9.0, 0.0), (9.0, 10.0)),
    ])
    def test_clear_miss(
        self, p1: tuple[float, float], p2: tuple[float, float]
    ) -> None:
        assert segment_intersects_rect(p1, p2, R) is False

    # ---- parametrised: clear hits (all expected True) ---- #

    @pytest.mark.parametrize("p1, p2", [
        # Horizontal through the centre
        ((0.0, 5.0), (10.0, 5.0)),
        # Vertical through the centre
        ((5.0, 0.0), (5.0, 10.0)),
        # Diagonal through the centre (NW → SE)
        ((0.0, 10.0), (10.0, 0.0)),
        # Diagonal through the centre (SW → NE)
        ((0.0, 0.0), (10.0, 10.0)),
        # Segment starts outside, ends inside (p2 is the centre)
        ((0.0, 5.0), (5.0, 5.0)),
        # Segment starts inside, ends outside
        ((5.0, 5.0), (10.0, 5.0)),
        # Both endpoints strictly inside R
        ((3.0, 3.0), (7.0, 7.0)),
    ])
    def test_clear_hit(
        self, p1: tuple[float, float], p2: tuple[float, float]
    ) -> None:
        assert segment_intersects_rect(p1, p2, R) is True

    # ---- edge and corner cases ---- #

    def test_segment_on_left_edge_is_not_interior(self) -> None:
        """A horizontal segment exactly on x = x_min (the left boundary) does
        not enter the open interior → False."""
        # p1=(2,3), p2=(2,7): vertical along x=2=R.x_min
        assert segment_intersects_rect((2.0, 3.0), (2.0, 7.0), R) is False

    def test_segment_on_right_edge_is_not_interior(self) -> None:
        assert segment_intersects_rect((8.0, 3.0), (8.0, 7.0), R) is False

    def test_segment_on_bottom_edge_is_not_interior(self) -> None:
        """Horizontal segment along y = y_min."""
        assert segment_intersects_rect((3.0, 2.0), (7.0, 2.0), R) is False

    def test_segment_on_top_edge_is_not_interior(self) -> None:
        """Horizontal segment along y = y_max."""
        assert segment_intersects_rect((3.0, 8.0), (7.0, 8.0), R) is False

    def test_full_bottom_edge_traversal_is_not_interior(self) -> None:
        """Segment crossing the entire wall along the bottom edge (exactly on
        y_min from x_min to x_max) must not count as an interior hit."""
        assert segment_intersects_rect((2.0, 2.0), (8.0, 2.0), R) is False

    def test_diagonal_grazes_bottom_left_corner(self) -> None:
        """Segment from (0,4) to (4,0) passes through corner (2,2) of R.
        It touches only a single boundary point and never enters the open
        interior → False.

        P(t) = (4t, 4-4t).  Corner (2,2) is hit at t=0.5.
        After the corner the segment moves into y<2, outside R.
        """
        assert segment_intersects_rect((0.0, 4.0), (4.0, 0.0), R) is False

    def test_diagonal_grazes_top_right_corner(self) -> None:
        """Symmetric case: segment from (6,10) to (10,6) grazes corner (8,8)."""
        assert segment_intersects_rect((6.0, 10.0), (10.0, 6.0), R) is False

    def test_segment_endpoint_exactly_on_boundary_enters_interior(self) -> None:
        """When p1 sits exactly on the left boundary and p2 is strictly
        inside, the segment immediately enters the open interior → True."""
        assert segment_intersects_rect((2.0, 5.0), (5.0, 5.0), R) is True

    def test_segment_endpoint_exactly_on_boundary_stays_outside(self) -> None:
        """p2 lands exactly on the left boundary but the path comes from the
        left → the path never enters the open interior → False."""
        assert segment_intersects_rect((-3.0, 5.0), (2.0, 5.0), R) is False

    def test_zero_length_segment_strictly_inside_is_hit(self) -> None:
        """A point inside the rect → the 'segment' is entirely in the
        interior → True."""
        assert segment_intersects_rect((5.0, 5.0), (5.0, 5.0), R) is True

    def test_zero_length_segment_on_boundary_is_not_hit(self) -> None:
        """A point exactly on the left boundary → False (boundary ≠ interior)."""
        assert segment_intersects_rect((2.0, 5.0), (2.0, 5.0), R) is False

    def test_zero_length_segment_outside_is_not_hit(self) -> None:
        assert segment_intersects_rect((1.0, 5.0), (1.0, 5.0), R) is False

    def test_segment_passes_between_two_close_rects_without_hitting_either(
        self,
    ) -> None:
        """Two side-by-side rectangles with a 1-unit gap; a vertical segment
        in the gap should not collide with either."""
        left  = Rectangle(0.0, 0.0, 4.0, 10.0)
        right = Rectangle(5.0, 0.0, 9.0, 10.0)
        # Segment runs through the gap at x=4.5
        assert segment_intersects_rect((4.5, 0.0), (4.5, 10.0), left) is False
        assert segment_intersects_rect((4.5, 0.0), (4.5, 10.0), right) is False

    def test_segment_reversed_gives_same_result(self) -> None:
        """Collision detection must be direction-agnostic."""
        p1, p2 = (0.0, 5.0), (10.0, 5.0)
        assert segment_intersects_rect(p1, p2, R) == segment_intersects_rect(p2, p1, R)

    def test_thin_rect_lumber_thickness(self) -> None:
        """A rect as thin as a real 2×4 (1.5 in wide) must be detected when
        the path crosses it perpendicularly."""
        stud_rect = Rectangle(10.0, 0.0, 11.5, 96.0)   # 1.5" stud, full height
        # Horizontal path at mid-height, crossing the stud
        assert segment_intersects_rect((0.0, 48.0), (20.0, 48.0), stud_rect) is True
        # Horizontal path missing entirely to the left
        assert segment_intersects_rect((0.0, 48.0), (9.9, 48.0), stud_rect) is False

    def test_degenerate_rect_zero_width(self) -> None:
        """Rectangle with zero width: a segment crossing it should not
        collide (no open interior exists in the x direction).
        The parallel-plus-q=0 rule handles this via the left or right boundary."""
        # x_min == x_max → zero width → no open interior along x
        zero_width = Rectangle(5.0, 0.0, 5.0, 10.0)
        assert segment_intersects_rect((0.0, 5.0), (10.0, 5.0), zero_width) is False


# ===================================================================== #
# path_collides                                                          #
# ===================================================================== #

class TestPathCollides:

    def test_empty_member_list_never_collides(self) -> None:
        assert path_collides((0.0, 0.0), (100.0, 100.0), []) is False

    def test_path_through_single_member_collides(self) -> None:
        obstacle = _member("obs", position=(4.0, 0.0), size=(2.0, 10.0))
        # Horizontal path crossing the obstacle at x=4–6
        assert path_collides((0.0, 5.0), (10.0, 5.0), [obstacle]) is True

    def test_path_missing_single_member_does_not_collide(self) -> None:
        obstacle = _member("obs", position=(4.0, 0.0), size=(2.0, 10.0))
        # Path passes well below the obstacle's x-range
        assert path_collides((0.0, 5.0), (3.9, 5.0), [obstacle]) is False

    def test_path_between_two_members_does_not_collide(self) -> None:
        """Two studs side-by-side with a gap; path runs through the gap."""
        left  = _member("left",  position=(0.0, 0.0), size=(1.5, 90.0))
        right = _member("right", position=(3.0, 0.0), size=(1.5, 90.0))
        # Vertical path in the gap at x=2.25
        assert path_collides((2.25, 95.0), (2.25, -5.0), [left, right]) is False

    def test_path_hits_second_of_two_members(self) -> None:
        far   = _member("far",  position=(10.0, 0.0), size=(2.0, 10.0))
        close = _member("close", position=(4.0,  0.0), size=(2.0, 10.0))
        # Path hits 'close' but not 'far'; order should not matter
        assert path_collides((0.0, 5.0), (7.0, 5.0), [far, close]) is True
        assert path_collides((0.0, 5.0), (7.0, 5.0), [close, far]) is True

    def test_path_hits_only_member_in_a_crowd(self) -> None:
        """Only the stud directly in the path should trigger a collision."""
        clear1 = _member("c1", position=(0.0,  0.0), size=(1.5, 90.0))
        blocker = _member("bl", position=(20.0, 0.0), size=(1.5, 90.0))
        clear2 = _member("c2", position=(40.0, 0.0), size=(1.5, 90.0))
        # Horizontal path at y=45 (mid-height), starting left of clear1
        # and ending right of clear2
        assert path_collides((-5.0, 45.0), (50.0, 45.0), [clear1, blocker, clear2]) is True

    def test_robot_starts_inside_member_collides(self) -> None:
        """If the robot starts inside a placed member's footprint the path is
        already in the interior and should be flagged."""
        obstacle = _member("obs", position=(0.0, 0.0), size=(10.0, 10.0))
        # Start point (5, 5) is inside the obstacle
        assert path_collides((5.0, 5.0), (20.0, 5.0), [obstacle]) is True

    def test_path_along_member_edge_does_not_collide(self) -> None:
        """A path running exactly along the left boundary of a member does
        not pierce its open interior → False."""
        stud = _member("stud", position=(10.0, 0.0), size=(1.5, 90.0))
        # Path along x=10 (the left edge of the stud)
        assert path_collides((10.0, -5.0), (10.0, 95.0), [stud]) is False

    def test_uses_rectangle_from_member_bounds(self) -> None:
        """path_collides must use the member's actual bounds, not some
        approximation.  A 1.5-inch-wide stud should be detectable."""
        thin_stud = _member(
            "ts",
            position=(inches(16), inches(1.5)),
            size=(inches(1.5), inches(93)),
        )
        # Path aimed at the stud's centre
        cx, cy = thin_stud.center
        assert path_collides((0.0, cy), (cx * 2, cy), [thin_stud]) is True
