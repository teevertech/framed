"""Pydantic data models for wall panel specifications.

A Panel fully describes one instance of the assembly problem: a wall of
given dimensions plus the framing members that compose it, each with
position, size, and the prerequisites that must be placed first.

Coordinate system
-----------------
The wall is modeled top-down (as it would lie flat on an assembly table).
The x axis runs along the wall's length; the y axis runs along the wall's
height (when erected). (0, 0) is the bottom-left corner. Each member
occupies an axis-aligned rectangle, given by its bottom-left `position`
and its `size` = (extent_x, extent_y).

All lengths, positions, and sizes are floats in the canonical unit (see
`framed.units`). The data model itself is unit-agnostic.

Member kinds and categories
---------------------------
Nine concrete kinds map onto four categories. The categories match the
color encoding in the README/anatomy diagram and group members by their
role and precedence pattern:

  PLATE             top plate, bottom plate, sill plate
  COMMON_STUD       common stud (full height, away from openings)
  OPENING_FRAMING   king stud, jack stud, header
  CRIPPLE           top cripple, bottom cripple

Precedence
----------
Each Member carries an explicit `prerequisites` list of other member ids
that must be placed first. The generator stamps these from physical
assembly rules at construction time, so the env never needs to reason
about "kind X needs kind Y." Cycle detection and dangling-reference
checks live in the Panel validators.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, Field, model_validator

from framed.units import feet, inches

# Coordinate / dimension aliases. Floats in the canonical unit.
Position = tuple[float, float]
Size = tuple[float, float]


# ----- Dimensional lumber and panel defaults -----

# Cross-section of a nominal 2x4. The 1.5" dimension is what's visible in
# our 2D top-down model (the wall is built on edge — the 3.5" face is the
# wall's depth, perpendicular to the modeled plane).
LUMBER_THICKNESS = inches(1.5)

DEFAULT_WALL_HEIGHT = feet(8)
DEFAULT_STUD_SPACING = inches(16)

# Header sits at the top of the rough opening. 82" = 6'10", the standard
# rough-opening top for both doors and most windows in residential
# construction.
DEFAULT_HEADER_BOTTOM_Y = inches(82)
DEFAULT_HEADER_DEPTH = inches(3.5)  # single 2x4 header; MVP simplification

# Sill sits at the bottom of the window rough opening, 36" off the floor.
DEFAULT_SILL_TOP_Y = feet(3)
DEFAULT_SILL_THICKNESS = LUMBER_THICKNESS

# Maximum wall length supported (no plate splicing above this).
MAX_WALL_LENGTH = feet(16)


class MemberKind(StrEnum):
    TOP_PLATE = "top_plate"
    BOTTOM_PLATE = "bottom_plate"
    SILL_PLATE = "sill_plate"
    COMMON_STUD = "common_stud"
    KING_STUD = "king_stud"
    JACK_STUD = "jack_stud"
    HEADER = "header"
    TOP_CRIPPLE = "top_cripple"
    BOTTOM_CRIPPLE = "bottom_cripple"


class MemberCategory(StrEnum):
    PLATE = "plate"
    COMMON_STUD = "common_stud"
    OPENING_FRAMING = "opening_framing"
    CRIPPLE = "cripple"


MEMBER_CATEGORY: dict[MemberKind, MemberCategory] = {
    MemberKind.TOP_PLATE: MemberCategory.PLATE,
    MemberKind.BOTTOM_PLATE: MemberCategory.PLATE,
    MemberKind.SILL_PLATE: MemberCategory.PLATE,
    MemberKind.COMMON_STUD: MemberCategory.COMMON_STUD,
    MemberKind.KING_STUD: MemberCategory.OPENING_FRAMING,
    MemberKind.JACK_STUD: MemberCategory.OPENING_FRAMING,
    MemberKind.HEADER: MemberCategory.OPENING_FRAMING,
    MemberKind.TOP_CRIPPLE: MemberCategory.CRIPPLE,
    MemberKind.BOTTOM_CRIPPLE: MemberCategory.CRIPPLE,
}


class Member(BaseModel):
    """A single framing member with a footprint and dependencies."""

    id: str
    kind: MemberKind
    position: Position  # (x, y) of the bottom-left corner
    size: Size  # (extent_x, extent_y), both > 0
    prerequisites: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def positive_size(self) -> Self:
        if self.size[0] <= 0 or self.size[1] <= 0:
            raise ValueError(
                f"Member {self.id} has non-positive size: {self.size}"
            )
        return self

    @property
    def category(self) -> MemberCategory:
        return MEMBER_CATEGORY[self.kind]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(x_min, y_min, x_max, y_max) of the member's footprint."""
        x, y = self.position
        w, h = self.size
        return (x, y, x + w, y + h)

    @property
    def center(self) -> Position:
        x, y = self.position
        w, h = self.size
        return (x + w / 2, y + h / 2)


class OpeningSpec(BaseModel):
    """Metadata about a single opening (window or door) in a panel.

    Stored on the Panel for the web UI and future CAD import validation.
    The env itself does not use this field — all placement logic is
    encoded directly in member prerequisites.
    """

    kind: Literal["window", "door"]
    center_x: float          # canonical units, determined at generation time
    width: float             # canonical units (rough opening width)
    member_ids: list[str]    # all member ids belonging to this opening


class Panel(BaseModel):
    """A wall panel: the full assembly specification."""

    wall_length: float = Field(gt=0)
    wall_height: float = Field(gt=0)
    members: list[Member] = Field(min_length=1)
    openings: list[OpeningSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        ids = [m.id for m in self.members]
        if len(ids) != len(set(ids)):
            seen: set[str] = set()
            dupes = sorted({i for i in ids if i in seen or seen.add(i)})  # type: ignore[func-returns-value]
            raise ValueError(f"Duplicate member ids: {dupes}")
        return self

    @model_validator(mode="after")
    def in_bounds(self) -> Self:
        for m in self.members:
            x_min, y_min, x_max, y_max = m.bounds
            # Allow exact-edge equality (e.g. a plate flush with x=0 or
            # x=wall_length); reject anything outside the rectangle.
            if x_min < 0 or x_max > self.wall_length:
                raise ValueError(
                    f"Member {m.id} x-extent [{x_min:.3f}, {x_max:.3f}] "
                    f"outside wall_length {self.wall_length}"
                )
            if y_min < 0 or y_max > self.wall_height:
                raise ValueError(
                    f"Member {m.id} y-extent [{y_min:.3f}, {y_max:.3f}] "
                    f"outside wall_height {self.wall_height}"
                )
        return self

    @model_validator(mode="after")
    def prereqs_resolvable(self) -> Self:
        """Every prerequisite must reference a real member id, and the
        precedence graph must be acyclic."""
        ids = {m.id for m in self.members}
        graph = {m.id: list(m.prerequisites) for m in self.members}

        for mid, prereqs in graph.items():
            for p in prereqs:
                if p not in ids:
                    raise ValueError(
                        f"Member {mid} has unknown prerequisite {p!r}"
                    )
                if p == mid:
                    raise ValueError(
                        f"Member {mid} cannot be its own prerequisite"
                    )

        # Cycle detection via depth-first search (white/gray/black).
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {mid: WHITE for mid in graph}

        def visit(node: str) -> None:
            if color[node] == GRAY:
                raise ValueError(f"Cycle in prerequisites involving {node}")
            if color[node] == BLACK:
                return
            color[node] = GRAY
            for neighbor in graph[node]:
                visit(neighbor)
            color[node] = BLACK

        for mid in graph:
            if color[mid] == WHITE:
                visit(mid)
        return self

    @model_validator(mode="after")
    def no_overlaps(self) -> Self:
        """Members may share edges (e.g. a stud's top touching a top
        plate's bottom) but their interiors must not overlap."""
        for i, a in enumerate(self.members):
            for b in self.members[i + 1:]:
                if _interiors_overlap(a, b):
                    raise ValueError(
                        f"Members {a.id} and {b.id} have overlapping footprints"
                    )
        return self


def _interiors_overlap(a: Member, b: Member) -> bool:
    """True iff the open interiors of `a` and `b` intersect (shared edges
    don't count as overlap)."""
    ax1, ay1, ax2, ay2 = a.bounds
    bx1, by1, bx2, by2 = b.bounds
    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2


# ------------------------------------------------------------------ #
# Plate prerequisite helper                                            #
# ------------------------------------------------------------------ #

def _plate_prereq_for_x(px: float, plate_segments: list[Member]) -> str:
    """Return the id of the plate segment whose x-extent contains `px`.

    A member at x position `px` (its left edge) belongs to the plate
    segment where ``segment.position[0] <= px <= segment.right_edge``.
    The last segment uses ``<=`` on both sides to correctly catch the
    rightmost king stud whose left edge coincides with the segment's
    right boundary (e.g. right jack stud at a door's right edge sits at
    the start of the right plate segment).

    Raises
    ------
    ValueError
        If no segment contains `px` — indicates a geometry bug in the
        caller (a member placed inside a door gap).
    """
    for seg in plate_segments:
        seg_left = seg.position[0]
        seg_right = seg_left + seg.size[0]
        # Use inclusive on both ends; overlapping segment boundaries are
        # disambiguated by left-to-right iteration (first match wins).
        if seg_left <= px <= seg_right:
            return seg.id
    raise ValueError(
        f"No plate segment found containing x={px:.4f}. "
        f"Segments: {[(s.id, s.position[0], s.position[0]+s.size[0]) for s in plate_segments]}"
    )


# ------------------------------------------------------------------ #
# New multi-opening generator                                          #
# ------------------------------------------------------------------ #

def generate_panel(
    wall_length: float = feet(16),
    openings: list[dict] | None = None,
    seed: int = 0,
    stud_spacing: float = DEFAULT_STUD_SPACING,
    wall_height: float = DEFAULT_WALL_HEIGHT,
) -> Panel:
    """Generate a wall panel with one or more windows and/or doors.

    Parameters
    ----------
    wall_length:
        Total wall length in canonical units. Must be ≤ ``feet(16)``.
    openings:
        List of dicts, each ``{"type": "window"|"door", "width": float}``.
        Widths are in canonical units. Defaults to a single 36" window.
        Order is the user's left-to-right intent; actual x positions are
        sampled from ``seed`` and members are named by sorted center_x order.
    seed:
        Seed for the internal NumPy RNG. Same seed + same config = same panel
        (the "RTS map seed" contract). The caller never passes center_x.
    stud_spacing:
        On-centre stud spacing (default 16").
    wall_height:
        Wall height in canonical units (default 8 ft).

    Raises
    ------
    ValueError
        If ``wall_length > feet(16)``, no openings are provided, any opening
        is too wide to fit on the wall, or the openings cannot be placed
        without overlapping each other after 200 attempts.
    """
    if openings is None:
        openings = [{"type": "window", "width": inches(36)}]

    # ---- Input validation ---- #
    if wall_length > MAX_WALL_LENGTH + 1e-9:
        raise ValueError(
            f"wall_length {wall_length:.2f} exceeds maximum {MAX_WALL_LENGTH:.2f} "
            f"(feet(16)). No plate splicing is modelled above this limit."
        )
    if len(openings) == 0:
        raise ValueError("At least one opening is required.")

    # Validate each opening fits individually
    for i, op in enumerate(openings):
        w = op["width"]
        margin = w / 2 + 2 * LUMBER_THICKNESS + feet(1)
        if margin * 2 > wall_length:
            raise ValueError(
                f"Opening {i} (width={w:.2f}) is too wide to fit on wall "
                f"(wall_length={wall_length:.2f}, required margin={margin:.2f} each side)."
            )

    # ---- Sample center_x positions with retry ---- #
    rng = np.random.default_rng(seed)
    center_xs: list[float] = []

    # Compute individual valid ranges for each opening
    ranges: list[tuple[float, float]] = []
    for op in openings:
        w = op["width"]
        margin = w / 2 + 2 * LUMBER_THICKNESS + feet(1)
        ranges.append((margin, wall_length - margin))

    sorted_openings: list[dict] = []
    for attempt in range(200):
        # Sample a center_x for each opening independently
        candidates = [
            float(rng.uniform(lo, hi)) for (lo, hi) in ranges
        ]
        # Sort openings by their sampled center_x to assign left-to-right
        # indices and validate minimum inter-opening gaps
        paired = sorted(zip(candidates, openings), key=lambda x: x[0])
        sorted_center_xs = [cx for cx, _ in paired]
        sorted_ops = [op for _, op in paired]

        # Check minimum gap between adjacent framing zones
        valid = True
        for j in range(len(sorted_ops) - 1):
            cx_left  = sorted_center_xs[j]
            w_left   = sorted_ops[j]["width"]
            cx_right = sorted_center_xs[j + 1]
            w_right  = sorted_ops[j + 1]["width"]

            # Outer right face of left opening's right king stud
            framing_right = cx_left + w_left / 2 + 2 * LUMBER_THICKNESS
            # Outer left face of right opening's left king stud
            framing_left  = cx_right - w_right / 2 - 2 * LUMBER_THICKNESS

            if framing_left - framing_right < stud_spacing - 1e-6:
                valid = False
                break

        if valid:
            center_xs = sorted_center_xs
            sorted_openings = sorted_ops
            break
    else:
        raise ValueError(
            f"Could not place {len(openings)} opening(s) on wall of length "
            f"{wall_length:.2f} without framing zones overlapping after 200 attempts. "
            f"Try fewer openings, narrower widths, or a longer wall."
        )

    # ---- Y-axis geometry (same for all openings) ---- #
    top_plate_y      = wall_height - LUMBER_THICKNESS
    header_bottom_y  = DEFAULT_HEADER_BOTTOM_Y
    header_top_y     = header_bottom_y + DEFAULT_HEADER_DEPTH
    full_stud_height = top_plate_y - LUMBER_THICKNESS

    if header_top_y >= top_plate_y:
        raise ValueError(
            f"Header top ({header_top_y}) reaches top plate bottom "
            f"({top_plate_y}); no room for top cripples."
        )

    sill_top_y   = DEFAULT_SILL_TOP_Y
    sill_bottom_y = sill_top_y - DEFAULT_SILL_THICKNESS
    if sill_bottom_y <= LUMBER_THICKNESS:
        raise ValueError(
            f"Sill bottom ({sill_bottom_y}) at or below bottom plate top "
            f"({LUMBER_THICKNESS}); no room for bottom cripples."
        )

    # ---- Build bottom plate segment(s) ---- #
    # Only DOOR openings split the bottom plate; windows do not.
    door_gaps: list[tuple[float, float]] = sorted(
        (cx - op["width"] / 2, cx + op["width"] / 2)
        for cx, op in zip(center_xs, sorted_openings)
        if op["type"] == "door"
    )

    plate_segments: list[Member] = []
    if not door_gaps:
        # Single bottom plate, no gaps.
        plate_segments.append(Member(
            id="bottom_plate",
            kind=MemberKind.BOTTOM_PLATE,
            position=(0.0, 0.0),
            size=(wall_length, LUMBER_THICKNESS),
            prerequisites=[],
        ))
    else:
        # One plate segment per gap between door openings.
        seg_left = 0.0
        for gap_idx, (door_left, door_right) in enumerate(door_gaps):
            if door_left - seg_left > 1e-6:
                plate_segments.append(Member(
                    id=f"bottom_plate_{len(plate_segments)}",
                    kind=MemberKind.BOTTOM_PLATE,
                    position=(seg_left, 0.0),
                    size=(door_left - seg_left, LUMBER_THICKNESS),
                    prerequisites=[],
                ))
            seg_left = door_right
        # Final segment after the last door (if any wall remains)
        if wall_length - seg_left > 1e-6:
            plate_segments.append(Member(
                id=f"bottom_plate_{len(plate_segments)}",
                kind=MemberKind.BOTTOM_PLATE,
                position=(seg_left, 0.0),
                size=(wall_length - seg_left, LUMBER_THICKNESS),
                prerequisites=[],
            ))

    members: list[Member] = list(plate_segments)

    # ---- Build per-opening members ---- #
    all_king_ids:        list[str] = []
    all_common_stud_ids: list[str] = []
    all_top_cripple_ids: list[str] = []

    # Exclusion zones for common studs: (framing_left, framing_right)
    exclusion_zones: list[tuple[float, float]] = []

    # Member IDs collected per opening during the build loop.  Populated
    # here so that OpeningSpec.member_ids is accurate — collecting by
    # prefix filter after the fact misses the associated plate segments.
    opening_member_ids: list[list[str]] = [[] for _ in sorted_openings]

    for i, (cx, op) in enumerate(zip(center_xs, sorted_openings)):
        prefix = f"opening_{i}_"
        opening_type: str = op["type"]
        opening_width: float = op["width"]

        opening_left_x  = cx - opening_width / 2
        opening_right_x = cx + opening_width / 2

        left_king_x  = opening_left_x  - 2 * LUMBER_THICKNESS
        right_king_x = opening_right_x + LUMBER_THICKNESS
        left_jack_x  = opening_left_x  - LUMBER_THICKNESS
        right_jack_x = opening_right_x

        # Track exclusion zone for this opening
        framing_left  = left_king_x
        framing_right = right_king_x + LUMBER_THICKNESS
        exclusion_zones.append((framing_left, framing_right))

        lk_plate = _plate_prereq_for_x(left_king_x,  plate_segments)
        rk_plate = _plate_prereq_for_x(right_king_x, plate_segments)
        lj_plate = _plate_prereq_for_x(left_jack_x,  plate_segments)
        rj_plate = _plate_prereq_for_x(right_jack_x, plate_segments)

        # Associate the plate segment(s) that support this opening's framing.
        # For windows this is always one segment; for doors it may be two
        # (left and right of the gap).  Use a set to deduplicate when both
        # sides land on the same segment.
        for plate_id in dict.fromkeys([lk_plate, rk_plate, lj_plate, rj_plate]):
            if plate_id not in opening_member_ids[i]:
                opening_member_ids[i].append(plate_id)

        # 1. King studs
        left_king_id  = f"{prefix}left_king"
        right_king_id = f"{prefix}right_king"
        members.append(Member(
            id=left_king_id,
            kind=MemberKind.KING_STUD,
            position=(left_king_x, LUMBER_THICKNESS),
            size=(LUMBER_THICKNESS, full_stud_height),
            prerequisites=[lk_plate],
        ))
        members.append(Member(
            id=right_king_id,
            kind=MemberKind.KING_STUD,
            position=(right_king_x, LUMBER_THICKNESS),
            size=(LUMBER_THICKNESS, full_stud_height),
            prerequisites=[rk_plate],
        ))
        all_king_ids.extend([left_king_id, right_king_id])
        opening_member_ids[i].extend([left_king_id, right_king_id])

        # 2. Jack studs
        jack_height   = header_bottom_y - LUMBER_THICKNESS
        left_jack_id  = f"{prefix}left_jack"
        right_jack_id = f"{prefix}right_jack"
        members.append(Member(
            id=left_jack_id,
            kind=MemberKind.JACK_STUD,
            position=(left_jack_x, LUMBER_THICKNESS),
            size=(LUMBER_THICKNESS, jack_height),
            prerequisites=[lj_plate],
        ))
        members.append(Member(
            id=right_jack_id,
            kind=MemberKind.JACK_STUD,
            position=(right_jack_x, LUMBER_THICKNESS),
            size=(LUMBER_THICKNESS, jack_height),
            prerequisites=[rj_plate],
        ))
        opening_member_ids[i].extend([left_jack_id, right_jack_id])

        # 3. Bottom cripples + sill (windows only)
        bottom_cripple_ids: list[str] = []
        if opening_type == "window":
            bot_cripple_height = sill_bottom_y - LUMBER_THICKNESS
            for j, x in enumerate(_cripple_x_positions(
                zone_left=opening_left_x,
                zone_right=opening_right_x,
                spacing=stud_spacing,
            )):
                cid = f"{prefix}bottom_cripple_{j}"
                bc_plate = _plate_prereq_for_x(x, plate_segments)
                members.append(Member(
                    id=cid,
                    kind=MemberKind.BOTTOM_CRIPPLE,
                    position=(x, LUMBER_THICKNESS),
                    size=(LUMBER_THICKNESS, bot_cripple_height),
                    prerequisites=[bc_plate],
                ))
                bottom_cripple_ids.append(cid)

            # 4. Sill plate
            sill_id = f"{prefix}sill"
            members.append(Member(
                id=sill_id,
                kind=MemberKind.SILL_PLATE,
                position=(opening_left_x, sill_bottom_y),
                size=(opening_width, DEFAULT_SILL_THICKNESS),
                prerequisites=[left_jack_id, right_jack_id] + bottom_cripple_ids,
            ))
            opening_member_ids[i].extend(bottom_cripple_ids + [sill_id])

        # 5. Header
        header_left_x = left_jack_x
        header_width  = (right_jack_x + LUMBER_THICKNESS) - left_jack_x
        header_id     = f"{prefix}header"
        members.append(Member(
            id=header_id,
            kind=MemberKind.HEADER,
            position=(header_left_x, header_bottom_y),
            size=(header_width, DEFAULT_HEADER_DEPTH),
            prerequisites=[left_jack_id, right_jack_id],
        ))
        opening_member_ids[i].append(header_id)

        # 6. Top cripples
        top_cripple_height = top_plate_y - header_top_y
        top_cripple_ids: list[str] = []
        for j, x in enumerate(_cripple_x_positions(
            zone_left=header_left_x,
            zone_right=header_left_x + header_width,
            spacing=stud_spacing,
        )):
            cid = f"{prefix}top_cripple_{j}"
            members.append(Member(
                id=cid,
                kind=MemberKind.TOP_CRIPPLE,
                position=(x, header_top_y),
                size=(LUMBER_THICKNESS, top_cripple_height),
                prerequisites=[header_id],
            ))
            top_cripple_ids.append(cid)
        all_top_cripple_ids.extend(top_cripple_ids)
        opening_member_ids[i].extend(top_cripple_ids)

    # ---- Common studs ---- #
    x_candidates = _common_stud_x_positions(wall_length, stud_spacing)
    stud_idx = 0
    for x in x_candidates:
        stud_left  = x
        stud_right = x + LUMBER_THICKNESS
        # Skip if this stud overlaps any opening's exclusion zone
        if any(
            stud_right > ez_left and stud_left < ez_right
            for (ez_left, ez_right) in exclusion_zones
        ):
            continue
        cid = f"common_stud_{stud_idx}"
        cs_plate = _plate_prereq_for_x(x, plate_segments)
        members.append(Member(
            id=cid,
            kind=MemberKind.COMMON_STUD,
            position=(x, LUMBER_THICKNESS),
            size=(LUMBER_THICKNESS, full_stud_height),
            prerequisites=[cs_plate],
        ))
        all_common_stud_ids.append(cid)
        stud_idx += 1

    # ---- Top plate ---- #
    top_plate_deps = all_king_ids + all_common_stud_ids + all_top_cripple_ids
    members.append(Member(
        id="top_plate",
        kind=MemberKind.TOP_PLATE,
        position=(0.0, top_plate_y),
        size=(wall_length, LUMBER_THICKNESS),
        prerequisites=top_plate_deps,
    ))

    # ---- OpeningSpec objects ---- #
    opening_specs: list[OpeningSpec] = []
    for i, (cx, op) in enumerate(zip(center_xs, sorted_openings)):
        opening_specs.append(OpeningSpec(
            kind=op["type"],   # type: ignore[arg-type]
            center_x=cx,
            width=op["width"],
            member_ids=opening_member_ids[i],
        ))

    return Panel(
        wall_length=wall_length,
        wall_height=wall_height,
        members=members,
        openings=opening_specs,
    )


# ------------------------------------------------------------------ #
# Internal helpers                                                     #
# ------------------------------------------------------------------ #

def _common_stud_x_positions(
    wall_length: float, spacing: float
) -> list[float]:
    """Left-edge x positions for common studs across the wall: one at
    each end (corner studs) plus interior studs at `spacing` intervals."""
    positions: list[float] = []
    x = 0.0
    while x + LUMBER_THICKNESS <= wall_length:
        positions.append(x)
        x += spacing
    # Ensure a stud flush with the right wall edge if the layout missed it.
    end_x = wall_length - LUMBER_THICKNESS
    if positions and positions[-1] < end_x - 1e-6:
        positions.append(end_x)
    return positions


def _cripple_x_positions(
    zone_left: float, zone_right: float, spacing: float
) -> list[float]:
    """Left-edge x positions for cripples within a zone, at `spacing`
    intervals from the zone's left edge. Cripples that wouldn't fit are
    dropped. The zone may end up with empty space at its right side if
    the width isn't an exact multiple of `spacing`."""
    positions: list[float] = []
    x = zone_left
    while x + LUMBER_THICKNESS <= zone_right:
        positions.append(x)
        x += spacing
    return positions
