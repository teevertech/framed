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

import random
from enum import StrEnum
from typing import Literal, Self

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


class Panel(BaseModel):
    """A wall panel: the full assembly specification."""

    wall_length: float = Field(gt=0)
    wall_height: float = Field(gt=0)
    members: list[Member] = Field(min_length=1)

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


# ----- Random panel generation -----

def generate_random_panel(
    wall_length: float | None = None,
    opening_type: Literal["window", "door"] = "window",
    opening_center_x: float | None = None,
    opening_width: float | None = None,
    stud_spacing: float = DEFAULT_STUD_SPACING,
    wall_height: float = DEFAULT_WALL_HEIGHT,
    seed: int | None = None,
) -> Panel:
    """Generate a wall panel with one window or door opening.

    All length arguments are in canonical units (see `framed.units`).
    Members come back with prerequisites pre-stamped from physical
    assembly rules:

      - Bottom plate has no prereqs (placed first).
      - Studs (common, king, jack) need the bottom plate.
      - Header needs both jack studs.
      - Sill (window) needs the jacks + the bottom cripples that
        support it.
      - Bottom cripples (window) need the bottom plate.
      - Top cripples need the header.
      - Top plate needs every full-height member underneath it
        (common studs, king studs, top cripples).

    Defaults: 8–14 ft wall, 8 ft tall, 16" OC studs, a 30–48" wide
    opening roughly centered with some jitter, window with sill at 3 ft.
    """
    rng = random.Random(seed)

    if wall_length is None:
        wall_length = feet(rng.choice([8, 10, 12, 14]))
    if opening_width is None:
        opening_width = inches(rng.choice([30, 36, 42, 48]))
    if opening_center_x is None:
        # Keep the opening framing (king studs included) clear of the wall
        # ends with at least 1 ft of margin to a wall edge.
        margin = opening_width / 2 + 2 * LUMBER_THICKNESS + feet(1)
        if margin * 2 > wall_length:
            raise ValueError(
                f"Wall too short ({wall_length}) for opening width "
                f"{opening_width} with edge margin"
            )
        opening_center_x = rng.uniform(margin, wall_length - margin)

    # Opening x bounds = inside-face-to-inside-face between jack studs
    # (the rough opening width).
    opening_left_x = opening_center_x - opening_width / 2
    opening_right_x = opening_center_x + opening_width / 2

    # y heights
    top_plate_y = wall_height - LUMBER_THICKNESS
    header_bottom_y = DEFAULT_HEADER_BOTTOM_Y
    header_top_y = header_bottom_y + DEFAULT_HEADER_DEPTH
    if header_top_y >= top_plate_y:
        raise ValueError(
            f"Header top ({header_top_y}) reaches top plate bottom "
            f"({top_plate_y}); no room for top cripples"
        )

    if opening_type == "window":
        sill_top_y = DEFAULT_SILL_TOP_Y
        sill_bottom_y = sill_top_y - DEFAULT_SILL_THICKNESS
        if sill_bottom_y <= LUMBER_THICKNESS:
            raise ValueError(
                f"Sill bottom ({sill_bottom_y}) at or below bottom plate "
                f"top ({LUMBER_THICKNESS}); no room for bottom cripples"
            )
    else:
        sill_top_y = None
        sill_bottom_y = None

    members: list[Member] = []

    # 1. Bottom plate (no prereqs).
    members.append(Member(
        id="bottom_plate",
        kind=MemberKind.BOTTOM_PLATE,
        position=(0.0, 0.0),
        size=(wall_length, LUMBER_THICKNESS),
        prerequisites=[],
    ))

    # 2. King studs: full height, outside the jacks.
    full_stud_height = top_plate_y - LUMBER_THICKNESS
    left_king_x = opening_left_x - 2 * LUMBER_THICKNESS
    right_king_x = opening_right_x + LUMBER_THICKNESS
    members.append(Member(
        id="left_king",
        kind=MemberKind.KING_STUD,
        position=(left_king_x, LUMBER_THICKNESS),
        size=(LUMBER_THICKNESS, full_stud_height),
        prerequisites=["bottom_plate"],
    ))
    members.append(Member(
        id="right_king",
        kind=MemberKind.KING_STUD,
        position=(right_king_x, LUMBER_THICKNESS),
        size=(LUMBER_THICKNESS, full_stud_height),
        prerequisites=["bottom_plate"],
    ))

    # 3. Jack studs: inside the kings, support the header.
    jack_height = header_bottom_y - LUMBER_THICKNESS
    left_jack_x = opening_left_x - LUMBER_THICKNESS
    right_jack_x = opening_right_x
    members.append(Member(
        id="left_jack",
        kind=MemberKind.JACK_STUD,
        position=(left_jack_x, LUMBER_THICKNESS),
        size=(LUMBER_THICKNESS, jack_height),
        prerequisites=["bottom_plate"],
    ))
    members.append(Member(
        id="right_jack",
        kind=MemberKind.JACK_STUD,
        position=(right_jack_x, LUMBER_THICKNESS),
        size=(LUMBER_THICKNESS, jack_height),
        prerequisites=["bottom_plate"],
    ))

    # 4. Common studs: at stud_spacing intervals along the wall, skipping
    # any whose footprint would collide with the opening framing region
    # (the four studs from outside of left king to outside of right king).
    opening_framing_left = left_king_x
    opening_framing_right = right_king_x + LUMBER_THICKNESS
    common_stud_ids: list[str] = []
    x_candidates = _common_stud_x_positions(wall_length, stud_spacing)
    for i, x in enumerate(x_candidates):
        stud_left = x
        stud_right = x + LUMBER_THICKNESS
        if stud_right > opening_framing_left and stud_left < opening_framing_right:
            # would overlap opening framing — skip
            continue
        cid = f"common_stud_{i}"
        members.append(Member(
            id=cid,
            kind=MemberKind.COMMON_STUD,
            position=(x, LUMBER_THICKNESS),
            size=(LUMBER_THICKNESS, full_stud_height),
            prerequisites=["bottom_plate"],
        ))
        common_stud_ids.append(cid)

    # 5. Bottom cripples (window only): support the sill from below.
    bottom_cripple_ids: list[str] = []
    if opening_type == "window":
        assert sill_bottom_y is not None
        bot_cripple_height = sill_bottom_y - LUMBER_THICKNESS
        for i, x in enumerate(_cripple_x_positions(
            zone_left=opening_left_x,
            zone_right=opening_right_x,
            spacing=stud_spacing,
        )):
            cid = f"bottom_cripple_{i}"
            members.append(Member(
                id=cid,
                kind=MemberKind.BOTTOM_CRIPPLE,
                position=(x, LUMBER_THICKNESS),
                size=(LUMBER_THICKNESS, bot_cripple_height),
                prerequisites=["bottom_plate"],
            ))
            bottom_cripple_ids.append(cid)

    # 6. Sill plate (window only): rests between jacks, on top of bottom
    # cripples.
    if opening_type == "window":
        assert sill_top_y is not None and sill_bottom_y is not None
        members.append(Member(
            id="sill",
            kind=MemberKind.SILL_PLATE,
            position=(opening_left_x, sill_bottom_y),
            size=(opening_width, DEFAULT_SILL_THICKNESS),
            prerequisites=["left_jack", "right_jack"] + bottom_cripple_ids,
        ))

    # 7. Header: rests on the jacks, spans between the kings.
    header_left_x = left_jack_x
    header_width = (right_jack_x + LUMBER_THICKNESS) - left_jack_x
    members.append(Member(
        id="header",
        kind=MemberKind.HEADER,
        position=(header_left_x, header_bottom_y),
        size=(header_width, DEFAULT_HEADER_DEPTH),
        prerequisites=["left_jack", "right_jack"],
    ))

    # 8. Top cripples: rest on the header, support the top plate.
    top_cripple_ids: list[str] = []
    top_cripple_height = top_plate_y - header_top_y
    for i, x in enumerate(_cripple_x_positions(
        zone_left=header_left_x,
        zone_right=header_left_x + header_width,
        spacing=stud_spacing,
    )):
        cid = f"top_cripple_{i}"
        members.append(Member(
            id=cid,
            kind=MemberKind.TOP_CRIPPLE,
            position=(x, header_top_y),
            size=(LUMBER_THICKNESS, top_cripple_height),
            prerequisites=["header"],
        ))
        top_cripple_ids.append(cid)

    # 9. Top plate: every full-height member that touches its bottom is
    # a prerequisite.
    top_plate_deps = (
        ["left_king", "right_king"] + common_stud_ids + top_cripple_ids
    )
    members.append(Member(
        id="top_plate",
        kind=MemberKind.TOP_PLATE,
        position=(0.0, top_plate_y),
        size=(wall_length, LUMBER_THICKNESS),
        prerequisites=top_plate_deps,
    ))

    return Panel(
        wall_length=wall_length,
        wall_height=wall_height,
        members=members,
    )


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
