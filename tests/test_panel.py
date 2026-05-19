"""Tests for the panel data model and random generator.

These pin down the invariants ``Panel`` enforces (unique ids, in-bounds,
acyclic prereqs, no overlaps), plus the behaviour of ``generate_random_panel``
(deterministic by seed, all members valid, prereqs form a complete DAG
that orders every member).

``TestGeneratePanel`` covers the new ``generate_panel`` function: multi-opening
support, namespaced member IDs (``opening_i_*``), split bottom plates for
doors, ``OpeningSpec`` metadata, and the wall-length / placement-fit guards.
The existing ``generate_random_panel`` tests are preserved unchanged — that
function's API and output format are frozen for backwards compatibility.
"""
from __future__ import annotations

from collections import defaultdict, deque

import pytest
from pydantic import ValidationError

from framed.panel import (
    DEFAULT_HEADER_BOTTOM_Y,
    DEFAULT_HEADER_DEPTH,
    DEFAULT_SILL_TOP_Y,
    DEFAULT_STUD_SPACING,
    DEFAULT_WALL_HEIGHT,
    LUMBER_THICKNESS,
    MEMBER_CATEGORY,
    Member,
    MemberCategory,
    MemberKind,
    OpeningSpec,
    Panel,
    generate_panel,
    generate_random_panel,
)
from framed.units import feet, inches


# ----- Member validation -----

class TestMember:
    def test_minimal_valid_member(self) -> None:
        m = Member(
            id="m1",
            kind=MemberKind.COMMON_STUD,
            position=(0.0, 0.0),
            size=(1.5, 90.0),
        )
        assert m.id == "m1"
        assert m.prerequisites == []
        assert m.category == MemberCategory.COMMON_STUD

    def test_bounds_property(self) -> None:
        m = Member(
            id="m1",
            kind=MemberKind.TOP_PLATE,
            position=(10.0, 20.0),
            size=(120.0, 1.5),
        )
        assert m.bounds == (10.0, 20.0, 130.0, 21.5)

    def test_center_property(self) -> None:
        m = Member(
            id="m1",
            kind=MemberKind.TOP_PLATE,
            position=(10.0, 20.0),
            size=(120.0, 1.5),
        )
        assert m.center == (70.0, 20.75)

    @pytest.mark.parametrize("bad_size", [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)])
    def test_non_positive_size_rejected(self, bad_size: tuple[float, float]) -> None:
        with pytest.raises(ValidationError, match="non-positive size"):
            Member(
                id="m1",
                kind=MemberKind.COMMON_STUD,
                position=(0.0, 0.0),
                size=bad_size,
            )

    def test_every_kind_has_category(self) -> None:
        for kind in MemberKind:
            assert kind in MEMBER_CATEGORY, f"{kind} missing category"


# ----- Panel validation -----

def _make_member(
    id: str,
    position: tuple[float, float] = (0.0, 0.0),
    size: tuple[float, float] = (1.5, 90.0),
    kind: MemberKind = MemberKind.COMMON_STUD,
    prerequisites: list[str] | None = None,
) -> Member:
    return Member(
        id=id,
        kind=kind,
        position=position,
        size=size,
        prerequisites=prerequisites or [],
    )


class TestPanelValidation:
    def test_minimal_valid_panel(self) -> None:
        panel = Panel(
            wall_length=feet(8),
            wall_height=feet(8),
            members=[_make_member("m1", position=(0.0, 0.0), size=(96.0, 1.5))],
        )
        assert len(panel.members) == 1

    def test_duplicate_ids_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate member ids"):
            Panel(
                wall_length=feet(8),
                wall_height=feet(8),
                members=[
                    _make_member("m1", position=(0.0, 0.0)),
                    _make_member("m1", position=(20.0, 0.0)),
                ],
            )

    def test_out_of_bounds_x_rejected(self) -> None:
        with pytest.raises(ValidationError, match="outside wall_length"):
            Panel(
                wall_length=feet(8),
                wall_height=feet(8),
                members=[_make_member("m1", position=(100.0, 0.0), size=(50.0, 1.5))],
            )

    def test_out_of_bounds_y_rejected(self) -> None:
        with pytest.raises(ValidationError, match="outside wall"):
            Panel(
                wall_length=feet(8),
                wall_height=feet(8),
                members=[_make_member("m1", position=(0.0, 90.0), size=(1.5, 20.0))],
            )

    def test_unknown_prereq_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unknown prerequisite"):
            Panel(
                wall_length=feet(8),
                wall_height=feet(8),
                members=[
                    _make_member("m1", prerequisites=["nonexistent"]),
                ],
            )

    def test_self_prereq_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cannot be its own"):
            Panel(
                wall_length=feet(8),
                wall_height=feet(8),
                members=[_make_member("m1", prerequisites=["m1"])],
            )

    def test_cycle_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Cycle in prerequisites"):
            Panel(
                wall_length=feet(8),
                wall_height=feet(8),
                members=[
                    _make_member("a", position=(0.0, 0.0), prerequisites=["b"]),
                    _make_member("b", position=(20.0, 0.0), prerequisites=["c"]),
                    _make_member("c", position=(40.0, 0.0), prerequisites=["a"]),
                ],
            )

    def test_overlapping_members_rejected(self) -> None:
        with pytest.raises(ValidationError, match="overlapping footprints"):
            Panel(
                wall_length=feet(8),
                wall_height=feet(8),
                members=[
                    _make_member("a", position=(0.0, 0.0), size=(10.0, 10.0)),
                    _make_member("b", position=(5.0, 5.0), size=(10.0, 10.0)),
                ],
            )

    def test_edge_touching_members_allowed(self) -> None:
        """A stud's top edge touching a plate's bottom edge is fine; only
        interior overlap is rejected."""
        panel = Panel(
            wall_length=feet(8),
            wall_height=feet(8),
            members=[
                _make_member(
                    "stud",
                    kind=MemberKind.COMMON_STUD,
                    position=(0.0, 1.5),
                    size=(1.5, 93.0),  # top at y=94.5
                ),
                _make_member(
                    "top_plate",
                    kind=MemberKind.TOP_PLATE,
                    position=(0.0, 94.5),  # bottom at y=94.5 — flush
                    size=(96.0, 1.5),
                    prerequisites=["stud"],
                ),
                _make_member(
                    "bottom_plate",
                    kind=MemberKind.BOTTOM_PLATE,
                    position=(0.0, 0.0),
                    size=(96.0, 1.5),
                ),
            ],
        )
        assert len(panel.members) == 3


# ===================================================================== #
# generate_random_panel (legacy function — frozen API)                  #
# ===================================================================== #

class TestGenerator:
    def test_default_generation_produces_valid_panel(self) -> None:
        panel = generate_random_panel(seed=0)
        # Panel construction itself validates; reaching here is the test.
        assert isinstance(panel, Panel)
        assert panel.wall_length > 0
        assert panel.wall_height > 0
        assert len(panel.members) >= 6  # at minimum: 2 plates + 2 kings + 2 jacks

    def test_same_seed_produces_same_panel(self) -> None:
        a = generate_random_panel(seed=42)
        b = generate_random_panel(seed=42)
        assert a.wall_length == b.wall_length
        assert len(a.members) == len(b.members)
        for ma, mb in zip(a.members, b.members):
            assert ma.id == mb.id
            assert ma.position == mb.position
            assert ma.size == mb.size
            assert ma.prerequisites == mb.prerequisites

    def test_different_seeds_produce_different_panels(self) -> None:
        seeds_seen: set[tuple[float, int]] = set()
        for s in range(10):
            p = generate_random_panel(seed=s)
            seeds_seen.add((p.wall_length, len(p.members)))
        # Across 10 seeds we should see variation
        assert len(seeds_seen) > 1

    def test_window_has_sill_and_bottom_cripples(self) -> None:
        panel = generate_random_panel(opening_type="window", seed=0)
        kinds = {m.kind for m in panel.members}
        assert MemberKind.SILL_PLATE in kinds
        assert MemberKind.BOTTOM_CRIPPLE in kinds

    def test_door_has_no_sill_or_bottom_cripples(self) -> None:
        panel = generate_random_panel(opening_type="door", seed=0)
        kinds = {m.kind for m in panel.members}
        assert MemberKind.SILL_PLATE not in kinds
        assert MemberKind.BOTTOM_CRIPPLE not in kinds

    def test_every_panel_has_one_top_and_bottom_plate(self) -> None:
        panel = generate_random_panel(seed=0)
        n_top = sum(1 for m in panel.members if m.kind == MemberKind.TOP_PLATE)
        n_bot = sum(1 for m in panel.members if m.kind == MemberKind.BOTTOM_PLATE)
        assert n_top == 1
        assert n_bot == 1

    def test_kings_and_jacks_come_in_pairs(self) -> None:
        panel = generate_random_panel(seed=0)
        n_king = sum(1 for m in panel.members if m.kind == MemberKind.KING_STUD)
        n_jack = sum(1 for m in panel.members if m.kind == MemberKind.JACK_STUD)
        assert n_king == 2
        assert n_jack == 2

    def test_explicit_parameters(self) -> None:
        panel = generate_random_panel(
            wall_length=feet(10),
            opening_type="window",
            opening_center_x=feet(5),
            opening_width=inches(36),
            seed=0,
        )
        assert panel.wall_length == feet(10)
        # Header should span 36" + 2 jack thicknesses
        header = next(m for m in panel.members if m.kind == MemberKind.HEADER)
        assert header.size[0] == pytest.approx(inches(36) + 2 * LUMBER_THICKNESS)

    def test_prereqs_form_complete_topological_order(self) -> None:
        """Every member must be placeable in some order respecting prereqs,
        and that order must include all members. This is the basic property
        the env relies on."""
        for seed in range(20):
            panel = generate_random_panel(seed=seed)
            order = _topological_order(panel)
            assert len(order) == len(panel.members), (
                f"Seed {seed}: only {len(order)}/{len(panel.members)} members "
                f"orderable — prereqs incomplete or cyclic"
            )

    def test_top_plate_depends_on_all_full_height_members(self) -> None:
        panel = generate_random_panel(seed=0)
        top_plate = next(m for m in panel.members if m.kind == MemberKind.TOP_PLATE)
        prereq_set = set(top_plate.prerequisites)

        # Every common stud, king stud, and top cripple must be a prereq
        for m in panel.members:
            if m.kind in (
                MemberKind.COMMON_STUD,
                MemberKind.KING_STUD,
                MemberKind.TOP_CRIPPLE,
            ):
                assert m.id in prereq_set, (
                    f"top_plate missing prereq on {m.kind.value} {m.id}"
                )

    def test_header_depends_on_both_jacks(self) -> None:
        panel = generate_random_panel(seed=0)
        header = next(m for m in panel.members if m.kind == MemberKind.HEADER)
        assert "left_jack" in header.prerequisites
        assert "right_jack" in header.prerequisites

    def test_sill_depends_on_jacks_and_bottom_cripples(self) -> None:
        panel = generate_random_panel(opening_type="window", seed=0)
        sill = next(m for m in panel.members if m.kind == MemberKind.SILL_PLATE)
        assert "left_jack" in sill.prerequisites
        assert "right_jack" in sill.prerequisites
        bottom_cripple_ids = [
            m.id for m in panel.members if m.kind == MemberKind.BOTTOM_CRIPPLE
        ]
        for bid in bottom_cripple_ids:
            assert bid in sill.prerequisites

    def test_top_cripples_depend_on_header(self) -> None:
        panel = generate_random_panel(seed=0)
        for m in panel.members:
            if m.kind == MemberKind.TOP_CRIPPLE:
                assert "header" in m.prerequisites

    @pytest.mark.parametrize("opening_type", ["window", "door"])
    def test_robust_across_many_seeds(self, opening_type: str) -> None:
        """Smoke test: generator should not raise across many seeds and
        the resulting panels should validate."""
        for seed in range(50):
            panel = generate_random_panel(opening_type=opening_type, seed=seed)  # type: ignore[arg-type]
            assert len(panel.members) >= 6


# ===================================================================== #
# generate_panel (new multi-opening function)                           #
# ===================================================================== #

class TestGeneratePanel:
    """Tests for ``generate_panel`` — multi-opening support with namespaced
    member IDs, split bottom plates for doors, and ``OpeningSpec`` metadata."""

    # ------------------------------------------------------------------ #
    # Basics                                                               #
    # ------------------------------------------------------------------ #

    def test_produces_valid_panel(self) -> None:
        """Panel construction validates all invariants; reaching here is the test."""
        panel = generate_panel(
            wall_length=feet(12),
            openings=[{"type": "window", "width": inches(36)}],
            seed=0,
        )
        assert isinstance(panel, Panel)
        assert len(panel.members) >= 6

    def test_same_seed_produces_same_panel(self) -> None:
        kwargs = dict(
            wall_length=feet(12),
            openings=[{"type": "window", "width": inches(36)}],
        )
        a = generate_panel(**kwargs, seed=7)
        b = generate_panel(**kwargs, seed=7)
        assert len(a.members) == len(b.members)
        for ma, mb in zip(a.members, b.members):
            assert ma.id == mb.id
            assert ma.position == mb.position
            assert ma.size == mb.size
            assert ma.prerequisites == mb.prerequisites

    def test_different_seeds_produce_different_center_positions(self) -> None:
        """The seed controls opening placement, so different seeds must yield
        different header positions (proxy for opening center_x)."""
        kwargs = dict(
            wall_length=feet(12),
            openings=[{"type": "window", "width": inches(36)}],
        )
        cx_values = set()
        for s in range(10):
            panel = generate_panel(**kwargs, seed=s)
            header = next(m for m in panel.members if m.id == "opening_0_header")
            cx_values.add(header.center[0])
        assert len(cx_values) > 1

    # ------------------------------------------------------------------ #
    # Single window — member IDs and structure                            #
    # ------------------------------------------------------------------ #

    def test_single_window_namespaced_ids(self) -> None:
        panel = generate_panel(
            wall_length=feet(12),
            openings=[{"type": "window", "width": inches(36)}],
            seed=0,
        )
        ids = {m.id for m in panel.members}
        assert "opening_0_left_king"  in ids
        assert "opening_0_right_king" in ids
        assert "opening_0_left_jack"  in ids
        assert "opening_0_right_jack" in ids
        assert "opening_0_header"     in ids
        assert "opening_0_sill"       in ids
        assert "top_plate"            in ids

    def test_single_window_single_bottom_plate(self) -> None:
        """Windows do not split the bottom plate."""
        panel = generate_panel(
            wall_length=feet(12),
            openings=[{"type": "window", "width": inches(36)}],
            seed=0,
        )
        ids = {m.id for m in panel.members}
        assert "bottom_plate"   in ids
        assert "bottom_plate_0" not in ids

    def test_single_window_has_sill_and_bottom_cripples(self) -> None:
        panel = generate_panel(
            wall_length=feet(12),
            openings=[{"type": "window", "width": inches(36)}],
            seed=0,
        )
        kinds = {m.kind for m in panel.members}
        assert MemberKind.SILL_PLATE    in kinds
        assert MemberKind.BOTTOM_CRIPPLE in kinds

    def test_single_window_header_prereqs(self) -> None:
        panel = generate_panel(
            wall_length=feet(12),
            openings=[{"type": "window", "width": inches(36)}],
            seed=0,
        )
        header = next(m for m in panel.members if m.id == "opening_0_header")
        assert "opening_0_left_jack"  in header.prerequisites
        assert "opening_0_right_jack" in header.prerequisites

    def test_single_window_sill_prereqs(self) -> None:
        panel = generate_panel(
            wall_length=feet(12),
            openings=[{"type": "window", "width": inches(36)}],
            seed=0,
        )
        sill = next(m for m in panel.members if m.id == "opening_0_sill")
        assert "opening_0_left_jack"  in sill.prerequisites
        assert "opening_0_right_jack" in sill.prerequisites
        bottom_cripple_ids = [
            m.id for m in panel.members if m.kind == MemberKind.BOTTOM_CRIPPLE
        ]
        assert bottom_cripple_ids, "expected at least one bottom cripple"
        for bid in bottom_cripple_ids:
            assert bid in sill.prerequisites

    def test_single_window_top_cripples_prereq_header(self) -> None:
        panel = generate_panel(
            wall_length=feet(12),
            openings=[{"type": "window", "width": inches(36)}],
            seed=0,
        )
        for m in panel.members:
            if m.kind == MemberKind.TOP_CRIPPLE:
                assert "opening_0_header" in m.prerequisites

    # ------------------------------------------------------------------ #
    # Single door — split bottom plate, no sill/cripples                 #
    # ------------------------------------------------------------------ #

    def test_single_door_split_bottom_plate(self) -> None:
        """A door opening splits the bottom plate into two named segments."""
        panel = generate_panel(
            wall_length=feet(12),
            openings=[{"type": "door", "width": inches(36)}],
            seed=0,
        )
        ids = {m.id for m in panel.members}
        assert "bottom_plate_0" in ids
        assert "bottom_plate_1" in ids
        assert "bottom_plate"   not in ids

    def test_single_door_no_sill_or_bottom_cripples(self) -> None:
        panel = generate_panel(
            wall_length=feet(12),
            openings=[{"type": "door", "width": inches(36)}],
            seed=0,
        )
        kinds = {m.kind for m in panel.members}
        assert MemberKind.SILL_PLATE     not in kinds
        assert MemberKind.BOTTOM_CRIPPLE not in kinds

    def test_single_door_bottom_plates_have_no_prereqs(self) -> None:
        panel = generate_panel(
            wall_length=feet(12),
            openings=[{"type": "door", "width": inches(36)}],
            seed=0,
        )
        for m in panel.members:
            if m.kind == MemberKind.BOTTOM_PLATE:
                assert m.prerequisites == [], (
                    f"{m.id} should have no prerequisites (all bottom plates are roots)"
                )

    def test_single_door_namespaced_ids(self) -> None:
        panel = generate_panel(
            wall_length=feet(12),
            openings=[{"type": "door", "width": inches(36)}],
            seed=0,
        )
        ids = {m.id for m in panel.members}
        assert "opening_0_left_king"  in ids
        assert "opening_0_right_king" in ids
        assert "opening_0_left_jack"  in ids
        assert "opening_0_right_jack" in ids
        assert "opening_0_header"     in ids

    # ------------------------------------------------------------------ #
    # Two windows — both openings namespaced                              #
    # ------------------------------------------------------------------ #

    def test_two_windows_namespaced_ids(self) -> None:
        panel = generate_panel(
            wall_length=feet(16),
            openings=[
                {"type": "window", "width": inches(24)},
                {"type": "window", "width": inches(24)},
            ],
            seed=0,
        )
        ids = {m.id for m in panel.members}
        for prefix in ("opening_0", "opening_1"):
            assert f"{prefix}_left_king"  in ids
            assert f"{prefix}_right_king" in ids
            assert f"{prefix}_left_jack"  in ids
            assert f"{prefix}_right_jack" in ids
            assert f"{prefix}_header"     in ids
            assert f"{prefix}_sill"       in ids

    def test_two_windows_single_bottom_plate(self) -> None:
        """Two windows — still no door, so bottom plate is not split."""
        panel = generate_panel(
            wall_length=feet(16),
            openings=[
                {"type": "window", "width": inches(24)},
                {"type": "window", "width": inches(24)},
            ],
            seed=0,
        )
        ids = {m.id for m in panel.members}
        assert "bottom_plate"   in ids
        assert "bottom_plate_0" not in ids

    def test_two_windows_top_plate_prereqs_cover_all_full_height_members(self) -> None:
        """Top plate depends on all king studs, common studs, and top cripples
        across both openings."""
        panel = generate_panel(
            wall_length=feet(16),
            openings=[
                {"type": "window", "width": inches(24)},
                {"type": "window", "width": inches(24)},
            ],
            seed=0,
        )
        top_plate = next(m for m in panel.members if m.id == "top_plate")
        prereq_set = set(top_plate.prerequisites)
        for m in panel.members:
            if m.kind in (MemberKind.KING_STUD, MemberKind.COMMON_STUD, MemberKind.TOP_CRIPPLE):
                assert m.id in prereq_set, (
                    f"top_plate missing prereq on {m.kind.value} '{m.id}'"
                )

    # ------------------------------------------------------------------ #
    # Mixed window + door                                                 #
    # ------------------------------------------------------------------ #

    def test_mixed_window_and_door_splits_bottom_plate(self) -> None:
        """The door splits the bottom plate; the window does not add more splits."""
        panel = generate_panel(
            wall_length=feet(16),
            openings=[
                {"type": "window", "width": inches(36)},
                {"type": "door",   "width": inches(32)},
            ],
            seed=0,
        )
        ids = {m.id for m in panel.members}
        assert "bottom_plate_0" in ids
        assert "bottom_plate_1" in ids
        assert "bottom_plate"   not in ids

    def test_mixed_window_and_door_window_has_sill(self) -> None:
        panel = generate_panel(
            wall_length=feet(16),
            openings=[
                {"type": "window", "width": inches(36)},
                {"type": "door",   "width": inches(32)},
            ],
            seed=0,
        )
        kinds = {m.kind for m in panel.members}
        assert MemberKind.SILL_PLATE in kinds

    # ------------------------------------------------------------------ #
    # Validation guards                                                    #
    # ------------------------------------------------------------------ #

    def test_wall_too_long_raises(self) -> None:
        with pytest.raises(ValueError, match="16"):
            generate_panel(
                wall_length=feet(16) + 0.1,
                openings=[{"type": "window", "width": inches(36)}],
                seed=0,
            )

    def test_opening_too_wide_for_wall_raises(self) -> None:
        """A 90" window cannot fit on an 8-foot wall with the required margins."""
        with pytest.raises(ValueError):
            generate_panel(
                wall_length=feet(8),
                openings=[{"type": "window", "width": inches(90)}],
                seed=0,
            )

    def test_no_openings_raises(self) -> None:
        """At least one opening is required."""
        with pytest.raises(ValueError):
            generate_panel(
                wall_length=feet(12),
                openings=[],
                seed=0,
            )

    # ------------------------------------------------------------------ #
    # OpeningSpec metadata                                                 #
    # ------------------------------------------------------------------ #

    def test_single_window_opening_spec(self) -> None:
        panel = generate_panel(
            wall_length=feet(12),
            openings=[{"type": "window", "width": inches(36)}],
            seed=0,
        )
        assert len(panel.openings) == 1
        spec = panel.openings[0]
        assert isinstance(spec, OpeningSpec)
        assert spec.kind == "window"
        assert spec.width == pytest.approx(inches(36))
        # All member_ids in the spec should actually be in the panel.
        panel_ids = {m.id for m in panel.members}
        for mid in spec.member_ids:
            assert mid in panel_ids, f"OpeningSpec references unknown id '{mid}'"

    def test_two_openings_spec_sorted_by_center_x(self) -> None:
        """OpeningSpec entries are ordered by center_x (left to right)."""
        panel = generate_panel(
            wall_length=feet(16),
            openings=[
                {"type": "window", "width": inches(24)},
                {"type": "door",   "width": inches(32)},
            ],
            seed=5,
        )
        assert len(panel.openings) == 2
        assert panel.openings[0].center_x < panel.openings[1].center_x

    def test_opening_spec_member_ids_prefixed(self) -> None:
        """All member IDs in each OpeningSpec should carry the matching prefix."""
        panel = generate_panel(
            wall_length=feet(16),
            openings=[
                {"type": "window", "width": inches(24)},
                {"type": "window", "width": inches(24)},
            ],
            seed=0,
        )
        for i, spec in enumerate(panel.openings):
            for mid in spec.member_ids:
                assert mid.startswith(f"opening_{i}_"), (
                    f"OpeningSpec[{i}] contains id '{mid}' without expected prefix"
                )

    # ------------------------------------------------------------------ #
    # Topological completeness                                             #
    # ------------------------------------------------------------------ #

    def test_prereqs_form_complete_topological_order_single_window(self) -> None:
        for seed in range(10):
            panel = generate_panel(
                wall_length=feet(12),
                openings=[{"type": "window", "width": inches(36)}],
                seed=seed,
            )
            order = _topological_order(panel)
            assert len(order) == len(panel.members), (
                f"Seed {seed}: only {len(order)}/{len(panel.members)} orderable"
            )

    def test_prereqs_form_complete_topological_order_mixed(self) -> None:
        for seed in range(10):
            panel = generate_panel(
                wall_length=feet(16),
                openings=[
                    {"type": "window", "width": inches(24)},
                    {"type": "door",   "width": inches(32)},
                ],
                seed=seed,
            )
            order = _topological_order(panel)
            assert len(order) == len(panel.members), (
                f"Seed {seed}: only {len(order)}/{len(panel.members)} orderable"
            )

    # ------------------------------------------------------------------ #
    # Smoke test                                                           #
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize("seed", list(range(20)))
    def test_robust_across_many_seeds_single_window(self, seed: int) -> None:
        panel = generate_panel(
            wall_length=feet(12),
            openings=[{"type": "window", "width": inches(36)}],
            seed=seed,
        )
        assert isinstance(panel, Panel)
        assert len(panel.members) >= 6


# ===================================================================== #
# Helpers                                                               #
# ===================================================================== #

def _topological_order(panel: Panel) -> list[str]:
    """Kahn's algorithm. Returns ids in a valid placement order, or a
    shorter list if a cycle prevents complete ordering (shouldn't happen
    for a valid Panel, but the test is defensive)."""
    indegree: dict[str, int] = {m.id: len(m.prerequisites) for m in panel.members}
    dependents: dict[str, list[str]] = defaultdict(list)
    for m in panel.members:
        for p in m.prerequisites:
            dependents[p].append(m.id)

    queue: deque[str] = deque(mid for mid, n in indegree.items() if n == 0)
    order: list[str] = []
    while queue:
        mid = queue.popleft()
        order.append(mid)
        for dep in dependents[mid]:
            indegree[dep] -= 1
            if indegree[dep] == 0:
                queue.append(dep)
    return order
