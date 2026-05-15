"""Tests for the panel data model and random generator.

These pin down the invariants `Panel` enforces (unique ids, in-bounds,
acyclic prereqs, no overlaps), plus the behavior of `generate_random_panel`
(deterministic by seed, all members valid, prereqs form a complete DAG
that orders every member).
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
    Panel,
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


# ----- Generator -----

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


# ----- Helpers -----

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
