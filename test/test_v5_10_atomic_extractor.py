from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from graphmem.build.atomic_extractor import (
    adaptive_fact_cap,
    scan_information_units,
    sentence_chunks,
)
from graphmem.build.semantic import QwenSemanticDistiller, strict_scene_schema
from graphmem.config import GraphMemV5Config


def _turn(text: str, turn_id: str = "turn:1") -> SimpleNamespace:
    return SimpleNamespace(
        turn_id=turn_id, raw_text=text, speaker="Alice",
        timestamp="2025-01-05T12:00:00Z")


def _scene(text: str) -> SimpleNamespace:
    return SimpleNamespace(scene_id="scene:1", summary="fallback summary", turns=[_turn(text)])


def _atomic_distiller(*, chunk_chars: int = 40) -> QwenSemanticDistiller:
    base = GraphMemV5Config()
    config = replace(base, models=replace(
        base.models,
        semantic_extraction_mode="strict_single",
        semantic_atomic_coverage=True,
        semantic_adaptive_fact_cap=True,
        semantic_max_facts_per_scene=4,
        semantic_adaptive_fact_cap_max=24,
        semantic_sentence_chunking=True,
        semantic_turn_input_chars=chunk_chars,
    ))
    distiller = QwenSemanticDistiller.__new__(QwenSemanticDistiller)
    distiller.config = config
    return distiller


def test_information_unit_scan_preserves_fragile_semantics() -> None:
    units = scan_information_units([_turn(
        "Alice did not buy 3 books in Paris last Friday because she plans to move next year.")])
    kinds = {unit.kind for unit in units}

    assert {"negation", "number_unit", "entity", "date", "modality", "state_change"} <= kinds
    assert len({unit.unit_id for unit in units}) == len(units)
    assert all(unit.text == _turn(
        "Alice did not buy 3 books in Paris last Friday because she plans to move next year."
    ).raw_text[unit.start:unit.end] for unit in units)


def test_sentence_chunking_is_lossless_and_keeps_middle_content() -> None:
    text = "First fact. " + "middle evidence " * 12 + "Final fact."
    chunks = sentence_chunks("turn:1", text, 40)

    assert len(chunks) > 2
    assert "".join(chunk.text for chunk in chunks) == text
    assert any("middle evidence" in chunk.text for chunk in chunks[1:-1])
    assert [(chunk.start, chunk.end) for chunk in chunks] == [
        (0 if index == 0 else chunks[index - 1].end, chunk.end)
        for index, chunk in enumerate(chunks)
    ]


def test_adaptive_fact_cap_follows_the_declared_formula() -> None:
    units = scan_information_units([_turn(
        "Bob bought 2 bikes in London on 2025-01-03 and plans to sell one next month.")])
    cap = adaptive_fact_cap(
        units, floor=4, ceiling=24, alpha=0.5, beta=0.25, gamma=0.5)
    entities = sum(unit.kind == "entity" for unit in units)
    temporal = sum(unit.kind in {"date", "duration"} for unit in units)
    import math
    assert cap == min(24, max(4, math.ceil(0.5 * len(units) + 0.25 * entities + 0.5 * temporal)))


def test_atomic_schema_requires_confidence_unit_links_and_rejections() -> None:
    schema = strict_scene_schema(
        1, 8, atomic_coverage=True, max_information_units=7, max_turn_index=4)
    scene = schema["properties"]["s"]["items"]
    fact = scene["properties"]["f"]["items"]

    assert {"c", "z"} <= set(fact["required"])
    assert fact["properties"]["c"] == {
        "type": "number", "minimum": 0.0, "maximum": 1.0}
    assert fact["properties"]["r"]["items"]["maximum"] == 4
    assert "u" in scene["required"]
    assert scene["properties"]["u"]["maxItems"] == 7


def test_atomic_payload_chunks_without_dropping_units() -> None:
    text = (
        "Alice bought 3 books in Paris last Friday. "
        "The decisive middle detail is that she did not return them. "
        "She plans to move to Berlin next year.")
    scene = _scene(text)
    distiller = _atomic_distiller(chunk_chars=55)
    units = scan_information_units(scene.turns)
    payload, aliases, turn_aliases = distiller._strict_payload(
        [scene], {scene.scene_id: units}, {scene.scene_id: 9})
    rows = payload["s"][0]["r"]

    assert "".join(row["t"] for row in rows) == text
    assert payload["s"][0]["k"] == 9
    assert aliases == {"0": scene.scene_id}
    assert len(turn_aliases["0"]) == len(rows)
    emitted = [entry[0] for row in rows for entry in row["u"]]
    assert sorted(emitted) == list(range(len(units)))


def test_atomic_validation_never_manufactures_confidence_and_exposes_fallback() -> None:
    text = "Alice bought 3 books in Paris last Friday."
    scene = _scene(text)
    units = scan_information_units(scene.turns)
    distiller = _atomic_distiller()
    all_ids = [unit.unit_id for unit in units]
    row = {"i": scene.scene_id, "f": [{
        "o": "Alice", "p": "bought", "v": "3 books in Paris",
        "g": "shopping", "n": "positive", "r": [0],
        "q": "bought 3 books in Paris", "z": all_ids,
        # Deliberately omit c: this fact must be rejected, never assigned 0.5.
        "e": [{"i": "turn:1", "q": "bought 3 books in Paris"}],
    }], "u": []}

    packet = distiller._validate_scene(scene, row, units=units, fact_cap=8)

    assert packet.facts == ()
    assert packet.missing_unit_ids == tuple(range(len(units)))
    assert packet.raw_fallback_turn_ids == ("turn:1",)
    assert packet.unit_coverage == 0.0


def test_atomic_validation_accepts_only_grounded_unit_links() -> None:
    text = "Alice bought 3 books in Paris last Friday."
    scene = _scene(text)
    units = scan_information_units(scene.turns)
    distiller = _atomic_distiller()
    grounded_ids = [unit.unit_id for unit in units
                    if unit.text.casefold() in "bought 3 books in paris".casefold()]
    unresolved_ids = [unit.unit_id for unit in units if unit.unit_id not in grounded_ids]
    row = {"i": scene.scene_id, "f": [{
        "o": "Alice", "p": "bought", "v": "3 books in Paris",
        "g": "shopping", "n": "positive", "r": [0],
        "q": "bought 3 books in Paris", "z": grounded_ids, "c": 0.93,
        "e": [{"i": "turn:1", "q": "bought 3 books in Paris"}],
    }], "u": [{"i": unit_id, "r": "not a separate durable fact"}
               for unit_id in unresolved_ids]}

    packet = distiller._validate_scene(scene, row, units=units, fact_cap=8)

    assert len(packet.facts) == 1 and packet.facts[0].confidence == 0.93
    assert set(packet.covered_unit_ids) == set(grounded_ids)
    assert set(packet.unresolved_unit_ids) == set(unresolved_ids)
    assert packet.missing_unit_ids == () and packet.unit_coverage == 1.0
