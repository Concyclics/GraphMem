#!/usr/bin/env python3
"""Run a real-model V5.10 lossless atomic extraction contract smoke test."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.build import QwenSemanticDistiller  # noqa: E402
from graphmem.build.pipeline import _SceneSlice  # noqa: E402
from graphmem.config import load_config  # noqa: E402
from graphmem.domain import Conversation, Session, SourceTurn, stable_id  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/v5/v5_10_report.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/report/v5_10/atomic_smoke")
    return parser.parse_args()


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    memory_id = "v5_10_atomic_smoke"
    raw_turns = (
        (
            "Alice", "2025-01-05T12:00:00Z",
            "I did not buy the first camera. I bought 3 Sony lenses in Paris last Friday, "
            "and I plan to return one lens next month.",
        ),
        (
            "Bob", "2025-02-10T09:00:00Z",
            "Bob finished the Kyoto marathon in 3 hours 42 minutes. He won second place "
            "in his age group, but he has never completed the Osaka race.",
        ),
        (
            "Carol", "2025-03-01T08:00:00Z",
            "Opening context about a routine status update. "
            + ("A neutral sentence keeps the long source realistic. " * 70)
            + "The decisive middle fact is that Carol cancelled the Berlin booking for 17 June 2025 "
            "and paid a 25 percent fee. "
            + ("Another neutral sentence follows the important detail. " * 70)
            + "At the end she said she might rebook next year, but has not decided.",
        ),
    )
    turns = []
    for index, (speaker, timestamp, text) in enumerate(raw_turns):
        turns.append(SourceTurn(
            stable_id("turn", memory_id, "session:1", index), memory_id,
            "session:1", index, speaker, "assistant", "user", timestamp,
            text, digest(text)))
    store = SQLiteGraphStore(args.output / "cache.sqlite")
    store.ingest_conversation(
        Conversation(memory_id, "smoke", memory_id, digest("".join(row.raw_text for row in turns))),
        [Session("session:1", memory_id, 0, raw_turns[0][1], digest("session:1"))],
        turns,
    )
    scenes = tuple(
        _SceneSlice(stable_id("scene", memory_id, index), "session:1", (turn,),
                    " ".join(turn.raw_text.split()[:24]))
        for index, turn in enumerate(turns)
    )
    distiller = QwenSemanticDistiller(store, config, "v5_10_atomic_smoke_v1")
    packets = distiller.extract(memory_id, scenes)
    output = {
        "config": str(args.config),
        "ledger": dict(distiller.ledger.snapshot()),
        "packets": [{
            "scene_id": packet.scene_id,
            "summary": packet.summary,
            "fact_cap": packet.fact_cap,
            "unit_coverage": packet.unit_coverage,
            "covered_unit_ids": packet.covered_unit_ids,
            "unresolved_unit_ids": packet.unresolved_unit_ids,
            "missing_unit_ids": packet.missing_unit_ids,
            "implicitly_covered_unit_ids": packet.implicitly_covered_unit_ids,
            "raw_fallback_turn_ids": packet.raw_fallback_turn_ids,
            "information_units": [asdict(unit) for unit in packet.information_units],
            "facts": [asdict(fact) for fact in packet.facts],
        } for packet in packets],
    }
    (args.output / "result.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "scenes": len(packets),
        "facts": sum(len(packet.facts) for packet in packets),
        "units": sum(len(packet.information_units) for packet in packets),
        "covered": sum(len(packet.covered_unit_ids) for packet in packets),
        "unresolved": sum(len(packet.unresolved_unit_ids) for packet in packets),
        "missing": sum(len(packet.missing_unit_ids) for packet in packets),
        "implicitly_linked": sum(
            len(packet.implicitly_covered_unit_ids) for packet in packets),
        "fallback_scenes": sum(bool(packet.raw_fallback_turn_ids) for packet in packets),
        "confidence_values": sorted({fact.confidence for packet in packets for fact in packet.facts}),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    store.close()


if __name__ == "__main__":
    main()
