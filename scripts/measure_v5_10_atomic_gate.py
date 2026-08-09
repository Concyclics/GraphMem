#!/usr/bin/env python3
"""Gate A: run V5.10 atomic extraction on the 109 known missing-fact turns.

The selected turns are the same query-agnostic source turns used by the V5.9
extraction-rescue experiment.  Extraction never receives a benchmark question.
A separate sufficiency judge compares current V5.9 facts, the previous free-form
lossless extractor, V5.10 strict atomic facts, and the raw-turn oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from graphmem.build import QwenSemanticDistiller  # noqa: E402
from graphmem.build.pipeline import _SceneSlice  # noqa: E402
from graphmem.config import load_config  # noqa: E402
from graphmem.domain import Conversation, Session, SourceTurn, stable_id  # noqa: E402
from graphmem.judging import OpenAICompatibleClient  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402
from graphmem.build.temporal import extract_time_expression  # noqa: E402
from measure_v5_9_extraction_rescue import (  # noqa: E402
    JUDGE_SYSTEM,
    load_questions,
    parse_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/v5/v5_10_report.json")
    parser.add_argument(
        "--db", type=Path,
        default=WORKSPACE / "artifacts/v5_9/full_benchmark_20260809/graph/report_graph.sqlite")
    parser.add_argument(
        "--previous-extracted", type=Path,
        default=ROOT / "artifacts/report/v5_9/extraction_rescue/extracted.jsonl")
    parser.add_argument(
        "--previous-sufficiency", type=Path,
        default=ROOT / "artifacts/report/v5_9/extraction_rescue/sufficiency.jsonl")
    parser.add_argument("--lme", type=Path,
                        default=WORKSPACE / "artifacts/data/longmemeval_s_cleaned.json")
    parser.add_argument("--locomo", type=Path,
                        default=WORKSPACE / "artifacts/data/locomo10_graphmem.json")
    parser.add_argument("--lme-gold", type=Path,
                        default=ROOT / "eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--judge-model", default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/report/v5_10/atomic_gate")
    parser.add_argument("--skip-judge", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def fact_text(fact: Any) -> str:
    if isinstance(fact, str):
        return fact
    return " | ".join(str(fact.get(key) or "") for key in (
        "subject", "owner", "predicate", "value", "object", "time", "polarity"))


def rate(rows: list[dict[str, Any]], field: str = "sufficient") -> float:
    return sum(bool(row.get(field)) for row in rows) / len(rows) if rows else 0.0


def main() -> None:
    args = parse_args()
    os.environ.setdefault("LOCAL_API_KEY", "local")
    args.output.mkdir(parents=True, exist_ok=True)
    selected_turn_ids = {
        str(row["turn_id"]) for row in read_jsonl(args.previous_extracted)
    }
    source = sqlite3.connect(f"file:{args.db.resolve()}?mode=ro", uri=True)
    placeholders = ",".join("?" for _ in selected_turn_ids)
    raw_rows = source.execute(
        "SELECT turn_id,memory_id,session_id,turn_index,speaker,listener,role,"
        "timestamp,raw_text,content_hash FROM source_turns WHERE turn_id IN ("
        + placeholders + ")", tuple(sorted(selected_turn_ids))).fetchall()
    original_turns = {
        str(row[0]): {
            "turn_id": str(row[0]), "memory_id": str(row[1]), "session_id": str(row[2]),
            "turn_index": int(row[3]), "speaker": str(row[4]), "listener": str(row[5]),
            "role": str(row[6]), "timestamp": row[7], "raw_text": str(row[8]),
            "content_hash": str(row[9]),
        } for row in raw_rows
    }
    if set(original_turns) != selected_turn_ids:
        missing = sorted(selected_turn_ids - set(original_turns))
        raise RuntimeError(f"selected source turns missing from DB: {missing[:5]}")

    current_facts: dict[str, list[str]] = defaultdict(list)
    for turn_id, summary in source.execute("""
        SELECT em.turn_id,n.summary
        FROM graph_nodes n JOIN evidence_members em
          ON em.evidence_group_id=n.evidence_group_id
        WHERE n.node_type='canonical_fact'
        UNION
        SELECT em.turn_id,n.summary
        FROM graph_nodes n, json_each(n.evidence_group_ids_json) groups
        JOIN evidence_members em ON em.evidence_group_id=groups.value
        WHERE n.node_type='canonical_fact'
    """):
        current_facts[str(turn_id)].append(str(summary))
    source.close()

    config = load_config(args.config)
    synthetic_memory = "v5_10_atomic_gate_missing_turns"
    synthetic_turns = []
    scenes = []
    scene_to_turn: dict[str, str] = {}
    for index, turn_id in enumerate(sorted(selected_turn_ids)):
        old = original_turns[turn_id]
        turn = SourceTurn(
            turn_id, synthetic_memory, "selected", index, old["speaker"], old["listener"],
            old["role"], old["timestamp"], old["raw_text"], old["content_hash"])
        synthetic_turns.append(turn)
        scene_id = stable_id("scene", synthetic_memory, turn_id)
        scenes.append(_SceneSlice(
            scene_id, "selected", (turn,), " ".join(turn.raw_text.split()[:24])))
        scene_to_turn[scene_id] = turn_id
    store = SQLiteGraphStore(args.output / "cache.sqlite")
    store.ingest_conversation(
        Conversation(synthetic_memory, "v5_10_gate", synthetic_memory,
                     digest("".join(turn.raw_text for turn in synthetic_turns))),
        [Session("selected", synthetic_memory, 0, None, digest("selected"))],
        synthetic_turns,
    )
    distiller = QwenSemanticDistiller(store, config, "v5_10_atomic_gate_v1")
    packets = distiller.extract(synthetic_memory, scenes)
    new_facts: dict[str, list[str]] = {}
    packet_rows = []
    for packet in packets:
        turn_id = scene_to_turn[packet.scene_id]
        source_turn = original_turns[turn_id]
        serialized_facts = []
        for fact in packet.facts:
            evidence_text = " ".join(
                source_turn["raw_text"][start:end]
                for cited_turn, start, end in fact.evidence if cited_turn == turn_id)
            explicit_time = (extract_time_expression(evidence_text)
                             or extract_time_expression(source_turn["raw_text"]))
            temporal = f" | observed_at={source_turn['timestamp'] or '<unknown>'}"
            if explicit_time:
                temporal += f" | event_time_expression={explicit_time}"
            serialized_facts.append(
                f"{fact.owner} | {fact.predicate} | {fact.value} | {fact.polarity}{temporal}")
        new_facts[turn_id] = serialized_facts
        packet_rows.append({
            "turn_id": turn_id, "scene_id": packet.scene_id,
            "fact_cap": packet.fact_cap, "unit_coverage": packet.unit_coverage,
            "units": [asdict(unit) for unit in packet.information_units],
            "covered_unit_ids": packet.covered_unit_ids,
            "implicitly_covered_unit_ids": packet.implicitly_covered_unit_ids,
            "unresolved_unit_ids": packet.unresolved_unit_ids,
            "missing_unit_ids": packet.missing_unit_ids,
            "raw_fallback_turn_ids": packet.raw_fallback_turn_ids,
            "facts": [asdict(fact) for fact in packet.facts],
        })
    (args.output / "extracted.json").write_text(
        json.dumps(packet_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    previous_extracted = {
        str(row["turn_id"]): [fact_text(fact) for fact in row.get("facts", ())]
        for row in read_jsonl(args.previous_extracted)
    }
    previous_sufficiency = {
        (str(row["question_id"]), str(row["condition"])): row
        for row in read_jsonl(args.previous_sufficiency)
    }
    selected_questions = sorted({qid for qid, _condition in previous_sufficiency})
    question_args = SimpleNamespace(lme=args.lme, locomo=args.locomo, lme_gold=args.lme_gold)
    questions = load_questions(question_args)

    # Re-open only for mapping annotated (memory, session, turn index) refs to
    # the exact raw source turns used by the extraction conditions.
    source = sqlite3.connect(f"file:{args.db.resolve()}?mode=ro", uri=True)
    turn_by_ref = {
        (str(memory), str(session), int(index)): {
            "turn_id": str(turn_id), "timestamp": str(timestamp or ""),
            "speaker": str(speaker), "raw_text": str(raw_text),
        }
        for turn_id, memory, session, index, timestamp, speaker, raw_text in source.execute(
            "SELECT turn_id,memory_id,session_id,turn_index,timestamp,speaker,raw_text "
            "FROM source_turns")
    }
    source.close()

    def condition_evidence(question: dict[str, Any], condition: str) -> str:
        parts = []
        for session_id, turn_index in question["refs"]:
            turn = turn_by_ref.get((question["memory_id"], session_id, turn_index))
            if not turn:
                continue
            if condition == "raw":
                parts.append(f"[{turn['timestamp']}] {turn['speaker']}: {turn['raw_text']}")
                continue
            rows = list(current_facts.get(turn["turn_id"], ()))
            if not rows:
                rows.extend(
                    new_facts.get(turn["turn_id"], ()) if condition == "v5_10"
                    else previous_extracted.get(turn["turn_id"], ()))
            parts.extend(rows)
        return "\n".join(parts) or "<NO INDEXED FACTS>"

    judged: list[dict[str, Any]] = []
    calls = []
    if not args.skip_judge:
        client = OpenAICompatibleClient(
            model=args.judge_model, base_url=args.base_url, api_key_env="LOCAL_API_KEY",
            request_profile="qwen", max_retries=3, timeout_sec=180)

        def judge(qid: str) -> tuple[dict[str, Any], list[Any]]:
            question = questions[qid]
            evidence = condition_evidence(question, "v5_10")
            payload = {
                "question": question["question"], "reference_answer": question["answer"],
                "evidence_condition": "v5_10_atomic", "evidence": evidence,
            }
            parsed: dict[str, Any] = {}
            records = []
            error: str | None = None
            for max_tokens in (512, 1024):
                result = client.chat(
                    question_id=qid, variant="v5_10_atomic", stage="sufficiency_judge",
                    thinking_mode="none", json_mode=True, max_tokens=max_tokens,
                    temperature=0.0, seed=0,
                    messages=[{"role": "system", "content": JUDGE_SYSTEM},
                              {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
                records.append(result.record)
                try:
                    parsed = parse_payload(result.text)
                    error = None
                    break
                except (json.JSONDecodeError, ValueError) as exception:
                    error = repr(exception)
            old = previous_sufficiency.get((qid, "current"), {})
            return ({
                "question_id": qid, "stratum": old.get("stratum", "unknown"),
                "condition": "v5_10", "sufficient": bool(parsed.get("sufficient")),
                "reason": str(parsed.get("reason", "")),
                "judge_error": error,
            }, records)

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(judge, qid) for qid in selected_questions]
            for future in as_completed(futures):
                row, records = future.result()
                judged.append(row); calls.extend(asdict(record) for record in records)
        judged.sort(key=lambda row: row["question_id"])
        (args.output / "sufficiency.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in judged),
            encoding="utf-8")
        (args.output / "judge_calls.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in calls),
            encoding="utf-8")

    total_units = sum(len(packet.information_units) for packet in packets)
    covered_units = sum(len(packet.covered_unit_ids) for packet in packets)
    missing_units = sum(len(packet.missing_unit_ids) for packet in packets)
    kinds = Counter(unit.kind for packet in packets for unit in packet.information_units)
    covered_kinds = Counter()
    for packet in packets:
        unit_map = {unit.unit_id: unit for unit in packet.information_units}
        covered_kinds.update(unit_map[unit_id].kind for unit_id in packet.covered_unit_ids)
    summary: dict[str, Any] = {
        "selected_turns": len(selected_turn_ids),
        "facts": sum(len(packet.facts) for packet in packets),
        "facts_per_turn": sum(len(packet.facts) for packet in packets) / len(packets),
        "information_units": total_units,
        "covered_units": covered_units,
        "missing_units": missing_units,
        "unit_coverage": covered_units / total_units if total_units else 1.0,
        "raw_fallback_turns": sum(bool(packet.raw_fallback_turn_ids) for packet in packets),
        "implicitly_linked_units": sum(
            len(packet.implicitly_covered_unit_ids) for packet in packets),
        "confidence_half_facts": sum(
            fact.confidence == 0.5 for packet in packets for fact in packet.facts),
        "per_kind": {
            kind: {"total": count, "covered": covered_kinds[kind],
                   "coverage": covered_kinds[kind] / count}
            for kind, count in sorted(kinds.items())
        },
        "build_token_ledger": dict(distiller.ledger.snapshot()),
    }
    if judged:
        by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in judged:
            by_stratum[row["stratum"]].append(row)
        summary["sufficiency"] = {
            "v5_10": rate(judged),
            "previous_augmented": rate([
                previous_sufficiency[(qid, "augmented")] for qid in selected_questions]),
            "current": rate([
                previous_sufficiency[(qid, "current")] for qid in selected_questions]),
            "raw_oracle": rate([
                previous_sufficiency[(qid, "raw")] for qid in selected_questions]),
            "per_stratum": {
                stratum: {
                    "n": len(rows), "v5_10": rate(rows),
                    "previous_augmented": rate([
                        previous_sufficiency[(row["question_id"], "augmented")]
                        for row in rows]),
                    "raw_oracle": rate([
                        previous_sufficiency[(row["question_id"], "raw")]
                        for row in rows]),
                } for stratum, rows in sorted(by_stratum.items())
            },
        }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    store.close()


if __name__ == "__main__":
    main()
