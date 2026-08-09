#!/usr/bin/env python3
"""Independent LLM audit of all materialized V5.10 typed relation edges."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from graphmem.config import load_config  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402


SYSTEM = """You audit typed edges in a conversational memory graph.
Judge only whether the two endpoint facts support the proposed relation.
Definitions:
- coreference: the facts refer to the same real-world fact, object, entity state, or paraphrased event.
- same_entity_state: the same specific entity and domain have a meaningful state progression.
- temporal_continuation: the same specific subject/predicate/object chain continues at distinct times.
- causal: one endpoint explicitly causes or explains the other.
- contradiction_update: the same proposition is negated, contradicted, replaced, or updated.
Return one JSON array, one object per row: {"i": integer, "valid": boolean,
"type_correct": boolean, "direction_correct": boolean,
"suggested_type": string or "NONE", "reason": <=18 words}.
Left is the stored edge source and right is its destination. For causal,
temporal_continuation and contradiction_update, direction_correct must judge
left->right; use true for symmetric relations.
Use a precision-first standard. Shared speakers or generic verbs alone are invalid.
No markdown and no extra text."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=WORKSPACE /
                        "artifacts/report/v5_10/hnsw_qwen_typed_dev200_graph_released/report_graph.sqlite")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/v5/v5_10_report.json")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument(
        "--per-relation-limit", type=int, default=0,
        help="0 audits every edge; positive values take a deterministic sample per type")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=WORKSPACE /
                        "artifacts/report/v5_10/typed_relation_judge")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def parse_response(content: str) -> list[dict]:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0].strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, list) else []
    except json.JSONDecodeError:
        rows = []
        for line in text.splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows


def endpoint_payload(summary: str, attributes_json: str) -> dict:
    try:
        attributes = json.loads(attributes_json)
    except (TypeError, json.JSONDecodeError):
        attributes = {}
    return {
        "summary": summary,
        "owner": attributes.get("owner_id", attributes.get("owners", ())),
        "predicate": attributes.get("predicate", attributes.get("predicates", ())),
        "value": attributes.get("value", attributes.get("values", ())),
        "scope": attributes.get("scope", attributes.get("scopes", ())),
        "polarity": attributes.get("polarity", ""),
        "observation_time": attributes.get(
            "observed_at", attributes.get("observation_time_range", ())),
        "event_time": attributes.get(
            "time_interval", attributes.get("event_time_range", attributes.get("times", ()))),
        "session": attributes.get("session_id", attributes.get("session_ids", ())),
    }


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "judgments.jsonl"
    completed = {}
    if args.resume and result_path.exists():
        completed = {
            row["edge_id"]: row for row in (
                json.loads(line) for line in result_path.read_text(
                    encoding="utf-8").splitlines() if line.strip())}
    store = SQLiteGraphStore(args.db, read_only=True)
    raw = store._read(
        "SELECT e.edge_id,e.memory_id,e.relation,e.confidence,e.source,"
        "a.node_type AS left_type,a.summary AS left_summary,"
        "a.attributes_json AS left_attributes,"
        "b.node_type AS right_type,b.summary AS right_summary,"
        "b.attributes_json AS right_attributes "
        "FROM graph_edges e JOIN graph_nodes a ON a.node_id=e.src_id "
        "JOIN graph_nodes b ON b.node_id=e.dst_id "
        "WHERE e.relation IN ('coreference','same_entity_state',"
        "'temporal_continuation','causal','contradiction_update') "
        "ORDER BY e.edge_id")
    all_edges = [dict(row) for row in raw]
    if args.per_relation_limit:
        edges = []
        for relation in sorted({row["relation"] for row in all_edges}):
            subset = [row for row in all_edges if row["relation"] == relation]
            subset.sort(key=lambda row: hashlib.sha256(
                f"{args.sample_seed}:{row['edge_id']}".encode()).hexdigest())
            edges.extend(subset[:args.per_relation_limit])
        edges.sort(key=lambda row: row["edge_id"])
    else:
        edges = all_edges
    pending = [row for row in edges if row["edge_id"] not in completed]
    config = load_config(args.config)
    from openai import OpenAI
    client = OpenAI(base_url=config.models.llm_base_url, api_key="local")
    usage = Counter()
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start:start + args.batch_size]
        payload = []
        for index, edge in enumerate(batch):
            payload.append({
                "i": index,
                "relation": edge["relation"],
                "left": endpoint_payload(
                    edge["left_summary"], edge["left_attributes"]),
                "right": endpoint_payload(
                    edge["right_summary"], edge["right_attributes"]),
            })
        response = client.chat.completions.create(
            model=config.models.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(
                    payload, ensure_ascii=False)},
            ],
            temperature=0,
            max_tokens=2048,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        parsed = {int(row["i"]): row for row in parse_response(
            response.choices[0].message.content or "")
            if isinstance(row, dict) and isinstance(row.get("i"), int)}
        records = []
        for index, edge in enumerate(batch):
            decision = parsed.get(index, {})
            valid = decision.get("valid")
            correct = decision.get("type_correct")
            direction_correct = decision.get("direction_correct")
            directional = edge["relation"] in {
                "temporal_continuation", "causal", "contradiction_update"}
            records.append({
                **edge,
                "valid": valid if isinstance(valid, bool) else None,
                "type_correct": correct if isinstance(correct, bool) else None,
                "direction_correct": (
                    direction_correct if isinstance(direction_correct, bool)
                    else (True if not directional else None)),
                "suggested_type": str(decision.get("suggested_type", "NONE")),
                "reason": str(decision.get("reason", "parse_failure")),
            })
        with result_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        response_usage = getattr(response, "usage", None)
        usage["input_tokens"] += int(getattr(response_usage, "prompt_tokens", 0) or 0)
        usage["output_tokens"] += int(getattr(response_usage, "completion_tokens", 0) or 0)
        print(f"{min(start + len(batch), len(pending))}/{len(pending)}", flush=True)
    rows = [json.loads(line) for line in result_path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    judged = [row for row in rows if isinstance(row.get("valid"), bool)
              and isinstance(row.get("type_correct"), bool)
              and isinstance(row.get("direction_correct"), bool)]
    by_relation = {}
    for relation in sorted({row["relation"] for row in judged}):
        subset = [row for row in judged if row["relation"] == relation]
        by_relation[relation] = {
            "n": len(subset),
            "valid_precision": sum(row["valid"] for row in subset) / len(subset),
            "typed_precision": sum(row["valid"] and row["type_correct"]
                                   and row["direction_correct"]
                                   for row in subset) / len(subset),
            "direction_precision": sum(row["direction_correct"]
                                       for row in subset) / len(subset),
        }
    summary = {
        "db": str(args.db), "materialized_typed_edges": len(all_edges),
        "edges": len(edges), "judged": len(judged),
        "sampling": {
            "per_relation_limit": args.per_relation_limit,
            "seed": args.sample_seed,
        },
        "parse_failures": len(rows) - len(judged),
        "valid_precision": (sum(row["valid"] for row in judged) / len(judged)
                            if judged else 0.0),
        "typed_precision": (sum(row["valid"] and row["type_correct"]
                                and row["direction_correct"] for row in judged)
                            / len(judged) if judged else 0.0),
        "direction_precision": (sum(row["direction_correct"] for row in judged)
                                / len(judged) if judged else 0.0),
        "by_relation": by_relation,
        "usage": dict(usage),
        "judge_model": config.models.llm_model,
        "judge_prompt": SYSTEM,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    store.close()


if __name__ == "__main__":
    main()
