#!/usr/bin/env python3
"""Independent LLM audit of all materialized V5.10 typed relation edges."""
from __future__ import annotations

import argparse
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
"type_correct": boolean, "suggested_type": string or "NONE", "reason": <=18 words}.
Use a precision-first standard. Shared speakers or generic verbs alone are invalid.
No markdown and no extra text."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=WORKSPACE /
                        "artifacts/report/v5_10/hnsw_qwen_typed_dev200_graph_released/report_graph.sqlite")
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/v5/v5_10_report.json")
    parser.add_argument("--batch-size", type=int, default=12)
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
        "WHERE e.source LIKE 'typed_%' ORDER BY e.edge_id")
    edges = [dict(row) for row in raw]
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
                "left": edge["left_summary"],
                "right": edge["right_summary"],
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
            records.append({
                **edge,
                "valid": valid if isinstance(valid, bool) else None,
                "type_correct": correct if isinstance(correct, bool) else None,
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
              and isinstance(row.get("type_correct"), bool)]
    by_relation = {}
    for relation in sorted({row["relation"] for row in judged}):
        subset = [row for row in judged if row["relation"] == relation]
        by_relation[relation] = {
            "n": len(subset),
            "valid_precision": sum(row["valid"] for row in subset) / len(subset),
            "typed_precision": sum(row["valid"] and row["type_correct"]
                                   for row in subset) / len(subset),
        }
    summary = {
        "db": str(args.db), "edges": len(edges), "judged": len(judged),
        "parse_failures": len(rows) - len(judged),
        "valid_precision": (sum(row["valid"] for row in judged) / len(judged)
                            if judged else 0.0),
        "typed_precision": (sum(row["valid"] and row["type_correct"] for row in judged)
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
