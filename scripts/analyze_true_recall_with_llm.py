#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem_demo.data import build_leaf_nodes, load_longmemeval_cases  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use a local LLM to label support leaf IDs and compute evidence-level recall."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--retrieval-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--question-type", default="all")
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL"))
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY", "dummy"))
    parser.add_argument("--leaf-text-max-chars", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def is_abstention_case(question_id: str, gold_answer: Any) -> bool:
    if question_id.endswith("_abs"):
        return True
    answer = str(gold_answer or "").lower()
    return bool(
        re.search(
            r"not enough|did not mention|does not mention|insufficient|cannot determine|unknown",
            answer,
        )
    )


def shorten(text: str, max_chars: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def build_prompt(
    *,
    question_id: str,
    question_type: str,
    question: str,
    gold_answer: Any,
    candidates: list[dict[str, Any]],
) -> str:
    lines = []
    for candidate in candidates:
        lines.append(
            f"- idx={candidate['idx']} node_id={candidate['node_id']} text={candidate['text']}"
        )
    blocks = "\n".join(lines)
    return f"""You are labeling gold evidence leaves for memory QA evaluation.

Task:
Given QUESTION, GOLD_ANSWER, and candidate LEAVES from gold answer sessions,
select the minimal set of leaf indices that are necessary to support the gold answer.

Rules:
1) Prefer precision over recall. Do not select loosely related leaves.
2) If a leaf is not needed to derive the gold answer, do not include it.
3) Multiple leaves can be required for aggregation / temporal ordering.
4) Return valid JSON only with keys:
   - support_leaf_indices: integer array
   - confidence: number in [0,1]
   - rationale: short string

QUESTION_ID: {question_id}
QUESTION_TYPE: {question_type}
QUESTION: {question}
GOLD_ANSWER: {gold_answer}

CANDIDATE_LEAVES:
{blocks}
"""


def parse_json_response(text: str) -> dict[str, Any]:
    raw = text.strip()
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise ValueError(f"invalid_json_response: {raw[:200]}")
        return json.loads(match.group(0))


def parse_fallback_response(text: str) -> dict[str, Any]:
    indices: list[int] = []
    list_match = re.search(
        r"support_leaf_indices\"\s*:\s*\[([^\]]*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if list_match:
        indices = [int(value) for value in re.findall(r"\d+", list_match.group(1))]
    confidence = None
    conf_match = re.search(
        r"confidence\"\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if conf_match:
        confidence = float(conf_match.group(1))
    rationale = "fallback_parse"
    rat_match = re.search(r"rationale\"\s*:\s*\"([^\"]*)", text, flags=re.IGNORECASE)
    if rat_match:
        rationale = rat_match.group(1)
    return {
        "support_leaf_indices": indices,
        "confidence": confidence if confidence is not None else 0.0,
        "rationale": rationale,
        "_fallback_parse": True,
    }


def judge_support_indices(
    *,
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    retries: int = 3,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Return strict JSON only, no markdown.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=500,
            )
            content = response.choices[0].message.content or "{}"
            try:
                payload = parse_json_response(content)
            except Exception:
                payload = parse_fallback_response(content)
            payload["_usage"] = {
                "prompt_tokens": int(getattr(response.usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(response.usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(response.usage, "total_tokens", 0) or 0),
            }
            return payload
        except Exception as error:
            last_error = error
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"judge_failed: {last_error}")


def main() -> None:
    args = parse_args()
    if not args.model:
        raise ValueError("model is required (set --model or DEEPSEEK_MODEL).")
    if not args.base_url:
        raise ValueError("base_url is required (set --base-url or DEEPSEEK_BASE_URL).")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=240)
    cases = load_longmemeval_cases(
        args.data,
        question_type=args.question_type,
        max_questions=args.max_questions,
    )
    case_by_id = {case.question_id: case for case in cases}
    retrieval_by_id = {
        str(row["question_id"]): row for row in read_jsonl(args.retrieval_results)
    }

    rows: list[dict[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    for idx, case in enumerate(cases, start=1):
        retrieval = retrieval_by_id.get(case.question_id)
        if retrieval is None:
            continue

        abstention = is_abstention_case(case.question_id, case.answer)
        leaves = build_leaf_nodes(case)
        candidate_leaves = [leaf for leaf in leaves if leaf.session_id in set(case.answer_session_ids)]
        candidate_rows = [
            {
                "idx": i,
                "node_id": leaf.node_id,
                "text": shorten(leaf.raw_text, args.leaf_text_max_chars),
            }
            for i, leaf in enumerate(candidate_leaves)
        ]

        label_payload: dict[str, Any]
        if abstention or not candidate_rows:
            label_payload = {
                "support_leaf_indices": [],
                "confidence": 1.0 if abstention else 0.0,
                "rationale": "abstention_or_no_candidates",
                "_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
        else:
            prompt = build_prompt(
                question_id=case.question_id,
                question_type=case.question_type,
                question=case.question,
                gold_answer=case.answer,
                candidates=candidate_rows,
            )
            try:
                label_payload = judge_support_indices(
                    client=client,
                    model=args.model,
                    prompt=prompt,
                    temperature=args.temperature,
                )
            except Exception as error:
                label_payload = {
                    "support_leaf_indices": [],
                    "confidence": 0.0,
                    "rationale": f"judge_error:{error}",
                    "_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }

        selected_indices = [
            int(value)
            for value in (label_payload.get("support_leaf_indices") or [])
            if str(value).strip().isdigit()
        ]
        selected_indices = sorted(set(i for i in selected_indices if 0 <= i < len(candidate_rows)))
        gold_leaf_ids = [candidate_rows[i]["node_id"] for i in selected_indices]
        gold_leaf_id_set = set(gold_leaf_ids)
        retrieved_leaf_ids = set(str(value) for value in retrieval.get("leaf_node_ids") or [])
        hit_leaf_ids = sorted(gold_leaf_id_set & retrieved_leaf_ids)
        gold_leaf_recall = (
            len(hit_leaf_ids) / len(gold_leaf_id_set) if gold_leaf_id_set else None
        )

        usage = label_payload.get("_usage", {})
        total_prompt_tokens += int(usage.get("prompt_tokens") or 0)
        total_completion_tokens += int(usage.get("completion_tokens") or 0)
        total_tokens += int(usage.get("total_tokens") or 0)

        row = {
            "question_id": case.question_id,
            "question_type": case.question_type,
            "is_abstention": abstention,
            "question": case.question,
            "gold_answer": case.answer,
            "answer_session_ids": case.answer_session_ids,
            "retrieved_answer_session_recall": float(retrieval.get("answer_session_recall") or 0.0),
            "candidate_leaf_count": len(candidate_rows),
            "gold_leaf_ids": gold_leaf_ids,
            "gold_leaf_count": len(gold_leaf_ids),
            "retrieved_leaf_count": len(retrieved_leaf_ids),
            "hit_gold_leaf_ids": hit_leaf_ids,
            "hit_gold_leaf_count": len(hit_leaf_ids),
            "gold_leaf_recall": gold_leaf_recall,
            "gold_leaf_all_hit": (
                len(gold_leaf_id_set) > 0 and len(hit_leaf_ids) == len(gold_leaf_id_set)
            ),
            "gold_leaf_any_hit": len(hit_leaf_ids) > 0,
            "llm_confidence": label_payload.get("confidence"),
            "llm_rationale": str(label_payload.get("rationale") or "")[:600],
            "llm_prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "llm_completion_tokens": int(usage.get("completion_tokens") or 0),
            "llm_total_tokens": int(usage.get("total_tokens") or 0),
        }
        rows.append(row)

        print(
            f"[{idx}/{len(cases)}] {case.question_id} "
            f"gold_leafs={row['gold_leaf_count']} hit={row['hit_gold_leaf_count']} "
            f"leaf_recall={row['gold_leaf_recall']}",
            flush=True,
        )

    non_abs_rows = [row for row in rows if not row["is_abstention"] and row["gold_leaf_count"] > 0]
    true_recall = (
        sum(float(row["gold_leaf_recall"]) for row in non_abs_rows if row["gold_leaf_recall"] is not None)
        / len(non_abs_rows)
        if non_abs_rows
        else None
    )
    all_hit_rate = (
        sum(bool(row["gold_leaf_all_hit"]) for row in non_abs_rows) / len(non_abs_rows)
        if non_abs_rows
        else None
    )
    any_hit_rate = (
        sum(bool(row["gold_leaf_any_hit"]) for row in non_abs_rows) / len(non_abs_rows)
        if non_abs_rows
        else None
    )

    summary = {
        "question_count": len(rows),
        "non_abstention_with_gold_leaf_count": len(non_abs_rows),
        "retrieval_answer_session_recall_avg_non_abs": (
            sum(row["retrieved_answer_session_recall"] for row in non_abs_rows) / len(non_abs_rows)
            if non_abs_rows
            else None
        ),
        "true_recall_llm_gold_leaf_avg": true_recall,
        "gold_leaf_all_hit_rate": all_hit_rate,
        "gold_leaf_any_hit_rate": any_hit_rate,
        "llm_label_prompt_tokens": total_prompt_tokens,
        "llm_label_completion_tokens": total_completion_tokens,
        "llm_label_total_tokens": total_tokens,
        "model": args.model,
        "base_url": args.base_url,
    }

    audit_path = args.output_dir / "llm_gold_leaf_recall_audit.jsonl"
    with audit_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = args.output_dir / "llm_gold_leaf_recall_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {summary_path}")
    print(f"saved: {audit_path}")


if __name__ == "__main__":
    main()
