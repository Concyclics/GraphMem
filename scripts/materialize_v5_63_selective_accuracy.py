#!/usr/bin/env python3
"""Compose label-free, high-confidence prompt refinements over Safe64.

The input arms are frozen full-corpus experiments.  This materializer selects
an arm using only the question, prompt structure and source-backed workspace;
it never reads a prediction, gold answer, category or judge verdict. Preference
and relation routes retain exact evidence IDs/order. The measured date-focus
route may repack the same 64-turn reservoir to make room for its reading index.
Every selected prompt stays within the configured per-question token increase.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable


VERSION = "graphmem-v5.63-selective-accuracy-v1"
_DATE_SURFACE_RE = re.compile(
    r"\b(?:ago|before|between|passed|take)\b", re.I)
_HOW_LONG_RE = re.compile(r"\bhow\s+long\b", re.I)
_ADDITIVE_DURATION_RE = re.compile(
    r"\bhow\s+many\s+(?:minutes?|hours?|days?|weeks?|months?|years?)\s+"
    r"did\s+it\s+take\b", re.I)
_NAMED_SPEAKER_RE = re.compile(
    r"^\[(?:CHAIN \d+ (?:step=\d+|support)|GRAPH \d+ step=\d+|"
    r"AUX \d+ rank=\d+)\]\s+\[[^\]]+\]\s+([^:\n]{1,48}):",
    re.M,
)
_GENERIC_SPEAKERS = frozenset({
    "", "assistant", "system", "tool", "user", "human",
})
_MONEY_RE = re.compile(
    r"[$\u00a3\u20ac\u00a5]\s*([0-9][0-9,]*(?:\.[0-9]+)?)")
_DIRECT_COUNT_RE = re.compile(
    r"\b(?:attend(?:ed)?|watch(?:ed)?|pack(?:ed)?|complete(?:d)?|"
    r"add(?:ed)?|wear|wore|spot(?:ted)?|catch|caught|try|tried|"
    r"lead|leads|led|buy|bought|acquire(?:d)?|visit(?:ed)?|"
    r"write|wrote|written|read|has|have|had)\s+"
    r"(?:a\s+total\s+of\s+|exactly\s+|about\s+|around\s+)?"
    r"(?P<count>one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|[0-9]+)\s+(?P<object>[a-z][a-z-]*)\b",
    re.I,
)
_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}
_GENERIC_TERMS = frozenset({
    "about", "after", "all", "amount", "and", "answer", "before",
    "between", "combined", "cost", "current", "currently", "day",
    "days", "did", "do", "does", "event", "events", "for", "from",
    "got", "had", "has", "have", "how", "including", "into", "last",
    "many", "month", "months", "money", "much", "new", "number", "of",
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "on", "past", "question",
    "recent", "recently", "since", "spend", "spent",
    "take", "the", "through", "time", "times", "total", "was", "were",
    "what", "when", "with", "year", "years", "you", "your",
})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(
        json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8")


def normalize(token: str) -> str:
    value = token.casefold()
    if len(value) > 5 and value.endswith("ies"):
        return value[:-3] + "y"
    if len(value) > 5 and value.endswith("ing"):
        return value[:-3]
    if len(value) > 4 and value.endswith("ed"):
        return value[:-2]
    if len(value) > 4 and value.endswith("s") and not value.endswith("ss"):
        return value[:-1]
    return value


def terms(text: str) -> set[str]:
    return {
        normalize(token) for token in re.findall(r"[a-z0-9]+", text.casefold())
        if len(token) > 2
    }


def question_of(metadata: dict[str, Any]) -> str:
    return " ".join(str(metadata.get("question") or "").split())


def operation_of(row: dict[str, Any]) -> str:
    ledger = (row.get("trace") or {}).get("aggregation_ledger") or {}
    return str(ledger.get("operation") or "")


def named_transcript(row: dict[str, Any]) -> bool:
    if not row.get("messages"):
        return False
    prompt = str(row["messages"][-1].get("content") or "")
    return any(match.group(1).casefold().strip() not in _GENERIC_SPEAKERS
               for match in _NAMED_SPEAKER_RE.finditer(prompt))


def relation_workspace(row: dict[str, Any]) -> str:
    if not row.get("messages"):
        return ""
    prompt = str(row["messages"][-1].get("content") or "")
    marker = "Source-backed relation workspace"
    if marker not in prompt:
        return ""
    return marker + prompt.split(marker, 1)[1]


def route_preference(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return bool(
        candidate.get("prompt_payload_hash") != baseline.get("prompt_payload_hash")
        and (candidate.get("trace") or {}).get("preference_focus_strategy"))


def route_additive_duration(question: str, operation: str) -> bool:
    return bool(
        operation == "date_difference"
        and _ADDITIVE_DURATION_RE.search(question)
        and re.search(r"\band\b", question, re.I))


def route_joint_minimum(question: str, operation: str) -> bool:
    return bool(
        operation == "minimum"
        and re.search(r"\bminimum\s+amount\b", question, re.I)
        and re.search(r"\band\b", question, re.I))


def route_complete_alternative(question: str, workspace: str,
                               operation: str) -> bool:
    if operation != "difference":
        return False
    first = re.search(
        r"\bby\s+taking\s+(?:(?:the|a|an)\s+)?([a-z][a-z-]*)", question,
        re.I)
    second = re.search(
        r"\binstead\s+of\s+(?:(?:the|a|an)\s+)?([a-z][a-z-]*)", question,
        re.I)
    if first is None or second is None:
        return False
    lowered = workspace.casefold()
    return all(re.search(rf"\b{re.escape(value.casefold())}\b", lowered)
               for value in (first.group(1), second.group(1)))


def route_complete_money_sum(question: str, workspace: str,
                             operation: str) -> bool:
    if operation != "sum":
        return False
    values = {
        float(value.replace(",", "")) for value in _MONEY_RE.findall(workspace)
    }
    if len(values) < 3:
        return False
    # A multi-operand question is unsafe when the workspace contains several
    # unrelated prices but omits named operands.  Content-term coverage rejects
    # that case while retaining open-ended totals over events/workshops.
    critical = terms(question) - {normalize(value) for value in _GENERIC_TERMS}
    if not critical:
        return False
    coverage = len(critical & terms(workspace)) / len(critical)
    return coverage >= 0.80


def direct_count_values(question: str, workspace: str) -> set[int]:
    question_terms = terms(question) - {
        normalize(value) for value in _GENERIC_TERMS
    }
    values: set[int] = set()
    for line in workspace.splitlines():
        if not re.match(r"^R\d+\s", line):
            continue
        relation = line.split("| source=", 1)[0]
        overlap = question_terms & terms(relation)
        if len(overlap) < 2:
            continue
        for match in _DIRECT_COUNT_RE.finditer(relation):
            if normalize(match.group("object")) not in question_terms:
                continue
            raw = match.group("count").casefold()
            values.add(_WORD_NUMBERS.get(raw, int(raw) if raw.isdigit() else -1))
    values.discard(-1)
    return values


def route_direct_count(question: str, workspace: str, operation: str) -> bool:
    return operation == "count_distinct" and len(
        direct_count_values(question, workspace)) == 1


def route_relation(question: str, candidate: dict[str, Any]) -> str | None:
    workspace = relation_workspace(candidate)
    if not workspace:
        return None
    operation = operation_of(candidate)
    if route_additive_duration(question, operation):
        return "relation_additive_duration"
    if route_joint_minimum(question, operation):
        return "relation_joint_minimum"
    if route_complete_alternative(question, workspace, operation):
        return "relation_complete_alternative"
    if route_complete_money_sum(question, workspace, operation):
        return "relation_complete_money_sum"
    if route_direct_count(question, workspace, operation):
        return "relation_direct_count"
    return None


def route_date_focus(question: str, candidate: dict[str, Any]) -> bool:
    return bool(
        not named_transcript(candidate)
        and operation_of(candidate) == "date_difference"
        and _DATE_SURFACE_RE.search(question)
        and not _HOW_LONG_RE.search(question))


def validate_arm(name: str, baseline: list[dict[str, Any]],
                 candidate: list[dict[str, Any]], *,
                 require_frozen_evidence: bool = True) -> dict[str, dict[str, Any]]:
    if len(candidate) != len(baseline):
        raise ValueError(
            f"{name}: question count {len(candidate)} != {len(baseline)}")
    by_id = {str(row["question_id"]): row for row in candidate}
    if len(by_id) != len(candidate):
        raise ValueError(f"{name}: duplicate question IDs")
    for source in baseline:
        question_id = str(source["question_id"])
        row = by_id.get(question_id)
        if row is None:
            raise ValueError(f"{name}: missing {question_id}")
        if row.get("memory_id") != source.get("memory_id"):
            raise ValueError(f"{name}: memory changed for {question_id}")
        if (require_frozen_evidence
                and row.get("evidence_turn_ids") != source.get("evidence_turn_ids")):
            raise ValueError(f"{name}: evidence/order changed for {question_id}")
    return by_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--metadata-answers", type=Path, required=True,
                        help="Only question and question_id are read")
    parser.add_argument("--preference", type=Path, required=True)
    parser.add_argument("--date-focus", type=Path, required=True)
    parser.add_argument("--relation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-token-increase", type=int, default=500)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    baseline = read_jsonl(args.baseline)
    metadata_rows = read_jsonl(args.metadata_answers)
    metadata = {str(row["question_id"]): {
        "question_id": str(row["question_id"]),
        "question": row.get("question", ""),
    } for row in metadata_rows}
    if set(metadata) != {str(row["question_id"]) for row in baseline}:
        raise ValueError("metadata and baseline question IDs differ")
    preference = validate_arm(
        "preference", baseline, read_jsonl(args.preference))
    date_focus = validate_arm(
        "date_focus", baseline, read_jsonl(args.date_focus),
        require_frozen_evidence=False)
    relation = validate_arm(
        "relation", baseline, read_jsonl(args.relation))

    output: list[dict[str, Any]] = []
    routes: dict[str, int] = {}
    deltas: list[int] = []
    for source in baseline:
        question_id = str(source["question_id"])
        question = question_of(metadata[question_id])
        selected = source
        route = "safe64"
        if route_preference(preference[question_id], source):
            selected = preference[question_id]
            route = "domain_preference_focus"
        else:
            relation_route = route_relation(question, relation[question_id])
            if relation_route is not None:
                selected = relation[question_id]
                route = relation_route
            elif route_date_focus(question, date_focus[question_id]):
                selected = date_focus[question_id]
                route = "date_query_focus"

        delta = int(selected.get("packing_prompt_tokens") or 0) - int(
            source.get("packing_prompt_tokens") or 0)
        if delta > args.max_token_increase:
            selected = source
            route = "safe64_budget_fallback"
            delta = 0
        row = dict(selected)
        trace = dict(row.get("trace") or {})
        trace.update({
            "v5_63_selective_route": route,
            "v5_63_source_prompt_payload_hash": source.get(
                "prompt_payload_hash"),
            "v5_63_token_delta": delta,
        })
        row["trace"] = trace
        output.append(row)
        routes[route] = routes.get(route, 0) + 1
        if row.get("prompt_payload_hash") != source.get("prompt_payload_hash"):
            deltas.append(delta)

    prepared_path = args.output / "prepared_answers.jsonl"
    write_jsonl(prepared_path, output)
    manifest = {
        "schema_version": VERSION,
        "baseline": str(args.baseline),
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "output": str(prepared_path),
        "output_sha256": hashlib.sha256(prepared_path.read_bytes()).hexdigest(),
        "questions": len(output),
        "changed_prompts": len(deltas),
        "routes": dict(sorted(routes.items())),
        "max_token_increase": args.max_token_increase,
        "token_delta": {
            "mean_changed": sum(deltas) / max(1, len(deltas)),
            "max": max(deltas, default=0),
            "min": min(deltas, default=0),
            "mean_all_questions": sum(deltas) / max(1, len(output)),
        },
        "uses_predictions_gold_categories_or_judges": False,
        "evidence_turn_ids_and_order_frozen": all(
            row.get("evidence_turn_ids") == source.get("evidence_turn_ids")
            for row, source in zip(output, baseline)),
        "date_query_focus_may_repack_within_same_64_turn_budget": True,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
