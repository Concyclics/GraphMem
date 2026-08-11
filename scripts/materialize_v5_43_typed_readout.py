#!/usr/bin/env python3
"""Materialize a budget-nonincreasing typed-readout prompt arm.

The source V5.40 prompt already repeats the question after the evidence.  This
arm replaces only the generic final reminder with a short instruction selected
from the observable question form.  For named multi-party memories, the
previously measured V5.42 strongest-last layout is retained only for
what/when/who/which questions.  Aggregation-ledger and preference-synthesis
requests remain byte-identical because they own their readout semantics.

No benchmark label, expected answer, gold evidence, or judge outcome is read.
The evidence set is frozen, and every changed prompt is required to use no more
packing tokens than its source row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.domain import canonical_json  # noqa: E402
from graphmem.tokenization import ExactTokenCounter  # noqa: E402


VERSION = "graphmem-v5.43-typed-readout-v1"
RECENCY_VERSION = "graphmem-v5.42-topological-recency-readout-v1"
MEMORY_MARKER = "Conversation memories:\n"
FOOTER_MARKER = "\n\nAnswer the original Question now:"
END_MARKERS = (
    "\n\nAggregation ledger (",
    "\n\nOutput contract:",
    "\n\nFinal check:",
    FOOTER_MARKER,
)
LABEL_RE = re.compile(
    r"^\[(?:CHAIN (?P<chain>\d+) (?:step=\d+|support)|"
    r"GRAPH (?P<graph>\d+) step=\d+|"
    r"AUX (?P<aux>\d+) rank=\d+)\]\s",
    re.M,
)
QUESTION_RE = re.compile(r"(?:^|\n)Question:\s*(?P<question>[^\n]+)")
NAMED_SPEAKER_RE = re.compile(
    r"^\[(?:CHAIN \d+ (?:step=\d+|support)|GRAPH \d+ step=\d+|"
    r"AUX \d+ rank=\d+)\]\s+\[[^\]]+\]\s+([^:\n]{1,48}):",
    re.M,
)
GENERIC_SPEAKERS = frozenset({"", "assistant", "system", "tool", "user", "human"})
RECENCY_PREFIXES = frozenset({"what", "when", "who", "which"})

_COUNT_RE = re.compile(
    r"\b(?:how many|count|total|sum|average|mean|difference|most|least|"
    r"fewest|number of)\b", re.I)
_DURATION_RE = re.compile(
    r"\b(?:how long|after how many|duration|time between|how much time)\b", re.I)
_TEMPORAL_RE = re.compile(
    r"\b(?:when|what date|which date|what year|which year|what month|"
    r"before|after|first|last|latest|earliest|recent(?:ly)?)\b", re.I)
_LIST_RE = re.compile(
    r"\b(?:which|what)\s+(?:activities|activity|books|classes|events|items|"
    r"locations|places|things|types|ways|games|movies|films|songs|people|"
    r"friends|sports|hobbies|projects|countries|cities|states|foods|meals|"
    r"restaurants|gifts|skills|languages|instruments|festivals|trips)\b",
    re.I,
)
_WHY_RE = re.compile(r"^(?:why|what (?:made|caused|reason)|how come)\b", re.I)


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _nearest(values: list[int], p: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)] if ordered else 0


def _stats(values: list[int]) -> dict:
    return {
        "count": len(values),
        "mean": sum(values) / max(1, len(values)),
        "p50": _nearest(values, .50),
        "p95": _nearest(values, .95),
        "p99": _nearest(values, .99),
        "max": max(values, default=0),
        "percentile_method": "nearest-rank",
    }


def _question(user: str) -> str:
    match = QUESTION_RE.search(user)
    if match is None:
        raise ValueError("question header is absent")
    return " ".join(match.group("question").split())


def _prefix(question: str) -> str:
    words = question.casefold().split()
    return words[0].strip("'\"([{.,?!:;") if words else ""


def _named_multi_party(evidence: str) -> bool:
    return any(match.group(1).casefold().strip() not in GENERIC_SPEAKERS
               for match in NAMED_SPEAKER_RE.finditer(evidence))


def _readout_kind(question: str) -> str:
    if _COUNT_RE.search(question):
        return "collection"
    if _DURATION_RE.search(question):
        return "duration"
    if _WHY_RE.search(question):
        return "causal"
    if _TEMPORAL_RE.search(question):
        return "temporal"
    if _LIST_RE.search(question):
        return "list"
    return "lookup"


def _readout_contract(kind: str) -> str:
    contracts = {
        "collection": (
            "Enumerate distinct qualifying facts, exclude plans and duplicates, "
            "then compute once. Return only the concise answer."),
        "duration": (
            "Match the exact event endpoints, resolve each from its own memory "
            "date, then compute once. Return only the concise answer."),
        "causal": (
            "Use the explicit reason for the exact subject and event; ignore "
            "same-topic explanations. Return only the concise answer."),
        "temporal": (
            "Match the exact subject and event, then use that memory's date or "
            "[source-time]; do not substitute another event. Return only the "
            "concise answer."),
        "list": (
            "Collect every distinct item satisfying the exact subject and "
            "relation; exclude plans, near-matches, and duplicates. Return only "
            "the concise answer."),
        "lookup": (
            "Match the exact subject, relation, and event; ignore same-topic "
            "near-matches. Return only the concise answer."),
    }
    return contracts[kind]


def _split_evidence(user: str) -> tuple[str, str, str]:
    start = user.find(MEMORY_MARKER)
    if start < 0:
        raise ValueError("conversation memory marker is absent")
    evidence_start = start + len(MEMORY_MARKER)
    stops = [position for marker in END_MARKERS
             if (position := user.find(marker, evidence_start)) >= 0]
    evidence_end = min(stops) if stops else len(user)
    return user[:evidence_start], user[evidence_start:evidence_end], user[evidence_end:]


def _chunks(evidence: str) -> list[tuple[str, str]]:
    matches = list(LABEL_RE.finditer(evidence))
    if not matches or matches[0].start() != 0:
        raise ValueError("topological evidence does not begin with a known label")
    rows: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(evidence)
        key = (
            f"chain:{match.group('chain')}" if match.group("chain") else
            f"graph:{match.group('graph')}" if match.group("graph") else
            f"aux:{match.group('aux')}"
        )
        rows.append((key, evidence[match.start():end]))
    return rows


def _reverse_blocks(evidence: str) -> tuple[str, list[int], int]:
    rows = _chunks(evidence)
    blocks: list[list[tuple[int, str, str]]] = []
    for index, (key, value) in enumerate(rows):
        if not blocks or blocks[-1][-1][1] != key:
            blocks.append([])
        blocks[-1].append((index, key, value))
    reversed_rows = [row for block in reversed(blocks) for row in block]
    return ("".join(row[2] for row in reversed_rows),
            [row[0] for row in reversed_rows], len(blocks))


def transform(row: dict, counter: ExactTokenCounter) -> tuple[dict, int, str, bool]:
    output = dict(row)
    messages = [dict(message) for message in row.get("messages", ())]
    if not messages:
        return output, 0, "deterministic", False
    if len(messages) != 2 or messages[1].get("role") != "user":
        raise ValueError(f"unexpected messages for {row['question_id']}")
    trace = dict(row.get("trace") or {})
    # Specialized paths retain their already measured contracts exactly.
    if trace.get("aggregation_ledger") or trace.get("preference_synthesis"):
        return output, 0, "specialized_frozen", False

    user = str(messages[1]["content"])
    prefix_text, evidence, suffix = _split_evidence(user)
    question = _question(user)
    kind = _readout_kind(question)
    recency = bool(
        _named_multi_party(evidence)
        and _prefix(question) in RECENCY_PREFIXES)
    evidence_ids = list(row.get("evidence_turn_ids") or ())
    block_count = 0
    if recency:
        evidence, permutation, block_count = _reverse_blocks(evidence)
        if len(permutation) != len(evidence_ids):
            raise ValueError(
                f"{row['question_id']}: rendered/evidence id length mismatch")
        evidence_ids = [evidence_ids[index] for index in permutation]
        old = "Blocks are ranked by relevance, then graph or time order."
        new = "Blocks run weakest-to-strongest; keep internal order."
        if old not in messages[0]["content"]:
            raise ValueError(
                f"compact topology contract absent for {row['question_id']}")
        messages[0]["content"] = messages[0]["content"].replace(old, new, 1)

    footer_at = suffix.find(FOOTER_MARKER)
    if footer_at < 0:
        raise ValueError(f"question recency footer absent for {row['question_id']}")
    footer_head = suffix[:footer_at]
    footer = (f"{FOOTER_MARKER} {question}\n"
              f"{_readout_contract(kind)}")
    messages[1]["content"] = prefix_text + evidence + footer_head + footer
    source_tokens = int(row.get("packing_prompt_tokens") or 0)
    prompt_tokens = sum(counter.count(message["content"]) for message in messages)
    if prompt_tokens > source_tokens:
        raise ValueError(
            f"{row['question_id']}: prompt tokens increased "
            f"{source_tokens} -> {prompt_tokens}")

    source_version = str(trace.get("prompt_version") or "")
    version_parts = [source_version, VERSION]
    if recency:
        version_parts.append(RECENCY_VERSION)
    prompt_version = "+".join(version_parts)
    prompt_hash = hashlib.sha256(
        (prompt_version + messages[0]["content"]).encode()).hexdigest()
    trace.update({
        "prompt_version": prompt_version,
        "typed_readout": True,
        "typed_readout_kind": kind,
        "recency_layout_gate": recency,
        "recency_layout_blocks": block_count,
        "evidence_order": (
            "topological_recency" if recency else trace.get("evidence_order")),
        "resolved_evidence_order": (
            "topological_recency" if recency
            else trace.get("resolved_evidence_order")),
        "typed_readout_source_payload_hash": row.get("prompt_payload_hash"),
    })
    output.update({
        "messages": messages,
        "evidence_turn_ids": evidence_ids,
        "packing_prompt_tokens": prompt_tokens,
        "prompt_hash": prompt_hash,
        "prompt_payload_hash": hashlib.sha256(
            canonical_json(messages).encode()).hexdigest(),
        "trace": trace,
        "preparation_latency_ms": 0.0,
    })
    return output, prompt_tokens - source_tokens, kind, recency


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    rows = _read(args.source)
    counter = ExactTokenCounter("frozen-packing-model", args.tokenizer)
    transformed: list[dict] = []
    deltas: list[int] = []
    kinds: dict[str, int] = {}
    recency_ids: list[str] = []
    for row in rows:
        output, delta, kind, recency = transform(row, counter)
        transformed.append(output)
        deltas.append(delta)
        kinds[kind] = kinds.get(kind, 0) + 1
        if recency:
            recency_ids.append(str(row["question_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=True) + "\n"
                      for row in transformed)
    args.output.write_text(payload, encoding="utf-8")
    changed = [str(before["question_id"])
               for before, after in zip(rows, transformed)
               if before.get("prompt_payload_hash")
               != after.get("prompt_payload_hash")]
    manifest = {
        "schema_version": "graphmem-v5.43-typed-readout-arm-v1",
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "output": str(args.output),
        "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "questions": len(rows),
        "changed": len(changed),
        "changed_question_ids": changed,
        "specialized_prompts_frozen": kinds.get("specialized_frozen", 0),
        "readout_kinds": dict(sorted(kinds.items())),
        "recency_gate_questions": len(recency_ids),
        "recency_gate_question_ids": recency_ids,
        "evidence_set_frozen": all(
            set(before.get("evidence_turn_ids") or ())
            == set(after.get("evidence_turn_ids") or ())
            for before, after in zip(rows, transformed)),
        "packing_token_delta": _stats(deltas),
        "increased_questions": sum(delta > 0 for delta in deltas),
        "tokenizer": counter.describe(),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
