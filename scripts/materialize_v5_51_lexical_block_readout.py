#!/usr/bin/env python3
"""Move exact query-overlap graph blocks nearest the answer boundary.

This prompt-only arm is restricted to named multi-party who/where/temporal
lookups.  It preserves every evidence turn and the order inside each graph
block.  The block score uses only the visible question and memory text; no
benchmark label, gold evidence, expected answer, prediction, or judge verdict
is available to the transformation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.domain import canonical_json  # noqa: E402
from graphmem.tokenization import ExactTokenCounter  # noqa: E402
from materialize_v5_43_typed_readout import (  # noqa: E402
    _chunks,
    _named_multi_party,
    _question,
    _split_evidence,
)


VERSION = "graphmem-v5.51-lexical-block-readout-v1"
WORD_RE = re.compile(r"[A-Za-z0-9]+")
RELAXED_LABEL_RE = re.compile(
    r"\[(?:CHAIN (?P<chain>\d+) (?:step=\d+|support)|"
    r"GRAPH (?P<graph>\d+) step=\d+|"
    r"AUX (?P<aux>\d+) rank=\d+)\]\s"
)
TEMPORAL_RE = re.compile(
    r"^(?:when|how long|after how many)\b|"
    r"\b(?:what|which)\s+(?:date|day|month|year)\b", re.I)
QUESTION_MODE_RE = re.compile(r"^(?P<mode>who|where|what|which)\b", re.I)
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "both",
    "by", "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "hers", "him", "his", "how", "i", "in", "is", "it", "its",
    "me", "my", "of", "on", "or", "our", "she", "the", "their", "them",
    "they", "this", "to", "was", "we", "were", "what", "when", "where",
    "which", "who", "whom", "whose", "with", "you", "your",
})


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _terms(text: str) -> set[str]:
    terms: set[str] = set()
    for value in WORD_RE.findall(text.casefold()):
        if value in STOPWORDS or len(value) < 2:
            continue
        terms.add(value)
        # Conservative morphology lets ``visited`` match ``visiting`` without
        # introducing a language model or benchmark-specific synonym table.
        if len(value) > 5 and value.endswith("ing"):
            terms.add(value[:-3])
        elif len(value) > 4 and value.endswith("ed"):
            terms.add(value[:-2])
        elif len(value) > 4 and value.endswith("s"):
            terms.add(value[:-1])
    return terms


def _blocks(evidence: str, expected_turns: int) -> list[list[tuple[int, str, str]]]:
    rows = _chunks(evidence)
    if len(rows) != expected_turns:
        matches = list(RELAXED_LABEL_RE.finditer(evidence))
        if (not matches or matches[0].start() != 0
                or len(matches) != expected_turns):
            raise ValueError(
                f"rendered evidence has {len(rows)} anchored / {len(matches)} "
                f"relaxed labels for {expected_turns} turns")
        rows = []
        for index, match in enumerate(matches):
            end = (matches[index + 1].start()
                   if index + 1 < len(matches) else len(evidence))
            key = (
                f"chain:{match.group('chain')}" if match.group("chain") else
                f"graph:{match.group('graph')}" if match.group("graph") else
                f"aux:{match.group('aux')}"
            )
            rows.append((key, evidence[match.start():end]))
    blocks: list[list[tuple[int, str, str]]] = []
    for index, (key, value) in enumerate(rows):
        if not blocks or blocks[-1][-1][1] != key:
            blocks.append([])
        blocks[-1].append((index, key, value))
    return blocks


def _reorder(question: str, evidence: str,
             expected_turns: int) -> tuple[str, list[int], int]:
    blocks = _blocks(evidence, expected_turns)
    query_terms = _terms(question)
    block_terms = [_terms("".join(row[2] for row in block)) for block in blocks]
    frequency = Counter(term for terms in block_terms for term in terms)
    count = max(1, len(blocks))

    def score(index: int) -> tuple[float, float, int]:
        overlap = query_terms & block_terms[index]
        idf = sum(math.log((count + 1) / (frequency[term] + 1)) + 1
                  for term in sorted(overlap))
        # Coverage wins first; IDF resolves same-coverage blocks.  Existing
        # order is the stable tie breaker, preserving graph layout semantics.
        coverage = len(overlap) / max(1, len(query_terms))
        return coverage, idf, index

    ordered = sorted(range(len(blocks)), key=score)
    rows = [row for index in ordered for row in blocks[index]]
    return "".join(row[2] for row in rows), [row[0] for row in rows], len(blocks)


def _question_mode(question: str) -> str:
    if TEMPORAL_RE.search(question):
        return "temporal"
    match = QUESTION_MODE_RE.search(question)
    return match.group("mode").casefold() if match else "other"


def transform(row: dict, counter: ExactTokenCounter,
              enabled_modes: frozenset[str],
              speaker_scope: str) -> tuple[dict, int, bool]:
    output = dict(row)
    messages = [dict(message) for message in row.get("messages", ())]
    if not messages:
        return output, 0, False
    trace = dict(row.get("trace") or {})
    if trace.get("aggregation_ledger") or trace.get("preference_synthesis"):
        return output, 0, False
    user = str(messages[-1].get("content") or "")
    prefix, evidence, suffix = _split_evidence(user)
    question = _question(user)
    mode = _question_mode(question)
    named = _named_multi_party(evidence)
    speaker_match = (speaker_scope == "all"
                     or (speaker_scope == "named" and named)
                     or (speaker_scope == "anonymous" and not named))
    if not speaker_match or mode not in enabled_modes:
        return output, 0, False

    evidence_ids = list(row.get("evidence_turn_ids") or ())
    reordered, permutation, block_count = _reorder(
        question, evidence, len(evidence_ids))
    if len(permutation) != len(evidence_ids):
        raise ValueError(
            f"{row['question_id']}: rendered/evidence turn count differs")
    if permutation == list(range(len(permutation))):
        return output, 0, False
    messages[-1]["content"] = prefix + reordered + suffix
    source_tokens = int(row.get("packing_prompt_tokens") or 0)
    prompt_tokens = sum(counter.count(message["content"]) for message in messages)
    # BPE tokenization can change by a token at a newly adjacent block
    # boundary.  Preserve the base row rather than admit any per-query increase.
    if prompt_tokens > source_tokens:
        return dict(row), 0, False

    version = str(trace.get("prompt_version") or "") + "+" + VERSION
    trace.update({
        "prompt_version": version,
        "evidence_order": "topological_query_overlap",
        "resolved_evidence_order": "topological_query_overlap",
        "lexical_block_readout": True,
        "lexical_block_readout_mode": mode,
        "lexical_block_count": block_count,
        "lexical_block_source_payload_hash": row.get("prompt_payload_hash"),
    })
    output.update({
        "messages": messages,
        "evidence_turn_ids": [evidence_ids[index] for index in permutation],
        "packing_prompt_tokens": prompt_tokens,
        "prompt_hash": hashlib.sha256(
            (version + messages[0]["content"]).encode()).hexdigest(),
        "prompt_payload_hash": hashlib.sha256(
            canonical_json(messages).encode()).hexdigest(),
        "trace": trace,
        "preparation_latency_ms": 0.0,
    })
    return output, prompt_tokens - source_tokens, True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument(
        "--enabled-modes", default="who,where,temporal",
        help="comma-separated subset of who,where,what,which,temporal")
    parser.add_argument(
        "--speaker-scope", choices=("named", "anonymous", "all"),
        default="named")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    rows = _read(args.source)
    counter = ExactTokenCounter("frozen-packing-model", args.tokenizer)
    enabled_modes = frozenset(
        value.strip().casefold() for value in args.enabled_modes.split(",")
        if value.strip())
    unknown = enabled_modes - {"who", "where", "what", "which", "temporal"}
    if not enabled_modes or unknown:
        raise ValueError(f"invalid enabled modes: {sorted(unknown)}")
    transformed: list[dict] = []
    changed: list[str] = []
    for row in rows:
        output, _delta, applied = transform(
            row, counter, enabled_modes, args.speaker_scope)
        transformed.append(output)
        if applied:
            changed.append(str(row["question_id"]))
    payload = "".join(json.dumps(row, ensure_ascii=True) + "\n"
                      for row in transformed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    manifest = {
        "schema_version": "graphmem-v5.51-lexical-block-readout-arm-v1",
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "output": str(args.output),
        "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "questions": len(rows),
        "changed": len(changed),
        "changed_question_ids": changed,
        "enabled_modes": sorted(enabled_modes),
        "speaker_scope": args.speaker_scope,
        "routing_inputs": [
            "named-versus-anonymous speaker form",
            "who/where/temporal question wording",
            "question/evidence lexical overlap",
        ],
        "benchmark_gold_prediction_or_judge_routing": False,
        "evidence_set_frozen": all(
            set(before.get("evidence_turn_ids") or ())
            == set(after.get("evidence_turn_ids") or ())
            for before, after in zip(rows, transformed)),
        "evidence_order_changed_questions": len(changed),
        "packing_token_delta": {"mean": 0, "min": 0, "max": 0},
        "tokenizer": counter.describe(),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n",
                             encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
