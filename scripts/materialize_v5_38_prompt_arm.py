#!/usr/bin/env python3
"""Create a budget-nonincreasing prompt arm from frozen PreparedAnswer rows.

The transform deliberately keeps evidence ids, order, spans, and graph layout
byte-identical.  It only (1) hides the global question date when the query has
no deictic phrase, (2) compacts the graph-label glossary, and (3) repeats the
question/source-time guard after the evidence.  Every non-deterministic request
must use no more packing tokens than its source row.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.answer import prompt_contract, question_needs_global_date  # noqa: E402
from graphmem.domain import canonical_json  # noqa: E402
from graphmem.text import content_terms, predicate_family  # noqa: E402
from graphmem.tokenization import ExactTokenCounter  # noqa: E402


FOOTER_MARKER = "Answer the original Question now:"
_QUESTION_HEADER_RE = re.compile(
    r"\A(?:Question date: (?P<date>[^\n]*)\n)?Question: "
    r"(?P<question>.*?)\n\nQuery operation:",
    re.S,
)
_TOPOLOGY_LABEL_RE = re.compile(
    r"^\[(?:CHAIN (?P<chain>\d+) step=(?P<chain_step>\d+)|"
    r"CHAIN (?P<support>\d+) support|"
    r"GRAPH (?P<graph>\d+) step=(?P<graph_step>\d+)|"
    r"AUX (?P<aux>\d+) rank=(?P<aux_rank>\d+))\]",
    re.M,
)
_EVIDENCE_LINE_RE = re.compile(
    r"^(?P<label>\[(?:CHAIN \d+ step=\d+|CHAIN \d+ support|"
    r"GRAPH \d+ step=\d+|AUX \d+ rank=\d+)\]) (?P<body>[^\n]+)$",
    re.M,
)
_RENDERED_HEADER_RE = re.compile(
    r"^(?P<header>\[[^\]]+\]\s+[^:]{1,64}:)\s*(?P<text>.*)$")
_SENTENCE_RE = re.compile(r"[^.!?。！？]+(?:[.!?。！？]+|$)")
_TEMPORAL_QUERY_RE = re.compile(
    r"\b(?:when|date|time|days?|weeks?|months?|years?|before|after|since|"
    r"first|last|latest|earliest|ago|order)\b", re.I)
_COUNT_QUERY_RE = re.compile(
    r"\b(?:how many|count|total|sum|average|difference)\b", re.I)
_TIME_VALUE_RE = re.compile(
    r"\b(?:20\d\d|january|february|march|april|may|june|july|august|"
    r"september|october|november|december|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday|today|yesterday|tomorrow|last|next)\b",
    re.I,
)
_NUMBER_RE = re.compile(r"(?:[$£€¥]\s*)?\b\d+(?:[.,]\d+)?\b")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _nearest(values: list[int], p: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)] if ordered else 0


def _stats(values: list[int]) -> dict:
    return {
        "count": len(values),
        "mean": sum(values) / max(1, len(values)),
        "p50": _nearest(values, 0.50),
        "p95": _nearest(values, 0.95),
        "p99": _nearest(values, 0.99),
        "max": max(values, default=0),
        "unit": "packing_tokens_per_question",
        "percentile_method": "nearest_rank",
    }


def _contract_flags(version: str) -> dict[str, bool]:
    return {
        "normalize_relative_time": "source-time" in version,
        "precision_grounding": "grounded-answer" in version,
        "topological_layout": "topological-evidence" in version,
        "aggregation_ledger": "aggregation-ledger" in version,
        "preference_synthesis": "preference-synthesis" in version,
        "exact_grounding_footer": "exact-grounding-footer" in version,
    }


def _compact_topology_labels(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        if match.group("chain"):
            return f"[C{match.group('chain')}.{match.group('chain_step')}]"
        if match.group("support"):
            return f"[C{match.group('support')}.S]"
        if match.group("graph"):
            return f"[G{match.group('graph')}.{match.group('graph_step')}]"
        return f"[A{match.group('aux')}.{match.group('aux_rank')}]"
    return _TOPOLOGY_LABEL_RE.sub(replace, text)


def _clip_words(text: str, limit: int = 220) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    clipped = text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:")
    return clipped + "..."


def _focus_terms(text: str) -> frozenset[str]:
    terms = content_terms(text) - {
        "he", "her", "hers", "him", "his", "its", "she", "them", "theirs",
    }
    return frozenset((*terms, *(predicate_family(term) for term in terms)))


def _query_focus_index(question: str, user_prompt: str,
                       counter: ExactTokenCounter, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    matches = list(_EVIDENCE_LINE_RE.finditer(user_prompt))
    if not matches:
        return ""
    query_terms = _focus_terms(question)
    line_terms = [_focus_terms(match.group("body")) for match in matches]
    document_frequency = Counter(
        term for terms in line_terms for term in (query_terms & terms))
    quoted = tuple(
        phrase.casefold() for phrase in re.findall(r"['\"]([^'\"]{3,})['\"]", question))
    temporal = bool(_TEMPORAL_QUERY_RE.search(question))
    numeric = bool(_COUNT_QUERY_RE.search(question))
    candidates: list[tuple[float, int, str, frozenset[str], str]] = []
    total_lines = len(matches)
    for index, (match, terms) in enumerate(zip(matches, line_terms)):
        body = match.group("body")
        parsed = _RENDERED_HEADER_RE.match(body)
        header = parsed.group("header") if parsed else ""
        source_text = parsed.group("text") if parsed else body
        sentences = [row.group(0).strip() for row in _SENTENCE_RE.finditer(
            source_text) if row.group(0).strip()] or [source_text]
        sentence_rows: list[tuple[float, int, str, frozenset[str], bool]] = []
        for sentence_index, sentence in enumerate(sentences):
            sentence_terms = _focus_terms(header + " " + sentence)
            overlap = query_terms & sentence_terms
            if "get" in overlap and re.search(r"\bget(?:s|ting)? along\b", sentence, re.I):
                overlap = frozenset(set(overlap) - {"get"})
            weighted_overlap = sum(
                1.0 + math.log((total_lines + 1) /
                               (document_frequency.get(term, 0) + 1))
                for term in overlap)
            score = 4.0 * weighted_overlap
            score += 5.0 * sum(phrase in sentence.casefold() for phrase in quoted)
            if temporal and _TIME_VALUE_RE.search(sentence):
                score += 2.5
            if temporal and _NUMBER_RE.search(sentence):
                # Durations often encode the requested date indirectly (for
                # example "had Max for 10 years").  Treat that operand as a
                # temporal endpoint, not as an incidental number.
                score += 4.5
            if numeric and _NUMBER_RE.search(sentence):
                score += 2.0
            critical = bool(
                (temporal and (_TIME_VALUE_RE.search(sentence)
                               or _NUMBER_RE.search(sentence)))
                or (numeric and _NUMBER_RE.search(sentence)))
            sentence_rows.append(
                (score, sentence_index, sentence, overlap, critical))
        best = max(sentence_rows, key=lambda item: (
            item[0], -len(item[2]), -item[1]))
        excerpt_rows = [best]
        critical_rows = [row for row in sentence_rows
                         if row[4] and row[1] != best[1]]
        if critical_rows:
            critical = max(critical_rows, key=lambda item: (
                item[0], -abs(item[1] - best[1]), -item[1]))
            excerpt_rows.append(critical)
        excerpt_rows.sort(key=lambda item: item[1])
        excerpt = " ".join(row[2] for row in excerpt_rows)
        overlap = frozenset(term for row in excerpt_rows for term in row[3])
        score = best[0]
        if len(excerpt_rows) > 1:
            score += 0.5 * excerpt_rows[1][0] + 2.0
        label = match.group("label")
        label_score = 0.0
        if label.startswith("[CHAIN"):
            label_score = 2.0
        elif label.startswith("[GRAPH"):
            label_score = 1.0
        else:
            rank = re.search(r"rank=(\d+)", label)
            label_score = 1.0 / max(1, int(rank.group(1))) if rank else 0.0
        score += label_score + 1.0 / (index + 1)
        if not overlap and label_score < 2.0:
            continue
        excerpt = _clip_words(excerpt)
        rendered = f"{header} {excerpt}".strip()
        session = header.split(" @ ", 1)[0] if header else f"line:{index}"
        candidates.append((score, index, rendered, overlap, session))

    title = "Query focus (exact excerpts from packed memories; verify full blocks):"
    selected: list[str] = []
    seen_text: set[str] = set()
    seen_session: Counter[str] = Counter()
    covered_terms: set[str] = set()
    pending = list(candidates)
    while pending and len(selected) < 6:
        def marginal(item: tuple[float, int, str, frozenset[str], str]):
            score, index, rendered, overlap, session = item
            novelty = len(set(overlap) - covered_terms)
            session_penalty = 1.5 * seen_session[session]
            return score + 2.0 * novelty - session_penalty, -index
        item = max(pending, key=marginal)
        pending.remove(item)
        _score, _index, rendered, overlap, session = item
        normalized = " ".join(rendered.casefold().split())
        if normalized in seen_text:
            continue
        proposal = selected + [f"[F{len(selected) + 1}] {rendered}"]
        text = "\n".join((title, *proposal))
        if counter.count(text) > max_tokens:
            continue
        selected = proposal
        seen_text.add(normalized)
        seen_session[session] += 1
        covered_terms.update(overlap)
    return "\n".join((title, *selected)) if selected else ""


def transform(row: dict, counter: ExactTokenCounter, *, scope: str = "all",
              compact_labels: bool = False, focus_index_tokens: int = 0,
              ) -> tuple[dict, int]:
    output = dict(row)
    messages = [dict(message) for message in row.get("messages", ())]
    if not messages:
        return output, 0
    if len(messages) != 2 or messages[0].get("role") != "system" \
            or messages[1].get("role") != "user":
        raise ValueError(f"unexpected message contract for {row['question_id']}")
    trace = dict(row.get("trace", {}))
    source_version = str(trace.get("prompt_version") or "")
    flags = _contract_flags(source_version)
    if scope == "default" and (
            flags["aggregation_ledger"] or flags["preference_synthesis"]):
        return output, 0
    if not flags["topological_layout"]:
        raise ValueError(f"source prompt is not topological for {row['question_id']}")
    old_version, old_system, old_hash = prompt_contract(**flags)
    if source_version != old_version or messages[0]["content"] != old_system \
            or str(row.get("prompt_hash") or "") != old_hash:
        raise ValueError(f"source prompt contract mismatch for {row['question_id']}")

    user = str(messages[1]["content"])
    if FOOTER_MARKER in user:
        raise ValueError(f"source already has recency footer for {row['question_id']}")
    match = _QUESTION_HEADER_RE.match(user)
    if match is None:
        raise ValueError(f"cannot parse question header for {row['question_id']}")
    question = match.group("question")
    suffix = user[match.end() - len("Query operation:"):]
    focus_index = _query_focus_index(
        question, user, counter, focus_index_tokens)
    if compact_labels:
        suffix = _compact_topology_labels(suffix)
    header = f"Question: {question}"
    if question_needs_global_date(question):
        header = f"Question date: {match.group('date') or 'unknown'}\n" + header
    footer = (
        f"\n\n{FOOTER_MARKER} {question}\n"
        "Resolve relative time inside each memory only from that memory's own "
        "date or [source-time] annotation. Return the concise answer once; do "
        "not quote evidence or repeat the conclusion."
    )
    focus_section = f"\n\n{focus_index}" if focus_index else ""
    messages[1]["content"] = header + "\n\n" + suffix + focus_section + footer
    new_version, new_system, new_hash = prompt_contract(
        **flags, contextual_question_date=True,
        question_recency_footer=True, compact_topological_contract=True,
        compact_topological_labels=compact_labels,
        query_focus_index=bool(focus_index))
    messages[0]["content"] = new_system
    prompt_tokens = sum(counter.count(message["content"]) for message in messages)
    source_tokens = int(row.get("packing_prompt_tokens") or 0)
    if prompt_tokens > source_tokens:
        raise ValueError(
            f"prompt budget increased for {row['question_id']}: "
            f"{source_tokens} -> {prompt_tokens}")

    trace.update({
        "prompt_version": new_version,
        "question_date_mode": "query_relative",
        "question_date_included": question_needs_global_date(question),
        "question_recency_footer": True,
        "compact_topological_contract": True,
        "compact_topological_labels": compact_labels,
        "query_focus_index": bool(focus_index),
        "query_focus_index_tokens": (
            counter.count(focus_index) if focus_index else 0),
        "prompt_source_payload_hash": row.get("prompt_payload_hash"),
    })
    output.update({
        "messages": messages,
        "packing_prompt_tokens": prompt_tokens,
        "prompt_hash": new_hash,
        "prompt_payload_hash": hashlib.sha256(
            canonical_json(messages).encode()).hexdigest(),
        "trace": trace,
        "preparation_latency_ms": 0.0,
    })
    return output, prompt_tokens - source_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--scope", choices=("all", "default"), default="all")
    parser.add_argument("--compact-labels", action="store_true")
    parser.add_argument("--query-focus-index-tokens", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = _read_jsonl(args.source)
    ids = [str(row["question_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("source PreparedAnswer artifact has duplicate question IDs")
    counter = ExactTokenCounter("frozen-packing-model", args.tokenizer)
    transformed: list[dict] = []
    deltas: list[int] = []
    for row in rows:
        result, delta = transform(
            row, counter, scope=args.scope,
            compact_labels=args.compact_labels,
            focus_index_tokens=args.query_focus_index_tokens)
        transformed.append(result)
        deltas.append(delta)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(
        json.dumps(row, ensure_ascii=True) + "\n" for row in transformed),
        encoding="utf-8")
    source_tokens = [int(row.get("packing_prompt_tokens") or 0) for row in rows]
    output_tokens = [int(row.get("packing_prompt_tokens") or 0)
                     for row in transformed]
    manifest = {
        "schema_version": "graphmem-v5.38-frozen-prompt-arm-v1",
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "output": str(args.output),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "questions": len(rows),
        "scope": args.scope,
        "compact_labels": args.compact_labels,
        "query_focus_index_tokens": args.query_focus_index_tokens,
        "message_prompts_transformed": sum(
            before.get("prompt_payload_hash") != after.get("prompt_payload_hash")
            for before, after in zip(rows, transformed)),
        "evidence_ids_and_order_frozen": all(
            before.get("evidence_turn_ids") == after.get("evidence_turn_ids")
            for before, after in zip(rows, transformed)),
        "packing_tokens": {
            "source": _stats(source_tokens),
            "candidate": _stats(output_tokens),
            "delta": _stats(deltas),
            "increased_questions": sum(delta > 0 for delta in deltas),
            "unchanged_questions": sum(delta == 0 for delta in deltas),
            "decreased_questions": sum(delta < 0 for delta in deltas),
        },
        "tokenizer": counter.describe(),
    }
    manifest_path = args.manifest or args.output.with_name("prompt_arm_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
