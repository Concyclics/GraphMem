#!/usr/bin/env python3
"""Materialize a domain-scoped preference focus over frozen 64-turn packs.

Only direct source turns already present in each PreparedAnswer are eligible.
The materializer replaces the previous one-row dense-dominated anchor with at
most two domain-matching rows and enforces a 500-token per-question increase.
It never reads answers, gold annotations, benchmark categories, or verdicts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem.domain import SourceTurn, canonical_json  # noqa: E402
from graphmem.storage import SQLiteGraphStore  # noqa: E402
from graphmem.tokenization import resolve_token_counter  # noqa: E402


STOP = frozenset({
    "about", "again", "also", "and", "any", "are", "can", "could", "do",
    "colleague", "colleagues", "gathering", "invite", "inviting", "for",
    "from", "have", "help", "i", "i'm", "in", "interesting", "is",
    "it", "me", "my", "new", "of", "on", "please", "recommend",
    "over", "recommendation", "small", "some", "suggest", "suggestion", "that", "the",
    "this", "think", "thinking", "tips", "to", "upcoming", "what", "with",
    "would", "you", "your",
})

ALIASES = (
    ({"publication", "conference"},
     {"paper", "article", "research", "workshop", "symposium", "journal"}),
    ({"hotel", "trip"},
     {"hotel", "accommodation", "room", "view", "pool", "balcony", "suite"}),
    ({"show", "movie", "watch"},
     {"netflix", "comedy", "standup", "special", "documentary", "series", "film"}),
    ({"bake", "baking"}, {"cake", "dessert", "pastry", "cookie", "recipe"}),
    ({"furniture", "bedroom"}, {"dresser", "bed", "decor", "design", "style"}),
    ({"creamer", "coffee"}, {"creamer", "almond", "vanilla", "milk", "honey"}),
    ({"nas", "storage"}, {"nas", "storage", "backup", "drive", "network"}),
    ({"meal", "prep"},
     {"quinoa", "vegetable", "cook", "food", "protein", "lunch", "dinner"}),
    ({"phone", "battery"}, {"phone", "power", "charger", "charging", "bank"}),
    ({"photography", "photo"}, {"camera", "lens", "flash", "tripod"}),
)

START = "Grounded user anchors (verbatim excerpts from packed memories; reading index only):"
END = "Grounded recommendation check:"


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def normalize(token: str) -> str:
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        value = token[:-3]
        return value[:-1] if len(value) > 3 and value[-1:] == value[-2:-1] else value
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def terms(text: str) -> set[str]:
    return {
        normalize(token) for token in re.findall(r"[a-z0-9']+", text.casefold())
        if len(token) > 2 and token not in STOP
    }


def query_terms(question: str) -> set[str]:
    # Expand only from the literal query.  Recursive alias closure lets a
    # generic bridge such as ``recipe`` pull in an unrelated meal-prep domain
    # after ``bake -> recipe``, which is exactly the drift this stage prevents.
    base = terms(question)
    result = set(base)
    for triggers, values in ALIASES:
        normalized_triggers = {normalize(value) for value in triggers}
        if base & normalized_triggers:
            result.update(normalize(value) for value in values)
    return result


def excerpt(text: str, anchors: set[str], limit: int = 340) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    lowered = compact.casefold()
    positions = [lowered.find(term) for term in anchors if lowered.find(term) >= 0]
    center = min(positions) if positions else 0
    start = max(0, min(len(compact) - limit, center - limit // 3))
    end = min(len(compact), start + limit)
    if start:
        boundary = compact.find(" ", start, min(end, start + 40))
        start = boundary + 1 if boundary >= 0 else start
    if end < len(compact):
        boundary = compact.rfind(" ", max(start, end - 40), end)
        end = boundary if boundary > start else end
    return ("..." if start else "") + compact[start:end].strip() + (
        "..." if end < len(compact) else "")


def routed_domain(question: str) -> bool:
    base = terms(question)
    return any(base & {normalize(value) for value in triggers}
               for triggers, _values in ALIASES)


def focus(question: str, packed: list[SourceTurn], limit: int = 1) -> tuple[str, list[str]]:
    q_terms = query_terms(question)
    row_terms = {turn.turn_id: terms(turn.raw_text) for turn in packed}
    df: dict[str, int] = {}
    for values in row_terms.values():
        for value in values:
            df[value] = df.get(value, 0) + 1
    ranked: list[tuple[float, int, SourceTurn]] = []
    for rank, turn in enumerate(packed):
        if (turn.role or "").casefold().strip() not in {"user", "human"}:
            continue
        overlap = q_terms & row_terms[turn.turn_id]
        if not overlap:
            continue
        lexical = sum(math.log((len(packed) + 1) / (df[value] + 0.5))
                      for value in overlap)
        personal = bool(re.search(
            r"\b(?:i(?:'ve|'m)?|my)\b.{0,80}\b(?:have|had|use|using|like|love|"
            r"prefer|enjoy|want|need|got|bought|grow|made|trying|struggl|issue)",
            turn.raw_text, re.I))
        # Exact domain overlap is authoritative; original packed order is only
        # a small stable tie-breaker so a broad dense match cannot dominate it.
        score = 3.0 * lexical + 1.5 * len(overlap) + (1.0 if personal else 0.0)
        # Very long turns (papers, transcripts, imported documents) overlap
        # many domains by chance.  Their overlap density is lower than a short
        # direct statement of a user preference or constraint.
        score -= min(6.0, len(turn.raw_text) / 800.0)
        score += 0.25 / (rank + 1)
        ranked.append((score, rank, turn))
    ranked.sort(key=lambda value: (-value[0], value[1], value[2].turn_id))
    chosen: list[SourceTurn] = []
    seen: set[str] = set()
    target = max(1, limit)
    for _score, _rank, turn in ranked:
        value = " ".join(turn.raw_text.casefold().split())
        if value in seen:
            continue
        chosen.append(turn); seen.add(value)
        # If the best match is an imported long document, retain one short
        # direct witness as a second anchor.  Long documents win lexical recall
        # by mentioning many domains but should not stand alone as preference.
        if len(chosen) == 1 and len(turn.raw_text) > 1000:
            target = max(target, 2)
        if len(chosen) >= target:
            break
    if not chosen:
        return "", []
    rendered = [START]
    for turn in chosen:
        rendered.append(f"- {turn.speaker}: {excerpt(turn.raw_text, q_terms)}")
    return "\n".join(rendered), [turn.turn_id for turn in chosen]


def replace_focus(user: str, replacement: str) -> str:
    if START not in user or END not in user:
        return user
    before, tail = user.split(START, 1)
    _old, after = tail.split(END, 1)
    return before + replacement + "\n\n" + END + after


def write(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(value, ensure_ascii=True) + "\n"
                            for value in values), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True,
                        help="Question metadata only; predictions are never read")
    parser.add_argument("--source-db", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packing-model", required=True)
    parser.add_argument("--max-token-increase", type=int, default=500)
    parser.add_argument("--focus-limit", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    prepared = rows(args.prepared)
    metadata = {str(row["question_id"]): row for row in rows(args.answers)}
    store = SQLiteGraphStore(args.source_db, read_only=True)
    counter = resolve_token_counter(args.packing_model, require_exact=True)
    memory_cache: dict[str, dict[str, SourceTurn]] = {}
    output: list[dict[str, Any]] = []
    changed = 0
    deltas: list[int] = []
    for source in prepared:
        row = dict(source)
        trace = dict(row.get("trace") or {})
        if (not trace.get("preference_synthesis")
                or not routed_domain(str(metadata.get(str(row["question_id"]), {}).get(
                    "question") or ""))):
            output.append(row); continue
        qid = str(row["question_id"])
        question = str(metadata[qid].get("question") or "")
        memory_id = str(row["memory_id"])
        if memory_id not in memory_cache:
            memory_cache[memory_id] = {
                turn.turn_id: turn for turn in store.turns(memory_id)}
        by_id = memory_cache[memory_id]
        packed = [by_id[turn_id] for turn_id in row.get("evidence_turn_ids", ())
                  if turn_id in by_id]
        rendered, turn_ids = focus(question, packed, limit=args.focus_limit)
        messages = [dict(value) for value in row.get("messages", ())]
        if not rendered or not messages:
            output.append(row); continue
        messages[-1]["content"] = replace_focus(
            str(messages[-1].get("content") or ""), rendered)
        new_tokens = sum(counter.count(str(value.get("content") or ""))
                         for value in messages)
        delta = new_tokens - int(row.get("packing_prompt_tokens") or 0)
        if delta > args.max_token_increase:
            output.append(row); continue
        row["messages"] = messages
        row["packing_prompt_tokens"] = new_tokens
        row["prompt_payload_hash"] = hashlib.sha256(
            canonical_json(messages).encode()).hexdigest()
        trace.update({
            "preference_focus_strategy": f"domain_idf_top{args.focus_limit}",
            "preference_focus_turn_ids": turn_ids,
            "preference_focus_turns": len(turn_ids),
            "preference_focus_token_delta": delta,
        })
        row["trace"] = trace
        output.append(row); changed += 1; deltas.append(delta)

    write(args.output / "prepared_answers.jsonl", output)
    manifest = {
        "schema_version": "graphmem-v5.62-domain-preference-focus-v1",
        "questions": len(output), "changed": changed,
        "max_token_increase": args.max_token_increase,
        "token_delta": {
            "mean_changed": sum(deltas) / max(1, len(deltas)),
            "max": max(deltas, default=0), "min": min(deltas, default=0)},
        "uses_answers_gold_or_judge": False,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
