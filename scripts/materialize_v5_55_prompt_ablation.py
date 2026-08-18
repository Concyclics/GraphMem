#!/usr/bin/env python3
"""Materialize benchmark-neutral V5.55 answer-prompt ablation arms.

The materializer never reads gold answers to construct a prompt.  An optional
question-id allowlist is only a diagnostic-set selector; every transformation
depends exclusively on the frozen question, evidence text, and graph labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ARMS = (
    "lean",
    "lean_route",
    "lean_route_primary_last",
    "lean_route_primary_last_closure",
    "baseline_route",
    "baseline_route_closure",
    "baseline_route_primary_last",
    "baseline_route_primary_last_closure",
    "selective_v1",
)

LEAN_SYSTEM = """You are the final readout stage of a conversation-memory database.
Use only the supplied memories. Prefer direct source statements over summaries or assistant suggestions. Keep the exact subject, relation, event, polarity, completion status, quantity, unit, and time scope. A [source-time] annotation is authoritative; an observation timestamp is not automatically the event time. Graph labels are navigation hints, not facts. Follow the execution card when present. Return one concise final answer and do not expose intermediate analysis."""

ROUTE_CARDS = {
    "lookup": """[EXECUTION CARD: LOOKUP]
Bind the exact subject and requested relation. Same-topic or similar-name evidence is not a match. If no direct or entailing statement supports the requested fact, answer that the information is insufficient.""",
    "state_update": """[EXECUTION CARD: STATE UPDATE]
Collect only states of the exact subject and relation. Order explicit updates by event time, apply replacements or cancellations, and answer from the latest valid state. Do not use a salient older state.""",
    "aggregate": """[EXECUTION CARD: AGGREGATION]
Enumerate all distinct qualifying operands for the exact requested scope. Keep distinct occurrences, remove duplicate mentions, exclude plans and near-matches, preserve units, and apply the requested arithmetic exactly once. A missing required operand is not zero.""",
    "comparison": """[EXECUTION CARD: COMPARISON]
Bind a qualifying value or dated event for every candidate explicitly compared by the question. Compare only after all required candidates are bound. If any candidate lacks the requested value or event, answer that the information is insufficient; never treat the only observed candidate as the winner.""",
    "temporal": """[EXECUTION CARD: TEMPORAL]
Bind the exact requested event and every required temporal endpoint. Resolve relative time from that memory's own date or [source-time], never from an unrelated observation date. Then compute or order the endpoints in the requested unit. If a required endpoint is missing, answer that the information is insufficient.""",
    "preference": """[EXECUTION CARD: PREFERENCE]
Treat demonstrated preferences, goals, and negative constraints as grounding. You may synthesize a useful new recommendation consistent with them; do not abstain merely because the recommendation itself is not already stated in memory. Do not invent a user preference.""",
}

CLOSURE_GATE = """[VERIFICATION GATE]
Before the final answer, silently create one binding row for every required candidate, item, state, or time endpoint and verify each row against an exact memory statement. If any required row is unbound, answer "Insufficient information." For A-versus-B, observing only B never proves B was first; for a sum of X, Y, and Z, missing Z never means zero. Do not output the binding rows."""

_QUESTION_RE = re.compile(r"^Question:\s*(.+)$", re.M)
_BLOCK_RE = re.compile(r"^\[(CHAIN|GRAPH|AUX)\s+([^\s\]]+)")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def canonical(value) -> str:
    # Match graphmem.domain.canonical_json exactly so an unchanged message
    # remains eligible for audited answer/usage reuse.
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"))


def question_from(message: str) -> str:
    match = _QUESTION_RE.search(message)
    if not match:
        raise ValueError("question marker is absent")
    return " ".join(match.group(1).split())


def route(question: str) -> str:
    q = question.casefold()
    if re.search(r"\b(?:recommend|suggest|advice|gift|should i|could i)\b", q):
        return "preference"
    if (re.search(r"\b(?:who|which|what)\b.*\b(?:first|last|most|least|more|less|earlier|later|older|younger)\b", q)
            or re.search(r"\b(?:first|last|most|least)\b", q)
            or re.search(r"\bwho\b.+\bor\b", q)):
        return "comparison"
    if re.search(r"\b(?:current|currently|latest|now|previous|before the update|used to)\b", q):
        return "state_update"
    if (re.search(r"\bwhen\b|\bwhat (?:date|time)\b|\bhow long\b|\bhow old\b", q)
            or re.search(r"\b(?:days?|weeks?|months?|years?) ago\b", q)):
        return "temporal"
    if re.search(r"\b(?:how many|how much|how often|total|sum|increase|decrease|difference|frequency)\b", q):
        return "aggregate"
    return "lookup"


def reverse_evidence_blocks(user: str, evidence_count: int) -> str:
    marker = "Conversation memories:\n"
    if marker not in user:
        raise ValueError("conversation-memory marker is absent")
    head, rest = user.split(marker, 1)
    lines = rest.splitlines(keepends=True)
    tagged = [(index, line) for index, line in enumerate(lines)
              if _BLOCK_RE.match(line)]
    # A source turn with empty rendered text can remain in evidence_turn_ids
    # without producing a visible line.  Reorder every rendered graph-labelled
    # row; do not manufacture text for an invisible turn.
    if not tagged:
        raise ValueError("no graph-labelled evidence lines found")
    selected = tagged[:evidence_count]
    evidence = [line for _, line in selected]
    tail = lines[selected[-1][0] + 1:]

    blocks: list[list[str]] = []
    keys: list[tuple[str, str]] = []
    for line in evidence:
        match = _BLOCK_RE.match(line)
        assert match is not None
        key = (match.group(1), match.group(2))
        if not keys or key != keys[-1]:
            keys.append(key)
            blocks.append([])
        blocks[-1].append(line)
    reordered = [line for block in reversed(blocks) for line in block]
    return head + marker + "".join(reordered + tail)


def transform(row: dict, arm: str) -> dict:
    if not row.get("messages"):
        updated = dict(row)
        trace = dict(row.get("trace") or {})
        trace.update({
            "prompt_ablation": arm,
            "prompt_ablation_version": "graphmem-v5.55-prompt-ablation-v1",
            "answer_route": "deterministic_bypass",
            "primary_last": False,
            "closure_gate": False,
            "source_prompt_payload_hash": row.get("prompt_payload_hash"),
        })
        updated["trace"] = trace
        return updated

    messages = [dict(message) for message in row["messages"]]
    if len(messages) != 2 or messages[0]["role"] != "system":
        raise ValueError(f"unsupported message contract for {row['question_id']}")
    question = question_from(messages[-1]["content"])
    selected_route = route(question)
    use_lean = arm.startswith("lean")
    use_route = "_route" in arm
    use_primary_last = "primary_last" in arm
    use_closure = arm.endswith("_closure")
    # The broad interventions regress on comparison/aggregation/preference.
    # This arm is deliberately surgical: temporal questions receive the
    # evidence-order and endpoint discipline that helped them, while state
    # questions receive only latest-valid-state semantics.  All other routes
    # preserve the frozen baseline prompt byte for byte.
    if arm == "selective_v1":
        use_route = selected_route in {"temporal", "state_update"}
        use_primary_last = selected_route == "temporal"
        use_closure = False

    if use_lean:
        messages[0]["content"] = LEAN_SYSTEM

    if use_route:
        messages[-1]["content"] = (
            messages[-1]["content"].rstrip() + "\n\n" +
            ROUTE_CARDS[selected_route])
    if use_primary_last:
        messages[-1]["content"] = reverse_evidence_blocks(
            messages[-1]["content"], len(row["evidence_turn_ids"]))
    if use_closure:
        messages[-1]["content"] = (
            messages[-1]["content"].rstrip() + "\n\n" + CLOSURE_GATE)

    updated = dict(row)
    updated["messages"] = messages
    updated["prompt_hash"] = hashlib.sha256(
        ("graphmem-v5.55-prompt-ablation-v1:" + arm + ":" +
         messages[0]["content"]).encode()).hexdigest()
    updated["prompt_payload_hash"] = hashlib.sha256(
        canonical(messages).encode()).hexdigest()
    updated["packing_prompt_tokens"] = 0  # replay records API usage separately
    trace = dict(row.get("trace") or {})
    trace.update({
        "prompt_ablation": arm,
        "prompt_ablation_version": "graphmem-v5.55-prompt-ablation-v1",
        "answer_route": selected_route,
        "primary_last": use_primary_last,
        "closure_gate": use_closure,
        "source_prompt_payload_hash": row["prompt_payload_hash"],
    })
    updated["trace"] = trace
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--question-ids", type=Path)
    parser.add_argument(
        "--retrieval-all-hit", type=Path,
        help=("select LongMemEval rows with turn-level gold and packed all-hit; "
              "selection is diagnostic-only and never changes prompt content"))
    parser.add_argument(
        "--include-deterministic", action="store_true",
        help="retain deterministic bypass rows for full-set replay")
    args = parser.parse_args()
    if args.question_ids and args.retrieval_all_hit:
        raise ValueError("use only one diagnostic selector")

    rows = read_jsonl(args.prepared)
    selected_ids = None
    if args.question_ids:
        selected_ids = {
            line.strip() for line in args.question_ids.read_text(
                encoding="utf-8").splitlines() if line.strip()}
        rows = [row for row in rows if str(row["question_id"]) in selected_ids]
        missing = selected_ids - {str(row["question_id"]) for row in rows}
        if missing:
            raise ValueError(f"missing selected questions: {sorted(missing)[:5]}")
    elif args.retrieval_all_hit:
        selected_ids = {
            str(row["dev_question_id"])
            for row in read_jsonl(args.retrieval_all_hit)
            if row.get("benchmark") == "longmemeval"
            and bool(row.get("has_turn_gold"))
            and bool(row.get("turn_all_hit"))
        }
        rows = [row for row in rows if str(row["question_id"]) in selected_ids]
        missing = selected_ids - {str(row["question_id"]) for row in rows}
        if missing:
            raise ValueError(f"missing selected questions: {sorted(missing)[:5]}")
    # Deterministic bypass rows have no model prompt and therefore cannot be
    # part of an answer-prompt ablation.
    if not args.include_deterministic:
        rows = [row for row in rows if row.get("messages")]
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "graphmem-v5.55-prompt-ablation-materialization-v1",
        "source": str(args.prepared),
        "questions": len(rows),
        "selection_uses_gold_only_for_diagnostic_membership": bool(selected_ids),
        "prompt_transform_reads_gold": False,
        "arms": {},
    }
    for arm in ARMS:
        output = args.output_root / arm / "prepared_answers.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        transformed = [transform(row, arm) for row in rows]
        output.write_text("".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in transformed),
            encoding="utf-8")
        manifest["arms"][arm] = {
            "prepared": str(output),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "questions": len(transformed),
            "routes": {name: sum(
                row["trace"]["answer_route"] == name for row in transformed)
                for name in ROUTE_CARDS},
        }
    (args.output_root / "materialize_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
