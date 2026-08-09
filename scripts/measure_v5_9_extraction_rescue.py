#!/usr/bin/env python3
"""Measure whether lossless atomic re-extraction rescues missing gold facts.

The extractor is deliberately query-agnostic: it sees one source turn and its
timestamp, but never the benchmark question or answer.  A separate sufficiency
judge then evaluates three evidence conditions for a stratified sample whose
gold evidence is not fully represented by current CanonicalFacts:

* current facts only;
* current facts plus newly extracted facts for missing turns;
* lossless annotated raw turns (oracle ceiling).

The experiment calls a local OpenAI-compatible model and records every call.
Because the configured extractor and judge may share a backbone, the output is
a confirmation signal rather than an independent final benchmark score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from graphmem.judging import OpenAICompatibleClient  # noqa: E402


EXTRACT_SYSTEM = """You are a lossless conversational-memory fact compiler.
Extract every independently queryable proposition in the supplied source turn.
Do not summarize multiple facts into one. Preserve exact people and ownership,
objects, actions, completion state, polarity/negation, modality, numbers, units,
dates, durations, relative-time wording and its supplied timestamp anchor.
Keep separate facts when a turn contains several items or comparisons. Never
invent information. Return JSON only: {"facts":[{"subject":str,
"predicate":str,"value":str,"time":str|null,"polarity":str,
"verbatim_support":str}]}.
"""

JUDGE_SYSTEM = """You audit whether a memory index preserves answer-supporting
information. Given a benchmark question, its reference answer and one evidence
condition, decide whether that evidence alone contains enough information to
derive the reference answer. Straightforward arithmetic, set operations and
date arithmetic are allowed. Do not credit facts absent from the evidence. Do
not penalize harmless extra evidence. Return JSON only:
{"sufficient":boolean,"reason":string}.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", type=Path,
        default=WORKSPACE / "artifacts/v5_9/full_benchmark_20260809/graph/report_graph.sqlite",
    )
    parser.add_argument(
        "--retrieval", type=Path,
        default=WORKSPACE / "artifacts/v5_9/full_benchmark_20260809/answers/merged/retrieval.jsonl",
    )
    parser.add_argument(
        "--lme", type=Path,
        default=WORKSPACE / "artifacts/data/longmemeval_s_cleaned.json",
    )
    parser.add_argument(
        "--locomo", type=Path,
        default=WORKSPACE / "artifacts/data/locomo10_graphmem.json",
    )
    parser.add_argument(
        "--lme-gold", type=Path,
        default=ROOT / "eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl",
    )
    parser.add_argument("--per-stratum", type=int, default=20)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--api-key-env", default="LOCAL_API_KEY")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/report/v5_9/extraction_rescue",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_locomo_refs(values: Iterable[str]) -> set[tuple[str, int]]:
    refs = set()
    for value in values or ():
        for part in str(value).split(";"):
            piece = part.strip()
            if ":" not in piece:
                continue
            day, turn = piece.split(":", 1)
            try:
                refs.add((f"session_{int(day[1:])}", int(turn) - 1))
            except ValueError:
                continue
    return refs


def load_questions(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    result = {}
    for row in json.loads(args.lme.read_text(encoding="utf-8")):
        qid = str(row["question_id"])
        result[qid] = {
            "question_id": qid, "memory_id": qid,
            "question": str(row["question"]), "answer": str(row["answer"]),
            "question_date": str(row.get("question_date") or ""), "refs": set(),
        }
    for row in read_jsonl(args.lme_gold):
        result[str(row["question_id"])]["refs"].add(
            (str(row["session_id"]), int(row["turn_index"])))
    for row in json.loads(args.locomo.read_text(encoding="utf-8")):
        if int(row["locomo_category"]) not in {1, 2, 3, 4}:
            continue
        qid = str(row["question_id"])
        result[qid] = {
            "question_id": qid,
            "memory_id": "locomo:" + str(row["locomo_sample_id"]),
            "question": str(row["question"]), "answer": str(row["answer"]),
            "question_date": str(row.get("question_date") or ""),
            "refs": parse_locomo_refs(row.get("locomo_evidence", ())),
        }
    return result


def parse_payload(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("response is not a JSON object")
    return value


def stable_order(value: str) -> str:
    return hashlib.sha256(("v5.9-extraction-rescue:" + value).encode()).hexdigest()


def main() -> None:
    args = parse_args()
    os.environ.setdefault(args.api_key_env, "local")
    questions = load_questions(args)
    retrieval = {str(row["dev_question_id"]): row for row in read_jsonl(args.retrieval)}
    db = sqlite3.connect(f"file:{args.db.resolve()}?mode=ro", uri=True)
    turns = {
        (str(memory), str(session), int(index)): {
            "turn_id": str(turn_id), "timestamp": str(timestamp or ""),
            "speaker": str(speaker), "raw_text": str(raw_text),
        }
        for turn_id, memory, session, index, timestamp, speaker, raw_text in db.execute(
            "SELECT turn_id,memory_id,session_id,turn_index,timestamp,speaker,raw_text "
            "FROM source_turns")
    }
    facts_by_turn: dict[str, list[str]] = defaultdict(list)
    for turn_id, summary in db.execute("""
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
        facts_by_turn[str(turn_id)].append(str(summary))
    db.close()

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for qid, question in questions.items():
        trace = retrieval.get(qid)
        if not trace or not question["refs"]:
            continue
        gold_turns = [turns.get((question["memory_id"], session, index))
                      for session, index in question["refs"]]
        gold_turns = [row for row in gold_turns if row]
        if gold_turns and any(not facts_by_turn.get(row["turn_id"]) for row in gold_turns):
            question = {**question, "stratum": str(trace["stratum"]),
                        "gold_turns": gold_turns}
            candidates[question["stratum"]].append(question)
    # Cat3 needs open-domain knowledge and makes an evidence-sufficiency oracle
    # ill-defined.  The remaining five strata directly test memory preservation.
    target_strata = (
        "lme_multi_session", "lme_temporal_reasoning",
        "locomo_cat1", "locomo_cat2", "locomo_cat4",
    )
    selected = []
    for stratum in target_strata:
        rows = sorted(candidates.get(stratum, ()),
                      key=lambda row: stable_order(row["question_id"]))
        selected.extend(rows[:args.per_stratum])
    selected.sort(key=lambda row: (row["stratum"], row["question_id"]))

    missing_turns = {}
    for question in selected:
        for turn in question["gold_turns"]:
            if not facts_by_turn.get(turn["turn_id"]):
                missing_turns[turn["turn_id"]] = turn
    client = OpenAICompatibleClient(
        model=args.model, base_url=args.base_url, api_key_env=args.api_key_env,
        request_profile="qwen", max_retries=3, timeout_sec=180,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    calls_path = args.output / "calls.jsonl"
    extracted_path = args.output / "extracted.jsonl"
    sufficiency_path = args.output / "sufficiency_partial.jsonl"
    failures_path = args.output / "extraction_failures.jsonl"
    if not args.resume:
        for path in (calls_path, extracted_path, sufficiency_path, failures_path):
            path.write_text("", encoding="utf-8")

    def extract(item: tuple[str, dict[str, Any]]):
        turn_id, turn = item
        payload = {
            "timestamp_anchor": turn["timestamp"], "speaker": turn["speaker"],
            "source_turn": turn["raw_text"],
        }
        records = []
        last_error: Exception | None = None
        for max_tokens in (1024, 2048):
            result = client.chat(
                question_id=turn_id, variant="v59_lossless_atomic_extractor",
                stage="semantic_extract", thinking_mode="none", json_mode=True,
                max_tokens=max_tokens, temperature=0.0, seed=0,
                messages=[{"role": "system", "content": EXTRACT_SYSTEM},
                          {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            )
            records.append(result.record)
            try:
                parsed = parse_payload(result.text)
                return turn_id, parsed.get("facts", []), records, None
            except (json.JSONDecodeError, ValueError) as error:
                last_error = error
        return turn_id, [], records, repr(last_error)

    extracted: dict[str, list[dict[str, Any]]] = {
        str(row["turn_id"]): list(row.get("facts") or ())
        for row in read_jsonl(extracted_path)
    } if extracted_path.exists() else {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(extract, item) for item in missing_turns.items()
                   if item[0] not in extracted]
        for future in as_completed(futures):
            turn_id, rows, records, error = future.result()
            extracted[turn_id] = [row for row in rows if isinstance(row, dict)]
            for record in records:
                append_jsonl(calls_path, asdict(record))
            if error:
                append_jsonl(failures_path, {"turn_id": turn_id, "error": error})
            append_jsonl(extracted_path, {
                "turn_id": turn_id, "facts": extracted[turn_id], "error": error,
            })
            print(f"extract {len(extracted)}/{len(missing_turns)}"
                  + (" FAILED" if error else ""), flush=True)

    def evidence(question: dict[str, Any], condition: str) -> str:
        parts = []
        for turn in question["gold_turns"]:
            turn_id = turn["turn_id"]
            if condition == "raw":
                parts.append(
                    f"[{turn['timestamp']}] {turn['speaker']}: {turn['raw_text']}")
                continue
            rows = list(facts_by_turn.get(turn_id, ()))
            if condition == "augmented" and not rows:
                rows.extend(
                    " | ".join(str(fact.get(key) or "") for key in
                               ("subject", "predicate", "value", "time", "polarity"))
                    for fact in extracted.get(turn_id, ())
                )
            parts.extend(rows)
        return "\n".join(parts) or "<NO INDEXED FACTS>"

    def judge(item: tuple[dict[str, Any], str]):
        question, condition = item
        prompt = {
            "question_date": question["question_date"],
            "question": question["question"], "reference_answer": question["answer"],
            "evidence_condition": condition, "evidence": evidence(question, condition),
        }
        records = []
        parsed: dict[str, Any] = {}
        error: str | None = None
        for max_tokens in (256, 512):
            result = client.chat(
                question_id=question["question_id"],
                variant=f"v59_fact_sufficiency_{condition}",
                stage="judge", thinking_mode="none", json_mode=True,
                max_tokens=max_tokens, temperature=0.0, seed=0,
                messages=[{"role": "system", "content": JUDGE_SYSTEM},
                          {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}],
            )
            records.append(result.record)
            try:
                parsed = parse_payload(result.text)
                error = None
                break
            except (json.JSONDecodeError, ValueError) as exc:
                error = repr(exc)
        return {
            "question_id": question["question_id"], "stratum": question["stratum"],
            "condition": condition, "sufficient": bool(parsed.get("sufficient")),
            "reason": str(parsed.get("reason") or ""),
            "judge_error": error,
            "gold_turns": len(question["gold_turns"]),
            "missing_fact_turns": sum(
                not facts_by_turn.get(turn["turn_id"]) for turn in question["gold_turns"]),
        }, records

    result_rows = read_jsonl(sufficiency_path) if sufficiency_path.exists() else []
    completed = {(str(row["question_id"]), str(row["condition"])) for row in result_rows}
    work = [(question, condition) for question in selected
            for condition in ("current", "augmented", "raw")]
    work = [item for item in work if (item[0]["question_id"], item[1]) not in completed]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(judge, item) for item in work]
        for future in as_completed(futures):
            row, records = future.result()
            result_rows.append(row)
            append_jsonl(sufficiency_path, row)
            for record in records:
                append_jsonl(calls_path, asdict(record))
            print(f"judge {len(result_rows)}/{len(selected) * 3}", flush=True)

    result_rows.sort(key=lambda row: (row["stratum"], row["question_id"], row["condition"]))
    by_question: dict[str, dict[str, Any]] = {}
    for row in result_rows:
        item = by_question.setdefault(row["question_id"], {
            "question_id": row["question_id"], "stratum": row["stratum"],
            "gold_turns": row["gold_turns"], "missing_fact_turns": row["missing_fact_turns"],
        })
        item[row["condition"]] = row["sufficient"]
    summary = {}
    for stratum in (*target_strata, "overall"):
        rows = list(by_question.values()) if stratum == "overall" else [
            row for row in by_question.values() if row["stratum"] == stratum]
        if not rows:
            continue
        raw_supported = [row for row in rows if row.get("raw")]
        rescuable = [row for row in raw_supported if not row.get("current")]
        rescued = [row for row in rescuable if row.get("augmented")]
        summary[stratum] = {
            "questions": len(rows),
            "current_sufficiency": sum(row.get("current", False) for row in rows) / len(rows),
            "augmented_sufficiency": sum(row.get("augmented", False) for row in rows) / len(rows),
            "raw_oracle_sufficiency": sum(row.get("raw", False) for row in rows) / len(rows),
            "raw_supported_current_failures": len(rescuable),
            "rescued_by_reextraction": len(rescued),
            "rescue_rate_given_raw_supported_current_failure": (
                len(rescued) / len(rescuable) if rescuable else 0.0),
        }
    calls = read_jsonl(calls_path)
    payload = {
        "schema_version": "graphmem-v5.9-extraction-rescue-v1",
        "method": {
            "query_agnostic_extractor": True,
            "same_backbone_extractor_and_judge": True,
            "sample_policy": f"up to {args.per_stratum} deterministic missing-fact questions per stratum",
            "excluded_stratum": "locomo_cat3 (open-domain evidence sufficiency is ill-defined)",
        },
        "model": args.model,
        "selected_questions": len(selected),
        "missing_turns_reextracted": len(missing_turns),
        "extracted_facts": sum(len(rows) for rows in extracted.values()),
        "call_usage": {
            "calls": len(calls),
            "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in calls),
            "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in calls),
            "latency_ms_mean": fmean(float(row.get("latency_sec") or 0) * 1000 for row in calls),
        },
        "summary": summary,
        "per_question": list(by_question.values()),
    }
    (args.output / "extraction_rescue.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.output / "sufficiency.jsonl").open("w", encoding="utf-8") as handle:
        for row in result_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), **summary.get("overall", {})}, indent=2))


if __name__ == "__main__":
    main()
