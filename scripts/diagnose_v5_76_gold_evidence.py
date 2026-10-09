#!/usr/bin/env python3
"""Isolate retrieval/presentation errors with LoCoMo gold evidence.

The diagnostic is evaluator-only.  It finds questions for which none of an
existing adaptive candidate pool was judged correct, reconstructs the exact
annotated source turns from LoCoMo, and asks the same local answer model for
``n`` answers using only those turns.  A later judge pass can therefore split
failures into retrieval/layout noise (gold-only succeeds) and answer-model or
annotation limits (gold-only still fails).
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import threading
import time
from typing import Any, Iterable
from urllib import request


EVIDENCE_ID_RE = re.compile(r"D\d+:\d+")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do",
    "does", "for", "from", "had", "has", "have", "how", "in", "is",
    "it", "of", "on", "or", "that", "the", "their", "they", "to",
    "was", "were", "what", "when", "where", "which", "who", "why",
    "with", "would",
})


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").split("\n") if line.strip()]


def keyed(path: Path, key: str = "question_id") -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in read_jsonl(path)}


def prediction_verdicts(roots: Iterable[Path]) -> dict[tuple[str, str], bool]:
    """Merge repeated judge records conservatively using any-correct."""

    verdicts: dict[tuple[str, str], bool] = {}
    for root in roots:
        for path in sorted(root.glob("candidate_*/auto_eval.jsonl")):
            for row in read_jsonl(path):
                key = (str(row["question_id"]),
                       str(row["prediction_sha256"]))
                verdicts[key] = verdicts.get(key, False) or bool(row["correct"])
    return verdicts


def evidence_rows(case: dict[str, Any]) -> list[str]:
    wanted = {
        match for value in case.get("locomo_evidence", ())
        for match in EVIDENCE_ID_RE.findall(str(value))
    }
    rows: list[str] = []
    for index, session in enumerate(case.get("haystack_sessions", ())):
        dates = case.get("haystack_dates", ())
        date = str(dates[index]) if index < len(dates) else "unknown date"
        for turn in session:
            dia_id = str(turn.get("dia_id") or "")
            if dia_id not in wanted:
                continue
            text = " ".join(str(turn.get("content") or "").split())
            rows.append(
                f"[{dia_id}; {date}] {turn.get('speaker', 'unknown')}: {text}")
    return rows


def gold_closure_rows(case: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Render annotated turns plus their immediate dialogue neighbours."""

    wanted = {
        match for value in case.get("locomo_evidence", ())
        for match in EVIDENCE_ID_RE.findall(str(value))
    }
    selected: list[tuple[int, int, dict[str, Any], str]] = []
    dates = case.get("haystack_dates", ())
    for session_index, session in enumerate(case.get("haystack_sessions", ())):
        gold_positions = {
            index for index, turn in enumerate(session)
            if str(turn.get("dia_id") or "") in wanted
        }
        positions = {
            neighbor for index in gold_positions for neighbor in (index - 1, index, index + 1)
            if 0 <= neighbor < len(session)
        }
        date = str(dates[session_index]) if session_index < len(dates) else ""
        selected.extend(
            (session_index, index, session[index], date)
            for index in sorted(positions))
    rendered = [
        f"[{turn.get('dia_id')}; {date}] {turn.get('speaker', 'unknown')}: "
        f"{' '.join(str(turn.get('content') or '').split())}"
        for _session, _index, turn, date in selected
    ]
    return rendered, [str(turn.get("dia_id") or "")
                      for _session, _index, turn, _date in selected]


def _terms(text: str) -> tuple[str, ...]:
    return tuple(token.casefold() for token in WORD_RE.findall(text)
                 if token.casefold() not in STOPWORDS)


def focus_rows(case: dict[str, Any], limit: int) -> tuple[list[str], list[str]]:
    """Return a label-free BM25/phrase view with dialogue closure.

    This deliberately reads every immutable source turn but never reads
    ``locomo_evidence`` or the reference answer.  It is a diagnostic prototype
    for the source-facing physical lane used by GraphMem.
    """

    documents: list[dict[str, Any]] = []
    dates = case.get("haystack_dates", ())
    for session_index, session in enumerate(case.get("haystack_sessions", ())):
        date = str(dates[session_index]) if session_index < len(dates) else ""
        for turn_index, turn in enumerate(session):
            text = " ".join(str(turn.get("content") or "").split())
            surface = f"{turn.get('speaker', '')} {date} {text}"
            documents.append({
                "id": str(turn.get("dia_id") or ""),
                "session": session_index,
                "index": turn_index,
                "date": date,
                "speaker": str(turn.get("speaker") or "unknown"),
                "text": text,
                "terms": _terms(surface),
            })
    if not documents:
        return [], []
    query = _terms(str(case.get("question") or ""))
    query_set = frozenset(query)
    query_bigrams = frozenset(zip(query, query[1:]))
    df = Counter(term for row in documents for term in set(row["terms"]))
    lengths = [len(row["terms"]) for row in documents]
    average_length = sum(lengths) / max(1, len(lengths))
    count = len(documents)
    explicit_speakers = {
        str(case.get("speaker_a") or "").casefold(),
        str(case.get("speaker_b") or "").casefold(),
    } & query_set

    scored: list[tuple[float, int]] = []
    for position, row in enumerate(documents):
        frequencies = Counter(row["terms"])
        score = 0.0
        for term in query_set:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            inverse = __import__("math").log(
                1.0 + (count - df[term] + 0.5) / (df[term] + 0.5))
            denominator = frequency + 1.2 * (
                0.25 + 0.75 * len(row["terms"]) / max(1.0, average_length))
            score += inverse * frequency * 2.2 / denominator
        row_bigrams = frozenset(zip(row["terms"], row["terms"][1:]))
        score += 1.5 * len(query_bigrams & row_bigrams)
        if row["speaker"].casefold() in explicit_speakers:
            score += 1.0
        scored.append((score, position))
    scored.sort(key=lambda item: (-item[0], item[1]))

    # Start with independent lexical seeds, then admit the immediately
    # adjacent response/clarification turns.  Finally group selected turns by
    # source session and dialogue order so the model reads coherent packets.
    seed_count = max(8, min(limit, (2 * limit) // 3))
    selected = [position for score, position in scored[:seed_count] if score > 0]
    selected_set = set(selected)
    by_coordinate = {
        (row["session"], row["index"]): position
        for position, row in enumerate(documents)
    }
    for position in tuple(selected):
        row = documents[position]
        for delta in (-1, 1):
            neighbor = by_coordinate.get((row["session"], row["index"] + delta))
            if neighbor is not None and neighbor not in selected_set:
                selected.append(neighbor)
                selected_set.add(neighbor)
                if len(selected) >= limit:
                    break
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        for _score, position in scored:
            if position not in selected_set:
                selected.append(position)
                selected_set.add(position)
                if len(selected) >= limit:
                    break
    selected = sorted(selected[:limit], key=lambda position: (
        documents[position]["session"], documents[position]["index"]))
    rendered = [
        f"[{documents[position]['id']}; {documents[position]['date']}] "
        f"{documents[position]['speaker']}: {documents[position]['text']}"
        for position in selected
    ]
    return rendered, [documents[position]["id"] for position in selected]


def messages(case: dict[str, Any], evidence: list[str]) -> list[dict[str, str]]:
    system = (
        "Answer the exact question from the supplied original conversation "
        "turns. Keep entity, action, polarity, quantity, unit, completion "
        "status and dates exact. Resolve relative dates from the timestamp "
        "attached to that source turn. For lists and counts, combine all "
        "supplied turns without inventing items. For an inference question, "
        "make the smallest ordinary inference supported by the evidence. "
        "Return one concise answer only, without analysis."
    )
    body = [
        "Selected original source turns:",
        *(evidence or ["[No parseable annotated source turn] "]),
        f"Question date: {case.get('question_date') or 'not provided'}",
        f"Question: {case['question']}",
        "Answer:",
    ]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(body)},
    ]


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url.rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer local"}, method="POST")
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--judge-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument(
        "--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument(
        "--evidence-mode", choices=("gold", "gold_closure", "focus"),
        default="gold")
    parser.add_argument("--focus-turns", type=int, default=32)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7608)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cases = {str(row["question_id"]): row for row in json.loads(
        args.data.read_text(encoding="utf-8"))}
    pools = keyed(args.candidate_pool)
    verdicts = prediction_verdicts(args.judge_root)
    unresolved = []
    unmapped_candidates = 0
    for question_id, row in pools.items():
        flags = []
        for candidate in row.get("candidates", ()):
            key = (question_id, str(candidate.get("prediction_sha256") or ""))
            if key not in verdicts:
                unmapped_candidates += 1
            flags.append(verdicts.get(key, False))
        if not any(flags):
            unresolved.append(question_id)

    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "candidates.jsonl"
    existing = keyed(output) if args.resume else {}
    lock = threading.Lock()
    if not args.resume:
        output.write_text("", encoding="utf-8")

    def run(question_id: str) -> dict[str, Any]:
        case = cases[question_id]
        if args.evidence_mode == "gold":
            evidence = evidence_rows(case)
            evidence_ids = EVIDENCE_ID_RE.findall(
                " ".join(map(str, case.get("locomo_evidence", ()))))
        elif args.evidence_mode == "gold_closure":
            evidence, evidence_ids = gold_closure_rows(case)
        else:
            evidence, evidence_ids = focus_rows(case, args.focus_turns)
        payload = {
            "model": args.model,
            "messages": messages(case, evidence),
            "n": args.n,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "max_tokens": 2000,
            "seed": args.seed,
        }
        error: Exception | None = None
        for attempt in range(args.retries):
            try:
                response = post_json(
                    args.base_url, payload, timeout=args.timeout)
                choices = sorted(response["choices"], key=lambda row: row["index"])
                candidates = []
                for index, choice in enumerate(choices, start=1):
                    prediction = str(choice["message"].get("content") or "").strip()
                    candidates.append({
                        "rank": index,
                        "prediction": prediction,
                        "prediction_sha256": hashlib.sha256(
                            prediction.encode("utf-8")).hexdigest(),
                        "finish_reason": choice.get("finish_reason"),
                    })
                if len(candidates) != args.n:
                    raise RuntimeError(
                        f"expected {args.n} choices, got {len(candidates)}")
                return {
                    "question_id": question_id,
                    "conversation_id": case.get("locomo_sample_id"),
                    "category": int(case["locomo_category"]),
                    "memory_id": f"locomo:{case.get('locomo_sample_id')}",
                    "diagnostic": f"{args.evidence_mode}_evidence_only",
                    "evidence_ids": evidence_ids,
                    "evidence_rows": len(evidence),
                    "candidates": candidates,
                    "usage": response.get("usage", {}),
                }
            except Exception as exc:  # service is supervised and may restart
                error = exc
                time.sleep(min(30.0, 1.5 ** min(attempt, 8)))
        raise RuntimeError(f"{question_id}: {error}")

    pending = [question_id for question_id in unresolved
               if question_id not in existing]
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run, question_id): question_id
                   for question_id in pending}
        for future in as_completed(futures):
            question_id = futures[future]
            try:
                row = future.result()
            except Exception as error:
                failures.append(f"{question_id}: {error!r}")
                print(f"[error] {failures[-1]}", flush=True)
                continue
            with lock:
                with output.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"gold-only {question_id}: {args.n} candidates", flush=True)

    rows = read_jsonl(output)
    manifest = {
        "schema_version": "graphmem-v5.76-gold-evidence-diagnostic-v1",
        "interpretation": (
            "Evaluator-only isolation test; gold evidence is never visible "
            "to the production retriever or budget controller."),
        "source_candidate_pool": str(args.candidate_pool),
        "source_candidate_pool_sha256": hashlib.sha256(
            args.candidate_pool.read_bytes()).hexdigest(),
        "source_questions": len(pools),
        "unresolved_by_available_verdicts": len(unresolved),
        "candidate_verdicts_missing": unmapped_candidates,
        "completed": len(rows),
        "failures": failures,
        "model": args.model,
        "evidence_mode": args.evidence_mode,
        "focus_turns": args.focus_turns if args.evidence_mode == "focus" else None,
        "n": args.n,
        "seed": args.seed,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
