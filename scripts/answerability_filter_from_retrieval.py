#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmem_demo.clients import DeepSeekClient, rough_token_count  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process answers with a Qwen answerability/evidence-owner classifier."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--retrieval-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", default="answerability_filter")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--context-rough-tokens", type=int, default=5200)
    parser.add_argument(
        "--diagnostics-only",
        action="store_true",
        help="Only write evidence-owner diagnostics; do not call the classifier model.",
    )
    parser.add_argument("--only-question-types", nargs="*", default=[])
    parser.add_argument(
        "--decision-policy",
        choices=["all", "abstain_only"],
        default="all",
        help=(
            "all applies keep/revise/abstain decisions; abstain_only only lets "
            "high-confidence abstain decisions override the original answer."
        ),
    )
    return parser.parse_args()


def read_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def trim_rough_tokens(text: str, token_limit: int) -> tuple[str, bool]:
    if rough_token_count(text) <= token_limit:
        return text, False
    kept: list[str] = []
    total = 0
    for line in text.splitlines():
        line_tokens = rough_token_count(line)
        if kept and total + line_tokens > token_limit:
            break
        kept.append(line)
        total += line_tokens
    return "\n".join(kept), True


_NAME_STOPWORDS = {
    "Assistant",
    "Evidence",
    "Final",
    "Question",
    "Session",
    "User",
    "When",
    "What",
    "Where",
    "Which",
    "Who",
    "Whose",
    "Why",
    "How",
    "Did",
    "Does",
    "Do",
    "Is",
    "Are",
    "Was",
    "Were",
    "The",
}

_TERM_STOPWORDS = {
    "about",
    "after",
    "and",
    "are",
    "answer",
    "before",
    "black",
    "could",
    "did",
    "does",
    "from",
    "for",
    "have",
    "her",
    "his",
    "into",
    "its",
    "make",
    "made",
    "mention",
    "plans",
    "photo",
    "picture",
    "respect",
    "shared",
    "that",
    "their",
    "there",
    "the",
    "these",
    "this",
    "what",
    "when",
    "where",
    "which",
    "white",
    "with",
    "would",
    "why",
}


def _dedupe(items: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        cleaned = re.sub(r"\s+", " ", item.strip())
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
        if len(output) >= limit:
            break
    return output


def _question_people(question: str) -> list[str]:
    possessive = [
        re.sub(r"(?:'s|’s)$", "", match)
        for match in re.findall(r"\b[A-Z][A-Za-z'-]{2,}(?:'s|’s)\b", question)
    ]
    remaining = [
        re.sub(r"(?:'s|’s)$", "", match)
        for match in re.findall(r"\b[A-Z][A-Za-z'-]{2,}(?:'s|’s)?\b", question)
    ]
    candidates = possessive + remaining
    return _dedupe([name for name in candidates if name not in _NAME_STOPWORDS], 6)


def _question_terms(question: str, people: list[str]) -> list[str]:
    people_tokens = {token.casefold() for name in people for token in re.findall(r"[A-Za-z]+", name)}
    terms = []
    for token in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", question):
        token = re.sub(r"(?:'s|’s)$", "", token)
        lowered = token.casefold()
        if lowered in _TERM_STOPWORDS or lowered in people_tokens:
            continue
        terms.append(lowered)
    return _dedupe(terms, 12)


def _speaker_for_line(line: str) -> str | None:
    match = re.search(r"^(?:User|Assistant)\s*\(([^)]+)\)\s*(?:->[^:]+)?:", line)
    if match:
        return match.group(1).strip()
    match = re.search(r"\[([A-Z][A-Za-z'-]{2,}) shared an image:", line)
    if match:
        return match.group(1).strip()
    match = re.search(r"^\s*-\s*Session [^:]+: Memory:\s*([A-Z][A-Za-z'-]{2,})\b", line)
    if match:
        return match.group(1).strip()
    return None


def _content_lines(evidence: str) -> list[str]:
    lines: list[str] = []
    for raw_line in evidence.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[Session ") or line.startswith("User") or line.startswith("Assistant"):
            lines.append(line)
        elif line.startswith("- Session ") and "Memory:" in line:
            lines.append(line)
    return lines


def _line_score(line: str, terms: list[str]) -> int:
    lowered = line.casefold()
    return sum(
        1
        for term in terms
        if re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", lowered)
    )


def _target_line_supports_owner(line: str, target: str) -> bool:
    lowered = line.casefold()
    if line.startswith("- Session ") and re.search(rf"\b{re.escape(target)}\b", line):
        return True
    other_names = [
        name
        for name in re.findall(r"\b[A-Z][A-Za-z'-]{2,}\b", line)
        if name not in _NAME_STOPWORDS and name.casefold() != target.casefold()
    ]
    if other_names and re.search(r"\b(?:you|your|yours|congrats|congratulations|thanks)\b", lowered):
        return False
    if re.search(r"\b(?:i|i'm|i've|i’d|i'd|my|mine|me|we|we're|we've|our|ours)\b", lowered):
        return True
    return not other_names


def evidence_owner_diagnostics(question: str, evidence: str, *, max_chars: int = 1400) -> str:
    people = _question_people(question)
    target = people[0] if people else ""
    terms = _question_terms(question, people)
    target_hits: list[str] = []
    target_response_hits: list[str] = []
    other_hits: list[str] = []
    photo_hits: list[str] = []
    unknown_hits: list[str] = []
    for line in _content_lines(evidence):
        lowered = line.casefold()
        speaker = _speaker_for_line(line)
        score = _line_score(line, terms)
        is_photo = bool(re.search(r"\b(?:image|photo|picture|pic|photography)\b", lowered))
        if score == 0 and not is_photo:
            continue
        compact = line[:260]
        if is_photo:
            photo_hits.append(compact)
        if target and speaker:
            if speaker.casefold() == target.casefold():
                if score:
                    if _target_line_supports_owner(line, target):
                        target_hits.append(compact)
                    else:
                        target_response_hits.append(compact)
            elif score:
                other_hits.append(f"{speaker}: {compact}")
        elif score:
            unknown_hits.append(compact)

    parts = [
        "Evidence-owner diagnostics (heuristic, non-authoritative):",
        f"- Target person from question: {target or 'unknown'}",
        f"- Question people: {', '.join(people) if people else 'none'}",
        f"- Question content terms: {', '.join(terms) if terms else 'none'}",
    ]
    if target_hits:
        parts.append("- Target-owned candidate evidence:")
        parts.extend(f"  * {line}" for line in _dedupe(target_hits, 4))
    else:
        parts.append("- Target-owned candidate evidence: none found by heuristic")
    if target_response_hits:
        parts.append("- Target-speaker response about another person/object:")
        parts.extend(f"  * {line}" for line in _dedupe(target_response_hits, 3))
    if other_hits:
        parts.append("- Other-speaker candidate evidence:")
        parts.extend(f"  * {line}" for line in _dedupe(other_hits, 4))
    if photo_hits:
        parts.append("- Photo/image candidate evidence:")
        parts.extend(f"  * {line}" for line in _dedupe(photo_hits, 4))
    if unknown_hits:
        parts.append("- Unattributed candidate evidence:")
        parts.extend(f"  * {line}" for line in _dedupe(unknown_hits, 3))
    text = "\n".join(parts)
    if len(text) > max_chars:
        return text[: max_chars - 16].rstrip() + "\n...[truncated]"
    return text


def classifier_messages(
    *,
    question: str,
    question_type: str,
    question_date: str | None,
    prediction: str,
    evidence: str,
    diagnostics: str = "",
) -> list[dict[str, str]]:
    system = (
        "You are an answerability and evidence-owner checker for memory QA. "
        "Use only the supplied retrieved memory evidence and the current prediction. "
        "Return JSON only with keys: decision, final_answer, reason. "
        "decision must be one of keep, revise, abstain. "
        "Use keep when the current prediction is directly supported by evidence or is a valid "
        "calculation/inference from it. Use revise when the evidence clearly supports a concise "
        "better answer. Use abstain only when the required fact is missing, or when the current "
        "prediction transfers a fact from a different speaker/person than the one asked about. "
        "In multi-speaker conversations, labels like 'User (Caroline) -> Melanie' and "
        "'Assistant (Melanie) -> Caroline' identify who said or owns the fact; do not transfer "
        "possessive items, plans, feelings, family members, dates, events, or reasons between "
        "speakers. Image captions such as '[Caroline shared an image: ...]' only prove who "
        "shared the image and what is visually described; they do not prove who made, owns, "
        "or is related to the object unless the surrounding text says so. Use the optional "
        "evidence-owner diagnostics only as a checklist; if it conflicts with the raw evidence, "
        "trust the raw evidence. Do not abstain for preference or inference questions if the "
        "evidence gives reasonable support. For temporal questions, revise if the evidence "
        "gives the event and relative date needed for the calculation. For category_5 or "
        "other answerability-style questions, be stricter: keep or revise only when the "
        "evidence directly supports the asked person's own item, plan, feeling, event, "
        "family member, reason, or action. If the evidence instead belongs to another "
        "speaker/person and the asked person only asks about it, congratulates them, "
        "reacts to it, or makes a generic supportive comment, choose abstain. A summary "
        "or prediction that merely mentions the asked person's name is not enough; identify "
        "who owns the underlying first-person fact. If the question contains a false "
        "premise, such as assigning another person's event or object to the asked person, "
        "choose abstain rather than answering the corrected premise."
    )
    diagnostics_block = f"\n\n{diagnostics}" if diagnostics else ""
    user = (
        f"Question date: {question_date or 'unknown'}\n"
        f"Question type: {question_type}\n"
        f"Question: {question}\n\n"
        f"Current prediction:\n{prediction}\n\n"
        f"Retrieved memory evidence:\n{evidence}"
        f"{diagnostics_block}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_payload(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            payload = json.loads(text[start : end + 1])
        else:
            raise
    decision = str(payload.get("decision") or "keep").strip().lower()
    if decision not in {"keep", "revise", "abstain"}:
        decision = "keep"
    return {
        "decision": decision,
        "final_answer": str(payload.get("final_answer") or "").strip(),
        "reason": str(payload.get("reason") or "").strip(),
    }


def render_prediction(original: str, payload: dict[str, Any], policy: str = "all") -> str:
    decision = payload["decision"]
    if policy == "abstain_only" and decision != "abstain":
        return original
    if decision == "keep":
        return original
    final_answer = payload.get("final_answer") or ""
    reason = payload.get("reason") or ""
    if decision == "abstain" and not final_answer:
        final_answer = "The supplied evidence is insufficient to answer this for the asked person."
    return f"Evidence facts: {reason}\n\nFinal answer: {final_answer}"


def filter_one(
    *,
    case: dict[str, Any],
    answer: dict[str, Any],
    retrieval: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    context, truncated = trim_rough_tokens(
        str(retrieval.get("context_text") or ""), args.context_rough_tokens
    )
    diagnostics = evidence_owner_diagnostics(str(case.get("question") or ""), context)
    llm = DeepSeekClient(model=args.model, base_url=args.base_url)
    started = time.perf_counter()
    result = llm.chat(
        question_id=str(case["question_id"]),
        variant=args.variant,
        stage="answerability_filter",
        thinking_mode="none",
        messages=classifier_messages(
            question=str(case.get("question") or ""),
            question_type=str(case.get("question_type") or ""),
            question_date=case.get("question_date"),
            prediction=str(answer.get("prediction") or ""),
            evidence=context,
            diagnostics=diagnostics,
        ),
        max_tokens=args.max_tokens,
        json_mode=True,
    )
    try:
        payload = parse_payload(result.text)
    except Exception as error:
        payload = {"decision": "keep", "final_answer": "", "reason": f"parse_error: {error}"}
    row = dict(answer)
    row["variant"] = args.variant
    row["prediction"] = render_prediction(
        str(answer.get("prediction") or ""), payload, args.decision_policy
    )
    row["answerability_effective_decision"] = (
        payload["decision"]
        if args.decision_policy == "all" or payload["decision"] == "abstain"
        else "keep_original"
    )
    row["answerability_decision"] = payload["decision"]
    row["answerability_reason"] = payload["reason"]
    row["answerability_final_answer"] = payload["final_answer"]
    row["answerability_diagnostics"] = diagnostics
    row["answerability_context_truncated"] = truncated
    row["answerability_latency_sec"] = time.perf_counter() - started
    record = asdict(result.record)
    record["context_truncated"] = truncated
    return row, record


def main() -> None:
    args = parse_args()
    if not args.diagnostics_only and not args.model:
        raise SystemExit("--model is required unless --diagnostics-only is set")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = {str(row["question_id"]): row for row in read_json_or_jsonl(args.data)}
    answers = read_json_or_jsonl(args.answers)
    retrieval = {str(row["question_id"]): row for row in read_json_or_jsonl(args.retrieval_results)}

    output_rows: list[dict[str, Any]] = []
    call_rows: list[dict[str, Any]] = []
    type_filter = set(args.only_question_types)
    if args.diagnostics_only:
        for answer in answers:
            qid = str(answer["question_id"])
            if qid not in cases or qid not in retrieval:
                continue
            row = dict(answer)
            row["variant"] = args.variant
            if type_filter and str(cases[qid].get("question_type")) not in type_filter:
                row["answerability_decision"] = "skipped"
                row["answerability_effective_decision"] = "skipped"
                row["answerability_reason"] = "question_type_not_selected"
                row["answerability_final_answer"] = ""
                row["answerability_diagnostics"] = ""
                row["answerability_context_truncated"] = False
                row["answerability_latency_sec"] = 0.0
                output_rows.append(row)
                continue
            context, truncated = trim_rough_tokens(
                str(retrieval[qid].get("context_text") or ""), args.context_rough_tokens
            )
            row["answerability_decision"] = "diagnostics_only"
            row["answerability_effective_decision"] = "keep_original"
            row["answerability_reason"] = ""
            row["answerability_final_answer"] = ""
            row["answerability_diagnostics"] = evidence_owner_diagnostics(
                str(cases[qid].get("question") or ""), context
            )
            row["answerability_context_truncated"] = truncated
            row["answerability_latency_sec"] = 0.0
            output_rows.append(row)
        order = {str(row["question_id"]): index for index, row in enumerate(answers)}
        output_rows.sort(key=lambda row: order[str(row["question_id"])])
        write_jsonl(args.output_dir / "answers.jsonl", output_rows)
        write_jsonl(
            args.output_dir / f"{args.variant}_hypothesis.jsonl",
            [
                {"question_id": row["question_id"], "hypothesis": row.get("prediction", "")}
                for row in output_rows
            ],
        )
        write_jsonl(args.output_dir / "answerability_calls.jsonl", call_rows)
        stats = {
            "question_count": len(output_rows),
            "decision_counts": {
                decision: sum(row["answerability_decision"] == decision for row in output_rows)
                for decision in ("diagnostics_only", "skipped")
            },
            "effective_decision_counts": {
                decision: sum(row.get("answerability_effective_decision") == decision for row in output_rows)
                for decision in ("keep_original", "skipped")
            },
            "decision_policy": args.decision_policy,
            "diagnostics_only": True,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "max_tokens_per_question": 0,
        }
        (args.output_dir / "answerability_stats.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for answer in answers:
            qid = str(answer["question_id"])
            if qid not in cases or qid not in retrieval:
                continue
            if type_filter and str(cases[qid].get("question_type")) not in type_filter:
                row = dict(answer)
                row["variant"] = args.variant
                row["answerability_decision"] = "skipped"
                row["answerability_effective_decision"] = "skipped"
                row["answerability_reason"] = "question_type_not_selected"
                row["answerability_final_answer"] = ""
                row["answerability_context_truncated"] = False
                row["answerability_latency_sec"] = 0.0
                output_rows.append(row)
                continue
            futures[
                executor.submit(
                    filter_one,
                    case=cases[qid],
                    answer=answer,
                    retrieval=retrieval[qid],
                    args=args,
                )
            ] = qid
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            row, record = future.result()
            output_rows.append(row)
            call_rows.append(record)
            if done % 25 == 0 or done == len(futures):
                decisions = {}
                for item in output_rows:
                    decisions[item["answerability_decision"]] = (
                        decisions.get(item["answerability_decision"], 0) + 1
                    )
                print(f"progress {done}/{len(futures)} decisions={decisions}", flush=True)

    order = {str(row["question_id"]): index for index, row in enumerate(answers)}
    output_rows.sort(key=lambda row: order[str(row["question_id"])])
    call_rows.sort(key=lambda row: order[str(row["question_id"])])

    write_jsonl(args.output_dir / "answers.jsonl", output_rows)
    write_jsonl(
        args.output_dir / f"{args.variant}_hypothesis.jsonl",
        [
            {"question_id": row["question_id"], "hypothesis": row.get("prediction", "")}
            for row in output_rows
        ],
    )
    write_jsonl(args.output_dir / "answerability_calls.jsonl", call_rows)
    stats = {
        "question_count": len(output_rows),
        "decision_counts": {
            decision: sum(row["answerability_decision"] == decision for row in output_rows)
            for decision in ("keep", "revise", "abstain", "skipped")
        },
        "effective_decision_counts": {
            decision: sum(row.get("answerability_effective_decision") == decision for row in output_rows)
            for decision in ("keep", "revise", "abstain", "keep_original", "skipped")
        },
        "decision_policy": args.decision_policy,
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in call_rows),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in call_rows),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in call_rows),
        "max_tokens_per_question": max(
            (
                int(row.get("prompt_tokens") or 0) + int(row.get("completion_tokens") or 0)
                for row in call_rows
            ),
            default=0,
        ),
    }
    (args.output_dir / "answerability_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
