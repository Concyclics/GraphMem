#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from nltk.stem import PorterStemmer


STEMMER = PorterStemmer()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _normalize(text: Any) -> str:
    value = str(text or "").replace(",", "")
    value = value.lower()
    value = "".join(character for character in value if character not in string.punctuation)
    value = re.sub(r"\b(a|an|the|and)\b", " ", value)
    return " ".join(value.split())


def _tokens(text: Any) -> list[str]:
    return [STEMMER.stem(token) for token in _normalize(text).split()]


def _f1(prediction: Any, answer: Any) -> float:
    prediction_tokens = _tokens(prediction)
    answer_tokens = _tokens(answer)
    common = Counter(prediction_tokens) & Counter(answer_tokens)
    overlap = sum(common.values())
    if not prediction_tokens or not answer_tokens:
        return float(prediction_tokens == answer_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def official_score(prediction: Any, answer: Any, category: int) -> float:
    prediction_text = str(prediction or "")
    answer_text = str(answer or "")
    if category == 5:
        lowered = prediction_text.lower()
        return float(
            "no information available" in lowered
            or "not mentioned" in lowered
        )
    if category == 1:
        predicted_items = [item.strip() for item in prediction_text.split(",") if item.strip()]
        gold_items = [item.strip() for item in answer_text.split(",") if item.strip()]
        if not gold_items:
            return float(not predicted_items)
        return sum(
            max((_f1(predicted, gold) for predicted in predicted_items), default=0.0)
            for gold in gold_items
        ) / len(gold_items)
    if category == 3:
        answer_text = answer_text.split(";", 1)[0]
    return _f1(prediction_text, answer_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GraphMem with the official LoCoMo F1.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cases = {row["question_id"]: row for row in json.loads(args.data.read_text(encoding="utf-8"))}
    answers = {row["question_id"]: row for row in _read_jsonl(args.answers)}
    missing = sorted(set(cases) - set(answers))
    extra = sorted(set(answers) - set(cases))
    rows = []
    category_scores: dict[int, list[float]] = defaultdict(list)
    conversation_scores: dict[str, list[float]] = defaultdict(list)
    for question_id, case in cases.items():
        if question_id not in answers:
            continue
        category = int(case["locomo_category"])
        prediction = answers[question_id].get("prediction", "")
        score = official_score(prediction, case.get("answer"), category)
        category_scores[category].append(score)
        conversation_scores[str(case["locomo_sample_id"])].append(score)
        rows.append(
            {
                "question_id": question_id,
                "conversation_id": case["locomo_sample_id"],
                "category": category,
                "question": case["question"],
                "gold_answer": case.get("answer"),
                "prediction": prediction,
                "official_f1": score,
            }
        )

    summary = {
        "metric": "official_locomo_token_f1",
        "question_count": len(cases),
        "evaluated_count": len(rows),
        "missing_answer_count": len(missing),
        "missing_question_ids": missing,
        "extra_answer_count": len(extra),
        "extra_question_ids": extra,
        "overall_f1": sum(row["official_f1"] for row in rows) / len(rows) if rows else 0.0,
        "by_category": {
            str(category): {
                "count": len(scores),
                "f1": sum(scores) / len(scores) if scores else 0.0,
            }
            for category, scores in sorted(category_scores.items())
        },
        "by_conversation": {
            conversation_id: {
                "count": len(scores),
                "f1": sum(scores) / len(scores) if scores else 0.0,
            }
            for conversation_id, scores in conversation_scores.items()
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "official_eval.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "official_eval.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = [
        "# LoCoMo official evaluation",
        "",
        f"- Evaluated: {len(rows)}/{len(cases)}",
        f"- Overall token F1: {summary['overall_f1']:.4f}",
        f"- Missing answers: {len(missing)}",
        "",
        "| Category | Questions | F1 |",
        "|---:|---:|---:|",
    ]
    for category, item in summary["by_category"].items():
        markdown.append(f"| {category} | {item['count']} | {item['f1']:.4f} |")
    (args.output_dir / "official_eval.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "metric": summary["metric"],
                "question_count": summary["question_count"],
                "evaluated_count": summary["evaluated_count"],
                "missing_answer_count": summary["missing_answer_count"],
                "extra_answer_count": summary["extra_answer_count"],
                "overall_f1": summary["overall_f1"],
                "by_category": summary["by_category"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
