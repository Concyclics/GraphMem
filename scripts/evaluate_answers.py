#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge GraphMem answers with an LLM.")
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    answers = [json.loads(line) for line in args.answers.read_text().splitlines() if line.strip()]
    source_rows = json.loads(args.data.read_text())
    question_types = {str(row["question_id"]): row.get("question_type") for row in source_rows}
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        timeout=120,
    )

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(judge_one, client, args.model, row, question_types): row
            for row in answers
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                result["question_id"],
                result["strict_correct"],
                result["relaxed_correct"],
                result["error_type"],
                flush=True,
            )

    order = {row["question_id"]: index for index, row in enumerate(answers)}
    results.sort(key=lambda row: order[row["question_id"]])
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in results) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(render_markdown(results), encoding="utf-8")
    print(f"wrote {args.output_jsonl}")
    print(f"wrote {args.output_md}")


def judge_one(
    client: OpenAI,
    model: str,
    row: dict[str, Any],
    question_types: dict[str, str | None],
) -> dict[str, Any]:
    prompt = f"""Judge whether a memory QA prediction matches the gold answer.

Strict correctness:
- Numeric/date/time/count/cost answers must contain the same final value as gold.
- If gold says information is insufficient, strict is true only if prediction clearly says insufficient / cannot determine and does not invent the missing answer.
- Extra explanation is allowed only if it does not change the final answer.

Relaxed correctness:
- Semantic equivalence is allowed.
- Minor wording differences or harmless extra evidence are allowed.
- Wrong final value, contradiction, or saying insufficient when gold is known is false.

Return JSON only with keys:
{{"strict_correct": boolean, "relaxed_correct": boolean, "error_type": string|null, "notes": string}}

Question: {row["question"]}
Gold answer: {row["gold_answer"]}
Prediction: {row["prediction"]}
"""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a careful benchmark judge. Return valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content or "{}"
            payload = json.loads(text)
            return _result_row(row, question_types, payload, response.usage)
        except Exception as error:
            last_error = error
            time.sleep(2**attempt)
    return _result_row(
        row,
        question_types,
        {
            "strict_correct": False,
            "relaxed_correct": False,
            "error_type": "judge_error",
            "notes": str(last_error),
        },
        None,
    )


def _result_row(
    row: dict[str, Any],
    question_types: dict[str, str | None],
    payload: dict[str, Any],
    usage: Any,
) -> dict[str, Any]:
    return {
        "question_id": row["question_id"],
        "question_type": question_types.get(row["question_id"]),
        "question": row["question"],
        "gold_answer": row["gold_answer"],
        "prediction": row["prediction"],
        "strict_correct": bool(payload.get("strict_correct")),
        "relaxed_correct": bool(payload.get("relaxed_correct")),
        "error_type": payload.get("error_type"),
        "notes": str(payload.get("notes") or ""),
        "judge_prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "judge_completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "judge_total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def render_markdown(results: list[dict[str, Any]]) -> str:
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        by_type.setdefault(str(row["question_type"]), []).append(row)
    lines = [
        "# DeepSeek Flash Auto Evaluation",
        "",
        f"- Rows: {len(results)}",
        (
            f"- Strict accuracy: {_accuracy(results, 'strict_correct'):.3f} "
            f"({sum(row['strict_correct'] for row in results)}/{len(results)})"
        ),
        (
            f"- Relaxed accuracy: {_accuracy(results, 'relaxed_correct'):.3f} "
            f"({sum(row['relaxed_correct'] for row in results)}/{len(results)})"
        ),
        f"- Judge tokens: {sum(row['judge_total_tokens'] for row in results)}",
        "",
        "| question_type | rows | strict | relaxed |",
        "| --- | ---: | ---: | ---: |",
    ]
    for question_type in sorted(by_type):
        rows = by_type[question_type]
        lines.append(
            f"| {question_type} | {len(rows)} | "
            f"{_accuracy(rows, 'strict_correct'):.3f} | "
            f"{_accuracy(rows, 'relaxed_correct'):.3f} |"
        )
    lines.extend(
        [
            "",
            "| question_id | type | strict | relaxed | error_type | notes |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in results:
        notes = str(row["notes"]).replace("|", "\\|").replace("\n", " ")[:240]
        lines.append(
            f"| {row['question_id']} | {row['question_type']} | {row['strict_correct']} | "
            f"{row['relaxed_correct']} | {row['error_type']} | {notes} |"
        )
    return "\n".join(lines) + "\n"


def _accuracy(rows: list[dict[str, Any]], field: str) -> float:
    return sum(bool(row[field]) for row in rows) / len(rows) if rows else 0.0


if __name__ == "__main__":
    main()
