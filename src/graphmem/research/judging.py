"""Frozen Qwen grading, separated from all retrieval and answer routing."""

import hashlib
import json
import re
from pathlib import Path
from .rubrics import locomo, longmemeval

RUBRIC_SHA = {
    "locomo": "8ebac1ef60e9ab5caf99079fdaac038b85472e81491ed35e2d2655f3927c76c2",
    "longmemeval": "ba8cf60d26f1390ecbef0f07b3e950556fe3bc5a37ba4b5343f28217f18c144f",
}
FORMAT = "\nResponse format: give at most three short sentences inside <judge_thinking>, close </judge_thinking>, then put exactly yes or no on the last line. Do not repeat your explanation. The grading criteria above are unchanged."


def payload(benchmark, meta, answer, attempt=0):
    path = Path(__file__).parent / "rubrics" / (benchmark + ".py")
    if hashlib.sha256(path.read_bytes()).hexdigest() != RUBRIC_SHA[benchmark]:
        raise ValueError("Rubric checksum mismatch")
    if benchmark == "locomo":
        category = int(meta["question_type"])
        prompt = locomo.get_judge_prompt(
            category,
            meta["question"],
            locomo.preprocess_answer(category, str(meta["reference"])),
            answer.strip(),
        )
        if attempt:
            prompt += '\nReturn exactly one JSON object with keys "reasoning" and "label". The label must be exactly "CORRECT" or "WRONG". This is only a response-format reminder; use the same grading rules above.'
        return dict(
            model="graphmem-eval-qwen30b",
            messages=[
                {"role": "system", "content": locomo.JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            seed=0,
            max_tokens=256,
            response_format={"type": "json_object"},
        )
    prompt = (
        longmemeval.get_judge_prompt(
            meta["question_type"],
            meta["qid"],
            meta["question"],
            meta["reference"],
            answer.strip(),
            meta["question_date"],
        )
        + FORMAT
    )
    if attempt:
        prompt += "\nResponse-format reminder only: keep the explanation inside <judge_thinking> to at most three short sentences, close </judge_thinking>, and put exactly yes or no on the last line. Apply the same grading criteria above."
    return dict(
        model="graphmem-eval-qwen30b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        seed=0,
        max_tokens=2048,
    )


def judge(benchmark, meta, answer, meter):
    group = hashlib.sha256(
        json.dumps(
            (benchmark, meta.get("public_qid", meta["qid"]), answer.strip()),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    keys = []
    for attempt in range(3):
        body = payload(benchmark, meta, answer, attempt)
        response, key = meter.call(group, "judge_" + str(attempt), body, api="judge")
        keys.append(key)
        choice = response["choices"][0]
        text = choice["message"]["content"].strip()
        try:
            if choice["finish_reason"] != "stop":
                raise ValueError("Truncated judge")
            if benchmark == "locomo":
                label = json.loads(text)["label"]
                if label not in {"CORRECT", "WRONG"}:
                    raise ValueError("Invalid label")
                correct = label == "CORRECT"
            else:
                final = (
                    text.rsplit("</judge_thinking>", 1)[-1].strip()
                    if "</judge_thinking>" in text
                    else text.splitlines()[-1]
                )
                if not re.fullmatch(r"(yes|no)[.!]?", final.lower()):
                    raise ValueError("Invalid label")
                correct = final.lower().startswith("yes")
            return dict(
                qid=meta["qid"],
                correct=correct,
                judge_group_id=group,
                answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
                question_type=meta["question_type"],
                call_keys=keys,
                usage=meter.usage(keys),
                rubric_sha256=RUBRIC_SHA[benchmark],
            )
        except (ValueError, KeyError, IndexError, TypeError):
            if attempt == 2:
                raise ValueError(
                    "Judge format invalid after three recorded attempts"
                ) from None
