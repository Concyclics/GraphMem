#!/usr/bin/env python3
"""Does the retrieval proxy predict judged correctness?

Every V5.4/V5.5/V5.6 conclusion -- "h6 beats h9", "the binding discriminant is
worse", "the fused-score fill carries gold_packed" -- is stated in terms of
strict turn all-hit, and no V5 run had an answer stage, so that proxy has never
been checked against a judged answer.  A 7pp spread between harness rungs means
nothing if the proxy correlates weakly with what we actually optimize.

Joins ``retrieval.jsonl`` to a judge's ``auto_eval.jsonl`` and reports, for each
candidate proxy, the point-biserial correlation with judged correctness plus the
accuracy split between proxy-true and proxy-false questions.  The split is the
readable number: "questions whose gold turns were all packed are judged correct
X% of the time, against Y% when they were not".

Turn-level proxies are only defined where turn gold exists, so rows without it
are excluded per-proxy rather than counted as zero.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

#: Proxies needing turn gold; ``turn_all_hit`` is vacuously true when a question
#: has no annotated gold turns, so those rows must be dropped rather than scored.
TURN_PROXIES = (
    "turn_all_hit", "turn_any_hit", "turn_recall", "turn_precision",
    "candidate_turn_all_hit", "candidate_turn_recall", "graph_reachable_turn_recall",
)
SESSION_PROXIES = ("session_all_hit", "session_any_hit", "session_recall")
OTHER_PROXIES = ("certificate_complete", "closed_form", "budget_relaxed",
                 "budget_exhausted", "path_provenance_complete")
PROXIES = TURN_PROXIES + SESSION_PROXIES + OTHER_PROXIES


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def point_biserial(pairs: list[tuple[float, bool]]) -> float | None:
    """Pearson correlation between a continuous proxy and a binary outcome."""
    if len(pairs) < 3:
        return None
    xs = [row[0] for row in pairs]
    ys = [1.0 if row[1] else 0.0 for row in pairs]
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denominator = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denominator


def bootstrap_ci(pairs: list[tuple[float, bool]], seed: int = 42,
                 resamples: int = 2000) -> tuple[float, float] | None:
    """Percentile CI for the correlation, so a near-zero value can be called zero."""
    if len(pairs) < 10:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(resamples):
        sample = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        value = point_biserial(sample)
        if value is not None:
            values.append(value)
    if not values:
        return None
    values.sort()
    return values[int(0.025 * len(values))], values[min(len(values) - 1, int(0.975 * len(values)))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--judge", type=Path, required=True,
                        help="auto_eval.jsonl from a judge run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    judged = {str(row["question_id"]): bool(row["correct"]) for row in read_jsonl(args.judge)}
    rows = [row for row in read_jsonl(args.retrieval)
            if str(row["dev_question_id"]) in judged]
    if not rows:
        raise SystemExit("no questions joined; check that the two files share question ids")
    for row in rows:
        row["judged_correct"] = judged[str(row["dev_question_id"])]

    accuracy = sum(row["judged_correct"] for row in rows) / len(rows)
    report: dict = {
        "questions_judged": len(rows),
        "judged_accuracy": accuracy,
        "by_stratum": {}, "proxies": {},
    }
    for stratum in sorted({row["stratum"] for row in rows}):
        subset = [row for row in rows if row["stratum"] == stratum]
        report["by_stratum"][stratum] = {
            "questions": len(subset),
            "accuracy": sum(row["judged_correct"] for row in subset) / len(subset),
        }

    for proxy in PROXIES:
        present = [row for row in rows if isinstance(row.get(proxy), (bool, int, float))]
        if proxy in TURN_PROXIES:
            # ``gold <= predicted`` is trivially true for a question with no
            # annotated gold turns, which would inflate every turn-level proxy.
            present = [row for row in present if int(row.get("gold_turns", 0)) > 0]
        if not present:
            continue
        pairs = [(float(row[proxy]), row["judged_correct"]) for row in present]
        true_rows = [row for row in present if float(row[proxy]) >= 0.5]
        false_rows = [row for row in present if float(row[proxy]) < 0.5]
        correlation = point_biserial(pairs)
        report["proxies"][proxy] = {
            "defined_on": len(present),
            "correlation": correlation,
            "correlation_ci95": bootstrap_ci(pairs),
            "accuracy_when_true": (sum(row["judged_correct"] for row in true_rows) / len(true_rows)
                                   if true_rows else None),
            "accuracy_when_false": (sum(row["judged_correct"] for row in false_rows) / len(false_rows)
                                    if false_rows else None),
            "n_true": len(true_rows), "n_false": len(false_rows),
        }

    tokens = sorted(int(row.get("prompt_tokens", 0)) for row in rows)
    report["prompt_tokens"] = {
        "mean": sum(tokens) / len(tokens), "p50": tokens[len(tokens) // 2],
        "p95": tokens[max(0, int(0.95 * len(tokens)) - 1)], "max": max(tokens),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
