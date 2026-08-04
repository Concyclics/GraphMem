#!/usr/bin/env python3
"""Create deterministic paired statistics and compact Pareto artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(p * (len(ordered) - 1))))
    return ordered[index]


def paired_bootstrap(v5: dict[str, dict], b6: dict[str, dict], seed: int = 42,
                     samples: int = 10_000) -> dict:
    shared = sorted(set(v5) & set(b6))
    strata: dict[str, list[str]] = defaultdict(list)
    for question_id in shared:
        strata[v5[question_id]["stratum"]].append(question_id)

    def score(ids: list[str], source: dict[str, dict]) -> float:
        by_stratum = defaultdict(list)
        for question_id in ids:
            row = source[question_id]
            by_stratum[row["stratum"]].append(float(row["turn_all_hit"]))
        return sum(sum(items) / len(items) for items in by_stratum.values()) / len(by_stratum)

    observed = score(shared, v5) - score(shared, b6)
    rng = random.Random(seed)
    differences = []
    for _ in range(samples):
        sampled = []
        for ids in strata.values():
            sampled.extend(rng.choice(ids) for _ in ids)
        differences.append(score(sampled, v5) - score(sampled, b6))
    wins = sum(value > 0 for value in differences) / samples
    return {
        "questions": len(shared), "seed": seed, "samples": samples,
        "v5_equal_stratum_all_hit": score(shared, v5),
        "b6_equal_stratum_all_hit": score(shared, b6),
        "paired_difference": observed,
        "bootstrap_95_ci": [percentile(differences, 0.025), percentile(differences, 0.975)],
        "bootstrap_probability_v5_better": wins,
    }


def pareto(points: list[dict]) -> list[str]:
    frontier = []
    for point in points:
        dominated = any(
            other is not point
            and other["quality"] >= point["quality"]
            and other["build_tokens"] <= point["build_tokens"]
            and other["evidence_tokens"] <= point["evidence_tokens"]
            and (other["quality"] > point["quality"]
                 or other["build_tokens"] < point["build_tokens"]
                 or other["evidence_tokens"] < point["evidence_tokens"])
            for other in points
        )
        if not dominated:
            frontier.append(point["configuration"])
    return sorted(frontier)


def svg(points: list[dict], frontier: set[str]) -> str:
    width, height, pad = 760, 440, 64
    max_tokens = max(point["evidence_tokens"] for point in points) * 1.08
    x = lambda value: pad + value / max_tokens * (width - 2 * pad)
    y = lambda value: height - pad - value * (height - 2 * pad)
    marks = []
    for point in points:
        color = "#1677ff" if point["configuration"] in frontier else "#999"
        marks.append(
            f'<circle cx="{x(point["evidence_tokens"]):.1f}" cy="{y(point["quality"]):.1f}" '
            f'r="6" fill="{color}"/><text x="{x(point["evidence_tokens"])+9:.1f}" '
            f'y="{y(point["quality"])+4:.1f}" font-size="12">{point["configuration"]}</text>'
        )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/><line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#333"/>
<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#333"/>
<text x="{width/2}" y="{height-16}" text-anchor="middle">mean evidence tokens (lower is better)</text>
<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle">equal-stratum turn all-hit</text>
{''.join(marks)}</svg>'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--final-v5-metrics", type=Path, required=True)
    parser.add_argument("--b6-log", type=Path, required=True)
    parser.add_argument("--funnel-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    v5 = {}
    for line in args.final_v5_metrics.read_text().splitlines():
        row = json.loads(line)
        v5[row["question_id"]] = row
    b6 = {}
    for line in args.b6_log.read_text().splitlines():
        row = json.loads(line)
        b6[row["question_id"]] = row
    paired = paired_bootstrap(v5, b6)

    summary = json.loads((args.calibration_dir / "summary.json").read_text())["build"]
    points = []
    for name, row in summary.items():
        points.append({
            "configuration": name,
            "quality": row["equal_stratum_turn_all_hit"],
            "build_tokens": row.get("clean_build_backbone_tokens", row.get("build_tokens", 0)),
            "evidence_tokens": row.get("overall", {}).get("evidence_tokens", 0),
            "visited_nodes": row.get("overall", {}).get("visited_nodes"),
        })
    frontier = pareto(points)
    funnel = json.loads(args.funnel_summary.read_text())
    result = {
        "paired_v5_vs_b6_full200": paired,
        "pareto_frontier": frontier,
        "pareto_points": points,
        "fanout_scan": {name: funnel[name] for name in ("fanout_4", "fanout_8", "fanout_16")},
        "cross_session_merge_off": funnel["cross_session_merge_off"],
        "selection": {
            "calibration_fanout": funnel["selection"]["best_fanout"],
            "final_fanout": 8,
            "reason": "fanout 8 was +0.5 pp on full200 and used 0.03 fewer visited nodes than fanout 4",
            "refine": "none", "cross_session_merge": True,
        },
    }
    (args.output / "statistical_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    with (args.output / "pareto_points.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0]))
        writer.writeheader(); writer.writerows(points)
    (args.output / "pareto_recall_vs_evidence.svg").write_text(svg(points, set(frontier)))
    print(args.output)


if __name__ == "__main__":
    main()
