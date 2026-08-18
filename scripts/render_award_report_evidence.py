#!/usr/bin/env python3
"""Render award-report tables from frozen GraphMem experiment manifests.

The script deliberately keeps unlike measurement boundaries in separate panels:
quality/Token results come from the full benchmark manifest, while serving
results come from the warm retrieval data-plane manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = REPO_ROOT / "GraphMem_report"
DEFAULT_ACCURACY = REPO_ROOT / "artifacts/report/v5_63/latest_accuracy/summary.json"
DEFAULT_QUERYIR = REPO_ROOT / "artifacts/report/v5_10/queryir_gate_dev200/summary.json"
DEFAULT_CERTIFICATE = (
    REPO_ROOT
    / "artifacts/report/v5_10/packer_gate_sparse_dev200_turn32_monotone/summary.json"
)
DEFAULT_SERVING = DEFAULT_REPORT / "generated/v5_18_mem0_pareto_manifest.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fmt_int(value: float | int) -> str:
    return f"{round(value):,}"


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.1f}\\%"


def fmt_float(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def select(rows: list[dict[str, Any]], method: str, benchmark: str, setting: str) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["method"] == method
        and row["benchmark"] == benchmark
        and row["retrieval_setting"] == setting
        and row.get("accuracy") is not None
        and row.get("status", "complete") == "complete"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {method}/{benchmark}/{setting}, got {len(matches)}")
    return matches[0]


def render_main_results(data: dict[str, Any]) -> str:
    graphmem = data["graphmem"]
    mem0 = data["mem0"]
    configs = [
        ("GraphMem", "32-turn", "32-turn"),
        ("GraphMem", "64-turn", "64-turn"),
        ("Mem0", "top-50", "top-50"),
        ("Mem0", "top-200", "top-200"),
    ]
    lines: list[str] = []
    for method, setting, label in configs:
        rows = graphmem if method == "GraphMem" else mem0
        lme = select(rows, method, "longmemeval", setting)
        locomo = select(rows, method, "locomo", setting)
        lines.append(
            " & ".join(
                [
                    method,
                    label,
                    fmt_pct(lme["accuracy"]),
                    fmt_int(lme["answer_tokens"]["mean"]),
                    fmt_pct(locomo["accuracy"]),
                    fmt_int(locomo["answer_tokens"]["mean"]),
                    fmt_int(lme["build_tokens"]["mean"]),
                    fmt_int(locomo["build_tokens"]["mean"]),
                ]
            )
            + r" \\"
        )
    return "\n".join(lines) + "\n\\bottomrule\n"


def select_serving(
    rows: list[dict[str, Any]], system: str, workers: int, clients: int
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["system"] == system
        and row["workers"] == workers
        and row["clients"] == clients
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one serving row for {system}/W{workers}/C{clients}, got {len(matches)}"
        )
    return matches[0]


def relative_change(current: float, baseline: float) -> float:
    if baseline == 0:
        raise ValueError("cannot compute a relative change from a zero baseline")
    return 100.0 * (current / baseline - 1.0)


def fmt_accuracy_gain(current: float, baseline: float) -> str:
    absolute_pp = 100.0 * (current - baseline)
    relative_pct = relative_change(current, baseline)
    return rf"+{absolute_pp:.1f} pp（+{relative_pct:.1f}\%）"


def render_quick_comparison(
    accuracy_data: dict[str, Any], serving_data: dict[str, Any]
) -> str:
    graphmem = accuracy_data["graphmem"]
    mem0 = accuracy_data["mem0"]
    gm_lme = select(graphmem, "GraphMem", "longmemeval", "64-turn")
    gm_locomo = select(graphmem, "GraphMem", "locomo", "64-turn")
    mem0_lme = select(mem0, "Mem0", "longmemeval", "top-200")
    mem0_locomo = select(mem0, "Mem0", "locomo", "top-200")

    serving_rows = serving_data["rows"]
    gm_w8 = max(
        (row for row in serving_rows if row["system"] == "GraphMem" and row["workers"] == 8),
        key=lambda row: row["qps"],
    )
    mem0_w8 = max(
        (row for row in serving_rows if row["system"] == "Mem0 OSS" and row["workers"] == 8),
        key=lambda row: row["qps"],
    )
    gm_w1_c1 = select_serving(serving_rows, "GraphMem", 1, 1)
    mem0_w1_c1 = select_serving(serving_rows, "Mem0 OSS", 1, 1)

    def data_row(label: str, lme: dict[str, Any], locomo: dict[str, Any], qps: float, latency: float) -> str:
        return (
            " & ".join(
                [
                    label,
                    fmt_int(lme["build_tokens"]["mean"]),
                    fmt_int(lme["answer_tokens"]["mean"]),
                    fmt_pct(lme["accuracy"]),
                    fmt_int(locomo["build_tokens"]["mean"]),
                    fmt_int(locomo["answer_tokens"]["mean"]),
                    fmt_pct(locomo["accuracy"]),
                    fmt_float(qps, 2),
                    f"{fmt_float(latency, 2)} ms",
                ]
            )
            + r" \\"
        )

    delta_row = " & ".join(
        [
            r"\textbf{相对 Mem0}",
            rf"$\downarrow${-relative_change(gm_lme['build_tokens']['mean'], mem0_lme['build_tokens']['mean']):.1f}\%",
            rf"$\downarrow${-relative_change(gm_lme['answer_tokens']['mean'], mem0_lme['answer_tokens']['mean']):.1f}\%",
            fmt_accuracy_gain(gm_lme["accuracy"], mem0_lme["accuracy"]),
            rf"$\downarrow${-relative_change(gm_locomo['build_tokens']['mean'], mem0_locomo['build_tokens']['mean']):.1f}\%",
            rf"$\downarrow${-relative_change(gm_locomo['answer_tokens']['mean'], mem0_locomo['answer_tokens']['mean']):.1f}\%",
            fmt_accuracy_gain(gm_locomo["accuracy"], mem0_locomo["accuracy"]),
            r"$2.41\times$",
            rf"$\downarrow${-relative_change(gm_w1_c1['latency_mean_ms'], mem0_w1_c1['latency_mean_ms']):.1f}\%",
        ]
    ) + r" \\"

    return "\n".join(
        [
            data_row(
                "Mem0 (top-200)",
                mem0_lme,
                mem0_locomo,
                mem0_w8["qps"],
                mem0_w1_c1["latency_mean_ms"],
            ),
            data_row(
                "GraphMem (64-turn)",
                gm_lme,
                gm_locomo,
                gm_w8["qps"],
                gm_w1_c1["latency_mean_ms"],
            ),
            delta_row,
        ]
    ) + "\n\\bottomrule\n"


def render_queryir(data: dict[str, Any]) -> str:
    labels = [
        ("split_h10", "分离式编译"),
        ("unified_h11", "统一 QueryIR"),
        ("unified_directed", "统一 QueryIR + 定向执行"),
    ]
    lines = []
    for key, label in labels:
        row = data["overall"][key]
        lines.append(
            " & ".join(
                [
                    label,
                    fmt_pct(row["all_hit"]),
                    fmt_pct(row["recall"]),
                    fmt_pct(row["false_complete"]),
                    fmt_float(row["tokens"], 0),
                    fmt_float(row["visited_nodes"], 1),
                    fmt_float(row["visited_edges"], 1),
                    fmt_float(row["latency_ms"], 1),
                ]
            )
            + r" \\"
        )
    return "\n".join(lines) + "\n"


def render_certificate(data: dict[str, Any]) -> str:
    labels = [("baseline", "相关性打包"), ("obligation", "义务感知打包 + 复核")]
    lines = []
    for key, label in labels:
        row = data["overall"][key]
        lines.append(
            " & ".join(
                [
                    label,
                    fmt_pct(row["all_hit"]),
                    fmt_pct(row["recall"]),
                    fmt_pct(row["false_complete"]),
                    fmt_pct(row["post_pack_complete"]),
                    fmt_float(row["mean_evidence_tokens"], 0),
                    fmt_float(row["p95_evidence_tokens"], 0),
                    fmt_float(row["mean_latency_ms"], 1),
                ]
            )
            + r" \\"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--accuracy", type=Path, default=DEFAULT_ACCURACY)
    parser.add_argument("--queryir", type=Path, default=DEFAULT_QUERYIR)
    parser.add_argument("--certificate", type=Path, default=DEFAULT_CERTIFICATE)
    parser.add_argument("--serving", type=Path, default=DEFAULT_SERVING)
    args = parser.parse_args()

    output = args.report / "generated"
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        "accuracy": args.accuracy.resolve(),
        "queryir": args.queryir.resolve(),
        "certificate": args.certificate.resolve(),
        "serving": args.serving.resolve(),
    }
    rendered = {
        "award_main_results_table.tex": render_main_results(load_json(args.accuracy)),
        "award_quick_comparison_table.tex": render_quick_comparison(
            load_json(args.accuracy), load_json(args.serving)
        ),
        "award_queryir_table.tex": render_queryir(load_json(args.queryir)),
        "award_certificate_table.tex": render_certificate(load_json(args.certificate)),
    }
    for name, content in rendered.items():
        (output / name).write_text(content, encoding="utf-8")

    audit = {
        "schema_version": "award-report-evidence-v2",
        "sources": {
            key: {"path": str(path), "sha256": sha256(path)} for key, path in sources.items()
        },
        "outputs": sorted(rendered),
        "notes": [
            "Full-benchmark accuracy and token rows use the V5.63 final-audit manifest.",
            "QueryIR and certificate tables are V5.10 hard200 diagnostics, not full-benchmark QA results.",
            "The quick comparison table combines quality/Token evidence with warm retrieval data-plane serving evidence and labels the boundary explicitly.",
            "Eight-core QPS is the maximum measured point per system; one-core latency is the W1/C1 arithmetic mean.",
        ],
    }
    (output / "award_report_sources.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
