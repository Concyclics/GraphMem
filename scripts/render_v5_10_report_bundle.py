#!/usr/bin/env python3
"""Build a provenance-preserving V5.10 data bundle for the Overleaf report.

The V5.10 end-to-end run deliberately reuses the frozen P8 fact projection.
This renderer therefore keeps V5.10 macros separate from the report's existing
macros and emits the experiment scope next to every derived table.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FANDOL_REGULAR = Path(
    "/usr/local/texlive/2024/texmf-dist/fonts/opentype/public/fandol/FandolHei-Regular.otf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT.parent / "GraphMem_report")
    parser.add_argument("--repo-artifacts", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--shared-artifacts", type=Path, default=ROOT.parent / "artifacts")
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}\\%"


def pp(value: float, digits: int = 1) -> str:
    return f"{value * 100:+.{digits}f} pp"


def fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def embedded_cjk_font(text_value: str) -> str:
    """Return a compact self-contained WOFF2 font face for visible CJK text."""
    if not FANDOL_REGULAR.exists():
        return ""
    try:
        from fontTools import subset
        from fontTools.ttLib import TTFont

        font = TTFont(FANDOL_REGULAR)
        options = subset.Options()
        options.layout_features = ["*"]
        subsetter = subset.Subsetter(options=options)
        subsetter.populate(text="".join(sorted(set(text_value))))
        subsetter.subset(font)
        buffer = io.BytesIO()
        font.save(buffer)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return (
            '@font-face { font-family: "GraphMem CJK"; '
            f'src: url("data:font/otf;base64,{encoded}") format("opentype"); }}')
    except (ImportError, ModuleNotFoundError):
        return ""


def svg_document(title: str, body: str, *, width: int = 1200, height: int = 650) -> str:
    font_face = embedded_cjk_font(title + body)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" rx="24" fill="#F7F9FC"/>
<style>
{font_face}
text {{ font-family: "GraphMem CJK", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; fill: #172033; }}
.title {{ font-size: 28px; font-weight: 700; }}
.subtitle {{ font-size: 15px; fill: #5F6B7A; }}
.axis {{ stroke: #B7C0CE; stroke-width: 1; }}
.grid {{ stroke: #DDE3EC; stroke-width: 1; stroke-dasharray: 4 5; }}
.label {{ font-size: 16px; font-weight: 600; }}
.small {{ font-size: 13px; fill: #5F6B7A; }}
.value {{ font-size: 15px; font-weight: 700; }}
.card {{ fill: #FFFFFF; stroke: #DDE3EC; stroke-width: 1.5; }}
</style>
<text x="55" y="55" class="title">{html.escape(title)}</text>
{body}
</svg>
'''


def bar_panel(
    *, x: int, y: int, width: int, height: int, title: str,
    labels: list[str], series: list[tuple[str, str, list[float]]],
    value_format: str = "percent", maximum: float | None = None,
) -> str:
    values = [value for _name, _colour, rows in series for value in rows]
    max_value = maximum or max(values or [1.0])
    top = y + 58
    bottom = y + height - 62
    chart_h = bottom - top
    group_w = width / max(1, len(labels))
    bar_w = min(42.0, group_w / (len(series) + 1.2))
    out = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="18" class="card"/>',
        f'<text x="{x + 24}" y="{y + 37}" class="label">{html.escape(title)}</text>',
    ]
    for tick in range(5):
        value = max_value * tick / 4
        yy = bottom - chart_h * tick / 4
        out.append(f'<line x1="{x + 54}" y1="{yy:.1f}" x2="{x + width - 20}" y2="{yy:.1f}" class="grid"/>')
        shown = f"{value * 100:.0f}%" if value_format == "percent" else f"{value:.0f}"
        out.append(f'<text x="{x + 47}" y="{yy + 5:.1f}" text-anchor="end" class="small">{shown}</text>')
    chart_x0 = x + 60
    usable_w = width - 86
    actual_group_w = usable_w / max(1, len(labels))
    for index, label in enumerate(labels):
        center = chart_x0 + actual_group_w * (index + 0.5)
        total_bar_w = bar_w * len(series) + 8 * max(0, len(series) - 1)
        left = center - total_bar_w / 2
        for sidx, (_name, colour, rows) in enumerate(series):
            value = rows[index]
            bh = max(1.0, chart_h * value / max_value)
            bx = left + sidx * (bar_w + 8)
            by = bottom - bh
            out.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" rx="5" fill="{colour}"/>')
            shown = f"{value * 100:.1f}%" if value_format == "percent" else f"{value:.1f}"
            out.append(f'<text x="{bx + bar_w / 2:.1f}" y="{by - 7:.1f}" text-anchor="middle" class="value">{shown}</text>')
        out.append(f'<text x="{center:.1f}" y="{bottom + 27}" text-anchor="middle" class="small">{html.escape(label)}</text>')
    legend_x = x + width - 22
    for index, (name, colour, _rows) in enumerate(reversed(series)):
        offset = sum(len(item[0]) * 9 + 44 for item in list(reversed(series))[:index + 1])
        lx = legend_x - offset
        out.append(f'<rect x="{lx}" y="{y + 18}" width="14" height="14" rx="3" fill="{colour}"/>')
        out.append(f'<text x="{lx + 20}" y="{y + 30}" class="small">{html.escape(name)}</text>')
    return "\n".join(out)


def write_accuracy_latency(path: Path, current: dict[str, Any], previous: dict[str, Any]) -> None:
    cur_lme = current["benchmarks"]["longmemeval"]
    cur_loco = current["benchmarks"]["locomo"]
    old_lme = previous["benchmarks"]["longmemeval"]
    old_loco = previous["benchmarks"]["locomo"]
    body = '<text x="55" y="84" class="subtitle">同题、同回答模型；V5.10 使用冻结 P8 fact projection，仅升级图、QueryIR、seed 与 packing</text>'
    body += bar_panel(
        x=45, y=110, width=540, height=480, title="端到端准确率",
        labels=["LongMemEval", "LoCoMo"], maximum=1.0,
        series=[
            ("V5.9", "#9AA8BB", [old_lme["accuracy"]["accuracy"], old_loco["accuracy"]["accuracy"]]),
            ("V5.10", "#2563EB", [cur_lme["accuracy"]["accuracy"], cur_loco["accuracy"]["accuracy"]]),
        ])
    body += bar_panel(
        x=615, y=110, width=540, height=480, title="检索 p95（ms，越低越好）",
        labels=["LongMemEval", "LoCoMo"], maximum=4000, value_format="number",
        series=[
            ("V5.9", "#9AA8BB", [old_lme["retrieval"]["retrieval_latency_ms"]["p95"], old_loco["retrieval"]["retrieval_latency_ms"]["p95"]]),
            ("V5.10", "#10B981", [cur_lme["retrieval"]["retrieval_latency_ms"]["p95"], cur_loco["retrieval"]["retrieval_latency_ms"]["p95"]]),
        ])
    path.write_text(svg_document("V5.10 全量端到端：准确率与检索延迟", body), encoding="utf-8")


def write_error_chain(path: Path, audit: dict[str, Any]) -> None:
    lme = audit["benchmarks"]["longmemeval"]
    loco = audit["benchmarks"]["locomo"]
    labels = ["Session命中", "图内可达", "候选命中", "最终Pack"]
    def values(row: dict[str, Any]) -> list[float]:
        funnel = row["funnel"]
        return [funnel["session_all_hit"], funnel["graph_all_reachable"], funnel["candidate_all_hit"], funnel["packed_all_hit"]]
    body = '<text x="55" y="84" class="subtitle">覆盖率诊断不是严格串联漏斗：seed fallback 可使候选覆盖高于图关系可达率</text>'
    body += bar_panel(
        x=45, y=110, width=735, height=480, title="原文 → 图 → 候选 → Token Pack 覆盖率",
        labels=labels, maximum=1.0,
        series=[("LongMemEval", "#7C3AED", values(lme)), ("LoCoMo", "#2563EB", values(loco))])
    box_x = 810
    body += f'<rect x="{box_x}" y="110" width="345" height="480" rx="18" class="card"/>'
    body += f'<text x="{box_x + 24}" y="147" class="label">错误答案的证据归因</text>'
    rows = [
        ("LongMemEval", lme["funnel"]["wrong_annotated"], lme["funnel"]["wrong_missing_packed_gold"], lme["funnel"]["wrong_despite_all_gold_packed"]),
        ("LoCoMo", loco["funnel"]["wrong_annotated"], loco["funnel"]["wrong_missing_packed_gold"], loco["funnel"]["wrong_despite_all_gold_packed"]),
    ]
    yy = 205
    for name, wrong, missing, reasoning in rows:
        body += f'<text x="{box_x + 28}" y="{yy}" class="label">{name}</text>'
        body += f'<text x="{box_x + 28}" y="{yy + 35}" class="small">标注集错误：{wrong}</text>'
        body += f'<rect x="{box_x + 28}" y="{yy + 54}" width="{270 * missing / wrong:.1f}" height="28" rx="5" fill="#F59E0B"/>'
        body += f'<rect x="{box_x + 28 + 270 * missing / wrong:.1f}" y="{yy + 54}" width="{270 * reasoning / wrong:.1f}" height="28" rx="5" fill="#EF4444"/>'
        body += f'<text x="{box_x + 28}" y="{yy + 105}" class="small">缺证据 {missing}　|　证据全但推理错 {reasoning}</text>'
        yy += 170
    body += f'<text x="{box_x + 28}" y="545" class="small">主瓶颈：LoCoMo Cat.1 Pack all-hit 18.8%</text>'
    path.write_text(svg_document("V5.10 端到端误差链", body), encoding="utf-8")


def write_capacity(path: Path, runs: list[dict[str, Any]], incremental: dict[str, Any]) -> None:
    labels = [f'{row["workload"]["workers"]}w/{row["workload"]["clients"]}c' for row in runs]
    body = '<text x="55" y="84" class="subtitle">60 秒 Zipf 多租户负载；QPS、尾延迟、RSS 与故障恢复联合报告</text>'
    body += bar_panel(
        x=45, y=110, width=540, height=480, title="多租户吞吐（QPS）",
        labels=labels, maximum=65, value_format="number",
        series=[("QPS", "#2563EB", [row["qps"] for row in runs])])
    body += bar_panel(
        x=615, y=110, width=540, height=300, title="p95 延迟（ms）",
        labels=labels, maximum=3000, value_format="number",
        series=[("p95", "#F59E0B", [row["latency_ms"]["p95"] for row in runs])])
    recovery = incremental["worker_recovery"]["rto_ms"]
    raw = incremental["foreground"]["raw_durable"]["p95_ms"]
    publish = incremental["foreground"]["route_publish"]["p95_ms"]
    body += '<rect x="615" y="430" width="540" height="160" rx="18" class="card"/>'
    body += '<text x="639" y="467" class="label">增量与高可用探针</text>'
    body += f'<text x="639" y="505" class="small">Raw durable p95</text><text x="840" y="505" class="value">{raw:.2f} ms</text>'
    body += f'<text x="639" y="537" class="small">Route publish p95</text><text x="840" y="537" class="value">{publish:.2f} ms</text>'
    body += f'<text x="639" y="569" class="small">Worker SIGKILL RTO</text><text x="840" y="569" class="value">{recovery:.1f} ms</text>'
    body += '<text x="970" y="505" class="small">0 torn read</text><text x="970" y="537" class="small">stale promote rejected</text><text x="970" y="569" class="small">checksum verified</text>'
    path.write_text(svg_document("V5.10 多租户容量、增量写入与恢复", body), encoding="utf-8")


def render_png(svg_path: Path) -> Path | None:
    """Rasterize through the installed browser for direct Overleaf inclusion."""
    browser = shutil.which("google-chrome") or shutil.which("chromium")
    if browser is None:
        return None
    png_path = svg_path.with_suffix(".png")
    with tempfile.TemporaryDirectory(prefix="graphmem-v510-chrome-") as profile:
        try:
            subprocess.run(
                [
                    browser, "--headless", "--no-sandbox", "--disable-gpu",
                    "--disable-dev-shm-usage", "--hide-scrollbars",
                    "--force-device-scale-factor=2", "--window-size=1200,720",
                    f"--user-data-dir={profile}", f"--screenshot={png_path.resolve()}",
                    svg_path.resolve().as_uri(),
                ],
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as exc:
            print(f"warning: browser rasterization failed for {svg_path}: {exc}")
            return None
    return png_path


def main() -> None:
    args = parse_args()
    repo = args.repo_artifacts / "report"
    shared = args.shared_artifacts / "report"
    paths = {
        "atomic": repo / "v5_10/atomic_gate_v3/summary.json",
        "hnsw": repo / "v5_10/hnsw_scaling_bounded_final/summary.json",
        "packer": shared / "v5_10/packer_gate_sparse_dev200_turn32_monotone/summary.json",
        "graph_gate": shared / "v5_10/graph_gate_dev200_bounded_frontier/summary.json",
        "queryir": shared / "v5_10/queryir_gate_dev200/summary.json",
        "seed": shared / "v5_10/seed_gate_dev200/summary.json",
        "full": shared / "v5_10/full_benchmark/summary.json",
        "previous_full": repo / "v5_9/full_benchmark/summary.json",
        "error_chain": shared / "v5_10/error_chain/error_chain.json",
        "capacity_4w16": shared / "v5_10/multi_tenant_replica2_60s/summary.json",
        "capacity_8w16": shared / "v5_10/multi_tenant_w8_c16_60s/summary.json",
        "capacity_8w32": shared / "v5_10/multi_tenant_w8_c32_60s/summary.json",
        "incremental_ha": shared / "v5_10/incremental_ha_gate/summary.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing V5.10 inputs:\n" + "\n".join(missing))
    data = {key: load(path) for key, path in paths.items()}

    generated = args.report / "generated"
    figures = args.report / "figures"
    generated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    atomic = data["atomic"]
    hnsw = data["hnsw"]
    h20 = hnsw["rows"][-1]
    pack = data["packer"]["overall"]
    graph = data["graph_gate"]["overall"]
    queryir = data["queryir"]["overall"]
    seed = data["seed"]["overall"]
    full = data["full"]["benchmarks"]
    error = data["error_chain"]
    cap = data["capacity_8w16"]
    inc = data["incremental_ha"]

    macros = {
        "VTenScope": "冻结 P8 fact projection；未重建抽取层",
        "VTenAtomicCoverage": pct(atomic["unit_coverage"]),
        "VTenAtomicSufficiency": pct(atomic["sufficiency"]["v5_10"]),
        "VTenAtomicNegationCoverage": pct(atomic["per_kind"]["negation"]["coverage"]),
        "VTenHNSWExponent": fmt(hnsw["candidate_work_exponent"], 3),
        "VTenHNSWWorkRatioTwentyK": pct(h20["work_vs_all_pairs"], 2),
        "VTenHNSWGoldEdgeRecall": pct(h20["gold_edge_recall"], 2),
        "VTenHNSWTwoHopReachability": pct(h20["gold_pair_reachable_within_2_hops"], 2),
        "VTenHNSWWallTwentyK": f'{h20["wall_seconds"]:.2f} s',
        "VTenPackAllHitBefore": pct(pack["baseline"]["all_hit"]),
        "VTenPackAllHitAfter": pct(pack["obligation"]["all_hit"]),
        "VTenPackTokenReduction": pct(1 - pack["obligation"]["mean_evidence_tokens"] / pack["baseline"]["mean_evidence_tokens"]),
        "VTenGraphTwoHopBefore": pct(graph["frozen"]["two_hop_gold_session_path"]),
        "VTenGraphTwoHopAfter": pct(graph["hnsw"]["two_hop_gold_session_path"]),
        "VTenQueryIRAllHit": pct(queryir["unified_h11"]["all_hit"]),
        "VTenQueryIRFalseComplete": pct(queryir["unified_h11"]["false_complete"]),
        "VTenSeedMeanBefore": f'{seed["sqlite_fts"]["latency_ms"]:.1f} ms',
        "VTenSeedMeanAfter": f'{seed["native_index"]["latency_ms"]:.1f} ms',
        "VTenLMEAccuracy": pct(full["longmemeval"]["accuracy"]["accuracy"]),
        "VTenLMEAccuracyDelta": pp(full["longmemeval"]["paired_vs_v5_9"]["delta"]),
        "VTenLMEPValue": fmt(full["longmemeval"]["paired_vs_v5_9"]["mcnemar_exact_p"], 3),
        "VTenLMEAllHit": pct(full["longmemeval"]["retrieval"]["turn_all_hit"]),
        "VTenLMEP95": f'{full["longmemeval"]["retrieval"]["retrieval_latency_ms"]["p95"]:.1f} ms',
        "VTenLoCoMoAccuracy": pct(full["locomo"]["accuracy"]["accuracy"]),
        "VTenLoCoMoAccuracyDelta": pp(full["locomo"]["paired_vs_v5_9"]["delta"]),
        "VTenLoCoMoPValue": fmt(full["locomo"]["paired_vs_v5_9"]["mcnemar_exact_p"], 3),
        "VTenLoCoMoAllHit": pct(full["locomo"]["retrieval"]["turn_all_hit"]),
        "VTenLoCoMoP95": f'{full["locomo"]["retrieval"]["retrieval_latency_ms"]["p95"]:.1f} ms',
        "VTenCandidateAllHitLME": pct(error["benchmarks"]["longmemeval"]["funnel"]["candidate_all_hit"]),
        "VTenCandidateAllHitLoCoMo": pct(error["benchmarks"]["locomo"]["funnel"]["candidate_all_hit"]),
        "VTenGoldFactRecallLME": pct(error["fact_coverage"]["longmemeval"]["gold_turn_fact_recall"]),
        "VTenGoldFactRecallLoCoMo": pct(error["fact_coverage"]["locomo"]["gold_turn_fact_recall"]),
        "VTenCrossSessionEdgeRate": pct(error["graph"]["coarse_related"]["cross_session_rate"], 2),
        "VTenSingleMemberManifestRate": pct(error["graph"]["collection_manifests"]["single_member_rate"]),
        "VTenCapacityQPS": fmt(cap["qps"], 1),
        "VTenCapacityP95": f'{cap["latency_ms"]["p95"]:.1f} ms',
        "VTenCapacityRSS": f'{cap["total_worker_rss_mib"]:.0f} MiB',
        "VTenRawDurableP95": f'{inc["foreground"]["raw_durable"]["p95_ms"]:.2f} ms',
        "VTenRoutePublishP95": f'{inc["foreground"]["route_publish"]["p95_ms"]:.2f} ms',
        "VTenWorkerRTO": f'{inc["worker_recovery"]["rto_ms"]:.1f} ms',
    }
    macro_lines = [
        "% Generated by scripts/render_v5_10_report_bundle.py; do not hand edit.",
        "% The end-to-end run uses a frozen P8 fact projection; it is not an extraction rebuild.",
    ]
    for key, value in sorted(macros.items()):
        macro_lines.append(f"\\newcommand{{\\{key}}}{{{value}}}")
    macro_path = generated / "v5_10_experiment_macros.tex"
    macro_path.write_text("\n".join(macro_lines) + "\n", encoding="utf-8")

    tables = {
        "scope": data["full"]["scope"],
        "end_to_end": {
            name: {
                "questions": row["accuracy"]["question_count"],
                "accuracy": row["accuracy"]["accuracy"],
                "paired_delta_vs_v5_9": row["paired_vs_v5_9"]["delta"],
                "mcnemar_exact_p": row["paired_vs_v5_9"]["mcnemar_exact_p"],
                "turn_all_hit": row["retrieval"]["turn_all_hit"],
                "prompt_tokens_mean": row["retrieval"]["prompt_tokens"]["mean"],
                "prompt_tokens_p95": row["retrieval"]["prompt_tokens"]["p95"],
                "retrieval_p95_ms": row["retrieval"]["retrieval_latency_ms"]["p95"],
            }
            for name, row in full.items()
        },
        "mechanism_gates": {
            "atomic_extraction": atomic,
            "hnsw_bounded_at_20k": h20,
            "hnsw_candidate_exponent": hnsw["candidate_work_exponent"],
            "obligation_packer": {"baseline": pack["baseline"], "obligation": pack["obligation"], "paired_delta": data["packer"]["paired_delta"]},
            "graph_gate": graph,
            "queryir": queryir,
            "native_seed": seed,
        },
        "capacity": [data["capacity_4w16"], data["capacity_8w16"], data["capacity_8w32"]],
        "incremental_ha": inc,
        "error_chain": error,
        "claim_boundaries": [
            "全量 benchmark 未重建 V5.10 atomic extractor；只能归因于检索与系统路径升级。",
            "typed relation 在 dev200 中未被 traversal 使用；不能宣称其带来主结果提升。",
            "LongMemEval 相对 V5.9 为 -0.6pp，McNemar p=0.828；LoCoMo +1.62pp，p=0.055，均不构成 p<0.05 的显著提升。",
            "fact/relation authority commit latency 不包含远端 LLM extraction 与 embedding latency。",
            "HNSW scaling 是受控结构实验；不能替代真实 QA 或 typed-edge precision。",
        ],
    }
    table_path = generated / "v5_10_tables.json"
    table_path.write_text(json.dumps(tables, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    accuracy_svg = figures / "v5_10_accuracy_latency.svg"
    error_svg = figures / "v5_10_error_chain.svg"
    capacity_svg = figures / "v5_10_capacity_ha.svg"
    write_accuracy_latency(accuracy_svg, data["full"], data["previous_full"])
    write_error_chain(error_svg, data["error_chain"])
    capacity_runs = [data["capacity_4w16"], data["capacity_8w16"], data["capacity_8w32"]]
    write_capacity(capacity_svg, capacity_runs, data["incremental_ha"])
    rasterized = [path for path in (
        render_png(accuracy_svg), render_png(error_svg), render_png(capacity_svg))
        if path is not None]

    manifest = {
        "schema_version": "graphmem-v5.10-report-bundle-v1",
        "scope": data["full"]["scope"],
        "generated_files": [
            str(macro_path), str(table_path),
            str(accuracy_svg), str(error_svg), str(capacity_svg),
            *(str(path) for path in rasterized),
        ],
        "sources": {
            key: {"path": str(path.resolve()), "sha256": sha256(path)}
            for key, path in paths.items()
        },
        "tokenizer": data["full"]["run_manifest"]["token_counter"],
    }
    manifest_path = generated / "v5_10_experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
