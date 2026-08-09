#!/usr/bin/env python3
"""Render measured experiment artifacts into the Overleaf report template."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path,
                        default=Path("artifacts/report/v5_9"))
    parser.add_argument("--report", type=Path,
                        default=Path("../GraphMem_report"))
    return parser.parse_args()


def pct(value: float) -> str:
    return f"{value * 100:.1f}\\%"


def number(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def quantile(rows, key: str, p: float) -> float:
    values = sorted(float(row[key]) for row in rows)
    if not values:
        return 0.0
    return values[min(len(values) - 1, round((len(values) - 1) * p))]


def main() -> None:
    args = parse_args()
    generated = args.report / "generated"
    figures = args.report / "figures"
    generated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    macros: dict[str, str] = {}
    sources = {}

    c1_path = args.artifacts / "c1" / "c1_scaling.json"
    if c1_path.exists():
        c1 = json.loads(c1_path.read_text(encoding="utf-8"))
        sources["c1"] = str(c1_path)
        exponents = c1["candidate_scaling_exponent"]
        macros.update({
            "COneAllPairsExponent": f"{exponents['all_pairs']:.2f}",
            "COneCIRExponent": f"{exponents['cir']:.2f}",
        })
        by_key = {(row["method"], int(row["n"])): row for row in c1["rows"]}
        largest = max(row["n"] for row in c1["rows"])
        cir = by_key[("cir", largest)]
        all_pairs = by_key[("all_pairs", largest)]
        flat = by_key[("flat_sparse", largest)]
        macros.update({
            "COneLargestN": number(largest),
            "COneAllPairsCandidates": number(all_pairs["candidate_relations"]),
            "COneCIRCandidates": number(cir["candidate_relations"]),
            "COneFlatCandidates": number(flat["candidate_relations"]),
            "COneCIRToken": number(cir["relation_decision_tokens"]),
            "COneFlatToken": number(flat["relation_decision_tokens"]),
            "COneCIRPathRetention": pct(cir["multi_hop_path_retention"]),
            "COneFlatPathRetention": pct(flat["multi_hop_path_retention"]),
            "COneCandidateReduction": (
                f"{(1 - cir['candidate_relations'] / all_pairs['candidate_relations']) * 100:.2f}\\%"),
            "COneTokenReductionVsFlat": pct(
                1 - cir["relation_decision_tokens"]
                / max(1, flat["relation_decision_tokens"])),
        })

    c23_path = args.artifacts / "c23" / "c23_results.json"
    if c23_path.exists():
        c23 = json.loads(c23_path.read_text(encoding="utf-8"))
        sources["c23"] = str(c23_path)
        arms = c23["arms"]
        if {"flat@32", "fixed@32", "adaptive@32"} <= set(arms):
            macros.update({
                "CTwoQuestionN": str(arms["adaptive@32"]["n"]),
                "CTwoFlatAllHit": pct(arms["flat@32"]["all_hit"]),
                "CTwoFixedAllHit": pct(arms["fixed@32"]["all_hit"]),
                "CTwoAdaptiveAllHit": pct(arms["adaptive@32"]["all_hit"]),
                "CTwoFlatRouteRecall": pct(arms["flat@32"]["route_gold_session_recall"]),
                "CTwoFixedRouteRecall": pct(arms["fixed@32"]["route_gold_session_recall"]),
                "CTwoAdaptiveRouteRecall": pct(arms["adaptive@32"]["route_gold_session_recall"]),
                "CTwoPortalRouteGain": pct(
                    arms["adaptive@32"]["route_gold_session_recall"]
                    - arms["fixed@32"]["route_gold_session_recall"]),
                "CThreeFalseComplete": pct(arms["adaptive@32"]["false_complete_rate"]),
                "CTwoAdaptiveP": f"{arms['adaptive@32']['latency_ms']['p95']:.1f}",
                "CTwoFlatP": f"{arms['flat@32']['latency_ms']['p95']:.1f}",
                "CTwoHierarchyRouteMs": (
                    f"{arms['adaptive@32']['stage_mean_ms'].get('hierarchical_route', 0.0):.2f}"),
                "CTwoBudgetSixteenToken": number(
                    arms["adaptive@16"]["evidence_tokens_mean"]),
                "CTwoBudgetSixteenAllHit": pct(arms["adaptive@16"]["all_hit"]),
                "CTwoBudgetThirtyTwoToken": number(
                    arms["adaptive@32"]["evidence_tokens_mean"]),
            })

    system_path = args.artifacts / "system" / "system_results.json"
    if system_path.exists():
        system = json.loads(system_path.read_text(encoding="utf-8"))
        sources["system"] = str(system_path)
        reads = system["reads"]
        target_concurrency = (32 if any(row["concurrency"] == 32 for row in reads)
                              else max(row["concurrency"] for row in reads))
        selected = {(row["mode"], row["concurrency"]): row for row in reads}
        flat = selected[("flat", target_concurrency)]
        hierarchy = selected[("hierarchical", target_concurrency)]
        process_hierarchy = selected.get(
            ("process_hierarchical", target_concurrency), hierarchy)
        updates = system["updates"]["rows"]
        full = [row for row in updates if row["mode"] == "full_snapshot"]
        delta = [row for row in updates if row["mode"] == "affected_path"]
        hot_mib = system["memory"]["hierarchical_cache"]["estimated_bytes"] / 1024 ** 2
        process_hot_mib = (
            system["memory"].get("process_worker_snapshot_bytes",
                                 system["memory"]["hierarchical_cache"]["estimated_bytes"])
            / 1024 ** 2)
        full_p95 = quantile(full, "commit_ms", 0.95)
        delta_p95 = quantile(delta, "commit_ms", 0.95)
        macros.update({
            "SystemConcurrency": str(target_concurrency),
            "SystemFlatQPS": f"{flat['qps']:.1f}",
            "SystemFlatTail": (f"{flat['latency_ms']['p95']:.1f}/"
                               f"{flat['latency_ms']['p99']:.1f} ms"),
            "SystemHierarchyQPS": f"{hierarchy['qps']:.1f}",
            "SystemHierarchyTail": (f"{hierarchy['latency_ms']['p95']:.1f}/"
                                    f"{hierarchy['latency_ms']['p99']:.1f} ms"),
            "SystemProcessQPS": f"{process_hierarchy['qps']:.1f}",
            "SystemProcessTail": (
                f"{process_hierarchy['latency_ms']['p95']:.1f}/"
                f"{process_hierarchy['latency_ms']['p99']:.1f} ms"),
            "SystemProcessWorkers": str(
                system["memory"].get("process_worker_count", 1)),
            "SystemProcessMemory": f"{process_hot_mib:.1f} MiB",
            "SystemProcessQPSGain": (
                f"\\ensuremath{{{process_hierarchy['qps'] / max(1e-9, hierarchy['qps']):.1f}\\times}}"),
            "SystemHotMemory": f"{hot_mib:.1f} MiB",
            "SystemFullUpdateP": f"{full_p95:.1f} ms",
            "SystemDeltaUpdateP": f"{delta_p95:.1f} ms",
            "SystemDeltaSpeedup": (
                f"\\ensuremath{{{full_p95 / max(1e-9, delta_p95):.1f}\\times}}"),
            "SystemDeltaReduction": pct(1 - delta_p95 / max(1e-9, full_p95)),
            "SystemFullTouched": number(quantile(full, "touched_rows", 0.50)),
            "SystemDeltaTouched": number(quantile(delta, "touched_rows", 0.50)),
            "SystemReaderErrors": str(system["availability"]["reader_errors"]),
        })

    benchmark_path = args.artifacts / "full_benchmark" / "summary.json"
    if benchmark_path.exists():
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        sources["full_benchmark"] = str(benchmark_path)
        lme = benchmark["benchmarks"]["longmemeval"]
        locomo = benchmark["benchmarks"]["locomo"]
        lme_accuracy = lme["accuracy"]
        locomo_accuracy = locomo["accuracy"]
        lme_retrieval = lme["retrieval"]
        locomo_retrieval = locomo["retrieval"]
        lme_paired = lme["paired_vs_v5_8"]
        locomo_paired = locomo["paired_vs_v5_8"]
        macros.update({
            "EndToEndLMEQuestions": str(lme_accuracy["question_count"]),
            "EndToEndLMEAccuracy": pct(lme_accuracy["accuracy"]),
            "EndToEndLMEAllHit": pct(lme_retrieval["turn_all_hit"]),
            "EndToEndLMEAnnotatedN": str(lme_retrieval["annotated_questions"]),
            "EndToEndLMEQueryToken": number(lme_retrieval["prompt_tokens"]["mean"]),
            "EndToEndLMEQueryTokenP": number(lme_retrieval["prompt_tokens"]["p95"]),
            "EndToEndLMEEvidenceToken": number(lme_retrieval["evidence_tokens"]["mean"]),
            "EndToEndLMELatencyP": f"{lme_retrieval['retrieval_latency_ms']['p95']:.1f} ms",
            "EndToEndLMEClosedForm": pct(lme_retrieval["closed_form_rate"]),
            "EndToEndLMEMultiSession": pct(
                lme_accuracy["by_question_type"]["multi-session"]["accuracy"]),
            "EndToEndLMETemporal": pct(
                lme_accuracy["by_question_type"]["temporal-reasoning"]["accuracy"]),
            "EndToEndLMEKnowledgeUpdate": pct(
                lme_accuracy["by_question_type"]["knowledge-update"]["accuracy"]),
            "EndToEndLMEDelta": f"{lme_paired['delta'] * 100:+.1f} pp",
            "EndToEndLMEPValue": f"{lme_paired['mcnemar_exact_p']:.3f}",
            "EndToEndLoCoMoQuestions": str(locomo_accuracy["question_count"]),
            "EndToEndLoCoMoAccuracy": pct(locomo_accuracy["accuracy"]),
            "EndToEndLoCoMoAllHit": pct(locomo_retrieval["turn_all_hit"]),
            "EndToEndLoCoMoAnnotatedN": str(locomo_retrieval["annotated_questions"]),
            "EndToEndLoCoMoQueryToken": number(
                locomo_retrieval["prompt_tokens"]["mean"]),
            "EndToEndLoCoMoQueryTokenP": number(
                locomo_retrieval["prompt_tokens"]["p95"]),
            "EndToEndLoCoMoEvidenceToken": number(
                locomo_retrieval["evidence_tokens"]["mean"]),
            "EndToEndLoCoMoLatencyP": (
                f"{locomo_retrieval['retrieval_latency_ms']['p95']:.1f} ms"),
            "EndToEndLoCoMoClosedForm": pct(locomo_retrieval["closed_form_rate"]),
            "EndToEndLoCoMoMultiHop": pct(
                locomo_accuracy["by_category"]["1"]["accuracy"]),
            "EndToEndLoCoMoTemporal": pct(
                locomo_accuracy["by_category"]["2"]["accuracy"]),
            "EndToEndLoCoMoOpenDomain": pct(
                locomo_accuracy["by_category"]["3"]["accuracy"]),
            "EndToEndLoCoMoSingleHop": pct(
                locomo_accuracy["by_category"]["4"]["accuracy"]),
            "EndToEndLoCoMoTokenFOne": pct(
                locomo["official_token_f1"]["overall_f1"]),
            "EndToEndLoCoMoDelta": f"{locomo_paired['delta'] * 100:+.1f} pp",
            "EndToEndLoCoMoPValue": f"{locomo_paired['mcnemar_exact_p']:.3f}",
        })

    defaults = {
        "COneAllPairsExponent", "COneCIRExponent", "COneLargestN",
        "COneAllPairsCandidates", "COneCIRCandidates", "COneFlatCandidates",
        "COneCIRToken", "COneFlatToken", "COneCIRPathRetention",
        "COneFlatPathRetention", "CTwoQuestionN", "CTwoFlatAllHit",
        "COneCandidateReduction", "COneTokenReductionVsFlat",
        "CTwoFixedAllHit", "CTwoAdaptiveAllHit", "CTwoFlatRouteRecall",
        "CTwoFixedRouteRecall", "CTwoAdaptiveRouteRecall", "CTwoPortalRouteGain",
        "CThreeFalseComplete", "CTwoAdaptiveP", "CTwoFlatP",
        "CTwoHierarchyRouteMs", "CTwoBudgetSixteenToken",
        "CTwoBudgetSixteenAllHit", "CTwoBudgetThirtyTwoToken", "SystemConcurrency",
        "SystemFlatQPS", "SystemFlatTail", "SystemHierarchyQPS",
        "SystemHierarchyTail", "SystemProcessQPS", "SystemProcessTail",
        "SystemProcessWorkers", "SystemProcessMemory", "SystemProcessQPSGain",
        "SystemHotMemory", "SystemFullUpdateP", "SystemDeltaSpeedup",
        "SystemDeltaReduction",
        "SystemDeltaUpdateP", "SystemFullTouched", "SystemDeltaTouched",
        "SystemReaderErrors",
        "EndToEndLMEQuestions", "EndToEndLMEAccuracy", "EndToEndLMEAllHit",
        "EndToEndLMEAnnotatedN", "EndToEndLMEQueryToken",
        "EndToEndLMEQueryTokenP", "EndToEndLMEEvidenceToken",
        "EndToEndLMELatencyP", "EndToEndLMEClosedForm",
        "EndToEndLMEMultiSession", "EndToEndLMETemporal",
        "EndToEndLMEKnowledgeUpdate", "EndToEndLMEDelta",
        "EndToEndLMEPValue", "EndToEndLoCoMoQuestions",
        "EndToEndLoCoMoAccuracy", "EndToEndLoCoMoAllHit",
        "EndToEndLoCoMoAnnotatedN", "EndToEndLoCoMoQueryToken",
        "EndToEndLoCoMoQueryTokenP", "EndToEndLoCoMoEvidenceToken",
        "EndToEndLoCoMoLatencyP", "EndToEndLoCoMoClosedForm",
        "EndToEndLoCoMoMultiHop", "EndToEndLoCoMoTemporal",
        "EndToEndLoCoMoOpenDomain", "EndToEndLoCoMoSingleHop",
        "EndToEndLoCoMoTokenFOne", "EndToEndLoCoMoDelta",
        "EndToEndLoCoMoPValue",
    }
    lines = ["% Generated by scripts/render_report_results.py; do not hand edit."]
    for key in sorted(defaults):
        value = macros.get(key, "\\pending")
        lines.append(f"\\newcommand{{\\{key}}}{{{value}}}")
    (generated / "experiment_macros.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    for group, stem in (("c1", "eval_c1"), ("c23", "eval_c23"),
                        ("system", "eval_system")):
        for suffix in ("pdf", "svg", "png"):
            source = args.artifacts / group / f"{stem}.{suffix}"
            if source.exists():
                shutil.copy2(source, figures / source.name)
    manifest = {"sources": sources, "macros": macros}
    (generated / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
