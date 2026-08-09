#!/usr/bin/env python3
"""Audit a frozen GraphMem raw-to-index-to-answer error chain.

This experiment is intentionally read-only.  It joins the full benchmark
retrieval/answer/judge logs to the frozen SQLite graph and reports losses at
four independently measurable boundaries:

1. annotated source turns that have at least one CanonicalFact;
2. gold evidence reachable through the graph and the candidate reservoir;
3. gold evidence retained by the final evidence pack;
4. judged correctness given complete/incomplete packed evidence.

The output is a machine-readable JSON result plus a compact Markdown report.
No LLM is called, so repeated executions must produce identical counts.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", type=Path,
        default=WORKSPACE / "artifacts/v5_9/full_benchmark_20260809/graph/report_graph.sqlite",
    )
    parser.add_argument(
        "--run-root", type=Path,
        default=WORKSPACE / "artifacts/v5_9/full_benchmark_20260809/answers/merged",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=WORKSPACE / "artifacts/v5_9/full_benchmark_20260809/graph/recoarsen_manifest.json",
    )
    parser.add_argument(
        "--lme-gold", type=Path,
        default=ROOT / "eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl",
    )
    parser.add_argument(
        "--locomo", type=Path,
        default=WORKSPACE / "artifacts/data/locomo10_graphmem.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "artifacts/report/v5_9/error_chain",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()]


def rate(rows: Iterable[Any], predicate=lambda row: bool(row)) -> float:
    values = list(rows)
    return sum(bool(predicate(row)) for row in values) / len(values) if values else 0.0


def quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * probability) - 1))
    return ordered[index]


def conditional_accuracy(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    present = [row for row in rows if row.get(field) is not None]
    yes = [row for row in present if bool(row[field])]
    no = [row for row in present if not bool(row[field])]
    return {
        "defined_on": len(present),
        "true_n": len(yes),
        "false_n": len(no),
        "accuracy_when_true": rate(yes, lambda row: row["correct"]),
        "accuracy_when_false": rate(no, lambda row: row["correct"]),
        "delta": (rate(yes, lambda row: row["correct"])
                  - rate(no, lambda row: row["correct"])),
        "delta_bootstrap_ci95": bootstrap_delta(yes, no),
    }


def bootstrap_delta(yes: list[dict[str, Any]], no: list[dict[str, Any]],
                    resamples: int = 4000) -> list[float] | None:
    if len(yes) < 5 or len(no) < 5:
        return None
    rng = random.Random(42)
    values = []
    for _ in range(resamples):
        yes_rate = sum(yes[rng.randrange(len(yes))]["correct"] for _ in yes) / len(yes)
        no_rate = sum(no[rng.randrange(len(no))]["correct"] for _ in no) / len(no)
        values.append(yes_rate - no_rate)
    values.sort()
    return [values[int(0.025 * len(values))], values[int(0.975 * len(values))]]


def parse_locomo_refs(values: Iterable[str]) -> set[tuple[str, int]]:
    refs: set[tuple[str, int]] = set()
    for value in values or ():
        for part in str(value).split(";"):
            piece = part.strip()
            if ":" not in piece:
                continue
            day, turn = piece.split(":", 1)
            try:
                refs.add((f"session_{int(day[1:])}", int(turn) - 1))
            except ValueError:
                continue
    return refs


def gold_refs(lme_gold: Path, locomo: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(lme_gold):
        qid = str(row["question_id"])
        item = result.setdefault(qid, {
            "benchmark": "longmemeval", "memory_id": qid, "refs": set(),
        })
        item["refs"].add((str(row["session_id"]), int(row["turn_index"])))
    for row in json.loads(locomo.read_text(encoding="utf-8")):
        if int(row["locomo_category"]) not in {1, 2, 3, 4}:
            continue
        refs = parse_locomo_refs(row.get("locomo_evidence", ()))
        if refs:
            result[str(row["question_id"])] = {
                "benchmark": "locomo",
                "memory_id": "locomo:" + str(row["locomo_sample_id"]),
                "refs": refs,
            }
    return result


def query_scalar(db: sqlite3.Connection, sql: str) -> int | float:
    return db.execute(sql).fetchone()[0]


def graph_audit(db: sqlite3.Connection, manifest_path: Path) -> dict[str, Any]:
    node_types = {str(row[0]): int(row[1]) for row in db.execute(
        "SELECT node_type,count(*) FROM graph_nodes GROUP BY node_type")}
    facts = node_types.get("canonical_fact", 0)
    fact_turns = query_scalar(db, """
        SELECT count(DISTINCT em.turn_id)
        FROM graph_nodes n JOIN evidence_members em
          ON em.evidence_group_id=n.evidence_group_id
        WHERE n.node_type='canonical_fact'
    """)
    source_turns = query_scalar(db, "SELECT count(*) FROM source_turns")
    confidence_half = query_scalar(db, """
        SELECT count(*) FROM graph_nodes
        WHERE node_type='canonical_fact' AND confidence=0.5
    """)
    coarse = db.execute("""
        SELECT e.memory_id,a.level,e.confidence,
               json_extract(a.attributes_json,'$.session_id'),
               json_extract(b.attributes_json,'$.session_id')
        FROM graph_edges e
        JOIN graph_nodes a ON a.node_id=e.src_id
        JOIN graph_nodes b ON b.node_id=e.dst_id
        WHERE e.relation='coarse_related'
    """).fetchall()
    cross = [row for row in coarse if row[3] is not None and row[4] is not None
             and row[3] != row[4]]
    manifests = db.execute("""
        SELECT CAST(json_extract(attributes_json,'$.member_count') AS INTEGER)
        FROM graph_nodes WHERE node_type='collection_manifest'
    """).fetchall()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_rows = list(manifest.get("rows", ()))
    qwen_session_vectors = sum(
        int(row.get("qwen_session_vectors", 0)) for row in manifest_rows)
    qwen_vector_memories = sum(
        int(row.get("qwen_session_vectors", 0)) > 0 for row in manifest_rows)
    aggregate = manifest.get("aggregate") or {}
    if not aggregate and manifest.get("rows"):
        aggregate = {
            key: sum(int(row.get(key, 0)) for row in manifest["rows"])
            for key in ("coarsen_comparisons", "relation_comparisons",
                        "relation_candidates", "accepted_relations",
                        "deferred_refine_candidates")
        }
    return {
        "source_turns": int(source_turns),
        "node_types": node_types,
        "canonical_facts": facts,
        "turns_with_canonical_fact": int(fact_turns),
        "turn_fact_coverage": fact_turns / source_turns,
        "canonical_fact_confidence_exactly_0_5": int(confidence_half),
        "canonical_fact_confidence_exactly_0_5_rate": confidence_half / max(1, facts),
        "embeddings": int(query_scalar(db, "SELECT count(*) FROM embeddings")),
        "recoarsen_vector_contract": {
            "memories": len(manifest_rows),
            "memories_with_qwen_session_vectors": qwen_vector_memories,
            "qwen_session_vectors": qwen_session_vectors,
            "fallback_memories": len(manifest_rows) - qwen_vector_memories,
        },
        "coarse_related": {
            "edges": len(coarse),
            "level_0": sum(int(row[1]) == 0 for row in coarse),
            "level_1_plus": sum(int(row[1]) >= 1 for row in coarse),
            "cross_session": len(cross),
            "cross_session_rate": len(cross) / max(1, len(coarse)),
            "memories_with_cross_session": len({str(row[0]) for row in cross}),
            "confidence_mean": fmean(float(row[2]) for row in coarse),
        },
        "collection_manifests": {
            "total": len(manifests),
            "single_member": sum(int(row[0] or 0) == 1 for row in manifests),
            "single_member_rate": rate(manifests, lambda row: int(row[0] or 0) == 1),
        },
        "recoarsen": aggregate,
    }


def fact_coverage(db: sqlite3.Connection, refs: dict[str, dict[str, Any]]) -> tuple[dict, dict]:
    turn_lookup = {
        (str(memory), str(session), int(index)): str(turn_id)
        for turn_id, memory, session, index in db.execute(
            "SELECT turn_id,memory_id,session_id,turn_index FROM source_turns")
    }
    fact_turn_ids = {str(row[0]) for row in db.execute("""
        SELECT DISTINCT em.turn_id
        FROM graph_nodes n JOIN evidence_members em
          ON em.evidence_group_id=n.evidence_group_id
        WHERE n.node_type='canonical_fact'
        UNION
        SELECT DISTINCT em.turn_id
        FROM graph_nodes n, json_each(n.evidence_group_ids_json) groups
        JOIN evidence_members em ON em.evidence_group_id=groups.value
        WHERE n.node_type='canonical_fact'
    """)}
    per_question: dict[str, dict[str, Any]] = {}
    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for qid, item in refs.items():
        turn_ids = {
            turn_lookup.get((str(item["memory_id"]), session, index))
            for session, index in item["refs"]
        }
        turn_ids.discard(None)
        covered = turn_ids & fact_turn_ids
        row = {
            "benchmark": str(item["benchmark"]),
            "gold_refs": len(item["refs"]),
            "resolved_gold_turns": len(turn_ids),
            "gold_turns_with_fact": len(covered),
            "all_gold_turns_have_fact": bool(turn_ids) and covered >= turn_ids,
            "gold_turn_fact_recall": len(covered) / max(1, len(turn_ids)),
        }
        per_question[qid] = row
        by_benchmark[row["benchmark"]].append(row)
    summary = {}
    for benchmark, rows in by_benchmark.items():
        total = sum(row["resolved_gold_turns"] for row in rows)
        covered = sum(row["gold_turns_with_fact"] for row in rows)
        summary[benchmark] = {
            "questions": len(rows),
            "gold_turns": total,
            "gold_turns_with_fact": covered,
            "gold_turn_fact_recall": covered / max(1, total),
            "questions_all_gold_turns_have_fact": sum(
                row["all_gold_turns_have_fact"] for row in rows),
            "question_all_fact_rate": rate(rows, lambda row: row["all_gold_turns_have_fact"]),
        }
    return summary, per_question


def benchmark_audit(benchmark: str, retrieval: list[dict[str, Any]],
                    judged: dict[str, bool], fact_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for source in retrieval:
        if source.get("benchmark") != benchmark:
            continue
        qid = str(source["dev_question_id"])
        if qid not in judged:
            continue
        row = dict(source)
        row["correct"] = judged[qid]
        row["all_gold_turns_have_fact"] = fact_rows.get(qid, {}).get(
            "all_gold_turns_have_fact")
        rows.append(row)
    annotated = [row for row in rows if row.get("has_turn_gold")]
    wrong = [row for row in annotated if not row["correct"]]
    packed_missing = [row for row in wrong if not row.get("turn_all_hit")]
    packed_complete = [row for row in wrong if row.get("turn_all_hit")]
    latency_fields = [
        "latency_query_compile_ms", "latency_seed_fusion_ms",
        "latency_hierarchical_route_ms", "latency_graph_read_view_ms",
        "latency_fact_reservoir_ms", "latency_evidence_pack_ms", "latency_total_ms",
    ]
    result = {
        "questions": len(rows),
        "accuracy": rate(rows, lambda row: row["correct"]),
        "annotated_questions": len(annotated),
        "funnel": {
            "session_all_hit": rate(annotated, lambda row: row.get("session_all_hit")),
            "graph_all_reachable": rate(
                annotated, lambda row: float(row.get("graph_reachable_turn_recall", 0)) >= 1),
            "candidate_all_hit": rate(
                annotated, lambda row: row.get("candidate_turn_all_hit")),
            "packed_all_hit": rate(annotated, lambda row: row.get("turn_all_hit")),
            "wrong_annotated": len(wrong),
            "wrong_missing_packed_gold": len(packed_missing),
            "wrong_despite_all_gold_packed": len(packed_complete),
            "failure_stage": dict(Counter(str(row.get("failure_stage")) for row in rows)),
        },
        "caps": {
            "hop": sum(bool(row.get("search_hop_cap_reached")) for row in rows),
            "frontier": sum(bool(row.get("search_frontier_truncated")) for row in rows),
            "turn": sum(bool(row.get("pack_turn_cap_reached")) for row in rows),
            "token": sum(bool(row.get("pack_token_cap_reached")) for row in rows),
        },
        "tokens": {
            "evidence_mean": fmean(float(row["evidence_tokens"]) for row in rows),
            "evidence_p95": quantile((row["evidence_tokens"] for row in rows), 0.95),
            "prompt_mean": fmean(float(row["prompt_tokens"]) for row in rows),
            "prompt_p95": quantile((row["prompt_tokens"] for row in rows), 0.95),
        },
        "latency_ms": {
            field.removeprefix("latency_").removesuffix("_ms"): {
                "mean": fmean(float(row[field]) for row in rows),
                "p95": quantile((row[field] for row in rows), 0.95),
            }
            for field in latency_fields
        },
        "conditional_accuracy": {
            key: conditional_accuracy(rows if key in {"certificate_complete", "closed_form"}
                                      else annotated, key)
            for key in ("turn_all_hit", "candidate_turn_all_hit", "certificate_complete",
                        "closed_form", "all_gold_turns_have_fact")
        },
        "by_stratum": {},
    }
    for stratum in sorted({str(row["stratum"]) for row in rows}):
        subset = [row for row in rows if str(row["stratum"]) == stratum]
        subset_annotated = [row for row in subset if row.get("has_turn_gold")]
        result["by_stratum"][stratum] = {
            "questions": len(subset),
            "accuracy": rate(subset, lambda row: row["correct"]),
            "session_all_hit": rate(subset_annotated, lambda row: row.get("session_all_hit")),
            "graph_all_reachable": rate(
                subset_annotated,
                lambda row: float(row.get("graph_reachable_turn_recall", 0)) >= 1),
            "packed_all_hit": rate(subset_annotated, lambda row: row.get("turn_all_hit")),
        }
    return result


def render_markdown(payload: dict[str, Any], *, version: str = "V5.9") -> str:
    graph = payload["graph"]
    vector_contract = graph["recoarsen_vector_contract"]
    lines = [
        f"# {version} 全链路误差审计",
        "",
        "本报告由冻结图、完整检索日志和冻结 judge 结果确定性生成；不调用 LLM。",
        "",
        "## 原文到索引",
        "",
        "| 数据集 | Gold turn | 含 CanonicalFact | 问题全部覆盖 |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (("longmemeval", "LongMemEval"), ("locomo", "LoCoMo")):
        row = payload["fact_coverage"][key]
        lines.append(
            f"| {label} | {row['gold_turns']} | {row['gold_turn_fact_recall']:.2%} | "
            f"{row['question_all_fact_rate']:.2%} |"
        )
    lines.extend([
        "",
        f"全图 {graph['source_turns']:,} 个 turn 中，{graph['turns_with_canonical_fact']:,} 个至少有一条 "
        f"CanonicalFact（{graph['turn_fact_coverage']:.2%}）。全部 {graph['canonical_facts']:,} 条 "
        f"Fact 的 confidence 均为 0.5。",
        "",
        "## 图结构",
        "",
        f"`coarse_related` 共 {graph['coarse_related']['edges']:,} 条；可识别为跨 session 的只有 "
        f"{graph['coarse_related']['cross_session']:,} 条（{graph['coarse_related']['cross_session_rate']:.3%}），"
        f"覆盖 {graph['coarse_related']['memories_with_cross_session']} 个 memory。目标图 embeddings 表为 "
        f"{graph['embeddings']} 行；recoarsen manifest 显示 "
        f"{vector_contract['memories_with_qwen_session_vectors']}/{vector_contract['memories']} 个 memory "
        f"使用了 {vector_contract['qwen_session_vectors']} 个 Qwen session vector，其余 "
        f"{vector_contract['fallback_memories']} 个 memory 使用确定性 fallback。因此这不是统一 dense 条件。",
        "",
        "## 检索—打包—答案漏斗",
        "",
        "| 数据集 | Accuracy | Session all-hit | Graph all-reachable | Candidate all-hit | Packed all-hit | 错误：缺证据 / 证据齐全 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for key, label in (("longmemeval", "LongMemEval"), ("locomo", "LoCoMo")):
        row = payload["benchmarks"][key]
        funnel = row["funnel"]
        lines.append(
            f"| {label} | {row['accuracy']:.2%} | {funnel['session_all_hit']:.2%} | "
            f"{funnel['graph_all_reachable']:.2%} | {funnel['candidate_all_hit']:.2%} | "
            f"{funnel['packed_all_hit']:.2%} | {funnel['wrong_missing_packed_gold']} / "
            f"{funnel['wrong_despite_all_gold_packed']} |"
        )
    lines.extend([
        "",
        "## 主要方法论结论",
        "",
        "1. Candidate all-hit 接近饱和而 Packed all-hit 明显下降，最终 evidence packing 是可直接干预的主损失点。",
        "2. Gold turn 的 Fact 覆盖不完整，且 confidence 未校准；Fact-only 路径不能安全替代 lossless raw evidence。",
        "3. CIR 在当前冻结图上几乎没有生成跨 session 关系，尚不足以支撑跨会话多跳这一核心主张。",
        "4. certificate/closed-form 的条件正确率差需要用真实 candidate-off/bypass 消融确认，不能由相关性推断因果。",
        "5. 目标图 embeddings 表为空，且只有部分 memory 在 recoarsen 时获得 Qwen session vector；本次不是统一 dense 条件，不能归因成 HNSW+dense 完整系统收益。",
        "",
        "机器可读结果见 `error_chain.json`。",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    version = "V5.10" if "v5_10" in str(args.output).lower() else "V5.9"
    retrieval = read_jsonl(args.run_root / "retrieval.jsonl")
    judges = {
        "longmemeval": {str(row["question_id"]): bool(row["correct"])
                        for row in read_jsonl(args.run_root / "judge_lme/auto_eval.jsonl")},
        "locomo": {str(row["question_id"]): bool(row["correct"])
                   for row in read_jsonl(args.run_root / "judge_locomo/auto_eval.jsonl")},
    }
    refs = gold_refs(args.lme_gold, args.locomo)
    uri = f"file:{args.db.resolve()}?mode=ro"
    db = sqlite3.connect(uri, uri=True)
    graph = graph_audit(db, args.manifest)
    fact_summary, fact_per_question = fact_coverage(db, refs)
    benchmarks = {
        key: benchmark_audit(key, retrieval, judges[key], fact_per_question)
        for key in ("longmemeval", "locomo")
    }
    db.close()
    payload = {
        "schema_version": f"graphmem-{version.lower()}-error-chain-v1",
        "method": {
            "read_only": True,
            "llm_calls": 0,
            "bootstrap_seed": 42,
            "bootstrap_resamples": 4000,
            "correctness_source": "frozen benchmark judge logs",
        },
        "inputs": {key: str(value) for key, value in vars(args).items() if key != "output"},
        "graph": graph,
        "fact_coverage": fact_summary,
        "benchmarks": benchmarks,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "error_chain.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "error_chain.md").write_text(
        render_markdown(payload, version=version), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "longmemeval_accuracy": benchmarks["longmemeval"]["accuracy"],
        "locomo_accuracy": benchmarks["locomo"]["accuracy"],
        "cross_session_edge_rate": graph["coarse_related"]["cross_session_rate"],
    }, indent=2))


if __name__ == "__main__":
    main()
