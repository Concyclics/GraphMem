#!/usr/bin/env python3
"""Fail closed when a V5.19 ablation or full benchmark violates its contract."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path


ALL_SIGNALS = {
    "scene_similar", "shared_entity", "state_compatible",
    "collection_related", "temporal_near", "lexical_rare",
}
LME_JUDGE_COMMIT = "bd063eea04de4f8a19927beea155afa094a01905"
LME_JUDGE_HASH = "ba8cf60d26f1390ecbef0f07b3e950556fe3bc5a37ba4b5343f28217f18c144f"
LOCOMO_JUDGE_COMMIT = "4b61c5d31b9c668a12b4f5e78064248a02c82d2b"
LOCOMO_JUDGE_HASH = "8ebac1ef60e9ab5caf99079fdaac038b85472e81491ed35e2d2655f3927c76c2"
ARMS = {
    "full": ALL_SIGNALS,
    "no_scene": ALL_SIGNALS - {"scene_similar"},
    "no_entity_family": ALL_SIGNALS - {
        "shared_entity", "state_compatible", "collection_related"},
    "no_temporal": ALL_SIGNALS - {"temporal_near"},
    "no_lexical": ALL_SIGNALS - {"lexical_rare"},
    "semantic_only": {"scene_similar"},
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").split("\n") if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def nearest_stats(values) -> dict:
    ordered = sorted(int(value) for value in values)
    def nearest(p: float) -> int:
        return ordered[max(0, math.ceil(p * len(ordered)) - 1)] if ordered else 0
    return {
        "count": len(ordered),
        "mean": sum(ordered) / max(1, len(ordered)),
        "p95": nearest(0.95), "p99": nearest(0.99),
        "max": max(ordered, default=0),
    }


def require_stats(actual: dict, expected: dict, label: str) -> None:
    for key in ("count", "p95", "p99", "max"):
        require(int(actual.get(key, -1)) == int(expected[key]),
                f"{label}: {key} differs: {actual.get(key)} != {expected[key]}")
    require(abs(float(actual.get("mean", -1)) - float(expected["mean"])) < 1e-9,
            f"{label}: mean differs")


def db_audit(path: Path) -> dict:
    signals = Counter()
    multi = 0
    checksums = []
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        for (source,) in connection.execute(
                "SELECT source FROM graph_edges "
                "WHERE source LIKE '%relation_mask:%'"):
            mask = str(source).split("relation_mask:", 1)[1].split("|", 1)[0]
            values = tuple(filter(None, mask.split(",")))
            signals.update(values)
            multi += len(values) > 1
        checksums = [str(row[0]) for row in connection.execute(
            "SELECT graph_checksum FROM graph_versions ORDER BY memory_id")]
        duplicate_edges = int(connection.execute(
            "SELECT COUNT(*) FROM (SELECT memory_id,src_id,dst_id,relation,COUNT(*) n "
            "FROM graph_edges GROUP BY memory_id,src_id,dst_id,relation "
            "HAVING n>1)").fetchone()[0])
    aggregate = hashlib.sha256("\n".join(checksums).encode()).hexdigest()
    return {"signals": dict(signals), "multi_attribute_edges": multi,
            "duplicate_endpoint_relation_edges": duplicate_edges,
            "memories": len(checksums), "aggregate_checksum": aggregate}


def semantic_cache_audit(path: Path) -> dict:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        rows = list(connection.execute(
            "SELECT stage,usage_json FROM llm_calls "
            "WHERE stage LIKE 'scene_semantic%'"))
    uncached = output = cached = 0
    for _, raw in rows:
        usage = json.loads(raw)
        uncached += int(usage.get("uncached_input_tokens", 0))
        output += int(usage.get("output_tokens", 0))
        cached += int(usage.get("cached_input_tokens", 0))
    return {"calls": len(rows), "cached_input_tokens": cached,
            "uncached_input_tokens": uncached, "output_tokens": output,
            "all_hits": bool(rows) and uncached == 0 and output == 0}


def answer_audit(root: Path, expected: int) -> dict:
    answers = jsonl(root / "answers.jsonl")
    prepared = jsonl(root / "prepared_answers.jsonl")
    retrieval = jsonl(root / "retrieval.jsonl")
    ids = {str(row["question_id"]) for row in answers}
    require(len(ids) == expected, f"{root}: expected {expected} answers, got {len(ids)}")
    require({str(row["question_id"]) for row in prepared} == ids,
            f"{root}: PreparedAnswer IDs differ")
    require({str(row["dev_question_id"]) for row in retrieval} == ids,
            f"{root}: retrieval IDs differ")
    prepared_hash = {str(row["question_id"]): str(
        row.get("prompt_payload_hash") or "") for row in prepared}
    require(all(str(row.get("prompt_payload_hash") or "") == prepared_hash[
        str(row["question_id"])] for row in answers),
        f"{root}: answer/prepared prompt hashes differ")
    require(all(int(row.get("api_prompt_tokens") or 0)
                + int(row.get("completion_tokens") or 0)
                == int(row.get("answer_total_tokens") or 0)
                for row in retrieval), f"{root}: non-additive API usage")
    manifest = read(root / "run_manifest.json")
    require(manifest.get("max_output_tokens") is None,
            f"{root}: answer completion cap must be omitted")
    if retrieval:
        token_rows = retrieval
        field_map = {"prompt": "api_prompt_tokens",
                     "completion": "completion_tokens",
                     "total": "answer_total_tokens"}
        recorded = manifest.get("answer_api_tokens", {})
    else:
        token_rows = jsonl(root / "answer_usage.jsonl")
        field_map = {"prompt": "api_prompt_tokens",
                     "completion": "completion_tokens",
                     "total": "total_tokens"}
        recorded = manifest.get("api_tokens", {})
    for metric, field in field_map.items():
        require_stats(dict(recorded.get(metric, {})), nearest_stats(
            row.get(field, 0) for row in token_rows),
            f"{root}:{metric} tokens")
    require(sum(int(row.get(field_map["prompt"]) or 0) for row in token_rows)
            + sum(int(row.get(field_map["completion"]) or 0) for row in token_rows)
            == sum(int(row.get(field_map["total"]) or 0) for row in token_rows),
            f"{root}: aggregate API usage differs")
    return {"answers": len(ids), "prompt_hashes": len(set(
        prepared_hash.values())), "token_stats_recomputed": True}


def judge_audit(root: Path, expected_lme: int, expected_locomo: int) -> dict:
    lme = jsonl(root / "judge_lme" / "auto_eval.jsonl")
    locomo = jsonl(root / "judge_locomo" / "auto_eval.jsonl")
    require(len({str(row["question_id"]) for row in lme}) == expected_lme,
            f"{root}: LongMemEval judge coverage is not {expected_lme}")
    require(len({str(row["question_id"]) for row in locomo}) == expected_locomo,
            f"{root}: LoCoMo judge coverage is not {expected_locomo}")
    require(all(row.get("judge_model") == "gpt-5.6-luna" for row in (*lme, *locomo)),
            f"{root}: unexpected judge model")
    require(all(row.get("judge_prompt_commit") == LME_JUDGE_COMMIT
                and row.get("judge_prompt_sha256") == LME_JUDGE_HASH
                for row in lme), f"{root}: LongMemEval judge prompt drift")
    require(all(row.get("judge_prompt_commit") == LOCOMO_JUDGE_COMMIT
                and row.get("judge_prompt_sha256") == LOCOMO_JUDGE_HASH
                for row in locomo), f"{root}: LoCoMo judge prompt drift")
    require(all(int(row.get("category", 1)) != 5 for row in locomo),
            f"{root}: LoCoMo Category 5 leaked into evaluation")
    for name in ("judge_lme", "judge_locomo"):
        stats = read(root / name / "judge_token_stats.json")
        require(stats.get("model") == "gpt-5.6-luna",
                f"{root}/{name}: unexpected judge stats model")
        require(int(stats.get("reasoning_tokens") or 0) == 0,
                f"{root}/{name}: judge reasoning was not disabled")
        require(int(stats.get("failure_count") or 0) == 0,
                f"{root}/{name}: unresolved judge failures")
        require(stats.get("temperature") == 0.0 and stats.get("seed") == 0,
                f"{root}/{name}: judge sampling contract drift")
    return {"longmemeval": len(lme), "locomo": len(locomo)}


def audit_ablation(args: argparse.Namespace) -> dict:
    result = {"mode": "ablation", "arms": {}}
    full_checksum = None
    for arm, expected_signals in ARMS.items():
        root = args.root / arm
        graph = db_audit(root / "graph" / "graphmem.sqlite")
        actual_signals = set(graph["signals"])
        require(actual_signals <= expected_signals,
                f"{arm}: disabled edge signal materialised: "
                f"{sorted(actual_signals - expected_signals)}")
        require(graph["memories"] == 110,
                f"{arm}: expected 110 memories, got {graph['memories']}")
        require(graph["duplicate_endpoint_relation_edges"] == 0,
                f"{arm}: duplicate relation edges materialised")
        if arm == "full":
            require(actual_signals == ALL_SIGNALS,
                    f"full: absent materialized signals {sorted(ALL_SIGNALS-actual_signals)}")
            require(graph["multi_attribute_edges"] > 0,
                    "full: no multi-attribute single edges were materialised")
            full_checksum = graph["aggregate_checksum"]
        else:
            require(graph["aggregate_checksum"] != full_checksum,
                    f"{arm}: aggregate graph checksum equals Full")
            cache = semantic_cache_audit(root / "graph" / "graphmem.sqlite")
            require(cache["all_hits"], f"{arm}: semantic extraction cache miss: {cache}")
            graph["semantic_cache"] = cache
        answer = answer_audit(root / "answer", args.expected_questions)
        judge = judge_audit(root / "answer", 100, 100)
        report = read(root / "build_report.json")
        require(set(report["summary"]["enabled_relation_signals"])
                == expected_signals, f"{arm}: report signal allow-list differs")
        candidate_signals = set()
        for row in report.get("rows", ()):
            diagnostic = dict(row.get("relation_candidate_diagnostics", {}))
            candidate_signals.update(map(str, dict(
                diagnostic.get("relation_mask_counts", {}))))
            candidate_signals.update(map(str, dict(
                diagnostic.get("atomic_candidate_signal_counts", {}))))
        require(candidate_signals <= expected_signals,
                f"{arm}: disabled candidate signal generated: "
                f"{sorted(candidate_signals - expected_signals)}")
        result["arms"][arm] = {"graph": graph, "answer": answer, "judge": judge}
    return result


def replay_audit(qwen: Path, gpt: Path, expected: int) -> dict:
    prepared = jsonl(qwen / "prepared_answers.jsonl")
    qwen_answers = jsonl(qwen / "answers.jsonl")
    gpt_answers = jsonl(gpt / "answers.jsonl")
    require(len(prepared) == expected and len(qwen_answers) == expected
            and len(gpt_answers) == expected, "dual-model answer coverage differs")
    frozen = {str(row["question_id"]): str(row["prompt_payload_hash"])
              for row in prepared}
    for label, rows in (("Qwen", qwen_answers), ("GPT", gpt_answers)):
        observed = {str(row["question_id"]): str(row.get(
            "prompt_payload_hash") or "") for row in rows}
        require(observed == frozen, f"{label}: prompt hash/ID map differs")
    usage = jsonl(gpt / "answer_usage.jsonl")
    require(len({str(row["question_id"]) for row in usage}) == expected,
            "GPT usage coverage differs")
    manifest = read(gpt / "run_manifest.json")
    require(manifest.get("max_output_tokens") is None,
            "GPT replay completion cap must be omitted")
    require(str(manifest.get("prepared_sha256")) == hashlib.sha256(
        (qwen / "prepared_answers.jsonl").read_bytes()).hexdigest(),
        "GPT replay did not record the frozen PreparedAnswer bytes")
    fields = {"prompt": "api_prompt_tokens", "completion": "completion_tokens",
              "total": "total_tokens"}
    for metric, field in fields.items():
        require_stats(dict(manifest.get("api_tokens", {}).get(metric, {})),
                      nearest_stats(row.get(field, 0) for row in usage),
                      f"GPT replay:{metric} tokens")
    require(sum(int(row.get("api_prompt_tokens") or 0) for row in usage)
            + sum(int(row.get("completion_tokens") or 0) for row in usage)
            == sum(int(row.get("total_tokens") or 0) for row in usage),
            "GPT replay aggregate API usage differs")
    return {"questions": expected, "prompt_hashes": len(set(frozen.values())),
            "identical": True, "token_stats_recomputed": True}


def audit_full(args: argparse.Namespace) -> dict:
    report = read(args.root / "build_report.json")
    summary = report["summary"]
    require(summary["memories_total"] == 510, "full build does not contain 510 memories")
    require(summary["token_ledger_memories"] == 510,
            "full token ledger does not contain 510 memories")
    require(not summary["failures"], "full build has failures")
    require(summary["retry_count"] == 0, "full build contains request retries")
    require(summary.get("fallback_and_degradation", {}).get(
        "extraction_retry_calls", 0) == 0,
        "full build contains semantic extraction retry calls")
    require(not summary["token_gate_violations"], "a memory exceeded 230K tokens")
    require(summary.get("build_diagnostic_memories") == 510,
            "fallback diagnostics do not cover all 510 memories")
    require(len(report.get("token_ledger", ())) == 510, "token ledger row mismatch")
    ledger = list(report["token_ledger"])
    for metric in ("input", "output", "total"):
        require_stats(
            dict(summary["build_token_stats"][metric]),
            nearest_stats(row[f"{metric}_tokens"] for row in ledger),
            f"build:{metric} tokens")
    require(sum(int(row["total_tokens"]) for row in ledger)
            == int(summary["tokens_total"]), "build token total differs")
    qwen = answer_audit(args.root / "qwen30", args.expected_lme + args.expected_locomo)
    qwen_judge = judge_audit(args.root / "qwen30", args.expected_lme,
                             args.expected_locomo)
    if args.qwen_only:
        return {"mode": "full-qwen", "build_memories": 510, "qwen": qwen,
                "judges": {"qwen30": qwen_judge}}
    gpt_manifest = read(args.root / "gpt54" / "run_manifest.json")
    require(gpt_manifest.get("api_usage_additivity_ok") is True,
            "GPT API usage is non-additive")
    replay = replay_audit(args.root / "qwen30", args.root / "gpt54",
                          args.expected_lme + args.expected_locomo)
    judges = {
        "qwen30": qwen_judge,
        "gpt54": judge_audit(args.root / "gpt54", args.expected_lme,
                             args.expected_locomo),
    }
    return {"mode": "full", "build_memories": 510, "qwen": qwen,
            "replay": replay, "judges": judges}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    ablation = sub.add_parser("ablation")
    ablation.add_argument("--root", type=Path, required=True)
    ablation.add_argument("--expected-questions", type=int, default=200)
    full = sub.add_parser("full")
    full.add_argument("--root", type=Path, required=True)
    full.add_argument("--expected-lme", type=int, default=500)
    full.add_argument("--expected-locomo", type=int, default=1540)
    full.add_argument("--qwen-only", action="store_true",
                      help="Audit the complete Qwen build/answer/judge contract "
                           "without requiring a cross-model replay")
    args = parser.parse_args()
    payload = audit_ablation(args) if args.mode == "ablation" else audit_full(args)
    output = args.root / "contract_audit.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
