"""CLI for frozen research retrieval, Qwen/Astra answers, grading and accounting."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import argparse
import fcntl
import hashlib
import json
import platform
import statistics
from .data import read_rows, write_rows, sha
from .transport import Meter, digest, read_ledger

ASSETS = Path(__file__).parent / "assets"
BASE_MODELS = {"locomo": "BAAI/bge-m3", "longmemeval": "Qwen/Qwen3-Embedding-0.6B"}


def verify_assets():
    manifest = json.loads((ASSETS / "release_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if sha(Path(__file__).parent / name) != expected:
            raise ValueError("Frozen research asset changed: " + name)
    return {
        "verified": True,
        "files": len(manifest["files"]),
        "protocol": manifest["protocol"],
    }


def endpoints(path, benchmark):
    result = json.loads(Path(path).read_text())
    result["base_embedding"] = result.pop("base_embedding_" + benchmark)
    result.pop("base_embedding_locomo", None)
    result.pop("base_embedding_longmemeval", None)
    return result


def checkpoint(path, rows, key, operation, workers):
    old = read_rows(path) if path.exists() else []
    done = {r[key]: r for r in old}
    expected = {r[key] for r in rows}
    if (
        len(done) != len(old)
        or len(expected) != len(rows)
        or not done.keys() <= expected
    ):
        raise ValueError("Checkpoint IDs differ from input cohort")
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {
            pool.submit(operation, r): r[key] for r in rows if r[key] not in done
        }
        for future in as_completed(pending):
            identifier = pending[future]
            try:
                row = future.result()
            except Exception as e:
                failures.append(identifier)
                print(f"Failed {identifier}: {type(e).__name__}", flush=True)
                continue
            with path.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
            done[identifier] = row
            if len(done) % 50 == 0:
                print(f"{path.stem}: {len(done)}/{len(rows)}", flush=True)
    if failures:
        raise RuntimeError(
            f"{len(failures)} incomplete requests; inspect the ledger before resuming"
        )
    return [done[r[key]] for r in rows]


def answer(args):
    import numpy as np
    from tokenizers import Tokenizer
    from .encoding import query_usage
    from . import astra
    from .qwen import Reader
    from .readers import astra_stage2_shared as common

    root = args.root
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    benchmark = json.loads((root / "inputs.json").read_text())["benchmark"]
    config = endpoints(args.config, benchmark)
    paths = [
        "inputs.json",
        "input/queries.jsonl",
        "input/turns.json",
        "input/tokenizer.json",
        "retrieval.jsonl",
        "scores.npz",
        "base_query_usage.jsonl",
        "enhanced_query_usage.jsonl",
    ]
    identity = {
        "phase": args.phase,
        "oracle": args.oracle,
        "input_hashes": {p: sha(root / p) for p in paths},
        "config": config,
        "release_manifest_sha256": sha(ASSETS / "release_manifest.json"),
        "limit": args.limit,
    }
    marker = out / "identity.json"
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise ValueError("Answer run identity changed; use a new output directory")
    marker.write_text(json.dumps(identity, indent=2) + "\n")
    queries = read_rows(root / "input/queries.jsonl")
    if args.limit:
        queries = queries[: args.limit]
    turns = json.loads((root / "input/turns.json").read_text())
    retrieval = {
        r["qid"]: r["selected_ids"] for r in read_rows(root / "retrieval.jsonl")
    }
    scores = np.load(root / "scores.npz")
    tok = Tokenizer.from_file(str(root / "input/tokenizer.json"))
    costs = query_usage(root)
    meter = Meter(out / "query_calls.jsonl", config)
    labels = (
        {r["qid"]: r for r in read_rows(root / "evaluation/labels.jsonl")}
        if args.oracle
        else None
    )
    if args.oracle and args.phase != "astra":
        raise ValueError("Matched oracle uses the frozen Astra branch instructions")

    def work(q):
        qid = q["qid"]
        memory = turns[q["memory_id"]]
        if args.phase == "astra":
            prepared = astra.prepare(
                benchmark,
                q["query"],
                q.get("question_date", ""),
                memory,
                retrieval[qid],
                scores[qid],
                tok,
            )
            if labels is not None:
                sessions = set(labels[qid].get("gold_session_ids", []))
                selected = [t for t in memory if t["session_id"] in sessions]
                if benchmark == "locomo":
                    from .contexts import records

                    context = common.render_locomo(
                        q["query"], "", records(selected, "locomo")
                    )
                else:
                    context = common.render_lme(
                        q["query"], q.get("question_date", ""), selected
                    )
                prepared["payload"]["input"][0]["content"][0]["text"] = context
                prepared["source_ids"] = sorted(t["turn_id"] for t in selected)
            value, key = meter.call(
                qid, "answer", prepared["payload"], api="astra", kind="responses"
            )
            text = "".join(
                c.get("text", "")
                for item in value.get("output", [])
                if item.get("type") == "message"
                for c in item.get("content", [])
                if c.get("type") == "output_text"
            )
            parsed = common.parse_answer({"text": text, "status": value.get("status")})
            result = {
                "answer": parsed or text,
                "valid_output": parsed is not None,
                "truncated": value.get("status") != "completed",
                "source_ids": prepared["source_ids"],
                "branch": prepared["branch"],
                "call_keys": [key],
            }
        else:
            reader = Reader(meter, qid, tok)
            if benchmark == "locomo":
                r, ids = reader.locomo(q["query"], memory, retrieval[qid], scores[qid])
            else:
                r, ids = reader.longmemeval(
                    q["query"],
                    q.get("question_date", ""),
                    memory,
                    retrieval[qid],
                    scores[qid],
                )
            result = {
                "answer": r["answer"],
                "truncated": r["truncated"],
                "valid_output": not r["truncated"]
                and r["parsed"]
                and bool(r["answer"].strip()),
                "source_ids": sorted(ids),
                "call_keys": reader.keys,
            }
        usage = meter.usage(result["call_keys"])
        return {
            **result,
            "qid": qid,
            "benchmark": benchmark,
            "phase": args.phase,
            "oracle": args.oracle,
            "usage": usage,
            "query_embedding_tokens": costs[qid],
            "all_query_model_tokens": usage["total_tokens"] + costs[qid]
            if usage
            else None,
            "policy_api_calls": len(result["call_keys"]),
        }

    checkpoint(out / "answers.jsonl", queries, "qid", work, args.workers)
    (out / "environment.json").write_text(
        json.dumps({"python": platform.python_version()}, indent=2) + "\n"
    )


def grading(args):
    from .judging import judge

    root = args.root
    out = args.output
    benchmark = json.loads((root / "inputs.json").read_text())["benchmark"]
    meta = {r["qid"]: r for r in read_rows(root / "evaluation/metadata.jsonl")}
    queries = {q["qid"]: q for q in read_rows(root / "input/queries.jsonl")}
    rows = read_rows(out / "answers.jsonl")
    config = endpoints(args.config, benchmark)
    identity = json.loads((out / "identity.json").read_text())
    if identity["config"] != config:
        raise ValueError("Judge config differs from recorded run identity")
    meter = Meter(out / "judge_calls.jsonl", config)

    def work(row):
        m = {
            **meta[row["qid"]],
            "question_date": queries[row["qid"]].get("question_date", ""),
        }
        return judge(benchmark, m, row["answer"], meter)

    checkpoint(out / "judge.jsonl", rows, "qid", work, args.workers)


def report(root, out):
    benchmark = json.loads((root / "inputs.json").read_text())["benchmark"]
    queries = read_rows(root / "input/queries.jsonl")
    expected = {q["qid"] for q in queries}
    answers = (
        read_rows(out / "answers.jsonl") if (out / "answers.jsonl").exists() else []
    )
    judged = read_rows(out / "judge.jsonl") if (out / "judge.jsonl").exists() else []
    labels = {r["qid"]: r for r in read_rows(root / "evaluation/labels.jsonl")}
    aa = {r["qid"]: r for r in answers}
    jj = {r["qid"]: r for r in judged}
    if (
        len(aa) != len(answers)
        or len(jj) != len(judged)
        or not aa.keys() <= expected
        or not jj.keys() <= aa.keys()
    ):
        raise ValueError("Duplicate or unexpected IDs in outputs")
    if any(
        j["answer_sha256"] != hashlib.sha256(aa[q]["answer"].encode()).hexdigest()
        for q, j in jj.items()
    ):
        raise ValueError("Judgment does not match saved answer")
    coverage = []
    complete_sources = 0
    for q, r in aa.items():
        label = labels[q]
        if not label["valid"] or label.get("abstention"):
            continue
        gold = set(label["gold_turn_ids"])
        hits = len(gold & set(r["source_ids"]))
        coverage.append(hits / len(gold))
        complete_sources += int(hits == len(gold))
    calls = (
        read_ledger(out / "query_calls.jsonl")
        if (out / "query_calls.jsonl").exists()
        else []
    )
    unresolved = sum(c["status"] != "ok" or c.get("usage") is None for c in calls)
    used = {k for r in answers for k in r["call_keys"]}
    ledger = {c["key"]: c for c in calls}
    if len(ledger) != len(calls) or not used <= ledger.keys():
        raise ValueError("Missing or duplicate query call ledger entries")
    for row in answers:
        keys = row["call_keys"]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate charged call")
        amounts = [ledger[k].get("usage") for k in keys]
        if all(a is not None for a in amounts):
            total = sum(a["total_tokens"] for a in amounts)
            if (
                row["usage"]["total_tokens"] != total
                or row["all_query_model_tokens"]
                != total + row["query_embedding_tokens"]
            ):
                raise ValueError("Query ledger/answer accounting mismatch")
    costs = [r["all_query_model_tokens"] for r in answers]
    unknown = unresolved > 0 or any(c is None for c in costs)
    types = defaultdict(lambda: [0, 0])
    for r in judged:
        types[r["question_type"]][0] += int(r["correct"])
        types[r["question_type"]][1] += 1
    result = {
        "benchmark": benchmark,
        "expected_questions": len(expected),
        "answered": len(answers),
        "judged": len(judged),
        "correct": sum(r["correct"] for r in judged),
        "accuracy": sum(r["correct"] for r in judged) / len(expected),
        "complete": aa.keys() == jj.keys() == expected,
        "full_benchmark": len(expected) == (1540 if benchmark == "locomo" else 500),
        "unknown_query_calls": unresolved,
        "unreferenced_query_calls": len(ledger.keys() - used),
        "mean_all_query_model_tokens": None
        if unknown or not costs
        else statistics.mean(costs),
        "invalid_outputs": sum(not r.get("valid_output", False) for r in answers),
        "source_recall_questions": len(coverage),
        "mean_source_id_recall": statistics.mean(coverage) if coverage else None,
        "complete_source_questions": complete_sources,
        "by_type": dict(types),
        "oracle": bool(answers and answers[0].get("oracle")),
        "query_cost_scope": "All reader/repair/fallback input+output, plus both query embedding views; judge separate.",
        "source_metric": "Mean annotated source-turn coverage, not semantic fact coverage or answer accuracy.",
    }
    result["checks_pass"] = (
        result["complete"]
        and not unknown
        and not result["invalid_outputs"]
        and not result["unreferenced_query_calls"]
    )
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify-assets")
    for name in ["build", "prepare"]:
        p = sub.add_parser(name)
        p.add_argument("--benchmark", choices=list(BASE_MODELS), required=True)
        p.add_argument("--data", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--tokenizer", type=Path, required=True)
        if name == "prepare":
            p.add_argument("--graph-db", type=Path, required=True)
        else:
            p.add_argument("--build-config", type=Path, required=True)
    for name in ["embed-base", "embed-views", "features", "rank"]:
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, required=True)
        if name == "embed-base":
            p.add_argument("--config", type=Path, required=True)
        if name == "embed-views":
            p.add_argument("--model-path", type=Path, required=True)
            p.add_argument("--batch-size", type=int, default=16)
            p.add_argument("--device", default="cuda")
    for name in ["answer", "judge", "report"]:
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        if name != "report":
            p.add_argument("--config", type=Path, required=True)
            p.add_argument("--workers", type=int, default=4)
        if name == "answer":
            p.add_argument("--phase", choices=["qwen", "astra"], required=True)
            p.add_argument("--limit", type=int, default=0)
            p.add_argument("--oracle", action="store_true")
    args = parser.parse_args()
    if (
        getattr(args, "workers", 1) < 1
        or getattr(args, "limit", 0) < 0
        or getattr(args, "batch_size", 1) < 1
    ):
        parser.error("Positive worker/batch counts and nonnegative limit required")
    if args.command == "verify-assets":
        print(json.dumps(verify_assets()))
        return
    # An exclusive lease prevents two processes from changing the same checkpoint.
    location = getattr(args, "output", None) or args.root
    location.mkdir(parents=True, exist_ok=True)
    with (location / ".research.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "prepare":
            from .data import prepare

            prepare(
                args.output, args.benchmark, args.data, args.graph_db, args.tokenizer
            )
        elif args.command == "build":
            from .build_index import build

            print(
                json.dumps(
                    build(
                        args.benchmark,
                        args.data,
                        args.output,
                        args.build_config,
                        args.tokenizer,
                    )
                )
            )
        elif args.command in ["answer", "judge", "report"]:
            verify_assets()
            if args.command == "answer":
                answer(args)
            elif args.command == "judge":
                grading(args)
            else:
                result = report(args.root, args.output)
                print(json.dumps(result, indent=2))
                if not result["checks_pass"]:
                    raise SystemExit(1)
        else:
            benchmark = json.loads((args.root / "inputs.json").read_text())["benchmark"]
            if args.command == "embed-base":
                from .encoding import base_queries

                config = endpoints(args.config, benchmark)
                if config["base_embedding"]["model"] != BASE_MODELS[benchmark]:
                    raise ValueError(
                        "Base embedding model differs from the frozen benchmark profile"
                    )
                base_queries(
                    args.root,
                    BASE_MODELS[benchmark],
                    Meter(args.root / "embedding_calls.jsonl", config),
                )
            elif args.command == "embed-views":
                from .encoding import encode_views

                encode_views(
                    args.root,
                    benchmark,
                    args.model_path,
                    batch_size=args.batch_size,
                    device=args.device,
                )
            elif args.command == "features":
                from .features import export
                from .enhanced_features import export as enhance

                export(
                    args.root,
                    BASE_MODELS[benchmark],
                    longmemeval=benchmark == "longmemeval",
                )
                enhance(args.root, benchmark)
            elif args.command == "rank":
                from .retrieval import rank

                verify_assets()
                rank(args.root, benchmark)


if __name__ == "__main__":
    main()
