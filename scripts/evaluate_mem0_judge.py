#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
sys.path.insert(0, str(ROOT / "src"))

from graphmem_demo.clients import DeepSeekClient  # noqa: E402
from graphmem_demo.mem0_longmemeval_prompts import get_judge_prompt  # noqa: E402

PINNED_COMMIT = "bd063eea04de4f8a19927beea155afa094a01905"
PROMPT_SOURCE_SHA256 = "ba8cf60d26f1390ecbef0f07b3e950556fe3bc5a37ba4b5343f28217f18c144f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge GraphMem answers with the pinned Mem0 LongMemEval prompt.")
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata-jsonl", type=Path, help="Optional prior eval JSONL used only to restore question_type/date metadata.")
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--mode", choices=["answer","retrieval-sufficiency"], default="answer")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows=[]
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip(): rows.append(json.loads(line))
    return rows


def verdict(text: str) -> str:
    matches = re.findall(r"(?im)^\s*(yes|no)\s*[.!]?\s*$", text)
    if not matches:
        matches = re.findall(r"(?i)\b(yes|no)\b", text)
    if not matches:
        raise ValueError("judge response has no yes/no verdict")
    return matches[-1].casefold()


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    args=parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_path=args.output_dir/"auto_eval.jsonl"
    calls_path=args.output_dir/"judge_calls.jsonl"
    if not args.resume:
        eval_path.write_text("")
        calls_path.write_text("")
    completed={row["question_id"] for row in read_jsonl(eval_path)} if eval_path.exists() else set()
    rows=[row for row in read_jsonl(args.answers) if str(row["question_id"]) not in completed]
    if args.metadata_jsonl:
        metadata={str(row["question_id"]):row for row in read_jsonl(args.metadata_jsonl)}
        rows=[{**metadata.get(str(row["question_id"]),{}),**row} for row in rows]
    client=DeepSeekClient(model=args.model, base_url=args.base_url)

    def judge(row: dict):
        if args.mode=="answer":
            prompt=get_judge_prompt(
                str(row.get("question_type") or ""), str(row["question_id"]), str(row.get("question") or ""),
                str(row.get("gold_answer", row.get("answer", ""))), str(row.get("prediction", row.get("response", ""))),
                str(row.get("question_date") or ""),
            )
        else:
            prompt=("You are evaluating retrieval evidence sufficiency for LongMemEval. Decide whether the supplied evidence contains enough information to derive the reference answer to the question. Extra irrelevant or contradictory evidence does not by itself make retrieval insufficient; focus on whether the necessary supporting facts are present and identifiable. For recommendation/preference questions, evidence is sufficient when it contains the user preferences or compatibility constraints needed to tailor a reasonable recommendation, even if an exact recommendation is not already written. For a reference answer that says information is insufficient, an explicit exact-entity absence check over the full indexed memory is sufficient evidence; do not require a positive fact that the reference says is absent. When the question asks for a field such as who, when, or how many, judge whether the reference field value can be derived; an unrelated mismatch in a question presupposition does not negate a clearly supported field value unless it creates multiple plausible answers. Deterministic ledger calculations count as derived evidence only when their cited source facts are present and identifiable. Use the supplied question date to resolve relative dates. Output a brief <judge_thinking> analysis, then output exactly yes or no on the final line.\n\nQuestion date: "+str(row.get("question_date") or "unknown")+"\nQuestion: "+str(row.get("question") or "")+"\nReference answer: "+str(row.get("gold_answer",row.get("answer","")))+"\nEvidence:\n"+str(row.get("prediction",row.get("response",""))) )
        result=client.chat(
            question_id=str(row["question_id"]), variant="mem0_longmemeval_judge" if args.mode=="answer" else "retrieval_sufficiency_judge",
            stage="judge", messages=[{"role":"user","content":prompt}], thinking_mode="none",
            max_tokens=args.max_tokens, temperature=0.0, seed=0,
        )
        result.record.excluded_from_budget=True
        if result.record.reasoning_tokens != 0:
            raise RuntimeError(f"judge reasoning_tokens must be 0 for {row['question_id']}")
        if result.record.prompt_cache_hit_tokens + result.record.prompt_cache_miss_tokens != result.record.prompt_tokens:
            raise RuntimeError(f"invalid judge cache breakdown for {row['question_id']}")
        return row, result, verdict(result.text)

    with ThreadPoolExecutor(max_workers=max(1,args.workers)) as pool:
        futures=[pool.submit(judge,row) for row in rows]
        for future in as_completed(futures):
            row,result,label=future.result()
            append_jsonl(calls_path, asdict(result.record))
            append_jsonl(eval_path, {
                "question_id":str(row["question_id"]), "question_type":row.get("question_type"),
                "correct":label=="yes", "verdict":label, "judge_response":result.text,
                "judge_model":result.record.model, "judge_mode":args.mode,
                "judge_prompt_commit":PINNED_COMMIT if args.mode=="answer" else None,
                "judge_prompt_sha256":PROMPT_SOURCE_SHA256 if args.mode=="answer" else None,
            })
            print(f"judge {row['question_id']}: {label}", flush=True)

    evaluations=read_jsonl(eval_path); calls=read_jsonl(calls_path)
    by_type={}
    for row in evaluations:
        key=str(row.get("question_type") or "unknown")
        bucket=by_type.setdefault(key,{"correct":0,"total":0})
        bucket["total"]+=1; bucket["correct"]+=int(row["correct"])
    for bucket in by_type.values(): bucket["accuracy"]=bucket["correct"]/bucket["total"] if bucket["total"] else 0.0
    stats={
        "excluded_from_build_and_answer_budgets":True, "model":args.model,
        "thinking":{"type":"disabled"}, "reasoning_effort_field_sent":False,
        "judge_mode":args.mode, "prompt_commit":PINNED_COMMIT if args.mode=="answer" else None, "prompt_source_sha256":PROMPT_SOURCE_SHA256 if args.mode=="answer" else None,
        "question_count":len(evaluations), "correct":sum(int(row["correct"]) for row in evaluations),
        "accuracy":sum(int(row["correct"]) for row in evaluations)/len(evaluations) if evaluations else 0.0,
        "by_question_type":by_type,
        "prompt_cache_miss_tokens":sum(int(row.get("prompt_cache_miss_tokens") or 0) for row in calls),
        "prompt_cache_hit_tokens":sum(int(row.get("prompt_cache_hit_tokens") or 0) for row in calls),
        "completion_tokens":sum(int(row.get("completion_tokens") or 0) for row in calls),
        "reasoning_tokens":sum(int(row.get("reasoning_tokens") or 0) for row in calls),
        "total_tokens":sum(int(row.get("total_tokens") or 0) for row in calls),
    }
    (args.output_dir/"judge_token_stats.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"accuracy":stats["accuracy"],"questions":stats["question_count"]}))

if __name__ == "__main__": main()
