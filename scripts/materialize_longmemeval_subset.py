#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path

def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def main():
    parser=argparse.ArgumentParser(description="Materialize a LongMemEval JSON subset in selection-file order.")
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--selection",type=Path,nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument(
        "--incorrect-only",action="store_true",
        help="Keep only selection rows whose correct field is explicitly false.",
    )
    args=parser.parse_args()
    rows=json.loads(args.data.read_text(encoding="utf-8"))
    by_id={str(row["question_id"]):row for row in rows}
    ids=[]
    for path in args.selection:
        selected_rows=read_jsonl(path)
        if args.incorrect_only:
            selected_rows=[row for row in selected_rows if row.get("correct") is False]
        ids.extend(str(row["question_id"]) for row in selected_rows)
    seen=set();ordered=[]
    for qid in ids:
        if qid not in seen:
            seen.add(qid);ordered.append(qid)
    missing=[qid for qid in ordered if qid not in by_id]
    if missing: raise SystemExit(f"Missing question ids: {missing[:10]}")
    selected=[by_id[qid] for qid in ordered]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(selected,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"questions":len(selected),"output":str(args.output)},ensure_ascii=False))

if __name__=="__main__": main()
