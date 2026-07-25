#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, random
from collections import defaultdict
from pathlib import Path

SEED=20260724

def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as h: return [json.loads(line) for line in h if line.strip()]

def allocation(groups, limit):
    total=sum(len(v) for v in groups.values())
    if not total or not limit: return {k:0 for k in groups}
    raw={k:limit*len(v)/total for k,v in groups.items()}
    result={k:min(len(groups[k]),math.floor(value)) for k,value in raw.items()}
    for key in sorted(groups,key=lambda k:(-(raw[k]-math.floor(raw[k])),-len(groups[k]),k)):
        if sum(result.values())>=min(limit,total): break
        if result[key]<len(groups[key]): result[key]+=1
    while sum(result.values())<min(limit,total):
        for key in sorted(groups,key=lambda k:(-len(groups[k]),k)):
            if result[key]<len(groups[key]): result[key]+=1
            if sum(result.values())>=min(limit,total): break
    return result

def sample_stratified(rows, limit, rng):
    groups=defaultdict(list)
    for row in rows: groups[str(row.get("question_type") or "unknown")].append(row)
    for values in groups.values(): rng.shuffle(values)
    alloc=allocation(groups,limit)
    return [row for key in sorted(groups) for row in groups[key][:alloc[key]]]

def write_jsonl(path,rows):
    with path.open("w",encoding="utf-8") as h:
        for row in rows: h.write(json.dumps(row,ensure_ascii=True)+"\n")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--answers",type=Path,required=True); p.add_argument("--judgments",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--max-dev-errors",type=int,default=48); p.add_argument("--control-size",type=int,default=24); a=p.parse_args()
    answers={str(r["question_id"]):r for r in read_jsonl(a.answers)}
    judgments=read_jsonl(a.judgments); joined=[]
    for judgment in judgments:
        qid=str(judgment["question_id"]); row={**answers.get(qid,{}),**judgment,"question_id":qid}; joined.append(row)
    errors=[r for r in joined if not bool(r.get("correct"))]; correct=[r for r in joined if bool(r.get("correct"))]
    rng=random.Random(SEED); dev=sample_stratified(errors,min(a.max_dev_errors,len(errors)),rng); dev_ids={r["question_id"] for r in dev}
    blind=[r for r in errors if r["question_id"] not in dev_ids]; control=sample_stratified(correct,min(a.control_size,len(correct)),rng)
    a.output_dir.mkdir(parents=True,exist_ok=True); write_jsonl(a.output_dir/"dev_errors.jsonl",dev); write_jsonl(a.output_dir/"control_correct.jsonl",control); write_jsonl(a.output_dir/"blind_errors.jsonl",blind)
    manifest={"seed":SEED,"source_questions":len(joined),"incorrect":len(errors),"correct":len(correct),"dev_errors":len(dev),"control_correct":len(control),"blind_errors":len(blind),"blind_rule":"Do not tune after blind evaluation.","files":{"dev":"dev_errors.jsonl","control":"control_correct.jsonl","blind":"blind_errors.jsonl"}}
    (a.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest))
if __name__=="__main__": main()
