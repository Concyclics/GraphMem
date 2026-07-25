#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description="Select a frozen V2 config by the predeclared lexicographic objective.");p.add_argument("--metrics-jsonl",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    rows=[json.loads(line) for line in a.metrics_jsonl.read_text().splitlines() if line.strip()]
    eligible=[r for r in rows if bool(r.get("all_budget_pass")) and int(r.get("control_regressions",999))<=2]
    ranked=sorted(eligible,key=lambda r:(-float(r.get("retrieval_sufficiency",0)),-float(r.get("answer_accuracy",0)),int(r.get("control_regressions",999)),str(r.get("config_id",""))))
    payload={"selected":ranked[0] if ranked else None,"eligible_config_ids":[r.get("config_id") for r in ranked],"objective":["all per-question budgets pass","max post-pack retrieval sufficiency","max pinned-Mem0 answer accuracy","control regressions <=2"],"blind_test_must_remain_frozen":True}
    a.output.write_text(json.dumps(payload,indent=2)+"\n");print(json.dumps(payload))
    if not ranked: raise SystemExit(2)
if __name__=="__main__":main()
