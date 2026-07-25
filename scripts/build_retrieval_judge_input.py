#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path

def rows(path):
    with path.open(encoding="utf-8") as h:
        for line in h:
            if line.strip(): yield json.loads(line)

def main():
    p=argparse.ArgumentParser();p.add_argument("--answers",type=Path,required=True);p.add_argument("--retrieval",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    retrieval={str(r["question_id"]):r for r in rows(a.retrieval)}
    with a.output.open("w",encoding="utf-8") as h:
        for answer in rows(a.answers):
            item=retrieval.get(str(answer["question_id"]));
            if not item: continue
            out={"question_id":answer["question_id"],"question_type":answer.get("question_type"),"question_date":answer.get("question_date"),"question":answer.get("question"),"gold_answer":answer.get("gold_answer"),"prediction":item.get("context_text","")}
            h.write(json.dumps(out,ensure_ascii=True)+"\n")
if __name__=="__main__":main()
