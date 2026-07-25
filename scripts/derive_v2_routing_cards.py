#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from graphmem_demo.hierarchical_v2 import _limit_rough, clean_entities

def rows(path):
    with path.open(encoding="utf-8") as h:
        for line in h:
            if line.strip():yield json.loads(line)

def values(parsed,*keys):
    out=[]
    for key in keys:
        value=parsed.get(key,[]) if isinstance(parsed,dict) else []
        if isinstance(value,str): value=[value]
        if isinstance(value,list):out.extend(str(x) for x in value if str(x).strip())
    return out

def main():
    p=argparse.ArgumentParser(description="E1: derive compact V2 routing cards from legacy parsed_summary without rebuilding memory.");p.add_argument("--legacy-nodes",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    candidates={}
    for row in rows(a.legacy_nodes):
        if row.get("node_type")!="summary":continue
        key=(str(row.get("question_id")),str(row.get("session_id")))
        if key not in candidates or int(row.get("level",0))>int(candidates[key].get("level",0)):candidates[key]=row
    with a.output.open("w",encoding="utf-8") as h:
        for (qid,sid),row in sorted(candidates.items()):
            parsed=row.get("parsed_summary") or {}; topics=values(parsed,"keywords","k")[:8]
            entities=clean_entities(values(row.get("anchor_terms") or {},"entities","keywords")+topics)[:12]
            events=values(parsed,"facts","updates","m")[:8];states=values(parsed,"updates","compact_summary","m")[-8:]
            time_range="; ".join(values(parsed,"time_anchors")[:4]) or str(row.get("session_date") or "unknown")
            text=_limit_rough(f"Session {sid} ({row.get('session_date') or 'unknown'}). Topics: {', '.join(topics)}. Entities: {', '.join(entities)}. Key events: {'; '.join(events)}. Current states: {'; '.join(states)}. Time range: {time_range}.",180)
            card={"schema_version":"graphmem_v2","node_type":"routing_card","node_id":f"{qid}:{sid}:routing:e1","question_id":qid,"session_id":sid,"session_date":row.get("session_date"),"topics":topics,"canonical_entities":entities,"key_events":events,"current_states":states,"time_range":time_range,"fact_ids":[],"leaf_ids":row.get("leaf_ids") or [],"retrieval_text":text,"derived_from_node_id":row.get("node_id"),"derived_without_llm":True}
            h.write(json.dumps(card,ensure_ascii=True)+"\n")
if __name__=="__main__":main()
