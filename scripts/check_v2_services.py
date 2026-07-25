#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys, urllib.request
from pathlib import Path
from dotenv import load_dotenv

ROOT=Path(__file__).resolve().parents[1]; load_dotenv(ROOT/".env",override=False)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--embedding-base-url",default=os.environ.get("EMBEDDING_BASE_URL","http://127.0.0.1:8001/v1")); p.add_argument("--embedding-model",default=os.environ.get("EMBEDDING_MODEL","Qwen3-Embedding-0.6B")); p.add_argument("--deepseek-model",default=os.environ.get("DEEPSEEK_MODEL","deepseek-v4-flash")); p.add_argument("--skip-deepseek-key",action="store_true"); a=p.parse_args()
    if a.deepseek_model!="deepseek-v4-flash": raise SystemExit("DEEPSEEK_MODEL must be deepseek-v4-flash")
    if not a.skip_deepseek_key and not os.environ.get("DEEPSEEK_API_KEY"): raise SystemExit("DEEPSEEK_API_KEY is missing from the environment/.env")
    base=a.embedding_base_url.rstrip("/")
    with urllib.request.urlopen(base+"/models",timeout=10) as response: models=json.load(response)
    ids=[str(row.get("id")) for row in models.get("data",[])]
    if a.embedding_model not in ids: raise SystemExit(f"embedding model {a.embedding_model!r} not served at {base}; available={ids}")
    request=urllib.request.Request(base+"/embeddings",data=json.dumps({"model":a.embedding_model,"input":["GraphMem V2 health check"]}).encode(),headers={"Content-Type":"application/json","Authorization":"Bearer local-embedding"},method="POST")
    with urllib.request.urlopen(request,timeout=30) as response: payload=json.load(response)
    dimension=len(payload["data"][0]["embedding"])
    if dimension!=1024: raise SystemExit(f"embedding dimension must be 1024, got {dimension}")
    print(json.dumps({"embedding_service":"healthy","embedding_model":a.embedding_model,"dimension":dimension,"deepseek_model":a.deepseek_model,"thinking_type":"disabled","reasoning_effort_field_sent":False}))
if __name__=="__main__": main()
