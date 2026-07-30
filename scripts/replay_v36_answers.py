#!/usr/bin/env python3
"""Replay V3.6 retrieval and one LLM answer from immutable persisted indexes."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any
import numpy as np
from dotenv import load_dotenv
ROOT=Path(__file__).resolve().parents[1]; load_dotenv(ROOT/'.env',override=False); sys.path.insert(0,str(ROOT/'src'))
from graphmem_demo.clients import EmbeddingClient, OpenAICompatibleClient
from graphmem_demo.data import load_longmemeval_cases
from graphmem_demo.v36.build import build_inverted_indexes
from graphmem_demo.v36.retrieval import answer_messages, build_query_ir, query_views, retrieve
from graphmem_demo.v36.schema import index_from_dict
VARIANT='hierarchical_role_graph_v3_6'

def args():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--index-dir',type=Path,action='append',required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--workers',type=int,default=16);p.add_argument('--context-token-budget',type=int,default=10000);p.add_argument('--max-answer-tokens',type=int,default=512);p.add_argument('--llm-model',default=os.environ.get('SGAO_MODEL','gpt-5.4-mini'));p.add_argument('--llm-base-url',default=os.environ.get('SGAO_BASE_URL','https://sub2api.sgao.me/v1/'));p.add_argument('--embedding-base-url',default=os.environ.get('EMBEDDING_BASE_URL','http://127.0.0.1:8001/v1'));p.add_argument('--embedding-model',default=os.environ.get('EMBEDDING_MODEL','Qwen3-Embedding-0.6B'));return p.parse_args()

def rows(path):
 if not path.exists(): return []
 return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def load_indexes(dirs,wanted):
 payloads={};sources={};buckets={'turn':'turns','role_frame':'frames','routing_card':'routing_cards','evidence_group':'evidence_groups'}
 for directory in dirs:
  for r in rows(directory/'nodes.jsonl'):
   q=str(r.get('question_id') or ''); bucket=buckets.get(str(r.get('node_type') or ''))
   if q not in wanted or bucket is None: continue
   p=payloads.setdefault(q,{'turns':[],'frames':[],'routing_cards':[],'evidence_groups':[],'edges':[],'state_chains':[],'coverage':[]});p[bucket].append({k:v for k,v in r.items() if k!='node_type'});sources[q]=directory
  for fn,bucket in [('edges.jsonl','edges'),('state_chains.jsonl','state_chains'),('coverage.jsonl','coverage')]:
   for r in rows(directory/fn):
    q=str(r.get('question_id') or '')
    if q in payloads: payloads[q][bucket].append({k:v for k,v in r.items() if k!='node_type'})
 missing=sorted(wanted-payloads.keys())
 if missing: raise RuntimeError(f'missing persisted indexes: {missing}')
 result={}
 for q,p in payloads.items():
  index=index_from_dict(p);key=hashlib.sha256(q.encode()).hexdigest()[:20];vd=sources[q]/'vectors';ids=json.loads((vd/f'{key}.ids.json').read_text());matrix=np.load(vd/f'{key}.npy',mmap_mode='r',allow_pickle=False);vectors={node_id:matrix[i].tolist() for i,node_id in enumerate(ids)}
  for node in [*index.routing_cards,*index.frames,*index.evidence_groups,*index.turns]: node.embedding=vectors.get(node.node_id)
  build_inverted_indexes(index);result[q]=index
 return result

def main():
 a=args();cases=load_longmemeval_cases(a.data,question_type='all');indexes=load_indexes(a.index_dir,{c.question_id for c in cases});a.output_dir.mkdir(parents=True,exist_ok=True);llm=OpenAICompatibleClient(model=a.llm_model,base_url=a.llm_base_url,api_key_env='SGAO_API_KEY',request_profile='openai');embed=EmbeddingClient(a.embedding_base_url,a.embedding_model)
 def run(case):
  ir=build_query_ir(case.question);qv=embed.embed(query_views(ir),question_id=case.question_id,variant=VARIANT);ret=retrieve(case=case,variant=VARIANT,index=indexes[case.question_id],query_vectors=qv,token_budget=a.context_token_budget);res=llm.chat(question_id=case.question_id,variant=VARIANT,stage='answer_qa',messages=answer_messages(case,ret),thinking_mode='none',max_tokens=a.max_answer_tokens);answer={'question_id':case.question_id,'variant':VARIANT,'question':case.question,'question_type':case.question_type,'question_date':case.question_date,'gold_answer':case.answer,'prediction':res.text.strip(),'answer_session_ids':case.answer_session_ids};trace=asdict(ret);trace['retrieval_trace']['answer_mode']='llm_from_persisted_v36_index';return answer,trace,asdict(res.record)
 answers=[];retrievals=[];calls=[]
 with ThreadPoolExecutor(max_workers=max(1,a.workers)) as pool:
  futures={pool.submit(run,c):c for c in cases}
  for f in as_completed(futures):
   x,r,c=f.result();answers.append(x);retrievals.append(r);calls.append(c);print(f'[{len(answers)}/{len(cases)}] {x["question_id"]} {x["prediction"][:100]}',flush=True)
 order={c.question_id:i for i,c in enumerate(cases)}
 for fn,data in [('answers.jsonl',answers),('retrieval_results.jsonl',retrievals),('llm_calls.jsonl',calls)]:
  data.sort(key=lambda x:order[x['question_id']])
  with (a.output_dir/fn).open('w') as h:
   for x in data:h.write(json.dumps(x,ensure_ascii=False)+'\n')
if __name__=='__main__':main()
