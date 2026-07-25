#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,re,statistics
from collections import Counter,defaultdict
from pathlib import Path
TOKEN_RE=re.compile(r"[\w][\w'-]*",re.UNICODE)
STOP={"a","an","and","are","as","at","be","by","for","from","i","in","is","it","my","of","on","or","the","to","was","were","with","you","your"}

def read(path):
    if not path or not path.exists():return []
    with path.open(encoding="utf-8") as h:return [json.loads(x) for x in h if x.strip()]
def terms(value):return {x.casefold() for x in TOKEN_RE.findall(str(value)) if x.casefold() not in STOP and len(x)>1}
def support(gold,text):
    wanted=terms(gold)
    if not wanted:return 1.0
    got=terms(text);return len(wanted&got)/len(wanted)
def avg(xs):return sum(xs)/len(xs) if xs else 0.0
def temporal_scope_ok(edge,node_by_id,chain_members):
    src=edge.get("src");dst=edge.get("dst")
    if chain_members.get(src,set())&chain_members.get(dst,set()):return True
    left=node_by_id.get(src,{});right=node_by_id.get(dst,{})
    generic={"","unknown","user","assistant","asst","conversation"}
    le={left.get("subject_key"),left.get("object_key")}-generic;re={right.get("subject_key"),right.get("object_key")}-generic
    if le&re:return True
    lp=left.get("predicate_key");rp=right.get("predicate_key")
    if lp==rp and lp not in generic:return True
    lc=left.get("context_key");return lc not in generic|{"default"} and lc==right.get("context_key")
def pct(xs,q):
    if not xs:return 0
    values=sorted(xs);return values[min(len(values)-1,max(0,int((len(values)-1)*q)))]
def main():
    p=argparse.ArgumentParser();p.add_argument("--run-dir",type=Path,required=True);p.add_argument("--judge-dir",type=Path);p.add_argument("--output-dir",type=Path);a=p.parse_args();out=a.output_dir or a.run_dir;out.mkdir(parents=True,exist_ok=True)
    nodes=read(a.run_dir/"nodes.jsonl");edges=read(a.run_dir/"edges.jsonl");chains=read(a.run_dir/"state_chains.jsonl");retrievals=read(a.run_dir/"retrieval_results.jsonl");answers=read(a.run_dir/"answers.jsonl");stats=read(a.run_dir/"question_stats.jsonl");index_diags=read(a.run_dir/"index_diagnostics.jsonl");judges=read(a.judge_dir/"auto_eval.jsonl") if a.judge_dir else []
    by_q_nodes=defaultdict(list);by_q_edges=defaultdict(list);by_q_chains=defaultdict(list);by_q_index_diags=defaultdict(list)
    for r in nodes:by_q_nodes[str(r.get("question_id"))].append(r)
    for r in edges:
        q=str(r.get("src","")).split(":",1)[0];by_q_edges[q].append(r)
    for r in chains:by_q_chains[str(r.get("question_id"))].append(r)
    for r in index_diags:by_q_index_diags[str(r.get("question_id"))].append(r)
    retrieval={str(r["question_id"]):r for r in retrievals};stat={str(r["question_id"]):r for r in stats};judge={str(r["question_id"]):r for r in judges}
    diagnostics=[]
    for answer in answers:
        qid=str(answer["question_id"]);qn=by_q_nodes[qid];qe=by_q_edges[qid];qc=by_q_chains[qid];qd=by_q_index_diags[qid];r=retrieval.get(qid,{});s=stat.get(qid,{});j=judge.get(qid)
        node_by_id={n["node_id"]:n for n in qn};facts=[n for n in qn if n.get("node_type")=="atomic_fact"];cards=[n for n in qn if n.get("node_type")=="routing_card"];leaves=[n for n in qn if n.get("node_type")=="leaf"]
        leaf_ids={n["node_id"] for n in leaves};fact_ids={n["node_id"] for n in facts};all_ids=set(node_by_id)
        extraction_diags=[row for row in qd if row.get("stage")=="session_extraction"]
        consolidation=next((row for row in reversed(qd) if row.get("stage")=="question_consolidation"),{})
        session_fact_counts=[int(row.get("fact_count",0)) for row in extraction_diags]
        card_token_estimates=[max(1,int(len(str(card.get("retrieval_text") or "").encode("utf-8"))/3.4+0.999)) for card in cards]
        edge_generators=Counter((edge.get("provenance") or {}).get("generator") for edge in qe)
        fact_source_errors=sum(not f.get("source_leaf_ids") or any(x not in leaf_ids for x in f.get("source_leaf_ids",[])) for f in facts)
        card_pointer_errors=sum(any(x not in fact_ids for x in c.get("fact_ids",[])) or any(x not in leaf_ids for x in c.get("leaf_ids",[])) for c in cards)
        edge_endpoint_errors=sum(e.get("src") not in all_ids or e.get("dst") not in all_ids for e in qe)
        relation_counts=Counter(e.get("relation") for e in qe)
        chain_members={f: {c["chain_id"] for c in qc if f in c.get("history_fact_ids",[])} for f in fact_ids}
        temporal_scope_warnings=sum(not temporal_scope_ok(e,node_by_id,chain_members) for e in qe if e.get("relation") in {"before","after"})
        chain_errors=sum(any(x not in c.get("history_fact_ids",[]) for x in c.get("current_fact_ids",[])) or c.get("update_order",[])!=c.get("history_fact_ids",[]) for c in qc)
        selected_fact_ids=set(r.get("fact_node_ids",[]));selected_leaf_ids=set(r.get("evidence_leaf_ids",[]));selected_card_ids=set(r.get("routing_card_ids",[]))
        selected_facts=[node_by_id[x] for x in selected_fact_ids if x in node_by_id];selected_leaves=[node_by_id[x] for x in selected_leaf_ids if x in node_by_id]
        source_ids={x for f in selected_facts for x in f.get("source_leaf_ids",[])}
        provenance_expansion_recall=len(source_ids&selected_leaf_ids)/len(source_ids) if source_ids else 1.0
        trace=r.get("retrieval_trace") or {};pre=trace.get("prepack") or {};post=trace.get("postpack") or {}
        index_text="\n".join(str(n.get("object") or n.get("raw_text") or "") for n in [*facts,*leaves])
        pre_text="\n".join(str(node_by_id.get(x,{}).get("object") or node_by_id.get(x,{}).get("raw_text") or "") for x in [*pre.get("fact_ids",[]),*pre.get("leaf_ids",[])])
        retrieval_text="\n".join(str(n.get("object") or n.get("raw_text") or "") for n in [*selected_facts,*selected_leaves])
        gold=answer.get("gold_answer","");index_support=support(gold,index_text);prepack_support=support(gold,pre_text);postpack_support=support(gold,retrieval_text);context_support=support(gold,r.get("context_text",""))
        correct=None if j is None else bool(j.get("correct"))
        if correct is True:failure="correct"
        elif correct is None:failure="unjudged"
        elif index_support<0.5:failure="index_extraction_miss"
        elif prepack_support<0.5:failure="retrieval_ranking_or_graph_miss"
        elif postpack_support<0.5:failure="evidence_packing_miss"
        elif context_support<0.5:failure="context_rendering_miss"
        else:failure="answer_reasoning_or_format_miss"
        diagnostics.append({"question_id":qid,"question_type":answer.get("question_type"),"correct":correct,"failure_stage":failure,
          "index":{"leaf_count":len(leaves),"fact_count":len(facts),"routing_card_count":len(cards),"state_chain_count":len(qc),"multi_fact_state_chain_count":sum(len(c.get("history_fact_ids",[]))>1 for c in qc),"edge_count":len(qe),"edges_per_index_node":len(qe)/max(1,len(facts)+len(cards)+len(leaves)),"avg_fact_confidence":avg([float(f.get("confidence",0)) for f in facts]),"fallback_fact_count":sum(float(f.get("confidence",0))<=0.45 for f in facts),"session_parse_error_count":sum(row.get("parse_error") is not None for row in extraction_diags),"session_length_finish_count":sum(row.get("finish_reason")=="length" for row in extraction_diags),"facts_per_session_p50":pct(session_fact_counts,.5),"facts_per_session_p95":pct(session_fact_counts,.95),"routing_card_token_p95":pct(card_token_estimates,.95),"routing_card_token_max":max(card_token_estimates,default=0),"routing_cards_over_180":sum(value>180 for value in card_token_estimates),"consolidation_allowed":consolidation.get("allowed"),"consolidation_parse_error":consolidation.get("parse_error"),"consolidation_accepted_edges":consolidation.get("accepted_edge_count",0),"fact_source_errors":fact_source_errors,"routing_pointer_errors":card_pointer_errors,"edge_endpoint_errors":edge_endpoint_errors,"state_chain_errors":chain_errors,"temporal_scope_warnings":temporal_scope_warnings,"relation_counts":dict(relation_counts),"edge_generator_counts":dict(edge_generators),"gold_term_support":index_support},
          "retrieval":{"query_kind":r.get("query_kind"),"card_count":len(selected_card_ids),"fact_count":len(selected_fact_ids),"leaf_count":len(selected_leaf_ids),"typed_expansion_additions":len(trace.get("typed_expanded_node_ids",[])),"adjacent_leaf_additions":len(trace.get("adjacent_leaf_ids",[])),"packer_card_retention":len(post.get("card_ids",[]))/max(1,len(pre.get("card_ids",[]))),"packer_fact_retention":len(post.get("fact_ids",[]))/max(1,len(pre.get("fact_ids",[]))),"packer_leaf_retention":len(post.get("leaf_ids",[]))/max(1,len(pre.get("leaf_ids",[]))),"packer_dropped_cards":len((trace.get("dropped_by_packer") or {}).get("card_ids",[])),"packer_dropped_facts":len((trace.get("dropped_by_packer") or {}).get("fact_ids",[])),"packer_dropped_leaves":len((trace.get("dropped_by_packer") or {}).get("leaf_ids",[])),"provider_token_estimate":r.get("packed_rough_tokens",0),"source_leaf_expansion_recall":provenance_expansion_recall,"gold_term_support_prepack":prepack_support,"gold_term_support_postpack":postpack_support,"gold_term_support_context":context_support,"gold_session_recall":r.get("answer_session_recall")},
          "tokens":{"build_miss":s.get("build_cache_miss_input_tokens",0),"build_hit":s.get("build_cache_hit_input_tokens",0),"build_output":s.get("build_output_tokens",0),"build_total":s.get("build_total_tokens",0),"answer_miss":s.get("answer_cache_miss_input_tokens",0),"answer_hit":s.get("answer_cache_hit_input_tokens",0),"answer_output":s.get("answer_output_tokens",0),"answer_total":s.get("answer_total_tokens",0),"reasoning":s.get("reasoning_tokens",0),"build_pass":s.get("build_budget_pass"),"answer_pass":s.get("answer_budget_pass")}})
    failures=Counter(d["failure_stage"] for d in diagnostics);types=defaultdict(list)
    for d in diagnostics:types[str(d.get("question_type") or "unknown")].append(d)
    summary={"question_count":len(diagnostics),"judged_count":sum(d["correct"] is not None for d in diagnostics),"accuracy":avg([int(d["correct"]) for d in diagnostics if d["correct"] is not None]),"failure_stages":dict(failures),
      "index_quality":{"fact_source_errors":sum(d["index"]["fact_source_errors"] for d in diagnostics),"routing_pointer_errors":sum(d["index"]["routing_pointer_errors"] for d in diagnostics),"edge_endpoint_errors":sum(d["index"]["edge_endpoint_errors"] for d in diagnostics),"state_chain_errors":sum(d["index"]["state_chain_errors"] for d in diagnostics),"temporal_scope_warnings":sum(d["index"]["temporal_scope_warnings"] for d in diagnostics),"fallback_fact_rate":sum(d["index"]["fallback_fact_count"] for d in diagnostics)/max(1,sum(d["index"]["fact_count"] for d in diagnostics)),"session_parse_error_rate":sum(d["index"]["session_parse_error_count"] for d in diagnostics)/max(1,sum(d["index"]["routing_card_count"] for d in diagnostics)),"session_length_finish_rate":sum(d["index"]["session_length_finish_count"] for d in diagnostics)/max(1,sum(d["index"]["routing_card_count"] for d in diagnostics)),"routing_cards_over_180":sum(d["index"]["routing_cards_over_180"] for d in diagnostics),"edge_density_avg":avg([d["index"]["edges_per_index_node"] for d in diagnostics])},
      "retrieval_quality":{"avg_gold_session_recall":avg([float(d["retrieval"]["gold_session_recall"] or 0) for d in diagnostics]),"avg_source_leaf_expansion_recall":avg([d["retrieval"]["source_leaf_expansion_recall"] for d in diagnostics]),"avg_index_gold_term_support":avg([d["index"]["gold_term_support"] for d in diagnostics]),"avg_postpack_gold_term_support":avg([d["retrieval"]["gold_term_support_postpack"] for d in diagnostics]),"avg_typed_expansion_additions":avg([d["retrieval"]["typed_expansion_additions"] for d in diagnostics]),"avg_adjacent_leaf_additions":avg([d["retrieval"]["adjacent_leaf_additions"] for d in diagnostics]),"avg_packer_card_retention":avg([d["retrieval"]["packer_card_retention"] for d in diagnostics]),"avg_packer_fact_retention":avg([d["retrieval"]["packer_fact_retention"] for d in diagnostics]),"avg_packer_leaf_retention":avg([d["retrieval"]["packer_leaf_retention"] for d in diagnostics])},
      "tokens":{"build_p50":pct([d["tokens"]["build_total"] for d in diagnostics],.5),"build_p95":pct([d["tokens"]["build_total"] for d in diagnostics],.95),"build_max":max([d["tokens"]["build_total"] for d in diagnostics],default=0),"answer_p50":pct([d["tokens"]["answer_total"] for d in diagnostics],.5),"answer_p95":pct([d["tokens"]["answer_total"] for d in diagnostics],.95),"answer_max":max([d["tokens"]["answer_total"] for d in diagnostics],default=0),"build_over":sum(d["tokens"]["build_pass"] is False for d in diagnostics),"answer_over":sum(d["tokens"]["answer_pass"] is False for d in diagnostics),"reasoning_tokens":sum(d["tokens"]["reasoning"] for d in diagnostics)},
      "by_question_type":{k:{"count":len(v),"accuracy":avg([int(x["correct"]) for x in v if x["correct"] is not None]),"failures":dict(Counter(x["failure_stage"] for x in v))} for k,v in types.items()}}
    with (out/"per_question_diagnostics.jsonl").open("w",encoding="utf-8") as h:
        for d in diagnostics:h.write(json.dumps(d,ensure_ascii=True)+"\n")
    (out/"pipeline_diagnostics.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    lines=["# GraphMem V2 Pipeline Diagnostics","",f"- Questions: {summary['question_count']}",f"- Judged accuracy: {summary['accuracy']:.3%}",f"- Failure stages: `{json.dumps(summary['failure_stages'],ensure_ascii=False)}`",f"- Index quality: `{json.dumps(summary['index_quality'],ensure_ascii=False)}`",f"- Retrieval quality: `{json.dumps(summary['retrieval_quality'],ensure_ascii=False)}`",f"- Tokens: `{json.dumps(summary['tokens'],ensure_ascii=False)}`"]
    (out/"pipeline_diagnostics.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(summary,ensure_ascii=False))
if __name__=="__main__":main()
