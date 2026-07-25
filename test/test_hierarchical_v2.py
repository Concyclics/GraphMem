from __future__ import annotations

import hashlib
import json
from pathlib import Path

from graphmem_demo.hierarchical_v2 import (
    _ALLOWED, _arithmetic_result, _assistant_recall_result, _category_count_result, _event_comparison_result, _event_sequence_result,
    _candidate_pool_is_complete, _exact_entity_result, _explicit_event_time_result, _mandatory_answer_hint, _operator_result, _pack_context,
    _current_competitive_record_result, _event_companion_result, _previous_status_result,
    apply_answer_constraint,
    _preference_context_facts, _preference_focus_instruction,
    _question_date_scope, _target_date_answer_result, _temporal_calculation_result,
    _typed_expand,
    apply_consolidation, build_evidence_ledger, build_graph_edges, build_state_chains,
    canonical_key, clean_entities, parse_session_extraction, provider_token_estimate, query_kind, retrieve,
)
from graphmem_demo.models import AtomicFactNode, DeepSeekCallRecord, GraphEdge, LeafNode, RetrievedContext, RoutingCardNode
from graphmem_demo.stats import build_question_stats


def leaf(node_id="q:s:leaf:0", turn=0, text="User: I like tea."):
    return LeafNode(node_id=node_id,question_id="q",session_id="s",session_date="2024-01-01",turn_index=turn,raw_text=text,user_text=text,message_count=1,retrieval_text=text,schema_version="graphmem_v2")


def fact(node_id, obj, *, op="set", polarity="positive", modality="asserted", when="2024-01-01", item=None, predicate="likes", kind="state", embedding=None):
    f=AtomicFactNode(node_id=node_id,question_id="q",session_id="s",subject="user",subject_key="user",predicate=predicate,predicate_key=canonical_key(predicate),object=obj,object_key=canonical_key(obj),kind=kind,polarity=polarity,modality=modality,state_op=op,context_key="default",item_key=canonical_key(item or obj),event_time=when,observed_at=when,valid_from=when,source_leaf_ids=["q:s:leaf:0"],retrieval_text=f"user {predicate} {obj}",embedding=embedding)
    return f


def card(facts):
    return RoutingCardNode(node_id="q:s:routing",question_id="q",session_id="s",session_date="2024-01-01",topics=["tea"],canonical_entities=["user"],key_events=[],current_states=["likes tea"],time_range="2024",fact_ids=[f.node_id for f in facts],leaf_ids=["q:s:leaf:0"],retrieval_text="tea preference")


def test_bad_json_falls_back_with_lossless_source():
    c,facts,error=parse_session_extraction("not-json",question_id="q",session_id="s",session_date="2024-01-01",leaves=[leaf()])
    assert error and facts[0].source_leaf_ids==["q:s:leaf:0"]
    assert c.schema_version=="graphmem_v2"




def test_compact_extraction_schema_preserves_semantics():
    payload={"r":{"t":["food"],"e":["Alice"],"v":[],"s":["dislikes olives"],"d":"2024"},"f":[{"s":"Alice","p":"likes","o":"olives","k":"p","n":"-","m":"a","x":"S","c":"dinner","i":"olives","t":"2024-01-01","z":["q:s:leaf:0"],"sp":"Alice","r":"u","q":0.97}]}
    c,facts,error=parse_session_extraction(json.dumps(payload),question_id="q",session_id="s",session_date="2024-01-01",leaves=[leaf()])
    assert error is None and c.topics==["food"]
    assert facts[0].kind=="preference" and facts[0].polarity=="negative" and facts[0].state_op=="set" and facts[0].role=="user"




def test_array_extraction_schema_preserves_semantics():
    payload={"r":[["food"],["Alice"],[],["dislikes olives"],"2024"],"f":[["Alice","likes","olives","p","-","a","S","dinner","olives","2024-01-01",["q:s:leaf:0"],"u",0.97,None]]}
    c,facts,error=parse_session_extraction(json.dumps(payload),question_id="q",session_id="s",session_date="2024-01-01",leaves=[leaf()])
    assert error is None and c.canonical_entities==["Alice"]
    assert facts[0].kind=="preference" and facts[0].polarity=="negative" and facts[0].source_leaf_ids==["q:s:leaf:0"]


def test_truncated_array_extraction_salvages_complete_fact_rows():
    text="{\"r\":[[\"food\"],[\"Alice\"],[],[\"likes tea\"],\"2024\"],\"f\":[[\"Alice\",\"likes\",\"tea\",\"p\",\"+\",\"a\",\"S\",\"default\",\"tea\",\"2024-01-01\",[\"q:s:leaf:0\"],\"u\",0.97,null],["
    c,facts,error=parse_session_extraction(text,question_id="q",session_id="s",session_date="2024-01-01",leaves=[leaf()])
    assert error=="partial_json_salvaged"
    assert c.topics==["food"]
    assert len(facts)==1 and facts[0].object=="tea"


def test_source_ids_are_rejected_unless_they_exist():
    payload={"routing_card":{},"facts":[{"subject":"user","predicate":"likes","object":"tea","kind":"preference","source_leaf_ids":["made-up"]}]}
    _,facts,error=parse_session_extraction(json.dumps(payload),question_id="q",session_id="s",session_date="2024-01-01",leaves=[leaf()])
    assert error=="empty_facts_fallback" and facts[0].confidence==0.45


def test_entity_cleanup_and_alias_canonicalization():
    assert canonical_key("The BLUE-Dress!")=="blue-dress"
    assert clean_entities(["the", "Alice", "Alice", "x"])==["Alice"]
    f=fact("q:s:fact:0","NYC")
    text=json.dumps({"aliases":{"NYC":"New York City"},"predicate_aliases":{"likes":"prefers"},"relations":[]})
    _,aliases,error=apply_consolidation(text,[f])
    assert not error and aliases["nyc"]=="new york city" and f.object_key=="new york city" and f.predicate_key=="prefers"


def test_invalid_consolidation_edges_are_rejected():
    f=fact("q:s:fact:0","tea")
    text=json.dumps({"relations":[{"src":f.node_id,"dst":"missing","relation":"supports","confidence":1.0}]})
    edges,_,_=apply_consolidation(text,[f])
    assert edges==[]


def test_directed_supersedes_validity_and_latest_state():
    old=fact("q:s:fact:0","tea",when="2024-01-01")
    new=fact("q:s:fact:1","coffee",when="2024-02-01")
    chains,edges=build_state_chains([new,old])
    assert chains[0].current_fact_ids==[new.node_id]
    assert old.valid_to=="2024-02-01"
    assert any(e.src==new.node_id and e.dst==old.node_id and e.relation=="supersedes" and e.directed for e in edges)


def test_completed_event_not_overwritten_by_later_plan():
    done=fact("q:s:fact:0","marathon",op="complete",when="2024-01-01",kind="event")
    plan=fact("q:s:fact:1","marathon",op="set",modality="planned",when="2024-02-01",kind="event")
    chains,_=build_state_chains([done,plan])
    assert chains[0].current_fact_ids==[done.node_id]


def test_opposite_polarity_builds_contradiction():
    positive=fact("q:s:fact:0","tea",polarity="positive",when="2024-01-01")
    negative=fact("q:s:fact:1","tea",polarity="negative",when="2024-02-01")
    _,edges=build_state_chains([positive,negative])
    assert any(e.relation=="contradicts" and e.src==negative.node_id for e in edges)


def test_no_global_temporal_edges_and_mutual_knn_floor():
    l1=leaf("q:s:leaf:0",0); l2=leaf("q:s:leaf:1",2)
    f1=fact("q:s:fact:0","tea",embedding=[1.0,0.0])
    f2=fact("q:s:fact:1","coffee",predicate="drinks",embedding=[0.99,0.01])
    f3=fact("q:s:fact:2","running",predicate="does",embedding=[0.0,1.0])
    c=card([f1,f2,f3]); c.embedding=[1.0,0.0]
    chains,_=build_state_chains([f1,f2,f3])
    edges=build_graph_edges([l1,l2],[c],[f1,f2,f3],chains,semantic_k=1,semantic_floor=0.8)
    assert not any(e.relation=="temporal_neighbor" for e in edges)
    assert any(e.relation=="next_turn" and e.directed for e in edges)
    semantic=[e for e in edges if e.relation=="semantic_neighbor"]
    assert semantic and all(e.confidence>=0.8 and not e.directed for e in semantic)


def test_query_classifier_and_typed_allowlists():
    assert query_kind("What is my current job?")=="current/update"
    assert query_kind("How many books did I finish?")=="count/list"
    assert query_kind("When did I visit?")=="temporal"
    assert query_kind("What was my previous goal before I updated it?")=="temporal"
    assert query_kind("What food do I dislike?")=="preference"
    assert "before" not in _ALLOWED["current/update"]
    edges=[GraphEdge("a","b",1.0,"before",True),GraphEdge("a","c",1.0,"same_predicate",False)]
    expanded=_typed_expand(["a"],edges,_ALLOWED["current/update"],2,10)
    assert "b" not in expanded and "c" in expanded


def test_count_latest_temporal_and_preference_operators():
    tea=fact("q:s:fact:0","tea",op="add",kind="preference")
    tea_dup=fact("q:s:fact:1","tea",op="add",kind="preference",when="2024-01-02")
    coffee=fact("q:s:fact:2","coffee",op="add",kind="preference",polarity="negative",when="2024-01-03")
    chains,_=build_state_chains([tea,tea_dup,coffee])
    name,result,ids=_operator_result("count/list",[tea,tea_dup,coffee],chains)
    assert name=="distinct_completed_items" and result["count"]==1 and ids==[tea_dup.node_id]
    name,result,_=_operator_result("preference",[tea,coffee],chains)
    assert result["positive"]==["tea"] and result["negative"]==["coffee"]
    assert set(result["context"])=={"tea","coffee"}
    assert _operator_result("current/update",[tea,tea_dup],chains)[0]=="latest_valid_state"
    assert _operator_result("temporal",[coffee,tea],chains)[1][0]["fact_id"]==tea.node_id


def test_query_aware_count_dedupes_pending_pickups_and_returns():
    blazer=fact("q:s:fact:0","navy blue blazer",predicate="needs_dry_cleaning_pickup",item="navy blue blazer",modality="planned")
    sweater=fact("q:s:fact:1","not returned yet",predicate="item_return_status",item="green sweater",polarity="negative")
    boots_old=fact("q:s:fact:2","new pair of boots",predicate="needs_to_pickup",item="new boots",modality="planned")
    boots_new=fact("q:s:fact:3","exchanged boots from Zara",predicate="needs_to_pick_up",item="boots",modality="planned")
    noise=fact("q:s:fact:4","store cloth bags",predicate="tip",kind="assistant_fact")
    noise.role="assistant"
    name,result,ids=_operator_result("count/list",[blazer,sweater,boots_old,boots_new,noise],[],"How many items of clothing do I need to pick up or return from a store?")
    assert name=="distinct_completed_items" and result["count"]==3
    assert set(ids)=={blazer.node_id,sweater.node_id,boots_new.node_id}


def test_evidence_packer_respects_budget_and_keeps_minimum_typed_evidence():
    fs=[fact(f"q:s:fact:{i}","x"*400,item=str(i)) for i in range(20)]
    cs=[card(fs) for _ in range(8)]
    for i,c in enumerate(cs): c.node_id=f"q:s{i}:routing"; c.session_id=f"s{i}"
    ls=[leaf(f"q:s:leaf:{i}",i,"User: "+"y"*1000) for i in range(20)]
    ledger=build_evidence_ledger("count/list",fs,build_state_chains(fs)[0],ls)
    kept_c,kept_f,kept_l,context=_pack_context("count","count/list",cs,fs,ls,ledger,1800)
    from graphmem_demo.hierarchical_v2 import provider_token_estimate
    assert provider_token_estimate(context)<=1800 and kept_c and kept_f and kept_l


def test_phase_token_accounting_and_judge_exclusion():
    def record(stage,miss,hit,out,excluded=False):
        return DeepSeekCallRecord(question_id="q",variant="v",stage=stage,call_id=stage,model="m",thinking_mode="none",prompt_tokens=miss+hit,completion_tokens=out,total_tokens=miss+hit+out,prompt_cache_miss_tokens=miss,prompt_cache_hit_tokens=hit,excluded_from_budget=excluded)
    stats=build_question_stats(question_id="q",variant="v",session_count=1,leaf_count=1,summary_count=1,edge_count=0,records=[record("build_fact_extraction",10,5,2),record("answer_qa",20,3,4),record("judge",100,0,50,True)],build_latency_sec=0,retrieval_latency_sec=0,answer_latency_sec=0,answer_session_hit=False)
    assert (stats.build_cache_miss_input_tokens,stats.build_cache_hit_input_tokens,stats.build_output_tokens,stats.build_total_tokens)==(10,5,2,17)
    assert (stats.answer_cache_miss_input_tokens,stats.answer_cache_hit_input_tokens,stats.answer_output_tokens,stats.answer_total_tokens)==(20,3,4,27)
    assert stats.total_deepseek_tokens==44 and stats.token_accounting_valid


def test_generic_predicates_do_not_create_false_state_updates():
    first=fact("q:s:fact:0","one",predicate="stated")
    second=fact("q:s:fact:1","two",predicate="recommended items",when="2024-02-01")
    _,edges=build_state_chains([first,second])
    assert not any(edge.relation=="supersedes" for edge in edges)


def test_consolidation_salvages_complete_edges_and_rejects_unrelated_time_edges():
    first=fact("q:s:fact:0","tea",predicate="likes")
    second=fact("q:s:fact:1","Paris",predicate="visited")
    edges,_,error=apply_consolidation("{\"a\":{},\"p\":{},\"e\":[[0,1,\"supports\",0.9],[",[first,second])
    assert error=="partial_json_salvaged" and len(edges)==1 and edges[0].relation=="supports"
    edges,_,_=apply_consolidation(json.dumps({"e":[[0,1,"before",0.9]]}),[first,second])
    assert edges==[]


def test_routing_card_uses_provider_token_cap():
    payload={"r":[["topic"*50]*8,["entity"*40]*12,[],[],"2024"],"f":[["user","likes","tea","p","+","a","S","default","tea","2024-01-01",["q:s:leaf:0"],"u",0.97,None]]}
    route,_,_=parse_session_extraction(json.dumps(payload),question_id="q",session_id="s",session_date="2024-01-01",leaves=[leaf()])
    assert provider_token_estimate(route.retrieval_text)<=180


def test_retrieval_expands_adjacent_source_turns():
    leaves=[leaf(f"q:s:leaf:{index}",index,f"User: turn {index}") for index in range(5)]
    target=fact("q:s:fact:0","tennis racket",embedding=[1.0,0.0]);target.source_leaf_ids=["q:s:leaf:2"]
    route=card([target]);route.leaf_ids=[item.node_id for item in leaves];route.embedding=[1.0,0.0]
    case=__import__("graphmem_demo.models",fromlist=["QuestionCase"]).QuestionCase(question_id="q",question_type="single-session-user",question="Where is the tennis racket?",answer=None,question_date=None,haystack_sessions=[],haystack_session_ids=[],haystack_dates=[],answer_session_ids=[])
    result=retrieve(case=case,variant="hierarchical_state_graph_v2",leaves=leaves,cards=[route],facts=[target],chains=build_state_chains([target])[0],edges=[],query_vector=[1.0,0.0],card_k=1,fact_k=1,leaf_k=3,token_budget=4000)
    assert set(result.evidence_leaf_ids)=={"q:s:leaf:1","q:s:leaf:2","q:s:leaf:3"}


def test_retrieval_has_independent_direct_leaf_channel():
    wrong=leaf("q:s:leaf:0",text="User: discussed gardening tools")
    wrong.embedding=[1.0,0.0]
    target=leaf("q:t:leaf:0",text="User: the passport is in the blue drawer")
    target.session_id="t";target.embedding=[0.0,1.0]
    memory_fact=fact("q:s:fact:0","gardening tools",embedding=[1.0,0.0])
    route=card([memory_fact]);route.embedding=[1.0,0.0]
    case=__import__("graphmem_demo.models",fromlist=["QuestionCase"]).QuestionCase(
        question_id="q",question_type="single-session-user",
        question="Where is the passport?",answer=None,question_date=None,
        haystack_sessions=[],haystack_session_ids=[],haystack_dates=[],
        answer_session_ids=[],
    )
    result=retrieve(
        case=case,variant="hierarchical_state_graph_v2",
        leaves=[wrong,target],cards=[route],facts=[memory_fact],
        chains=build_state_chains([memory_fact])[0],edges=[],
        query_vector=[0.0,1.0],card_k=1,fact_k=1,leaf_k=2,
        token_budget=4000,
    )
    assert target.node_id in result.evidence_leaf_ids
    assert result.retrieval_trace["direct_leaf_ids"][0]==target.node_id


def test_count_operator_does_not_fallback_to_unrelated_pool():
    unrelated=fact(
        "q:s:fact:0","Dr. Chen",predicate="visited",
        kind="event",embedding=[1.0,0.0],
    )
    result=_operator_result(
        "count/list",[unrelated],[],
        "How many baby supplies did I buy?",[leaf()],
    )
    assert result is None


def test_generic_arithmetic_aggregates_market_income_and_inventory():
    market_texts=[
        "User: I sold herbs at the market, earning a total of $120.",
        "User: I sold jam at the market, earning $225.",
        "User: I sold 20 potted plants at the market for $7.5 each.",
    ]
    market_leaves=[];market_facts=[]
    for index,text in enumerate(market_texts):
        src=leaf(f"q:m{index}:leaf:0",text=text);src.session_id=f"m{index}"
        row=fact(f"q:m{index}:fact:0","market sale",predicate="earned",kind="quantity")
        row.session_id=src.session_id;row.source_leaf_ids=[src.node_id]
        market_leaves.append(src);market_facts.append(row)
    name,result,_=_arithmetic_result(
        "What is the total amount of money I earned from selling at markets?",
        market_facts,market_leaves,
    )
    assert name=="generic_calculation" and result["formatted_result"]=="$495.0"

    inventory_texts=[
        "User: I have 57 rare records.",
        "User: I have 25 rare coins.",
        "User: I have 12 rare figurines.",
        "User: I have a rare book collection of 5 books.",
    ]
    inventory_leaves=[];inventory_facts=[]
    for index,text in enumerate(inventory_texts):
        src=leaf(f"q:r{index}:leaf:0",text=text);src.session_id=f"r{index}"
        row=fact(f"q:r{index}:fact:0","rare items",predicate="has",kind="quantity")
        row.session_id=src.session_id;row.source_leaf_ids=[src.node_id]
        inventory_leaves.append(src);inventory_facts.append(row)
    _,result,_=_arithmetic_result(
        "How many rare items do I have in total?",inventory_facts,inventory_leaves
    )
    assert result["formatted_result"]=="99"


def test_generic_arithmetic_resolves_points_and_adjusted_wake_time():
    points_leaf=leaf(
        text="User: I have 200 points at Sephora and need 300 points to redeem a product."
    )
    points_fact=fact("q:s:fact:0","200 points",predicate="has",kind="quantity")
    _,result,_=_arithmetic_result(
        "How many points do I need to earn to redeem a product at Sephora?",
        [points_fact],[points_leaf],
    )
    assert result["formatted_result"]=="100 points"

    base=leaf("q:s1:leaf:0",text="User: I started waking up at 7:00 AM.")
    base.session_id="s1"
    adjustment=leaf(
        "q:s2:leaf:0",
        text="User: On Tuesdays and Thursdays I wake up 15 minutes earlier.",
    )
    adjustment.session_id="s2"
    base_fact=fact("q:s1:fact:0","7:00 AM",predicate="wake_time")
    base_fact.session_id="s1";base_fact.source_leaf_ids=[base.node_id]
    adjustment_fact=fact(
        "q:s2:fact:0","15 minutes earlier",predicate="wake_time_adjustment"
    )
    adjustment_fact.session_id="s2";adjustment_fact.source_leaf_ids=[adjustment.node_id]
    _,result,_=_arithmetic_result(
        "What time do I wake up on Tuesdays and Thursdays?",
        [base_fact,adjustment_fact],[base,adjustment],
    )
    assert result["formatted_result"]=="6:45 AM"


def test_same_day_current_state_uses_observation_order_and_stays_compact():
    old=fact("q:s:fact:0","1250",predicate="follower_count",kind="quantity",embedding=[1.0,0.0])
    new=fact("q:s:fact:1","1300",predicate="follower_count",kind="quantity",embedding=[0.0,1.0])
    old.observation_order=4;new.observation_order=9
    name,result,ids=_operator_result("current/update",[old,new],[],"How many followers do I have now?",[leaf()])
    assert name=="latest_valid_state" and ids==[new.node_id]
    assert result[0]["object"]=="1300" and "embedding" not in result[0]


def test_count_operator_uses_source_turn_for_exact_action():
    source=leaf(text="User: I just re-watched Avengers: Endgame, a Marvel movie.")
    avengers=fact("q:s:fact:0","Avengers: Endgame",predicate="watched",kind="event")
    name,result,ids=_operator_result("count/list",[avengers],[],"How many Marvel movies did I re-watch?",[source])
    assert name=="distinct_completed_items" and result["count"]==1
    assert ids==[avengers.node_id]


def test_cashback_and_exact_entity_ledger_operators():
    purchase_leaf=leaf("q:purchase:leaf:0",text="User: I spent $75 at SaveMart last Thursday.")
    rate_leaf=leaf("q:rate:leaf:0",text="User: I have a SaveMart membership and earn 1% cashback on all purchases.")
    amount=fact("q:purchase:fact:0","75 dollars",predicate="amount",kind="quantity")
    amount.session_id="purchase";amount.source_leaf_ids=[purchase_leaf.node_id]
    rate=fact("q:rate:fact:0","1% on all purchases",predicate="earns_cashback",kind="quantity")
    rate.session_id="rate";rate.source_leaf_ids=[rate_leaf.node_id]
    ledger=build_evidence_ledger("temporal",[amount,rate],[],[purchase_leaf,rate_leaf],"How much cashback did I earn at SaveMart last Thursday?")
    cashback=next(row for row in ledger if row.get("operator")=="cashback_calculation")
    assert cashback["result"]["formatted_cashback"]=="$0.75"

    tennis_leaf=leaf(text="User: I play tennis weekly at the local park.")
    tennis=fact("q:s:fact:1","with friends",predicate="plays_tennis",kind="event")
    entity_ledger=build_evidence_ledger("fact",[tennis],[],[tennis_leaf],"How often do I play table tennis at the park?")
    check=next(row for row in entity_ledger if row.get("operator")=="exact_entity_check")
    assert check["result"]=={"requested_entity":"table tennis","exact_match":False,"partial_entity_only":True,"partial_entity":"tennis"}


def test_count_precedes_time_window_and_advice_is_preference():
    assert query_kind("How many health-related devices do I use in a day?")=="count/list"
    assert query_kind("How many art events did I attend last month?")=="count/list"
    assert query_kind("Any tips for my phone battery life?")=="preference"
    assert query_kind("Should I buy a NAS now or wait? What do you think?")=="preference"


def test_category_device_count_and_observation_time_label():
    names=[("uses_device","Fitbit Versa 3"),("uses_device","Accu-Chek Aviva Nano"),("uses_treatment","nebulizer machine"),("relies_on","hearing aids")]
    leaves=[];facts=[]
    for index,(predicate,obj) in enumerate(names):
        item_leaf=leaf(f"q:s{index}:leaf:0",text=f"User: I use {obj} every day for my health.")
        item=fact(f"q:s{index}:fact:0",obj,predicate=predicate,kind="state")
        item.session_id=f"s{index}";item.source_leaf_ids=[item_leaf.node_id]
        leaves.append(item_leaf);facts.append(item)
    name,result,_=_operator_result("count/list",facts,[],"How many health-related devices do I use in a day?",leaves)
    assert name=="distinct_completed_items" and result["count"]==4
    _,temporal,_=_operator_result("temporal",[facts[0]],[],"When did I use it?",leaves)
    assert temporal[0]["time_basis"]=="event_time"
    facts[0].event_time=None
    _,temporal,_=_operator_result("temporal",[facts[0]],[],"When did I use it?",leaves)
    assert temporal[0]["time_basis"]=="observation_only" and temporal[0]["event_time"] is None


def test_relative_date_scope_and_elapsed_days_operator():
    start_leaf=leaf("q:start:leaf:0",text='User: I started "The Nightingale" today.')
    start_leaf.session_id="start";start_leaf.session_date="2023/01/10 (Tue) 10:34"
    end_leaf=leaf("q:end:leaf:0",text='User: I finished "The Nightingale" today.')
    end_leaf.session_id="end";end_leaf.session_date="2023/01/31 (Tue) 23:59"
    start=fact("q:start:fact:0","The Nightingale",predicate="started_reading",kind="event")
    start.session_id="start";start.source_leaf_ids=[start_leaf.node_id]
    end=fact("q:end:fact:0","The Nightingale",predicate="finished_reading",kind="event")
    end.session_id="end";end.source_leaf_ids=[end_leaf.node_id]
    result=_temporal_calculation_result("How many days did it take me to finish The Nightingale?",[start,end],[start_leaf,end_leaf],"2023/05/01")
    assert result and result[1]["formatted_result"]=="21 days"
    target,_=_question_date_scope("What appliance did I buy 10 days ago?","2023/03/25 (Sat) 18:26")
    assert target and target.isoformat()=="2023-03-15"


def test_target_date_extracts_music_companion_relative_event_and_cooked_item():
    music_leaf=leaf("q:music:leaf:0",text="User: I saw Queen live with my parents.")
    music_leaf.session_id="music";music_leaf.session_date="2023/04/15 (Sat) 03:11"
    music=fact("q:music:fact:0","Queen concert with my parents",predicate="attended music event",kind="event")
    music.session_id="music";music.source_leaf_ids=[music_leaf.node_id]
    result=_target_date_answer_result("Who did I go with to the music event last Saturday?",[music],[music_leaf],"2023/04/22 (Sat) 08:01")
    assert result and result[1]["value"].casefold()=="my parents"

    wedding_leaf=leaf("q:wedding:leaf:0",text="User: I walked down the aisle as a bridesmaid at my cousin’s wedding.")
    wedding_leaf.session_id="wedding";wedding_leaf.session_date="2023/06/15 (Thu) 10:02"
    wedding=fact("q:wedding:fact:0","my cousin wedding",predicate="participated in relative life event",kind="event")
    wedding.session_id="wedding";wedding.source_leaf_ids=[wedding_leaf.node_id]
    result=_target_date_answer_result("What life event of a relative did I participate in a week ago?",[wedding],[wedding_leaf],"2023/06/22 (Thu) 18:33")
    assert result and "cousin" in result[1]["value"].casefold() and "wedding" in result[1]["value"].casefold()

    cake_leaf=leaf("q:cake:leaf:0",text="User: I baked a chocolate cake for my friend’s birthday party.")
    cake_leaf.session_id="cake";cake_leaf.session_date="2022/04/10 (Sun) 23:24"
    cake=fact("q:cake:fact:0","chocolate cake",predicate="baked for friend",kind="event")
    cake.session_id="cake";cake.source_leaf_ids=[cake_leaf.node_id]
    result=_target_date_answer_result("I mentioned cooking something for my friend a couple of days ago. What was it?",[cake],[cake_leaf],"2022/04/12 (Tue) 22:57")
    assert result and result[1]["value"].casefold()=="chocolate cake"


def test_direct_daily_duration_operator():
    source=leaf(text="User: I have been practicing guitar for 30 minutes daily.")
    duration=fact("q:s:fact:0","30 minutes daily",predicate="practice_duration",kind="quantity")
    result=_temporal_calculation_result("How much time do I dedicate to practicing guitar every day?",[duration],[source],"2023/05/30")
    assert result and result[1]["formatted_result"]=="30 minutes"


def test_page_sum_survives_as_generic_calculation():
    first_leaf=leaf("q:a:leaf:0",text="User: I finished a 416-page novel.")
    second_leaf=leaf("q:b:leaf:0",text="User: I finished The Nightingale, which had 440 pages.")
    first=fact("q:a:fact:0","416-page novel",predicate="finished_novel",kind="event");first.session_id="a";first.source_leaf_ids=[first_leaf.node_id]
    finished=fact("q:b:fact:0","The Nightingale",predicate="finished_reading",kind="event");finished.session_id="b";finished.source_leaf_ids=[second_leaf.node_id]
    pages=fact("q:b:fact:1","440 pages",predicate="book_page_count",kind="quantity");pages.session_id="b";pages.source_leaf_ids=[second_leaf.node_id]
    ledger=build_evidence_ledger("count/list",[first,finished,pages],[],[first_leaf,second_leaf],"What was the page count of the two novels I finished in January and March?")
    calculation=next(row for row in ledger if row.get("operator")=="generic_calculation")
    assert calculation["result"]["result_pages"]=="856"
    assert calculation["result"]["formatted_result"]=="856 pages"


def test_mandatory_hint_uses_packed_operator_value():
    context='[EVIDENCE LEDGER]\n[{"operator":"explicit_event_time","result":{"value":"February 1st"}}]\n\n[ROUTING CARD x]'
    assert _mandatory_answer_hint(context) is None

def test_collection_increment_and_amount_sum_are_deduplicated():
    base_leaf=leaf("q:base:leaf:0",text="User: I have a total of 37 pre-1920 American coins.")
    add_leaf=leaf("q:add:leaf:0",text="User: I added a 1915-S Barber quarter to my pre-1920 collection.")
    base=fact("q:base:fact:0","37 pre-1920 American coins",predicate="has",kind="quantity");base.session_id="base";base.source_leaf_ids=[base_leaf.node_id]
    added=fact("q:add:fact:0","1915-S Barber quarter",predicate="added_to_collection",kind="event");added.session_id="add";added.source_leaf_ids=[add_leaf.node_id]
    result=_category_count_result("How many pre-1920 American coins do I have?",[base,added],[base_leaf,add_leaf])
    assert result and result[0]["count"]==38


def test_arrival_time_accepts_word_number_duration():
    left_leaf=leaf("q:left:leaf:0",text="User: I left home at 7 AM for the clinic.")
    trip_leaf=leaf("q:trip:leaf:0",text="User: It took me two hours to get to the clinic.")
    left=fact("q:left:fact:0","7 AM",predicate="left_home",kind="event");left.session_id="left";left.source_leaf_ids=[left_leaf.node_id]
    trip=fact("q:trip:fact:0","two hours",predicate="commute_duration",kind="quantity");trip.session_id="trip";trip.source_leaf_ids=[trip_leaf.node_id]
    result=_temporal_calculation_result("What time did I reach the clinic?",[left,trip],[left_leaf,trip_leaf],"2023/05/30")
    assert result and result[1]["formatted_result"]=="9:00 AM"


def test_event_sequence_keeps_first_occurrence_of_recalled_event():
    rows=[
        ("a","2023-03-18","User: I just got back from a Billie Eilish concert at the Wells Fargo Center in Philly."),
        ("b","2023-04-01","User: I just got back from a music festival in Brooklyn."),
        ("c","2023-04-15","User: I saw Queen live with Adam Lambert; I also recently attended a music festival in Brooklyn."),
    ]
    leaves=[];facts=[]
    for sid,when,text in rows:
        item_leaf=leaf(f"q:{sid}:leaf:0",text=text);item_leaf.session_id=sid;item_leaf.session_date=when;leaves.append(item_leaf)
        item=fact(f"q:{sid}:fact:0",text,predicate="attended",kind="event");item.session_id=sid;item.source_leaf_ids=[item_leaf.node_id];facts.append(item)
    result=_event_sequence_result("What is the order of the concerts and musical events I attended, earliest first?",facts,leaves)
    assert result and [row["date"] for row in result[1]]==["2023-03-18","2023-04-01","2023-04-15"]


def test_packer_preserves_complete_deterministic_count():
    sources=[];facts=[]
    for index in range(4):
        item_leaf=leaf(f"q:s{index}:leaf:0",text=f"User: I attended charity event {index} in May.");item_leaf.session_id=f"s{index}";sources.append(item_leaf)
        item=fact(f"q:s{index}:fact:0",f"event {index}",predicate="attended",kind="event");item.session_id=f"s{index}";item.source_leaf_ids=[item_leaf.node_id];facts.append(item)
    ledger=[{"operator":"distinct_completed_items","result":{"count":4,"items":[f"event {i}" for i in range(4)]},"source_fact_ids":[f.node_id for f in facts]}]
    _,_,_,context=_pack_context("How many charity events?","count/list",[],facts,sources,ledger,8200)
    assert _mandatory_answer_hint(context) is None


def test_pinned_mem0_prompt_source_is_byte_exact():
    path=Path(__file__).parents[1]/"src/graphmem_demo/mem0_longmemeval_prompts.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest()=="ba8cf60d26f1390ecbef0f07b3e950556fe3bc5a37ba4b5343f28217f18c144f"


def test_query_planner_handles_current_collections_and_assistant_recall():
    assert query_kind("How many musical instruments do I currently own?")=="count/list"
    assert query_kind("Can you remind me how many mummies were in our previous chat?")=="fact"


def test_sibling_count_uses_max_per_relationship():
    sister_leaf=leaf("q:a:leaf:0",text="User: I have 3 sisters.");sister_leaf.session_id="a"
    brother_leaf=leaf("q:b:leaf:0",text="User: I have a brother.");brother_leaf.session_id="b"
    sister=fact("q:a:fact:0","3",predicate="family_sisters_count",kind="quantity");sister.session_id="a";sister.source_leaf_ids=[sister_leaf.node_id]
    brother=fact("q:b:fact:0","brother",predicate="has",kind="state");brother.session_id="b";brother.source_leaf_ids=[brother_leaf.node_id]
    duplicate=fact("q:c:fact:0","brother",predicate="attended_with",kind="event");duplicate.session_id="c"
    duplicate_leaf=leaf("q:c:leaf:0",text="User: I attended with my brother.");duplicate_leaf.session_id="c";duplicate.source_leaf_ids=[duplicate_leaf.node_id]
    result=_category_count_result("What is the total number of siblings I have?",[sister,brother,duplicate],[sister_leaf,brother_leaf,duplicate_leaf])
    assert result and result[0]["count"]==4


def test_exact_capacity_entity_rejects_other_tank_size():
    source=leaf(text="User: I have a 20-gallon tank with 10 neon tetras.")
    tank=fact("q:s:fact:0","20-gallon",predicate="tank_size",kind="quantity")
    result=_exact_entity_result("How many fish are in my 30-gallon tank?",[tank],[source])
    assert result and result[1]["exact_match"] is False
    assert result[1]["partial_entity"]=="20-gallon tank"


def test_lossless_age_augmentation_from_user_leaf():
    source=leaf(text="User: Do you think 32 is considered young or old?")
    payload={"r":{"t":["travel"]},"f":[["user","likes","hostels","p","+","a","S","travel","hostels",None,[source.node_id],"u",0.9,None]]}
    _,facts,_=parse_session_extraction(json.dumps(payload),question_id="q",session_id="s",session_date="2024-01-01",leaves=[source])
    age=next(item for item in facts if item.predicate_key=="age")
    assert age.object=="32 years old" and age.source_leaf_ids==[source.node_id]


def test_assistant_recall_extracts_structured_count_and_handle():
    mummy_leaf=leaf(text='User: make a one shot.\nAssistant: * Mummies (4): Armor Class 11')
    mummy=fact("q:s:fact:0","Mummies (4)",predicate="provided_enemy",kind="assistant_fact");mummy.role="assistant"
    count=_assistant_recall_result("Can you remind me how many mummies were in our previous chat?",[mummy],[mummy_leaf])
    assert count and count[1]["value"]=="4"
    ring_leaf=leaf(text='User: list designers.\nAssistant: 1. Jessica Poole (@jessica\\_poole\\_jewellery): UK-based and works with unusual gemstones.\n2. Rachel Boston (@rachel): London-based and uses diamonds.')
    ring=fact("q:s:fact:1","@jessica_poole_jewellery",predicate="designer_instagram",kind="assistant_fact");ring.role="assistant"
    handle=_assistant_recall_result("In our previous conversation, remind me of the Instagram handle of the UK-based designer using unusual gemstones.",[ring],[ring_leaf])
    assert handle and handle[1]["value"]=="@jessica_poole_jewellery"


def test_target_date_from_whom_rejects_article_and_accepts_relationship():
    source=leaf(text="User: I received a crystal chandelier from my aunt today.")
    item=fact("q:s:fact:0","crystal chandelier from aunt",predicate="acquired_item",kind="event")
    result=_target_date_answer_result("I received jewelry last Monday from whom?",[item],[source],"2024-01-08")
    assert result and result[1]["value"]=="my aunt"


def test_sculpting_weeks_ignore_unrelated_earlier_rows():
    unrelated_leaf=leaf("q:x:leaf:0",text="User: I need to update my medical license.");unrelated_leaf.session_id="x";unrelated_leaf.session_date="2023-01-14"
    start_leaf=leaf("q:a:leaf:0",text="User: I started taking sculpting classes at a local studio.");start_leaf.session_id="a";start_leaf.session_date="2023-02-11"
    tools_leaf=leaf("q:b:leaf:0",text="User: I entered a sculpture competition and invested in sculpting tools.");tools_leaf.session_id="b";tools_leaf.session_date="2023-03-04"
    rows=[]
    for sid,pred,obj,src in [("x","needs","license",unrelated_leaf),("a","started_sculpting_classes","studio",start_leaf),("b","has_sculpting_tools","tools",tools_leaf)]:
        item=fact(f"q:{sid}:fact:0",obj,predicate=pred,kind="event");item.session_id=sid;item.source_leaf_ids=[src.node_id];rows.append(item)
    result=_temporal_calculation_result("How many weeks have I been taking sculpting classes when I invested in my own set of sculpting tools?",rows,[unrelated_leaf,start_leaf,tools_leaf],"2023-03-04")
    assert result and result[1]["formatted_result"]=="3 weeks"


def test_album_count_accepts_buying_and_keeps_purchase_representative_per_session():
    rows = [
        ("a", "User: I downloaded the album Happier Than Ever.", "downloaded_album", "Happier Than Ever"),
        ("b", "User: I purchased the EP Midnight Sky.", "discovered", "The Whiskey Wanderers"),
        ("c", "User: I ended up buying their EP Midnight Sky.", "purchased_item", "EP Midnight Sky"),
    ]
    leaves=[];facts=[]
    for sid,text,predicate,obj in rows:
        source=leaf(f"q:{sid}:leaf:0",text=text);source.session_id=sid;leaves.append(source)
        item=fact(f"q:{sid}:fact:0",obj,predicate=predicate,kind="event",op="complete")
        item.session_id=sid;item.source_leaf_ids=[source.node_id];facts.append(item)
    result=_category_count_result(
        "How many music albums or EPs have I purchased or downloaded?",facts,leaves,
    )
    assert result and result[0]["count"]==3
    assert "EP Midnight Sky" in result[0]["items"]


def test_nas_preference_context_excludes_other_domains_and_triggers_focus():
    nas_leaf=leaf("q:nas:leaf:0",text="User: I rely on an external hard drive and need more storage; I want a NAS as central backup.")
    nas_leaf.session_id="nas"
    clay_leaf=leaf("q:clay:leaf:0",text="User: I want to try stoneware clay.")
    clay_leaf.session_id="clay"
    nas=fact("q:nas:fact:0","NAS as central backup",predicate="wants",kind="preference");nas.session_id="nas";nas.source_leaf_ids=[nas_leaf.node_id]
    external=fact("q:nas:fact:1","external hard drive",predicate="currently uses",kind="state");external.session_id="nas";external.source_leaf_ids=[nas_leaf.node_id]
    clay=fact("q:clay:fact:0","stoneware clay",predicate="wants to try",kind="preference");clay.session_id="clay";clay.source_leaf_ids=[clay_leaf.node_id]
    question="Should I buy a NAS now or wait?"
    selected=_preference_context_facts(question,[nas,external,clay],[nas_leaf,clay_leaf])
    assert {item.node_id for item in selected}=={nas.node_id,external.node_id}
    context=[item.object for item in selected]+[nas_leaf.user_text]
    focus=_preference_focus_instruction(question,context)
    assert focus and "storage-capacity" in focus and "external hard drives" in focus


def test_rewatch_count_does_not_count_non_event_state_from_same_turn():
    source=leaf(text="User: I re-watched Spider-Man and want new movies to watch.")
    watched=fact("q:s:fact:0","Spider-Man",predicate="re_watched",kind="event",op="complete")
    interest=fact("q:s:fact:1","new movies to watch",predicate="interested_in",kind="state")
    name,result,_=_operator_result("count/list",[watched,interest],[],"How many Marvel movies did I re-watch?",[source])
    assert name=="distinct_completed_items" and result["count"]==1


def test_current_state_locks_to_question_entity_before_predicate_grouping():
    source=leaf(text="User: I use a water bottle and Trader Joe's lavender shampoo.")
    bottle=fact("q:s:fact:0","refillable water bottle",predicate="uses",kind="state")
    shampoo=fact("q:s:fact:1","Trader Joe's lavender shampoo",predicate="uses",kind="state")
    name,result,_=_operator_result("current/update",[bottle,shampoo],[],"What brand of shampoo do I currently use?",[source])
    assert name=="latest_valid_state" and result[0]["object"]=="Trader Joe's lavender shampoo"


def test_most_recent_streaming_service_uses_relative_age():
    apple_leaf=leaf("q:a:leaf:0",text="User: I have been using Apple TV+ for a few months now.");apple_leaf.session_id="a"
    disney_leaf=leaf("q:d:leaf:0",text="User: I started a free trial of Disney+ last month.");disney_leaf.session_id="d"
    apple=fact("q:a:fact:0","Apple TV+",predicate="subscribed to",kind="event");apple.session_id="a";apple.source_leaf_ids=[apple_leaf.node_id]
    disney=fact("q:d:fact:0","Disney+",predicate="had free trial of",kind="event");disney.session_id="d";disney.source_leaf_ids=[disney_leaf.node_id]
    question="Which streaming service did I start using most recently?"
    assert query_kind(question)=="current/update"
    _,result,_=_operator_result("current/update",[apple,disney],[],question,[apple_leaf,disney_leaf])
    assert result[0]["object"]=="Disney+"


def test_event_comparison_binds_relative_duration_to_each_entity():
    case_leaf=leaf("q:c:leaf:0",text="User: I got my new phone case about a month ago.");case_leaf.session_id="c"
    charger_leaf=leaf("q:h:leaf:0",text="User: I lost my phone charger about two weeks ago.");charger_leaf.session_id="h"
    case=fact("q:c:fact:0","new phone case",predicate="received",kind="event");case.session_id="c";case.source_leaf_ids=[case_leaf.node_id]
    charger=fact("q:h:fact:0","phone charger",predicate="lost",kind="event");charger.session_id="h";charger.source_leaf_ids=[charger_leaf.node_id]
    result=_event_comparison_result("Which event happened first, the narrator losing their phone charger or the narrator receiving their new phone case?",[case,charger],[case_leaf,charger_leaf])
    assert result and "receiving their new phone case" in result[1]["value"]


def test_sports_event_sequence_keeps_all_january_events():
    rows=[
        ("a","2023-01-05","User: I went to an NBA game at the Staples Center today."),
        ("b","2023-01-15","User: I watched the College Football National Championship game yesterday."),
        ("c","2023-01-22","User: I watched the NFL playoffs last weekend."),
    ]
    leaves=[];facts=[]
    for sid,when,text in rows:
        source=leaf(f"q:{sid}:leaf:0",text=text);source.session_id=sid;source.session_date=when;leaves.append(source)
        item=fact(f"q:{sid}:fact:0",text,predicate="watched",kind="event",op="complete");item.session_id=sid;item.source_leaf_ids=[source.node_id];facts.append(item)
    result=_event_sequence_result("What is the order of the sports events I watched in January?",facts,leaves)
    assert result and [row["event"] for row in result[1]]==[
        "NBA game at the Staples Center","College Football National Championship game","NFL playoffs",
    ]


def test_current_follower_count_uses_latest_source_timestamp_and_normalizes_approximation():
    old_leaf=leaf("q:a:leaf:0",text="User: I have 1250 Instagram followers.");old_leaf.session_id="a";old_leaf.session_date="2023/05/25 05:26"
    new_leaf=leaf("q:b:leaf:0",text="User: I am now near 1300 Instagram followers.");new_leaf.session_id="b";new_leaf.session_date="2023/05/25 09:28"
    old=fact("q:a:fact:0","1250",predicate="has_followers",kind="quantity");old.session_id="a";old.source_leaf_ids=[old_leaf.node_id]
    new=fact("q:b:fact:0","near 1300",predicate="follower_count",kind="quantity");new.session_id="b";new.source_leaf_ids=[new_leaf.node_id]
    _,result,_=_operator_result("current/update",[old,new],[],"How many followers do I have on Instagram now?",[old_leaf,new_leaf])
    assert result[0]["object"]=="1300"


def test_event_comparison_understands_past_duration_without_ago():
    festival_leaf=leaf("q:f:leaf:0",text="User: I attended a cultural festival yesterday.");festival_leaf.session_id="f"
    spanish_leaf=leaf("q:s:leaf:0",text="User: I have been taking Spanish classes for the past three months.");spanish_leaf.session_id="s"
    festival=fact("q:f:fact:0","cultural festival",predicate="attended",kind="event");festival.session_id="f";festival.source_leaf_ids=[festival_leaf.node_id]
    spanish=fact("q:s:fact:0","Spanish classes",predicate="started",kind="event");spanish.session_id="s";spanish.source_leaf_ids=[spanish_leaf.node_id]
    result=_event_comparison_result("Which event happened first, my attendance at a cultural festival or the start of my Spanish classes?",[festival,spanish],[festival_leaf,spanish_leaf])
    assert result and "spanish classes" in result[1]["value"]


def test_gift_total_uses_source_amount_when_l1_cost_fact_is_missing():
    necklace_leaf=leaf("q:a:leaf:0",text="User: I bought my sister a necklace for $200.");necklace_leaf.session_id="a"
    card_leaf=leaf("q:b:leaf:0",text="User: Last time I got my sister a $100 gift card.");card_leaf.session_id="b"
    necklace=fact("q:a:fact:0","$200",predicate="gift_cost",kind="quantity");necklace.session_id="a";necklace.source_leaf_ids=[necklace_leaf.node_id]
    card=fact("q:b:fact:0","spa gift card",predicate="bought",kind="event",op="complete");card.session_id="b";card.source_leaf_ids=[card_leaf.node_id]
    result=_category_count_result("How much did I spend on gifts for my sister?",[necklace,card],[necklace_leaf,card_leaf])
    assert result and result[0]["formatted_total"]=="$300"


def test_prior_professional_experience_subtracts_current_role_tenure():
    total=fact("q:a:fact:0","9 years",predicate="total_professional_experience",kind="quantity")
    tenure=fact("q:b:fact:0","4 years and 3 months",predicate="current_role_tenure",kind="quantity")
    result=_operator_result("temporal",[total,tenure],[],"How long have I been working before I started my current job at NovaTech?",[leaf()])
    # Arithmetic operators are appended outside the base temporal operator.
    ledger=build_evidence_ledger("temporal",[total,tenure],[],[leaf()],"How long have I been working before I started my current job at NovaTech?")
    calc=next(row for row in ledger if row.get("operator")=="generic_calculation")
    assert calc["result"]["formatted_result"]=="4 years and 9 months"


def test_prior_experience_rejects_conflicting_requested_employer():
    total_leaf=leaf("q:a:leaf:0",text="User: I have worked professionally for 9 years.")
    tenure_leaf=leaf("q:b:leaf:0",text="User: I have been working at NovaTech for 4 years and 3 months.")
    total=fact("q:a:fact:0","9 years",predicate="total_professional_experience",kind="quantity");total.session_id="a";total.source_leaf_ids=[total_leaf.node_id]
    tenure=fact("q:b:fact:0","4 years and 3 months",predicate="current_role_tenure",kind="quantity");tenure.session_id="b";tenure.source_leaf_ids=[tenure_leaf.node_id]
    ledger=build_evidence_ledger("temporal",[total,tenure],[],[total_leaf,tenure_leaf],"How long did I work before my current job at Google?")
    assert not any(row.get("operator")=="generic_calculation" for row in ledger)


def test_feed_weight_sum_uses_distinct_purchases():
    first_leaf=leaf("q:a:leaf:0",text="User: I purchased new layer feed and got a 50-pound batch.");first_leaf.session_id="a"
    second_leaf=leaf("q:b:leaf:0",text="User: I also bought 20 pounds of organic scratch grains.");second_leaf.session_id="b"
    first=fact("q:a:fact:0","50 pounds",predicate="feed weight",kind="quantity");first.session_id="a";first.source_leaf_ids=[first_leaf.node_id]
    second=fact("q:b:fact:0","20 pounds",predicate="grain weight",kind="quantity");second.session_id="b";second.source_leaf_ids=[second_leaf.node_id]
    result=_arithmetic_result("What is the total weight of the new feed I purchased?",[first,second],[first_leaf,second_leaf])
    assert result and result[1]["formatted_result"]=="70 pounds"


def test_typed_expansion_uses_query_relevance_and_does_not_flood_contains_edges():
    edges=[
        GraphEdge("card",f"fact:{index}",1.0,"contains",True,confidence=1.0)
        for index in range(20)
    ]
    scores={f"fact:{index}":(1.0 if index==7 else 0.0) for index in range(20)}
    scores["card"]=1.0
    expanded=_typed_expand(
        ["card"],edges,{"contains"},2,20,
        node_scores=scores,
        eligible_nodes={"card",*scores},
        min_score=0.15,
    )
    assert "fact:7" in expanded
    assert len(expanded)<6



def test_typed_expansion_can_inspect_incoming_state_edge_and_traces_direction():
    edge=GraphEdge("new","old",0.96,"supersedes",True,confidence=0.96)
    trace={}
    expanded=_typed_expand(
        ["old"],[edge],{"supersedes"},2,4,
        node_scores={"old":1.0,"new":0.8},
        eligible_nodes={"old","new"},
        trace=trace,
    )
    assert "new" in expanded
    assert trace["paths"]["new"]["traversal"]=="directed_reverse"
    assert trace["paths"]["new"]["relations"]==["supersedes"]


def test_baby_count_deduplicates_named_births_and_twins():
    texts=[
        "User: My son Max was born this month.",
        "User: We welcomed a baby girl named Charlotte.",
        "User: The twins, Ava and Lily, were born yesterday.",
        "User: A baby boy named Jasper arrived this week.",
    ]
    leaves=[];facts=[]
    for index,text in enumerate(texts):
        source=leaf(f"q:s{index}:leaf:0",text=text);source.session_id=f"s{index}";leaves.append(source)
        item=fact(f"q:s{index}:fact:0",text,predicate="birth",kind="event",op="complete")
        item.session_id=f"s{index}";item.source_leaf_ids=[source.node_id];facts.append(item)
    result=_category_count_result("How many babies were born?",facts,leaves)
    assert result and result[0]["count"]==5
    assert set(result[0]["items"])=={"Max","Charlotte","Ava","Lily","Jasper"}


def test_baking_count_deduplicates_same_cake_but_keeps_distinct_cookie_dates():
    rows=[
        ("bread","2023-05-23","User: I tried out a new bread recipe."),
        ("cake-a","2023-05-21","User: I baked a chocolate cake for my sister birthday."),
        ("cake-b","2023-05-22","User: I just baked a chocolate cake for my sister birthday."),
        ("cookie-a","2023-05-24","User: I baked a batch of cookies last Thursday."),
        ("cookie-b","2023-05-28","User: I baked a batch of cookies last Thursday."),
    ]
    leaves=[];facts=[]
    for sid,when,text in rows:
        source=leaf(f"q:{sid}:leaf:0",text=text);source.session_id=sid;source.session_date=when;leaves.append(source)
        item=fact(f"q:{sid}:fact:0",text,predicate="baked",kind="event",op="complete")
        item.session_id=sid;item.source_leaf_ids=[source.node_id];facts.append(item)
    result=_category_count_result("How many times did I bake?",facts,leaves)
    assert result and result[0]["count"]==4


def test_points_remaining_prefers_current_total_over_earned_and_target_total():
    current_leaf=leaf("q:current:leaf:0",text="User: I earned 50 points, bringing my total to 200 points.")
    current_leaf.session_id="current"
    target_leaf=leaf("q:target:leaf:0",text="User: I am close to redeeming a free product; I just need a total of 300 points.")
    target_leaf.session_id="target"
    current=fact("q:current:fact:0","200 points",predicate="total_points",kind="quantity")
    current.session_id="current";current.source_leaf_ids=[current_leaf.node_id]
    target=fact("q:target:fact:0","300 points total",predicate="needs_points",kind="quantity")
    target.session_id="target";target.source_leaf_ids=[target_leaf.node_id]
    result=_arithmetic_result("How many points do I need to earn to redeem a free skincare product at Sephora?",[current,target],[current_leaf,target_leaf])
    assert result and result[1]["formatted_result"]=="100 points"


def test_paired_expense_binds_currency_after_each_named_item():
    wash_leaf=leaf("q:wash:leaf:0",text="User: I got a car wash on February 3 for $15.");wash_leaf.session_id="wash"
    parking_leaf=leaf("q:park:leaf:0",text="User: Air filter was $25, gas was $30, and my parking ticket was $50.");parking_leaf.session_id="park"
    wash=fact("q:wash:fact:0","$15",predicate="car wash cost",kind="quantity");wash.session_id="wash";wash.source_leaf_ids=[wash_leaf.node_id]
    parking=fact("q:park:fact:0","$50",predicate="parking ticket cost",kind="quantity");parking.session_id="park";parking.source_leaf_ids=[parking_leaf.node_id]
    result=_arithmetic_result("How much did I spend on car wash and parking ticket?",[wash,parking],[wash_leaf,parking_leaf])
    assert result and result[1]["formatted_result"]=="$65"


def test_rare_inventory_understands_collection_of_books_word_order():
    rows=[
        ("records","User: I have 57 rare records."),
        ("figurines","User: I have 12 rare figurines."),
        ("coins","User: I have 25 rare coins."),
        ("books","User: I collect rare books and have a small collection of 5 books."),
    ]
    leaves=[];facts=[]
    for sid,text in rows:
        source=leaf(f"q:{sid}:leaf:0",text=text);source.session_id=sid;leaves.append(source)
        item=fact(f"q:{sid}:fact:0",text,predicate="has collection",kind="quantity")
        item.session_id=sid;item.source_leaf_ids=[source.node_id];facts.append(item)
    result=_arithmetic_result("How many rare items do I have in total?",facts,leaves)
    assert result and result[1]["formatted_result"]=="99"


def test_exact_comparison_handles_possessive_plural_and_verb_morphology():
    fence_leaf=leaf("q:fence:leaf:0",text="User: I fixed the fence last week.");fence_leaf.session_id="fence"
    hoof_leaf=leaf("q:hoof:leaf:0",text="User: I trimmed the goat hooves two weeks ago.");hoof_leaf.session_id="hoof"
    fence=fact("q:fence:fact:0","last week",predicate="fixed_fence",kind="event",op="complete");fence.session_id="fence";fence.source_leaf_ids=[fence_leaf.node_id]
    hoof=fact("q:hoof:fact:0","two weeks ago",predicate="trimmed_goat_hooves",kind="event",op="complete");hoof.session_id="hoof";hoof.source_leaf_ids=[hoof_leaf.node_id]
    result=_exact_entity_result("Which task did I complete first, fixing the fence or trimming the goats hooves?",[fence,hoof],[fence_leaf,hoof_leaf])
    assert result is None


def test_relative_date_scope_accepts_articles_and_couple_of_days():
    week,_=_question_date_scope("What happened a week ago?","2024/05/22 (Wed) 12:00")
    couple,_=_question_date_scope("What did I bake a couple of days ago?","2024/05/22 (Wed) 12:00")
    assert str(week)=="2024-05-15"
    assert str(couple)=="2024-05-20"


def test_doctor_count_deduplicates_repeated_specialty_mentions():
    rows=[
        ("pcp","User: I visited my primary care physician for a checkup."),
        ("ent","User: I got back from an appointment with the ENT specialist."),
        ("derm-a","User: My dermatologist did a biopsy."),
        ("derm-b","User: I visited the dermatologist again for the result."),
    ]
    leaves=[];facts=[]
    for sid,text in rows:
        source=leaf(f"q:{sid}:leaf:0",text=text);source.session_id=sid;leaves.append(source)
        item=fact(f"q:{sid}:fact:0",text,predicate="doctor visit",kind="event",op="complete")
        item.session_id=sid;item.source_leaf_ids=[source.node_id];facts.append(item)
    result=_category_count_result("How many different doctors did I visit?",facts,leaves)
    assert result and result[0]["count"]==3


def test_rollercoaster_count_understands_word_number_and_named_list():
    rows=[
        ("repeat","User: I rode the Mako rollercoaster three times."),
        ("list","User: I rode the Space Mountain, Kraken and Manta rollercoasters."),
    ]
    leaves=[];facts=[]
    for sid,text in rows:
        source=leaf(f"q:{sid}:leaf:0",text=text);source.session_id=sid;leaves.append(source)
        item=fact(f"q:{sid}:fact:0",text,predicate="rode rollercoaster",kind="event",op="complete")
        item.session_id=sid;item.source_leaf_ids=[source.node_id];facts.append(item)
    result=_category_count_result("How many times did I ride rollercoasters?",facts,leaves)
    assert result and result[0]["count"]==6


def test_trip_duration_sum_aggregates_distinct_destinations():
    rows=[
        ("hawaii","User: I stayed in Hawaii for 8 days."),
        ("nyc","User: My New York City trip was 7-days long."),
    ]
    leaves=[];facts=[]
    for sid,text in rows:
        source=leaf(f"q:{sid}:leaf:0",text=text);source.session_id=sid;leaves.append(source)
        item=fact(f"q:{sid}:fact:0",text,predicate="trip duration",kind="quantity")
        item.session_id=sid;item.source_leaf_ids=[source.node_id];facts.append(item)
    result=_arithmetic_result("How many total days did I spend in Hawaii and New York City?",facts,leaves)
    assert result and result[1]["formatted_result"]=="15 days"


def test_arrival_question_does_not_use_explicit_departure_time():
    source=leaf(
        "q:clinic:leaf:0",
        text="User: I left for the clinic at 8 AM and the trip took one hour.",
    )
    source.session_id="clinic"
    item=fact(
        "q:clinic:fact:0",
        "left at 8 AM and traveled for one hour",
        predicate="clinic trip",
        kind="event",
        op="complete",
    )
    item.session_id="clinic";item.source_leaf_ids=[source.node_id]
    from graphmem_demo.hierarchical_v2 import _explicit_event_time_result
    assert _explicit_event_time_result(
        "What time did I arrive at the clinic?",[item],[source]
    ) is None


def test_candidate_pool_operator_survives_final_fact_budget():
    sources=[];facts=[]
    for index in range(4):
        src=leaf(f"q:s{index}:leaf:0",text=f"User: I attended event {index}.")
        src.session_id=f"s{index}";sources.append(src)
        item=fact(f"q:s{index}:fact:0",f"event {index}",predicate="attended",kind="event")
        item.session_id=f"s{index}";item.source_leaf_ids=[src.node_id];facts.append(item)
    ledger=[{
        "operator":"distinct_completed_items",
        "result":{"count":4,"items":[f"event {index}" for index in range(4)]},
        "source_fact_ids":[item.node_id for item in facts],
        "candidate_pool_complete":True,
    }]
    _,_,_,context=_pack_context(
        "How many events did I attend?","count/list",[],facts[:1],sources[:1],ledger,4000
    )
    assert "final count must be 4" in _mandatory_answer_hint(context).casefold()


def test_answer_guard_does_not_override_unsafe_candidate_pool_count():
    source=leaf("q:s:leaf:0",text="User: I attended four events.")
    item=fact("q:s:fact:0","event one",predicate="attended",kind="event")
    ledger=[{
        "operator":"distinct_completed_items",
        "result":{"count":4,"items":["one","two","three","four"]},
        "source_fact_ids":[item.node_id],
        "candidate_pool_complete":True,
    }]
    _,_,_,context=_pack_context(
        "How many events did I attend?","count/list",[],[item],[source],ledger,4000
    )
    retrieval=RetrievedContext(
        question_id="q",variant="hierarchical_state_graph_v2",
        summary_node_ids=[],leaf_node_ids=[],edge_count=0,context_text=context,
        answer_session_hit=False,retrieved_session_ids=[],latency_sec=0.0,
    )
    answer,trace=apply_answer_constraint(
        "How many events did I attend?",retrieval,"I found three events."
    )
    assert answer=="I found three events."
    assert trace["applied"] is False


def test_answer_guard_applies_only_complete_safe_calculation():
    context=("[EVIDENCE LEDGER]\n"
        "[{\"operator\":\"generic_calculation\",\"result\":{\"calculation_type\":\"feed_weight_sum\",\"formatted_result\":\"70 pounds\"},\"candidate_pool_complete\":true}]")
    retrieval=RetrievedContext(
        question_id="q",variant="hierarchical_state_graph_v2",
        summary_node_ids=[],leaf_node_ids=[],edge_count=0,context_text=context,
        answer_session_hit=False,retrieved_session_ids=[],latency_sec=0.0,
    )
    answer,trace=apply_answer_constraint("What is the total feed weight?",retrieval,"50 pounds")
    assert answer=="70 pounds" and trace["operator"]=="generic_calculation"



def test_operand_graph_links_same_measure_collection_and_operands():
    cost_leaf=leaf("q:a:leaf:0",text="User: I spent $60 on coffee mugs for coworkers.");cost_leaf.session_id="a"
    count_leaf=leaf("q:b:leaf:0",text="User: I purchased 5 coffee mugs, one for each coworker.");count_leaf.session_id="b"
    cost=fact("q:a:fact:0","$60",predicate="gift cost",kind="quantity");cost.session_id="a";cost.source_leaf_ids=[cost_leaf.node_id]
    count=fact("q:b:fact:0","5 coffee mugs",predicate="purchased",kind="quantity");count.session_id="b";count.source_leaf_ids=[count_leaf.node_id]
    edges=build_graph_edges([cost_leaf,count_leaf],[],[cost,count],[],semantic_k=1,semantic_floor=0.99)
    relations={edge.relation for edge in edges if {edge.src,edge.dst}=={cost.node_id,count.node_id}}
    assert {"same_measure","same_collection","operand_of"} <= relations
    assert cost.collection_key==count.collection_key=="coffee_mugs"


def test_generic_per_unit_and_grouped_follower_delta_operators():
    cost_leaf=leaf("q:a:leaf:0",text="User: I spent $60 on coffee mugs for my coworkers.");cost_leaf.session_id="a"
    count_leaf=leaf("q:b:leaf:0",text="User: I purchased 5 coffee mugs, one for each coworker.");count_leaf.session_id="b"
    cost=fact("q:a:fact:0","$60",predicate="gift cost",kind="quantity");cost.session_id="a";cost.source_leaf_ids=[cost_leaf.node_id]
    count=fact("q:b:fact:0","5 mugs",predicate="purchased",kind="quantity");count.session_id="b";count.source_leaf_ids=[count_leaf.node_id]
    result=_arithmetic_result("How much did I spend on each coffee mug for my coworkers?",[cost,count],[cost_leaf,count_leaf])
    assert result and result[1]["calculation_type"]=="per_unit_price" and result[1]["formatted_result"]=="$12"

    twitter_leaf=leaf("q:t:leaf:0",text="User: My Twitter follower count jumped from 420 to 540 over the past month.");twitter_leaf.session_id="t"
    tiktok_leaf=leaf("q:k:leaf:0",text="User: On TikTok I gained around 200 followers over the past three weeks.");tiktok_leaf.session_id="k"
    facebook_leaf=leaf("q:f:leaf:0",text="User: My Facebook follower count remained steady at 800.");facebook_leaf.session_id="f"
    fs=[]
    for source in (twitter_leaf,tiktok_leaf,facebook_leaf):
        item=fact(source.node_id.replace("leaf","fact"),"followers",predicate="follower change",kind="quantity")
        item.session_id=source.session_id;item.source_leaf_ids=[source.node_id];fs.append(item)
    result=_arithmetic_result("Which social media platform did I gain the most followers on?",fs,[twitter_leaf,tiktok_leaf,facebook_leaf])
    assert result and result[1]["formatted_result"]=="TikTok"


def test_generic_collection_sums_and_airline_frequency():
    writing_texts=[
        "User: I've written 17 poems in the past two weeks.",
        "User: I've written five short stories so far.",
        "User: In the writing challenge, I wrote a piece titled The Smell of Old Books.",
    ]
    leaves=[];facts=[]
    for index,text in enumerate(writing_texts):
        source=leaf(f"q:w{index}:leaf:0",text=text);source.session_id=f"w{index}";leaves.append(source)
        item=fact(f"q:w{index}:fact:0",text,predicate="completed writing",kind="quantity");item.session_id=source.session_id;item.source_leaf_ids=[source.node_id];facts.append(item)
    result=_arithmetic_result("How many total pieces of writing have I completed, including short stories, poems, and pieces for the writing challenge?",facts,leaves)
    assert result and result[1]["formatted_result"]=="23"

    united_leaf=leaf("q:u:leaf:0",text="User: In March I flew with United Airlines, with two flights each way.");united_leaf.session_id="u"
    southwest_leaf=leaf("q:s:leaf:0",text="User: In March I took a direct flight with Southwest Airlines.");southwest_leaf.session_id="s"
    american_leaf=leaf("q:a:leaf:0",text="User: In April we flew with American Airlines to Honolulu and then took a connecting flight to Maui.");american_leaf.session_id="a"
    fs=[]
    for source in (united_leaf,southwest_leaf,american_leaf):
        item=fact(source.node_id.replace("leaf","fact"),"flight",predicate="flew",kind="event",op="complete")
        item.session_id=source.session_id;item.source_leaf_ids=[source.node_id];fs.append(item)
    result=_arithmetic_result("Which airline did I fly with the most in March and April?",fs,[united_leaf,southwest_leaf,american_leaf])
    assert result and result[1]["formatted_result"]=="United Airlines"


def test_event_anchor_valentine_assistant_metric_and_current_company():
    website_leaf=leaf("q:w:leaf:0",text="User: I just launched my website.");website_leaf.session_id="w";website_leaf.session_date="2023-02-10"
    contract_leaf=leaf("q:c:leaf:0",text="User: I signed a contract with my first client today.");contract_leaf.session_id="c";contract_leaf.session_date="2023-03-01"
    website=fact("q:w:fact:0","website",predicate="launched",kind="event",op="complete");website.session_id="w";website.source_leaf_ids=[website_leaf.node_id]
    contract=fact("q:c:fact:0","first client contract",predicate="signed",kind="event",op="complete");contract.session_id="c";contract.source_leaf_ids=[contract_leaf.node_id]
    result=_temporal_calculation_result("How many days ago did I launch my website when I signed a contract with my first client?",[website,contract],[website_leaf,contract_leaf],"2023-03-06")
    assert result and result[1]["formatted_result"]=="19 days ago"

    shelter_leaf=leaf("q:s:leaf:0",text="User: I volunteered at the animal shelter fundraising dinner back on Valentine's Day.")
    shelter=fact("q:s:fact:0","animal shelter fundraising dinner",predicate="volunteered",kind="event",op="complete")
    event=_explicit_event_time_result("When did I volunteer at the animal shelter's fundraising dinner?",[shelter],[shelter_leaf])
    assert event and event[1]["value"]=="February 14th"

    metric_leaf=leaf("q:m:leaf:0",text="User: Summarize the paper.\nAssistant: The average improvement in framerate was approximately 20% when using the HAMT agent, with a further 4x improvement from active modulation.")
    metric=fact("q:m:fact:0","approximately 20%",predicate="average framerate improvement",kind="assistant_fact");metric.role="assistant";metric.source_leaf_ids=[metric_leaf.node_id]
    recalled=_assistant_recall_result("Can you remind me what was the average improvement in framerate when using the HAMT agent?",[metric],[metric_leaf])
    assert recalled and recalled[1]["value"]=="approximately 20%"

    company_leaf=leaf("q:r:leaf:0",text="User: Rachel, an old colleague, who's currently at TechCorp.");company_leaf.session_date="2023-05-26"
    company_fact=fact("q:r:fact:0","Rachel",predicate="old colleague",kind="state");company_fact.source_leaf_ids=[company_leaf.node_id]
    assert query_kind("What company is Rachel, an old colleague from my previous company, currently working at?")=="current/update"
    ledger=build_evidence_ledger("current/update",[company_fact],[],[company_leaf],"What company is Rachel currently working at?",operator_facts=[company_fact],operator_leaves=[company_leaf])
    latest=next(row for row in ledger if row.get("operator")=="latest_valid_state")
    assert latest["candidate_pool_complete"] and latest["result"][0]["object"]=="TechCorp"


def test_temporal_operator_uses_complete_memory_when_shortlist_drops_operand():
    start_leaf=leaf("q:start:leaf:0",text="User: I started reading The Nightingale today.")
    start_leaf.session_id="start";start_leaf.session_date="2024-01-10"
    end_leaf=leaf("q:end:leaf:0",text="User: I finished reading The Nightingale today.")
    end_leaf.session_id="end";end_leaf.session_date="2024-01-31"
    distractor_leaf=leaf("q:d:leaf:0",text="User: I recommended The Nightingale to a friend.")
    distractor_leaf.session_id="d";distractor_leaf.session_date="2024-02-02"

    start=fact("q:start:fact:0","The Nightingale",predicate="started reading",kind="event",op="complete")
    start.session_id="start";start.source_leaf_ids=[start_leaf.node_id];start.event_time="2024-01-10"
    end=fact("q:end:fact:0","The Nightingale",predicate="finished reading",kind="event",op="complete")
    end.session_id="end";end.source_leaf_ids=[end_leaf.node_id];end.event_time="2024-01-31"
    distractor=fact("q:d:fact:0","The Nightingale",predicate="recommended",kind="event",op="complete")
    distractor.session_id="d";distractor.source_leaf_ids=[distractor_leaf.node_id];distractor.event_time="2024-02-02"

    ledger=build_evidence_ledger(
        "temporal",[distractor],[],[distractor_leaf],
        "How many days did it take me to finish reading The Nightingale?",
        operator_facts=[distractor],operator_leaves=[distractor_leaf],
        complete_facts=[start,end,distractor],
        complete_leaves=[start_leaf,end_leaf,distractor_leaf],
    )
    row=next(
        item for item in ledger
        if item.get("operator")=="generic_calculation"
        and item.get("result",{}).get("calculation_type")=="elapsed_days"
    )
    assert row["candidate_pool_complete"] is True
    assert row["result"]["formatted_result"]=="21 days"


def test_education_duration_keeps_explicit_duration_after_later_degree_mention():
    rows=[
        ("high","User: I attended high school from 2004 to 2008."),
        ("associate","User: I completed my associate's degree in 2010."),
        ("bachelor-duration","User: My bachelor's degree took me four years."),
        ("bachelor-later","User: I later mentioned my bachelor degree while discussing work."),
    ]
    leaves=[];facts=[]
    for sid,text in rows:
        source=leaf(f"q:{sid}:leaf:0",text=text);source.session_id=sid;leaves.append(source)
        item=fact(f"q:{sid}:fact:0",text,predicate="education",kind="event",op="complete")
        item.session_id=sid;item.source_leaf_ids=[source.node_id];facts.append(item)
    result=_arithmetic_result(
        "How many total years of formal education did I complete through my bachelor's degree?",
        facts,leaves,
    )
    assert result and result[1]["formatted_result"]=="10 years"


def test_music_event_sequence_ignores_plans_and_orders_completed_events():
    rows=[
        ("billie","2024-03-18","User: I saw Billie Eilish in concert at the Wells Fargo Center in Philly."),
        ("outdoor","2024-03-25","User: I went to a free outdoor concert series in the park."),
        ("planned-jazz","2024-03-25","User: I plan to go to jazz night at a local bar next month."),
        ("festival","2024-04-01","User: I just got back from a music festival in Brooklyn."),
        ("jazz","2024-04-08","User: I enjoyed jazz night at a local bar today."),
        ("queen","2024-04-15","User: I saw Queen and Adam Lambert live at the Prudential Center in Newark, NJ."),
    ]
    leaves=[];facts=[]
    for sid,when,text in rows:
        source=leaf(f"q:{sid}:leaf:0",text=text);source.session_id=sid;source.session_date=when;leaves.append(source)
        item=fact(f"q:{sid}:fact:0",text,predicate="musical event",kind="event",op="complete")
        item.session_id=sid;item.source_leaf_ids=[source.node_id];facts.append(item)
    result=_event_sequence_result(
        "What is the chronological order of the concerts and musical events I attended?",
        facts,leaves,
    )
    assert result
    assert [row["event"] for row in result[1]]==[
        "Billie Eilish concert at the Wells Fargo Center in Philly",
        "Free outdoor concert series in the park",
        "Music festival in Brooklyn",
        "Jazz night at a local bar",
        "Queen + Adam Lambert concert at the Prudential Center in Newark, NJ",
    ]


def test_blind2_cross_session_revenue_percentage_and_comment_sum():
    texts=[
        ("eggs", "User: I sold a total of 40 dozen eggs this month."),
        ("rate", "User: I charge $3 per dozen for my eggs."),
        ("women", "User: Women hold 20 of the leadership positions."),
        ("total", "User: We have a total of 100 leadership positions."),
        ("facebook", "User: My recent Facebook Live session received 12 comments."),
        ("youtube", "User: My most popular YouTube video has 21 comments."),
    ]
    leaves=[];facts=[]
    for sid,text in texts:
        source=leaf(f"q:{sid}:leaf:0",text=text);source.session_id=sid;leaves.append(source)
        item=fact(f"q:{sid}:fact:0",text,predicate="quantity",kind="quantity")
        item.session_id=sid;item.source_leaf_ids=[source.node_id];facts.append(item)
    revenue=_arithmetic_result("How much have I made from selling eggs this month?",facts,leaves)
    ratio=_arithmetic_result("What percentage of leadership positions do women hold in my company?",facts,leaves)
    comments=_arithmetic_result("What is the total number of comments on my recent Facebook Live session and my most popular YouTube video?",facts,leaves)
    assert revenue and revenue[1]["formatted_result"]=="$120"
    assert ratio and ratio[1]["formatted_result"]=="20%"
    assert comments and comments[1]["formatted_result"]=="33"


def test_blind2_relative_purchase_comparison_binds_adjacent_sentence_duration():
    bed=leaf(
        "q:bed:leaf:0",
        text="User: I got a new dog bed for Max about three weeks ago.",
    );bed.session_id="bed";bed.session_date="2023-05-29"
    pads=leaf(
        "q:pads:leaf:0",
        text=("User: I've been using eco-friendly training pads for Luna. "
              "I got a set of 10 for $25 about a month ago."),
    );pads.session_id="pads";pads.session_date="2023-05-29"
    bed_fact=fact("q:bed:fact:0","dog bed for Max",predicate="purchased",kind="event",op="complete")
    bed_fact.session_id="bed";bed_fact.source_leaf_ids=[bed.node_id]
    pads_fact=fact("q:pads:fact:0","training pads for Luna",predicate="purchased",kind="event",op="complete")
    pads_fact.session_id="pads";pads_fact.source_leaf_ids=[pads.node_id]
    result=_temporal_calculation_result(
        "Which item did I purchase first, the dog bed for Max or the training pads for Luna?",
        [bed_fact,pads_fact],[bed,pads],"2023-05-29",
    )
    assert result and result[1]["calculation_type"]=="relative_event_comparison"
    assert result[1]["formatted_result"]=="the training pads for luna"


def test_blind2_exact_residence_ignores_friend_city_and_month_name():
    friend=leaf("q:f:leaf:0",text="User: I invited friends who live in Tokyo.");friend.session_id="f"
    current=leaf("q:h:leaf:0",text="User: I've been living in Harajuku for three months.");current.session_id="h"
    move=leaf("q:m:leaf:0",text="User: I moved to my new apartment in March.");move.session_id="m"
    facts=[]
    for source in (friend,current,move):
        item=fact(source.node_id.replace("leaf","fact"),source.user_text,predicate="residence",kind="state")
        item.session_id=source.session_id;item.source_leaf_ids=[source.node_id];facts.append(item)
    result=_exact_entity_result(
        "How long have I been living in my current apartment in Shinjuku?",
        facts,[friend,current,move],
    )
    assert result and result[1]["exact_match"] is False
    assert result[1]["partial_entity"]=="Harajuku"


def test_blind2_assistant_recall_chapter_and_store_name():
    chapter_leaf=leaf(
        "q:c:leaf:0",
        text=("User: Which chapter discusses vocal prayer and meditation?\n"
              "Assistant: Chapter 4 of Book 1, titled 'Vocal Prayer and Meditation'."),
    );chapter_leaf.session_id="c"
    chapter_fact=fact("q:c:fact:0","Chapter 4 of Book 1",predicate="answered",kind="assistant_fact")
    chapter_fact.role="assistant";chapter_fact.session_id="c";chapter_fact.source_leaf_ids=[chapter_leaf.node_id]
    chapter=_assistant_recall_result(
        "Can you remind me from our previous chat which chapter of the second part discusses vocal prayer and meditation?",
        [chapter_fact],[chapter_leaf],
    )
    assert chapter and chapter[1]["value"].startswith("Chapter 4 of Book 1")

    store_leaf=leaf(
        "q:n:leaf:0",
        text=("User: Name an Indian online fabric store.\nAssistant: "
              "1. Nostalgia: An online store based in India selling traditional fabrics, threads, and embellishments."),
    );store_leaf.session_id="n"
    store_fact=fact("q:n:fact:0","Nostalgia",predicate="recommended",kind="assistant_fact")
    store_fact.role="assistant";store_fact.session_id="n";store_fact.source_leaf_ids=[store_leaf.node_id]
    store=_assistant_recall_result(
        "Can you remind me from our previous conversation of the name of the online store based in India that sells traditional Indian fabrics, threads, and embellishments?",
        [store_fact],[store_leaf],
    )
    assert store and store[1]["value"]=="Nostalgia"


def test_blind2_doctor_count_and_evening_preference_focus():
    attended=leaf("q:a:leaf:0",text="User: I went to my primary care physician on March 3.");attended.session_id="a"
    orthopedic=leaf("q:o:leaf:0",text="User: I saw an orthopedic surgeon on March 20.");orthopedic.session_id="o"
    planned=leaf("q:p:leaf:0",text="User: I scheduled a dentist appointment for March 28.");planned.session_id="p"
    facts=[]
    for source in (attended,orthopedic,planned):
        item=fact(source.node_id.replace("leaf","fact"),source.user_text,predicate="doctor appointment",kind="event",op="complete")
        item.session_id=source.session_id;item.source_leaf_ids=[source.node_id];facts.append(item)
    count=_category_count_result("How many doctor's appointments did I go to in March?",facts,[attended,orthopedic,planned])
    assert count and count[0]["count"]==2
    focus=_preference_focus_instruction(
        "Can you suggest some activities that I can do in the evening?",
        ["I relax before 9:30 pm", "Phone and TV have been affecting my sleep quality"],
    )
    assert focus and "before 9:30 pm" in focus and "avoid phone" in focus.casefold()

    distractor=leaf("q:d:leaf:0",text="User: I enjoy outdoor activities.");distractor.session_id="d"
    constraint=leaf(
        "q:e:leaf:0",
        text=("User: I need relaxing activities before 9:30 pm. "
              "Using my phone or watching TV has been affecting my sleep quality."),
    );constraint.session_id="e"
    distractor_fact=fact("q:d:fact:0","outdoor activities",predicate="likes",kind="preference")
    distractor_fact.session_id="d";distractor_fact.source_leaf_ids=[distractor.node_id]
    constraint_fact=fact("q:e:fact:0","relax before 9:30 without phone or TV",predicate="evening preference",kind="preference")
    constraint_fact.session_id="e";constraint_fact.source_leaf_ids=[constraint.node_id]
    ledger=build_evidence_ledger(
        "preference",[distractor_fact],[],[distractor],
        "Can you suggest some activities that I can do in the evening?",
        operator_facts=[distractor_fact],operator_leaves=[distractor],
        complete_facts=[distractor_fact,constraint_fact],
        complete_leaves=[distractor,constraint],
    )
    preference=next(row for row in ledger if row.get("operator")=="contextual_preferences")
    assert preference["candidate_pool_complete"] is True
    assert "before 9:30 pm" in preference["result"]["focus_instruction"]


def test_blind3_ratio_window_value_and_latest_routine():
    old=leaf("q:old:leaf:0",text="User: My French press ratio is 1 tablespoon of coffee for every 6 ounces of water.")
    old.session_id="old";old.session_date="2023-02-11"
    new=leaf("q:new:leaf:0",text="User: My French press ratio is 1 tablespoon of coffee for every 5 ounces of water.")
    new.session_id="new";new.session_date="2023-06-30"
    first=leaf("q:first:leaf:0",text="User: That's 15 autographed baseballs since I started collecting three months ago!")
    first.session_id="first"
    later=leaf("q:later:leaf:0",text="User: I've added 20 autographed baseballs in the past few months.")
    later.session_id="later"
    gym7=leaf("q:g7:leaf:0",text="User: My gym sessions, which I usually go to at 7:00 pm, are Monday, Wednesday, Friday.")
    gym7.session_id="g7";gym7.session_date="2023-02-11"
    gym6=leaf("q:g6:leaf:0",text="User: I head to the gym, which is usually at 6:00 pm.")
    gym6.session_id="g6";gym6.session_date="2023-05-30"
    leaves=[old,new,first,later,gym7,gym6];facts=[]
    for source in leaves:
        item=fact(source.node_id.replace("leaf","fact"),source.user_text,predicate="stated",kind="quantity")
        item.session_id=source.session_id;item.source_leaf_ids=[source.node_id];facts.append(item)
    ratio=_arithmetic_result("For the coffee-to-water ratio in my French press, did I switch to more water per tablespoon of coffee, or less?",facts,leaves)
    collection=_arithmetic_result("How many autographed baseballs have I added to my collection in the first three months of collection?",facts,leaves)
    gym=_arithmetic_result("What time do I usually go to the gym?",facts,leaves)
    assert ratio and ratio[1]["formatted_result"]=="less water (5 ounces) per tablespoon of coffee"
    assert collection and collection[1]["formatted_result"]=="15"
    assert gym and gym[1]["formatted_result"]=="6:00 pm"


def test_blind3_duration_sums_for_trips_and_social_breaks():
    japan=leaf("q:j:leaf:0",text="User: I went to Japan before from April 15th to 22nd.");japan.session_id="j"
    chicago=leaf("q:c:leaf:0",text="User: I loved my last 4-day trip to Chicago.");chicago.session_id="c"
    ten=leaf("q:t:leaf:0",text="User: I just got back from a 10-day social media break.");ten.session_id="t"
    week=leaf("q:w:leaf:0",text="User: I took a week-long break from social media.");week.session_id="w"
    leaves=[japan,chicago,ten,week];facts=[]
    for source in leaves:
        item=fact(source.node_id.replace("leaf","fact"),source.user_text,predicate="completed",kind="event",op="complete")
        item.session_id=source.session_id;item.source_leaf_ids=[source.node_id];facts.append(item)
    trips=_arithmetic_result("What is the total number of days I spent in Japan and Chicago?",facts,leaves)
    breaks=_arithmetic_result("How many days did I take social media breaks in total?",facts,leaves)
    assert trips and trips[1]["formatted_result"]=="11 days"
    assert breaks and breaks[1]["formatted_result"]=="17 days"


def test_blind3_direct_date_difference_and_last_week_location():
    bought=leaf("q:b:leaf:0",text="User: I bought my laptop backpack on 1/15.");bought.session_id="b";bought.session_date="2023-01-24"
    arrived=leaf("q:a:leaf:0",text="User: My new laptop backpack arrived on 1/20.");arrived.session_id="a";arrived.session_date="2023-01-24"
    church=leaf("q:e:leaf:0",text="User: I got to attend the Maundy Thursday service at the Episcopal Church.")
    church.session_id="e";church.session_date="2023-04-06"
    leaves=[bought,arrived,church];facts=[]
    for source in leaves:
        item=fact(source.node_id.replace("leaf","fact"),source.user_text,predicate="attended",kind="event",op="complete")
        item.session_id=source.session_id;item.source_leaf_ids=[source.node_id];facts.append(item)
    elapsed=_temporal_calculation_result("How many days did it take for my laptop backpack to arrive after I bought it?",facts,leaves,"2023-01-24")
    location=_temporal_calculation_result("Where did I attend the religious activity last week?",facts,leaves,"2023-04-10")
    assert elapsed and elapsed[1]["formatted_result"]=="5 days"
    assert location and location[1]["formatted_result"]=="the Episcopal Church"


def test_blind3_numbered_assistant_item_and_missing_parent_evidence():
    listing=leaf(
        "q:l:leaf:0",
        text=("User: Give me prompt parameters.\nAssistant: 26. Soliloquy\n"
              "27. Sound effects (e.g., ambient, diegetic, non-diegetic, etc.)\n28. Music"),
    );listing.session_id="l"
    listing_fact=fact("q:l:fact:0","prompt parameter list",predicate="provided",kind="assistant_fact")
    listing_fact.role="assistant";listing_fact.session_id="l";listing_fact.source_leaf_ids=[listing.node_id]
    recalled=_assistant_recall_result(
        "Can you remind me what was the 27th parameter on that list?",
        [listing_fact],[listing],
    )
    assert recalled and recalled[1]["value"].startswith("Sound effects")

    alex=leaf("q:x:leaf:0",text="User: My cousin Alex just adopted a baby girl from China in January.")
    alex.session_id="x"
    unrelated=leaf("q:t:leaf:0",text="User: Tom recommended a restaurant.");unrelated.session_id="t"
    alex_fact=fact("q:x:fact:0","Alex adopted a baby girl",predicate="became parent",kind="event",op="complete")
    alex_fact.session_id="x";alex_fact.source_leaf_ids=[alex.node_id]
    tom_fact=fact("q:t:fact:0","Tom recommended a restaurant",predicate="recommended",kind="event",op="complete")
    tom_fact.session_id="t";tom_fact.source_leaf_ids=[unrelated.node_id]
    exact=_exact_entity_result("Who became a parent first, Tom or Alex?",[alex_fact,tom_fact],[alex,unrelated])
    assert exact and exact[1]["exact_match"] is False
    assert exact[1]["partial_entity"]=="Alex" and exact[1]["missing_alternatives"]==["Tom"]


def test_blind4_monthly_faith_activity_days_are_distinct_and_completed():
    rows=[
        ("drive","User: I helped out at the church holiday food drive on December 10th."),
        ("study","User: I actually did a Bible study at my church on December 17th."),
        ("mass","User: I just got back from midnight mass at St. Mary's Church on December 24th."),
        ("repeat","User: That Bible study on December 17th was thought-provoking."),
        ("plan","User: I plan to attend a church service on December 30th."),
    ]
    leaves=[];facts=[]
    for sid,text in rows:
        source=leaf(f"q:{sid}:leaf:0",text=text)
        source.session_id=sid;source.session_date="2024-01-10";leaves.append(source)
        item=fact(f"q:{sid}:fact:0",text,predicate="faith activity",kind="event",op="complete")
        item.session_id=sid;item.source_leaf_ids=[source.node_id];facts.append(item)
    result=_temporal_calculation_result(
        "How many days did I spend participating in faith-related activities in December?",
        facts,leaves,"2024-01-10",
    )
    assert result and result[1]["calculation_type"]=="distinct_event_days_in_month"
    assert result[1]["formatted_result"]=="3 days"
    assert result[1]["dates"]==["2023-12-10","2023-12-17","2023-12-24"]


def test_blind4_event_comparison_prefers_completed_event_fact_and_source_date():
    tomatoes=leaf(
        "q:t:leaf:0",
        text="User: I've been starting seeds indoors since February 20th - tomatoes are doing well.",
    );tomatoes.session_id="t";tomatoes.session_date="2023-03-10"
    marigolds=leaf(
        "q:m:leaf:0",
        text="User: I just started some marigold seeds that arrived on March 3rd.",
    );marigolds.session_id="m";marigolds.session_date="2023-03-10"
    tomato_event=fact("q:t:fact:0","2023-02-20",predicate="started seeds indoors",kind="event",op="complete",when="2023-02-20")
    tomato_event.session_id="t";tomato_event.source_leaf_ids=[tomatoes.node_id]
    tomato_height=fact("q:t:fact:1","2-3 inches",predicate="tomato seedling height",kind="quantity",when=None)
    tomato_height.session_id="t";tomato_height.source_leaf_ids=[tomatoes.node_id]
    marigold_event=fact("q:m:fact:0","started marigold seeds",predicate="started",kind="event",op="complete",when="2023-03-03")
    marigold_event.session_id="m";marigold_event.source_leaf_ids=[marigolds.node_id]
    result=_event_comparison_result(
        "Which seeds were started first, the tomatoes or the marigolds?",
        [tomato_event,tomato_height,marigold_event],[tomatoes,marigolds],
    )
    assert result and result[1]["value"]=="the tomatoes"
    assert result[1]["resolved_dates"]=={
        "the tomatoes":"2023-02-20","the marigolds":"2023-03-03",
    }
    assert result[1]["high_confidence"] is True
    assert _candidate_pool_is_complete("event_comparison",result[1],True) is True


def test_event_comparison_is_not_forced_without_two_distinct_anchored_dates():
    assert _candidate_pool_is_complete(
        "event_comparison",
        {"high_confidence":False,"resolved_dates":{"a":"2023-05-01","b":"2023-05-02"}},
        True,
    ) is False
    assert _candidate_pool_is_complete(
        "event_comparison",
        {"high_confidence":True,"resolved_dates":{"a":"2023-05-01","b":"2023-05-01"}},
        True,
    ) is False


def test_answer_guard_prefers_exact_mismatch_and_anchored_calculation_over_comparison():
    rows=[
        {"operator":"event_comparison","result":{"value":"dog bed"},"candidate_pool_complete":True},
        {"operator":"generic_calculation","result":{"calculation_type":"relative_event_comparison","formatted_result":"training pads"},"candidate_pool_complete":True},
        {"operator":"exact_entity_check","result":{"requested_entity":"Porsche","exact_match":False},"candidate_pool_complete":True},
    ]
    context="[EVIDENCE LEDGER]\n"+json.dumps(rows)
    retrieval=RetrievedContext(
        question_id="q",variant="hierarchical_state_graph_v2",
        summary_node_ids=[],leaf_node_ids=[],edge_count=0,context_text=context,
        answer_session_hit=False,retrieved_session_ids=[],latency_sec=0.0,
    )
    answer,trace=apply_answer_constraint("Which Porsche purchase happened first?",retrieval,"dog bed")
    assert "not enough information" in answer.casefold()
    assert trace["operator"]=="exact_entity_check"

    rows.pop()
    retrieval.context_text="[EVIDENCE LEDGER]\n"+json.dumps(rows)
    answer,trace=apply_answer_constraint("Which purchase happened first?",retrieval,"dog bed")
    assert answer=="training pads"
    assert trace["operator"]=="generic_calculation"


def test_blind4_source_anchored_between_dates_use_explicit_and_relative_dates():
    start=leaf("q:s:leaf:0",text="User: I started working with Rachel on 2/15.")
    start.session_id="s";start.session_date="2022-03-02"
    house=leaf("q:h:leaf:0",text="User: I saw a house that I really love on 3/1.")
    house.session_id="h";house.session_date="2022-03-02"
    start_fact=fact("q:s:fact:0","2022-02-15",predicate="started working with Rachel",kind="event",op="complete",when=None)
    start_fact.session_id="s";start_fact.source_leaf_ids=[start.node_id]
    house_fact=fact("q:h:fact:0","2022-03-01",predicate="found house loved",kind="event",op="complete",when=None)
    house_fact.session_id="h";house_fact.source_leaf_ids=[house.node_id]
    house_background=fact("q:h:fact:1","Rachel",predicate="works with agent",kind="state",when=None)
    house_background.session_id="h";house_background.source_leaf_ids=[house.node_id]
    fan=leaf("q:d1:leaf:0",text="User: Rachel is a huge Billie Eilish fan.")
    fan.session_id="d1";fan.session_date="2022-03-02"
    fan_fact=fact("q:d1:fact:0","Billie Eilish",predicate="Rachel is fan of",kind="state",when=None)
    fan_fact.session_id="d1";fan_fact.source_leaf_ids=[fan.node_id]
    center=leaf("q:d2:leaf:0",text="User: I plan to find an Irish cultural center.")
    center.session_id="d2";center.session_date="2022-03-02"
    center_fact=fact("q:d2:fact:0","find Irish cultural center",predicate="plan to",kind="event",when=None)
    center_fact.session_id="d2";center_fact.source_leaf_ids=[center.node_id]
    elapsed=_temporal_calculation_result(
        "How many days did it take between starting to work with Rachel and finding a house I loved?",
        [fan_fact,center_fact,start_fact,house_background,house_fact],
        [fan,center,start,house],"2022-03-02",
    )
    assert elapsed and elapsed[1]["formatted_result"]=="14 days"

    feedback=leaf("q:f:leaf:0",text="User: I received feedback that my car suspension was too soft.")
    feedback.session_id="f";feedback.session_date="2023-03-17"
    testing=leaf("q:x:leaf:0",text="User: Tomorrow I will test my new suspension setup at the track.")
    testing.session_id="x";testing.session_date="2023-04-23"
    feedback_fact=fact("q:f:fact:0","suspension too soft",predicate="received feedback",kind="event",op="complete",when="2023-03-17")
    feedback_fact.session_id="f";feedback_fact.source_leaf_ids=[feedback.node_id]
    testing_fact=fact("q:x:fact:0","new suspension setup",predicate="tested",kind="event",op="complete",when=None)
    testing_fact.session_id="x";testing_fact.source_leaf_ids=[testing.node_id]
    elapsed=_temporal_calculation_result(
        "How many days passed between the day I received feedback about my car's suspension and the day I tested my new suspension setup?",
        [feedback_fact,testing_fact],[feedback,testing],"2023-06-01",
    )
    assert elapsed and elapsed[1]["formatted_result"]=="38 days"
    assert elapsed[1]["right_basis"]=="source_relative_tomorrow"


def test_blind4_current_storage_prefers_latest_source_location():
    old=leaf("q:o:leaf:0",text="User: I've been keeping my old sneakers under my bed for storage.")
    old.session_id="o";old.session_date="2023-05-25"
    new=leaf("q:n:leaf:0",text="User: I need to organize my closet and I'm looking forward to storing my old sneakers in a shoe rack in it.")
    new.session_id="n";new.session_date="2023-05-29"
    old_fact=fact("q:o:fact:0","old sneakers under bed",predicate="has",kind="state")
    old_fact.session_id="o";old_fact.source_leaf_ids=[old.node_id]
    new_fact=fact("q:n:fact:0","old sneakers in shoe rack",predicate="storage update",kind="state")
    new_fact.session_id="n";new_fact.source_leaf_ids=[new.node_id]
    result=_operator_result(
        "current/update",[old_fact,new_fact],[],
        "Where do I currently keep my old sneakers?",[old,new],
    )
    assert result and result[0]=="current_storage_location"
    assert result[1]["value"]=="in a shoe rack in my closet"


def test_blind4_assistant_count_binds_teams_and_location_window():
    source=leaf(
        "q:nfl:leaf:0",
        text=("User: How many times have the Chiefs played the Jaguars?\n"
              "Assistant: The Chiefs and Jaguars have played 23 games. "
              "Of those, 12 games were played at Arrowhead Stadium in Kansas City. "
              "The 49ers and Cowboys have played 28 games."),
    );source.session_id="nfl"
    answer=fact("q:nfl:fact:0","NFL historical counts",predicate="answered",kind="assistant_fact")
    answer.role="assistant";answer.session_id="nfl";answer.source_leaf_ids=[source.node_id]
    result=_assistant_recall_result(
        "Looking back at our previous chat, how many times did the Chiefs play the Jaguars at Arrowhead Stadium?",
        [answer],[source],
    )
    assert result and result[1]["value"]=="12"


def test_blind4_exact_role_rejects_partial_title_match():
    source=leaf(
        "q:r:leaf:0",
        text="User: I just started my new role as Senior Software Engineer and lead four engineers.",
    );source.session_id="r"
    role=fact("q:r:fact:0","Senior Software Engineer",predicate="role",kind="state")
    role.session_id="r";role.source_leaf_ids=[source.node_id]
    result=_exact_entity_result(
        "How many engineers do I lead when I just started my new role as Software Engineer Manager?",
        [role],[source],
    )
    assert result and result[1]["exact_match"] is False
    assert result[1]["requested_entity"]=="Software Engineer Manager"
    assert result[1]["partial_entity"]=="Senior Software Engineer"


def test_blind5_podcast_episode_sum_ignores_contraction_apostrophe():
    built=leaf(
        "q:b:leaf:0",
        text='User: I have finished around 15 episodes so far of "How I Built This".',
    );built.session_id="b"
    murder=leaf(
        "q:m:leaf:0",
        text='User: I just finished episode 12 of the "My Favorite Murder" podcast.',
    );murder.session_id="m"
    built_fact=fact("q:b:fact:0","15 episodes",predicate="finished podcast episodes",kind="quantity")
    built_fact.session_id="b";built_fact.source_leaf_ids=[built.node_id]
    murder_fact=fact("q:m:fact:0","episode 12",predicate="finished podcast episode",kind="quantity")
    murder_fact.session_id="m";murder_fact.source_leaf_ids=[murder.node_id]
    result=_arithmetic_result(
        "What is the total number of episodes I've listened to from 'How I Built This' and 'My Favorite Murder'?",
        [built_fact,murder_fact],[built,murder],
    )
    assert result and result[1]["calculation_type"]=="podcast_episode_sum"
    assert result[1]["formatted_result"]=="27"
    assert set(result[2])=={built_fact.node_id,murder_fact.node_id}


def test_blind5_campaign_reach_sum_uses_reach_predicate_not_other_facebook_metric():
    fb=leaf("q:f:leaf:0",text="User: My previous Facebook ad reached 2,000 people.");fb.session_id="f"
    clicks=leaf("q:c:leaf:0",text="User: My Facebook ad got 50 clicks.");clicks.session_id="c"
    influencer=leaf("q:i:leaf:0",text="User: The Instagram influencer has 10,000 followers.");influencer.session_id="i"
    fb_fact=fact("q:f:fact:0","2,000",predicate="previous_facebook_ad_reached",kind="quantity")
    fb_fact.session_id="f";fb_fact.source_leaf_ids=[fb.node_id]
    click_fact=fact("q:c:fact:0","50",predicate="facebook_ad_clicks",kind="quantity")
    click_fact.session_id="c";click_fact.source_leaf_ids=[clicks.node_id]
    influencer_fact=fact("q:i:fact:0","10,000 followers",predicate="audience size",kind="quantity")
    influencer_fact.session_id="i";influencer_fact.source_leaf_ids=[influencer.node_id]
    result=_arithmetic_result(
        "What was the total number of people reached by my Facebook ad campaign and Instagram influencer collaboration?",
        [click_fact,fb_fact,influencer_fact],[clicks,fb,influencer],
    )
    assert result and result[1]["formatted_result"]=="12,000"
    assert set(result[2])=={fb_fact.node_id,influencer_fact.node_id}


def test_blind5_latest_record_and_previous_airline_status_follow_source_chronology():
    old=leaf("q:o:leaf:0",text="User: Our volleyball league record is 3-2.");old.session_id="o";old.session_date="2023-05-01"
    new=leaf("q:n:leaf:0",text="User: We won again and our volleyball record is now 5-2.");new.session_id="n";new.session_date="2023-05-08"
    old_fact=fact("q:o:fact:0","3-2",predicate="volleyball record",kind="state",when="2023-05-01")
    old_fact.session_id="o";old_fact.source_leaf_ids=[old.node_id]
    new_fact=fact("q:n:fact:0","5-2",predicate="volleyball record",kind="state",when="2023-05-08")
    new_fact.session_id="n";new_fact.source_leaf_ids=[new.node_id]
    record=_current_competitive_record_result(
        "What is my volleyball league's current record?",[old_fact,new_fact],[old,new],
    )
    assert record and record[1]["value"]=="5-2" and record[2]==[new_fact.node_id]

    silver=leaf("q:s:leaf:0",text="User: I had United Premier Silver status.");silver.session_id="s";silver.session_date="2023-03-01"
    gold=leaf("q:g:leaf:0",text="User: I now have United Premier Gold status.");gold.session_id="g";gold.session_date="2023-04-01"
    silver_fact=fact("q:s:fact:0","Premier Silver",predicate="United status",when="2023-03-01")
    silver_fact.session_id="s";silver_fact.source_leaf_ids=[silver.node_id];silver_fact.observed_at="2023-03-01"
    gold_fact=fact("q:g:fact:0","Premier Gold",predicate="United status",when="2023-04-01")
    gold_fact.session_id="g";gold_fact.source_leaf_ids=[gold.node_id];gold_fact.observed_at="2023-04-01"
    status=_previous_status_result(
        "What was my previous United status before my current status?",
        [gold_fact,silver_fact],[gold,silver],
    )
    assert status and status[1]["value"]=="Premier Silver"
    assert status[1]["current_value"]=="Premier Gold"


def test_blind5_relative_event_companion_uses_nearest_target_event():
    museum=leaf("q:m:leaf:0",text="User: I visited the science museum and explored it by myself.")
    museum.session_id="m";museum.session_date="2023-03-01"
    lecture=leaf("q:l:leaf:0",text="User: I attended a lecture with my friend.")
    lecture.session_id="l";lecture.session_date="2023-04-01"
    museum_fact=fact("q:m:fact:0","science museum",predicate="visited",kind="event",op="complete",when="2023-03-01")
    museum_fact.session_id="m";museum_fact.source_leaf_ids=[museum.node_id]
    lecture_fact=fact("q:l:fact:0","lecture with friend",predicate="attended",kind="event",op="complete",when="2023-04-01")
    lecture_fact.session_id="l";lecture_fact.source_leaf_ids=[lecture.node_id]
    result=_event_companion_result(
        "Did I visit the science museum with a friend two months ago or not?",
        [museum_fact,lecture_fact],[museum,lecture],"2023-05-01",
    )
    assert result and result[1]["accompanied"] is False
    assert result[2]==[museum_fact.node_id]


def test_blind5_assistant_dish_recall_binds_fruit_snapper_list_item():
    source=leaf(
        "q:d:leaf:0",
        text=("User: Suggest snapper dishes.\nAssistant: 1. Jamaican Escovitch Snapper - spicy peppers and vinegar\n"
              "2. Grilled Snapper with Mango Salsa - mango, lime, and cilantro"),
    );source.session_id="d"
    answer=fact("q:d:fact:0","snapper dish recommendations",predicate="answered",kind="assistant_fact")
    answer.role="assistant";answer.session_id="d";answer.source_leaf_ids=[source.node_id]
    result=_assistant_recall_result(
        "What was the name of the snapper dish with fruit you recommended in our previous conversation?",
        [answer],[source],
    )
    assert result and result[1]["value"]=="Grilled Snapper with Mango Salsa"
    assert result[2]==[answer.node_id]


def test_blind5_poster_relation_slot_mismatch_and_documentary_focus():
    poster=leaf("q:p:leaf:0",text="User: I presented my thesis research poster at Stanford University.")
    poster.session_id="p"
    attendance=leaf("q:h:leaf:0",text="User: I attended Harvard University for undergrad.")
    attendance.session_id="h"
    poster_fact=fact("q:p:fact:0","Stanford University",predicate="presented thesis poster at",kind="event",op="complete")
    poster_fact.session_id="p";poster_fact.source_leaf_ids=[poster.node_id]
    attendance_fact=fact("q:h:fact:0","Harvard University",predicate="attended for undergrad",kind="state")
    attendance_fact.session_id="h";attendance_fact.source_leaf_ids=[attendance.node_id]
    mismatch=_exact_entity_result(
        "At which university did I present a poster for my undergrad course research project?",
        [poster_fact,attendance_fact],[poster,attendance],
    )
    assert mismatch and mismatch[1]["exact_match"] is False
    assert mismatch[1]["entity_type"]=="event_relation_slot"

    focus=_preference_focus_instruction(
        "What documentaries would you recommend that I watch next?",
        ["I enjoyed Our Planet and Free Solo.","I also liked Tiger King."],
    )
    assert focus and "do not merely repeat viewing history" in focus


def _dated_source_fact(node_id, source, obj, predicate, observed, *, op="set", modality="asserted"):
    value=fact(node_id,obj,predicate=predicate,when=observed,op=op,modality=modality)
    value.session_id=source.session_id
    value.source_leaf_ids=[source.node_id]
    value.observed_at=observed
    return value


def test_blind6_elapsed_days_classifies_temporal_before_generic_count():
    assert query_kind("How many days had passed since I finished one book when I started another?")=="temporal"
    assert query_kind("How many days per week do I practice?")=="count/list"


def test_blind6_current_subscription_set_and_distinct_tank_assets():
    sources=[];facts=[]
    for index,(text,predicate,obj,op) in enumerate((
        ("User: I canceled my Forbes subscription.","canceled subscription","Forbes","remove"),
        ("User: I subscribed to The New Yorker.","subscribed to","The New Yorker","add"),
        ("User: I subscribed to Architectural Digest.","subscribed to","Architectural Digest","add"),
    )):
        source=leaf(f"q:s{index}:leaf:0",text=text);source.session_id=f"s{index}"
        value=_dated_source_fact(f"q:s{index}:fact:0",source,obj,predicate,f"2023-05-{20+index:02d}",op=op)
        value.observation_order=index
        sources.append(source);facts.append(value)
    result=_arithmetic_result(
        "How many magazine subscriptions do I currently have?",facts,sources,
    )
    assert result and result[1]["formatted_result"]=="2"

    tank_sources=[];tank_facts=[]
    for index,text in enumerate((
        "User: I have a 1-gallon fish tank.",
        "User: My old 5-gallon tank is at my cousin's house.",
        "User: I maintain a 20-gallon community tank.",
        "User: I cleaned my 20-gallon community tank again.",
    )):
        source=leaf(f"q:t{index}:leaf:0",text=text);source.session_id=f"t{index}"
        value=_dated_source_fact(f"q:t{index}:fact:0",source,text,"has tank",f"2023-06-{index+1:02d}")
        tank_sources.append(source);tank_facts.append(value)
    result=_arithmetic_result("How many tanks do I currently have?",tank_facts,tank_sources)
    assert result and result[1]["formatted_result"]=="3"


def test_blind6_latest_scoped_quantities_and_latest_owned_lens():
    old=leaf("q:o:leaf:0",text="User: My to-watch list has 20 movies.");old.session_id="o"
    new=leaf("q:n:leaf:0",text="User: My to-watch list now has 25 movies.");new.session_id="n"
    old_fact=_dated_source_fact("q:o:fact:0",old,"20","to-watch list size","2023-05-20")
    new_fact=_dated_source_fact("q:n:fact:0",new,"25","to-watch list size","2023-05-22")
    result=_arithmetic_result("How many movies are currently on my to-watch list?",[new_fact,old_fact],[old,new])
    assert result and result[1]["formatted_result"]=="25"

    fifty=leaf("q:f:leaf:0",text="User: I recently got a 50mm prime lens.");fifty.session_id="f"
    zoom=leaf("q:z:leaf:0",text="User: I now have a 70-200mm zoom lens.");zoom.session_id="z"
    planned=leaf("q:p:leaf:0",text="User: I am considering a wide-angle lens.");planned.session_id="p"
    fifty_fact=_dated_source_fact("q:f:fact:0",fifty,"50mm prime lens","has_lens","2023-03-11")
    zoom_fact=_dated_source_fact("q:z:fact:0",zoom,"70-200mm zoom lens","has_lens","2023-08-30")
    planned_fact=_dated_source_fact("q:p:fact:0",planned,"wide-angle lens","considering lens","2023-09-01",modality="planned")
    result=_arithmetic_result("What is the most recent camera lens I have?",[fifty_fact,planned_fact,zoom_fact],[fifty,zoom,planned])
    assert result and "70-200mm" in result[1]["formatted_result"]


def test_blind6_minimum_value_and_period_baseline_delta():
    necklace=leaf("q:n:leaf:0",text="User: My vintage diamond necklace is worth $5,000.");necklace.session_id="n"
    vanity=leaf("q:v:leaf:0",text="User: The restored vanity should be worth at least $150.");vanity.session_id="v"
    necklace_fact=_dated_source_fact("q:n:fact:0",necklace,"$5,000","necklace worth","2023-01-01")
    vanity_fact=_dated_source_fact("q:v:fact:0",vanity,"at least $150","vanity worth","2023-01-02")
    result=_arithmetic_result("What is the minimum I could get if I sold my necklace and vanity?",[necklace_fact,vanity_fact],[necklace,vanity])
    assert result and result[1]["formatted_result"]=="$5,150"

    after=leaf("q:a:leaf:0",text="User: After two weeks, I reached 350 Instagram followers.");after.session_id="a"
    start=leaf("q:s:leaf:0",text="User: I started the year with 250 Instagram followers.");start.session_id="s"
    after_fact=_dated_source_fact("q:a:fact:0",after,"350","instagram follower count","2023-05-23")
    start_fact=_dated_source_fact("q:s:fact:0",start,"250","instagram follower baseline","2023-05-28")
    result=_arithmetic_result("Approximately how much did my Instagram followers increase after two weeks?",[start_fact,after_fact],[after,start])
    assert result and result[1]["formatted_result"]=="100"


def test_blind6_weddings_are_deduplicated_by_couple():
    texts=(
        "User: I attended Rachel and Mike's wedding and Emily and Sarah's wedding.",
        "User: I attended Emily and Sarah's wedding in the city.",
        "User: I went to the wedding of Jen and Tom.",
    )
    sources=[];facts=[]
    for index,text in enumerate(texts):
        source=leaf(f"q:w{index}:leaf:0",text=text);source.session_id=f"w{index}"
        value=_dated_source_fact(f"q:w{index}:fact:0",source,"wedding","attended wedding",f"2023-0{index+1}-01",op="complete")
        sources.append(source);facts.append(value)
    result=_arithmetic_result("How many weddings have I attended this year and which couples?",facts,sources)
    assert result and result[1]["formatted_result"]=="3"
    assert len(result[1]["items"])==3


def test_blind6_temporal_explicit_dates_recurring_latest_and_relative_years():
    walk=leaf("q:w:leaf:0",text="User: I completed the Walk for Hunger on February 21.");walk.session_id="w";walk.session_date="2023-03-14"
    cleanup=leaf("q:c:leaf:0",text="User: I participated in Coastal Cleanup on March 7.");cleanup.session_id="c";cleanup.session_date="2023-03-14"
    walk_fact=_dated_source_fact("q:w:fact:0",walk,"Walk for Hunger","completed","2023-03-14",op="complete")
    cleanup_fact=_dated_source_fact("q:c:fact:0",cleanup,"Coastal Cleanup","completed","2023-03-14",op="complete")
    elapsed=_temporal_calculation_result("How many days passed between the Walk for Hunger and Coastal Cleanup?",[walk_fact,cleanup_fact],[walk,cleanup],"2023-03-20")
    assert elapsed and elapsed[1]["formatted_result"]=="14 days"

    one=leaf("q:1:leaf:0",text="User: I spend 1 hour coding daily.");one.session_id="1";one.session_date="2023-05-20"
    two=leaf("q:2:leaf:0",text="User: I dedicate 2 hours each day to coding exercises.");two.session_id="2";two.session_date="2023-05-29"
    one_fact=_dated_source_fact("q:1:fact:0",one,"1 hour","daily coding duration","2023-05-20")
    two_fact=_dated_source_fact("q:2:fact:0",two,"2 hours","daily coding duration","2023-05-29")
    duration=_temporal_calculation_result("How much time do I spend coding each day?",[one_fact,two_fact],[one,two],"2023-06-01")
    assert duration and duration[1]["formatted_result"]=="2 hours"

    europe=leaf("q:e:leaf:0",text="User: I traveled through Europe solo last summer.");europe.session_id="e";europe.session_date="2023-05-22"
    southwest=leaf("q:s:leaf:0",text="User: I took a family road trip through the Southwest a few years ago.");southwest.session_id="s";southwest.session_date="2023-05-22"
    ef=_dated_source_fact("q:e:fact:0",europe,"Europe solo trip","completed trip","2023-05-22",op="complete")
    sf=_dated_source_fact("q:s:fact:0",southwest,"Southwest family road trip","completed trip","2023-05-22",op="complete")
    comparison=_event_comparison_result("Which trip happened first, my Europe solo trip or the Southwest family road trip?",[ef,sf],[europe,southwest])
    assert comparison and "southwest" in comparison[1]["value"].casefold()


def test_blind6_complete_l0_assistant_budget_and_exact_collection_mismatch():
    source=leaf("q:d:leaf:0",text="User: Make a DHL campaign plan.\nAssistant: Budget:\n* Influencer marketing: $2,000\n* Creative: $500")
    source.session_id="d"
    answer=fact("q:d:fact:0","campaign plan",predicate="provided",kind="assistant_fact")
    answer.role="assistant";answer.session_id="d";answer.source_leaf_ids=[source.node_id]
    result=_assistant_recall_result("In the previous chat, how much was allocated to influencer marketing for the DHL campaign?",[answer],[source])
    assert result and result[1]["value"]=="$2,000"

    cameras=leaf("q:c:leaf:0",text="User: I have been collecting vintage cameras for 3 months.");cameras.session_id="c"
    camera_fact=_dated_source_fact("q:c:fact:0",cameras,"vintage cameras","collecting","2023-01-01")
    mismatch=_exact_entity_result("How long have I been collecting vintage films?",[camera_fact],[cameras])
    assert mismatch and mismatch[1]["exact_match"] is False
    assert "cameras" in mismatch[1]["partial_entity"]


def test_blind6_theme_park_focus_preserves_all_requested_constraints():
    focus=_preference_focus_instruction(
        "What theme parks should I visit next for thrill rides, special events, unique food, and nighttime shows?",
        ["I have visited four parks during seasonal events."],
    )
    assert focus
    for expected in ("thrill rides","special or seasonal events","distinctive food","nighttime shows"):
        assert expected in focus


def test_blind7_label_bound_deltas_and_cross_entity_percentages():
    old_mpg=leaf("q:o:leaf:0",text="User: A few months ago my car was getting 30 MPG.");old_mpg.session_id="o"
    new_mpg=leaf("q:n:leaf:0",text="User: Lately it has been getting 28 MPG.");new_mpg.session_id="n"
    old_fact=_dated_source_fact("q:o:fact:0",old_mpg,"30 MPG","fuel economy","2023-03-01")
    new_fact=_dated_source_fact("q:n:fact:0",new_mpg,"28 MPG","fuel economy","2023-05-01")
    result=_arithmetic_result(
        "How much more miles per gallon was my car getting a few months ago compared to now?",
        [old_fact,new_fact],[old_mpg,new_mpg],
    )
    assert result and result[1]["formatted_result"]=="2 MPG"

    property_source=leaf("q:p:leaf:0",text="User: The 5-acre countryside property is listed at $200,000.");property_source.session_id="p"
    renovation_source=leaf("q:r:leaf:0",text="User: Renovations to my current house will cost $20,000.");renovation_source.session_id="r"
    property_fact=_dated_source_fact("q:p:fact:0",property_source,"$200,000","property price","2023-05-01")
    renovation_fact=_dated_source_fact("q:r:fact:0",renovation_source,"$20,000","renovation cost","2023-05-02")
    ratio=_arithmetic_result(
        "What percentage of the countryside property's price is the cost of the renovations I plan to do on my current house?",
        [renovation_fact,property_fact],[renovation_source,property_source],
    )
    assert ratio and ratio[1]["formatted_result"]=="10%"

    quote=leaf("q:q:leaf:0",text="User: They initially quoted me $2,500, then corrected the final price to $2,800.");quote.session_id="q"
    quote_fact=_dated_source_fact("q:q:fact:0",quote,"quote revision","trip price","2023-05-03")
    delta=_arithmetic_result(
        "How much more did I have to pay for the trip after the initial quote?",
        [quote_fact],[quote],
    )
    assert delta and delta[1]["formatted_result"]=="$300"


def test_blind7_lossless_l0_ratio_relative_week_and_related_count():
    worn=leaf("q:w:leaf:0",text="User: On my last trip I ended up only wearing two - my sneakers and sandals.");worn.session_id="w"
    packed=leaf("q:p:leaf:0",text="User: Since I packed 5 pairs of shoes, space was tight.");packed.session_id="p"
    worn_fact=_dated_source_fact("q:w:fact:0",worn,"two pairs","wore shoes","2023-05-02")
    packed_anchor=_dated_source_fact("q:p:fact:0",packed,"packing note","trip packing","2023-05-01")
    ratio=_arithmetic_result(
        "What percentage of packed shoes did I wear on my last trip?",
        [worn_fact,packed_anchor],[worn,packed],
    )
    assert ratio and ratio[1]["formatted_result"]=="40%"

    market=leaf("q:m:leaf:0",text="User: I attended the Holiday Market a week before Black Friday.");market.session_id="m"
    phone=leaf("q:i:leaf:0",text="User: I got my iPhone 13 Pro from Best Buy on Black Friday.");phone.session_id="i"
    market_fact=_dated_source_fact("q:m:fact:0",market,"Holiday Market","attended","2023-11-17",op="complete")
    phone_fact=_dated_source_fact("q:i:fact:0",phone,"iPhone 13 Pro","bought","2023-11-24",op="complete")
    elapsed=_arithmetic_result(
        "How many days before I bought the iPhone 13 Pro did I attend the Holiday Market?",
        [market_fact,phone_fact],[market,phone],
    )
    assert elapsed and elapsed[1]["formatted_result"]=="7 days"

    poster=leaf("q:a:leaf:0",text="User: I have a signed poster from my favorite artist's debut album, which is a limited edition of only 500 copies worldwide.");poster.session_id="a"
    poster_fact=_dated_source_fact("q:a:fact:0",poster,"signed poster","owns collectible","2023-05-01")
    copies=_arithmetic_result(
        "How many copies of my favorite artist's debut album were released worldwide?",
        [poster_fact],[poster],
    )
    assert copies and copies[1]["formatted_result"]=="500"


def test_blind7_exact_destination_and_completed_item_mismatches():
    airbnb=leaf("q:a:leaf:0",text="User: I stayed at an Airbnb in San Francisco and booked it three months in advance.");airbnb.session_id="a"
    airbnb_fact=_dated_source_fact("q:a:fact:0",airbnb,"San Francisco Airbnb","booked lodging","2023-03-01")
    mismatch=_exact_entity_result("When did I book the Airbnb in Sacramento?",[airbnb_fact],[airbnb])
    assert mismatch and mismatch[1]["exact_match"] is False
    assert mismatch[1]["entity_type"]=="booking_destination"

    cake=leaf("q:c:leaf:0",text="User: I baked a chocolate cake for my sister's birthday.");cake.session_id="c"
    cake_fact=_dated_source_fact("q:c:fact:0",cake,"chocolate cake","baked","2023-05-01",op="complete")
    mismatch=_exact_entity_result("How many times did I bake egg tarts in the past two weeks?",[cake_fact],[cake])
    assert mismatch and mismatch[1]["exact_match"] is False
    assert mismatch[1]["requested_entity"]=="egg tarts"


def test_blind7_latest_family_trip_relative_weekend_game_and_named_journal():
    hawaii=leaf("q:h:leaf:0",text="User: My recent family trip was to Hawaii.");hawaii.session_id="h";hawaii.session_date="2023-05-26"
    paris=leaf("q:p:leaf:0",text="User: My recent trip was to Paris with family.");paris.session_id="p";paris.session_date="2023-05-28"
    hf=_dated_source_fact("q:h:fact:0",hawaii,"Hawaii","recent_family_trip","2023-05-26",op="complete")
    pf=_dated_source_fact("q:p:fact:0",paris,"Paris with family","recent_trip","2023-05-28",op="complete")
    latest=_arithmetic_result("Where did I go on my most recent family trip?",[hf,pf],[hawaii,paris])
    assert latest and latest[1]["formatted_result"]=="Paris"

    game=leaf("q:g:leaf:0",text="User: I finally beat that last boss in the Dark Souls 3 DLC last weekend.")
    game.session_id="g";game.session_date="2023-05-26"
    gf=_dated_source_fact("q:g:fact:0",game,"Dark Souls 3 DLC","beat game","2023-05-26",op="complete")
    target=_target_date_answer_result(
        "What game did I finally beat last weekend?",[gf],[game],"2023-06-01",
    )
    assert target and target[1]["value"]=="Dark Souls 3 DLC"

    source=leaf(
        "q:s:leaf:0",
        text=("User: Give examples.\nAssistant: A study in Alternative Therapies involved 15 subjects. "
              "Another study published in the journal Music and Medicine involved 38 subjects "
              "and found significant reductions in depression, anxiety, and stress."),
    );source.session_id="s"
    answer=fact("q:s:fact:0","study examples",predicate="answered",kind="assistant_fact")
    answer.role="assistant";answer.session_id="s";answer.source_leaf_ids=[source.node_id]
    recalled=_assistant_recall_result(
        "In our previous conversation, how many subjects were in the study published in the journal Music and Medicine that found significant reductions?",
        [answer],[source],
    )
    assert recalled and recalled[1]["value"]=="38"


def test_blind8_lossless_attributes_and_completed_graduation_count():
    dog=leaf(
        "q:d:leaf:0",
        text="User: What collar type would suit a Golden Retriever like Max?",
    );dog.session_id="d"
    dog_fact=_dated_source_fact("q:d:fact:0",dog,"Max","owns dog","2023-05-25")
    breed=_arithmetic_result("What breed is my dog?",[dog_fact],[dog])
    assert breed and breed[1]["formatted_result"]=="Golden Retriever"

    necklace=leaf(
        "q:n:leaf:0",
        text="User: My grandma gave me the silver necklace on my 18th birthday.",
    );necklace.session_id="n"
    necklace_fact=_dated_source_fact("q:n:fact:0",necklace,"silver necklace","received gift","2023-05-22")
    age=_arithmetic_result(
        "How old was I when my grandma gave me the silver necklace?",
        [necklace_fact],[necklace],
    )
    assert age and age[1]["formatted_result"]=="18"

    texts=(
        "User: I just attended my little cousin Emma's preschool graduation about two months ago.",
        "User: I just attended my colleague Alex's graduation from a leadership development program a few weeks ago.",
        "User: I just attended my best friend Rachel's master's degree graduation ceremony a couple of weeks ago.",
        "User: I feel guilty about missing my nephew Jack's graduation ceremony last month.",
    )
    sources=[];facts=[]
    for index,text in enumerate(texts):
        source=leaf(f"q:g{index}:leaf:0",text=text);source.session_id=f"g{index}"
        value=_dated_source_fact(
            f"q:g{index}:fact:0",source,"graduation",
            "attended graduation" if index<3 else "missed graduation",
            f"2023-07-{index+1:02d}",op="complete",
        )
        sources.append(source);facts.append(value)
    count=_arithmetic_result(
        "How many graduation ceremonies have I attended in the past three months?",
        facts,sources,
    )
    assert count and count[1]["formatted_result"]=="3"


def test_blind8_relative_art_event_routes_to_nearest_attended_exhibit():
    city=leaf(
        "q:c:leaf:0",
        text='User: I attended the "Impressionist Masterpieces" exhibition at the City Art Museum on a Saturday.',
    );city.session_id="c";city.session_date="2023-01-14"
    met=leaf(
        "q:m:leaf:0",
        text='User: I attended the "Ancient Civilizations" exhibit at the Metropolitan Museum of Art today.',
    );met.session_id="m";met.session_date="2023-01-15"
    city_fact=_dated_source_fact("q:c:fact:0",city,"City Art Museum","attended exhibit","2023-01-14",op="complete")
    met_fact=_dated_source_fact("q:m:fact:0",met,"Metropolitan Museum of Art","attended exhibit","2023-01-15",op="complete")
    result=_target_date_answer_result(
        "I mentioned that I participated in an art-related event two weeks ago. Where was that event held at?",
        [city_fact,met_fact],[city,met],"2023-02-01",
    )
    assert result and result[1]["value"]=="the Metropolitan Museum of Art"
