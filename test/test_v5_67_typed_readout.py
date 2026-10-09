from __future__ import annotations

from graphmem.answer.typed_readout import (
    extract_evidence_blocks,
    make_audit_messages,
    make_answer_messages,
    make_readout_messages,
    parse_json_object,
    route_question,
    select_compact_evidence,
)


MESSAGES = [
    {"role": "system", "content": "old answer rules"},
    {"role": "user", "content": (
        "Question: How many dogs did Maria adopt?\n\n"
        "Conversation memories:\n"
        "[AUX 1 rank=3] [s1 @ 2023-01-01] Maria: I plan to adopt a dog.\n"
        "[GRAPH 1 step=0] [s2 @ 2023-02-01] Maria: I adopted Luna.\n"
        "[CHAIN 1 support] [s3 @ 2023-03-01] Maria: I adopted Max.\n\n"
        "Answer the original Question now: How many dogs did Maria adopt?")},
]


def test_extracts_only_numbered_evidence_blocks() -> None:
    blocks = extract_evidence_blocks(MESSAGES)
    assert len(blocks) == 3
    assert blocks[0].startswith("[AUX")
    assert "Answer the original" not in blocks[-1]


def test_routes_count_before_multihop_metadata() -> None:
    assert route_question(
        "How many dogs did Maria adopt?", "locomo_cat1") == "count"
    assert route_question("When did Maria adopt Luna?", "locomo_cat2") == "temporal"
    assert route_question("What job might Maria pursue?", "locomo_cat3") == "inference"
    assert route_question("What color was the car?", "locomo_cat4") == "lookup"


def test_citations_and_lexical_fallback_build_compact_source_view() -> None:
    blocks = extract_evidence_blocks(MESSAGES)
    readout = parse_json_object(
        '{"witnesses":[{"evidence_id":"E03"}],"missing_slots":[]}')
    selected = select_compact_evidence(
        question="How many dogs did Maria adopt?", readout=readout,
        evidence_blocks=blocks, max_blocks=2,
    )
    assert selected[0][0] == 3
    assert len(selected) == 2


def test_prompts_mark_baseline_as_fallible_and_never_add_gold() -> None:
    blocks = extract_evidence_blocks(MESSAGES)
    readout_messages = make_readout_messages(
        question="How many dogs did Maria adopt?", question_date="2023-04-01",
        route="count", evidence_blocks=blocks,
    )
    answer_messages = make_answer_messages(
        question="How many dogs did Maria adopt?", question_date="2023-04-01",
        route="count", baseline="one", readout_text='{"result":"two"}',
        compact_evidence=[(2, blocks[1]), (3, blocks[2])],
    )
    assert "Fallible baseline answer" in answer_messages[-1]["content"]
    assert "gold" not in "\n".join(
        row["content"] for row in readout_messages + answer_messages).casefold()


def test_answer_can_be_derived_without_baseline_anchor() -> None:
    messages = make_answer_messages(
        question="Who adopted Luna?", question_date="2023-04-01",
        route="lookup", baseline=None, readout_text='{"result":"Maria"}',
        compact_evidence=[(1, "[GRAPH 1] Maria adopted Luna.")],
    )
    prompt = messages[-1]["content"]
    assert "No baseline answer is supplied" in prompt
    assert "Fallible baseline answer" not in prompt


def test_adversarial_audit_requires_full_rescan_without_gold() -> None:
    blocks = extract_evidence_blocks(MESSAGES)
    messages = make_audit_messages(
        question="How many dogs did Maria adopt?", question_date="2023-04-01",
        route="count", baseline=None, readout_text='{"confidence":"high"}',
        evidence_blocks=list(enumerate(blocks, 1)),
    )
    prompt = "\n".join(row["content"] for row in messages)
    assert "Rescan every numbered source memory" in prompt
    assert "candidate_ledger" in prompt
    assert "[E03]" in prompt
    assert "gold" not in prompt.casefold()


def test_empty_deterministic_message_has_no_evidence_blocks() -> None:
    assert extract_evidence_blocks([]) == []
