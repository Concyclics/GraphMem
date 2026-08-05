from graphmem_demo.v3.direct_answer import (
    direct_lossless_answer_messages,
    scalar_delta_proposal,
)


def test_direct_answer_prompt_contains_no_planner_candidate_or_gold_metadata() -> None:
    messages = direct_lossless_answer_messages(
        question="How many places did Alex visit?",
        question_date="2026-07-28",
        evidence_text="[EVIDENCE] Alex visited Lisbon.",
    )
    rendered = "\n".join(message["content"] for message in messages)
    assert "Alex visited Lisbon" in rendered
    assert "candidate_answer" not in rendered
    assert "answer_session_ids" not in rendered
    assert "gold" not in rendered.casefold()


def test_direct_answer_exposes_generic_scalar_delta_contract() -> None:
    messages = direct_lossless_answer_messages(
        question="How many credits do I need to earn to reach the target?",
        question_date="2026-07-28",
        evidence_text="Current: 200. Target: 300.",
    )
    rendered = "\n".join(message["content"] for message in messages)
    assert "scalar_delta" in rendered
    assert "target amount - latest current amount" in rendered


def test_scalar_delta_proposal_is_generic_and_evidence_bound() -> None:
    proposal = scalar_delta_proposal(
        "How many credits do I need to earn?",
        "I currently have 200 credits. I need a total of 300 credits.",
    )
    assert proposal is not None
    assert proposal["target"] == 300
    assert proposal["current"] == 200
    assert proposal["proposed_answer"] == 100


def test_scalar_delta_proposal_does_not_run_for_unrelated_count() -> None:
    assert scalar_delta_proposal(
        "How many cities did I visit?",
        "I visited 2 cities and have 100 credits.",
    ) is None
