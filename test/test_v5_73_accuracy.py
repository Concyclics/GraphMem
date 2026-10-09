from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from graphmem.answer import (
    AnswerConfig, build_answer_messages, build_aggregation_ledger,
    build_candidate_verifier_messages, build_ensemble_family_messages,
    build_typed_evidence_card, combine_candidate_families,
    cross_family_consensus_choice, parse_verifier_choice, prompt_contract,
)
from graphmem.config import load_runtime_config
from graphmem.domain import CandidateScore, SourceTurn
from graphmem.retrieval.packer import (
    pack_obligation_aware, select_obligation_witnesses,
    source_date_match_score,
)
from graphmem.tokenization import HeuristicTokenCounter


def _turn(turn_id: str, session: str, index: int, text: str,
          speaker: str = "Maria", timestamp: str = "2023-05-08") -> SourceTurn:
    return SourceTurn(
        turn_id, "memory", session, index, speaker, "Caroline", "user",
        timestamp, text, hashlib.sha256(text.encode()).hexdigest())


def _candidate(turn: SourceTurn, score: float, *, operands=(),
               exact: float = 0.0, dense: float = 0.0) -> CandidateScore:
    return CandidateScore(
        turn.turn_id, turn.session_id, exact, 0.0, dense, 0.0, 0.0, 0.0,
        len(turn.raw_text.split()), score, (), operand_ids=tuple(operands))


def test_witness_reserve_recovers_terse_dialogue_answer_below_precision_floor() -> None:
    prompt = _turn(
        "prompt", "s-gold", 0,
        "What are the names of the two dogs you adopted?")
    response = _turn("response", "s-gold", 1, "Luna and Oliver!")
    noise = [
        _turn(f"noise-{index}", f"s-{index}", 0,
              f"We discussed dogs and adoption topic {index}.")
        for index in range(8)
    ]
    rows = tuple(
        [_candidate(turn, 20.0 - index, exact=0.2)
         for index, turn in enumerate(noise)]
        + [_candidate(prompt, 2.0, exact=1.0), _candidate(response, 1.0)])
    turns = {turn.turn_id: turn for turn in (*noise, prompt, response)}
    floor = tuple(turn.turn_id for turn in noise[:4])

    witnesses, trace = select_obligation_witnesses(
        rows, turns, query="What are Maria's dogs' names?",
        answer_kind="lookup", max_witnesses=2, baseline_turn_ids=floor)

    assert "response" in witnesses
    assert trace["dialogue"] >= 1
    packed, _dropped, flags, _units, _tokens = pack_obligation_aware(
        (), rows, turns, query="What are Maria's dogs' names?",
        answer_kind="lookup", max_turns=6, max_tokens=200,
        count_text_tokens=lambda text: len(text.split()),
        baseline_floor=floor, witness_floor=witnesses)
    assert set(floor) <= set(packed)
    assert "response" in packed
    assert flags["witness_floor_packed_count"] >= 1


def test_v573_runtime_enables_only_bounded_witness_reserve() -> None:
    config = load_runtime_config(
        __import__("pathlib").Path("configs/v5/runtime_v5_73_accuracy64.json"))
    assert config.query_budget.max_evidence_turns == 64
    assert config.retrieval.obligation_witness_reserve
    assert config.retrieval.obligation_witness_reserve_turns == 8
    options = config.retrieval.navigator_options()
    assert options["obligation_witness_reserve"] is True


def test_typed_evidence_card_is_source_only_time_aware_and_capped() -> None:
    gold = _turn(
        "gold", "s1", 0,
        "Maria said yesterday that she adopted Luna at the dog shelter.")
    distractor = _turn(
        "other", "s2", 0,
        "Caroline planned to visit a homeless shelter next month.",
        speaker="Caroline")
    rows = (
        _candidate(gold, 4.0, exact=1.0),
        _candidate(distractor, 3.0, dense=0.8),
    )
    config = AnswerConfig.v5_73(typed_evidence_card_max_tokens=160)
    counter = HeuristicTokenCounter()

    card = build_typed_evidence_card(
        "When did Maria adopt Luna?", {"gold": gold, "other": distractor},
        ("gold", "other"), rows, answer_kind="temporal",
        config=config, counter=counter)

    assert card is not None
    assert card.tokens <= 160
    assert "not a proposed answer" in card.text
    assert "source-time" in card.text
    assert "Candidate answer" not in card.text
    assert card.turn_ids[0] == "gold"


def test_bounded_common_knowledge_is_inference_only_and_hashed() -> None:
    base = build_answer_messages(
        question="What console would she likely need for Xenoblade 2?",
        question_date=None, evidence_text="She enjoyed Xenoblade 2.",
        typed_evidence_card="Answer-critical evidence card (source excerpts only; not a proposed answer):\nK1 source=t1: Xenoblade 2",
        bounded_common_knowledge=True)
    lookup = build_answer_messages(
        question="What game did she enjoy?", question_date=None,
        evidence_text="She enjoyed Xenoblade 2.")
    assert "stable ordinary knowledge" in base[0]["content"]
    assert "stable ordinary knowledge" not in lookup[0]["content"]
    assert prompt_contract(typed_evidence_card=True,
                           bounded_common_knowledge=True)[2] != (
        prompt_contract()[2])


def test_event_table_separates_exact_dog_shelter_from_near_match_and_ordinals() -> None:
    dog = _turn(
        "dog", "s1", 0,
        "On May 2, I completed my seventh volunteer shift at the dog shelter.")
    homeless = _turn(
        "homeless", "s2", 0,
        "On May 3, I volunteered at a homeless shelter.")
    planned = _turn(
        "plan", "s3", 0,
        "I plan to volunteer at the dog shelter next Friday.")
    turns = {turn.turn_id: turn for turn in (dog, homeless, planned)}

    ledger = build_aggregation_ledger(
        "How many times did Maria volunteer at the dog shelter?",
        turns, tuple(turns), event_aware=True)

    assert ledger is not None
    assert ledger.schema_version == "graphmem-v5.73-event-ledger-v1"
    assert ledger.max_ordinal == 7
    dog_line = next(line for line in ledger.event_lines if "source=dog" in line)
    homeless_line = next(
        line for line in ledger.event_lines if "source=homeless" in line)
    plan_line = next(line for line in ledger.event_lines if "source=plan" in line)
    assert "scope=exact_scope" in dog_line
    assert "scope=near_match" in homeless_line
    assert "planned_or_hypothetical" in plan_line


def test_event_aware_list_gets_a_typed_ledger_without_changing_legacy() -> None:
    turn = _turn("trip", "s1", 0, "Maria completed a kayaking trip on Monday.")
    question = "Which trips did Maria complete?"
    assert build_aggregation_ledger(
        question, {turn.turn_id: turn}, (turn.turn_id,)) is None
    ledger = build_aggregation_ledger(
        question, {turn.turn_id: turn}, (turn.turn_id,), event_aware=True)
    assert ledger is not None and ledger.operation == "list_distinct"


def test_split_prompt_families_are_distinct_and_verifier_cannot_rewrite() -> None:
    base = build_answer_messages(
        question="When did Maria adopt Luna?", question_date=None,
        evidence_text="[s1] Maria: I adopted Luna yesterday.",
        typed_evidence_card=(
            "Answer-critical evidence card (source excerpts only; not a proposed "
            "answer):\nK1 source=t1: adopted Luna yesterday"))
    extractive = build_ensemble_family_messages(base, "extractive")
    audit = build_ensemble_family_messages(base, "audit")
    assert extractive != audit
    assert "adopted Luna yesterday" in extractive[-1]["content"]
    assert "adopted Luna yesterday" in audit[-1]["content"]
    candidates = (
        {"prediction": "May 7", "family": "extractive", "family_rank": 1},
        {"prediction": "May 8", "family": "audit", "family_rank": 1},
    )
    verifier = build_candidate_verifier_messages(base, candidates)
    assert "must not synthesize" in verifier[0]["content"]
    assert "C1" in verifier[-1]["content"] and "C2" in verifier[-1]["content"]
    assert parse_verifier_choice('{"choice": 2, "reason_code": "correct_time"}', 2) == 1
    assert parse_verifier_choice("I would write May 9", 2) is None


def test_direct_structured_candidate_pair_preserves_four_by_four_provenance() -> None:
    direct = tuple({
        "rank": index, "prediction": f"direct {index}",
        "prediction_sha256": hashlib.sha256(
            f"direct {index}".encode()).hexdigest(),
    } for index in range(1, 6))
    structured = tuple({
        "rank": index, "family": "extractive",
        "prediction": f"structured {index}",
        "prediction_sha256": hashlib.sha256(
            f"structured {index}".encode()).hexdigest(),
    } for index in range(1, 6))

    rows = combine_candidate_families(direct, structured)

    assert len(rows) == 8
    assert [row["family"] for row in rows[:4]] == ["direct_v563"] * 4
    assert [row["family"] for row in rows[4:]] == ["structured_v573"] * 4
    assert [row["rank"] for row in rows] == list(range(1, 9))

    repeated = [dict(rows[0]) for _ in range(8)]
    verifier = build_candidate_verifier_messages(
        build_answer_messages(
            question="What happened?", question_date=None,
            evidence_text="[s1] Maria: A direct event happened."),
        [{**repeated[0], "support_count": 8,
          "support_families": ["direct_v563", "structured_v573"]}],
    )
    assert "support=8" in verifier[-1]["content"]
    assert "direct_v563,structured_v573" in verifier[-1]["content"]

    mixed = [dict(row) for row in rows]
    mixed[4]["prediction"] = mixed[0]["prediction"]
    mixed[4]["prediction_sha256"] = mixed[0]["prediction_sha256"]
    assert cross_family_consensus_choice(mixed) == 0
    assert cross_family_consensus_choice(rows) is None


def test_v563_defaults_remain_off_and_v573_is_opt_in() -> None:
    old = AnswerConfig.v5_63()
    new = AnswerConfig.v5_73()
    assert not old.typed_evidence_card_enabled
    assert not old.aggregation_event_table_enabled
    assert new.typed_evidence_card_enabled
    assert new.aggregation_event_table_enabled
    assert new.typed_evidence_card_max_tokens <= 500
    with pytest.raises(ValueError, match=r"\[64, 500\]"):
        replace(new, typed_evidence_card_max_tokens=501)


def test_witness_reserve_does_not_treat_owner_or_question_mark_as_proof() -> None:
    noise_question = _turn(
        "noise-question", "s-noise", 0,
        "What did you do today?", speaker="Maria")
    noise_response = _turn(
        "noise-response", "s-noise", 1,
        "I reorganized the kitchen.", speaker="Maria")
    topical = _turn(
        "topical", "s-gold", 0,
        "I adopted a dog named Luna from the shelter.", speaker="Maria")
    turns = {turn.turn_id: turn for turn in (
        noise_question, noise_response, topical)}
    rows = tuple(_candidate(turn, 3.0 - index, operands=("owner",))
                 for index, turn in enumerate(turns.values()))

    witnesses, trace = select_obligation_witnesses(
        rows, turns, query="What is Maria's dog's name?",
        answer_kind="lookup", max_witnesses=8)

    # A merely topical owner turn is already handled by the dual-lane rank; it
    # must not consume a hard reserve seat without proof/operator provenance.
    assert "topical" not in witnesses
    assert "noise-response" not in witnesses
    assert trace["operand"] == 0


def test_source_date_witness_recovers_metadata_only_session() -> None:
    dated = _turn(
        "dated", "s-gold", 0, "We watched The Farewell together.",
        timestamp="8:15 pm on 3 June, 2023")
    noise = [
        _turn(f"noise-{index}", f"s-{index}", 0,
              f"We discussed movies and weekend plans {index}.",
              timestamp=f"2023-05-{index + 1:02d}")
        for index in range(8)
    ]
    turns = {turn.turn_id: turn for turn in (*noise, dated)}
    rows = tuple(
        [_candidate(turn, 20.0 - index, exact=0.2)
         for index, turn in enumerate(noise)]
        + [_candidate(dated, 0.1)])
    floor = tuple(turn.turn_id for turn in noise[:4])

    witnesses, trace = select_obligation_witnesses(
        rows, turns, query="What movie did we watch on June 3, 2023?",
        answer_kind="lookup", max_witnesses=4, baseline_turn_ids=floor)

    assert source_date_match_score(
        "What happened on June 3, 2023?", dated.timestamp) == 3.0
    assert source_date_match_score(
        "What happened in May 2023?", "2023-05-08") == 2.0
    assert witnesses[0] == "dated"
    assert trace["source_date"] == 1
