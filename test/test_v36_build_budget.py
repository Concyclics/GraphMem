from __future__ import annotations

import time

import pytest

import graphmem_demo.pipeline as pipeline
from graphmem_demo.models import DeepSeekCallRecord, QuestionCase
from graphmem_demo.pipeline import DemoConfig, InflightLimiter
from graphmem_demo.v36.runtime import (
    _BUDGET_RESERVE, _CONSOLIDATION_RESERVE, _HARD_SAFETY_RESERVE,
    _REPAIR_RESERVE, _completion_caps, _consolidation_budget_remaining,
    _load_call_checkpoint, _repair_budget_remaining, _repair_positions,
    _save_call_checkpoint,
)


def test_parallel_completion_caps_fit_conservative_hard_envelope() -> None:
    estimates = {
        f"s{index}": 1800 + index * 37 for index in range(47)
    }
    caps = _completion_caps(
        estimates, requested=4096, budget=300_000
    )
    assumed_prompt = sum(
        int(value * 1.05 + 0.999999) for value in estimates.values()
    )
    assert set(caps) == set(estimates)
    assert min(caps.values()) >= 128
    assert max(caps.values()) <= 4096
    assert (
        assumed_prompt + sum(caps.values()) + _BUDGET_RESERVE
        <= 300_000
    )


def test_initial_reserve_is_split_into_spendable_stage_pools() -> None:
    assert _BUDGET_RESERVE == 25_000
    assert (
        _REPAIR_RESERVE + _CONSOLIDATION_RESERVE + _HARD_SAFETY_RESERVE
        == _BUDGET_RESERVE
    )
    assert _repair_budget_remaining(
        build_budget_tokens=300_000, spent=275_000, prompt_cost=2_000,
    ) == 13_000
    assert _consolidation_budget_remaining(
        build_budget_tokens=300_000, spent=288_000, prompt_cost=2_000,
    ) == 5_000


def test_denser_sessions_receive_no_less_output_budget() -> None:
    caps = _completion_caps(
        {"short": 500, "long": 5000},
        requested=4096,
        budget=100_000,
    )
    assert caps["long"] >= caps["short"]


def test_all_structurally_invalid_sessions_are_repair_candidates() -> None:
    rows = [
        ("s0", [], None, [], None, None),
        ("s1", [], None, [], "empty_frames", None),
        ("s2", [], None, [], "invalid_json", None),
        ("s3", [], None, [], "empty_frames", None),
        ("s4", [], None, [], "coverage_gap", None),
        ("s5", [], None, [], "empty_durable_memory", None),
    ]
    assert _repair_positions(rows) == [1, 2, 3, 4]


def test_call_checkpoint_restores_text_and_original_token_record(tmp_path) -> None:
    messages = [{"role": "user", "content": "compact memory"}]
    record = DeepSeekCallRecord(
        question_id="q", variant="hierarchical_role_graph_v3_6",
        stage="build_v36_session", call_id="call", model="model",
        thinking_mode="none", prompt_tokens=11, completion_tokens=7,
        total_tokens=18, reasoning_tokens=0,
    )
    _save_call_checkpoint(
        tmp_path, stage="build_v36_session", key="session",
        messages=messages, max_tokens=128, text="{\"frames\":[]}",
        record=record,
    )
    restored = _load_call_checkpoint(
        tmp_path, stage="build_v36_session", key="session",
        messages=messages, max_tokens=128,
    )
    assert restored is not None
    assert restored[0] == "{\"frames\":[]}"
    assert restored[1].total_tokens == 18
    assert restored[1].reasoning_tokens == 0
    assert _load_call_checkpoint(
        tmp_path, stage="build_v36_session", key="session",
        messages=messages, max_tokens=129,
    ) is None
    assert not list(tmp_path.glob("*.tmp"))


def test_parallel_group_error_cancels_pending_work_without_queue_deadlock(
    tmp_path, monkeypatch,
) -> None:
    cases = [
        QuestionCase(
            question_id=f"q{index}", question_type="unknown",
            question="question", answer="answer", question_date=None,
            haystack_sessions=[[{"role": "user", "content": "memory"}]],
            haystack_session_ids=[f"s{index}"], haystack_dates=[None],
            answer_session_ids=[], memory_cache_key=f"cache-{index}",
        )
        for index in range(80)
    ]
    config = DemoConfig(
        data_path=tmp_path / "unused.json", output_dir=tmp_path / "out",
        variants=("hierarchical_role_graph_v3_6",),
        question_workers=4, mock_services=True,
    )
    original = pipeline._run_cases_with_memory_cache

    def controlled(config_value, case_values, *args, **kwargs):
        if len(case_values) > 1:
            yield from original(config_value, case_values, *args, **kwargs)
            return
        if case_values[0].question_id == "q0":
            raise RuntimeError("synthetic group failure")
        yield case_values[0], None, [], []

    monkeypatch.setattr(pipeline, "_run_cases_with_memory_cache", controlled)
    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="synthetic group failure"):
        list(controlled(
            config, cases, "hierarchical_role_graph_v3_6",
            tmp_path / "variant", InflightLimiter(8),
            allow_memory_cache_read=True,
        ))
    assert time.perf_counter() - started < 2.0
