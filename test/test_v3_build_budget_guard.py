from graphmem_demo.v3.runtime import (
    _BUILD_BUDGET_RESERVE,
    _bounded_optional_max_tokens,
    _uniform_session_completion_cap,
)


def test_session_completion_cap_reserves_hard_build_headroom() -> None:
    prompts = [3_800] * 50
    cap = _uniform_session_completion_cap(
        prompts,
        requested_max_tokens=3_072,
        build_budget_tokens=300_000,
    )

    assert cap == 2_140
    assert sum(prompts) + len(prompts) * cap <= 300_000 - _BUILD_BUDGET_RESERVE


def test_session_completion_cap_keeps_requested_limit_when_budget_allows() -> None:
    assert _uniform_session_completion_cap(
        [1_000] * 10,
        requested_max_tokens=3_072,
        build_budget_tokens=300_000,
    ) == 3_072


def test_optional_call_is_skipped_or_clipped_by_remaining_budget() -> None:
    assert _bounded_optional_max_tokens(
        spent_tokens=295_000,
        prompt_estimate=1_500,
        requested_max_tokens=2_048,
        build_budget_tokens=300_000,
        minimum=128,
    ) == 500
    assert _bounded_optional_max_tokens(
        spent_tokens=296_000,
        prompt_estimate=1_000,
        requested_max_tokens=2_048,
        build_budget_tokens=300_000,
        minimum=128,
    ) == 0
