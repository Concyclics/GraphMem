from __future__ import annotations

from types import SimpleNamespace

import pytest

import graphmem_demo.clients as clients_module
from graphmem_demo.clients import OpenAICompatibleClient


def test_openai_profile_maps_none_to_reasoning_effort_without_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **request):
            captured.update(request)
            return SimpleNamespace(
                id="fake-call",
                model=request["model"],
                usage={
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "total_tokens": 4,
                },
                choices=[
                    SimpleNamespace(
                        finish_reason="stop",
                        message=SimpleNamespace(content="ok"),
                    )
                ],
            )

    class FakeOpenAI:
        def __init__(self, **_: object) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv("TEST_PROVIDER_KEY", "secret")
    monkeypatch.setattr(clients_module, "_openai_client_class", lambda: FakeOpenAI)
    result = OpenAICompatibleClient(
        model="gpt-5.4-mini",
        base_url="https://provider.invalid/v1",
        api_key_env="TEST_PROVIDER_KEY",
        request_profile="openai",
    ).chat(
        question_id="q",
        variant="v",
        stage="answer_qa",
        messages=[{"role": "user", "content": "answer"}],
        thinking_mode="none",
        max_tokens=64,
    )

    assert captured["reasoning_effort"] == "none"
    assert "extra_body" not in captured
    assert captured["max_completion_tokens"] == 64
    assert "max_tokens" not in captured
    assert result.record.reasoning_tokens == 0
    assert result.record.breakdown_inferred is True


def test_provider_rejects_unknown_request_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_PROVIDER_KEY", "secret")
    with pytest.raises(ValueError, match="request_profile"):
        OpenAICompatibleClient(
            api_key_env="TEST_PROVIDER_KEY", request_profile="unknown"
        )


def test_qwen_profile_disables_thinking_via_chat_template_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **request):
            captured.update(request)
            return SimpleNamespace(
                id="fake-qwen-call", model=request["model"],
                usage={"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                choices=[SimpleNamespace(
                    finish_reason="stop", message=SimpleNamespace(content="ok")
                )],
            )

    class FakeOpenAI:
        def __init__(self, **_: object) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv("TEST_PROVIDER_KEY", "secret")
    monkeypatch.setattr(clients_module, "_openai_client_class", lambda: FakeOpenAI)
    OpenAICompatibleClient(
        model="Qwen3-32B-FP8", base_url="http://127.0.0.1:8002/v1",
        api_key_env="TEST_PROVIDER_KEY", request_profile="qwen",
    ).chat(
        question_id="q", variant="v", stage="answer_qa",
        messages=[{"role": "user", "content": "answer"}],
        thinking_mode="none", max_tokens=64,
    )

    assert captured["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert "reasoning_effort" not in captured
    assert captured["max_tokens"] == 64


def test_provider_retries_transient_service_errors_with_bounded_backoff() -> None:
    transient = RuntimeError("503 Service temporarily unavailable")
    rate_limit = RuntimeError("429 Too Many Requests")
    permanent = RuntimeError("400 invalid request")
    assert clients_module._is_retryable_llm_error(transient)
    assert clients_module._is_retryable_llm_error(rate_limit)
    assert not clients_module._is_retryable_llm_error(permanent)
    assert clients_module._retry_sleep_sec(8, transient) == 60.0
    assert clients_module._retry_sleep_sec(8, rate_limit) == 60.0
