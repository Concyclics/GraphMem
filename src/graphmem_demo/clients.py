from __future__ import annotations

import hashlib
import math
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable

from .models import CompressionRecord, DeepSeekCallRecord, EmbeddingCallRecord


_COMPRESSOR_LOAD_LOCK = threading.Lock()


def rough_token_count(text: str) -> int:
    words = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return max(len(words), 1) if text else 0


def parse_usage(usage: Any) -> dict[str, int]:
    data = _as_mapping(usage)
    prompt_details = _as_mapping(data.get("prompt_tokens_details"))
    completion_details = _as_mapping(data.get("completion_tokens_details"))
    cache_hit = data.get("prompt_cache_hit_tokens")
    cache_miss = data.get("prompt_cache_miss_tokens")
    return {
        "prompt_tokens": _int_value(data.get("prompt_tokens")),
        "completion_tokens": _int_value(data.get("completion_tokens")),
        "total_tokens": _int_value(data.get("total_tokens")),
        "prompt_cache_hit_tokens": _int_value(
            cache_hit if cache_hit is not None else prompt_details.get("cached_tokens")
        ),
        "prompt_cache_miss_tokens": _int_value(cache_miss),
        "reasoning_tokens": _int_value(completion_details.get("reasoning_tokens")),
    }


def cosine_similarity(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


@dataclass
class LLMResult:
    text: str
    record: DeepSeekCallRecord


@dataclass
class LocalSummaryResult:
    text: str
    record: CompressionRecord


def _openai_client_class() -> Any:
    try:
        from openai import OpenAI
    except ModuleNotFoundError as error:  # pragma: no cover - depends on environment
        raise RuntimeError("The openai package is required for real API clients") from error
    return OpenAI


class DeepSeekClient:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        max_retries: int = 2,
        timeout_sec: float = 180.0,
    ) -> None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required for real DeepSeek calls")
        self.model = model or os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
        OpenAI = _openai_client_class()
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            timeout=timeout_sec,
        )
        self.max_retries = max_retries

    def chat(
        self,
        *,
        question_id: str,
        variant: str,
        stage: str,
        messages: list[dict[str, str]],
        thinking_mode: str,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResult:
        start = time.perf_counter()
        last_error: Exception | None = None
        for retry_count in range(self.max_retries + 1):
            try:
                request: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                }
                if thinking_mode == "enabled":
                    request["extra_body"] = {"thinking": {"type": "enabled"}}
                    request["reasoning_effort"] = "high"
                elif thinking_mode in {"disabled", "none"}:
                    request["extra_body"] = {"thinking": {"type": "disabled"}}
                else:
                    raise ValueError(f"Unsupported thinking_mode: {thinking_mode}")
                if max_tokens is not None:
                    request["max_tokens"] = max_tokens
                if json_mode:
                    request["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(**request)
                usage = parse_usage(response.usage)
                record = DeepSeekCallRecord(
                    question_id=question_id,
                    variant=variant,
                    stage=stage,  # type: ignore[arg-type]
                    call_id=response.id or str(uuid.uuid4()),
                    model=response.model or self.model,
                    thinking_mode=thinking_mode,  # type: ignore[arg-type]
                    latency_sec=time.perf_counter() - start,
                    retry_count=retry_count,
                    finish_reason=response.choices[0].finish_reason,
                    max_tokens=max_tokens,
                    response_format="json_object" if json_mode else "text",
                    **usage,
                )
                text = response.choices[0].message.content or ""
                return LLMResult(text=text.strip(), record=record)
            except Exception as error:  # pragma: no cover - depends on remote API behavior
                last_error = error
                if retry_count < self.max_retries:
                    time.sleep(min(2**retry_count, 4))
        raise RuntimeError(f"DeepSeek {stage} call failed: {last_error}") from last_error


class MockDeepSeekClient:
    model = "mock-deepseek-v4-pro"

    def chat(
        self,
        *,
        question_id: str,
        variant: str,
        stage: str,
        messages: list[dict[str, str]],
        thinking_mode: str,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResult:
        prompt = "\n".join(message["content"] for message in messages)
        if stage.startswith("build_summary"):
            compact = " ".join(prompt.split()[-24:])
            if json_mode:
                if '"m"' in prompt or '"m":' in prompt:
                    text = (
                        '{"m": ["'
                        + compact.replace('"', "'")
                        + '", "mock fact"], "k": ["mock"]}'
                    )
                else:
                    text = (
                        '{"compact_summary": "'
                        + compact.replace('"', "'")
                        + '", "facts": ["mock fact"], "updates": [], '
                        '"time_anchors": [], "keywords": ["mock"]}'
                    )
            else:
                text = "Summary: " + compact
        else:
            text = "Mock answer based on retrieved evidence."
        prompt_tokens = rough_token_count(prompt)
        completion_tokens = rough_token_count(text)
        reasoning_tokens = 7 if thinking_mode == "enabled" else 0
        record = DeepSeekCallRecord(
            question_id=question_id,
            variant=variant,
            stage=stage,  # type: ignore[arg-type]
            call_id=f"mock-{uuid.uuid4()}",
            model=self.model,
            thinking_mode=thinking_mode,  # type: ignore[arg-type]
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            reasoning_tokens=reasoning_tokens,
            latency_sec=0.0,
            finish_reason="stop",
            max_tokens=max_tokens,
            response_format="json_object" if json_mode else "text",
        )
        return LLMResult(text=text, record=record)


class EmbeddingClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "local-embedding",
        batch_size: int = 64,
    ) -> None:
        OpenAI = _openai_client_class()
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.batch_size = batch_size
        self.records: list[EmbeddingCallRecord] = []

    def embed(
        self,
        texts: list[str],
        *,
        question_id: str,
        variant: str,
    ) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in _batches(texts, self.batch_size):
            start = time.perf_counter()
            try:
                response = self.client.embeddings.create(model=self.model, input=batch)
                ordered = sorted(response.data, key=lambda item: item.index)
                vectors.extend([list(item.embedding) for item in ordered])
                usage = parse_usage(response.usage)
                self.records.append(
                    EmbeddingCallRecord(
                        question_id=question_id,
                        variant=variant,
                        item_count=len(batch),
                        prompt_tokens=usage["prompt_tokens"],
                        total_tokens=usage["total_tokens"],
                        latency_sec=time.perf_counter() - start,
                        model=response.model or self.model,
                    )
                )
            except Exception as error:
                self.records.append(
                    EmbeddingCallRecord(
                        question_id=question_id,
                        variant=variant,
                        item_count=len(batch),
                        prompt_tokens=0,
                        total_tokens=0,
                        latency_sec=time.perf_counter() - start,
                        model=self.model,
                        error_status=str(error),
                    )
                )
                raise
        return vectors


class MockEmbeddingClient:
    model = "mock-embedding"

    def __init__(self) -> None:
        self.records: list[EmbeddingCallRecord] = []

    def embed(
        self,
        texts: list[str],
        *,
        question_id: str,
        variant: str,
    ) -> list[list[float]]:
        vectors = [_hashed_vector(text) for text in texts]
        token_count = sum(rough_token_count(text) for text in texts)
        self.records.append(
            EmbeddingCallRecord(
                question_id=question_id,
                variant=variant,
                item_count=len(texts),
                prompt_tokens=token_count,
                total_tokens=token_count,
                latency_sec=0.0,
                model=self.model,
            )
        )
        return vectors


class LLMLinguaCompressor:
    def __init__(
        self,
        ratio: float,
        model_name: str | None = None,
        device_map: str | None = None,
        use_llmlingua2: bool = False,
    ) -> None:
        self.ratio = ratio
        default_model = (
            "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
            if use_llmlingua2
            else "NousResearch/Llama-2-7b-hf"
        )
        self.model_name = model_name or os.environ.get("LLMLINGUA_MODEL_NAME", default_model)
        self.device_map = device_map or os.environ.get("LLMLINGUA_DEVICE_MAP", "cuda")
        self.use_llmlingua2 = use_llmlingua2
        self._compressor: Any = None
        self.records: list[CompressionRecord] = []

    def compress(
        self,
        text: str,
        *,
        question_id: str,
        variant: str,
        stage: str,
        chunk_rough_tokens: int = 0,
    ) -> str:
        if self._compressor is None:
            with _COMPRESSOR_LOAD_LOCK:
                if self._compressor is None:
                    from llmlingua import PromptCompressor

                    self._compressor = PromptCompressor(
                        model_name=self.model_name,
                        device_map=self.device_map,
                        use_llmlingua2=self.use_llmlingua2,
                    )
        chunks = _rough_text_chunks(text, chunk_rough_tokens)
        start = time.perf_counter()
        compressed_chunks: list[str] = []
        origin_tokens = 0
        compressed_tokens = 0
        try:
            for chunk in chunks:
                result = self._compressor.compress_prompt([chunk], rate=self.ratio)
                compressed_chunk = str(result["compressed_prompt"])
                compressed_chunks.append(compressed_chunk)
                origin_tokens += _int_value(result.get("origin_tokens")) or rough_token_count(chunk)
                compressed_tokens += _int_value(
                    result.get("compressed_tokens")
                ) or rough_token_count(compressed_chunk)
        except Exception as error:
            raise RuntimeError(f"LLMLingua compression failed during {stage}: {error}") from error
        compressed = "\n\n".join(compressed_chunks)
        self.records.append(
            CompressionRecord(
                question_id=question_id,
                variant=variant,
                stage=stage,
                origin_tokens=origin_tokens,
                compressed_tokens=compressed_tokens,
                latency_sec=time.perf_counter() - start,
                compressor=f"{'llmlingua2' if self.use_llmlingua2 else 'llmlingua'}:{self.model_name}",
                chunk_count=len(chunks),
            )
        )
        return compressed


class NoOpCompressor:
    def __init__(self) -> None:
        self.records: list[CompressionRecord] = []

    def compress(
        self,
        text: str,
        *,
        question_id: str,
        variant: str,
        stage: str,
        chunk_rough_tokens: int = 0,
    ) -> str:
        return text


class LocalSummarizer:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        max_retries: int = 1,
        timeout_sec: float = 180.0,
    ) -> None:
        OpenAI = _openai_client_class()
        self.client = OpenAI(
            api_key=api_key or os.environ.get("LOCAL_SUMMARIZER_API_KEY", "local-summarizer"),
            base_url=base_url,
            timeout=timeout_sec,
        )
        self.model = model
        self.max_retries = max_retries
        self.records: list[CompressionRecord] = []

    def summarize(
        self,
        *,
        question_id: str,
        variant: str,
        stage: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        json_mode: bool = True,
    ) -> LocalSummaryResult:
        start = time.perf_counter()
        prompt = "\n".join(message["content"] for message in messages)
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
        if json_mode:
            request["response_format"] = {"type": "json_object"}
        last_error: Exception | None = None
        for retry_count in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**request)
                text = (response.choices[0].message.content or "").strip()
                usage = parse_usage(response.usage)
                record = CompressionRecord(
                    question_id=question_id,
                    variant=variant,
                    stage=stage,
                    origin_tokens=usage["prompt_tokens"] or rough_token_count(prompt),
                    compressed_tokens=usage["completion_tokens"] or rough_token_count(text),
                    latency_sec=time.perf_counter() - start,
                    compressor=f"qwen_local:{self.model}",
                    chunk_count=1,
                )
                self.records.append(record)
                return LocalSummaryResult(text=text, record=record)
            except Exception as error:  # pragma: no cover - depends on remote API behavior
                last_error = error
                if retry_count < self.max_retries:
                    time.sleep(min(2**retry_count, 4))
        text = _fallback_summary_json(prompt)
        record = CompressionRecord(
            question_id=question_id,
            variant=variant,
            stage=stage,
            origin_tokens=rough_token_count(prompt),
            compressed_tokens=rough_token_count(text),
            latency_sec=time.perf_counter() - start,
            compressor=f"qwen_local:{self.model}",
            chunk_count=1,
            error_status=str(last_error),
        )
        self.records.append(record)
        return LocalSummaryResult(text=text, record=record)


class MockLocalSummarizer:
    def __init__(self, model: str = "mock-qwen-local") -> None:
        self.model = model
        self.records: list[CompressionRecord] = []

    def summarize(
        self,
        *,
        question_id: str,
        variant: str,
        stage: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        json_mode: bool = True,
    ) -> LocalSummaryResult:
        prompt = "\n".join(message["content"] for message in messages)
        text = _fallback_summary_json(prompt)
        record = CompressionRecord(
            question_id=question_id,
            variant=variant,
            stage=stage,
            origin_tokens=rough_token_count(prompt),
            compressed_tokens=rough_token_count(text),
            latency_sec=0.0,
            compressor=f"qwen_local:{self.model}",
            chunk_count=1,
        )
        self.records.append(record)
        return LocalSummaryResult(text=text, record=record)


class MockCompressor:
    def __init__(self, ratio: float) -> None:
        self.ratio = ratio
        self.records: list[CompressionRecord] = []

    def compress(
        self,
        text: str,
        *,
        question_id: str,
        variant: str,
        stage: str,
        chunk_rough_tokens: int = 0,
    ) -> str:
        chunks = _rough_text_chunks(text, chunk_rough_tokens)
        compressed_chunks: list[str] = []
        for chunk in chunks:
            tokens = chunk.split()
            keep = max(1, int(len(tokens) * self.ratio))
            compressed_chunks.append(" ".join(tokens[:keep]))
        compressed = "\n\n".join(compressed_chunks)
        self.records.append(
            CompressionRecord(
                question_id=question_id,
                variant=variant,
                stage=stage,
                origin_tokens=rough_token_count(text),
                compressed_tokens=rough_token_count(compressed),
                latency_sec=0.0,
                compressor="mock-ratio",
                chunk_count=len(chunks),
            )
        )
        return compressed


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def _int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _batches(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _rough_text_chunks(text: str, max_rough_tokens: int) -> list[str]:
    if not text or max_rough_tokens <= 0 or rough_token_count(text) <= max_rough_tokens:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for block in [block for block in text.split("\n\n") if block]:
        block_tokens = rough_token_count(block)
        if block_tokens > max_rough_tokens:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_tokens = 0
            words = block.split()
            for start in range(0, len(words), max_rough_tokens):
                chunks.append(" ".join(words[start : start + max_rough_tokens]))
            continue
        if current and current_tokens + block_tokens > max_rough_tokens:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0
        current.append(block)
        current_tokens += block_tokens
    if current:
        chunks.append("\n\n".join(current))
    return chunks or [text]


def _fallback_summary_json(prompt: str) -> str:
    lines = [
        line.strip()
        for line in prompt.splitlines()
        if line.strip()
        and not line.startswith(("Build stage:", "Session:", "Date:", "Child memory:"))
    ]
    facts = [_compact_line(line) for line in lines if _compact_line(line)][:6]
    keywords = _keywords_from_text(" ".join(facts))[:8]
    if '"facts"' in prompt or "facts" in prompt.lower() and "events" in prompt.lower():
        return _json_dumps_ascii(
            {
                "facts": facts,
                "events": [],
                "counts": [fact for fact in facts if re.search(r"\d", fact)],
                "dates": [fact for fact in facts if _looks_temporal(fact)],
                "updates": [fact for fact in facts if _looks_update(fact)],
                "keywords": keywords,
            }
        )
    if '"m"' in prompt or "short memory fact" in prompt:
        return _json_dumps_ascii({"m": facts, "k": keywords})
    return _json_dumps_ascii(
        {
            "compact_summary": "; ".join(facts[:3]),
            "facts": facts,
            "updates": [fact for fact in facts if _looks_update(fact)],
            "time_anchors": [fact for fact in facts if _looks_temporal(fact)],
            "keywords": keywords,
        }
    )


def _compact_line(line: str, max_chars: int = 180) -> str:
    line = re.sub(r"\s+", " ", line).strip()
    line = re.sub(r"^\[Child \d+\]\s*", "", line)
    return line[:max_chars].strip()


def _keywords_from_text(text: str) -> list[str]:
    stop = {
        "user",
        "assistant",
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "have",
        "我",
        "你",
        "了",
        "的",
    }
    seen: set[str] = set()
    keywords: list[str] = []
    for token in re.findall(r"[\w\u4e00-\u9fff]+", text.lower()):
        if len(token) < 2 or token in stop or token in seen:
            continue
        seen.add(token)
        keywords.append(token)
    return keywords


def _looks_temporal(text: str) -> bool:
    return bool(
        re.search(
            r"\b\d{1,2}(:\d{2})?\s*(am|pm)?\b|day|week|month|year|today|yesterday|tomorrow|最近|今天|昨天|明天|周|月|年",
            text,
            flags=re.IGNORECASE,
        )
    )


def _looks_update(text: str) -> bool:
    return bool(
        re.search(
            r"cancel|currently|now|instead|changed|update|not|didn't|did not|no longer|取消|现在|当前|改为|不再|没有",
            text,
            flags=re.IGNORECASE,
        )
    )


def _json_dumps_ascii(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=True)


def _hashed_vector(text: str, dimension: int = 24) -> list[float]:
    vector = [0.0] * dimension
    for token in re.findall(r"\w+", text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        vector[digest[0] % dimension] += 1.0 if digest[1] % 2 else -1.0
    if not any(vector):
        vector[0] = 1.0
    return vector
