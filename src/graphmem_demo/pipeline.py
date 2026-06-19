from __future__ import annotations

import csv
import hashlib
import json
import re
import threading
import time
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .clients import (
    DeepSeekClient,
    EmbeddingClient,
    LLMLinguaCompressor,
    LocalSummarizer,
    MockCompressor,
    MockDeepSeekClient,
    MockEmbeddingClient,
    MockLocalSummarizer,
    NoOpCompressor,
    cosine_similarity,
    rough_token_count,
)
from .data import build_leaf_nodes, group_by_session, load_longmemeval_cases
from .models import (
    DeepSeekCallRecord,
    GraphEdge,
    LeafNode,
    QuestionCase,
    QuestionStats,
    RetrievedContext,
    SummaryNode,
    VariantStats,
)
from .stats import (
    aggregate_variant_stats,
    build_question_stats,
    build_stats_payload,
    query_stats_payload,
)


@dataclass(frozen=True)
class VariantSpec:
    tree_mode: str
    compression: bool
    graph: bool
    fanout_k: int | None = None
    summary_max_tokens: int | None = None
    summary_schema: str = "minimal_memory_v1"
    build_leaf_text: str = "raw"
    retrieval_leaf_text: str = "raw"
    raw_question_types: tuple[str, ...] = ()
    hybrid_retrieval: bool = False
    local_summary: bool = False
    default_summarizer_model: str | None = None
    enhanced_retrieval: bool = False
    enhanced_qa: bool = False


VARIANT_SPECS = {
    "raw_rag": VariantSpec("raw_rag", False, False),
    "summary_tree_k4_no_compress": VariantSpec("legacy_kway", False, False, fanout_k=4),
    "summary_tree_k4_graphmem": VariantSpec("legacy_kway", True, True, fanout_k=4),
    "direct_session_k16_no_compress": VariantSpec("direct_session", False, False, fanout_k=16),
    "direct_session_k16_graphmem": VariantSpec("direct_session", True, True, fanout_k=16),
    "direct_session_k16_compact_no_compress": VariantSpec(
        "direct_session",
        False,
        False,
        fanout_k=16,
        summary_schema="compact_memory_v2",
        build_leaf_text="user_only",
        retrieval_leaf_text="user_only",
        hybrid_retrieval=True,
    ),
    "direct_session_k16_compact_graphmem": VariantSpec(
        "direct_session",
        True,
        True,
        fanout_k=16,
        summary_schema="compact_memory_v2",
        build_leaf_text="user_only",
        retrieval_leaf_text="user_only",
        hybrid_retrieval=True,
    ),
    "qwen35_2b_summary_graphmem": VariantSpec(
        "direct_session",
        False,
        True,
        fanout_k=16,
        summary_schema="multilingual_memory_v1",
        build_leaf_text="user_only",
        retrieval_leaf_text="user_only",
        hybrid_retrieval=True,
        local_summary=True,
        default_summarizer_model="Qwen/Qwen3.5-2B",
        enhanced_retrieval=True,
        enhanced_qa=True,
    ),
    "qwen35_08b_summary_graphmem": VariantSpec(
        "direct_session",
        False,
        True,
        fanout_k=16,
        summary_schema="multilingual_memory_v1",
        build_leaf_text="user_only",
        retrieval_leaf_text="user_only",
        hybrid_retrieval=True,
        local_summary=True,
        default_summarizer_model="Qwen/Qwen3.5-0.8B",
        enhanced_retrieval=True,
        enhanced_qa=True,
    ),
    "qwen35_2b_summary_graphmem_no_retrieval_enhance": VariantSpec(
        "direct_session",
        False,
        True,
        fanout_k=16,
        summary_schema="multilingual_memory_v1",
        build_leaf_text="user_only",
        retrieval_leaf_text="user_only",
        hybrid_retrieval=True,
        local_summary=True,
        default_summarizer_model="Qwen/Qwen3.5-2B",
        enhanced_retrieval=False,
        enhanced_qa=True,
    ),
    "qwen35_2b_summary_graphmem_no_qa_enhance": VariantSpec(
        "direct_session",
        False,
        True,
        fanout_k=16,
        summary_schema="multilingual_memory_v1",
        build_leaf_text="user_only",
        retrieval_leaf_text="user_only",
        hybrid_retrieval=True,
        local_summary=True,
        default_summarizer_model="Qwen/Qwen3.5-2B",
        enhanced_retrieval=True,
        enhanced_qa=False,
    ),
    "single_llm_summary_graphmem": VariantSpec(
        "direct_session",
        False,
        True,
        fanout_k=16,
        summary_max_tokens=512,
        summary_schema="compact_memory_v2",
        build_leaf_text="user_only",
        retrieval_leaf_text="user_only",
        raw_question_types=("single-session-assistant", "single-session-preference"),
        hybrid_retrieval=True,
        local_summary=False,
        enhanced_retrieval=True,
        enhanced_qa=True,
    ),
    # Keep old names for existing run directories and resume workflows.
    "summary_tree_no_compress": VariantSpec("legacy_kway", False, False),
    "token_efficient_graphmem": VariantSpec("legacy_kway", True, True),
}
VARIANTS = set(VARIANT_SPECS)


@dataclass
class DemoConfig:
    data_path: Path
    output_dir: Path
    question_type: str = "multi-session"
    variants: tuple[str, ...] = (
        "direct_session_k16_compact_no_compress",
        "direct_session_k16_compact_graphmem",
    )
    deepseek_model: str | None = None
    embedding_base_url: str = "http://127.0.0.1:8002/v1"
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    tree_mode: str | None = None
    fanout_k: int = 16
    max_group_rough_tokens: int = 6000
    leaf_top_k: int = 14
    root_top_k: int = 4
    root_candidate_k: int = 8
    global_leaf_top_k: int = 24
    qa_summary_top_k: int = 4
    per_session_leaf_k: int = 2
    graph_neighbor_k: int = 2
    qa_context_token_budget: int = 10000
    qa_max_tokens: int = 1024
    compression_ratio: float = 0.5
    max_questions: int = 10
    question_workers: int = 2
    summary_workers: int = 0
    max_inflight_deepseek: int = 0
    summary_schema: str | None = None
    summarizer_kind: str = "auto"
    summarizer_base_url: str = "http://127.0.0.1:8003/v1"
    summarizer_model: str | None = None
    summary_token_budget: int = 320
    build_leaf_text: str = "auto"
    retrieval_leaf_text: str = "auto"
    compressor_chunk_rough_tokens: int = 384
    raw_group_summary_max_tokens: int = 256
    session_summary_max_tokens: int = 320
    legacy_internal_summary_max_tokens: int = 224
    resume: bool = False
    mock_services: bool = False
    mock_llm: bool = False
    mock_embedding: bool = False
    mock_compressor: bool = False
    mock_summarizer: bool = False
    llmlingua_model: str | None = None
    llmlingua_device_map: str | None = None
    use_llmlingua2: bool = False
    enable_speaker_profiles: bool = False
    enable_speaker_neighbor_window: bool = False
    enable_speaker_retrieval_text: bool = False
    enable_typed_root_edges: bool = False
    enable_multilevel_summary_retrieval: bool = False

    def __post_init__(self) -> None:
        unknown = set(self.variants) - VARIANTS
        if unknown:
            raise ValueError(f"Unknown variants: {', '.join(sorted(unknown))}")
        if self.fanout_k < 2:
            raise ValueError("fanout_k must be at least 2")
        if not 0 < self.compression_ratio <= 1:
            raise ValueError("compression_ratio must be in (0, 1]")
        if self.question_workers < 1:
            raise ValueError("question_workers must be at least 1")
        if self.summary_workers < 0 or self.max_inflight_deepseek < 0:
            raise ValueError("summary_workers and max_inflight_deepseek cannot be negative")
        if self.tree_mode is not None and self.tree_mode not in {"legacy_kway", "direct_session"}:
            raise ValueError("tree_mode must be legacy_kway or direct_session")
        if self.summary_schema not in {
            None,
            "minimal_memory_v1",
            "compact_memory_v2",
            "multilingual_memory_v1",
        }:
            raise ValueError(
                "summary_schema must be minimal_memory_v1, compact_memory_v2, or multilingual_memory_v1"
            )
        if self.summarizer_kind not in {"auto", "none", "llmlingua2", "qwen_local"}:
            raise ValueError("summarizer_kind must be auto, none, llmlingua2, or qwen_local")
        if self.qa_context_token_budget < 1000:
            raise ValueError("qa_context_token_budget must be at least 1000")
        if self.qa_max_tokens < 128:
            raise ValueError("qa_max_tokens must be at least 128")
        if self.summary_token_budget < 32:
            raise ValueError("summary_token_budget must be at least 32")

    def use_mock_llm(self) -> bool:
        return self.mock_services or self.mock_llm

    def use_mock_embedding(self) -> bool:
        return self.mock_services or self.mock_embedding

    def use_mock_compressor(self) -> bool:
        return self.mock_services or self.mock_compressor

    def use_mock_summarizer(self) -> bool:
        return self.mock_services or self.mock_summarizer
        for field_name in ("build_leaf_text", "retrieval_leaf_text"):
            if getattr(self, field_name) not in {"auto", "raw", "user_only"}:
                raise ValueError(f"{field_name} must be auto, raw, or user_only")
        if min(
            self.root_candidate_k,
            self.global_leaf_top_k,
            self.qa_summary_top_k,
            self.per_session_leaf_k,
        ) < 1:
            raise ValueError("V2 retrieval k values must be at least 1")


@dataclass
class CaseRun:
    leaves: list[LeafNode]
    summaries: list[SummaryNode]
    edges: list[GraphEdge]
    retrieval: RetrievedContext
    answer: str
    llm_records: list[DeepSeekCallRecord]
    stats: QuestionStats


@dataclass
class MemoryBuild:
    leaves: list[LeafNode]
    summaries: list[SummaryNode]
    roots: list[SummaryNode]
    edges: list[GraphEdge]
    llm_records: list[DeepSeekCallRecord]
    metrics: "BuildMetrics"
    build_latency_sec: float


@dataclass
class BuildMetrics:
    ready_job_counts: list[dict[str, Any]] = field(default_factory=list)
    summary_parse_error_count: int = 0
    summary_truncation_count: int = 0
    peak_inflight_deepseek: int = 0
    _active_deepseek: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def begin_call(self) -> None:
        with self._lock:
            self._active_deepseek += 1
            self.peak_inflight_deepseek = max(
                self.peak_inflight_deepseek, self._active_deepseek
            )

    def end_call(self) -> None:
        with self._lock:
            self._active_deepseek -= 1


class InflightLimiter:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.peak = 0
        self._active = 0
        self._lock = threading.Lock()
        self._semaphore = threading.Semaphore(limit) if limit else None

    @contextmanager
    def track(self, metrics: BuildMetrics):
        if self._semaphore is not None:
            self._semaphore.acquire()
        with self._lock:
            self._active += 1
            self.peak = max(self.peak, self._active)
        metrics.begin_call()
        try:
            yield
        finally:
            metrics.end_call()
            with self._lock:
                self._active -= 1
            if self._semaphore is not None:
                self._semaphore.release()


@dataclass(frozen=True)
class SummaryJob:
    session_id: str
    session_date: str | None
    children: list[LeafNode | SummaryNode]
    stage: str
    level: int
    group_number: int
    summary_mode: str
    max_tokens: int


def run_demo(
    config: DemoConfig,
    *,
    llm: Any | None = None,
    embedder: Any | None = None,
    compressor: Any | None = None,
    summarizer: Any | None = None,
) -> list[VariantStats]:
    cases = load_longmemeval_cases(
        config.data_path, question_type=config.question_type, max_questions=config.max_questions
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    has_injected_services = any(
        service is not None for service in (llm, embedder, compressor, summarizer)
    )

    aggregates: list[VariantStats] = []
    for variant in config.variants:
        variant_llm, variant_embedder, variant_compressor, variant_summarizer = _complete_services(
            config, variant, llm, embedder, compressor, summarizer
        )
        variant_started = time.perf_counter()
        limiter = InflightLimiter(config.max_inflight_deepseek)
        variant_dir = config.output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        if not config.resume:
            _reset_jsonl_outputs(variant_dir)
        stats = _read_question_stats(variant_dir / "question_stats.jsonl") if config.resume else []
        completed = {item.question_id for item in stats}

        pending_cases = [case for case in cases if case.question_id not in completed]
        if _has_memory_cache_keys(pending_cases):
            for case, run, embedding_records, compression_records in _run_cases_with_memory_cache(
                config,
                pending_cases,
                variant,
                variant_dir,
                limiter,
                allow_memory_cache_read=config.resume,
                llm=llm,
                embedder=embedder,
                compressor=compressor,
                summarizer=summarizer,
            ):
                stats.append(run.stats)
                _write_case_outputs(
                    variant_dir, case, run, embedding_records, compression_records
                )
                _print_case_progress(run.stats)
        elif config.question_workers > 1 and not has_injected_services:
            with ThreadPoolExecutor(max_workers=config.question_workers) as executor:
                futures = {
                    executor.submit(
                        _run_case_with_fresh_services, config, case, variant, limiter
                    ): case
                    for case in pending_cases
                }
                for future in as_completed(futures):
                    case = futures[future]
                    run, embedding_records, compression_records = future.result()
                    stats.append(run.stats)
                    _write_case_outputs(
                        variant_dir, case, run, embedding_records, compression_records
                    )
                    _print_case_progress(run.stats)
        else:
            for case in pending_cases:
                embedding_start = len(variant_embedder.records)
                compression_start = len(variant_compressor.records)
                summarizer_start = len(variant_summarizer.records)
                run = run_case(
                    config,
                    case,
                    variant,
                    variant_llm,
                    variant_embedder,
                    variant_compressor,
                    limiter,
                    summarizer=variant_summarizer,
                )
                stats.append(run.stats)
                _write_case_outputs(
                    variant_dir,
                    case,
                    run,
                    variant_embedder.records[embedding_start:],
                    [
                        *variant_compressor.records[compression_start:],
                        *variant_summarizer.records[summarizer_start:],
                    ],
                )
                _print_case_progress(run.stats)

        aggregate = aggregate_variant_stats(stats, variant)
        aggregate.metadata.update(
            {
                "question_workers": config.question_workers,
                "summary_workers": config.summary_workers,
                "max_inflight_deepseek": config.max_inflight_deepseek,
                "peak_inflight_deepseek": limiter.peak,
                "summary_schema": config.summary_schema or VARIANT_SPECS[variant].summary_schema,
                "report_run_wall_time_sec": time.perf_counter() - variant_started,
            }
        )
        aggregates.append(aggregate)
        stage_totals = _deepseek_stage_totals(_read_jsonl(variant_dir / "llm_calls.jsonl"))
        local_summary_totals = _local_summary_stage_totals(
            _read_jsonl(variant_dir / "compression_stats.jsonl")
        )
        build_payload = build_stats_payload(stats, aggregate)
        build_payload["deepseek_token_by_stage"] = stage_totals
        build_payload["local_summarizer_by_stage"] = local_summary_totals
        query_payload = query_stats_payload(stats, aggregate)
        query_payload["deepseek_token_by_stage"] = stage_totals
        query_payload["local_summarizer_by_stage"] = local_summary_totals
        _write_json(variant_dir / "build_stats.json", build_payload)
        _write_json(variant_dir / "query_stats.json", query_payload)
        if variant in {
            "direct_session_k16_compact_graphmem",
            "qwen35_2b_summary_graphmem",
            "qwen35_08b_summary_graphmem",
            "qwen35_2b_summary_graphmem_no_retrieval_enhance",
            "qwen35_2b_summary_graphmem_no_qa_enhance",
        }:
            _write_manual_eval_template(variant_dir)

    _write_summary(config.output_dir, aggregates)
    return aggregates


def _complete_services(
    config: DemoConfig,
    variant: str,
    llm: Any | None = None,
    embedder: Any | None = None,
    compressor: Any | None = None,
    summarizer: Any | None = None,
) -> tuple[Any, Any, Any, Any]:
    spec = _variant_spec(config, variant)
    use_llmlingua2 = (
        config.use_llmlingua2
        or config.summarizer_kind == "llmlingua2"
        or (config.summarizer_kind == "auto" and spec.compression)
    )
    use_compressor = spec.compression and config.summarizer_kind != "none" and not _uses_local_summary(config, spec)
    summarizer_model = config.summarizer_model or spec.default_summarizer_model or "Qwen/Qwen3.5-2B"
    return (
        llm
        or (MockDeepSeekClient() if config.use_mock_llm() else DeepSeekClient(model=config.deepseek_model)),
        embedder
        or (
            MockEmbeddingClient()
            if config.use_mock_embedding()
            else EmbeddingClient(config.embedding_base_url, config.embedding_model)
        ),
        compressor
        or (
            MockCompressor(config.compression_ratio)
            if config.use_mock_compressor()
            else NoOpCompressor()
            if not use_compressor
            else LLMLinguaCompressor(
                ratio=config.compression_ratio,
                model_name=config.llmlingua_model,
                device_map=config.llmlingua_device_map,
                use_llmlingua2=use_llmlingua2,
            )
        ),
        summarizer
        or (
            MockLocalSummarizer(summarizer_model)
            if config.use_mock_summarizer()
            else LocalSummarizer(config.summarizer_base_url, summarizer_model)
        ),
    )


def _run_case_with_fresh_services(
    config: DemoConfig,
    case: QuestionCase,
    variant: str,
    limiter: InflightLimiter,
) -> tuple[CaseRun, list[Any], list[Any]]:
    return _run_case_with_services(config, case, variant, limiter)


def _record_start(service: Any) -> int:
    records = getattr(service, "records", None)
    return len(records) if records is not None else 0


def _records_since(service: Any, start: int) -> list[Any]:
    records = getattr(service, "records", None)
    if records is None:
        return []
    return list(records[start:])


def _memory_cache_path(
    config: DemoConfig,
    variant_dir: Path,
    case: QuestionCase,
    variant: str,
) -> Path:
    if not case.memory_cache_key:
        raise ValueError("memory_cache_key is required for memory cache path")
    key = _safe_cache_part(case.memory_cache_key)
    fingerprint = _memory_cache_fingerprint(config, case, variant)
    return variant_dir / "memory_cache" / f"{key}-{fingerprint[:16]}.json"


def _safe_cache_part(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe[:96] or "memory"


def _memory_cache_fingerprint(config: DemoConfig, case: QuestionCase, variant: str) -> str:
    spec = _variant_spec(config, variant)
    data_payload = {
        "haystack_session_ids": case.haystack_session_ids,
        "haystack_dates": case.haystack_dates,
        "haystack_sessions": case.haystack_sessions,
    }
    data_hash = hashlib.sha256(
        json.dumps(data_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    payload = {
        "version": 1,
        "variant": variant,
        "tree_mode": spec.tree_mode,
        "compression": spec.compression,
        "graph": spec.graph,
        "fanout_k": spec.fanout_k or config.fanout_k,
        "summary_schema": _summary_schema(config, spec),
        "summary_max_tokens": _summary_max_tokens(config, spec),
        "summary_token_budget": config.summary_token_budget,
        "summarizer_kind": config.summarizer_kind,
        "summarizer_model": config.summarizer_model or spec.default_summarizer_model,
        "uses_local_summary": _uses_local_summary(config, spec),
        "deepseek_model": config.deepseek_model,
        "embedding_model": config.embedding_model,
        "build_leaf_text": _effective_leaf_text_mode(
            config.build_leaf_text, spec, case, phase="build"
        ),
        "retrieval_leaf_text": _effective_leaf_text_mode(
            config.retrieval_leaf_text, spec, case, phase="retrieval"
        ),
        "max_group_rough_tokens": config.max_group_rough_tokens,
        "raw_group_summary_max_tokens": config.raw_group_summary_max_tokens,
        "session_summary_max_tokens": config.session_summary_max_tokens,
        "legacy_internal_summary_max_tokens": config.legacy_internal_summary_max_tokens,
        "compression_ratio": config.compression_ratio,
        "compressor_chunk_rough_tokens": config.compressor_chunk_rough_tokens,
        "llmlingua_model": config.llmlingua_model,
        "use_llmlingua2": config.use_llmlingua2,
        "enable_speaker_profiles": config.enable_speaker_profiles,
        "enable_speaker_retrieval_text": config.enable_speaker_retrieval_text,
        "enable_typed_root_edges": config.enable_typed_root_edges,
        "enable_multilevel_summary_retrieval": config.enable_multilevel_summary_retrieval,
        "graph_neighbor_k": config.graph_neighbor_k,
        "leaf_retrieval_text_version": 1,
        "summary_retrieval_text_version": 3,
        "summary_anchor_terms_version": 4,
        "keyword_edge_version": 3,
        "data_hash": data_hash,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _write_memory_cache(
    path: Path,
    memory: MemoryBuild,
    case: QuestionCase,
    variant: str,
    config: DemoConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "memory_cache_key": case.memory_cache_key,
        "fingerprint": _memory_cache_fingerprint(config, case, variant),
        "source_question_id": case.question_id,
        "leaves": [asdict(leaf) for leaf in memory.leaves],
        "summaries": [asdict(summary) for summary in memory.summaries],
        "root_ids": [root.node_id for root in memory.roots],
        "edges": [asdict(edge) for edge in memory.edges],
        "llm_records": [asdict(record) for record in memory.llm_records],
        "metrics": {
            "ready_job_counts": memory.metrics.ready_job_counts,
            "summary_parse_error_count": memory.metrics.summary_parse_error_count,
            "summary_truncation_count": memory.metrics.summary_truncation_count,
            "peak_inflight_deepseek": memory.metrics.peak_inflight_deepseek,
        },
        "build_latency_sec": memory.build_latency_sec,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
    tmp_path.replace(path)


def _load_memory_cache(path: Path) -> MemoryBuild | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            return None
        leaves = [LeafNode(**row) for row in payload.get("leaves", [])]
        summaries = [SummaryNode(**row) for row in payload.get("summaries", [])]
        summary_by_id = {summary.node_id: summary for summary in summaries}
        roots = [
            summary_by_id[root_id]
            for root_id in payload.get("root_ids", [])
            if root_id in summary_by_id
        ]
        edges = [GraphEdge(**row) for row in payload.get("edges", [])]
        llm_records = [
            DeepSeekCallRecord(**row) for row in payload.get("llm_records", [])
        ]
        metrics_payload = payload.get("metrics") or {}
        metrics = BuildMetrics(
            ready_job_counts=list(metrics_payload.get("ready_job_counts") or []),
            summary_parse_error_count=int(
                metrics_payload.get("summary_parse_error_count") or 0
            ),
            summary_truncation_count=int(
                metrics_payload.get("summary_truncation_count") or 0
            ),
            peak_inflight_deepseek=int(metrics_payload.get("peak_inflight_deepseek") or 0),
        )
        return MemoryBuild(
            leaves=leaves,
            summaries=summaries,
            roots=roots,
            edges=edges,
            llm_records=llm_records,
            metrics=metrics,
            build_latency_sec=float(payload.get("build_latency_sec") or 0.0),
        )
    except Exception:
        return None


def _run_case_with_services(
    config: DemoConfig,
    case: QuestionCase,
    variant: str,
    limiter: InflightLimiter,
    llm: Any | None = None,
    embedder: Any | None = None,
    compressor: Any | None = None,
    summarizer: Any | None = None,
) -> tuple[CaseRun, list[Any], list[Any]]:
    case_llm, case_embedder, case_compressor, case_summarizer = _complete_services(
        config, variant, llm, embedder, compressor, summarizer
    )
    embedding_start = _record_start(case_embedder)
    compression_start = _record_start(case_compressor)
    summarizer_start = _record_start(case_summarizer)
    run = run_case(
        config,
        case,
        variant,
        case_llm,
        case_embedder,
        case_compressor,
        limiter,
        summarizer=case_summarizer,
    )
    return (
        run,
        _records_since(case_embedder, embedding_start),
        [
            *_records_since(case_compressor, compression_start),
            *_records_since(case_summarizer, summarizer_start),
        ],
    )


def _has_memory_cache_keys(cases: list[QuestionCase]) -> bool:
    return any(case.memory_cache_key for case in cases)


def _run_cases_with_memory_cache(
    config: DemoConfig,
    cases: list[QuestionCase],
    variant: str,
    variant_dir: Path,
    limiter: InflightLimiter,
    *,
    allow_memory_cache_read: bool,
    llm: Any | None = None,
    embedder: Any | None = None,
    compressor: Any | None = None,
    summarizer: Any | None = None,
) -> Any:
    has_injected_services = any(
        service is not None for service in (llm, embedder, compressor, summarizer)
    )
    grouped: dict[str, list[QuestionCase]] = {}
    for case in cases:
        key = case.memory_cache_key or f"question:{case.question_id}"
        grouped.setdefault(key, []).append(case)

    for group_cases in grouped.values():
        build_case = group_cases[0]
        memory_cache_key = build_case.memory_cache_key
        if not memory_cache_key:
            case = group_cases[0]
            run, embedding_records, compression_records = _run_case_with_services(
                config,
                case,
                variant,
                limiter,
                llm=llm,
                embedder=embedder,
                compressor=compressor,
                summarizer=summarizer,
            )
            yield case, run, embedding_records, compression_records
            continue

        group_build_started = time.perf_counter()
        memory_cache_path = _memory_cache_path(config, variant_dir, build_case, variant)
        memory = (
            _load_memory_cache(memory_cache_path)
            if allow_memory_cache_read and memory_cache_path.exists()
            else None
        )
        loaded_from_cache = memory is not None
        build_embedding_records: list[Any] = []
        build_compression_records: list[Any] = []
        if memory is None:
            build_llm, build_embedder, build_compressor, build_summarizer = _complete_services(
                config, variant, llm, embedder, compressor, summarizer
            )
            build_embedding_start = _record_start(build_embedder)
            build_compression_start = _record_start(build_compressor)
            build_summarizer_start = _record_start(build_summarizer)
            memory = build_memory(
                config,
                build_case,
                variant,
                build_llm,
                build_embedder,
                build_compressor,
                limiter,
                summarizer=build_summarizer,
            )
            build_embedding_records = _records_since(build_embedder, build_embedding_start)
            build_compression_records = [
                *_records_since(build_compressor, build_compression_start),
                *_records_since(build_summarizer, build_summarizer_start),
            ]
            _write_memory_cache(memory_cache_path, memory, build_case, variant, config)
        build_record_question_id = build_case.question_id

        worker_count = min(config.question_workers, len(group_cases))
        if has_injected_services:
            worker_count = 1
        if worker_count <= 1:
            for case in group_cases:
                include_build_records = (
                    not loaded_from_cache and case.question_id == build_record_question_id
                )
                run, embedding_records, compression_records = _run_case_with_cached_memory(
                    config,
                    case,
                    variant,
                    limiter,
                    memory,
                    include_build_records,
                    group_build_started if include_build_records else None,
                    llm=llm,
                    embedder=embedder,
                )
                if include_build_records:
                    embedding_records = [*build_embedding_records, *embedding_records]
                    compression_records = [*build_compression_records, *compression_records]
                yield case, run, embedding_records, compression_records
            continue

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _run_case_with_cached_memory,
                    config,
                    case,
                    variant,
                    limiter,
                    memory,
                    (not loaded_from_cache and case.question_id == build_record_question_id),
                    (
                        group_build_started
                        if not loaded_from_cache and case.question_id == build_record_question_id
                        else None
                    ),
                    llm=llm,
                    embedder=embedder,
                ): case
                for case in group_cases
            }
            for future in as_completed(futures):
                case = futures[future]
                run, embedding_records, compression_records = future.result()
                if not loaded_from_cache and case.question_id == build_record_question_id:
                    embedding_records = [*build_embedding_records, *embedding_records]
                    compression_records = [*build_compression_records, *compression_records]
                yield case, run, embedding_records, compression_records


def _run_case_with_cached_memory(
    config: DemoConfig,
    case: QuestionCase,
    variant: str,
    limiter: InflightLimiter,
    memory: MemoryBuild,
    include_build_records: bool,
    case_started: float | None = None,
    llm: Any | None = None,
    embedder: Any | None = None,
) -> tuple[CaseRun, list[Any], list[Any]]:
    case_llm, case_embedder, _compressor, _summarizer = _complete_services(
        config, variant, llm, embedder
    )
    embedding_start = _record_start(case_embedder)
    run = run_case_with_memory(
        config,
        case,
        variant,
        memory,
        case_llm,
        case_embedder,
        limiter,
        include_build_records=include_build_records,
        case_started=case_started,
    )
    return run, _records_since(case_embedder, embedding_start), []


def _write_case_outputs(
    variant_dir: Path,
    case: QuestionCase,
    run: CaseRun,
    embedding_records: list[Any],
    compression_records: list[Any],
) -> None:
    _append_jsonl(variant_dir / "llm_calls.jsonl", [asdict(item) for item in run.llm_records])
    _append_jsonl(
        variant_dir / "embedding_calls.jsonl",
        [asdict(item) for item in embedding_records],
    )
    _append_jsonl(
        variant_dir / "compression_stats.jsonl",
        [asdict(item) for item in compression_records],
    )
    _append_jsonl(
        variant_dir / "nodes.jsonl",
        [_node_row(item) for item in [*run.leaves, *run.summaries]],
    )
    _append_jsonl(variant_dir / "edges.jsonl", [asdict(item) for item in run.edges])
    _append_jsonl(variant_dir / "question_stats.jsonl", [asdict(run.stats)])
    _append_jsonl(variant_dir / "retrieval_results.jsonl", [asdict(run.retrieval)])
    _append_jsonl(
        variant_dir / "answers.jsonl",
        [
            {
                "question_id": case.question_id,
                "variant": run.stats.variant,
                "question": case.question,
                "gold_answer": case.answer,
                "prediction": run.answer,
                "answer_session_ids": case.answer_session_ids,
                "retrieved_answer_session_hit": run.retrieval.answer_session_hit,
                "retrieved_answer_session_any_hit": run.retrieval.answer_session_hit,
                "retrieved_answer_session_all_hit": run.retrieval.answer_session_all_hit,
                "retrieved_answer_session_recall": run.retrieval.answer_session_recall,
            }
        ],
    )


def _print_case_progress(stats: QuestionStats) -> None:
    print(
        f"{stats.variant}: question={stats.question_id} "
        f"calls={stats.deepseek_call_count} deepseek_tokens={stats.total_deepseek_tokens}",
        flush=True,
    )


def run_case(
    config: DemoConfig,
    case: QuestionCase,
    variant: str,
    llm: Any,
    embedder: Any,
    compressor: Any,
    limiter: InflightLimiter | None = None,
    summarizer: Any | None = None,
) -> CaseRun:
    case_started = time.perf_counter()
    limiter = limiter or InflightLimiter(config.max_inflight_deepseek)
    memory = build_memory(
        config,
        case,
        variant,
        llm,
        embedder,
        compressor,
        limiter,
        summarizer=summarizer,
    )
    return run_case_with_memory(
        config,
        case,
        variant,
        memory,
        llm,
        embedder,
        limiter,
        include_build_records=True,
        case_started=case_started,
    )


def build_memory(
    config: DemoConfig,
    case: QuestionCase,
    variant: str,
    llm: Any,
    embedder: Any,
    compressor: Any,
    limiter: InflightLimiter,
    summarizer: Any | None = None,
) -> MemoryBuild:
    metrics = BuildMetrics()
    spec = _variant_spec(config, variant)
    llm_records: list[DeepSeekCallRecord] = []
    build_started = time.perf_counter()
    leaves = build_leaf_nodes(case)
    retrieval_leaf_mode = _effective_leaf_text_mode(
        config.retrieval_leaf_text, spec, case, phase="retrieval"
    )
    _embed_nodes(
        leaves,
        embedder,
        case.question_id,
        variant,
        attr=_leaf_embedding_attr(config, case, retrieval_leaf_mode),
    )
    summaries: list[SummaryNode] = []
    roots: list[SummaryNode] = []
    edges: list[GraphEdge] = []

    if spec.tree_mode != "raw_rag":
        summaries, roots = _build_summary_roots(
            config,
            case,
            leaves,
            variant,
            spec,
            llm,
            compressor,
            llm_records,
            metrics,
            limiter,
            summarizer,
        )
        if config.enable_speaker_profiles:
            speaker_profiles = _build_speaker_profile_roots(
                config,
                case,
                roots,
                variant,
                spec,
                llm,
                llm_records,
                metrics,
                limiter,
            )
            summaries.extend(speaker_profiles)
            roots.extend(speaker_profiles)
        _embed_nodes(summaries, embedder, case.question_id, variant, attr="retrieval_text")
        if spec.graph:
            graph_roots = summaries if config.enable_multilevel_summary_retrieval else roots
            edges = _build_root_graph(
                graph_roots,
                config.graph_neighbor_k,
                enable_typed_edges=config.enable_typed_root_edges,
            )
    build_latency = time.perf_counter() - build_started
    return MemoryBuild(
        leaves=leaves,
        summaries=summaries,
        roots=roots,
        edges=edges,
        llm_records=llm_records,
        metrics=metrics,
        build_latency_sec=build_latency,
    )


def run_case_with_memory(
    config: DemoConfig,
    case: QuestionCase,
    variant: str,
    memory: MemoryBuild,
    llm: Any,
    embedder: Any,
    limiter: InflightLimiter,
    *,
    include_build_records: bool,
    case_started: float | None = None,
) -> CaseRun:
    case_started = case_started if case_started is not None else time.perf_counter()
    spec = _variant_spec(config, variant)
    answer_metrics = BuildMetrics()
    llm_records: list[DeepSeekCallRecord] = list(memory.llm_records) if include_build_records else []
    leaves, summaries, roots, edges = _clone_memory_for_case(memory, case.question_id)
    retrieval_roots = summaries if config.enable_multilevel_summary_retrieval else roots
    retrieval = _retrieve(
        config=config,
        case=case,
        variant=variant,
        leaves=leaves,
        roots=retrieval_roots,
        edges=edges,
        embedder=embedder,
        graph_enabled=spec.graph,
        hybrid_retrieval=spec.hybrid_retrieval,
        enhanced_retrieval=spec.enhanced_retrieval,
        enhanced_qa=spec.enhanced_qa,
    )
    answer_started = time.perf_counter()
    answer_result = _tracked_chat(
        llm,
        limiter,
        answer_metrics,
        question_id=case.question_id,
        variant=variant,
        stage="answer_qa",
        thinking_mode="none",
        messages=_answer_messages(case, retrieval.context_text, enhanced=spec.enhanced_qa),
        max_tokens=config.qa_max_tokens,
    )
    llm_records.append(answer_result.record)
    answer_latency = time.perf_counter() - answer_started
    build_metrics = memory.metrics if include_build_records else BuildMetrics()
    stats = build_question_stats(
        question_id=case.question_id,
        variant=variant,
        session_count=len(case.haystack_session_ids),
        leaf_count=len(leaves),
        summary_count=len(summaries),
        edge_count=len(edges),
        records=llm_records,
        build_latency_sec=memory.build_latency_sec if include_build_records else 0.0,
        retrieval_latency_sec=retrieval.latency_sec,
        answer_latency_sec=answer_latency,
        answer_session_hit=retrieval.answer_session_hit,
        answer_session_all_hit=retrieval.answer_session_all_hit,
        answer_session_recall=retrieval.answer_session_recall,
        retrieved_answer_session_count=retrieval.retrieved_answer_session_count,
        gold_answer_session_count=retrieval.gold_answer_session_count,
        wall_time_sec=time.perf_counter() - case_started,
        summary_parse_error_count=build_metrics.summary_parse_error_count,
        summary_truncation_count=build_metrics.summary_truncation_count,
        ready_job_counts=build_metrics.ready_job_counts,
        peak_inflight_deepseek=max(
            build_metrics.peak_inflight_deepseek,
            answer_metrics.peak_inflight_deepseek,
        ),
    )
    return CaseRun(
        leaves=leaves,
        summaries=summaries,
        edges=edges,
        retrieval=retrieval,
        answer=answer_result.text,
        llm_records=llm_records,
        stats=stats,
    )


def _clone_memory_for_case(
    memory: MemoryBuild,
    question_id: str,
) -> tuple[list[LeafNode], list[SummaryNode], list[SummaryNode], list[GraphEdge]]:
    id_map: dict[str, str] = {}

    def remap(node_id: str) -> str:
        existing = id_map.get(node_id)
        if existing is not None:
            return existing
        suffix = node_id.split(":", 1)[1] if ":" in node_id else node_id
        mapped = f"{question_id}:{suffix}"
        id_map[node_id] = mapped
        return mapped

    leaves = [
        LeafNode(
            node_id=remap(leaf.node_id),
            question_id=question_id,
            session_id=leaf.session_id,
            session_date=leaf.session_date,
            turn_index=leaf.turn_index,
            raw_text=leaf.raw_text,
            user_text=leaf.user_text,
            message_count=leaf.message_count,
            retrieval_text=leaf.retrieval_text or leaf.raw_text,
            embedding=list(leaf.embedding) if leaf.embedding is not None else None,
        )
        for leaf in memory.leaves
    ]
    summaries = [
        SummaryNode(
            node_id=remap(summary.node_id),
            question_id=question_id,
            session_id=summary.session_id,
            session_date=summary.session_date,
            level=summary.level,
            child_ids=[remap(node_id) for node_id in summary.child_ids],
            leaf_ids=[remap(node_id) for node_id in summary.leaf_ids],
            summary=summary.summary,
            retrieval_text=summary.retrieval_text or summary.summary,
            anchor_terms=summary.anchor_terms or _summary_anchor_terms(
                summary.parsed_summary,
                summary.raw_summary_text or summary.summary,
                summary.session_date,
            ),
            summary_mode=summary.summary_mode,
            summary_schema_version=summary.summary_schema_version,
            parsed_summary=summary.parsed_summary,
            raw_summary_text=summary.raw_summary_text,
            truncated=summary.truncated,
            parse_error=summary.parse_error,
            source_level=summary.source_level,
            embedding=list(summary.embedding) if summary.embedding is not None else None,
        )
        for summary in memory.summaries
    ]
    summary_by_id = {summary.node_id: summary for summary in summaries}
    root_ids = {remap(root.node_id) for root in memory.roots}
    roots = [summary_by_id[node_id] for node_id in root_ids if node_id in summary_by_id]
    edges = [
        GraphEdge(
            src=remap(edge.src),
            dst=remap(edge.dst),
            score=edge.score,
            relation=edge.relation,
        )
        for edge in memory.edges
    ]
    return leaves, summaries, roots, edges


def _variant_spec(config: DemoConfig, variant: str) -> VariantSpec:
    spec = VARIANT_SPECS[variant]
    if config.tree_mode is None or spec.tree_mode == "raw_rag":
        return spec
    return VariantSpec(
        tree_mode=config.tree_mode,
        compression=spec.compression,
        graph=spec.graph,
        fanout_k=None,
        summary_max_tokens=spec.summary_max_tokens,
        summary_schema=spec.summary_schema,
        build_leaf_text=spec.build_leaf_text,
        retrieval_leaf_text=spec.retrieval_leaf_text,
        raw_question_types=spec.raw_question_types,
        hybrid_retrieval=spec.hybrid_retrieval,
        local_summary=spec.local_summary,
        default_summarizer_model=spec.default_summarizer_model,
        enhanced_retrieval=spec.enhanced_retrieval,
        enhanced_qa=spec.enhanced_qa,
    )


def _summary_schema(config: DemoConfig, spec: VariantSpec) -> str:
    return config.summary_schema or spec.summary_schema


def _summary_max_tokens(config: DemoConfig, spec: VariantSpec) -> int:
    return spec.summary_max_tokens or config.session_summary_max_tokens


def _uses_local_summary(config: DemoConfig, spec: VariantSpec) -> bool:
    if config.summarizer_kind == "qwen_local":
        return True
    if config.summarizer_kind in {"none", "llmlingua2"}:
        return False
    return spec.local_summary


def _leaf_text_mode(config_mode: str, variant_mode: str) -> str:
    return variant_mode if config_mode == "auto" else config_mode


def _effective_leaf_text_mode(config_mode: str, spec: VariantSpec, case: QuestionCase, phase: str) -> str:
    variant_mode = spec.build_leaf_text if phase == "build" else spec.retrieval_leaf_text
    mode = _leaf_text_mode(config_mode, variant_mode)
    if config_mode == "auto" and _has_explicit_speakers(case):
        return "raw"
    if config_mode == "auto" and case.question_type in spec.raw_question_types:
        return "raw"
    return mode


def _has_explicit_speakers(case: QuestionCase) -> bool:
    return any(
        bool(str(message.get("speaker", "")).strip())
        for session in case.haystack_sessions
        for message in session
    )


def _leaf_text_attr(mode: str) -> str:
    return "user_text" if mode == "user_only" else "raw_text"


def _leaf_embedding_attr(config: DemoConfig, case: QuestionCase, mode: str) -> str:
    if mode == "user_only":
        return "user_text"
    if config.enable_speaker_retrieval_text and _has_explicit_speakers(case):
        return "retrieval_text"
    return "raw_text"


def _tracked_chat(
    llm: Any,
    limiter: InflightLimiter,
    metrics: BuildMetrics,
    **kwargs: Any,
) -> Any:
    with limiter.track(metrics):
        return llm.chat(**kwargs)


def _build_summary_roots(
    config: DemoConfig,
    case: QuestionCase,
    leaves: list[LeafNode],
    variant: str,
    spec: VariantSpec,
    llm: Any,
    compressor: Any,
    llm_records: list[DeepSeekCallRecord],
    metrics: BuildMetrics,
    limiter: InflightLimiter,
    summarizer: Any | None,
) -> tuple[list[SummaryNode], list[SummaryNode]]:
    if spec.tree_mode == "direct_session":
        return _build_direct_session_roots(
            config,
            case,
            leaves,
            variant,
            spec,
            llm,
            compressor,
            llm_records,
            metrics,
            limiter,
            summarizer,
        )
    return _build_legacy_kway_roots(
        config,
        case,
        leaves,
        variant,
        spec,
        llm,
        compressor,
        llm_records,
        metrics,
        limiter,
        summarizer,
        )


def _build_speaker_profile_roots(
    config: DemoConfig,
    case: QuestionCase,
    roots: list[SummaryNode],
    variant: str,
    spec: VariantSpec,
    llm: Any,
    llm_records: list[DeepSeekCallRecord],
    metrics: BuildMetrics,
    limiter: InflightLimiter,
) -> list[SummaryNode]:
    speakers = _explicit_speakers(case)
    if not speakers or not roots or spec.tree_mode == "raw_rag":
        return []

    timeline = _speaker_profile_timeline(roots, config.max_group_rough_tokens)
    profiles: list[SummaryNode] = []
    for speaker in speakers:
        result = _tracked_chat(
            llm,
            limiter,
            metrics,
            question_id=case.question_id,
            variant=variant,
            stage="build_summary_speaker_profile",
            thinking_mode="none",
            messages=_speaker_profile_messages(speaker, timeline),
            max_tokens=max(512, _summary_max_tokens(config, spec)),
            json_mode=True,
        )
        llm_records.append(result.record)
        parsed, parse_error = _parse_summary(result.text, "compact_memory_v2")
        rendered_summary = _render_summary(parsed, result.text, "compact_memory_v2")
        leaf_ids: list[str] = []
        for root in roots:
            leaf_ids.extend(root.leaf_ids)
        profiles.append(
            SummaryNode(
                node_id=f"{case.question_id}:profile:{_safe_node_part(speaker)}",
                question_id=case.question_id,
                session_id=f"profile:{speaker}",
                session_date=case.question_date,
                level=99,
                child_ids=[root.node_id for root in roots],
                leaf_ids=sorted(set(leaf_ids)),
                summary=rendered_summary,
                retrieval_text=_summary_retrieval_text(
                    rendered_summary,
                    parsed,
                    timeline,
                    case.question_date,
                ),
                anchor_terms=_summary_anchor_terms(parsed, timeline, case.question_date),
                summary_mode="speaker_profile",
                summary_schema_version="compact_memory_v2",
                parsed_summary=parsed,
                raw_summary_text=result.text,
                truncated=result.record.finish_reason == "length",
                parse_error=parse_error,
                source_level=99,
            )
        )
    return profiles


def _explicit_speakers(case: QuestionCase) -> list[str]:
    speakers: list[str] = []
    seen: set[str] = set()
    for session in case.haystack_sessions:
        for message in session:
            speaker = str(message.get("speaker", "")).strip()
            if not speaker or speaker in seen:
                continue
            speakers.append(speaker)
            seen.add(speaker)
    return speakers


def _safe_node_part(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug or "speaker"


def _speaker_profile_timeline(roots: list[SummaryNode], rough_token_limit: int) -> str:
    chunks: list[str] = []
    total = 0
    for root in sorted(roots, key=lambda item: (item.session_date or "", item.session_id)):
        chunk = (
            f"[Session {root.session_id} | {root.session_date or 'unknown'}]\n"
            f"{root.summary}"
        )
        token_count = rough_token_count(chunk)
        if chunks and total + token_count > rough_token_limit:
            break
        chunks.append(chunk)
        total += token_count
    return "\n\n".join(chunks)


def _speaker_profile_messages(speaker: str, timeline: str) -> list[dict[str, str]]:
    system_prompt = (
        "Build a speaker-specific long-term memory profile from timeline summaries. "
        "Return JSON only: {\"m\":[\"short profile fact\"],\"k\":[\"keyword\"]}. "
        "Only include facts about the named speaker. Do not transfer facts from another speaker. "
        "Preserve stable identity, relationships, family, work or education goals, preferences, "
        "recurring activities, dated events, counts, and current status. Include uncertainty when "
        "the timeline does not explicitly support a value. Use at most 18 short m strings and "
        "12 keywords."
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Speaker: {speaker}\n\nTimeline summaries:\n{timeline}",
        },
    ]


def _build_legacy_kway_roots(
    config: DemoConfig,
    case: QuestionCase,
    leaves: list[LeafNode],
    variant: str,
    spec: VariantSpec,
    llm: Any,
    compressor: Any,
    llm_records: list[DeepSeekCallRecord],
    metrics: BuildMetrics,
    limiter: InflightLimiter,
    summarizer: Any | None,
) -> tuple[list[SummaryNode], list[SummaryNode]]:
    all_summaries: list[SummaryNode] = []
    roots: list[SummaryNode] = []
    dates = dict(zip(case.haystack_session_ids, case.haystack_dates))
    fanout_k = spec.fanout_k or config.fanout_k
    current: dict[str, list[LeafNode | SummaryNode]] = {
        session_id: list(session_leaves)
        for session_id, session_leaves in group_by_session(leaves).items()
    }
    level = 1
    while current:
        jobs: list[SummaryJob] = []
        for session_id, nodes in current.items():
            for group_number, children in enumerate(_chunks(nodes, fanout_k)):
                is_leaf_stage = isinstance(children[0], LeafNode)
                jobs.append(
                    SummaryJob(
                        session_id=session_id,
                        session_date=dates.get(session_id),
                        children=children,
                        stage="build_summary_leaf" if is_leaf_stage else "build_summary_internal",
                        level=level,
                        group_number=group_number,
                        summary_mode="legacy_kway",
                        max_tokens=(
                            config.raw_group_summary_max_tokens
                            if is_leaf_stage
                            else config.legacy_internal_summary_max_tokens
                        ),
                    )
                )
        next_nodes = _run_summary_jobs(
            config,
            case,
            variant,
            spec,
            jobs,
            llm,
            compressor,
            llm_records,
            metrics,
            limiter,
            summarizer,
        )
        grouped = _group_summaries_by_session(next_nodes)
        current = {}
        for session_id, session_nodes in grouped.items():
            all_summaries.extend(session_nodes)
            if len(session_nodes) == 1:
                roots.append(session_nodes[0])
            else:
                current[session_id] = session_nodes
        level += 1
    return all_summaries, roots


def _build_direct_session_roots(
    config: DemoConfig,
    case: QuestionCase,
    leaves: list[LeafNode],
    variant: str,
    spec: VariantSpec,
    llm: Any,
    compressor: Any,
    llm_records: list[DeepSeekCallRecord],
    metrics: BuildMetrics,
    limiter: InflightLimiter,
    summarizer: Any | None,
) -> tuple[list[SummaryNode], list[SummaryNode]]:
    all_summaries: list[SummaryNode] = []
    roots: list[SummaryNode] = []
    pending_merge: dict[str, list[SummaryNode]] = {}
    dates = dict(zip(case.haystack_session_ids, case.haystack_dates))
    fanout_k = spec.fanout_k or config.fanout_k
    session_summary_max_tokens = _summary_max_tokens(config, spec)
    build_leaf_mode = _effective_leaf_text_mode(config.build_leaf_text, spec, case, phase="build")
    first_jobs: list[SummaryJob] = []
    direct_session_ids: set[str] = set()
    for session_id, session_leaves in group_by_session(leaves).items():
        raw_groups = _raw_leaf_groups(
            session_leaves,
            fanout_k,
            config.max_group_rough_tokens,
            build_leaf_mode,
        )
        direct = len(raw_groups) == 1
        if direct:
            direct_session_ids.add(session_id)
        for group_number, children in enumerate(raw_groups):
            first_jobs.append(
                SummaryJob(
                    session_id=session_id,
                    session_date=dates.get(session_id),
                    children=children,
                    stage="build_summary_session_direct" if direct else "build_summary_raw_group",
                    level=1,
                    group_number=group_number,
                    summary_mode="direct_session",
                    max_tokens=(
                        session_summary_max_tokens
                        if direct
                        else config.raw_group_summary_max_tokens
                    ),
                )
            )
    first_nodes = _run_summary_jobs(
        config,
        case,
        variant,
        spec,
        first_jobs,
        llm,
        compressor,
        llm_records,
        metrics,
        limiter,
        summarizer,
    )
    all_summaries.extend(first_nodes)
    for session_id, session_nodes in _group_summaries_by_session(first_nodes).items():
        if session_id in direct_session_ids:
            roots.extend(session_nodes)
            continue
        pending_merge[session_id] = session_nodes

    level = 2
    while pending_merge:
        jobs: list[SummaryJob] = []
        for session_id, nodes in pending_merge.items():
            for group_number, children in enumerate(_chunks(nodes, fanout_k)):
                jobs.append(
                    SummaryJob(
                        session_id=session_id,
                        session_date=dates.get(session_id),
                        children=children,
                        stage="build_summary_session_merge",
                        level=level,
                        group_number=group_number,
                        summary_mode="direct_session",
                        max_tokens=session_summary_max_tokens,
                    )
                )
        merge_nodes = _run_summary_jobs(
            config,
            case,
            variant,
            spec,
            jobs,
            llm,
            compressor,
            llm_records,
            metrics,
            limiter,
            summarizer,
        )
        all_summaries.extend(merge_nodes)
        pending_merge = {}
        for session_id, session_nodes in _group_summaries_by_session(merge_nodes).items():
            if len(session_nodes) == 1:
                roots.append(session_nodes[0])
            else:
                pending_merge[session_id] = session_nodes
        level += 1
    return all_summaries, roots


def _run_summary_jobs(
    config: DemoConfig,
    case: QuestionCase,
    variant: str,
    spec: VariantSpec,
    jobs: list[SummaryJob],
    llm: Any,
    compressor: Any,
    llm_records: list[DeepSeekCallRecord],
    metrics: BuildMetrics,
    limiter: InflightLimiter,
    summarizer: Any | None,
) -> list[SummaryNode]:
    if not jobs:
        return []
    metrics.ready_job_counts.append(
        {
            "level": jobs[0].level,
            "job_count": len(jobs),
            "stages": {
                stage: sum(job.stage == stage for job in jobs)
                for stage in sorted({job.stage for job in jobs})
            },
        }
    )
    worker_count = len(jobs) if config.summary_workers == 0 else min(config.summary_workers, len(jobs))
    if worker_count == 1:
        results = [
            _summarize_job(
                config, case, variant, spec, job, llm, compressor, metrics, limiter, summarizer
            )
            for job in jobs
        ]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(
                executor.map(
                    lambda job: _summarize_job(
                        config,
                        case,
                        variant,
                        spec,
                        job,
                        llm,
                        compressor,
                        metrics,
                        limiter,
                        summarizer,
                    ),
                    jobs,
                )
            )
    for node, record in results:
        if record is not None:
            llm_records.append(record)
        metrics.summary_parse_error_count += int(node.parse_error is not None)
        metrics.summary_truncation_count += int(node.truncated)
    return [node for node, _ in results]


def _summarize_job(
    config: DemoConfig,
    case: QuestionCase,
    variant: str,
    spec: VariantSpec,
    job: SummaryJob,
    llm: Any,
    compressor: Any,
    metrics: BuildMetrics,
    limiter: InflightLimiter,
    summarizer: Any | None,
) -> tuple[SummaryNode, DeepSeekCallRecord | None]:
    schema = _summary_schema(config, spec)
    child_text = _child_text(
        job.children,
        _leaf_text_mode(config.build_leaf_text, spec.build_leaf_text),
    )
    if spec.compression and config.summarizer_kind != "none" and not _uses_local_summary(config, spec):
        child_text = compressor.compress(
            child_text,
            question_id=case.question_id,
            variant=variant,
            stage=job.stage,
            chunk_rough_tokens=config.compressor_chunk_rough_tokens,
        )
    messages = _summary_messages(job.session_id, job.session_date, job.stage, child_text, schema)
    if _uses_local_summary(config, spec):
        if summarizer is None:
            raise RuntimeError("qwen_local summarizer is required for this variant")
        summary_result = summarizer.summarize(
            question_id=case.question_id,
            variant=variant,
            stage=job.stage,
            messages=messages,
            max_tokens=config.summary_token_budget,
            json_mode=True,
        )
        summary_text = summary_result.text
        record: DeepSeekCallRecord | None = None
        finish_reason = None
    else:
        result = _tracked_chat(
            llm,
            limiter,
            metrics,
            question_id=case.question_id,
            variant=variant,
            stage=job.stage,
            thinking_mode="none",
            messages=messages,
            max_tokens=job.max_tokens,
            json_mode=True,
        )
        summary_text = result.text
        record = result.record
        finish_reason = result.record.finish_reason
    parsed, parse_error = _parse_summary(summary_text, schema)
    rendered_summary = _render_summary(parsed, summary_text, schema)
    node = SummaryNode(
        node_id=(
            f"{case.question_id}:{job.session_id}:summary:"
            f"{job.stage}:l{job.level}:g{job.group_number}"
        ),
        question_id=case.question_id,
        session_id=job.session_id,
        session_date=job.session_date,
        level=job.level,
        child_ids=[child.node_id for child in job.children],
        leaf_ids=_leaf_ids(job.children),
        summary=rendered_summary,
        retrieval_text=_summary_retrieval_text(
            rendered_summary,
            parsed,
            child_text,
            job.session_date,
        ),
        anchor_terms=_summary_anchor_terms(parsed, child_text, job.session_date),
        summary_mode=job.summary_mode,
        summary_schema_version=schema,
        parsed_summary=parsed,
        raw_summary_text=summary_text,
        truncated=finish_reason == "length",
        parse_error=parse_error,
        source_level=job.level,
    )
    return node, record


def _retrieve(
    *,
    config: DemoConfig,
    case: QuestionCase,
    variant: str,
    leaves: list[LeafNode],
    roots: list[SummaryNode],
    edges: list[GraphEdge],
    embedder: Any,
    graph_enabled: bool,
    hybrid_retrieval: bool,
    enhanced_retrieval: bool,
    enhanced_qa: bool,
) -> RetrievedContext:
    started = time.perf_counter()
    query_vector = embedder.embed([case.question], question_id=case.question_id, variant=variant)[0]
    leaf_by_id = {leaf.node_id: leaf for leaf in leaves}
    root_by_id = {root.node_id: root for root in roots}
    selected_roots: list[SummaryNode] = []
    selected_leaves: list[LeafNode]
    retrieval_edges: list[GraphEdge] = []

    if hybrid_retrieval:
        selected_roots, selected_leaves, retrieval_edges = _retrieve_hybrid(
            config,
            case,
            leaves,
            roots,
            edges,
            query_vector,
            graph_enabled,
            enhanced_retrieval,
        )
    elif roots:
        ranked_roots = _rank_nodes(roots, query_vector)
        root_ids = [root.node_id for root in ranked_roots[: config.root_top_k]]
        if graph_enabled:
            root_ids, retrieval_edges = _expand_root_ids(root_ids, edges, config.graph_neighbor_k)
        selected_roots = [root_by_id[node_id] for node_id in root_ids if node_id in root_by_id]
        candidate_leaf_ids = {leaf_id for root in selected_roots for leaf_id in root.leaf_ids}
        candidate_leaves = [leaf_by_id[node_id] for node_id in candidate_leaf_ids if node_id in leaf_by_id]
        selected_leaves = _rank_leaves(
            candidate_leaves or leaves,
            query_vector,
            case.question,
            enhanced=enhanced_retrieval,
        )[: _effective_leaf_top_k(config, case)]
    else:
        selected_leaves = _rank_leaves(
            leaves, query_vector, case.question, enhanced=enhanced_retrieval
        )[: _effective_leaf_top_k(config, case)]
    selected_roots, selected_leaves = _fit_context_budget(
        selected_roots,
        selected_leaves,
        _evidence_context_budget(config, case, enhanced_qa),
    )
    retrieved_sessions = sorted({leaf.session_id for leaf in selected_leaves})
    answer_sessions = set(case.answer_session_ids)
    answer_session_count = len(set(retrieved_sessions) & answer_sessions)
    gold_answer_session_count = len(answer_sessions)
    answer_session_recall = (
        answer_session_count / gold_answer_session_count if gold_answer_session_count else 0.0
    )
    context = _context_text(selected_roots, selected_leaves)
    return RetrievedContext(
        question_id=case.question_id,
        variant=variant,
        summary_node_ids=[root.node_id for root in selected_roots],
        leaf_node_ids=[leaf.node_id for leaf in selected_leaves],
        edge_count=len(retrieval_edges),
        context_text=context,
        answer_session_hit=bool(answer_session_count),
        retrieved_session_ids=retrieved_sessions,
        latency_sec=time.perf_counter() - started,
        answer_session_all_hit=(
            gold_answer_session_count > 0 and answer_session_count == gold_answer_session_count
        ),
        answer_session_recall=answer_session_recall,
        retrieved_answer_session_count=answer_session_count,
        gold_answer_session_count=gold_answer_session_count,
    )


def _retrieve_hybrid(
    config: DemoConfig,
    case: QuestionCase | str,
    leaves: list[LeafNode],
    roots: list[SummaryNode],
    edges: list[GraphEdge],
    query_vector: list[float],
    graph_enabled: bool,
    enhanced: bool,
) -> tuple[list[SummaryNode], list[LeafNode], list[GraphEdge]]:
    question = case.question if isinstance(case, QuestionCase) else case
    question_type = case.question_type if isinstance(case, QuestionCase) else ""
    leaf_by_id = {leaf.node_id: leaf for leaf in leaves}
    root_by_id = {root.node_id: root for root in roots}
    roots_by_session = {root.session_id: root for root in roots}
    ranked_roots = _rank_nodes(roots, query_vector)
    root_ids = [root.node_id for root in ranked_roots[: config.root_candidate_k]]
    retrieval_edges: list[GraphEdge] = []
    if graph_enabled:
        root_ids, retrieval_edges = _expand_root_ids(root_ids, edges, config.graph_neighbor_k)
    candidate_leaf_ids: set[str] = set()
    for root_id in root_ids:
        root = root_by_id.get(root_id)
        if root is not None:
            candidate_leaf_ids.update(root.leaf_ids)
    global_leaf_ids = {
        leaf.node_id
        for leaf in _rank_leaves(leaves, query_vector, question, enhanced=enhanced)[
            : _effective_global_leaf_top_k(config, case)
        ]
    }
    candidate_leaves = [
        leaf_by_id[leaf_id]
        for leaf_id in candidate_leaf_ids | global_leaf_ids
        if leaf_id in leaf_by_id
    ]
    ranked_leaves = _rank_leaves(candidate_leaves or leaves, query_vector, question, enhanced=enhanced)
    seed_roots = (
        [root_by_id[root_id] for root_id in root_ids if root_id in root_by_id]
        if enhanced
        else ranked_roots[: config.root_top_k]
    )
    root_seed_leaves = _root_seed_leaves(
        seed_roots,
        leaf_by_id,
        query_vector,
        question,
        enhanced,
    )
    selected_leaves = _diversify_leaves(
        ranked_leaves,
        limit=_effective_leaf_top_k(config, case),
        per_session_k=_effective_per_session_leaf_k(config, case),
        seed_leaves=root_seed_leaves,
    )
    if enhanced:
        selected_leaves = _expand_selected_session_context(
            selected_leaves,
            leaves,
            question,
            question_type,
            _effective_leaf_top_k(config, case),
            explicit_speaker=(
                isinstance(case, QuestionCase) and _has_explicit_speakers(case)
                and config.enable_speaker_neighbor_window
            ),
        )
    context_sessions = {leaf.session_id for leaf in selected_leaves}
    summary_rank = {root.node_id: index for index, root in enumerate(ranked_roots)}
    selected_roots = sorted(
        [roots_by_session[session_id] for session_id in context_sessions if session_id in roots_by_session],
        key=lambda root: (summary_rank.get(root.node_id, len(summary_rank)), root.node_id),
    )
    if enhanced and len(selected_roots) < config.qa_summary_top_k:
        selected_root_ids = {root.node_id for root in selected_roots}
        for root in ranked_roots:
            if root.node_id in selected_root_ids:
                continue
            selected_roots.append(root)
            selected_root_ids.add(root.node_id)
            if len(selected_roots) >= config.qa_summary_top_k:
                break
    selected_roots = selected_roots[: config.qa_summary_top_k]
    return selected_roots, selected_leaves, retrieval_edges


def _effective_leaf_top_k(config: DemoConfig, case: QuestionCase | str) -> int:
    if isinstance(case, QuestionCase) and _has_explicit_speakers(case):
        return max(config.leaf_top_k, 24)
    return config.leaf_top_k


def _effective_global_leaf_top_k(config: DemoConfig, case: QuestionCase | str) -> int:
    if isinstance(case, QuestionCase) and _has_explicit_speakers(case):
        return max(config.global_leaf_top_k, 48)
    return config.global_leaf_top_k


def _effective_per_session_leaf_k(config: DemoConfig, case: QuestionCase | str) -> int:
    if isinstance(case, QuestionCase) and _has_explicit_speakers(case):
        return max(config.per_session_leaf_k, 3)
    return config.per_session_leaf_k


def _expand_selected_session_context(
    selected_leaves: list[LeafNode],
    leaves: list[LeafNode],
    question: str,
    question_type: str,
    limit: int,
    *,
    explicit_speaker: bool = False,
) -> list[LeafNode]:
    if not selected_leaves or limit <= len(selected_leaves) // 2:
        return selected_leaves
    grouped = group_by_session(leaves)
    expanded: list[LeafNode] = []
    seen: set[str] = set()

    def add(leaf: LeafNode) -> bool:
        if leaf.node_id in seen:
            return False
        expanded.append(leaf)
        seen.add(leaf.node_id)
        return len(expanded) >= limit

    if explicit_speaker:
        for leaf in selected_leaves:
            if add(leaf):
                return expanded
        sibling_sets: list[tuple[list[LeafNode], int]] = []
        for leaf in selected_leaves:
            siblings = sorted(grouped.get(leaf.session_id, []), key=lambda item: item.turn_index)
            index = next((idx for idx, item in enumerate(siblings) if item.node_id == leaf.node_id), -1)
            if index < 0:
                continue
            sibling_sets.append((siblings, index))
        for offset in (-1, 1, 2, 3):
            for siblings, index in sibling_sets:
                neighbor_index = index + offset
                if 0 <= neighbor_index < len(siblings) and add(siblings[neighbor_index]):
                    return expanded
        return expanded

    previous_chat = question_type == "single-session-assistant" or bool(
        re.search(
            r"previous (conversation|chat)|our previous|looking back|remind me what|finally decided",
            question,
            flags=re.IGNORECASE,
        )
    )
    if not previous_chat:
        return selected_leaves

    top_session = selected_leaves[0].session_id
    for sibling in sorted(grouped.get(top_session, []), key=lambda item: item.turn_index)[:8]:
        if add(sibling):
            return expanded

    for leaf in selected_leaves:
        if add(leaf):
            return expanded
        siblings = sorted(grouped.get(leaf.session_id, []), key=lambda item: item.turn_index)
        index = next((idx for idx, item in enumerate(siblings) if item.node_id == leaf.node_id), -1)
        if index < 0:
            continue
        window = siblings[max(0, index - 1) : min(len(siblings), index + 5)]
        for sibling in window:
            if add(sibling):
                return expanded

    for leaf in selected_leaves:
        if add(leaf):
            break
    return expanded


def _diversify_leaves(
    ranked_leaves: list[LeafNode],
    *,
    limit: int,
    per_session_k: int,
    seed_leaves: list[LeafNode] | None = None,
) -> list[LeafNode]:
    selected: list[LeafNode] = []
    selected_ids: set[str] = set()
    per_session: dict[str, int] = {}
    for leaf in seed_leaves or []:
        if leaf.node_id in selected_ids:
            continue
        if per_session.get(leaf.session_id, 0) >= per_session_k:
            continue
        selected.append(leaf)
        selected_ids.add(leaf.node_id)
        per_session[leaf.session_id] = per_session.get(leaf.session_id, 0) + 1
        if len(selected) >= limit:
            return selected
    for leaf in ranked_leaves:
        if leaf.node_id in selected_ids:
            continue
        if per_session.get(leaf.session_id, 0) >= per_session_k:
            continue
        selected.append(leaf)
        selected_ids.add(leaf.node_id)
        per_session[leaf.session_id] = per_session.get(leaf.session_id, 0) + 1
        if len(selected) >= limit:
            return selected
    for leaf in ranked_leaves:
        if leaf.node_id in selected_ids:
            continue
        selected.append(leaf)
        if len(selected) >= limit:
            break
    return selected


def _root_seed_leaves(
    roots: list[SummaryNode],
    leaf_by_id: dict[str, LeafNode],
    query_vector: list[float],
    question: str = "",
    enhanced: bool = False,
) -> list[LeafNode]:
    seed_leaves: list[LeafNode] = []
    for root in roots:
        children = [leaf_by_id[leaf_id] for leaf_id in root.leaf_ids if leaf_id in leaf_by_id]
        ranked_children = _rank_leaves(children, query_vector, question, enhanced=enhanced)
        if ranked_children:
            seed_leaves.append(ranked_children[0])
    return seed_leaves


def _build_root_graph(
    roots: list[SummaryNode],
    graph_neighbor_k: int,
    *,
    enable_typed_edges: bool = False,
) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    seen: set[tuple[str, str, str]] = set()
    temporal_roots = sorted(roots, key=lambda root: (root.session_date or "", root.session_id))
    for left, right in zip(temporal_roots, temporal_roots[1:]):
        _add_edge(edges, seen, left.node_id, right.node_id, 1.0, "temporal_neighbor")
    root_terms = {root.node_id: _summary_term_set(root.retrieval_text or root.summary) for root in roots}
    keyword_neighbors: dict[str, list[tuple[float, SummaryNode]]] = {root.node_id: [] for root in roots}
    typed_neighbors: dict[str, list[tuple[float, SummaryNode, str]]] = {
        root.node_id: [] for root in roots
    }
    root_anchors = (
        {root.node_id: _typed_anchor_sets(root) for root in roots}
        if enable_typed_edges
        else {}
    )
    for index, root in enumerate(roots):
        terms = root_terms[root.node_id]
        anchors = root_anchors.get(root.node_id, {})
        for candidate in roots[index + 1 :]:
            if enable_typed_edges:
                candidate_anchors = root_anchors.get(candidate.node_id, {})
                for relation, score in _typed_anchor_edge_scores(anchors, candidate_anchors):
                    typed_neighbors[root.node_id].append((score, candidate, relation))
                    typed_neighbors[candidate.node_id].append((score, root, relation))
            if not terms:
                continue
            candidate_terms = root_terms[candidate.node_id]
            if not candidate_terms:
                continue
            shared = terms & candidate_terms
            if len(shared) < 2:
                continue
            score = min(0.99, 0.45 + len(shared) / max(len(terms | candidate_terms), 1))
            keyword_neighbors[root.node_id].append((score, candidate))
            keyword_neighbors[candidate.node_id].append((score, root))
    for root in roots:
        per_relation: dict[str, int] = {}
        for score, candidate, relation in sorted(
            typed_neighbors[root.node_id],
            key=lambda item: (item[0], item[2], item[1].node_id),
            reverse=True,
        ):
            if per_relation.get(relation, 0) >= graph_neighbor_k:
                continue
            _add_edge(edges, seen, root.node_id, candidate.node_id, score, relation)
            per_relation[relation] = per_relation.get(relation, 0) + 1
    for root in roots:
        for score, candidate in sorted(
            keyword_neighbors[root.node_id],
            key=lambda item: (item[0], item[1].node_id),
            reverse=True,
        )[:graph_neighbor_k]:
            _add_edge(edges, seen, root.node_id, candidate.node_id, score, "keyword_neighbor")
    for root in roots:
        neighbors = sorted(
            (
                (cosine_similarity(root.embedding, candidate.embedding), candidate)
                for candidate in roots
                if candidate.node_id != root.node_id
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        for score, candidate in neighbors[:graph_neighbor_k]:
            _add_edge(edges, seen, root.node_id, candidate.node_id, score, "semantic_neighbor")
    return edges


def _typed_anchor_sets(root: SummaryNode) -> dict[str, set[str]]:
    anchors = root.anchor_terms or _summary_anchor_terms(
        root.parsed_summary,
        root.raw_summary_text or root.retrieval_text or root.summary,
        root.session_date,
    )
    typed: dict[str, set[str]] = {}
    for key, values in anchors.items():
        cleaned = {
            _normalize_anchor(value)
            for value in values
            if _normalize_anchor(value)
        }
        if cleaned:
            typed[key] = cleaned
    return typed


def _normalize_anchor(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value).strip(" \t\r\n.,;:")).casefold()
    if len(text) < 2 or text in _SUMMARY_CUE_STOPWORDS:
        return ""
    return text


def _typed_anchor_edge_scores(
    left: dict[str, set[str]],
    right: dict[str, set[str]],
) -> list[tuple[str, float]]:
    scores: list[tuple[str, float]] = []
    relation_specs = (
        ("entities", "entity_neighbor", 0.86, 0.04, 1),
        ("times", "time_neighbor", 0.76, 0.04, 1),
        ("actions", "event_neighbor", 0.68, 0.04, 2),
        ("state_phrases", "state_neighbor", 0.74, 0.03, 1),
        ("keywords", "keyword_neighbor", 0.58, 0.03, 2),
    )
    for key, relation, base, step, threshold in relation_specs:
        shared = left.get(key, set()) & right.get(key, set())
        if len(shared) < threshold:
            continue
        scores.append((relation, min(0.97, base + len(shared) * step)))
    if (left.get("actions", set()) & right.get("actions", set())) and (
        left.get("entities", set()) & right.get("entities", set())
        or left.get("keywords", set()) & right.get("keywords", set())
    ):
        scores.append(("update_neighbor", 0.82))
    return scores


def _add_edge(
    edges: list[GraphEdge],
    seen: set[tuple[str, str, str]],
    src: str,
    dst: str,
    score: float,
    relation: str,
) -> None:
    pair = tuple(sorted((src, dst)))
    key = (pair[0], pair[1], relation)
    if key in seen:
        return
    edges.append(GraphEdge(src=pair[0], dst=pair[1], score=score, relation=relation))  # type: ignore[arg-type]
    seen.add(key)


def _expand_root_ids(
    root_ids: list[str],
    edges: list[GraphEdge],
    graph_neighbor_k: int,
) -> tuple[list[str], list[GraphEdge]]:
    expanded = list(root_ids)
    seen = set(root_ids)
    used_edges: list[GraphEdge] = []
    neighbor_counts = {root_id: 0 for root_id in root_ids}
    for edge in sorted(edges, key=_edge_expansion_sort_key, reverse=True):
        for source, destination in ((edge.src, edge.dst), (edge.dst, edge.src)):
            if source in neighbor_counts and destination not in seen:
                if neighbor_counts[source] >= graph_neighbor_k:
                    continue
                expanded.append(destination)
                seen.add(destination)
                used_edges.append(edge)
                neighbor_counts[source] += 1
    return expanded, used_edges


_EDGE_RELATION_EXPANSION_BONUS = {
    "update_neighbor": 0.08,
    "entity_neighbor": 0.06,
    "state_neighbor": 0.06,
    "event_neighbor": 0.04,
    "time_neighbor": 0.03,
    "keyword_neighbor": 0.01,
    "temporal_neighbor": 0.0,
    "semantic_neighbor": 0.0,
}


def _edge_expansion_sort_key(edge: GraphEdge) -> tuple[float, float, str, str, str]:
    return (
        edge.score + _EDGE_RELATION_EXPANSION_BONUS.get(edge.relation, 0.0),
        edge.score,
        edge.relation,
        edge.src,
        edge.dst,
    )


def _embed_nodes(nodes: list[Any], embedder: Any, question_id: str, variant: str, attr: str) -> None:
    if not nodes:
        return
    vectors = embedder.embed(
        [getattr(node, attr) for node in nodes], question_id=question_id, variant=variant
    )
    for node, vector in zip(nodes, vectors):
        node.embedding = vector


def _rank_nodes(nodes: list[Any], query_vector: list[float]) -> list[Any]:
    return sorted(
        nodes,
        key=lambda node: (cosine_similarity(node.embedding, query_vector), node.node_id),
        reverse=True,
    )


def _rank_leaves(
    leaves: list[LeafNode],
    query_vector: list[float],
    question: str,
    *,
    enhanced: bool,
) -> list[LeafNode]:
    if not enhanced:
        return _rank_nodes(leaves, query_vector)
    query_terms = _important_query_terms(question)
    update_query = _is_update_sensitive_question(question)
    return sorted(
        leaves,
        key=lambda leaf: (
            cosine_similarity(leaf.embedding, query_vector)
            + _lexical_overlap_score(leaf.raw_text, query_terms)
            + _update_signal_score(leaf.raw_text if update_query else ""),
            leaf.node_id,
        ),
        reverse=True,
    )


def _important_query_terms(question: str) -> set[str]:
    stop = {
        "how",
        "many",
        "much",
        "what",
        "when",
        "where",
        "did",
        "have",
        "the",
        "and",
        "for",
        "currently",
        "total",
        "recently",
        "多少",
        "几个",
        "什么",
        "当前",
        "现在",
        "最近",
        "总共",
    }
    terms = {
        token.lower()
        for token in re.findall(r"[\w\u4e00-\u9fff]+", question)
        if len(token) > 2 and token.lower() not in stop
    }
    return terms


def _lexical_overlap_score(text: str, query_terms: set[str]) -> float:
    if not query_terms:
        return 0.0
    lowered = text.lower()
    hits = sum(term in lowered for term in query_terms)
    return min(0.18, hits * 0.04)


def _is_update_sensitive_question(question: str) -> bool:
    return bool(
        re.search(
            r"currently|how many|how much|total|recent|since|now|current|arrive|arrival|leave|left|cost|spent|最近|当前|现在|多少|几个|总共|花了|到达|离开",
            question,
            flags=re.IGNORECASE,
        )
    )


def _update_signal_score(text: str) -> float:
    if not text:
        return 0.0
    patterns = (
        r"cancel|subscribe|subscription|currently|now|current|no longer|instead|changed",
        r"buy|bought|purchase|purchased|cost|spent|total|\$\d+",
        r"arrive|arrival|reach|reached|leave|left|\b\d{1,2}(:\d{2})?\s*(am|pm)\b",
        r"取消|订阅|现在|当前|不再|购买|买了|花了|总共|到达|抵达|离开",
    )
    hits = sum(bool(re.search(pattern, text, flags=re.IGNORECASE)) for pattern in patterns)
    return min(0.24, hits * 0.06)


_SUMMARY_CUE_STOPWORDS = {
    "assistant",
    "build",
    "child",
    "counts",
    "date",
    "dates",
    "events",
    "facts",
    "keywords",
    "memory",
    "session",
    "summary",
    "updates",
    "user",
}

_SUMMARY_ANCHOR_KEYS = (
    "entities",
    "times",
    "quantities",
    "actions",
    "state_phrases",
    "keywords",
)


_SUMMARY_ACTION_CUES = (
    "accepted",
    "arrived",
    "arrival",
    "attended",
    "bought",
    "canceled",
    "cancelled",
    "changed",
    "completed",
    "cost",
    "current",
    "currently",
    "decided",
    "delivered",
    "flight",
    "joined",
    "left",
    "moved",
    "ordered",
    "planned",
    "purchased",
    "read",
    "recommended",
    "replaced",
    "returned",
    "spent",
    "subscribed",
    "subscription",
    "visited",
)


def _summary_retrieval_text(
    rendered_summary: str,
    parsed: dict[str, Any] | None,
    source_text: str,
    session_date: str | None,
) -> str:
    anchors = _summary_anchor_terms(parsed, source_text, session_date)
    cues = _summary_search_cues(parsed, source_text, anchors=anchors)
    anchor_text = _summary_anchor_text(anchors)
    blocks = []
    if session_date:
        blocks.append(f"Session date: {session_date}")
    if rendered_summary.strip():
        blocks.append(rendered_summary.strip())
    if anchor_text:
        blocks.append("Anchor terms:\n" + anchor_text)
    if cues:
        blocks.append("Search cues: " + "; ".join(cues))
    return "\n".join(blocks)


def _summary_anchor_terms(
    parsed: dict[str, Any] | None,
    source_text: str,
    session_date: str | None,
    limit_per_type: int = 32,
) -> dict[str, list[str]]:
    anchors: dict[str, list[str]] = {
        key: []
        for key in _SUMMARY_ANCHOR_KEYS
    }
    keyword_candidates: list[str] = []
    if parsed:
        for key in ("keywords", "k"):
            keyword_candidates.extend(_summary_string_list(parsed.get(key)))
    anchors["keywords"] = _dedupe_cues(keyword_candidates, limit_per_type)
    anchors["entities"] = _dedupe_cues(_proper_name_cues(source_text), limit_per_type)
    time_candidates = _numeric_time_cues(source_text)
    if session_date:
        time_candidates.append(session_date)
    anchors["times"] = _dedupe_cues(time_candidates, limit_per_type)
    anchors["quantities"] = _dedupe_cues(_quantity_cues(source_text), limit_per_type)
    lowered = source_text.lower()
    anchors["actions"] = _dedupe_cues(
        [cue for cue in _SUMMARY_ACTION_CUES if cue in lowered],
        limit_per_type,
    )
    anchors["state_phrases"] = _dedupe_cues(
        [
            *_state_phrase_cues(source_text),
            *_speaker_attribute_cues(source_text),
            *_action_object_cues(source_text),
        ],
        limit_per_type,
    )
    return {key: value for key, value in anchors.items() if value}


def _summary_anchor_text(anchors: dict[str, list[str]]) -> str:
    lines: list[str] = []
    labels = {
        "entities": "Entities",
        "times": "Times",
        "quantities": "Quantities",
        "actions": "Actions",
        "state_phrases": "State phrases",
        "keywords": "Keywords",
    }
    for key in _SUMMARY_ANCHOR_KEYS:
        values = anchors.get(key) or []
        if values:
            lines.append(f"{labels[key]}: " + "; ".join(values[:16]))
    return "\n".join(lines)


def _summary_search_cues(
    parsed: dict[str, Any] | None,
    source_text: str,
    limit: int = 48,
    *,
    anchors: dict[str, list[str]] | None = None,
) -> list[str]:
    candidates: list[str] = []
    if anchors is not None:
        for key in _SUMMARY_ANCHOR_KEYS:
            candidates.extend(anchors.get(key) or [])
        return _dedupe_cues(candidates, limit)
    if parsed:
        for key in ("keywords", "k"):
            candidates.extend(_summary_string_list(parsed.get(key)))
    candidates.extend(_proper_name_cues(source_text))
    candidates.extend(_numeric_time_cues(source_text))
    candidates.extend(_quantity_cues(source_text))
    candidates.extend(_state_phrase_cues(source_text))
    lowered = source_text.lower()
    candidates.extend(cue for cue in _SUMMARY_ACTION_CUES if cue in lowered)
    return _dedupe_cues(candidates, limit)


def _proper_name_cues(text: str) -> list[str]:
    cues: list[str] = []
    for match in re.finditer(r'"([^"\n]{3,80})"', text):
        cue = match.group(1).strip()
        if cue and cue.lower() not in _SUMMARY_CUE_STOPWORDS:
            cues.append(cue)
    for match in re.finditer(r"(?<![A-Za-z])'([^'\n]{3,80})'(?![A-Za-z])", text):
        cue = match.group(1).strip()
        if cue and cue.lower() not in _SUMMARY_CUE_STOPWORDS:
            cues.append(cue)
    for match in re.finditer(
        r"\b[A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*){0,4}\b",
        text,
    ):
        cue = match.group(0).strip(" .,:;")
        normalized = cue.lower()
        if len(cue) < 3 or normalized in _SUMMARY_CUE_STOPWORDS:
            continue
        cues.append(cue)
    return cues


def _state_phrase_cues(text: str) -> list[str]:
    cues: list[str] = []
    patterns = (
        r"\b(?:currently|now|still)\s+(?:reading|devouring|using|keeping|storing|wearing|owning|have|having)\s+([^.;\n|]{3,90})",
        r"\b(?:stored|keeping|kept|storing)\s+(?:it|them|my\s+[^.;\n|]{2,40}?)\s+(?:in|on|under|at)\s+([^.;\n|]{3,80})",
        r"\b(?:moved|switched|changed|replaced)\s+(?:to|into|from)\s+([^.;\n|]{3,80})",
        r"\b(?:subscribed to|canceled|cancelled|bought|purchased|ordered)\s+([^.;\n|]{3,80})",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            cue = _normalize_summary_state_phrase(match.group(1))
            if cue:
                cues.append(cue)
    return cues


def _speaker_attribute_cues(text: str) -> list[str]:
    cues: list[str] = []
    speaker_pattern = r"[A-Z][A-Za-z.'-]{1,40}"
    verb_pattern = (
        r"is|was|has|had|wants?|likes?|loves?|enjoys?|prefers?|works|volunteers|"
        r"reads?|ran|went|moved|camped|attended|signed|joined|started|finished|"
        r"plans?|planned|studies|studying|pursues?|pursuing"
    )
    for match in re.finditer(
        rf"\b({speaker_pattern})\s+({verb_pattern})\b\s+([^.;\n|]{{2,90}})",
        text,
    ):
        speaker = match.group(1).strip()
        verb = match.group(2).strip()
        obj = _normalize_summary_state_phrase(match.group(3))
        if obj:
            cues.append(f"{speaker} {verb} {obj}")
    for match in re.finditer(
        rf"\b({speaker_pattern})'s\s+([^.;\n|]{{3,90}})",
        text,
    ):
        speaker = match.group(1).strip()
        attr = _normalize_summary_state_phrase(match.group(2))
        if attr:
            cues.append(f"{speaker}'s {attr}")
    return cues


def _action_object_cues(text: str) -> list[str]:
    cues: list[str] = []
    action_pattern = (
        r"went to|going to|go to|attended|ran|signed up for|planning(?: on)?|"
        r"moved from|moved to|read|recommended|camped|camping|visited|made|"
        r"painted|created|joined|started|finished|volunteered at|works at|"
        r"pursue|pursuing|studying"
    )
    for match in re.finditer(
        rf"\b({action_pattern})\b\s+([^.;\n|]{{3,90}})",
        text,
        flags=re.IGNORECASE,
    ):
        action = re.sub(r"\s+", " ", match.group(1).strip().lower())
        obj = _normalize_summary_state_phrase(match.group(2))
        if obj:
            cues.append(f"{action} {obj}")
    return cues


def _normalize_summary_state_phrase(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip(" \t\r\n.,;:!?*\"'()[]"))
    value = re.sub(
        r"\s+(?:and|but|because|which|that|while|so|by the way)\b.*$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"^(?:my|a|an|the)\s+", "", value, flags=re.IGNORECASE)
    if len(value) < 3 or value.casefold() in _SUMMARY_CUE_STOPWORDS:
        return ""
    return value[:90]


def _numeric_time_cues(text: str) -> list[str]:
    patterns = (
        r"\$\s?\d+(?:[,.]\d+)?",
        r"\b\d{1,2}:\d{2}\s?(?:AM|PM|am|pm)?\b",
        r"\b\d{1,2}\s?(?:AM|PM|am|pm)\b",
        r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
        r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b",
        r"\b\d+(?:[,.]\d+)?\s?(?:minutes?|hours?|days?|weeks?|months?|years?)\b",
    )
    cues: list[str] = []
    for pattern in patterns:
        cues.extend(match.group(0).strip() for match in re.finditer(pattern, text))
    return cues


def _quantity_cues(text: str) -> list[str]:
    patterns = (
        r"\$\s?\d+(?:[,.]\d+)?",
        r"\b\d+(?:[,.]\d+)?\s?(?:minutes?|hours?|days?|weeks?|months?|years?|people|persons|items|tickets|books|episodes|classes|sessions|miles|km|kilometers?)\b",
        r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:minutes?|hours?|days?|weeks?|months?|years?|people|persons|items|tickets|books|episodes|classes|sessions)\b",
    )
    cues: list[str] = []
    for pattern in patterns:
        cues.extend(match.group(0).strip() for match in re.finditer(pattern, text, flags=re.IGNORECASE))
    return cues


def _dedupe_cues(candidates: list[str], limit: int) -> list[str]:
    cues: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        cue = re.sub(r"\s+", " ", str(candidate).strip(" \t\r\n.,;:"))
        if len(cue) < 2:
            continue
        key = cue.casefold()
        if key in seen or key in _SUMMARY_CUE_STOPWORDS:
            continue
        seen.add(key)
        cues.append(cue)
        if len(cues) >= limit:
            break
    return cues


def _summary_term_set(text: str) -> set[str]:
    terms = {
        token.casefold()
        for token in re.findall(r"[\w&.'-]+", text)
        if len(token) >= 3 and token.casefold() not in _SUMMARY_CUE_STOPWORDS
    }
    terms.update(cue.casefold() for cue in _numeric_time_cues(text))
    return terms


def _summary_messages(
    session_id: str,
    session_date: str | None,
    stage: str,
    child_text: str,
    schema: str,
) -> list[dict[str, str]]:
    if schema == "multilingual_memory_v1":
        system_prompt = (
            "Extract multilingual user memory as JSON only with exactly these keys: "
            '{"facts":[],"events":[],"counts":[],"dates":[],"updates":[],"keywords":[]}. '
            "Use short atomic strings in the original language when possible. Preserve numbers, "
            "dates, times, costs, negations, cancellations, purchases, subscriptions, arrivals, "
            "departures, current state, and updates. Ignore assistant filler and generic advice. "
            "Use at most 8 facts/events total, 6 counts/dates/updates total, and 10 keywords."
        )
    elif schema == "compact_memory_v2":
        system_prompt = (
            'Extract memory as JSON only: {"m":["short memory fact or update"],'
            '"k":["keyword"]}. Keep user facts, preferences, plans, purchases, visits, '
            "events, counts, costs, dates, negations, and updates. Also keep assistant-provided "
            "answers, recommendations, named entities, methods, options, tables, numbers, and "
            "rubrics that could answer a later 'previous conversation' question. Use at most 8 "
            "short atomic m strings and 8 keywords. Drop only unrelated filler and repeated wording."
        )
    else:
        system_prompt = (
            "Extract compact memory as a JSON object only. Use this schema exactly: "
            '{"compact_summary":"short session or group summary","facts":["short durable memory fact"],'
            '"updates":["short update or contradiction"],"time_anchors":["short temporal anchor"],'
            '"keywords":["keyword"]}. Use short strings. Keep user facts, preferences, plans, '
            "purchases, visits, events, updates, negations, and temporal anchors. Drop assistant "
            "filler and repeated wording. Return empty arrays when a list has no content."
        )
    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": (
                f"Build stage: {stage}\nSession: {session_id}\n"
                f"Date: {session_date or 'unknown'}\nChild memory:\n{child_text}"
            ),
        },
    ]


def _answer_messages(case: QuestionCase, context: str, *, enhanced: bool = False) -> list[dict[str, str]]:
    if enhanced:
        system_content = (
            "Answer the user memory question from the supplied evidence only. First write a short "
            "'Evidence facts:' section with the facts you used, then write 'Final answer:'. For "
            "time, money, count, and total questions, explicitly perform the arithmetic or elapsed "
            "time calculation when the evidence provides the needed values. For current-state or "
            "knowledge-update questions, treat the latest known value as current unless later "
            "evidence contradicts it; do not say insufficient just because there is no newer update. "
            "Treat user memory statements as authoritative even if a later statement is a recollection; "
            "prefer the later dated value for update questions. If an earlier total is followed by a "
            "later addition and no later total is stated, add the increment to the earlier total. "
            "For relative date words such as today, yesterday, last Monday, last week, or next month, "
            "resolve them using the date shown in that evidence item's session header. For ordering "
            "questions, list each event with its date, then sort by date before writing the final order. "
            "For questions asking how many times an event happened, if the event is not mentioned, "
            "answer that the information is insufficient or not mentioned; do not convert absence of "
            "evidence into a numeric zero. "
            "For preference or advice questions, use the evidence as user-specific constraints and "
            "give a useful personalized answer; do not require that the exact recommendation already "
            "appears in memory. Preserve negative preferences and avoidances, such as avoiding phone "
            "or TV use when the evidence says those hurt sleep. For questions about a previous "
            "conversation, assistant messages and "
            "assistant-provided tables, recommendations, names, methods, and numbers are valid "
            "evidence. If the user asks what was finally decided or chosen, prefer the last accepted "
            "or named option in the relevant conversation over earlier suggestions. If a required "
            "value is truly missing, say the information is insufficient and "
            "name what is missing. Do not invent unstated facts. Scan all supplied evidence before "
            "the final answer; do not ignore an explicit phrase like 'it cost me', 'my new X', or "
            "'my previous role as Y'. For previous-conversation questions, later turns override "
            "earlier drafts: a user's praise, acceptance, or repeated use of a name should be "
            "treated as the final choice, and a later named table should override an earlier table "
            "with generic Agent labels. For recommendation questions, if no exact local event or "
            "publication list is present, still answer with tailored categories, venues, search "
            "targets, or conference/publication areas grounded in the user's interests and "
            "avoidances; do not abstain just because the evidence lacks a live event calendar. "
            "For count and total questions, include every explicit service, brand, doctor, trip, "
            "or cost item mentioned in relevant evidence, including restaurants or delivery "
            "platforms used for convenience. Before giving a numeric count, write the counted "
            "items as a short list and check whether each item satisfies the wording of the "
            "question. Do not count recommendations, examples, budgets, price ranges, or future "
            "plans unless the question explicitly asks about planned items. For currently-own or "
            "currently-use questions, exclude items only considered, suggested, replaced, canceled, "
            "returned, or not yet acquired. For attended/visited/completed questions, exclude "
            "missed, planned, suggested, or merely discussed events. Count each explicitly named "
            "person, baby, item, device, appointment, trip, or event separately when the evidence "
            "states separate entities, including twins or multiple named items in the same turn. "
            "For holiday/date questions, if a session date is the "
            "holiday and the user describes a flight or event as today/recent in that session, use "
            "that dated evidence unless another retrieved fact directly contradicts it."
        )
    else:
        system_content = (
            "Answer the user memory question from the supplied evidence. If evidence is "
            "insufficient, say so. Compute direct counts, totals, elapsed times, and clock "
            "times when the evidence gives the needed values or time anchors. Keep the answer "
            "concise and state the evidence-based calculation when one is needed."
        )
    return [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": (
                f"Question date: {case.question_date or 'unknown'}\n"
                f"Question: {case.question}\n\nRetrieved memory evidence:\n{context}"
            ),
        },
    ]


def _evidence_context_budget(config: DemoConfig, case: QuestionCase, enhanced_qa: bool) -> int:
    overhead = rough_token_count(
        "\n".join(message["content"] for message in _answer_messages(case, "", enhanced=enhanced_qa))
    )
    # Reserve enough room for completion plus tokenizer mismatch. The context
    # budget uses a rough local counter, while provider accounting is model-side.
    answer_margin = max(2400, config.qa_max_tokens + 1400)
    return max(1000, config.qa_context_token_budget - overhead - answer_margin)


def _child_text(children: list[LeafNode | SummaryNode], leaf_text_mode: str) -> str:
    chunks = []
    for index, child in enumerate(children, start=1):
        text = _leaf_text(child, leaf_text_mode) if isinstance(child, LeafNode) else child.summary
        chunks.append(f"[Child {index}]\n{text}")
    return "\n\n".join(chunks)


def _leaf_text(leaf: LeafNode, mode: str) -> str:
    return leaf.user_text if mode == "user_only" else leaf.raw_text


def _leaf_ids(children: list[LeafNode | SummaryNode]) -> list[str]:
    ids: list[str] = []
    for child in children:
        ids.extend([child.node_id] if isinstance(child, LeafNode) else child.leaf_ids)
    return ids


def _raw_leaf_groups(
    leaves: list[LeafNode],
    fanout_k: int,
    max_group_rough_tokens: int,
    leaf_text_mode: str,
) -> list[list[LeafNode]]:
    groups: list[list[LeafNode]] = []
    current: list[LeafNode] = []
    current_tokens = 0
    for leaf in leaves:
        leaf_tokens = rough_token_count(_leaf_text(leaf, leaf_text_mode))
        would_exceed_budget = (
            max_group_rough_tokens > 0
            and current
            and current_tokens + leaf_tokens > max_group_rough_tokens
        )
        if len(current) >= fanout_k or would_exceed_budget:
            groups.append(current)
            current = []
            current_tokens = 0
        current.append(leaf)
        current_tokens += leaf_tokens
    if current:
        groups.append(current)
    return groups


def _group_summaries_by_session(nodes: list[SummaryNode]) -> dict[str, list[SummaryNode]]:
    grouped: dict[str, list[SummaryNode]] = {}
    for node in nodes:
        grouped.setdefault(node.session_id, []).append(node)
    return grouped


def _parse_summary(text: str, schema: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError) as error:
        extracted = _extract_json_object(text)
        if extracted is None:
            return None, f"invalid_json: {error}"
        payload = extracted
    if not isinstance(payload, dict):
        return None, "summary_json_must_be_object"

    if schema == "multilingual_memory_v1":
        parsed = {
            "facts": _summary_string_list(payload.get("facts")),
            "events": _summary_string_list(payload.get("events")),
            "counts": _summary_string_list(payload.get("counts")),
            "dates": _summary_string_list(payload.get("dates")),
            "updates": _summary_string_list(payload.get("updates")),
            "keywords": _summary_string_list(payload.get("keywords")),
        }
    elif schema == "compact_memory_v2":
        parsed: dict[str, Any] = {
            "m": _summary_string_list(payload.get("m")),
            "k": _summary_string_list(payload.get("k")),
        }
    else:
        parsed = {
            "compact_summary": _summary_string(payload.get("compact_summary")),
            "facts": _summary_string_list(payload.get("facts")),
            "updates": _summary_string_list(payload.get("updates")),
            "time_anchors": _summary_string_list(payload.get("time_anchors")),
            "keywords": _summary_string_list(payload.get("keywords")),
        }
    return parsed, None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    decoder = json.JSONDecoder()
    for start in (index for index, char in enumerate(text) if char == "{"):
        try:
            payload, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _summary_string(value: Any) -> str:
    return str(value).strip() if isinstance(value, (str, int, float, bool)) else ""


def _summary_string_list(value: Any) -> list[str]:
    if isinstance(value, (str, int, float, bool)):
        value = [value]
    if not isinstance(value, list):
        return []
    return [item for item in (_summary_string(item) for item in value) if item]


def _render_summary(parsed: dict[str, Any] | None, raw_text: str, schema: str) -> str:
    if parsed is None:
        return raw_text.strip()
    if schema == "multilingual_memory_v1":
        blocks: list[str] = []
        for label, key in (
            ("Facts", "facts"),
            ("Events", "events"),
            ("Counts", "counts"),
            ("Dates", "dates"),
            ("Updates", "updates"),
            ("Keywords", "keywords"),
        ):
            values = parsed[key]
            if values:
                blocks.append(f"{label}: " + "; ".join(values))
        return "\n".join(blocks)
    if schema == "compact_memory_v2":
        memory = "; ".join(parsed["m"])
        keywords = "; ".join(parsed["k"])
        return "\n".join(
            block for block in (f"Memory: {memory}" if memory else "", f"Keywords: {keywords}" if keywords else "") if block
        )
    blocks: list[str] = []
    if parsed["compact_summary"]:
        blocks.append(parsed["compact_summary"])
    for label, key in (
        ("Facts", "facts"),
        ("Updates", "updates"),
        ("Times", "time_anchors"),
        ("Keywords", "keywords"),
    ):
        values = parsed[key]
        if values:
            blocks.append(f"{label}: " + "; ".join(values))
    return "\n".join(blocks)


def _context_text(summaries: list[SummaryNode], leaves: list[LeafNode]) -> str:
    blocks = []
    if summaries:
        blocks.append("Relevant session summaries:")
        blocks.extend(
            f"- Session {summary.session_id} ({summary.session_date or 'unknown'}): {summary.summary}"
            for summary in summaries
        )
    blocks.append("Raw evidence:")
    blocks.extend(
        f"[Session {leaf.session_id} | {leaf.session_date or 'unknown'} | turn {leaf.turn_index}]\n{leaf.raw_text}"
        for leaf in leaves
    )
    return "\n\n".join(blocks)


def _fit_context_budget(
    summaries: list[SummaryNode],
    leaves: list[LeafNode],
    token_budget: int,
) -> tuple[list[SummaryNode], list[LeafNode]]:
    kept_summaries = list(summaries)
    kept_leaves = list(leaves)
    while kept_summaries and rough_token_count(_context_text(kept_summaries, kept_leaves)) > token_budget:
        kept_summaries.pop()
    while len(kept_leaves) > 1 and rough_token_count(_context_text(kept_summaries, kept_leaves)) > token_budget:
        kept_leaves.pop()
    return kept_summaries, kept_leaves


def _chunks(values: list[Any], size: int) -> list[list[Any]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def _node_row(node: LeafNode | SummaryNode) -> dict[str, Any]:
    row = asdict(node)
    row["node_type"] = "leaf" if isinstance(node, LeafNode) else "summary"
    row.pop("embedding", None)
    return row


def _reset_jsonl_outputs(directory: Path) -> None:
    for name in (
        "llm_calls.jsonl",
        "embedding_calls.jsonl",
        "compression_stats.jsonl",
        "nodes.jsonl",
        "edges.jsonl",
        "question_stats.jsonl",
        "retrieval_results.jsonl",
        "answers.jsonl",
        "manual_eval.jsonl",
    ):
        (directory / name).write_text("", encoding="utf-8")


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _read_question_stats(path: Path) -> list[QuestionStats]:
    return [QuestionStats(**row) for row in _read_jsonl(path)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _deepseek_stage_totals(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = {}
    for row in rows:
        stage = str(row["stage"])
        stage_total = totals.setdefault(
            stage,
            {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": 0,
            },
        )
        stage_total["calls"] += 1
        for field in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"):
            stage_total[field] += int(row.get(field) or 0)
    return totals


def _local_summary_stage_totals(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    totals: dict[str, dict[str, float | int]] = {}
    for row in rows:
        compressor_name = str(row.get("compressor") or "")
        if not compressor_name.startswith("qwen_local:"):
            continue
        stage = str(row["stage"])
        stage_total = totals.setdefault(
            stage,
            {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "latency_sec": 0.0,
                "failure_count": 0,
            },
        )
        stage_total["calls"] = int(stage_total["calls"]) + 1
        prompt_tokens = int(row.get("origin_tokens") or 0)
        completion_tokens = int(row.get("compressed_tokens") or 0)
        stage_total["prompt_tokens"] = int(stage_total["prompt_tokens"]) + prompt_tokens
        stage_total["completion_tokens"] = int(stage_total["completion_tokens"]) + completion_tokens
        stage_total["total_tokens"] = int(stage_total["total_tokens"]) + prompt_tokens + completion_tokens
        stage_total["latency_sec"] = float(stage_total["latency_sec"]) + float(
            row.get("latency_sec") or 0.0
        )
        stage_total["failure_count"] = int(stage_total["failure_count"]) + int(
            bool(row.get("error_status"))
        )
    return totals


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_summary(output_dir: Path, aggregates: list[VariantStats]) -> None:
    rows = [asdict(aggregate) for aggregate in aggregates]
    if not rows:
        return
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    columns = [
        "variant",
        "question_count",
        "build_prompt_tokens",
        "build_completion_tokens",
        "answer_prompt_tokens",
        "answer_completion_tokens",
        "reasoning_tokens",
        "total_deepseek_tokens",
        "deepseek_call_count",
        "avg_tokens_per_question",
        "token_budget_avg_under_300k",
        "retrieval_answer_session_hit_rate",
        "retrieval_answer_session_all_hit_rate",
        "avg_retrieved_answer_session_recall",
        "summary_count",
        "edge_count",
        "build_calls_per_session",
        "summary_parse_error_count",
        "summary_truncation_count",
        "wall_time_sec",
    ]
    markdown = [
        "# GraphMem Token Demo Summary",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        markdown.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    (output_dir / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def _write_manual_eval_template(variant_dir: Path) -> None:
    prior = {
        row["question_id"]: row
        for row in _read_jsonl(variant_dir / "manual_eval.jsonl")
        if row.get("question_id")
    }
    rows: list[dict[str, Any]] = []
    for answer in _read_jsonl(variant_dir / "answers.jsonl"):
        existing = prior.get(answer["question_id"], {})
        rows.append(
            {
                "question_id": answer["question_id"],
                "question": answer["question"],
                "gold_answer": answer["gold_answer"],
                "prediction": answer["prediction"],
                "strict_correct": existing.get("strict_correct"),
                "relaxed_correct": existing.get("relaxed_correct"),
                "error_type": existing.get("error_type"),
                "notes": existing.get("notes", ""),
            }
        )
    path = variant_dir / "manual_eval.jsonl"
    path.write_text("", encoding="utf-8")
    _append_jsonl(path, rows)
    strict_rows = [row for row in rows if isinstance(row["strict_correct"], bool)]
    relaxed_rows = [row for row in rows if isinstance(row["relaxed_correct"], bool)]
    markdown = [
        "# GraphMem Manual Evaluation",
        "",
        "Strict: numeric answers must state the gold value; insufficient gold must be explicit.",
        "Relaxed: report semantic matches separately; no LLM judge token is used.",
        "",
        f"- Rows: {len(rows)}",
        f"- Strict judged: {len(strict_rows)}",
        f"- Strict accuracy: {_manual_accuracy(strict_rows, 'strict_correct')}",
        f"- Relaxed judged: {len(relaxed_rows)}",
        f"- Relaxed accuracy: {_manual_accuracy(relaxed_rows, 'relaxed_correct')}",
        "",
        "| question_id | strict | relaxed | error_type | notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    markdown.extend(
        "| "
        + " | ".join(
            str(row[field]).replace("|", "\\|")
            for field in ("question_id", "strict_correct", "relaxed_correct", "error_type", "notes")
        )
        + " |"
        for row in rows
    )
    (variant_dir / "manual_eval.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def _manual_accuracy(rows: list[dict[str, Any]], field: str) -> str:
    if not rows:
        return "pending"
    return f"{sum(bool(row[field]) for row in rows) / len(rows):.3f}"
