from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import graphmem_demo.clients as clients_module
from graphmem_demo.clients import (
    DeepSeekClient,
    LLMResult,
    MockCompressor,
    MockDeepSeekClient,
    MockEmbeddingClient,
    MockLocalSummarizer,
    parse_usage,
)
from graphmem_demo.data import (
    build_leaf_nodes,
    load_longmemeval_cases,
    speaker_retrieval_text_from_raw,
)
from graphmem_demo.models import DeepSeekCallRecord, GraphEdge, LeafNode, QuestionCase, SummaryNode
from graphmem_demo.pipeline import (
    DemoConfig,
    _answer_messages,
    _build_root_graph,
    _effective_global_leaf_top_k,
    _effective_leaf_top_k,
    _effective_leaf_text_mode,
    _effective_per_session_leaf_k,
    _evidence_context_budget,
    _edge_expansion_sort_key,
    _expand_root_ids,
    _expand_selected_session_context,
    _fit_context_budget,
    _leaf_embedding_attr,
    _parse_summary,
    _render_summary,
    _retrieve_hybrid,
    _summary_anchor_terms,
    _summary_retrieval_text,
    _summary_messages,
    _variant_spec,
    run_demo,
    run_case,
)
from graphmem_demo.stats import aggregate_variant_stats, build_question_stats


GRAPHMEM_ROOT = Path(__file__).resolve().parents[1]


def _load_generic_ops_module():
    module_path = GRAPHMEM_ROOT / "scripts" / "apply_generic_memory_ops.py"
    spec = importlib.util.spec_from_file_location("apply_generic_memory_ops", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_loader_selects_multi_session_cases() -> None:
    cases = load_longmemeval_cases(
        GRAPHMEM_ROOT / "data" / "longmemeval_s_subset_10_per_type.json",
        question_type="multi-session",
    )
    assert len(cases) == 10
    assert all(case.question_type == "multi-session" for case in cases)
    assert all(
        len(case.haystack_sessions) == len(case.haystack_session_ids) == len(case.haystack_dates)
        for case in cases
    )
    assert cases[0].answer_session_ids


def test_loader_can_select_all_question_types() -> None:
    cases = load_longmemeval_cases(
        GRAPHMEM_ROOT / "data" / "longmemeval_s_subset_10_per_type.json",
        question_type="all",
    )
    assert len(cases) == 60
    assert len({case.question_type for case in cases}) == 6


def test_turn_pairing_keeps_nonstandard_messages() -> None:
    case = QuestionCase(
        question_id="q1",
        question_type="multi-session",
        question="question",
        answer="answer",
        question_date=None,
        haystack_sessions=[
            [
                {"role": "assistant", "content": "opening"},
                {"role": "user", "content": "paired user"},
                {"role": "assistant", "content": "paired assistant"},
                {"role": "user", "content": "trailing user"},
            ]
        ],
        haystack_session_ids=["s1"],
        haystack_dates=["2026/05/22"],
        answer_session_ids=["s1"],
    )
    leaves = build_leaf_nodes(case)
    assert [leaf.message_count for leaf in leaves] == [1, 2, 1]
    assert leaves[1].raw_text.startswith("User: paired user")
    assert "paired assistant" not in leaves[1].user_text
    assert leaves[0].user_text == leaves[0].raw_text
    assert leaves[-1].turn_index == 3


def test_leaf_text_preserves_explicit_speaker_names() -> None:
    case = QuestionCase(
        question_id="q1",
        question_type="category_1",
        question="What did Melanie do?",
        answer="answer",
        question_date=None,
        haystack_sessions=[
            [
                {
                    "role": "assistant",
                    "speaker": "Melanie",
                    "listener": "Caroline",
                    "content": "I ran a charity race.",
                },
                {
                    "role": "user",
                    "speaker": "Caroline",
                    "listener": "Melanie",
                    "content": "That sounds great.",
                },
            ]
        ],
        haystack_session_ids=["s1"],
        haystack_dates=["2026/05/22"],
        answer_session_ids=["s1"],
    )

    leaves = build_leaf_nodes(case)

    assert leaves[0].raw_text.startswith("Assistant (Melanie) -> Caroline:")
    assert "User (Caroline) -> Melanie:" in leaves[1].raw_text
    assert "Melanie said: I ran a charity race." in leaves[0].retrieval_text
    assert "Caroline said: That sounds great." in leaves[1].retrieval_text


def test_speaker_retrieval_text_can_be_backfilled_from_raw_leaf() -> None:
    raw_text = (
        "Assistant (Melanie) -> Caroline: I ran a charity race.\n"
        "User (Caroline) -> Melanie: That sounds great."
    )

    retrieval_text = speaker_retrieval_text_from_raw(raw_text)

    assert raw_text in retrieval_text
    assert "Melanie said: I ran a charity race." in retrieval_text
    assert "Caroline said: That sounds great." in retrieval_text


def test_explicit_speaker_cases_use_raw_leaf_text_in_auto_mode() -> None:
    case = QuestionCase(
        question_id="q1",
        question_type="category_1",
        question="What did Melanie do?",
        answer="answer",
        question_date=None,
        haystack_sessions=[
            [
                {
                    "role": "assistant",
                    "speaker": "Melanie",
                    "listener": "Caroline",
                    "content": "I ran a charity race.",
                }
            ]
        ],
        haystack_session_ids=["s1"],
        haystack_dates=["2026/05/22"],
        answer_session_ids=["s1"],
    )
    spec = _variant_spec(DemoConfig(data_path=Path("x"), output_dir=Path("y")), "single_llm_summary_graphmem")

    assert _effective_leaf_text_mode("auto", spec, case, phase="build") == "raw"
    assert _effective_leaf_text_mode("auto", spec, case, phase="retrieval") == "raw"
    assert _leaf_embedding_attr(
        DemoConfig(data_path=Path("x"), output_dir=Path("y")), case, "raw"
    ) == "raw_text"
    assert _leaf_embedding_attr(
        DemoConfig(
            data_path=Path("x"),
            output_dir=Path("y"),
            enable_speaker_retrieval_text=True,
        ),
        case,
        "raw",
    ) == "retrieval_text"


def test_explicit_speaker_cases_get_wider_retrieval_caps() -> None:
    case = QuestionCase(
        question_id="q1",
        question_type="category_1",
        question="What did Melanie do?",
        answer="answer",
        question_date=None,
        haystack_sessions=[
            [
                {
                    "role": "assistant",
                    "speaker": "Melanie",
                    "listener": "Caroline",
                    "content": "I ran a charity race.",
                }
            ]
        ],
        haystack_session_ids=["s1"],
        haystack_dates=["2026/05/22"],
        answer_session_ids=["s1"],
    )
    config = DemoConfig(
        data_path=Path("x"),
        output_dir=Path("y"),
        leaf_top_k=14,
        global_leaf_top_k=24,
        per_session_leaf_k=2,
    )

    assert _effective_leaf_top_k(config, case) == 24
    assert _effective_global_leaf_top_k(config, case) == 48
    assert _effective_per_session_leaf_k(config, case) == 3


def test_parse_usage_preserves_deepseek_token_breakdown() -> None:
    usage = {
        "prompt_tokens": 21,
        "completion_tokens": 8,
        "total_tokens": 29,
        "prompt_cache_hit_tokens": 4,
        "prompt_cache_miss_tokens": 17,
        "completion_tokens_details": {"reasoning_tokens": 5},
    }
    assert parse_usage(usage) == {
        "prompt_tokens": 21,
        "completion_tokens": 8,
        "total_tokens": 29,
        "prompt_cache_hit_tokens": 4,
        "prompt_cache_miss_tokens": 17,
        "reasoning_tokens": 5,
    }
    assert parse_usage({"prompt_tokens": 3})["reasoning_tokens"] == 0


def test_deepseek_none_thinking_disables_reasoning_request(monkeypatch) -> None:
    captured_request: dict[str, object] = {}

    class FakeCompletions:
        def create(self, **request):
            captured_request.update(request)
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

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(clients_module, "_openai_client_class", lambda: FakeOpenAI)

    result = DeepSeekClient(model="deepseek-v4-flash").chat(
        question_id="q",
        variant="v",
        stage="answer_qa",
        messages=[{"role": "user", "content": "answer"}],
        thinking_mode="none",
        max_tokens=64,
    )

    assert captured_request["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in captured_request
    assert result.record.thinking_mode == "none"
    assert result.record.reasoning_tokens == 0


def test_stats_keep_build_and_answer_tokens_separate() -> None:
    records = [
        _record("build_summary_leaf", 10, 2),
        _record("build_summary_internal", 4, 1),
        _record("answer_qa", 9, 3, reasoning_tokens=7),
    ]
    question_stats = build_question_stats(
        question_id="q",
        variant="token_efficient_graphmem",
        session_count=2,
        leaf_count=3,
        summary_count=2,
        edge_count=1,
        records=records,
        build_latency_sec=1.0,
        retrieval_latency_sec=0.2,
        answer_latency_sec=0.3,
        answer_session_hit=True,
    )
    aggregate = aggregate_variant_stats([question_stats], "token_efficient_graphmem")
    assert question_stats.build_prompt_tokens == 14
    assert question_stats.build_completion_tokens == 3
    assert question_stats.answer_prompt_tokens == 9
    assert question_stats.answer_completion_tokens == 3
    assert aggregate.total_deepseek_tokens == 29
    assert aggregate.retrieval_answer_session_hit_rate == 1.0


def test_mock_cli_runs_all_variants_and_writes_recall(tmp_path: Path) -> None:
    data_path = tmp_path / "cases.json"
    output_dir = tmp_path / "run"
    data_path.write_text(json.dumps([_synthetic_row()]), encoding="utf-8")
    command = [
        sys.executable,
        str(GRAPHMEM_ROOT / "scripts" / "run_token_demo.py"),
        "--data",
        str(data_path),
        "--output-dir",
        str(output_dir),
        "--max-questions",
        "1",
        "--variants",
        "direct_session_k16_compact_no_compress",
        "direct_session_k16_compact_graphmem",
        "qwen35_2b_summary_graphmem",
        "qwen35_08b_summary_graphmem",
        "qwen35_2b_summary_graphmem_no_retrieval_enhance",
        "qwen35_2b_summary_graphmem_no_qa_enhance",
        "single_llm_summary_graphmem",
        "--question-workers",
        "2",
        "--mock-services",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "direct_session_k16_compact_graphmem" in result.stdout
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "direct_session_k16_compact_graphmem" / "llm_calls.jsonl").exists()
    query_stats = json.loads(
        (output_dir / "direct_session_k16_compact_graphmem" / "query_stats.json").read_text(
            encoding="utf-8"
        )
    )
    assert query_stats["aggregate"]["edge_count"] > 0
    assert query_stats["questions"][0]["retrieved_answer_session_hit"] is True
    assert query_stats["questions"][0]["retrieved_answer_session_recall"] == 1.0
    assert query_stats["deepseek_token_by_stage"]["answer_qa"]["prompt_tokens"] > 0
    llm_rows = [
        json.loads(line)
        for line in (output_dir / "direct_session_k16_compact_no_compress" / "llm_calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["stage"] for row in llm_rows} >= {"build_summary_session_direct", "answer_qa"}
    assert {row["thinking_mode"] for row in llm_rows} == {"none"}
    assert sum(row["reasoning_tokens"] for row in llm_rows) == 0
    assert all(
        row["response_format"] == "json_object"
        for row in llm_rows
        if row["stage"].startswith("build_summary")
    )
    manual_dir = output_dir / "direct_session_k16_compact_graphmem"
    assert (manual_dir / "manual_eval.jsonl").exists()
    assert "pending" in (manual_dir / "manual_eval.md").read_text(encoding="utf-8")
    qwen_stats = json.loads(
        (output_dir / "qwen35_2b_summary_graphmem" / "build_stats.json").read_text(
            encoding="utf-8"
        )
    )
    assert qwen_stats["local_summarizer_by_stage"]["build_summary_session_direct"]["calls"] == 2
    qwen_llm_rows = [
        json.loads(line)
        for line in (output_dir / "qwen35_2b_summary_graphmem" / "llm_calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["stage"] for row in qwen_llm_rows} == {"answer_qa"}
    assert {row["thinking_mode"] for row in qwen_llm_rows} == {"none"}
    assert sum(row["reasoning_tokens"] for row in qwen_llm_rows) == 0


def test_locomo_memory_cache_builds_each_sample_once(tmp_path: Path) -> None:
    first = _synthetic_row()
    first["question_id"] = "conv-1_qa000"
    first["locomo_sample_id"] = "conv-1"
    second = _synthetic_row()
    second["question_id"] = "conv-1_qa001"
    second["question"] = "What did I do after work?"
    second["locomo_sample_id"] = "conv-1"
    data_path = tmp_path / "locomo.json"
    output_dir = tmp_path / "run"
    data_path.write_text(json.dumps([first, second]), encoding="utf-8")

    command = [
        sys.executable,
        str(GRAPHMEM_ROOT / "scripts" / "run_token_demo.py"),
        "--data",
        str(data_path),
        "--question-type",
        "all",
        "--output-dir",
        str(output_dir),
        "--max-questions",
        "10",
        "--variants",
        "single_llm_summary_graphmem",
        "--question-workers",
        "2",
        "--mock-services",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)

    llm_rows = [
        json.loads(line)
        for line in (output_dir / "single_llm_summary_graphmem" / "llm_calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    stage_counts = {stage: sum(row["stage"] == stage for row in llm_rows) for stage in {row["stage"] for row in llm_rows}}
    assert stage_counts["build_summary_session_direct"] == 2
    assert stage_counts["answer_qa"] == 2
    assert len([row for row in llm_rows if row["stage"].startswith("build_summary")]) == 2


def test_locomo_memory_cache_applies_with_injected_services(tmp_path: Path) -> None:
    first = _synthetic_row()
    first["question_id"] = "conv-1_qa000"
    first["locomo_sample_id"] = "conv-1"
    second = _synthetic_row()
    second["question_id"] = "conv-1_qa001"
    second["question"] = "What did I do after work?"
    second["locomo_sample_id"] = "conv-1"
    data_path = tmp_path / "locomo.json"
    output_dir = tmp_path / "run"
    data_path.write_text(json.dumps([first, second]), encoding="utf-8")

    run_demo(
        DemoConfig(
            data_path=data_path,
            question_type="all",
            output_dir=output_dir,
            max_questions=10,
            variants=("single_llm_summary_graphmem",),
            question_workers=2,
            mock_services=True,
        ),
        llm=MockDeepSeekClient(),
        embedder=MockEmbeddingClient(),
        compressor=MockCompressor(0.5),
    )

    llm_rows = [
        json.loads(line)
        for line in (output_dir / "single_llm_summary_graphmem" / "llm_calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    build_rows = [row for row in llm_rows if row["stage"].startswith("build_summary")]
    answer_rows = [row for row in llm_rows if row["stage"] == "answer_qa"]
    question_stats = [
        json.loads(line)
        for line in (output_dir / "single_llm_summary_graphmem" / "question_stats.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert len(build_rows) == 2
    assert len(answer_rows) == 2
    assert sum(row["build_prompt_tokens"] > 0 for row in question_stats) == 1


def test_locomo_memory_cache_reuses_disk_cache_on_resume(tmp_path: Path) -> None:
    first = _synthetic_row()
    first["question_id"] = "conv-1_qa000"
    first["locomo_sample_id"] = "conv-1"
    second = _synthetic_row()
    second["question_id"] = "conv-1_qa001"
    second["question"] = "What did I do after work?"
    second["locomo_sample_id"] = "conv-1"
    data_path = tmp_path / "locomo.json"
    output_dir = tmp_path / "run"
    data_path.write_text(json.dumps([first, second]), encoding="utf-8")

    base_config = dict(
        data_path=data_path,
        question_type="all",
        output_dir=output_dir,
        variants=("single_llm_summary_graphmem",),
        question_workers=2,
        mock_services=True,
    )
    run_demo(DemoConfig(max_questions=1, **base_config))
    run_demo(DemoConfig(max_questions=10, resume=True, **base_config))

    variant_dir = output_dir / "single_llm_summary_graphmem"
    llm_rows = [
        json.loads(line)
        for line in (variant_dir / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    question_stats = [
        json.loads(line)
        for line in (variant_dir / "question_stats.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert list((variant_dir / "memory_cache").glob("*.json"))
    assert len([row for row in llm_rows if row["stage"].startswith("build_summary")]) == 2
    assert len([row for row in llm_rows if row["stage"] == "answer_qa"]) == 2
    assert sum(row["build_prompt_tokens"] > 0 for row in question_stats) == 1


def test_relabel_leaf_nodes_from_data_preserves_summaries_and_restores_speakers(
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "locomo.json"
    nodes_path = tmp_path / "old_nodes.jsonl"
    output_path = tmp_path / "new_nodes.jsonl"
    row = _synthetic_row()
    row["question_id"] = "conv-1_qa000"
    row["question_type"] = "category_1"
    row["locomo_sample_id"] = "conv-1"
    row["haystack_sessions"][0][0]["speaker"] = "Caroline"
    row["haystack_sessions"][0][0]["listener"] = "Melanie"
    row["haystack_sessions"][0][1]["speaker"] = "Melanie"
    row["haystack_sessions"][0][1]["listener"] = "Caroline"
    data_path.write_text(json.dumps([row]), encoding="utf-8")
    old_leaf = {
        "node_id": "conv-1_qa000:distractor-session:leaf:0",
        "question_id": "conv-1_qa000",
        "session_id": "distractor-session",
        "session_date": "2026/05/20",
        "turn_index": 0,
        "raw_text": "User: old\nAssistant: old",
        "user_text": "User: old",
        "message_count": 2,
        "node_type": "leaf",
    }
    summary = {
        "node_id": "summary-1",
        "question_id": "conv-1_qa000",
        "session_id": "distractor-session",
        "session_date": "2026/05/20",
        "level": 1,
        "child_ids": [old_leaf["node_id"]],
        "leaf_ids": [old_leaf["node_id"]],
        "summary": "old summary",
        "node_type": "summary",
    }
    nodes_path.write_text(
        json.dumps(old_leaf) + "\n" + json.dumps(summary) + "\n",
        encoding="utf-8",
    )

    command = [
        sys.executable,
        str(GRAPHMEM_ROOT / "scripts" / "relabel_leaf_nodes_from_data.py"),
        "--data",
        str(data_path),
        "--nodes",
        str(nodes_path),
        "--output",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert rows[0]["raw_text"].startswith("User (Caroline) -> Melanie:")
    assert "Assistant (Melanie) -> Caroline:" in rows[0]["raw_text"]
    assert rows[1] == summary


def test_speaker_mismatch_operator_abstains_on_cross_speaker_transfer() -> None:
    module = _load_generic_ops_module()
    chunks = [
        module.MemoryChunk(
            question_id="q",
            node_type="leaf",
            session_id="s1",
            session_date=None,
            turn_index=0,
            text=(
                "User (Caroline) -> Melanie: My grandma in my home country, Sweden gave me a necklace.\n"
                "Assistant (Melanie) -> Caroline: That sounds meaningful."
            ),
        )
    ]

    result = module.op_speaker_mismatch_abstain(
        "What country is Melanie's grandma from?",
        chunks,
    )

    assert result is not None
    assert result.operator == "speaker_mismatch_abstain"
    assert "does not mention this information for Melanie" in result.answer


def test_target_speaker_prefers_question_subject_over_possessive_modifier() -> None:
    module = _load_generic_ops_module()

    assert (
        module.target_speaker(
            "What book did Melanie read from Caroline's suggestion?",
            ["Caroline", "Melanie"],
        )
        == "Melanie"
    )


def test_explicit_speaker_profile_calls_are_recorded(tmp_path: Path) -> None:
    case = QuestionCase(
        question_id="conv-1_qa000",
        question_type="category_1",
        question="What does Melanie do?",
        answer="runs",
        question_date="2026/05/22",
        haystack_sessions=[
            [
                {
                    "role": "assistant",
                    "speaker": "Melanie",
                    "listener": "Caroline",
                    "content": "I run every morning.",
                },
                {
                    "role": "user",
                    "speaker": "Caroline",
                    "listener": "Melanie",
                    "content": "I paint on weekends.",
                },
            ]
        ],
        haystack_session_ids=["conv-1_session_1"],
        haystack_dates=["2026/05/22"],
        answer_session_ids=["conv-1_session_1"],
    )
    case.enable_speaker_profiles = True  # type: ignore[attr-defined]

    run = _run_synthetic_case(
        tmp_path,
        case,
        variant="single_llm_summary_graphmem",
    )

    stage_counts = {
        stage: sum(record.stage == stage for record in run.llm_records)
        for stage in {record.stage for record in run.llm_records}
    }
    assert stage_counts["build_summary_speaker_profile"] == 2
    assert stage_counts["answer_qa"] == 1
    assert sum(node.summary_mode == "speaker_profile" for node in run.summaries) == 2
    assert run.stats.build_prompt_tokens > 0


def test_partial_mock_flags_keep_real_llm_config(tmp_path: Path) -> None:
    run = run_case(
        DemoConfig(
            data_path=tmp_path / "unused.json",
            output_dir=tmp_path / "run",
            variants=("qwen35_2b_summary_graphmem",),
            mock_embedding=True,
            mock_summarizer=True,
        ),
        _case_with_leaf_count(1),
        "qwen35_2b_summary_graphmem",
        MockDeepSeekClient(),
        MockEmbeddingClient(),
        MockCompressor(0.5),
        summarizer=MockLocalSummarizer("Qwen/Qwen3.5-2B"),
    )
    assert {record.stage for record in run.llm_records} == {"answer_qa"}


def test_direct_session_short_session_skips_leaf_summary(tmp_path: Path) -> None:
    run = _run_synthetic_case(
        tmp_path,
        _case_with_leaf_count(3),
        variant="direct_session_k16_no_compress",
    )
    stages = [record.stage for record in run.llm_records]
    assert stages.count("build_summary_session_direct") == 1
    assert "build_summary_leaf" not in stages
    assert len([node for node in run.summaries if node.level == 1]) == 1
    assert run.summaries[0].parsed_summary


def test_direct_session_long_session_splits_raw_groups_and_merges(tmp_path: Path) -> None:
    run = _run_synthetic_case(
        tmp_path,
        _case_with_leaf_count(3),
        variant="direct_session_k16_no_compress",
        max_group_rough_tokens=1,
    )
    stages = [record.stage for record in run.llm_records]
    assert stages.count("build_summary_raw_group") == 3
    assert stages.count("build_summary_session_merge") == 1
    assert "build_summary_leaf" not in stages


def test_bad_structured_summary_keeps_raw_text_and_parse_error(tmp_path: Path) -> None:
    llm = _MalformedSummaryLLM()
    run = _run_synthetic_case(
        tmp_path,
        _case_with_leaf_count(1),
        variant="direct_session_k16_no_compress",
        llm=llm,
    )
    assert run.summaries[0].raw_summary_text == "not json"
    assert run.summaries[0].summary == "not json"
    assert run.summaries[0].parse_error
    assert run.summaries[0].truncated is True
    assert run.stats.summary_parse_error_count == 1
    assert run.stats.summary_truncation_count == 1


def test_ready_summary_jobs_overlap_and_parent_waits(tmp_path: Path) -> None:
    llm = _SlowTrackingLLM()
    run = _run_synthetic_case(
        tmp_path,
        _case_with_leaf_count(8),
        variant="summary_tree_k4_no_compress",
        llm=llm,
    )
    leaf_ends = [end for stage, _, end in llm.calls if stage == "build_summary_leaf"]
    internal_starts = [start for stage, start, _ in llm.calls if stage == "build_summary_internal"]
    assert llm.peak_build_calls >= 2
    assert run.stats.peak_inflight_deepseek >= 2
    assert internal_starts and max(leaf_ends) <= min(internal_starts)


def test_inflight_cap_limits_sibling_summary_calls(tmp_path: Path) -> None:
    llm = _SlowTrackingLLM()
    run = _run_synthetic_case(
        tmp_path,
        _case_with_leaf_count(8),
        variant="summary_tree_k4_no_compress",
        llm=llm,
        max_inflight_deepseek=1,
    )
    assert llm.peak_build_calls == 1
    assert run.stats.peak_inflight_deepseek == 1


def test_compact_v2_schema_and_compression_chunks(tmp_path: Path) -> None:
    compressor = MockCompressor(0.5)
    run = run_case(
        DemoConfig(
            data_path=tmp_path / "unused.json",
            output_dir=tmp_path / "run",
            variants=("direct_session_k16_compact_graphmem",),
            compressor_chunk_rough_tokens=4,
            mock_services=True,
        ),
        _case_with_leaf_count(3),
        "direct_session_k16_compact_graphmem",
        MockDeepSeekClient(),
        MockEmbeddingClient(),
        compressor,
    )
    assert set(run.summaries[0].parsed_summary or {}) == {"m", "k"}
    assert run.summaries[0].summary.startswith("Memory:")
    assert max(record.chunk_count for record in compressor.records) > 1
    assert all("Remember visit" not in record.compressor for record in compressor.records)


def test_hybrid_global_leaf_fallback_adds_non_root_leaf(tmp_path: Path) -> None:
    config = DemoConfig(
        data_path=tmp_path / "unused.json",
        output_dir=tmp_path / "run",
        root_candidate_k=1,
        global_leaf_top_k=1,
        leaf_top_k=2,
        mock_services=True,
    )
    distractor = _leaf("distractor-leaf", "distractor", [0.0, 1.0])
    answer = _leaf("answer-leaf", "answer", [1.0, 0.0])
    roots = [
        _root("distractor-root", "distractor", [1.0, 0.0], [distractor.node_id]),
        _root("answer-root", "answer", [0.0, 1.0], [answer.node_id]),
    ]
    summaries, leaves, edges = _retrieve_hybrid(
        config,
        "Which answer is current?",
        [distractor, answer],
        roots,
        [],
        [1.0, 0.0],
        False,
        False,
    )
    assert not edges
    assert answer in leaves
    assert {summary.session_id for summary in summaries} >= {"answer"}


def test_hybrid_root_seed_preserves_root_session_leaf(tmp_path: Path) -> None:
    config = DemoConfig(
        data_path=tmp_path / "unused.json",
        output_dir=tmp_path / "run",
        root_top_k=1,
        root_candidate_k=1,
        global_leaf_top_k=1,
        leaf_top_k=1,
        mock_services=True,
    )
    root_leaf = _leaf("root-leaf", "root-session", [0.0, 1.0])
    global_leaf = _leaf("global-leaf", "global-session", [1.0, 0.0])
    roots = [
        _root("root-summary", "root-session", [1.0, 0.0], [root_leaf.node_id]),
        _root("global-summary", "global-session", [0.0, 1.0], [global_leaf.node_id]),
    ]
    _, leaves, _ = _retrieve_hybrid(
        config,
        "Which root session?",
        [root_leaf, global_leaf],
        roots,
        [],
        [1.0, 0.0],
        False,
        False,
    )
    assert leaves == [root_leaf]


def test_multilingual_schema_parse_and_render() -> None:
    payload = (
        '{"facts":["用户订阅了 The New Yorker"],"events":["arrived at 9 AM"],'
        '"counts":["2 subscriptions"],"dates":["Monday"],'
        '"updates":["cancelled Forbes"],"keywords":["subscription"]}'
    )
    parsed, error = _parse_summary(payload, "multilingual_memory_v1")
    assert error is None
    assert parsed and set(parsed) == {"facts", "events", "counts", "dates", "updates", "keywords"}
    rendered = _render_summary(parsed, payload, "multilingual_memory_v1")
    assert "Facts: 用户订阅了 The New Yorker" in rendered
    assert "Updates: cancelled Forbes" in rendered


def test_summary_retrieval_text_adds_search_cues_without_raw_copy() -> None:
    parsed = {
        "m": ["User bought a pass for $120."],
        "k": ["Architectural Digest", "subscription"],
    }
    source = (
        "[Child 1]\nUser: I bought an Architectural Digest subscription on 2026/05/21 "
        "and cancelled Forbes."
    )

    retrieval_text = _summary_retrieval_text(
        "Memory: User bought a pass for $120.",
        parsed,
        source,
        "2026/05/21",
    )

    assert "Session date: 2026/05/21" in retrieval_text
    assert "Anchor terms:" in retrieval_text
    assert "Keywords: Architectural Digest; subscription" in retrieval_text
    assert "Search cues:" in retrieval_text
    assert "Architectural Digest" in retrieval_text
    assert "2026/05/21" in retrieval_text
    assert "bought" in retrieval_text
    assert "[Child 1]" not in retrieval_text


def test_root_graph_adds_keyword_neighbor_edges() -> None:
    first = _root("root-a", "a", [1.0, 0.0], ["leaf-a"])
    second = _root("root-b", "b", [0.0, 1.0], ["leaf-b"])
    third = _root("root-c", "c", [0.0, -1.0], ["leaf-c"])
    first.retrieval_text = "Memory: subscribed. Search cues: Architectural Digest; subscription; cancelled"
    second.retrieval_text = "Memory: updated. Search cues: Architectural Digest; subscription; current"
    third.retrieval_text = "Memory: visited a clinic. Search cues: clinic; 9 AM; arrived"

    edges = _build_root_graph([first, second, third], graph_neighbor_k=1)

    assert any(
        edge.relation == "keyword_neighbor"
        and {edge.src, edge.dst} == {"root-a", "root-b"}
        for edge in edges
    )


def test_root_graph_adds_typed_anchor_edges_and_expands_them() -> None:
    first = _root("root-a", "a", [1.0, 0.0], ["leaf-a"])
    second = _root("root-b", "b", [0.0, 1.0], ["leaf-b"])
    third = _root("root-c", "c", [0.0, -1.0], ["leaf-c"])
    first.anchor_terms = {"entities": ["Architectural Digest"], "actions": ["subscribed"]}
    second.anchor_terms = {"entities": ["Architectural Digest"], "actions": ["cancelled"]}
    third.anchor_terms = {"entities": ["Clinic"], "times": ["9 AM"]}

    edges = _build_root_graph(
        [first, second, third],
        graph_neighbor_k=1,
        enable_typed_edges=True,
    )
    expanded, used_edges = _expand_root_ids(["root-a"], edges, graph_neighbor_k=1)

    assert any(
        edge.relation == "entity_neighbor"
        and {edge.src, edge.dst} == {"root-a", "root-b"}
        for edge in edges
    )
    assert "root-b" in expanded
    assert used_edges


def test_summary_anchor_terms_extracts_quoted_titles_and_state_phrases() -> None:
    anchors = _summary_anchor_terms(
        None,
        'User: I am currently devouring "The Seven Husbands of Evelyn Hugo". '
        "I am storing my old sneakers in a shoe rack in my closet.",
        "2026/05/22",
    )

    assert "The Seven Husbands of Evelyn Hugo" in anchors["entities"]
    assert any("Seven Husbands" in phrase for phrase in anchors["state_phrases"])
    assert any("shoe rack" in phrase for phrase in anchors["state_phrases"])


def test_summary_anchor_terms_extracts_speaker_action_attribute_phrases() -> None:
    anchors = _summary_anchor_terms(
        None,
        "Assistant (Melanie) -> Caroline: I ran a charity race on Sunday. "
        "Summary: Melanie ran a charity race. Caroline moved from Sweden. "
        "Caroline signed up for a counseling certification.",
        "2023/05/25",
    )

    phrases = " | ".join(anchors["state_phrases"])
    assert "Melanie ran charity race" in phrases
    assert "moved from Sweden" in phrases
    assert "signed up for counseling certification" in phrases


def test_root_graph_adds_state_phrase_edges() -> None:
    first = _root("root-a", "a", [1.0, 0.0], ["leaf-a"])
    second = _root("root-b", "b", [0.0, 1.0], ["leaf-b"])
    third = _root("root-c", "c", [0.0, -1.0], ["leaf-c"])
    first.anchor_terms = {"state_phrases": ["currently reading seven husbands"]}
    second.anchor_terms = {"state_phrases": ["currently reading seven husbands"]}
    third.anchor_terms = {"state_phrases": ["storing sneakers in closet"]}

    edges = _build_root_graph(
        [first, second, third],
        graph_neighbor_k=1,
        enable_typed_edges=True,
    )

    assert any(
        edge.relation == "state_neighbor"
        and {edge.src, edge.dst} == {"root-a", "root-b"}
        for edge in edges
    )


def test_typed_edges_get_small_expansion_priority() -> None:
    typed = GraphEdge("root-a", "root-b", 0.75, "state_neighbor")
    semantic = GraphEdge("root-a", "root-c", 0.76, "semantic_neighbor")

    assert _edge_expansion_sort_key(typed) > _edge_expansion_sort_key(semantic)


def test_qwen_local_summary_variant_uses_local_records_not_deepseek_build(tmp_path: Path) -> None:
    summarizer = MockLocalSummarizer("Qwen/Qwen3.5-2B")
    run = run_case(
        DemoConfig(
            data_path=tmp_path / "unused.json",
            output_dir=tmp_path / "run",
            variants=("qwen35_2b_summary_graphmem",),
            mock_services=True,
        ),
        _case_with_leaf_count(2),
        "qwen35_2b_summary_graphmem",
        MockDeepSeekClient(),
        MockEmbeddingClient(),
        MockCompressor(0.5),
        summarizer=summarizer,
    )
    assert {record.stage for record in run.llm_records} == {"answer_qa"}
    assert summarizer.records
    assert summarizer.records[0].compressor == "qwen_local:Qwen/Qwen3.5-2B"
    assert run.summaries[0].summary


def test_single_llm_variant_preserves_raw_assistant_evidence_and_larger_summary_budget(
    tmp_path: Path,
) -> None:
    config = DemoConfig(
        data_path=tmp_path / "unused.json",
        output_dir=tmp_path / "run",
        variants=("single_llm_summary_graphmem",),
        mock_services=True,
    )
    spec = _variant_spec(config, "single_llm_summary_graphmem")
    assert spec.summary_schema == "compact_memory_v2"
    assert spec.build_leaf_text == "user_only"
    assert spec.retrieval_leaf_text == "user_only"
    assert spec.raw_question_types == ("single-session-assistant", "single-session-preference")
    assert spec.summary_max_tokens == 512
    assistant_case = _synthetic_case("What did you recommend?")
    assistant_case.question_type = "single-session-assistant"
    update_case = _synthetic_case("How many Instagram followers do I currently have?")
    update_case.question_type = "knowledge-update"
    assert _effective_leaf_text_mode("auto", spec, assistant_case, "build") == "raw"
    assert _effective_leaf_text_mode("auto", spec, assistant_case, "retrieval") == "raw"
    assert _effective_leaf_text_mode("auto", spec, update_case, "build") == "user_only"

    run = run_case(
        config,
        _case_with_leaf_count(2),
        "single_llm_summary_graphmem",
        MockDeepSeekClient(),
        MockEmbeddingClient(),
        MockCompressor(0.5),
    )
    summary_calls = [record for record in run.llm_records if record.stage.startswith("build_summary")]
    assert summary_calls
    assert all(record.max_tokens == 512 for record in summary_calls)


def test_compact_summary_prompt_keeps_assistant_previous_conversation_answers() -> None:
    messages = _summary_messages(
        "s1",
        "2026/05/22",
        "build_summary_session_direct",
        "Assistant: The SIAC_GEE tool implements the 6S algorithm.",
        "compact_memory_v2",
    )
    system = messages[0]["content"]
    assert "assistant-provided" in system
    assert "tables" in system
    assert "previous conversation" in system


def test_enhanced_hybrid_retrieval_seeds_each_candidate_root_session(tmp_path: Path) -> None:
    config = DemoConfig(
        data_path=tmp_path / "unused.json",
        output_dir=tmp_path / "run",
        root_candidate_k=3,
        root_top_k=1,
        global_leaf_top_k=1,
        leaf_top_k=3,
        per_session_leaf_k=1,
        mock_services=True,
    )
    leaves = [
        _leaf("new-yorker-leaf", "new-yorker", [1.0, 0.0]),
        _leaf("architectural-leaf", "architectural", [0.9, 0.1]),
        _leaf("forbes-leaf", "forbes", [0.8, 0.2]),
    ]
    leaves[1].raw_text = "User: I currently have an Architectural Digest subscription."
    roots = [
        _root("new-yorker-root", "new-yorker", [1.0, 0.0], [leaves[0].node_id]),
        _root("architectural-root", "architectural", [0.9, 0.1], [leaves[1].node_id]),
        _root("forbes-root", "forbes", [0.8, 0.2], [leaves[2].node_id]),
    ]
    _, selected, _ = _retrieve_hybrid(
        config,
        "How many magazine subscriptions do I currently have?",
        leaves,
        roots,
        [],
        [1.0, 0.0],
        False,
        True,
    )
    assert {leaf.session_id for leaf in selected} >= {"new-yorker", "architectural", "forbes"}


def test_multilevel_summary_candidates_can_narrow_leaf_candidates(tmp_path: Path) -> None:
    config = DemoConfig(
        data_path=tmp_path / "unused.json",
        output_dir=tmp_path / "run",
        root_candidate_k=1,
        global_leaf_top_k=0,
        leaf_top_k=1,
        per_session_leaf_k=1,
        mock_services=True,
    )
    target_leaf = _leaf("leaf-target", "long-session", [0.0, 1.0])
    target_leaf.raw_text = "User: I stored the old sneakers in the closet shoe rack."
    distractor_leaf = _leaf("leaf-distractor", "long-session", [1.0, 0.0])
    distractor_leaf.raw_text = "User: I discussed unrelated closet organization ideas."
    session_root = _root(
        "root-session",
        "long-session",
        [0.7, 0.0],
        [target_leaf.node_id, distractor_leaf.node_id],
    )
    target_group = _root("summary-target", "long-session", [1.0, 0.0], [target_leaf.node_id])
    distractor_group = _root(
        "summary-distractor",
        "long-session",
        [0.0, 1.0],
        [distractor_leaf.node_id],
    )

    _, root_only_leaves, _ = _retrieve_hybrid(
        config,
        "Where do I currently keep my old sneakers?",
        [target_leaf, distractor_leaf],
        [session_root],
        [],
        [1.0, 0.0],
        False,
        False,
    )
    _, multilevel_leaves, _ = _retrieve_hybrid(
        config,
        "Where do I currently keep my old sneakers?",
        [target_leaf, distractor_leaf],
        [session_root, target_group, distractor_group],
        [],
        [1.0, 0.0],
        False,
        False,
    )

    assert [leaf.node_id for leaf in root_only_leaves] == ["leaf-distractor"]
    assert [leaf.node_id for leaf in multilevel_leaves] == ["leaf-target"]


def test_context_budget_trims_summaries_before_raw_leaves() -> None:
    leaves = [_leaf("leaf-1", "s1", [1.0, 0.0]), _leaf("leaf-2", "s2", [0.0, 1.0])]
    leaves[0].raw_text = "User: " + "important " * 120
    leaves[1].raw_text = "User: " + "secondary " * 120
    summaries = [
        _root("summary-1", "s1", [1.0, 0.0], [leaves[0].node_id]),
        _root("summary-2", "s2", [0.0, 1.0], [leaves[1].node_id]),
    ]
    summaries[0].summary = "summary " * 200
    summaries[1].summary = "summary " * 200
    kept_summaries, kept_leaves = _fit_context_budget(summaries, leaves, 260)
    assert len(kept_summaries) < len(summaries)
    assert kept_leaves
    assert len(kept_leaves) <= len(leaves)


def test_evidence_context_budget_reserves_answer_completion_margin(tmp_path: Path) -> None:
    case = _synthetic_case("What is the total cost of the items I bought?")
    config = DemoConfig(
        data_path=tmp_path / "unused.json",
        output_dir=tmp_path / "run",
        qa_context_token_budget=10000,
        qa_max_tokens=1024,
        mock_services=True,
    )

    budget = _evidence_context_budget(config, case, enhanced_qa=True)

    assert budget <= 10000 - 2400


def test_explicit_speaker_context_expands_neighboring_turns() -> None:
    leaves = [
        _leaf("s1-l0", "s1", [1.0, 0.0]),
        _leaf("s1-l1", "s1", [1.0, 0.0]),
        _leaf("s1-l2", "s1", [1.0, 0.0]),
        _leaf("s2-l0", "s2", [0.0, 1.0]),
    ]
    for index, leaf in enumerate(leaves[:3]):
        leaf.turn_index = index * 2
    selected = [leaves[1], leaves[3]]

    expanded = _expand_selected_session_context(
        selected,
        leaves,
        "Where did Caroline move from?",
        "category_1",
        4,
        explicit_speaker=True,
    )

    assert [leaf.node_id for leaf in expanded[:2]] == ["s1-l1", "s2-l0"]
    assert {leaf.node_id for leaf in expanded[2:]} == {"s1-l0", "s1-l2"}


def test_enhanced_answer_prompt_demands_evidence_and_time_math() -> None:
    case = _synthetic_case("What time did I reach the clinic on Monday?")
    prompt = _answer_messages(case, "Left at 7 AM. Trip took two hours.", enhanced=True)
    system = prompt[0]["content"]
    assert "Evidence facts:" in system
    assert "Final answer:" in system
    assert "elapsed time calculation" in system
    assert "latest known value as current" in system
    assert "later addition" in system
    assert "sort by date" in system
    assert "not mentioned" in system
    assert "preference or advice questions" in system
    assert "assistant messages" in system
    assert "Before giving a numeric count" in system
    assert "Do not count recommendations" in system
    assert "currently-own or currently-use" in system
    assert "attended/visited/completed" in system


def _record(stage: str, prompt: int, completion: int, reasoning_tokens: int = 0) -> DeepSeekCallRecord:
    return DeepSeekCallRecord(
        question_id="q",
        variant="v",
        stage=stage,  # type: ignore[arg-type]
        call_id=stage,
        model="m",
        thinking_mode="none",
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        reasoning_tokens=reasoning_tokens,
    )


def _synthetic_row() -> dict:
    return {
        "question_id": "demo-q",
        "question_type": "multi-session",
        "question": "What time did I reach the clinic on Monday?",
        "answer": "9:00 AM",
        "question_date": "2026/05/22",
        "answer_session_ids": ["answer-session"],
        "haystack_session_ids": ["distractor-session", "answer-session"],
        "haystack_dates": ["2026/05/20", "2026/05/21"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I practiced guitar after work."},
                {"role": "assistant", "content": "Keep a steady schedule."},
            ],
            [
                {"role": "user", "content": "On Monday I reached the clinic at 9:00 AM."},
                {"role": "assistant", "content": "I will remember the clinic arrival time."},
            ],
        ],
    }


def _synthetic_case(question: str) -> QuestionCase:
    row = _synthetic_row()
    row["question"] = question
    return QuestionCase(
        question_id=row["question_id"],
        question_type=row["question_type"],
        question=row["question"],
        answer=row["answer"],
        question_date=row["question_date"],
        haystack_sessions=row["haystack_sessions"],
        haystack_session_ids=row["haystack_session_ids"],
        haystack_dates=row["haystack_dates"],
        answer_session_ids=row["answer_session_ids"],
    )


def _run_synthetic_case(
    tmp_path: Path,
    case: QuestionCase,
    *,
    variant: str,
    llm: MockDeepSeekClient | None = None,
    max_group_rough_tokens: int = 6000,
    max_inflight_deepseek: int = 0,
):
    config = DemoConfig(
        data_path=tmp_path / "unused.json",
        output_dir=tmp_path / "run",
        variants=(variant,),
        max_group_rough_tokens=max_group_rough_tokens,
        max_inflight_deepseek=max_inflight_deepseek,
        enable_speaker_profiles=getattr(case, "enable_speaker_profiles", False),
        mock_services=True,
    )
    return run_case(
        config,
        case,
        variant,
        llm or MockDeepSeekClient(),
        MockEmbeddingClient(),
        MockCompressor(config.compression_ratio),
    )


def _case_with_leaf_count(leaf_count: int) -> QuestionCase:
    messages = []
    for index in range(leaf_count):
        messages.extend(
            [
                {"role": "user", "content": f"On day {index} I visited place {index}."},
                {"role": "assistant", "content": f"Remember visit {index}."},
            ]
        )
    return QuestionCase(
        question_id=f"many-leaves-{leaf_count}",
        question_type="multi-session",
        question="Where did I visit?",
        answer="places",
        question_date="2026/05/22",
        haystack_sessions=[messages],
        haystack_session_ids=["session-1"],
        haystack_dates=["2026/05/21"],
        answer_session_ids=["session-1"],
    )


def _leaf(node_id: str, session_id: str, embedding: list[float]) -> LeafNode:
    return LeafNode(
        node_id=node_id,
        question_id="q",
        session_id=session_id,
        session_date=None,
        turn_index=0,
        raw_text=f"User: {session_id}",
        user_text=f"User: {session_id}",
        message_count=1,
        embedding=embedding,
    )


def _root(node_id: str, session_id: str, embedding: list[float], leaf_ids: list[str]) -> SummaryNode:
    return SummaryNode(
        node_id=node_id,
        question_id="q",
        session_id=session_id,
        session_date=None,
        level=1,
        child_ids=leaf_ids,
        leaf_ids=leaf_ids,
        summary=session_id,
        embedding=embedding,
    )


class _MalformedSummaryLLM(MockDeepSeekClient):
    def chat(self, **kwargs):
        result = super().chat(**kwargs)
        if str(kwargs["stage"]).startswith("build_summary"):
            result.record.finish_reason = "length"
            return LLMResult(text="not json", record=result.record)
        return result


class _SlowTrackingLLM(MockDeepSeekClient):
    def __init__(self) -> None:
        self.calls: list[tuple[str, float, float]] = []
        self.peak_build_calls = 0
        self._active = 0
        self._lock = threading.Lock()

    def chat(self, **kwargs):
        stage = str(kwargs["stage"])
        start = time.perf_counter()
        if stage.startswith("build_summary"):
            with self._lock:
                self._active += 1
                self.peak_build_calls = max(self.peak_build_calls, self._active)
            time.sleep(0.03)
        result = super().chat(**kwargs)
        end = time.perf_counter()
        if stage.startswith("build_summary"):
            with self._lock:
                self._active -= 1
                self.calls.append((stage, start, end))
        return result
