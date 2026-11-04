"""Cold-build adapter for the repository's existing index implementation."""

from dataclasses import replace, asdict
from pathlib import Path
from types import SimpleNamespace
import json
import os
from .data import sha, read_rows, write_rows
from .transport import Meter, digest


def build(benchmark, data, output, config_path, tokenizer):
    from openai.types.chat import ChatCompletion
    from openai.types.create_embedding_response import CreateEmbeddingResponse
    from graphmem.service.config import ServiceConfig
    from graphmem.storage import SQLiteGraphStore
    from graphmem.eval.fullset import load_full_questions
    from graphmem.eval.devset import ingest_questions
    from graphmem.build import GraphBuildPipeline, QwenSemanticDistiller
    from graphmem.build.canonicalize import PredicateCanonicalizer
    from graphmem.embedding import QwenEmbeddingIndex
    from .locomo_format import convert_sample

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = replace(ServiceConfig.from_file(config_path), data_dir=str(output))
    build_cfg, runtime = config.profiles()
    identity = dict(
        benchmark=benchmark,
        data_sha256=sha(data),
        tokenizer_sha256=sha(tokenizer),
        config=config.to_dict(),
    )
    marker = output / "build_identity.json"
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise ValueError("Build inputs/config changed; use a new directory")
    marker.write_text(json.dumps(identity, indent=2) + "\n")
    (output / "resolved_build_profile.json").write_text(
        json.dumps(asdict(build_cfg), indent=2) + "\n"
    )
    os.environ["GRAPHMEM_TOKENIZER_PATH"] = str(Path(tokenizer).resolve())
    if benchmark == "locomo":
        samples = json.loads(Path(data).read_text())
        converted = output / "converted_locomo.json"
        converted.write_text(
            json.dumps(
                [r for i, s in enumerate(samples) for r in convert_sample(s, i)],
                ensure_ascii=False,
            )
        )
        questions = load_full_questions(locomo_path=converted)
    else:
        questions = load_full_questions(lme_path=Path(data))
    endpoints = {
        "build_chat": dict(
            base_url=config.llm_base_url,
            model=config.llm_model,
            key_env=config.llm_api_key_env,
        ),
        "build_embedding": dict(
            base_url=config.embedding_base_url,
            model=config.embedding_model,
            key_env=config.embedding_api_key_env,
        ),
    }
    meter = Meter(output / "build_calls.jsonl", endpoints)

    def client(api, kind, response_type):
        def create(**kwargs):
            body = dict(kwargs)
            body.pop("timeout", None)
            body.update(body.pop("extra_body", {}) or {})
            value, _ = meter.call(
                "build", api + ":" + digest(body), body, api=api, kind=kind
            )
            return response_type.model_validate(value)

        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
            embeddings=SimpleNamespace(create=create),
        )

    chat = client("build_chat", "chat", ChatCompletion)
    embed = client("build_embedding", "embedding", CreateEmbeddingResponse)
    store = SQLiteGraphStore(output / "memory.sqlite")
    ingest_questions(store, questions)
    embedding = QwenEmbeddingIndex(
        store, build_cfg, client=embed, **runtime.retrieval.embedding_options()
    )
    done = (
        {r["memory_id"]: r for r in read_rows(output / "built.jsonl")}
        if (output / "built.jsonl").exists()
        else {}
    )
    try:
        for mid in sorted({q.memory_id for q in questions}):
            if mid in done:
                continue
            embedding.index_memory(mid)
            pipeline = GraphBuildPipeline(
                store,
                dataset_hash=identity["data_sha256"],
                distiller=QwenSemanticDistiller(
                    store,
                    build_cfg,
                    identity["data_sha256"],
                    client=chat,
                    worker_limit=4,
                    request_profile=config.request_profile,
                ),
                predicate_canonicalizer=PredicateCanonicalizer(
                    store, build_cfg, client=embed
                ),
                coarsen_vector_provider=embedding.embed_graph_nodes,
                relation_vector_provider=embedding.embed_graph_nodes,
            )
            pipeline.build(mid, build_cfg)
            embedding.embed_graph_nodes(mid, store.nodes(mid))
            done[mid] = dict(
                memory_id=mid,
                graph_version=store.graph_version(mid),
                turns=len(store.turns(mid)),
            )
            write_rows(output / "built.jsonl", done.values())
            print(f"Built {len(done)} histories", flush=True)
    finally:
        store.close()
    groups = {}
    for api in endpoints:
        keys = [k for k, r in meter.cache.items() if r["api"] == api]
        groups[api] = {"calls": len(keys), "usage": meter.usage(keys)}
    summary = {
        "histories": len(done),
        "model_usage": groups,
        "scope": "Fresh rebuild; distinct from the historical frozen graph snapshot.",
    }
    (output / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
