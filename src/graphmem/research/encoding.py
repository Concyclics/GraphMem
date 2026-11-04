"""Two embedding views used by the frozen ranker; explicit token ledgers."""

from pathlib import Path
from collections import Counter
import json
import sqlite3
import time
import hashlib
import numpy as np
from .data import read_rows, write_rows, sha
from .embedding_views import jobs

BASE_INSTRUCTION = "Instruct: Retrieve conversation turns that provide all evidence needed to answer the memory question.\nQuery: "


def base_queries(root, model, meter):
    """Use the same query-cache schema as QwenEmbeddingIndex; no hidden retries."""
    root = Path(root)
    db = sqlite3.connect(root / "input/query_embeddings.sqlite")
    db.execute(
        "CREATE TABLE IF NOT EXISTS query_embeddings (cache_key TEXT PRIMARY KEY, dimension INTEGER NOT NULL, vector BLOB NOT NULL, created_at REAL NOT NULL, last_accessed REAL NOT NULL, hits INTEGER NOT NULL DEFAULT 0)"
    )
    usage = []
    for q in read_rows(root / "input/queries.jsonl"):
        text = BASE_INSTRUCTION + q["query"]
        # Meter caches by request hash and retains the original real API usage.
        value, key = meter.call(
            q["qid"],
            "base_query_embedding",
            {"model": model, "input": [text]},
            api="base_embedding",
            kind="embedding",
        )
        vector = np.asarray(value["data"][0]["embedding"], dtype=np.float32)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("Invalid query embedding")
        vector /= max(np.linalg.norm(vector), 1e-9)
        cache_key = hashlib.sha256(
            (model + "\ngraphmem-turn-evidence-v1\n" + text).encode()
        ).hexdigest()
        db.execute(
            "INSERT OR REPLACE INTO query_embeddings VALUES (?,?,?,?,?,?)",
            (cache_key, len(vector), vector.tobytes(), time.time(), time.time(), 0),
        )
        db.commit()
        usage.append(
            dict(
                qid=q["qid"],
                base_query_tokens=value["usage"]["prompt_tokens"],
                call_key=key,
            )
        )
    db.close()
    write_rows(root / "base_query_usage.jsonl", usage)


def encode_views(root, benchmark, model_path, *, batch_size=16, device="cuda"):
    """Local BF16 Qwen encoder: left padding, 2048-token cap, last-token pooling.

    Vectors are normalized in float32, stored as float16, then normalized again
    by the feature exporter. These precision details match the evaluated cache.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    root = Path(root)
    out = root / "qwen_embeddings"
    out.mkdir(exist_ok=True)
    rows = jobs(root, benchmark)
    write_rows(root / "input/embedding_jobs.jsonl", rows)
    identity = {
        "jobs_sha256": sha(root / "input/embedding_jobs.jsonl"),
        "model": str(model_path),
        "max_length": 2048,
        "dtype": "bfloat16",
        "pooling": "last_token",
        "stored_dtype": "float16",
        "batch_size": batch_size,
    }
    marker = out / "identity.json"
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise ValueError("Embedding identity changed; use a new output directory")
    marker.write_text(json.dumps(identity, indent=2) + "\n")
    tok = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, padding_side="left"
    )
    model = (
        AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )
    lengths = []
    for start in range(0, len(rows), 256):
        lengths.extend(
            len(r.ids)
            for r in tok.backend_tokenizer.encode_batch(
                [r["text"] for r in rows[start : start + 256]], add_special_tokens=True
            )
        )
    order = sorted(range(len(rows)), key=lambda i: lengths[i])
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        path = out / f"batch_{start // batch_size:05d}.npz"
        ids = [rows[i]["id"] for i in idx]
        if path.exists():
            with np.load(path) as z:
                if z["ids"].tolist() != ids or not np.isfinite(z["vectors"]).all():
                    raise ValueError("Invalid embedding checkpoint")
            continue
        inputs = tok(
            [rows[i]["text"] for i in idx],
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state[:, -1]
            vectors = (
                torch.nn.functional.normalize(hidden.float(), p=2, dim=1)
                .cpu()
                .numpy()
                .astype(np.float16)
            )
        temp = path.with_suffix(".tmp.npz")
        np.savez_compressed(temp, ids=np.array(ids), vectors=vectors)
        temp.replace(path)
        if start % (batch_size * 100) == 0:
            print(f"Embedded {start + len(idx)}/{len(rows)}", flush=True)
    manifest = {
        **identity,
        "jobs": len(rows),
        "total_tokens": sum(lengths),
        "tokens_by_kind": {
            k: sum(n for r, n in zip(rows, lengths) if r["kind"] == k)
            for k in ["raw", "graph", "query"]
        },
        "embedding_truncated_jobs": sum(n > 2048 for n in lengths),
        "attended_tokens": sum(min(n, 2048) for n in lengths),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_rows(
        root / "enhanced_query_usage.jsonl",
        [
            dict(qid=r["id"][6:], enhanced_query_tokens=n, attended_tokens=min(n, 2048))
            for r, n in zip(rows, lengths)
            if r["kind"] == "query"
        ],
    )
    return manifest


def query_usage(root):
    root = Path(root)
    # An externally supplied verified cache may carry this exact same schema.
    base = {r["qid"]: r for r in read_rows(root / "base_query_usage.jsonl")}
    enhanced = {r["qid"]: r for r in read_rows(root / "enhanced_query_usage.jsonl")}
    expected = {q["qid"] for q in read_rows(root / "input/queries.jsonl")}
    if base.keys() != expected or enhanced.keys() != expected:
        raise ValueError("Incomplete query embedding usage")
    result = {}
    for qid in expected:
        a, b = base[qid]["base_query_tokens"], enhanced[qid]["enhanced_query_tokens"]
        if any(type(v) != int or v < 0 for v in [a, b]):
            raise ValueError("Missing or invalid query embedding tokens")
        result[qid] = a + b
    return result
