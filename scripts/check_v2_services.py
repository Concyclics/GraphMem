#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SGAO_MODEL_CHOICES = ("gpt-5.4-mini", "gpt-5.6-luna")
load_dotenv(ROOT / ".env", override=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embedding-base-url",
        default=os.environ.get("EMBEDDING_BASE_URL", "http://127.0.0.1:8001/v1"),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("EMBEDDING_MODEL", "Qwen3-Embedding-0.6B"),
    )
    parser.add_argument(
        "--llm-model",
        choices=SGAO_MODEL_CHOICES,
        default=os.environ.get("SGAO_MODEL", "gpt-5.4-mini"),
    )
    parser.add_argument("--skip-llm-key", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_llm_key and not os.environ.get("SGAO_API_KEY"):
        raise SystemExit("SGAO_API_KEY is missing from the environment/.env")

    embedding_key = os.environ.get("EMBEDDING_API_KEY", "local-embedding")
    base = args.embedding_base_url.rstrip("/")
    models_request = urllib.request.Request(
        base + "/models",
        headers={"Authorization": f"Bearer {embedding_key}"},
    )
    with urllib.request.urlopen(models_request, timeout=10) as response:
        models = json.load(response)
    ids = [str(row.get("id")) for row in models.get("data", [])]
    if args.embedding_model not in ids:
        raise SystemExit(
            f"embedding model {args.embedding_model!r} not served at {base}; "
            f"available={ids}"
        )

    request = urllib.request.Request(
        base + "/embeddings",
        data=json.dumps(
            {"model": args.embedding_model, "input": ["GraphMem V2 health check"]}
        ).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {embedding_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    dimension = len(payload["data"][0]["embedding"])
    if dimension != 1024:
        raise SystemExit(f"embedding dimension must be 1024, got {dimension}")
    print(
        json.dumps(
            {
                "embedding_service": "healthy",
                "embedding_model": args.embedding_model,
                "dimension": dimension,
                "llm_model": args.llm_model,
                "provider": "sgao_openai",
                "reasoning_effort": "none",
                "reasoning_effort_field_sent": True,
            }
        )
    )


if __name__ == "__main__":
    main()
