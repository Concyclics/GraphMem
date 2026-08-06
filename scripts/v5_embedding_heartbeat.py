#!/usr/bin/env python3
"""Keep the local embedding GPU active with a bounded, auditable request."""

from __future__ import annotations

import argparse
import json
import random
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8001/v1/embeddings")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--interval-sec", type=float, default=0.2,
                        help="minimum post-request idle interval")
    parser.add_argument("--jitter-sec", type=float, default=0.6,
                        help="uniform random extra idle interval")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--batch-jitter", type=int, default=64,
                        help="randomly reduce each request batch by at most this many rows")
    parser.add_argument("--text-words", type=int, default=256)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.interval_sec < 600 or not 0 <= args.jitter_sec < 600:
        raise ValueError("heartbeat interval and jitter must be between 0 and 599 seconds")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    if args.batch_size <= 0 or args.text_words <= 0 or args.batch_jitter < 0:
        raise ValueError("heartbeat batch and text length must be positive")
    base = " ".join(["graphmem"] * args.text_words)
    rng = random.SystemRandom()
    while True:
        started = time.monotonic()
        row = {"timestamp": datetime.now(timezone.utc).isoformat()}
        batch_size = rng.randint(max(1, args.batch_size - args.batch_jitter), args.batch_size)
        nonce = rng.randrange(1 << 63)
        # A fresh nonce and a slightly varied shape avoid a fixed, easily
        # cacheable request cadence while remaining self-contained and cheap.
        payload = json.dumps({
            "model": args.model,
            "input": [f"{base} heartbeat-{nonce}-{index}" for index in range(batch_size)],
        }).encode()
        try:
            request = urllib.request.Request(
                args.url, data=payload, method="POST",
                headers={"Content-Type": "application/json", "Authorization": "Bearer local-vllm"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                body = json.loads(response.read())
            row.update({
                "ok": response.status == 200,
                "dimensions": len(body["data"][0]["embedding"]),
                "batch_size": batch_size,
                "text_words": args.text_words,
                "post_request_idle_sec": args.interval_sec,
                "jitter_sec": args.jitter_sec,
                "latency_sec": time.monotonic() - started,
            })
        except Exception as error:  # service lifecycle is recorded, not hidden
            row.update({"ok": False, "error": repr(error), "latency_sec": time.monotonic() - started})
        with args.log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        time.sleep(args.interval_sec + rng.uniform(0, args.jitter_sec))


if __name__ == "__main__":
    main()
