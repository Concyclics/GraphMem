#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=args.base_url, api_key="local")
    while True:
        started = time.perf_counter()
        row = {"timestamp": datetime.now(timezone.utc).isoformat(), "heartbeat": True}
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": "Reply only: OK"}],
                temperature=0,
                max_tokens=2,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            message = response.choices[0].message
            row.update({
                "ok": True, "content": message.content,
                "reasoning_empty": not bool(getattr(message, "reasoning_content", None)),
                "total_tokens": int(getattr(response.usage, "total_tokens", 0) or 0),
            })
        except Exception as error:
            row.update({"ok": False, "error": repr(error)})
        row["latency_sec"] = time.perf_counter() - started
        with args.log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        time.sleep(max(1.0, args.interval))


if __name__ == "__main__":
    main()
