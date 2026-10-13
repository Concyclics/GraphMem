#!/usr/bin/env python3
"""Verify real-model HTTP write, answer, SIGKILL restart and backup restore."""
from __future__ import annotations

import argparse
import http.client
import json
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import sys


def start(config, directory):
    process = subprocess.Popen(
        [sys.executable, "-m", "graphmem", "serve", "--config", str(config),
         "--data-dir", str(directory), "--port", "0"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        if not selector.select(timeout=30):
            stop(process)
            raise RuntimeError("Memory service startup timeout")
        line = process.stdout.readline()
    if not line:
        raise RuntimeError(process.stderr.read())
    return process, int(json.loads(line)["listening"].rsplit(":", 1)[1])


def request(port, route, payload=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    headers = {"Content-Type": "application/json", "X-Tenant-ID": "smoke",
               "Authorization": "Bearer " + os.environ["GRAPHMEM_API_KEY"]}
    try:
        connection.request("GET" if payload is None else "POST", route,
                           None if payload is None else json.dumps(payload), headers)
        response = connection.getresponse()
        result = json.loads(response.read())
        if response.status != 200:
            raise RuntimeError((response.status, result))
        return result
    finally:
        connection.close()


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/memory-service.json"))
    parser.add_argument("--output", type=Path, required=True, help="New directory for durable smoke artifacts")
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    config["generate_answers"] = True
    config_path = root / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    os.environ["GRAPHMEM_API_KEY"] = "local-smoke-" + os.urandom(16).hex()
    base = "/v1/memories/alice"
    payload = {"session_id": "chat-1", "request_id": "event-1", "messages": [
        {"role": "user", "speaker": "Alice", "listener": "Bob", "timestamp": "2026-10-13T10:00:00",
         "content": "I adopted a dog named Luna. Luna is a golden retriever."},
        {"role": "assistant", "speaker": "Bob", "listener": "Alice", "timestamp": "2026-10-13T10:01:00",
         "content": "Congratulations on adopting Luna!"}]}
    question = {"query": "What is the name of Alice's dog?"}
    process, port = start(config_path, root / "data")
    try:
        written = request(port, base + "/messages", payload)
        first = request(port, base + "/prepare", question)
        answer = request(port, base + "/answer", question)
        assert "Luna" in str(first["messages"])
        assert "luna" in answer["prediction"].lower(), answer
        process.kill()
        process.wait(timeout=5)
    finally:
        stop(process)
    process, port = start(config_path, root / "data")
    try:
        second = request(port, base + "/prepare", question)
        replay = request(port, base + "/messages", payload)
        assert second["prompt_payload_hash"] == first["prompt_payload_hash"]
        assert replay["replayed"] and replay["turn_count"] == 2
        assert replay["graph_version"] == written["graph_version"]
        backup = root / "backup.sqlite"
        subprocess.run([sys.executable, "-m", "graphmem", "backup", "--data-dir", str(root / "data"),
                        "--output", str(backup)], check=True, capture_output=True, text=True)
    finally:
        stop(process)
    restored = root / "restored"
    restored.mkdir()
    shutil.copy2(backup, restored / "memory.sqlite")
    process, port = start(config_path, restored)
    try:
        assert request(port, base)["status"] == "ready"
        assert request(port, base + "/prepare", question)["prompt_payload_hash"] == first["prompt_payload_hash"]
        restored_answer = request(port, base + "/answer", question)
        assert restored_answer["prediction"] == answer["prediction"]
        assert restored_answer["cached"] is True
    finally:
        stop(process)
    result = {"http_write_prepare_answer": True, "sigkill_restart": True,
              "idempotent_retry": True, "online_backup_restore": True,
              "models_called": True, "prediction": answer["prediction"],
              "answer_prompt_tokens": answer["api_prompt_tokens"],
              "answer_completion_tokens": answer["completion_tokens"]}
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
