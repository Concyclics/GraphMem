"""Resumable HTTP ledger. Failed transports remain unknown, never zero-cost."""

from pathlib import Path
import hashlib
import json
import os
import threading
import time
import urllib.request
from urllib.parse import urlsplit
from .data import read_rows


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def native_usage(value):
    if not isinstance(value, dict):
        return None
    a = value.get("input_tokens", value.get("prompt_tokens"))
    b = value.get("output_tokens", value.get("completion_tokens", 0))
    if not isinstance(a, int) or not isinstance(b, int) or min(a, b) < 0:
        return None
    total = value.get("total_tokens", a + b)
    if total != a + b:
        raise ValueError("Provider total tokens disagree with input + output")
    return {"prompt_tokens": a, "completion_tokens": b, "total_tokens": total}


def read_ledger(path):
    """Fold durable start/finish events; retain orphan starts as unknown calls."""
    rows = read_rows(path) if Path(path).exists() else []
    calls = {}
    for row in rows:
        key = row["key"]
        prior = calls.get(key)
        if prior is not None:
            if (
                prior.get("event") != "start"
                or row.get("event") != "finish"
                or row.get("request_sha256") != prior.get("request_sha256")
            ):
                raise ValueError("Duplicate or mismatched call lifecycle: " + key)
        elif row.get("event") == "finish":
            raise ValueError("Call finish without durable start: " + key)
        calls[key] = row
    return list(calls.values())


class Meter:
    def __init__(self, path, endpoints, *, timeout=1800):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.endpoints = endpoints
        self.timeout = timeout
        self.lock = threading.Lock()
        self.key_locks = {}
        rows = read_ledger(self.path)
        self.cache = {r["key"]: r for r in rows}
        if len(self.cache) != len(rows):
            raise ValueError("Duplicate call key in ledger")
        for config in endpoints.values():
            url = urlsplit(config["base_url"])
            if url.username or url.password or url.query or url.fragment:
                raise ValueError(
                    "Endpoint must not contain credentials, query parameters or fragments"
                )
            if url.scheme not in {"http", "https"}:
                raise ValueError("HTTP(S) endpoint required")

    def call(self, qid, stage, body, api="qwen", kind="chat"):
        with self.lock:
            guard = self.key_locks.setdefault(qid + ":" + stage, threading.Lock())
        with guard:
            return self._call(qid, stage, body, api, kind)

    def _record(self, row):
        with self.lock:
            with self.path.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            self.cache[row["key"]] = dict(row)

    def _call(self, qid, stage, body, api, kind):
        config = self.endpoints[api]
        body = {**body, "model": config["model"]}
        suffix = {
            "chat": "chat/completions",
            "responses": "responses",
            "embedding": "embeddings",
        }[kind]
        url = config["base_url"].rstrip("/") + "/" + suffix
        request_hash = digest({"url": url, "body": body})
        key = qid + ":" + stage
        with self.lock:
            saved = self.cache.get(key)
        if saved:
            if saved["request_sha256"] != request_hash:
                raise ValueError("Request changed within a run: " + key)
            if saved.get("status") != "ok" or saved.get("usage") is None:
                raise RuntimeError(
                    "Unresolved prior call; inspect its ledger before a new run: " + key
                )
            return saved["response"], key
        headers = {"Content-Type": "application/json"}
        env = config.get("key_env", "")
        secret = os.environ.get(env, "") if env else ""
        if config.get("require_key") and not secret:
            raise ValueError("Set the API key environment variable " + env)
        if secret:
            headers["Authorization"] = "Bearer " + secret
        row = dict(
            key=key,
            qid=qid,
            stage=stage,
            api=api,
            kind=kind,
            model=body["model"],
            request_sha256=request_hash,
            started_at=time.time(),
            status="unknown",
            usage=None,
            event="start",
        )
        self._record(row)
        start = time.monotonic()
        try:
            request = urllib.request.Request(
                url, data=json.dumps(body).encode(), headers=headers
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as stream:
                value = json.load(stream)
            row["response"] = value
            row["usage"] = native_usage(value.get("usage"))
            if row["usage"] is None:
                raise ValueError("Response has no valid token usage")
            row["status"] = "ok"
        except Exception as error:
            # Error bodies/headers can include secrets. Record only the class/code.
            row["error_type"] = type(error).__name__
            row["http_status"] = getattr(error, "code", None)
            raise
        finally:
            row["seconds"] = time.monotonic() - start
            row["event"] = "finish"
            self._record(row)
        return value, key

    def usage(self, keys):
        if len(keys) != len(set(keys)):
            raise ValueError("A call was charged twice")
        rows = [self.cache[k] for k in keys]
        if any(r["status"] != "ok" or r["usage"] is None for r in rows):
            return None
        return {
            k: sum(r["usage"][k] for r in rows)
            for k in ["prompt_tokens", "completion_tokens", "total_tokens"]
        }
