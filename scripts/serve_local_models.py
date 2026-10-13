#!/usr/bin/env python3
"""Run Qwen chat and embedding on one GPU, without touching other services."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

CHAT = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
EMBED = "Qwen/Qwen3-Embedding-0.6B"


def snapshot(model: str, supplied: str | None) -> str:
    if supplied:
        path = Path(supplied).expanduser().resolve()
        if not (path / "config.json").is_file():
            raise ValueError(f"model config missing: {path}")
        return str(path)
    from huggingface_hub import snapshot_download
    return snapshot_download(model, local_files_only=True)


def identity(pid: int) -> str:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return "" if fields[0] == "Z" else fields[19]
    except FileNotFoundError:
        return ""


def alive(row: dict) -> bool:
    return bool(row.get("start_time")) and identity(row["pid"]) == row["start_time"]


def ready(row: dict) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{row['port']}/v1/models", timeout=2) as r:
            return any(x["id"] == row["model"] for x in json.load(r)["data"])
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["start", "status", "stop"])
    parser.add_argument("--state-dir", type=Path, default=Path("runs/local-models"))
    parser.add_argument("--python", default=sys.executable, help="Python with vLLM installed")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--chat-path")
    parser.add_argument("--embedding-path")
    parser.add_argument("--chat-port", type=int, default=8002)
    parser.add_argument("--embedding-port", type=int, default=8001)
    parser.add_argument("--chat-memory", type=float, default=0.72)
    parser.add_argument("--embedding-memory", type=float, default=0.10)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--wait-seconds", type=int, default=600)
    args = parser.parse_args()
    state = args.state_dir.resolve()
    state.mkdir(parents=True, exist_ok=True)
    manifest = state / "services.json"
    rows = json.loads(manifest.read_text()) if manifest.exists() else []
    if args.command == "status":
        print(json.dumps([{**r, "alive": alive(r), "ready": ready(r)} for r in rows], indent=2))
        return
    if args.command == "stop":
        for row in rows:
            if alive(row):
                os.killpg(row["pid"], signal.SIGTERM)
        deadline = time.monotonic() + 30
        while any(alive(r) for r in rows) and time.monotonic() < deadline:
            time.sleep(1)
        for row in rows:
            if alive(row):
                raise RuntimeError(f"service still exiting; inspect {row['log']}")
        print("Managed model services stopped")
        return
    if any(alive(r) for r in rows):
        raise RuntimeError("managed services already running; use status or stop")
    if not (0 < args.chat_memory < 1 and 0 < args.embedding_memory < 1
            and args.chat_memory + args.embedding_memory < 0.95):
        raise ValueError("GPU memory fractions must be positive and sum to less than 0.95")
    if args.chat_port == args.embedding_port:
        raise ValueError("model ports must differ")
    for port in (args.chat_port, args.embedding_port):
        with socket.socket() as s:
            s.bind(("127.0.0.1", port))
    gpu_rows = subprocess.check_output([
        "nvidia-smi", f"--id={args.gpu}", "--query-gpu=uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True).strip().split(",")
    uuid, used, utilization = (x.strip() for x in gpu_rows)
    if int(used) > 512 or int(utilization) > 5:
        raise RuntimeError(f"GPU {args.gpu} is occupied ({used} MiB, {utilization}%)")
    models = [
        ("chat", CHAT, snapshot(CHAT, args.chat_path), args.chat_port, args.chat_memory),
        ("embedding", EMBED, snapshot(EMBED, args.embedding_path), args.embedding_port,
         args.embedding_memory),
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
           "HF_HUB_OFFLINE": "1",
           "OMP_NUM_THREADS": "4", "VLLM_USE_DEEP_GEMM": "0",
           "VLLM_MOE_USE_DEEP_GEMM": "0", "VLLM_USE_DEEP_GEMM_E8M0": "0",
           "VLLM_DISABLED_KERNELS": "FlashInferFp8DeepGEMMDynamicBlockScaledKernel,DeepGemmFp8BlockScaledMMKernel"}
    rows = []
    # Start sequentially: vLLM profiles free VRAM during startup.
    for name, model, path, port, fraction in models:
        command = [args.python, "-m", "vllm.entrypoints.openai.api_server",
                   "--model", path, "--served-model-name", model, "--host", "127.0.0.1",
                   "--port", str(port), "--max-model-len", str(args.max_model_len),
                   "--gpu-memory-utilization", str(fraction),
                   "--max-num-seqs", str(args.max_num_seqs)]
        if name == "embedding":
            command += ["--runner", "pooling", "--enforce-eager"]
        else:
            command += ["--max-num-batched-tokens", "16384", "--generation-config", "vllm"]
        logfile = state / f"{name}.log"
        with logfile.open("a") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        row = dict(name=name, pid=process.pid, start_time=identity(process.pid), gpu=uuid,
                   model=model, path=path, port=port, log=str(logfile), command=command)
        rows.append(row)
        manifest.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"Starting {name} on GPU {args.gpu}, port {port}; log: {logfile}", flush=True)
        deadline = time.monotonic() + args.wait_seconds
        while not ready(row):
            if process.poll() is not None:
                raise RuntimeError(f"{name} exited {process.returncode}; see {logfile}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"{name} startup timeout; see {logfile}; use status/stop")
            time.sleep(2)
        print(f"{name} ready", flush=True)


if __name__ == "__main__":
    main()
