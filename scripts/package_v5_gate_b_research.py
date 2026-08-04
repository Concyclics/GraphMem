#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--funnel", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = {
        "README.md": args.repo / "docs/V5_GATE_B_REPORT.md",
        **{f"calibration40/{name}": args.calibration / name for name in (
            "summary.json", "run_manifest.json", "metrics.csv", "metrics.parquet",
            "error_cases.jsonl", "token_usage.csv", "graph_manifest.json",
            "llm_calls.jsonl.gz", "ablation_report.html",
        )},
        **{f"final200/{name}": args.final / name for name in (
            "summary.json", "metrics.jsonl", "graph_manifest.json", "neo4j_parity.json",
        )},
        **{f"funnel40/{name}": args.funnel / name for name in (
            "summary.json", "metrics.jsonl",
        )},
        **{f"analysis/{name}": args.analysis / name for name in (
            "statistical_analysis.json", "pareto_points.csv",
            "pareto_recall_vs_evidence.svg",
        )},
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing package inputs: " + ", ".join(missing))
    manifest = {
        name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for name, path in files.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    with tempfile.TemporaryDirectory() as directory:
        manifest_path = Path(directory) / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        with tarfile.open(args.output, "w:gz") as archive:
            archive.add(manifest_path, arcname="manifest.json")
            for name, path in files.items():
                archive.add(path, arcname=name)
    print(args.output, args.output.stat().st_size)


if __name__ == "__main__":
    main()
