#!/usr/bin/env python3
"""Emit docs/V5_6_FREEZE.md from the frozen V5.4 full-200 artifacts."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path("/ssd3/chenhan/Spark_MemGraph_Dev")
RUN = ROOT / "artifacts/v5_4/full200_resume/v5_1_graph_ablation_full_20260805T103058Z"
REPO = ROOT / "GraphMem"
DB_SHA = "130cd67d58a063487b53a64b7dad743f8ea338337202b2492d92b039973944cd"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifests = json.loads((RUN / "graph_manifest.json").read_text())
    run_manifest = json.loads((RUN / "run_manifest.json").read_text())
    summary = json.loads((RUN / "summary.json").read_text())
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()

    config_hashes = sorted({row["config_hash"] for row in manifests})
    prompt_hashes = sorted({row["prompt_hashes"]["semantic_distill"] for row in manifests})
    schema_versions = sorted({row["schema_version"] for row in manifests})
    models = sorted({json.dumps(row["model_ids"], sort_keys=True) for row in manifests})
    checksum_of_checksums = hashlib.sha256(
        "\n".join(f"{row['memory_id']}:{row['graph_version']}:{row['graph_checksum']}"
                  for row in sorted(manifests, key=lambda item: item["memory_id"])).encode()
    ).hexdigest()

    gold_draft = ROOT / "artifacts/v5/lme_gold_turn_merged_draft_20260804.jsonl"
    gold_final = REPO / "eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl"

    rows = sorted(manifests, key=lambda item: item["memory_id"])
    nodes = sum(row["node_count"] for row in rows)
    edges = sum(row["edge_count"] for row in rows)
    groups = sum(row["evidence_group_count"] for row in rows)

    lines: list[str] = []
    add = lines.append
    add("# V5.6 freeze record: the immutable V5.4 authority graph")
    add("")
    add("V5.6 development treats the artifacts below as read-only. Track A opens the")
    add("SQLite authority with `read_only=True`; Track B writes its rebuilt graph to a")
    add("separate database with its own checksums. Any run whose manifest does not")
    add("reproduce the hashes on this page is not comparable to the V5.6 baseline.")
    add("")
    add("## Authority artifact")
    add("")
    add("| Field | Value |")
    add("| --- | --- |")
    add(f"| Run directory | `{RUN.relative_to(ROOT)}` |")
    add(f"| SQLite authority | `graphmem.sqlite` |")
    add(f"| SQLite sha256 | `{DB_SHA}` |")
    add(f"| Size | {(RUN / 'graphmem.sqlite').stat().st_size:,} bytes |")
    add(f"| GraphMem commit | `{commit}` |")
    add(f"| Schema version | `{', '.join(schema_versions)}` |")
    add(f"| Config hash | `{', '.join(config_hashes)}` |")
    add(f"| Semantic prompt hash | `{', '.join(prompt_hashes)}` |")
    add(f"| Models | `{'; '.join(models)}` |")
    add(f"| Memories | {len(rows)} |")
    add(f"| Questions | {run_manifest['questions']} |")
    add(f"| Nodes / edges / evidence groups | {nodes:,} / {edges:,} / {groups:,} |")
    add(f"| Per-memory checksum digest | `{checksum_of_checksums}` |")
    add("")
    add("The per-memory checksum digest is the SHA-256 of the newline-joined")
    add("`memory_id:graph_version:graph_checksum` triples listed below, sorted by")
    add("memory id. Recomputing it is the cheapest way to prove a graph is untouched.")
    add("")
    add("## Frozen retrieval baseline (as originally reported)")
    add("")
    overall = summary["v54"]["navigation"]["overall"]
    add("| Metric | Value |")
    add("| --- | ---: |")
    for key in ("turn_all_hit", "turn_recall", "candidate_turn_all_hit",
                "candidate_turn_recall", "session_all_hit", "evidence_tokens",
                "visited_nodes", "visited_edges"):
        add(f"| {key} | {overall[key]:.4f} |")
    add(f"| average_cold_equivalent_backbone_tokens | "
        f"{summary['v54']['average_cold_equivalent_backbone_tokens']:,.2f} |")
    add("")
    add("> These numbers were computed against the **pre-adjudication draft**")
    add("> annotations (see below). They are recorded here only so the re-baseline in")
    add("> `V5_6_REBASELINE.md` has something to be compared against. Do not quote them.")
    add("")
    add("## Gold annotation provenance (defect D0)")
    add("")
    add("The frozen run consumed the draft annotation file, not the finalized asset")
    add("that ships in the repository:")
    add("")
    add("| Role | Path | sha256 |")
    add("| --- | --- | --- |")
    add(f"| Used by the frozen run | `{gold_draft.relative_to(ROOT)}` | "
        f"`{sha256(gold_draft) if gold_draft.exists() else 'MISSING'}` |")
    add(f"| Finalized, adjudicated | `{gold_final.relative_to(ROOT)}` | "
        f"`{sha256(gold_final)}` |")
    add("")
    add("`run_manifest.json` records the draft hash as")
    add(f"`{run_manifest['input_hashes'][str(gold_draft)]}`. Every V5.6 run must instead")
    add("pass the finalized file, and its manifest must record the finalized hash.")
    add("")
    add("## Benchmark inputs")
    add("")
    add("| Input | sha256 |")
    add("| --- | --- |")
    for path, value in sorted(run_manifest["input_hashes"].items()):
        name = path.replace(str(ROOT) + "/", "")
        add(f"| `{name}` | `{value}` |")
    add("")
    add("## Per-memory graph checksums")
    add("")
    add("| memory_id | graph_version | node_count | edge_count | evidence_groups | graph_checksum |")
    add("| --- | ---: | ---: | ---: | ---: | --- |")
    for row in rows:
        add(f"| `{row['memory_id']}` | {row['graph_version']} | {row['node_count']} | "
            f"{row['edge_count']} | {row['evidence_group_count']} | `{row['graph_checksum']}` |")
    add("")

    out = REPO / "docs/V5_6_FREEZE.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(lines)} lines, {len(rows)} memories)")


if __name__ == "__main__":
    main()
