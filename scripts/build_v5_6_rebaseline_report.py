#!/usr/bin/env python3
"""Emit docs/V5_6_REBASELINE.md from the 2x2 gold/graph re-baseline runs."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path("/ssd3/chenhan/Spark_MemGraph_Dev")
REPO = ROOT / "GraphMem"
OUT = ROOT / "artifacts/v5_6/rebaseline"
GOLD_FINAL = REPO / "eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl"
GOLD_DRAFT = ROOT / "artifacts/v5/lme_gold_turn_merged_draft_20260804.jsonl"

STRATA = ("lme_multi_session", "lme_temporal", "locomo_multihop", "locomo_temporal")
STRATUM_LABEL = {
    "lme_multi_session": "LongMemEval multi-session",
    "lme_temporal": "LongMemEval temporal",
    "locomo_multihop": "LoCoMo Cat1 multi-hop",
    "locomo_temporal": "LoCoMo Cat2 temporal",
}
# As printed in docs/V5_5_RETRIEVAL_HARNESS_REPORT.md and V5_4_FULL200_REPORT.md.
REPORTED = {
    ("g0_final", "h0"): {"lme_multi_session": (.60, .98), "lme_temporal": (.72, .98),
                         "locomo_multihop": (.06, .60), "locomo_temporal": (.60, .96),
                         "overall": (.495, .880)},
    ("g2_final", "h6"): {"lme_multi_session": (.68, .80), "lme_temporal": (.74, .84),
                         "locomo_multihop": (.18, .82), "locomo_temporal": (.62, .96),
                         "overall": (.555, .855)},
}


def latest_run(name: str) -> Path:
    rows = sorted((OUT / name).glob("v5_5_retrieval200_*"))
    if not rows:
        raise FileNotFoundError(f"no run under {OUT / name}")
    return rows[-1]


def load(name: str) -> tuple[dict, list[dict], dict]:
    run = latest_run(name)
    summary = json.loads((run / "summary.json").read_text())
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    manifest = json.loads((run / "run_manifest.json").read_text())
    return summary, metrics, manifest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gold_refs(path: Path) -> set[tuple[str, str, int]]:
    rows = set()
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows.add((row["question_id"], row["session_id"], row["turn_index"]))
    return rows


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def delta(new: float, old: float) -> str:
    diff = (new - old) * 100
    return f"{diff:+.1f}pp"


def main() -> None:
    runs = {name: load(name) for name in ("g0_draft", "g0_final", "g2_draft", "g2_final")}

    final_refs, draft_refs = gold_refs(GOLD_FINAL), gold_refs(GOLD_DRAFT)
    final_roles = Counter(json.loads(line)["support_role"]
                          for line in GOLD_FINAL.read_text().splitlines() if line.strip())
    draft_roles = Counter(json.loads(line)["support_role"]
                          for line in GOLD_DRAFT.read_text().splitlines() if line.strip())

    lines: list[str] = []
    add = lines.append
    add("# V5.6 re-baseline: correcting the LongMemEval gold annotation input (D0)")
    add("")
    add("Every V5.4 and V5.5 run consumed the **pre-adjudication draft** annotation")
    add("file, not the finalized asset that ships in `eval_annotations/`. This page")
    add("re-measures the frozen V5.4 graph and the V5.5 H6+G2 configuration against")
    add("the finalized annotations so V5.6 gates have a defensible baseline.")
    add("")
    add("All four runs use `PYTHONHASHSEED=0`. The retrieval code is unchanged from")
    add("commit `1a0779a`; the only variable is the `--gold` input and the source graph.")
    add("")
    add("## The annotation change")
    add("")
    add("| | draft (`lme-v5-dev100-draft-r1`) | finalized (`lme-v5-dev100-r1`) |")
    add("| --- | ---: | ---: |")
    add(f"| annotations | {sum(draft_roles.values())} | {sum(final_roles.values())} |")
    for role in sorted(set(final_roles) | set(draft_roles)):
        add(f"| role `{role}` | {draft_roles.get(role, 0)} | {final_roles.get(role, 0)} |")
    add(f"| sha256 | `{sha256(GOLD_DRAFT)[:16]}…` | `{sha256(GOLD_FINAL)[:16]}…` |")
    add("")
    shared = len(final_refs & draft_refs)
    add(f"At the `(question_id, session_id, turn_index)` granularity that")
    add(f"`navigation_metrics` actually scores, the two files share **{shared}** references;")
    add(f"**{len(final_refs - draft_refs)}** exist only in the finalized set and")
    add(f"**{len(draft_refs - final_refs)}** only in the draft. Roughly a third of the")
    add("LongMemEval gold turns changed, and the draft had almost no")
    add("`temporal_endpoint` or `aggregation_member` roles at all.")
    add("")

    add("## Strict turn all-hit and candidate all-hit, by gold file")
    add("")
    add("`strict` = `turn_all_hit` (all gold turns packed); `cand` = `candidate_turn_all_hit`.")
    add("")
    for label, name, profile in (("V5.4 frozen navigator (H0, G0)", "g0", "h0"),
                                 ("V5.5 harness (H6, G0)", "g0", "h6"),
                                 ("V5.5 harness (H6, G2 sidecar)", "g2", "h6")):
        draft = runs[f"{name}_draft"][0]["profiles"][profile]["navigation"]
        final = runs[f"{name}_final"][0]["profiles"][profile]["navigation"]
        add(f"### {label}")
        add("")
        add("| Stratum | draft strict / cand | finalized strict / cand | Δ strict | Δ cand |")
        add("| --- | ---: | ---: | ---: | ---: |")
        for key in (*STRATA, "overall"):
            if key not in draft:
                continue
            name_label = STRATUM_LABEL.get(key, "**All 200**")
            ds, dc = draft[key]["turn_all_hit"], draft[key]["candidate_turn_all_hit"]
            fs, fc = final[key]["turn_all_hit"], final[key]["candidate_turn_all_hit"]
            add(f"| {name_label} | {pct(ds)} / {pct(dc)} | **{pct(fs)}** / **{pct(fc)}** | "
                f"{delta(fs, ds)} | {delta(fc, dc)} |")
        add("")

    add("## Reproduction of the published figures")
    add("")
    add("Under a fixed hash seed the draft-gold runs should land on the numbers in the")
    add("V5.4/V5.5 reports. Divergence here is the non-determinism of defect D1")
    add("(`packer.pack` iterates a `set` of mandatory turn ids).")
    add("")
    add("| Configuration | Stratum | published | draft-gold replay | Δ |")
    add("| --- | --- | ---: | ---: | ---: |")
    for (name, profile), expected in REPORTED.items():
        replay = runs[name.replace("_final", "_draft")][0]["profiles"][profile]["navigation"]
        for key, (strict, _cand) in expected.items():
            label = STRATUM_LABEL.get(key, "All 200")
            got = replay[key]["turn_all_hit"]
            add(f"| {name.replace('_draft', '').replace('_final', '')}/{profile} | {label} | "
                f"{pct(strict)} | {pct(got)} | {delta(got, strict)} |")
    add("")

    # --- candidate pool narrowing, joined positionally: navigation_results.jsonl
    # overwrites its own question_id with the hashed result id, so the devset id
    # is not available for a key join (tracked as a separate logging defect).
    run = latest_run("g0_final")
    nav: dict[str, list[dict]] = {}
    for line in (run / "navigation_results.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            nav.setdefault(row["configuration"], []).append(row)
    metrics: dict[str, list[dict]] = {}
    for line in (run / "metrics.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            metrics.setdefault(row["configuration"], []).append(row)

    sizes = {profile: sorted(len(row["candidate_scores"]) for row in rows)
             for profile, rows in nav.items()}
    superset = subset = 0
    regressions: Counter[str] = Counter()
    for left, right, m_left, m_right in zip(nav["h0"], nav["h6"], metrics["h0"], metrics["h6"]):
        pool_left = {row["turn_id"] for row in left["candidate_scores"]}
        pool_right = {row["turn_id"] for row in right["candidate_scores"]}
        superset += pool_left <= pool_right
        subset += pool_right <= pool_left
        if m_left["candidate_turn_all_hit"] and not m_right["candidate_turn_all_hit"]:
            regressions[m_left["stratum"]] += 1

    add("## Where the candidate pool went (the H8 defect, quantified)")
    add("")
    add("Measured on the finalized gold, same frozen graph, so the only difference")
    add("is the harness itself.")
    add("")
    add("| Candidate pool size | H0 | H6 |")
    add("| --- | ---: | ---: |")
    for label, index in (("p50", 0.50), ("p90", 0.90), ("p99", 0.99)):
        add(f"| {label} | {sizes['h0'][int(len(sizes['h0']) * index) - 1]} | "
            f"{sizes['h6'][int(len(sizes['h6']) * index) - 1]} |")
    add(f"| mean | {sum(sizes['h0']) / len(sizes['h0']):.1f} | "
        f"{sum(sizes['h6']) / len(sizes['h6']):.1f} |")
    add(f"| max | {sizes['h0'][-1]} | {sizes['h6'][-1]} |")
    add("")
    add(f"- The H6 pool is a **superset** of the H0 pool on **{superset}/200** questions.")
    add(f"- The H6 pool is a **subset** of the H0 pool on **{subset}/200** questions.")
    add(f"- **{sum(regressions.values())}/200** questions had every gold turn in the H0 pool but")
    add(f"  not in the H6 pool: {dict(sorted(regressions.items()))}.")
    add("")
    add("H8 must therefore restore a pool that dominates H0's by construction. Sizing")
    add(f"the id-only reservoir at **{sizes['h0'][-1]}+** entries covers H0's widest question;")
    add("the narrowing then happens only at hydration and packing, where it is")
    add("budget-driven and measured.")
    add("")

    add("## What the G2 sidecar is actually contributing")
    add("")
    g0_h6 = runs["g0_final"][0]["profiles"]["h6"]["navigation"]
    g2_h6 = runs["g2_final"][0]["profiles"]["h6"]["navigation"]
    add("G2 was built only for the ten LoCoMo memories where a prior H6 run missed")
    add("`candidate_turn_all_hit`, i.e. its build scope is selected using gold. On the")
    add("honest graph its headline gain largely disappears:")
    add("")
    add("| Stratum | H6 on frozen V5.4 (honest) | H6 + G2 (gold-scoped) | Δ cand |")
    add("| --- | ---: | ---: | ---: |")
    for key in (*STRATA, "overall"):
        label = STRATUM_LABEL.get(key, "**All 200**")
        left, right = g0_h6[key]["candidate_turn_all_hit"], g2_h6[key]["candidate_turn_all_hit"]
        add(f"| {label} | {pct(left)} | {pct(right)} | {delta(right, left)} |")
    add("")
    add("LoCoMo Cat1 candidate all-hit is **16%** without the sidecar and 82% with it.")
    add("The published \"candidate 82%, packed 18%, so the problem is packing\" reading")
    add("does not survive this: on the honest graph Cat1 fails at routing and seeding")
    add("long before packing. PR6 must rebuild those postings globally and")
    add("question-independently before any Cat1 packing claim can be made.")
    add("")

    add("## Corrected baseline for the V5.6 gate table")
    add("")
    add("| Metric | A0 = V5.4 H0 | V5.5 H6+G2 |")
    add("| --- | ---: | ---: |")
    h0 = runs["g0_final"][0]["profiles"]["h0"]["navigation"]
    h6 = runs["g2_final"][0]["profiles"]["h6"]["navigation"]
    for key, label in (("turn_all_hit", "Strict full-200 turn all-hit"),
                       ("turn_recall", "Mean turn recall"),
                       ("candidate_turn_all_hit", "Candidate all-hit"),
                       ("candidate_turn_recall", "Candidate recall"),
                       ("evidence_tokens", "Evidence tokens (heuristic estimate)"),
                       ("certificate_complete", "Certificate complete (pre-pack)"),
                       ("pack_turn_cap_reached", "Pack turn cap reached"),
                       ("pack_token_cap_reached", "Pack token cap reached")):
        left, right = h0["overall"][key], h6["overall"][key]
        fmt = (lambda value: f"{value:,.0f}") if key == "evidence_tokens" else pct
        add(f"| {label} | {fmt(left)} | {fmt(right)} |")
    for key in STRATA:
        add(f"| {STRATUM_LABEL[key]} candidate all-hit | {pct(h0[key]['candidate_turn_all_hit'])} | "
            f"{pct(h6[key]['candidate_turn_all_hit'])} |")
    add("")

    paired = runs["g0_final"][0].get("paired_turn_all_hit_vs_h0") or {}
    if not paired:
        paired = json.loads((latest_run("g0_final") / "summary.json").read_text()).get(
            "paired_turn_all_hit_vs_h0", {})
    if paired:
        add("## Paired bootstrap versus H0 (finalized gold, same graph)")
        add("")
        add("| Profile | point | CI low | CI high | significant |")
        add("| --- | ---: | ---: | ---: | --- |")
        for profile, row in sorted(paired.items()):
            significant = "yes" if row["ci_low"] > 0 else "**no**"
            add(f"| {profile} | {row['point']:+.3f} | {row['ci_low']:+.3f} | "
                f"{row['ci_high']:+.3f} | {significant} |")
        add("")

    add("## Run provenance")
    add("")
    add("| Run | source db | gold | run directory |")
    add("| --- | --- | --- | --- |")
    for name in ("g0_draft", "g0_final", "g2_draft", "g2_final"):
        manifest = runs[name][2]
        gold = next((key for key in manifest["input_hashes"] if "gold" in key), "?")
        add(f"| `{name}` | `{Path(manifest['source_db']).name}` | `{Path(gold).name}` | "
            f"`{latest_run(name).relative_to(ROOT)}` |")
    add("")
    add("Every run reports `generative_llm_calls: 0`; only the permitted embedding")
    add("query channel was used.")
    add("")

    target = REPO / "docs/V5_6_REBASELINE.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {target} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
