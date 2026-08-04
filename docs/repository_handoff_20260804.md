# GraphMem repository handoff and artifact index

Date: 2026-08-04

## Recommended continuation branch

Use `codex/graphmem-v419-qwen32b` for new development. It descends from the
V4.1 query-time implementation and contains the GPT-5.4-mini/Qwen3-32B runner,
sharding, checkpoint, token-audit, retrieval, operator, and test changes.

```bash
git clone https://github.com/Concyclics/GraphMem.git
cd GraphMem
git switch codex/graphmem-v419-qwen32b
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Populate `.env` locally. Never commit credentials.

## Version history preserved in Git

| Line | Main commit/branch | Purpose |
|---|---|---|
| Baseline | `main` | Original graph-first retrieval pipeline |
| V2 | `codex/graphmem-v2` | Hierarchical state graph and evidence ledger |
| V3/V3.6 | history of `codex/graphmem-v3` | Hypergraph, role graph, lossless-first navigation and locked evaluation |
| V4/V4.1 | `codex/graphmem-v419-qwen32b` | Unified hybrid topology plus online Graph Harness and dual-backbone runners |

Use `git log --graph --oneline --all` to inspect the full iteration chain.

## Key implementation entry points

- `src/graphmem_demo/hierarchical_v2.py` — V2 state graph and generic memory operations.
- `src/graphmem_demo/v3/` — V3 hypergraph/lossless navigation components.
- `src/graphmem_demo/v36/` — Role-frame graph, generic operators, retrieval and runtime.
- `src/graphmem_demo/v4/` — Unified hybrid graph topology.
- `src/graphmem_demo/v41/` — Online Query IR, sidecar retrieval, evidence certificates,
  typed expansion, planners, and answer policies.
- `src/graphmem_demo/pipeline.py` — shared orchestration and token accounting.
- `scripts/run_token_demo.py` — main benchmark CLI.
- `scripts/run_v419_gpt54_lme_full.sh` and
  `scripts/run_v419_qwen32b_unified_full.sh` — resumable full-run examples.

## Design and iteration reports

### Architecture and plans

- `docs/2026-07-08_graphmem_architecture_overview.md`
- `docs/graphmem_v2.md`
- `docs/v3_6_single_version_redesign_20260728.md`
- `docs/graphmem_v4_0.md`
- `docs/V4_1_QUERY.md`

### V2 and V3 experiments

- `docs/graphmem_v2_experiment_report_20260724.md`
- `docs/graphmem_v2_final_500_report_20260725.md`
- `docs/locomo_v2_full_report_20260725.md`
- `docs/graphmem_v3_subset_report_20260725.md`
- `docs/graphmem_v3_blind_report_20260725.md`
- `docs/graphmem_v3_unseen_report_20260725.md`
- `docs/graphmem_v3_validation_20260726.md`
- `docs/v3_5_full_error_analysis_20260728.md`
- `docs/v3_6_full_eval_report_20260730.md`
- `docs/v3_6_lock_report_20260730.md`
- `docs/v3_6_retrieval_answer_gate_report_20260730.md`
- `docs/graphmem_v3_7_frozen_scorecard_20260730.md`

### V4.1 and cross-system reports

- `docs/graphmem_v4_1_dual_backbone_report_20260804.md`
- `docs/graphmem_v2_gpt_mem0_comparison_report_20260804.md`
- `docs/graphmem_mem0_comprehensive_report_20260804.md`
- `docs/GraphMem_vs_Mem0_Report_20260804.pdf`
- `docs/GraphMem_vs_Mem0_Innovation_Deck_20260804.pdf`
- `docs/graph_harness_query_ir_speaker_notes_20260804.md`

## Reproducibility and data policy

The Git repository contains source code, tests, run scripts, configuration
examples, aggregate result tables, and reports. It intentionally does not contain:

- `.env` or API keys;
- benchmark source datasets under `data/`;
- provider caches, embeddings, model weights, and local virtual environments;
- the local `runs/` tree (approximately 291 GB), including duplicated indexes,
  full retrieved contexts, and provider call caches;
- the nested `.qwen-worktree`, which is another checked-out Git branch rather
  than a project directory.

These exclusions keep the public repository credential-safe and avoid
redistributing benchmark conversations. Aggregate accuracy, per-type scores,
token breakdowns, failure analyses, configuration decisions, and reproducible
commands are preserved in the reports above.

## Validation

```bash
python -m pytest test/ -q
```

The benchmark runners require local datasets, an OpenAI-compatible backbone,
and the configured embedding endpoint. Judge prompts and pinning behavior are
documented in the experiment reports and evaluation scripts.
