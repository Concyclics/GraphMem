#!/usr/bin/env bash
# V5.6 PR0.5 re-baseline: isolate the gold-annotation change (D0) from everything else.
# 2x2 = {draft gold, finalized gold} x {V5.4 authority (G0), V5.5 G2 sidecar}.
set -euo pipefail

AP=/home/chenhan/miniconda3/envs/agent/bin/python
REPO=/ssd3/chenhan/Spark_MemGraph_Dev/GraphMem
ART=/ssd3/chenhan/Spark_MemGraph_Dev/artifacts
DEV=$ART/development_sets/hard_lme_multisession_temporal_locomo_cat1_cat2_200_20260804
V54=$ART/v5_4/full200_resume/v5_1_graph_ablation_full_20260805T103058Z/graphmem.sqlite
G2=$ART/v5_5/cat1_g2_20260805/graphmem_g2.sqlite
GOLD_FINAL=$REPO/eval_annotations/longmemeval_v5_dev100_gold_turns.jsonl
GOLD_DRAFT=$ART/v5/lme_gold_turn_merged_draft_20260804.jsonl
OUT=$ART/v5_6/rebaseline

cd "$REPO"
mkdir -p "$OUT"

run () {  # name db gold profiles
  local name=$1 db=$2 gold=$3 profiles=$4
  echo "=== [$name] db=$(basename "$db") gold=$(basename "$gold") profiles=$profiles ==="
  PYTHONHASHSEED=0 PYTHONPATH=src "$AP" scripts/run_v5_5_retrieval_ablation.py \
    --source-db "$db" \
    --output-root "$OUT/$name" \
    --lme "$DEV/longmemeval_hard_multisession50_temporal50.json" \
    --locomo "$DEV/locomo_hard_cat1_multihop50_cat2_temporal50.json" \
    --gold "$gold" \
    --config configs/v5/v5_4_navigable.json \
    --profiles "$profiles" \
    --embedding
}

run g0_draft "$V54" "$GOLD_DRAFT" h0,h6
run g0_final "$V54" "$GOLD_FINAL" h0,h6
run g2_draft "$G2"  "$GOLD_DRAFT" h6
run g2_final "$G2"  "$GOLD_FINAL" h6

echo "ALL REBASELINE RUNS COMPLETE"
