# V5.9 report results audit

This file is the compact, source-of-truth handoff for the measurements rendered
into `../GraphMem_report`. Raw observations are under the ignored local path
`artifacts/report/v5_9/`; the report repository contains generated figures and a
JSON macro manifest.

## C1: recursive coarsening and parent-gated relations

The controlled scaling run covers 1K, 2K, 5K, 10K and 20K Session Cards.

- Fitted candidate exponent: All-pairs `2.0003`; CIR `0.9845`.
- At 20K: All-pairs has exactly `199,990,000` relation candidates; CIR has
  `108,820` (`99.95%` fewer).
- Under the fixed `2*96 input + 256 output` relation-decision contract, CIR uses
  `22,283,968` Token versus `55,720,000` for flat sparse construction (`60.0%`
  less).
- Controlled multi-hop path retention is `86.38%` for CIR and `75.0%` for the
  flat sparse baseline.

Candidate counts are exact. All-pairs wall time above 2K is explicitly projected
from the same materialized enumeration loop; it is not a claimed 20K execution.
Path retention is a controlled structural metric, not QA accuracy.

## C2/C3: QueryIR route, evidence completeness and certificate safety

The paired run contains 100 questions with turn-level gold. It preserves the
frozen V5.8 non-routing fact graph, replaces only parent Routing Cards with the
report recursive hierarchy, and makes no extractor or answer-model call. Of 100
memories, 71 contain Session Cards and are recoarsened; 29 source snapshots have
no Session Cards and remain unchanged. Dense retrieval is disabled in this run.

| Arm | all-hit | Turn recall | Gold Session route recall | p95 | false-complete |
|---|---:|---:|---:|---:|---:|
| flat@32 | 55.0% | 65.90% | 57.14% | 655.0 ms | 0.0% |
| fixed@32 | 57.0% | 67.28% | 45.71% | 606.0 ms | 0.0% |
| adaptive@32 | 57.0% | 66.82% | 53.81% | 630.9 ms | 0.0% |

The bounded lexical portal adds 8.1 percentage points of route recall over pure
top-down fixed beam while preserving an ancestor corridor. The hierarchy stage
itself averages 0.93 ms. At the smaller 16-turn budget, mean evidence decreases
from about 5.0K to 4.3K Token and all-hit decreases to 52.0%; this setting has not
met a non-inferiority criterion. These are retrieval/evidence results, not final
answer accuracy, and the +2 point all-hit difference is not claimed significant.

## System microbenchmark

The synthetic workload has 256 Sessions and four turns per Session; its graph
contains 2,374 nodes and 7,716 edges. The process query plane uses eight
persistent workers. At 32 concurrent clients:

- Single-process hierarchical: `17.09 QPS`, p95/p99
  `9,204.99/13,234.92 ms`.
- Eight-worker process-sharded hierarchical: `160.55 QPS`, p95/p99
  `245.32/255.65 ms` (`9.4x` throughput).
- Aggregate immutable Graph/Index snapshot cache across workers: `31.6 MiB`.
  This is not total Python runtime RSS.

For five paired publications, full-snapshot p95 is `527.35 ms` and touches
11,114 rows; affected-path p95 is `4.52 ms` and touches four rows (`116.6x`,
`99.1%` lower). The concurrent publication probe records 64,793 reader
operations, zero reader errors and an immutable old read view. Cold construction
of the first new in-memory snapshot is recorded separately and is not included
in authority commit latency.

## Full end-to-end benchmark

The frozen end-to-end run recoarsens all 510 benchmark memories from the V5.8
P8 fact projection, then executes the current H10 QueryIR, recursive hierarchy,
AST algebra, post-pack certificate and answer stage. Dense retrieval is disabled.
The answer and judge backbone is local
`Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`, with temperature zero and reasoning
disabled. LongMemEval uses the pinned Mem0 judge prompt; LoCoMo uses the pinned
memory-benchmarks prompt and excludes Category 5.

| Benchmark | Questions | Accuracy | Annotated all-hit | Prompt Token mean/p95 | Retrieval p95 |
|---|---:|---:|---:|---:|---:|
| LongMemEval | 500 | 72.60% | 56.00% (n=100) | 6,169 / 6,442 | 3,596.0 ms |
| LoCoMo Cat1--4 | 1,540 | 81.95% | 58.06% (n=1,533) | 2,637 / 2,918 | 1,989.3 ms |

No question exceeded the 10K answer-prompt soft budget. Closed-form execution
was used by 13.4% of LongMemEval and 14.2% of LoCoMo questions. LongMemEval
Multi-session/Temporal/Knowledge-update accuracy is 53.38%/70.68%/73.08%; LoCoMo
Multi-hop/Temporal/Open-domain/Single-hop is 78.01%/78.19%/77.08%/85.26%.
The separate format-sensitive LoCoMo official token-F1 is 22.41%.

Against the frozen V5.8 rank-mandatory run on the same questions and judge,
LongMemEval changes from 347/500 to 363/500 (+3.20 pp; new-only/old-only 48/32;
exact McNemar p=0.0929). LoCoMo changes from 1245/1540 to 1262/1540 (+1.10 pp;
112/95; p=0.2661). Both point estimates improve, but neither difference is
significant at 0.05. The closed-form subset is not a causal bypass ablation:
on those harder questions its paired net is +2 LongMemEval and -9 LoCoMo.

The compact reproducibility bundle is in
`artifacts/report/v5_9/full_benchmark/`; raw answers, retrieval rows and judge
calls remain under the ignored workspace artifact root.

## Still pending

- Fixed-index per-module answer ablations, fair long-context/vector/flat-graph
  baselines and an independent judge/backbone replication.
- New-Session insertion, split/merge, global relation maintenance and ANN delta
  merge in the synchronous update path.
- Admission control, worker restart, cross-node replication and explicit
  RPO/RTO fault-injection experiments.
