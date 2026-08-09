# GraphMem V5.8 Ablation Visual Report — Design Contract

## Audience and purpose

- Audience: technical researchers and GraphMem engineers.
- Decision: which ablation arms deserve the next compute and judge budget.
- Primary surface: self-contained portable HTML report.

## Visual system

- Single-column report reading path; all charts use full width.
- Palette policy: single blue root for single-series rankings; blue/orange for
  legacy-versus-harness comparisons. Neutral tones carry baselines and notes.
- Charts use direct category labels, zero-based scales for absolute rates, and
  explicit percent or millisecond units.
- Method names describe the mechanism; experiment IDs appear only as secondary
  aliases in tables.

## Chart map

| Section | Question | Family | Fields | Claim |
|---|---|---|---|---|
| Build extraction | Which extraction feature changes recall? | Horizontal bar | method, all_hit | Quoted evidence is the only strong positive single factor. |
| Retrieval ladder | Which retrieval mechanism wins overall? | Horizontal bar | method, all_hit | Legacy N5 beats every harness configuration overall. |
| Retrieval latency | What is the runtime cost? | Horizontal bar | method, p50_ms | Harness stages add substantial latency. |
| Evidence budget | Does packing more turns help? | Bar | budget_turns, all_hit | H10 improves monotonically with turn budget. |
| Category trade-off | Where does harness help? | Grouped bar | stratum, method, all_hit | H10 helps Cat1 but loses on Cat4. |

## QA

- Quantitative claims are backed by the V5.8 parameter-space document and the
  saved Phase A/Phase B JSON artifacts.
- LoCoMo dominates the 761-question dataset; LongMemEval has only one sample.
- Retrieval all-hit is not answer/judge accuracy.
- Nine-way concurrent latency is appropriate for relative ranking, not capacity
  planning.
