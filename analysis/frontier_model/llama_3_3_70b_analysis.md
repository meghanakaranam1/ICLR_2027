# Llama 3.3 70B — Empirical Analysis

Read-only. No inference run, no result files modified, no manuscript changes. Data sources: `analysis/frontier_model/llama_70b/completed.jsonl` (HopGPT-routed, primarily HateModerate + partial AEGIS) + `analysis/frontier_model/llama_70b_spark_local/completed.jsonl` (local Ollama on Spark, AEGIS gap-fill) — confirmed zero overlap, zero duplicates between the two sources.

## A. Data integrity

| Dataset | Method | N | Unique IDs | Duplicates |
|---|---|---|---|---|
| hatemoderate | single_call | 400 | 400 | 0 |
| hatemoderate | graph_mv | 400 | 400 | 0 |
| hatemoderate | graph_ucb_B75 | 400 | 400 | 0 |
| hatemoderate | graph_ucb_B100 | 400 | 400 | 0 |
| hatemoderate | graph_ucb_B124 | 400 | 400 | 0 |
| aegis | single_call | 400 | 400 | 0 |
| aegis | graph_mv | 400 | 400 | 0 |
| aegis | graph_ucb_B75 | 400 | 400 | 0 |
| aegis | graph_ucb_B100 | 400 | 400 | 0 |
| aegis | graph_ucb_B124 | 400 | 400 | 0 |

- **Total: 4000/4000 rows, 0 duplicates.** All 5 methods share the *identical* 400-example ID set within each dataset (verified directly, not assumed).
- **576 error rows** exist across both sources (units that never produced a completed row — mostly Haiku-style preamble/parse failures during the original HopGPT run, now fully backfilled by successful retries or the Spark-local run; the one specific example flagged in an earlier turn this session was confirmed resolved).
- **Schema is leaner than the primary 7-8B `raw.csv`**: only `model, dataset, exampleId, method, budget, groundTruth, finalLabel, totalPulls, elapsedSeconds, finalConfidence, leadingCandidate` are present. **Missing entirely**: `arm_elimination_trace`, `arm_estimates_json`, `input_tokens`/`output_tokens`, separate `self_refused`/`prompt_injected`/`injection_detected`/`parse_failed` boolean columns, `category`. This caps what can be reconstructed (see Section E).
- **Parse-failure handling differs from the primary pipeline**: the primary `run_condition` catches `LabelParseError` and marks a row `parse_failed` after 3 consecutive failures *within the completed CSV itself*. The frontier runner instead lets a `LabelParseError` (or any other exception) abort the whole unit and land in `errors.jsonl` — it never appears as a `parse_failed` row in `completed.jsonl`. A retried/resumed unit that eventually succeeds shows up as a normal resolved row with no trace of the earlier failure(s). This is a **real difference in what "coverage" means operationally** between the two pipelines (see Section H).

## B. Per-condition metrics

| Dataset | Method | N | Resolved | Coverage | Acc (resolved) | FPR | FNR | Escalation | Self-refusal | Injection | Mean pulls | Median pulls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hatemoderate | single_call | 400 | 388 | 0.970 | 0.969 | 0.190 | 0.012 | 0.030 | 0.000 | 0.000 | 1.0 | 1.0 |
| hatemoderate | graph_mv | 400 | 400 | 1.000 | 0.948 | 0.186 | 0.036 | 0.000 | 0.000 | 0.000 | 5.1 | 5.0 |
| hatemoderate | graph_ucb_B75 | 400 | 396 | 0.990 | 0.952 | 0.186 | 0.031 | 0.010 | 0.000 | 0.000 | 76.7 | 72.0 |
| hatemoderate | graph_ucb_B100 | 400 | 399 | 0.998 | 0.947 | 0.186 | 0.037 | 0.003 | 0.000 | 0.000 | 77.3 | 72.0 |
| hatemoderate | graph_ucb_B124 | 400 | 397 | 0.993 | 0.947 | 0.186 | 0.037 | 0.007 | 0.000 | 0.000 | 77.9 | 72.0 |
| aegis | single_call | 400 | 325 | 0.812 | 0.892 | 0.181 | 0.050 | 0.188 | 0.000 | 0.000 | 1.0 | 1.0 |
| aegis | graph_mv | 400 | 351 | 0.877 | 0.875 | 0.209 | 0.053 | 0.122 | 0.000 | 0.000 | 6.6 | 5.0 |
| aegis | graph_ucb_B75 | 400 | 337 | 0.843 | 0.887 | 0.187 | 0.049 | 0.158 | 0.000 | 0.000 | 103.7 | 72.0 |
| aegis | graph_ucb_B100 | 400 | 341 | 0.853 | 0.883 | 0.196 | 0.049 | 0.147 | 0.000 | 0.000 | 106.2 | 72.0 |
| aegis | graph_ucb_B124 | 400 | 343 | 0.858 | 0.878 | 0.201 | 0.054 | 0.142 | 0.000 | 0.000 | 108.6 | 72.0 |

**Zero self-refusal, zero prompt-injection anywhere** — a real, clean finding (Llama 3.3 70B never refused or was hijacked on either dataset), not an artifact of missing data (the `finalLabel` field itself would show these as distinct string values if they occurred, and it never does).

**FNR denominators are small for HateModerate**: only 43 `not_hate`-ground-truth... wait — FNR here means missed `hate` cases, denominator is the `hate`-ground-truth count. Given the severe class imbalance (357 hate / 43 not_hate, see Section H), **FPR's denominator is only 43 examples** — noisy, wide-uncertainty estimate, not a stable rate.

## C. MV vs single_call

| Dataset | N_joint | Agree | Disagree | single-only-correct | MV-only-correct | McNemar p |
|---|---|---|---|---|---|---|
| hatemoderate | 388 | 387 | 1 | 1 | 0 | 1.0000 |
| aegis | 325 | 325 | 0 | 0 | 0 | 1.0000 |

MV and single-call agree almost universally on jointly-resolved examples for this model — even more so than typically seen in the 7-8B models, though single_call resolves noticeably fewer AEGIS examples overall (325/400 vs MV's 351/400 coverage), consistent with single_call having no elimination/voting mechanism to recover from an initial escalate-leaning response.

## D. SE vs MV

| Dataset | Budget | N_joint | Agree | Disagree | MV-only-correct | SE-only-correct | Exact agreement | McNemar p |
|---|---|---|---|---|---|---|---|---|
| hatemoderate | B75 | 396 | 395 | 1 | 1 | 0 | 0.9975 | 1.0000 |
| hatemoderate | B100 | 399 | 396 | 3 | 2 | 1 | 0.9925 | 1.0000 |
| hatemoderate | B124 | 397 | 395 | 2 | 2 | 0 | 0.9950 | 0.5000 |
| aegis | B75 | 337 | 337 | 0 | 0 | 0 | 1.0000 | 1.0000 |
| aegis | B100 | 341 | 340 | 1 | 1 | 0 | 0.9971 | 1.0000 |
| aegis | B124 | 343 | 343 | 0 | 0 | 0 | 1.0000 | 1.0000 |

**Pooled SE-vs-MV agreement**: HateModerate 1186/1192 = **99.50%**; AEGIS 1020/1021 = **99.90%**. Both extremely close to the 7-8B pooled figure (99.67%, from `analysis/risk_coverage/seMvAgreement.csv`) — the same near-total agreement phenomenon holds at 70B scale.

## E. Confidence vs correctness

**Separation-margin AUROC is NOT reconstructible** for this dataset — `arm_elimination_trace`/`arm_estimates_json` are absent from this schema (see Section A). Only `finalConfidence` (p̂_final) itself is available, and only for `graph_mv`/SE rows (never for `single_call`, which doesn't populate it).

| Dataset | Method | N | AUROC(p̂) | Fraction at p̂=1.0 | Of those, incorrect |
|---|---|---|---|---|---|
| hatemoderate | graph_mv | 400 | 0.562 | 0.975 (390/400) | 4.6% (18/390) |
| hatemoderate | B75 | 396 | 0.500 | 1.000 (396/396) | 4.8% (19/396) |
| hatemoderate | B100 | 399 | 0.546 | 0.992 (396/399) | 4.8% (19/396) |
| hatemoderate | B124 | 397 | 0.520 | 0.990 (393/397) | 5.1% (20/393) |
| aegis | graph_mv | 351 | 0.541 | 0.937 (329/351) | 11.6% (38/329) |
| aegis | B75 | 337 | 0.500 | 1.000 (337/337) | 11.3% (38/337) |
| aegis | B100 | 341 | 0.602 | 0.956 (326/341) | 9.5% (31/326) |
| aegis | B124 | 343 | 0.618 | 0.945 (324/343) | 9.6% (31/324) |

**Pooled AUROC: 0.557 (n=2964)** — still close to chance, but a real, measurable *step up* from the 7-8B pooled figure (0.494, from `analysis/confidence_correctness/confidenceAnalysis.md`). Not strong discrimination either way, but worth noting as a directional difference, not claiming it as meaningful calibration.

## F. Risk-coverage

Coverage/conditional-accuracy pairs, both datasets (from Section B):

- **HateModerate**: coverage ranges 0.970 (single_call) → 1.000 (MV) → 0.990-0.998 (SE, all three budgets), while conditional accuracy stays essentially flat (0.947-0.969) across every method. **Very little risk-coverage tradeoff visible** — extra compute (MV, SE) buys slightly *more* coverage without a corresponding accuracy cost, but the accuracy differences (0.947-0.969) are small enough that this reads as noise more than a clear trend.
- **AEGIS**: coverage ranges 0.812 (single_call) → 0.877 (MV) → 0.843-0.858 (SE), accuracy 0.875-0.892. Here there IS a mild coverage-for-accuracy tradeoff signal: single_call has the *lowest* coverage but *highest* accuracy (0.892) of any method, while MV has the *highest* coverage but *lowest* accuracy (0.875) — a small, real risk-coverage tradeoff, though the SE conditions sit in between without a clean monotonic budget trend (B75: 0.843/0.887, B100: 0.853/0.883, B124: 0.858/0.878 — coverage rises with budget while accuracy drifts slightly down, a genuine but modest pattern).
- **Compared to the 7-8B risk-coverage findings** (`analysis/risk_coverage/riskCoverageAnalysis.md`, which found *heterogeneous* per-pair behavior — some pairs showing a real tradeoff, others flat): the 70B AEGIS result (mild real tradeoff) and HateModerate result (flat) individually resemble different ends of the SAME heterogeneous pattern already documented across the 7-8B pairs — this is **consistent with**, not a departure from, the existing heterogeneity finding.

## G. Comparison with 7-8B results — does the central phenomenon reproduce?

1. **Does SE still identify the model's modal response?** Yes — SE and MV agree on the exact same final label 99.5-99.9% of jointly-resolved examples, the same pattern as 7-8B.
2. **Does high response concentration still fail to reliably predict correctness?** Yes — pooled AUROC 0.557, still near chance (though modestly higher than 7-8B's 0.494). 83-100% of resolved examples land at p̂=1.0 depending on condition, of which 4.6-11.6% are still wrong — the same qualitative "full confidence is not reliable" finding as 7-8B.
3. **Do SE and MV still agree strongly on jointly resolved examples?** Yes, if anything *more* strongly than 7-8B's pooled 99.67% (70B: 99.50-99.90%, comparable/slightly better).
4. **Does SE primarily change which examples get resolved vs escalated, more than which label is chosen?** Consistent with this — disagreement counts are tiny (0-3 per condition) relative to coverage differences across methods (up to 6.5 percentage points on AEGIS).
5. **Is risk-coverage behavior heterogeneous?** Yes — HateModerate shows almost no tradeoff, AEGIS shows a mild one, matching the already-documented cross-pair heterogeneity in the 7-8B results rather than contradicting it.
6. **Does scaling to 70B change the phenomenon substantially?** **No evidence of a substantial change** — every one of the above phenomena replicates directionally at 70B. The only measurable difference is a modestly higher (still weak) confidence-AUROC, which is a difference of degree, not of kind.

This is genuinely encouraging **descriptive, single-model cross-scale evidence** that the paper's central phenomenon is not an artifact specific to 7-8B models — but it is one model, one scale point, and should be framed exactly that narrowly.

## H. Protocol differences — flagged prominently

1. **Class balance differs sharply for HateModerate.** The 70B sample is **357 hate / 43 not_hate (89%/11%)** — nowhere close to the primary 7-8B experiments' balanced 200/200 split (via `balanced_sample_generic`, seeded). AEGIS is reasonably close (208 unsafe / 192 safe) but not identically constructed either. **This means HateModerate's FPR (denominator 43) is a small, noisy estimate, and none of the 70B accuracy/FPR/FNR numbers should be pooled or directly compared point-for-point against the primary 7-8B balanced numbers without explicitly disclosing this imbalance.**
2. **Two different serving backends for AEGIS.** ~842 units came through HopGPT (Bedrock-routed `llama3-3-70b-instruct`); ~1157 came through local Ollama on Spark (`llama3.3:70b`, a GGUF quantization). These are not guaranteed byte-identical serving stacks for the "same" model — flagged already in this session's earlier work, repeated here for completeness. HateModerate is 100% HopGPT-routed (no local-Ollama mixing).
3. **Parse-failure/coverage bookkeeping differs from the primary pipeline** (Section A) — the frontier runner's resumable retry-until-success design means a unit that failed multiple times before eventually succeeding shows up as a normal resolved row, with no record of the intermediate failures in `completed.jsonl` itself (only in the separate `errors.jsonl`). The primary pipeline's `parse_failed` classification (3-strikes-and-mark-it rule) has no direct analogue here — a `parse_failed` *outcome* essentially cannot occur in this schema, since failures either get retried to a real answer or land in `errors.jsonl`, never as a terminal "parse_failed" `finalLabel`. Confirmed empirically: zero rows anywhere in the 4000 have `finalLabel=="parse_failed"`.
4. **Same, confirmed unchanged**: temperature (`TEMPERATURE=0.7`, reused directly from `run_all_datasets.py`, never overridden anywhere in either frontier runner), node prompts (`build_hatemoderate_node_systems()`/`build_aegis_node_systems()`, called directly, unmodified), graph structure (`GRAPH = ["Screener","Analyst","Adjudicator"]`, same constant), candidate labels (`build_node_arms`, same function), SE implementation (`ucb_node`/`run_graph_ucb`, imported and called directly, never reimplemented for HopGPT; the Spark-local runner also reuses these unmodified), budget definitions (75/100/124, identical meaning — per-node cap, same `BUDGET_GRID`-derived values).

## I. What this result supports

- Descriptive, single-model evidence that the paper's central phenomena (SE≈MV agreement, weak confidence-correctness discrimination, heterogeneous risk-coverage behavior, SE mainly shifting resolution/escalation rather than label choice) replicate directionally at a materially larger open-weight model (70B vs 7-8B).
- A genuine, clean AEGIS completion story: 400/400 across all 5 methods, sourced from two consistent serving paths with zero duplication.
- A real (if modest) confidence-AUROC improvement at scale (0.557 vs 0.494) — worth reporting as an observation, not a strong claim.

## J. What this result does NOT support

- **Not** a claim that this reproduces the paper's exact 7-8B protocol closely enough to pool numerically — the HateModerate class-imbalance difference alone rules that out without explicit disclosure.
- **Not** evidence of "model scaling improves calibration" in any strong sense — the AUROC difference (0.494→0.557) is a single data point at a single scale, both still close to chance.
- **Not** generalizable evidence across frontier/larger models in general — this is one model (Llama 3.3 70B) via two serving backends, not a scaling sweep.
- **Not** evidence about self-refusal or prompt-injection behavior differences at scale, since both rates are exactly zero here — absence of these failure modes at 70B could reflect genuinely more robust behavior, or could reflect the specific serving stacks used; this analysis cannot distinguish those.

## Recommendation

**Better suited to appendix.**

The phenomenon-level replication (Section G) is genuinely valuable and worth including, but the HateModerate class-imbalance protocol mismatch (Section H.1) is serious enough that this cannot be pooled into or presented alongside the main 7-8B balanced-sample tables without extensive caveats. An appendix framing — "descriptive cross-scale check on the central phenomenon, single model, protocol differences disclosed" — captures the actual evidentiary weight honestly. It is not strong enough, on protocol-comparability grounds alone, to serve as the main-text scale/generalization claim as-is.
