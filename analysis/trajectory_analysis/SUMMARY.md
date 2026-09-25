# SE Trajectory-Feature Analysis — Summary

Read-only analysis over the existing SE B=75/100/124 raw.csv outputs
(5 pairs). No LLM calls, no changes to the SE algorithm, no changes to any
existing result file.

## Schema findings (read this before trusting any feature below)

- `arm_elimination_trace` records, per graph node, a list of
  `{round, active_before, eliminated, active_after}`. **Elimination is a
  single terminal event, never incremental**: across every row inspected,
  `eliminated` is empty at every round except (at most) the very last one,
  where the losing candidate(s) are cut all at once. There is **no
  per-round vote proportion or per-round leading-candidate stored
  anywhere**.
- Consequence: **`earlyVoteMargin`, `trajectoryStability`,
  `modalLabelChanges`, and "did the leader ever change after first
  becoming leader" are NOT reconstructable from the stored data.** They are
  not computed anywhere in this analysis, per instruction, rather than
  approximated.
- `arm_estimates_json` / `final_confidence` / `leading_estimate` are
  populated **only for rows generated after the 2026-09-20 schema
  addition** (the N=100→N=400 extension rows); the original 100-row
  baseline in every file lacks them. `leading_candidate` itself has much
  broader coverage (~100% of resolved, ~97% of escalated) because it was
  backfilled separately in an earlier session. `finalVoteMargin` and
  `finalEntropy` (which require `arm_estimates_json`) are therefore
  computed on a **subset** of rows — exact n reported per row in the CSVs,
  never imputed.
- `numberOfSurvivingCandidates` is **exactly 1 for every resolved row, with
  zero exceptions** (confirmed directly) — it is a constant within the
  resolved population and cannot discriminate correct vs incorrect by
  construction. It is reported for Question B only, and even there it is
  close to tautological with the resolved/escalated label itself (see
  below).

## Reconstructed features (used below)

| Feature | Reconstructable? | Caveat |
|---|---|---|
| `finalConfidence` | Yes, subset (post-2026-09-20 rows) | |
| `totalPulls` | Yes, all rows | |
| `eliminationRound` | Yes, all rows | ≈ total rounds run for the terminating node; strongly collinear with `totalPulls` / budget-exhaustion, not an independent "convergence speed" signal |
| `finalVoteMargin` | Yes, subset | requires `arm_estimates_json` |
| `finalEntropy` | Yes, subset | requires `arm_estimates_json`; computed over the real-label distribution only |
| `numberOfEliminations` | Yes, all rows | confirmed always 0 or 1 — a binary indicator of "did this node's trace ever terminate", not a graded count |
| `numberOfSurvivingCandidates` | Yes, all rows | constant (=1) for all resolved rows — see above |

## Question A: Correct vs Incorrect (resolved examples only)

**Aggregate AUROC (pooled across all 5 pairs, all 3 budgets):**

| Feature | AUROC | Interpretation |
|---|---|---|
| finalEntropy | 0.506 | chance |
| eliminationRound | 0.499 | chance |
| totalPulls | 0.498 | chance |
| numberOfSurvivingCandidates | 0.500 | constant feature, undefined discrimination |
| finalConfidence | 0.494 | chance |
| finalVoteMargin | 0.494 | chance |
| numberOfEliminations | 0.494 | chance |

**Every single feature sits within ~1 point of exactly 0.5 (chance).** None
of the reconstructable trajectory statistics discriminate correct from
incorrect resolutions in the pooled data. This is an independent
confirmation, via a completely different metric (AUROC on trajectory
features, not a raw confidence-gap comparison), of the paper's existing
Section 6.3 finding.

**Per-pair heterogeneity:** the strongest per-pair result is
llama3.1:8b×HateModerate, finalConfidence AUROC=0.594 (n_correct=278,
n_incorrect=110, pooled over budgets). Broken down by budget, B100 and
B124 show nominally significant Mann-Whitney p-values (p=0.028, p=0.034),
but the effect size is small (rank-biserial ≈ −0.20) and the raw means
differ by only ~1 percentage point (0.985 vs 0.976 at B100). **Given 105
Mann-Whitney tests were run for Question A alone, 1–2 nominal p<0.05 hits
are exactly what chance predicts (~5%) — this is not treated as a real
finding.** No other pair reaches even this level.

## Question B: Resolved vs Escalated

**Aggregate AUROC:**

| Feature | AUROC | Interpretation |
|---|---|---|
| finalVoteMargin | 0.989 | very strong — **but see circularity note below** |
| numberOfEliminations | 0.781 | strong — **near-tautological, see below** |
| eliminationRound | 0.248 | strong (inverted: resolved rows finish in fewer rounds) |
| numberOfSurvivingCandidates | 0.172 | strong — **tautological by construction (always 1 iff resolved)** |
| totalPulls | 0.084 | very strong (inverted: resolved rows use far fewer pulls than escalated, which run to budget exhaustion) |
| finalEntropy | 0.019 | very strong (inverted) |
| finalConfidence | undefined | escalated rows never have a `final_confidence` value (0 negatives) — cannot be computed, correctly left blank rather than fabricated |

**Important circularity caveat**: several of these are not independent
"difficulty signals" — they are close to definitional restatements of how
SE decides to stop. A node resolves *because* its vote margin crossed the
elimination threshold; it escalates *because* it never did, which is also
exactly why it used more pulls (ran to budget) and shows a low margin/high
entropy. `numberOfSurvivingCandidates` (1 vs >1) and `numberOfEliminations`
(1 vs 0) are the most extreme cases of this — they are near-restatements of
the resolved/escalated label itself, not new information. `totalPulls`,
`eliminationRound`, `finalEntropy`, and `finalVoteMargin` are less
extreme but still substantially mechanical consequences of the algorithm's
own stopping rule, not evidence about the underlying *content* being
harder. **Treat these AUROCs as confirming the SE algorithm's stopping
rule behaves consistently with its own design, not as evidence that SE
"detects difficulty" in some deeper sense.**

## Question 9 — the key pattern

**Pattern A** ("trajectory features predict resolved-vs-escalated but not
correct-vs-incorrect") **is present in the aggregate data**: every feature
that discriminates resolution status strongly (AUROC far from 0.5) sits at
chance for correctness (AUROC ≈ 0.5). This holds consistently across all
five pairs for Question A (no pair shows a robust correctness signal after
accounting for multiple comparisons) — so unlike the risk-coverage
analysis (where llama showed a genuine, non-circular difficulty gradient
via resolved-set accuracy varying with coverage), this trajectory-feature
version of the same question does **not** provide independent
non-circular evidence for "SE identifies easier cases" — the Question-B
AUROCs available here are mostly circular restatements of the stopping
rule, as noted above. The one exception is `totalPulls`/`eliminationRound`
timing, which is at least a real, externally-meaningful quantity (how much
compute was spent) even if its relationship to resolution is expected by
construction.

**So: Pattern A holds, but the Question-B half of it should not be read as
a strong new confirmation of "SE detects difficulty" beyond what the
risk-coverage analysis already showed** — most of the available trajectory
features are too close to the algorithm's own stopping criterion to serve
as independent evidence.

## Answers to the 10 questions

1. **Which features distinguish correct vs incorrect?** None, reliably.
   All aggregate AUROCs are within ~1 point of chance (0.494–0.506).
2. **Which features distinguish resolved vs escalated?** All of them, but
   most (numberOfSurvivingCandidates, numberOfEliminations, finalVoteMargin)
   are circular with the stopping rule itself; totalPulls and
   eliminationRound are the least circular (they measure compute spent,
   not just a restated decision).
3. **Is final SE confidence useful for correctness?** No — AUROC 0.494
   aggregate; only chance-level per-pair variation.
4. **Is final SE confidence useful for predicting resolution?** Cannot be
   assessed directly (it's undefined/absent for every escalated row by
   construction — there is no negative class to compare against).
5. **Is elimination time useful?** For resolution status yes (mechanically
   expected); for correctness, no (AUROC 0.499).
6. **Is vote margin useful?** For resolution status yes, but largely
   circular (see above); for correctness, no (AUROC 0.494).
7. **Is entropy useful?** For resolution status yes (inverted, expected);
   for correctness, no (AUROC 0.506, the closest of any feature to exactly
   chance).
8. **Is trajectory stability useful?** Cannot be evaluated — not
   reconstructable from the stored data (no per-round leader is recorded).
9. **Does the answer differ across pairs?** For Question A, no — every
   pair is at or near chance; the one nominally-significant result
   (llama×HateModerate finalConfidence, some budgets) has a small effect
   size and is consistent with a chance finding under 105 comparisons. For
   Question B, the direction is consistent across all 5 pairs (resolved
   examples use fewer pulls, higher margin, lower entropy, everywhere).
10. **Is there evidence for "difficulty is detectable, correctness is
    not"?** Weak-to-moderate. The pattern technically holds, but the
    resolution-side evidence here is mostly circular with the algorithm's
    own stopping rule rather than an independent difficulty signal. The
    risk-coverage analysis (see `analysis/risk_coverage/SUMMARY.md`) is the
    stronger, non-circular evidence for this claim (it showed llama's
    resolved-set accuracy genuinely varies with coverage) — this
    trajectory analysis is best read as a **weaker, partially circular
    corroboration**, not a second independent confirmation.

## What this analysis does NOT claim

No causal claim. No claim of statistical significance beyond the one
llama×HateModerate nominal result, which is explicitly flagged as likely a
multiple-comparisons artifact, not a real effect. No claim that a feature
is useful merely because group means differ — every reported "useful"
result above is backed by an AUROC substantially off 0.5, not just a mean
difference. No feature was fabricated or imputed where the underlying data
was absent.

---

## Files created

- `analysis/seTrajectoryFeatures.csv` (6000 rows: 5 pairs × 3 budgets × 400 examples)
- `analysis/seCorrectVsIncorrect.csv` (105 rows: 5 pairs × 3 budgets × 7 features)
- `analysis/seResolvedVsEscalated.csv` (105 rows)
- `analysis/seAuroc.csv` (84 rows: 2 tasks × (5 pairs + 1 aggregate) × 7 features)
- `analysis/trajectory_analysis/computeTrajectoryFeatures.py`
- `analysis/trajectory_analysis/figures/figure1_correct_vs_incorrect.png|svg`
- `analysis/trajectory_analysis/figures/figure2_resolved_vs_escalated.png|svg`
- `analysis/trajectory_analysis/figures/figure3_auroc_comparison.png|svg`
- `analysis/trajectory_analysis/SUMMARY.md` (this file)

## Commands to reproduce

```bash
cd /Users/meghanakarnam/Desktop/MLHC/ICLR_2027
python3 analysis/trajectory_analysis/computeTrajectoryFeatures.py
```

No LLM calls; reads only existing `raw.csv` files under `results_02/` and
`results_02_n400_from_spark/`.

## Parsing problems encountered

One escalated row (during initial inspection, not in the final run — the
script guards for this) had an empty `arm_elimination_trace` string,
consistent with `parse_failed` rows never running the elimination loop at
all. Handled via `safeJson()` returning `None` rather than crashing; such
rows simply contribute `None` for trace-derived features, not a fabricated
value.

## Features that could not be reliably reconstructed

`earlyVoteMargin`, `trajectoryStability`, `modalLabelChanges`, and
"whether the leading candidate changed after first becoming leader" — none
of these exist in the stored trace, which records only the final
elimination event, never per-round intermediate state.

## Five most important numerical findings

1. **Every trajectory feature's aggregate AUROC for correctness is within
   1.2 points of exactly chance (0.494–0.506)** — no feature discriminates
   correct from incorrect resolutions.
2. **`finalVoteMargin`'s AUROC for resolved-vs-escalated is 0.989** — but
   this is near-circular with the algorithm's own stopping rule, not
   independent evidence.
3. **`totalPulls`' AUROC for resolved-vs-escalated is 0.084** (i.e.
   resolved examples use far fewer pulls than escalated ones, which run to
   budget) — the least circular of the strong Question-B results.
4. **`numberOfSurvivingCandidates` is exactly 1 for 100% of resolved rows,
   zero exceptions** — confirmed constant, not usable for Question A by
   construction.
5. **The one nominally-significant correctness result (llama×HateModerate,
   finalConfidence, p=0.028–0.034 at B100/B124) has a small effect size
   (rank-biserial ≈ −0.20) and is not distinguishable from a chance hit
   given 105 total comparisons for Question A** — not reported as a real
   finding.
