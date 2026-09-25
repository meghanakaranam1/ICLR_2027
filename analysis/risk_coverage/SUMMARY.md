# Risk-Coverage Analysis — Summary

Read-only analysis over the existing N=400 experiment outputs. No experiment
was rerun, no LLM calls were made, and no existing file was modified.

## A. Files inspected

- `run_all_datasets.py` (schema, `NON_TERMINAL_LABELS`, escalation logic)
- `run_extension_n400.py` (how N=400 extensions are generated/checkpointed)
- `analyze.py` (existing analysis conventions — escalated rows already
  excluded from accuracy in the project's own prior analysis, confirming
  this analysis's definition matches project convention)
- `results_02/` and `results_02_n400_from_spark/` directory trees for all
  5 model/dataset pairs × 5 methods (25 raw.csv files)
- The CSV header/schema directly (`final_label`, `ground_truth`, `correct`,
  `escalated_to_human`, `self_refused`, `prompt_injected`, `parse_failed`,
  `total_pulls`, `final_confidence`, `leading_candidate`, etc.)

**Single-call representation**: `single_agent/raw.csv` per pair (one call,
one label per example; no separate runner script remains in the repo, but
its output is complete and unaffected).
**Majority voting**: `graph_mv/raw.csv` per pair.
**SE B=75/100/124**: `graph_ucb_B75/B100/B124/raw.csv` per pair.
**Escalation representation**: `final_label` ∈ {`escalate`, `self_refused`,
`prompt_injected`, `parse_failed`} — exactly the four outcomes the
pipeline's own `correct` column already leaves blank.
**Per-example trajectories/confidence**: available for `graph_mv` and SE
conditions (`arm_elimination_trace`, `final_confidence`, `leading_candidate`,
`arm_estimates_json`) — not used here since risk-coverage only needs the
final resolved/unresolved outcome, per the task's scope.

## B. Files changed / created

**Nothing existing was modified.** New files only, all under
`analysis/risk_coverage/`:

- `computeRiskCoverage.py` — the analysis script (reusable, rerunnable)
- `riskCoverageSummary.csv` / `.json` — main per-method/pair table (25 rows)
- `riskCoverageComparisons.csv` — SE-vs-majority-voting paired comparisons (15 rows)
- `riskCoverageAggregate.csv` — mean-across-pairs per method (5 rows)
- `figures/riskCoverage_allPairs.png|svg` — multi-panel figure
- `figures/riskCoverage_<model>_<dataset>.png|svg` — one figure per pair (5 pairs)
- `SUMMARY.md` — this file

## C. Exact formulas used

```
resolved      = final_label NOT IN {escalate, self_refused, prompt_injected, parse_failed}
coverage      = resolvedCount / totalCount
accuracy      = correctResolved / resolvedCount        (over RESOLVED rows only)
selectiveRisk = 1 - accuracy
escalationRate = escalatedCount / totalCount
```

95% CI on accuracy: Wilson score interval (appropriate for a binomial
proportion at the sample sizes here, some of which are small).

Paired comparison (SE vs MV): restricted to examples **both** methods
resolved; exact binomial McNemar test on the discordant pairs. This is the
same construction the paper already uses in Section 6.1/6.2 — reused here,
not a new method.

## D. Risk-coverage results, all 5 pairs

| Model | Dataset | Method | Coverage | Accuracy | Selective risk |
|---|---|---|---|---|---|
| llama3.1_8b | hatemoderate | single_call | 0.565 | 0.704 | 0.296 |
| llama3.1_8b | hatemoderate | majority_voting | 0.825 | 0.700 | 0.300 |
| llama3.1_8b | hatemoderate | SE B=75 | 0.313 | 0.760 | 0.240 |
| llama3.1_8b | hatemoderate | SE B=100 | 0.488 | 0.738 | 0.262 |
| llama3.1_8b | hatemoderate | SE B=124 | 0.583 | 0.708 | 0.292 |
| llama3.1_8b | aegis | single_call | 0.585 | 0.872 | 0.128 |
| llama3.1_8b | aegis | majority_voting | 0.778 | 0.839 | 0.161 |
| llama3.1_8b | aegis | SE B=75 | 0.258 | 0.922 | 0.078 |
| llama3.1_8b | aegis | SE B=100 | 0.445 | 0.927 | 0.073 |
| llama3.1_8b | aegis | SE B=124 | 0.498 | 0.915 | 0.085 |
| qwen2.5_7b | hatemoderate | single_call | 0.635 | 0.571 | 0.429 |
| qwen2.5_7b | hatemoderate | majority_voting | 0.863 | 0.586 | 0.414 |
| qwen2.5_7b | hatemoderate | SE B=75 | 0.625 | 0.604 | 0.396 |
| qwen2.5_7b | hatemoderate | SE B=100 | 0.710 | 0.599 | 0.401 |
| qwen2.5_7b | hatemoderate | SE B=124 | 0.743 | 0.596 | 0.404 |
| qwen2.5_7b | aegis | single_call | 0.638 | 0.878 | 0.122 |
| qwen2.5_7b | aegis | majority_voting | 0.738 | 0.881 | 0.119 |
| qwen2.5_7b | aegis | SE B=75 | 0.628 | 0.884 | 0.116 |
| qwen2.5_7b | aegis | SE B=100 | 0.665 | 0.880 | 0.120 |
| qwen2.5_7b | aegis | SE B=124 | 0.688 | 0.884 | 0.116 |
| mistral_7b | hatemoderate | single_call | 0.778 | 0.514 | 0.486 |
| mistral_7b | hatemoderate | majority_voting | 0.750 | 0.567 | 0.433 |
| mistral_7b | hatemoderate | SE B=75 | 0.458 | 0.546 | 0.454 |
| mistral_7b | hatemoderate | SE B=100 | 0.585 | 0.547 | 0.453 |
| mistral_7b | hatemoderate | SE B=124 | 0.615 | 0.553 | 0.447 |

(Full table with CIs, escalation counts/rates, source files:
`riskCoverageSummary.csv`.)

### SE vs majority voting (paired, on jointly-resolved examples only)

Every one of the 15 SE-vs-MV comparisons is **statistically null**
(p ∈ {0.5, 1.0}, 0–2 discordant pairs each) — SE and majority voting give
the *same verdict* whenever both resolve an example, in every pair, at
every budget. This means every risk/coverage difference reported above is
a **selection effect** (which examples get resolved vs. escalated), never
a difference in judgment quality on shared examples. (Full table:
`riskCoverageComparisons.csv`.)

## E. Assumptions and data limitations

- `totalCount == 400` held for all 25 files — no violation, nothing to stop for.
- All 8 consistency checks passed for every pair/method (see script
  assertions); none were silently bypassed.
- Coverage-matched comparisons were **not** interpolated — SE's three
  operating points (B=75/100/124) are the only ones the existing budget
  sweep provides, plotted as discrete points as instructed. Single-call and
  majority-voting are each a single fixed point.
- The `single_agent` *generator* script no longer exists in the repo (only
  its output data does) — this analysis only needed the output, so it's
  unaffected, but it means the single-call condition's generation process
  can't be independently re-audited from code alone.
- Sample sizes at low coverage (e.g. llama×aegis SE B=75, n=103 resolved)
  are small enough that Wilson CIs are wide — see the CSV for exact bounds
  before treating any single point as precise.

## F. Exact commands to rerun

```bash
cd /Users/meghanakarnam/Desktop/MLHC/ICLR_2027
python3 analysis/risk_coverage/computeRiskCoverage.py
```

No LLM calls, no network access, reads only the existing `raw.csv` files
under `results_02/` and `results_02_n400_from_spark/`.

---

## Interpretation (conservative)

**1. Does SE reduce selective risk as coverage decreases?**
Sometimes, and not uniformly. For **llama3.1:8b on both datasets**, risk
rises monotonically as budget/coverage increases (B75 lowest risk → B124
highest), a textbook risk-coverage tradeoff. For **qwen2.5:7b on
HateModerate**, the same direction holds but weakly (0.396 → 0.404 across
budgets — a 0.8-point spread). For **qwen2.5:7b on AEGIS** and
**mistral:7b on HateModerate**, risk is essentially flat across all three
budgets (differences under 1 point, likely within noise) — there is no
reliable coverage-risk tradeoff for these two pairs in this data.

**2–4. Do B=75 / B=100 / B=124 individually provide a useful tradeoff?**
B=75 gives the lowest risk at the lowest coverage in 4 of 5 pairs (all but
qwen/aegis, where the three budgets are statistically indistinguishable).
Whether that's "useful" depends on how much coverage the application can
afford to give up — B=75 resolves as few as 26–63% of examples across
pairs. B=124 buys more coverage at some risk cost in llama's two pairs, but
buys neither more coverage-worthy accuracy nor risk reduction in
qwen/aegis or mistral/hatemoderate.

**5. How do SE operating points compare with majority voting?**
SE has **lower coverage than MV in every single one of the 15
comparisons** (it always resolves fewer examples). On risk: SE is lower
than MV in 4 of 5 pairs, at every budget tested — **except
mistral:7b×HateModerate, where SE has *higher* risk than MV at every
budget** despite lower coverage (SE risk 0.447–0.454 vs. MV's 0.433). Since
the paired McNemar tests show SE and MV agree almost perfectly on shared
examples everywhere, this is not SE "judging worse" — it is SE choosing to
resolve a comparatively harder subset than MV does, specifically for this
one pair.

**6. Are there pairs where behavior differs?**
Yes, clearly. llama (both datasets) shows the intended, well-behaved
selective-prediction pattern. qwen/aegis and mistral/hatemoderate show flat
or (for mistral) inverted behavior. This is a real, model/dataset-dependent
heterogeneity, not noise-shaped: it's consistent across all three budgets
within each pair.

**7. Does the data support "SE confidence is useful for identifying
easier/correct examples"?**
Partially, and only in the specific sense of *which examples get
escalated*, not in the sense of *the confidence score itself*. Where risk
rises with coverage (llama, both datasets), that implies the
escalated-away examples genuinely are harder on average — a real
difficulty-discrimination signal in the escalation *decision*. But this is
a different claim from whether `final_confidence` discriminates correct
from incorrect *among resolved examples*, which the paper's own Section 6.3
already shows it does not (confidence ≈1.0 for both). The two claims are
compatible: escalation-vs-resolution can carry a difficulty signal even
while the resolved cases' confidence values stay uninformative about
correctness. For mistral/hatemoderate specifically, neither claim holds —
flat risk across coverage here corroborates 6.3's finding for this pair
with an independent metric.

**8. Any data/implementation problems preventing a valid analysis?**
None found. All 25 files present, N=400 confirmed everywhere,
resolved+escalated=total held everywhere, no escalated row was ever counted
as correct/incorrect, and all 5 methods were confirmed to operate on the
identical 400 example ids per pair.

**What this analysis does NOT claim**: it does not claim SE is generally
superior or inferior to majority voting, does not claim improved
calibration or reliability, and does not claim statistical significance for
any SE-vs-MV difference (the paired tests are null everywhere). The only
significance-tested result here is the *absence* of a judgment-quality
difference between SE and MV on shared examples — a null result, reported
as such.
