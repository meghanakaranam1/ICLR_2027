# SE vs Majority Voting — Exact Agreement Analysis

Descriptive analysis of how often successive elimination (SE) and majority voting (MV) produce the exact same final label on examples **both methods resolve** (jointly resolved: neither escalates, self-refuses, is prompt-injected, nor parse-fails). Read-only over existing `raw.csv` outputs; no LLM calls, no experiment file modified, no reruns.

## Methodology

- **Jointly resolved**: an example is included only if both MV's `graph_mv` and SE's `graph_ucb_B{budget}` produced a real task label for it (not one of `escalate`/`self_refused`/`prompt_injected`/`parse_failed`).
- **Exact agreement**: MV's `final_label` equals SE's `final_label` on that example (a stricter criterion than "both correct" — two methods can both be wrong and still agree, or both be right and still agree; this row is agreement on the *decision*, not on correctness).
- **McNemar's exact test**: binomial test on the discordant pairs only (`b` = MV-correct/SE-incorrect, `c` = SE-correct/MV-incorrect), identical construction to the paper's existing Section 6.1/6.2 tables and to `riskCoverageComparisons.csv`.
- **95% CI on paired accuracy difference** (SE − MV): Wald-type CI for a difference of matched/paired binary proportions, built directly from the discordant-pair counts: `diff = (c - b) / n`, `Var(diff) = [n(b+c) - (c-b)^2] / n^3`, `CI = diff ± 1.96*sqrt(Var(diff))`. This is the discordant-pair formulation the task instructions permit when no dedicated paired-proportion routine already exists in the codebase — it is not a new statistical method invented for this analysis, just the standard McNemar-adjacent variance estimator applied directly.
- **Non-significance ≠ equivalence**: a McNemar p-value of 1.0 means the data give no evidence of a difference; it does **not** prove SE and MV are equivalent, especially at the small discordant-pair counts seen here (many rows have 0–3 discordant pairs, so power to detect a real but modest difference is low).

## Full 15-row table

| Model | Dataset | Budget | N (jointly resolved) | Agree | Disagree | Agreement rate | Both correct | Both incorrect | MV-only correct | SE-only correct | Paired acc. diff (SE-MV) | 95% CI | McNemar p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3.1_8b | hatemoderate | B75 | 125 | 124 | 1 | 0.992 | 95 | 29 | 1 | 0 | -0.008 | [-0.024, +0.008] | 1.0000 |
| llama3.1_8b | hatemoderate | B100 | 195 | 193 | 2 | 0.990 | 144 | 49 | 2 | 0 | -0.010 | [-0.024, +0.004] | 0.5000 |
| llama3.1_8b | hatemoderate | B124 | 233 | 231 | 2 | 0.991 | 165 | 66 | 2 | 0 | -0.009 | [-0.020, +0.003] | 0.5000 |
| llama3.1_8b | aegis | B75 | 103 | 103 | 0 | 1.000 | 95 | 8 | 0 | 0 | +0.000 | [+0.000, +0.000] | 1.0000 |
| llama3.1_8b | aegis | B100 | 177 | 177 | 0 | 1.000 | 164 | 13 | 0 | 0 | +0.000 | [+0.000, +0.000] | 1.0000 |
| llama3.1_8b | aegis | B124 | 199 | 199 | 0 | 1.000 | 182 | 17 | 0 | 0 | +0.000 | [+0.000, +0.000] | 1.0000 |
| qwen2.5_7b | hatemoderate | B75 | 250 | 250 | 0 | 1.000 | 151 | 99 | 0 | 0 | +0.000 | [+0.000, +0.000] | 1.0000 |
| qwen2.5_7b | hatemoderate | B100 | 284 | 284 | 0 | 1.000 | 170 | 114 | 0 | 0 | +0.000 | [+0.000, +0.000] | 1.0000 |
| qwen2.5_7b | hatemoderate | B124 | 297 | 296 | 1 | 0.997 | 177 | 119 | 1 | 0 | -0.003 | [-0.010, +0.003] | 1.0000 |
| qwen2.5_7b | aegis | B75 | 250 | 249 | 1 | 0.996 | 221 | 28 | 1 | 0 | -0.004 | [-0.012, +0.004] | 1.0000 |
| qwen2.5_7b | aegis | B100 | 263 | 261 | 2 | 0.992 | 232 | 29 | 1 | 1 | +0.000 | [-0.011, +0.011] | 1.0000 |
| qwen2.5_7b | aegis | B124 | 271 | 269 | 2 | 0.993 | 240 | 29 | 1 | 1 | +0.000 | [-0.010, +0.010] | 1.0000 |
| mistral_7b | hatemoderate | B75 | 183 | 183 | 0 | 1.000 | 100 | 83 | 0 | 0 | +0.000 | [+0.000, +0.000] | 1.0000 |
| mistral_7b | hatemoderate | B100 | 234 | 234 | 0 | 1.000 | 128 | 106 | 0 | 0 | +0.000 | [+0.000, +0.000] | 1.0000 |
| mistral_7b | hatemoderate | B124 | 246 | 246 | 0 | 1.000 | 136 | 110 | 0 | 0 | +0.000 | [+0.000, +0.000] | 1.0000 |

## Pooled descriptive statistics

**Important**: the 15 rows are NOT independent statistical replicates — the same underlying 400 examples per pair reappear at all three budgets, so pooling across rows pools example-*comparisons*, not unique examples. The pooled numbers below are purely descriptive (a weighted overall rate), not a combined hypothesis test.

- Total example-comparisons pooled across all 15 rows: **3310**
- For reference, unique examples per pair (counted once, budget-independent): 400 × 5 pairs = **2000**
- Total agree: **3299**, total disagree: **11**
- Pooled exact agreement rate: **0.9967**
- Both correct: 2400, both incorrect: 899
- MV-correct/SE-incorrect: 9, SE-correct/MV-incorrect: 2
- Total discordant correctness pairs: **11** (out of 3310 comparisons)
- Agreement rate range across the 15 rows: [0.990, 1.000]
- Paired accuracy difference range across the 15 rows: [-0.010, +0.000]

## Interpretation

SE and MV make the exact same decision on the large majority of jointly-resolved examples (pooled agreement rate 99.7%), and the total number of examples where they disagree on correctness (11 out of 3310 comparisons) is small relative to the shared/agreeing population. This supports the descriptive claim that whatever correctness differences exist between SE and MV are concentrated in a small number of discordant examples, not spread broadly across the jointly-resolved set.

This is **not** the same as concluding SE and MV are statistically equivalent. Every McNemar p-value in the 15-row table is non-significant, but several rows have only 0-3 discordant pairs — at that count, the test has very little power to detect anything but a large effect, so "non-significant" here mainly reflects "too few discordant examples to say anything," not confirmed sameness. The 95% CIs on the paired accuracy difference are wide relative to the point estimates in most rows, which is the more honest way to see the same limitation: the data are compatible with a range of true differences, not pinned down to zero.

