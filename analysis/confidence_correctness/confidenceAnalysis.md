# Confidence vs Correctness

## 1. Method

Uses only the already-computed `analysis/seTrajectoryFeatures.csv` (itself built read-only from the existing SE `raw.csv` outputs). `finalConfidence` comes directly from the `final_confidence` column; `finalVoteMargin` and `finalEntropy` are derived from `arm_estimates_json` in that earlier pass, not recomputed here. **Schema note**: these three fields are populated for only 2463 of 3319 resolved examples (74.2%) — the pre-2026-09-20 baseline rows in every file lack them and are excluded from every table below, never imputed. AUROC uses a rank-based (Mann-Whitney U) estimator; AUPRC uses average precision with the 'correct' class as positive. No LLM calls were made and no existing file was modified to produce this analysis.

## 2. Discrimination

Per-condition AUROC/AUPRC for predicting correctness among resolved examples:

| Model | Dataset | Budget | Feature | N usable | N correct | N incorrect | AUROC | AUPRC | Low-n flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3.1_8b | hatemoderate | B75 | finalConfidence | 85 | 64 | 21 | 0.500 | 1.000 |  |
| llama3.1_8b | hatemoderate | B75 | finalVoteMargin | 85 | 64 | 21 | 0.500 | 1.000 |  |
| llama3.1_8b | hatemoderate | B75 | finalEntropy | 85 | 64 | 21 | 0.500 | 1.000 |  |
| llama3.1_8b | hatemoderate | B100 | finalConfidence | 138 | 100 | 38 | 0.603 | 0.930 |  |
| llama3.1_8b | hatemoderate | B100 | finalVoteMargin | 138 | 100 | 38 | 0.606 | 0.930 |  |
| llama3.1_8b | hatemoderate | B100 | finalEntropy | 138 | 100 | 38 | 0.394 | 0.733 |  |
| llama3.1_8b | hatemoderate | B124 | finalConfidence | 165 | 114 | 51 | 0.598 | 0.879 |  |
| llama3.1_8b | hatemoderate | B124 | finalVoteMargin | 165 | 114 | 51 | 0.586 | 0.872 |  |
| llama3.1_8b | hatemoderate | B124 | finalEntropy | 165 | 114 | 51 | 0.419 | 0.709 |  |
| llama3.1_8b | aegis | B75 | finalConfidence | 72 | 67 | 5 | 0.500 | 1.000 | YES |
| llama3.1_8b | aegis | B75 | finalVoteMargin | 72 | 67 | 5 | 0.500 | 1.000 | YES |
| llama3.1_8b | aegis | B75 | finalEntropy | 72 | 67 | 5 | 0.500 | 1.000 | YES |
| llama3.1_8b | aegis | B100 | finalConfidence | 130 | 121 | 9 | 0.506 | 0.972 | YES |
| llama3.1_8b | aegis | B100 | finalVoteMargin | 130 | 121 | 9 | 0.506 | 0.972 | YES |
| llama3.1_8b | aegis | B100 | finalEntropy | 130 | 121 | 9 | 0.494 | 0.955 | YES |
| llama3.1_8b | aegis | B124 | finalConfidence | 149 | 137 | 12 | 0.506 | 0.964 | YES |
| llama3.1_8b | aegis | B124 | finalVoteMargin | 149 | 137 | 12 | 0.506 | 0.964 | YES |
| llama3.1_8b | aegis | B124 | finalEntropy | 149 | 137 | 12 | 0.494 | 0.940 | YES |
| qwen2.5_7b | hatemoderate | B75 | finalConfidence | 184 | 110 | 74 | 0.500 | 1.000 |  |
| qwen2.5_7b | hatemoderate | B75 | finalVoteMargin | 184 | 110 | 74 | 0.500 | 1.000 |  |
| qwen2.5_7b | hatemoderate | B75 | finalEntropy | 184 | 110 | 74 | 0.500 | 1.000 |  |
| qwen2.5_7b | hatemoderate | B100 | finalConfidence | 208 | 124 | 84 | 0.478 | 0.954 |  |
| qwen2.5_7b | hatemoderate | B100 | finalVoteMargin | 208 | 124 | 84 | 0.475 | 0.951 |  |
| qwen2.5_7b | hatemoderate | B100 | finalEntropy | 208 | 124 | 84 | 0.525 | 0.880 |  |
| qwen2.5_7b | hatemoderate | B124 | finalConfidence | 219 | 130 | 89 | 0.495 | 0.928 |  |
| qwen2.5_7b | hatemoderate | B124 | finalVoteMargin | 219 | 130 | 89 | 0.495 | 0.928 |  |
| qwen2.5_7b | hatemoderate | B124 | finalEntropy | 219 | 130 | 89 | 0.505 | 0.778 |  |
| qwen2.5_7b | aegis | B75 | finalConfidence | 186 | 165 | 21 | 0.500 | 1.000 |  |
| qwen2.5_7b | aegis | B75 | finalVoteMargin | 186 | 165 | 21 | 0.500 | 1.000 |  |
| qwen2.5_7b | aegis | B75 | finalEntropy | 186 | 165 | 21 | 0.500 | 1.000 |  |
| qwen2.5_7b | aegis | B100 | finalConfidence | 194 | 173 | 21 | 0.492 | 0.993 |  |
| qwen2.5_7b | aegis | B100 | finalVoteMargin | 194 | 173 | 21 | 0.515 | 0.993 |  |
| qwen2.5_7b | aegis | B100 | finalEntropy | 194 | 173 | 21 | 0.483 | 0.960 |  |
| qwen2.5_7b | aegis | B124 | finalConfidence | 200 | 179 | 21 | 0.430 | 0.984 |  |
| qwen2.5_7b | aegis | B124 | finalVoteMargin | 200 | 179 | 21 | 0.430 | 0.984 |  |
| qwen2.5_7b | aegis | B124 | finalEntropy | 200 | 179 | 21 | 0.570 | 1.000 |  |
| mistral_7b | hatemoderate | B75 | finalConfidence | 151 | 82 | 69 | 0.500 | 1.000 |  |
| mistral_7b | hatemoderate | B75 | finalVoteMargin | 151 | 82 | 69 | 0.500 | 1.000 |  |
| mistral_7b | hatemoderate | B75 | finalEntropy | 151 | 82 | 69 | 0.500 | 1.000 |  |
| mistral_7b | hatemoderate | B100 | finalConfidence | 186 | 100 | 86 | 0.500 | 0.941 |  |
| mistral_7b | hatemoderate | B100 | finalVoteMargin | 186 | 100 | 86 | 0.500 | 0.941 |  |
| mistral_7b | hatemoderate | B100 | finalEntropy | 186 | 100 | 86 | 0.500 | 0.771 |  |
| mistral_7b | hatemoderate | B124 | finalConfidence | 196 | 107 | 89 | 0.470 | 0.906 |  |
| mistral_7b | hatemoderate | B124 | finalVoteMargin | 196 | 107 | 89 | 0.470 | 0.906 |  |
| mistral_7b | hatemoderate | B124 | finalEntropy | 196 | 107 | 89 | 0.530 | 0.797 |  |

**Pooled descriptive summary** (across all 15 conditions × 3 features = 45 rows; conditions are NOT independent — the same underlying examples reappear across budgets within a pair, so this is descriptive only, not a combined test):

- `finalConfidence`: AUROC range [0.430, 0.603], mean 0.505 (n=15 conditions)
- `finalVoteMargin`: AUROC range [0.430, 0.606], mean 0.506 (n=15 conditions)
- `finalEntropy`: AUROC range [0.394, 0.570], mean 0.494 (n=15 conditions)

**AUPRC baseline caveat**: AUPRC's uninformative baseline equals the positive-class prevalence (the fraction of resolved examples that are correct), not 0.5. Since accuracy on resolved examples is high in most conditions, a high AUPRC here is expected even under no real discrimination and should not be read as strong evidence on its own — AUROC (baseline 0.5 regardless of prevalence) is the more informative of the two metrics in this table.


No AUROC in this table is treated as "proof of no relationship" — values near 0.5 are reported as **no useful discrimination observed in these data**, which is a weaker and more accurate claim than statistical independence. Any single condition with a higher AUROC is checked against sample size and the total number of tests run (45) before being read as anything beyond a candidate for chance variation — see Section 6.

## 3. Calibration

Confidence-bin breakdown (bins with zero observations omitted, not forced):

| Model | Dataset | Budget | Bin | N | Mean confidence | Empirical accuracy | Empirical error rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama3.1_8b | hatemoderate | B75 | [0.9,1.0] | 85 | 1.000 | 0.753 | 0.247 |
| llama3.1_8b | hatemoderate | B100 | [0.8,0.9) | 1 | 0.886 | 1.000 | 0.000 |
| llama3.1_8b | hatemoderate | B100 | [0.9,1.0] | 137 | 0.983 | 0.723 | 0.277 |
| llama3.1_8b | hatemoderate | B124 | [0.8,0.9) | 11 | 0.877 | 0.636 | 0.364 |
| llama3.1_8b | hatemoderate | B124 | [0.9,1.0] | 154 | 0.974 | 0.695 | 0.305 |
| llama3.1_8b | aegis | B75 | [0.9,1.0] | 72 | 1.000 | 0.931 | 0.069 |
| llama3.1_8b | aegis | B100 | [0.8,0.9) | 1 | 0.879 | 1.000 | 0.000 |
| llama3.1_8b | aegis | B100 | [0.9,1.0] | 129 | 0.979 | 0.930 | 0.070 |
| llama3.1_8b | aegis | B124 | [0.8,0.9) | 11 | 0.866 | 0.909 | 0.091 |
| llama3.1_8b | aegis | B124 | [0.9,1.0] | 138 | 0.976 | 0.920 | 0.080 |
| qwen2.5_7b | hatemoderate | B75 | [0.9,1.0] | 184 | 1.000 | 0.598 | 0.402 |
| qwen2.5_7b | hatemoderate | B100 | [0.9,1.0] | 208 | 0.995 | 0.596 | 0.404 |
| qwen2.5_7b | hatemoderate | B124 | [0.8,0.9) | 8 | 0.880 | 0.500 | 0.500 |
| qwen2.5_7b | hatemoderate | B124 | [0.9,1.0] | 211 | 0.993 | 0.597 | 0.403 |
| qwen2.5_7b | aegis | B75 | [0.9,1.0] | 186 | 1.000 | 0.887 | 0.113 |
| qwen2.5_7b | aegis | B100 | [0.9,1.0] | 194 | 0.997 | 0.892 | 0.108 |
| qwen2.5_7b | aegis | B124 | [0.8,0.9) | 4 | 0.890 | 1.000 | 0.000 |
| qwen2.5_7b | aegis | B124 | [0.9,1.0] | 196 | 0.995 | 0.893 | 0.107 |
| mistral_7b | hatemoderate | B75 | [0.9,1.0] | 151 | 1.000 | 0.543 | 0.457 |
| mistral_7b | hatemoderate | B100 | [0.9,1.0] | 186 | 0.994 | 0.538 | 0.462 |
| mistral_7b | hatemoderate | B124 | [0.8,0.9) | 3 | 0.881 | 1.000 | 0.000 |
| mistral_7b | hatemoderate | B124 | [0.9,1.0] | 193 | 0.991 | 0.539 | 0.461 |

## 4. Confidence-1.0 analysis

| Model | Dataset | Budget | N at confidence=1.0 | Correct | Incorrect | Accuracy | Error rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama3.1_8b | hatemoderate | B75 | 85 | 64 | 21 | 0.753 | 0.247 |
| llama3.1_8b | hatemoderate | B100 | 90 | 72 | 18 | 0.800 | 0.200 |
| llama3.1_8b | hatemoderate | B124 | 80 | 63 | 17 | 0.787 | 0.212 |
| llama3.1_8b | aegis | B75 | 72 | 67 | 5 | 0.931 | 0.069 |
| llama3.1_8b | aegis | B100 | 76 | 71 | 5 | 0.934 | 0.066 |
| llama3.1_8b | aegis | B124 | 80 | 74 | 6 | 0.925 | 0.075 |
| qwen2.5_7b | hatemoderate | B75 | 184 | 110 | 74 | 0.598 | 0.402 |
| qwen2.5_7b | hatemoderate | B100 | 188 | 110 | 78 | 0.585 | 0.415 |
| qwen2.5_7b | hatemoderate | B124 | 181 | 107 | 74 | 0.591 | 0.409 |
| qwen2.5_7b | aegis | B75 | 186 | 165 | 21 | 0.887 | 0.113 |
| qwen2.5_7b | aegis | B100 | 182 | 162 | 20 | 0.890 | 0.110 |
| qwen2.5_7b | aegis | B124 | 175 | 154 | 21 | 0.880 | 0.120 |
| mistral_7b | hatemoderate | B75 | 151 | 82 | 69 | 0.543 | 0.457 |
| mistral_7b | hatemoderate | B100 | 162 | 87 | 75 | 0.537 | 0.463 |
| mistral_7b | hatemoderate | B124 | 161 | 85 | 76 | 0.528 | 0.472 |

**580 incorrect examples occur at final_confidence == 1.0**, summed across all 15 conditions (exact per-condition counts in the table above). This directly shows that a confidence value of exactly 1.0 does not guarantee correctness in this data.

## 5. Budget behavior

| Model | Dataset | Budget | Fraction at confidence=1.0 | Accuracy at confidence=1.0 | AUROC (finalConfidence) |
| --- | --- | --- | --- | --- | --- |
| llama3.1_8b | hatemoderate | B75 | 1.000 | 0.753 | 0.500 |
| llama3.1_8b | hatemoderate | B100 | 0.652 | 0.800 | 0.603 |
| llama3.1_8b | hatemoderate | B124 | 0.485 | 0.787 | 0.598 |
| llama3.1_8b | aegis | B75 | 1.000 | 0.931 | 0.500 |
| llama3.1_8b | aegis | B100 | 0.585 | 0.934 | 0.506 |
| llama3.1_8b | aegis | B124 | 0.537 | 0.925 | 0.506 |
| qwen2.5_7b | hatemoderate | B75 | 1.000 | 0.598 | 0.500 |
| qwen2.5_7b | hatemoderate | B100 | 0.904 | 0.585 | 0.478 |
| qwen2.5_7b | hatemoderate | B124 | 0.826 | 0.591 | 0.495 |
| qwen2.5_7b | aegis | B75 | 1.000 | 0.887 | 0.500 |
| qwen2.5_7b | aegis | B100 | 0.938 | 0.890 | 0.492 |
| qwen2.5_7b | aegis | B124 | 0.875 | 0.880 | 0.430 |
| mistral_7b | hatemoderate | B75 | 1.000 | 0.543 | 0.500 |
| mistral_7b | hatemoderate | B100 | 0.871 | 0.537 | 0.500 |
| mistral_7b | hatemoderate | B124 | 0.821 | 0.528 | 0.470 |

No causal claim is made about the effect of increasing budget. The table above reports how these three quantities co-vary with budget within each pair; any trend is descriptive.

## 6. Interpretation

This analysis sits downstream of three earlier findings: (1) SE and MV agree on 99.67% of jointly-resolved examples (`seMvAgreement.md`); (2) SE changes coverage substantially relative to MV (`riskCoverageAnalysis.md`); (3) the resulting risk changes are heterogeneous across the 5 pairs, from clearly lower to unchanged to inverted. Given (1)-(3), the open question is specifically whether SE's own reported confidence carries information about correctness — independent of whether coverage or risk changes overall.

The data show: aggregate AUROC for `finalConfidence` sits within a narrow band close to chance across all 15 conditions (see pooled range in Section 2). A substantial fraction of resolved examples land at confidence exactly 1.0 in most conditions (Section 5), and **580 of these are wrong** (Section 4) — the model can be maximally, deterministically confident and still incorrect. This is consistent with the mechanism already documented in the paper's Section 6.3: SE's confidence reflects the model's own self-consistency across repeated samples (successive elimination converging to a single surviving candidate), not an estimate of whether that candidate matches ground truth. A model that is repeatedly, consistently wrong produces the same maximal confidence value as one that is repeatedly, consistently right.

## 7. What this does NOT establish

- AUROC values near 0.5 are **not** proof of statistical independence between confidence and correctness — they show no useful discrimination was observed in this data, at this sample size, which is a narrower claim.
- No claim of causality: confidence does not "cause" correctness or incorrectness, and budget does not "cause" any observed change in the confidence-1.0 fraction or accuracy.
- No new algorithm or confidence-correction method is proposed here — this is a diagnostic analysis only.
- The one-off higher-AUROC conditions noted with a low-sample flag in Section 2 are not treated as real effects; with 45 AUROC values computed, some variation is expected by chance alone, and no multiple-comparison correction was applied because none of the values were extreme enough to warrant one — this is stated explicitly rather than either running an ad hoc correction or silently ignoring the multiplicity.

