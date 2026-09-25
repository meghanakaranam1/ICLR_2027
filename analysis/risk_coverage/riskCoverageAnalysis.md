# Risk vs. Coverage — Main Analysis

Does SE's reduced coverage correspond to lower selective risk among the examples it resolves? Read-only analysis over the existing `riskCoverageSummary.csv` / `riskCoverageComparisons.csv` outputs. No new data, no LLM calls, no manuscript or experiment file modified.

## 1. Method

For each of the 5 model×dataset pairs, majority voting (MV) gives one (coverage, selective risk) point; SE gives three, one per budget (B75/B100/B124). `coverage = resolvedCount/totalCount`, `selectiveRisk = 1 - accuracy`, where `accuracy` is computed only over resolved examples (escalated examples are never counted as incorrect). All values are read directly from the existing `riskCoverageSummary.csv` — nothing is recomputed from raw predictions here. Differences are reported as SE − MV at each budget.

## 2. Full numerical table

| Model | Dataset | SE budget | MV coverage | MV accuracy | MV risk | SE coverage | SE accuracy | SE risk | Cov. diff | Risk diff | Acc. diff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3.1_8b | hatemoderate | B75 | 0.825 | 0.700 | 0.300 | 0.312 | 0.760 | 0.240 | -0.512 | -0.060 | +0.060 |
| llama3.1_8b | hatemoderate | B100 | 0.825 | 0.700 | 0.300 | 0.487 | 0.738 | 0.262 | -0.337 | -0.038 | +0.038 |
| llama3.1_8b | hatemoderate | B124 | 0.825 | 0.700 | 0.300 | 0.583 | 0.708 | 0.292 | -0.242 | -0.008 | +0.008 |
| llama3.1_8b | aegis | B75 | 0.777 | 0.839 | 0.161 | 0.258 | 0.922 | 0.078 | -0.520 | -0.083 | +0.083 |
| llama3.1_8b | aegis | B100 | 0.777 | 0.839 | 0.161 | 0.445 | 0.927 | 0.073 | -0.332 | -0.088 | +0.088 |
| llama3.1_8b | aegis | B124 | 0.777 | 0.839 | 0.161 | 0.497 | 0.915 | 0.085 | -0.280 | -0.075 | +0.075 |
| qwen2.5_7b | hatemoderate | B75 | 0.863 | 0.586 | 0.414 | 0.625 | 0.604 | 0.396 | -0.238 | -0.018 | +0.018 |
| qwen2.5_7b | hatemoderate | B100 | 0.863 | 0.586 | 0.414 | 0.710 | 0.599 | 0.401 | -0.153 | -0.013 | +0.013 |
| qwen2.5_7b | hatemoderate | B124 | 0.863 | 0.586 | 0.414 | 0.743 | 0.596 | 0.404 | -0.120 | -0.010 | +0.010 |
| qwen2.5_7b | aegis | B75 | 0.738 | 0.881 | 0.119 | 0.627 | 0.884 | 0.116 | -0.110 | -0.003 | +0.003 |
| qwen2.5_7b | aegis | B100 | 0.738 | 0.881 | 0.119 | 0.665 | 0.880 | 0.120 | -0.073 | +0.002 | -0.002 |
| qwen2.5_7b | aegis | B124 | 0.738 | 0.881 | 0.119 | 0.688 | 0.884 | 0.116 | -0.050 | -0.002 | +0.002 |
| mistral_7b | hatemoderate | B75 | 0.750 | 0.567 | 0.433 | 0.458 | 0.546 | 0.454 | -0.292 | +0.020 | -0.020 |
| mistral_7b | hatemoderate | B100 | 0.750 | 0.567 | 0.433 | 0.585 | 0.547 | 0.453 | -0.165 | +0.020 | -0.020 |
| mistral_7b | hatemoderate | B124 | 0.750 | 0.567 | 0.433 | 0.615 | 0.553 | 0.447 | -0.135 | +0.014 | -0.014 |

## 3. Model-dataset observations (B75 → B100 → B124 trajectory)

**llama3.1_8b × hatemoderate**: Lower SE coverage is accompanied by lower selective risk at all three budgets (risk diff negative throughout: -0.060, -0.038, -0.008), with the gap narrowing as budget/coverage increases toward MV's level.

**llama3.1_8b × aegis**: Same qualitative pattern as HateModerate, with a larger risk reduction at every budget (-0.083, -0.088, -0.075) — the largest SE-favoring gap of any pair in this table.

**qwen2.5_7b × hatemoderate**: Only modest risk improvement (-0.018, -0.013, -0.010) — directionally the same as llama, but the magnitude is small relative to llama's pairs.

**qwen2.5_7b × aegis**: Risk is essentially unchanged from MV at every budget (-0.003, +0.002, -0.002) — coverage drops modestly but selective risk does not meaningfully move.

**mistral_7b × hatemoderate**: SE has lower coverage but HIGHER selective risk than MV at every budget (+0.020, +0.020, +0.014) — the one pair where the coverage/risk relationship runs opposite to the usual selective-prediction expectation.

## 4. Cross-condition interpretation

The five pairs show genuinely heterogeneous behavior — this is not noise around one universal pattern. Two pairs (llama on both datasets) show SE trading coverage for meaningfully lower risk; one pair (qwen/HateModerate) shows the same direction but weakly; one pair (qwen/AEGIS) shows no meaningful risk change; and one pair (mistral/HateModerate) shows SE's lower coverage paired with *higher* risk, the opposite of the usual selective-prediction expectation.

Critically, the paired agreement analysis (`seMvAgreement.md`) already established that SE and MV make the exact same decision on 99.67% of jointly-resolved examples, with only 11 discordant examples across all 3,310 example-comparisons. **This means the coverage/risk differences reported above should be read primarily as differences in *which* examples each method resolves, not as evidence that SE judges shared examples differently or better.** Where SE's resolved-set risk is lower than MV's, it is because SE's selection of which examples to resolve happens to exclude examples that would have been wrong — not because SE reaches a different, better verdict on the same examples.

## 5. What this does NOT establish

- **Not causal.** A negative risk-diff at lower coverage does not mean *lowering coverage causes* lower risk — both are downstream consequences of which examples SE's stopping rule happens to resolve at a given budget, not a controlled intervention on coverage itself.
- **Not a universal difficulty detector.** The pattern holds for 2 of 5 pairs clearly, weakly for 1, is absent for 1, and is *inverted* for 1 (mistral/HateModerate). Any claim that "SE identifies easier examples" must be scoped to the specific pairs where the data support it, not stated as a general property of the method.
- **Selective risk, overall correctness, and coverage are three different quantities and are not interchangeable.** Selective risk is conditional on resolution (it says nothing about the escalated majority in some low-coverage conditions); overall correctness across *all* 400 examples would need to fold in the escalation rate itself (not computed here); coverage alone says nothing about accuracy. A pair can have simultaneously low coverage, low selective risk, AND low overall usefulness if escalation is too aggressive — this table does not adjudicate that tradeoff, it only reports the two axes requested.

