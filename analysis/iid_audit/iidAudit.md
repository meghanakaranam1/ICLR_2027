# IID / Repeated-Sampling Diagnostic Audit

## 1. Purpose

The paper's theoretical analysis models repeated LLM outputs for a fixed example x as IID draws from a fixed categorical distribution p_c(x). This audit checks whether the existing repeated-call data provide obvious diagnostic evidence against that assumption (stationarity across call position, lack of serial dependence, stable label frequencies). It is explicitly a diagnostic sweep, not a formal statistical test of IID, and a failure to find evidence of dependence does not prove independence.

## 2. Data and representation

**Central finding: individual per-call outcomes are not stored anywhere in the existing outputs.** This was verified by direct inspection of `run_all_datasets.py`, not assumed:

- `run_node_mv` (majority-vote path) builds an in-memory `votes` list — one parsed label per call — used once to build the Adjudicator's context string (`build_adjudicator_context_mv`), then discarded. `votes` is not among the ~26 columns `run_condition` writes to `raw.csv`.
- `ucb_node` (successive-elimination path) fires one **identical** request per currently-active arm each round (confirmed at `run_all_datasets.py:1162-1173`, comment: "a 'pull for arm X' here is an IDENTICAL request regardless of X"), and increments per-arm win/pull counters — but the parsed label of each individual pull is never stored. The persisted `arm_elimination_trace` records only `{round, active_before, eliminated, active_after}` per round: the surviving/eliminated ARM SET, not which label each call produced.
- `arm_estimates_json` (where present) is a single FINAL snapshot of each arm's cumulative win-rate at the moment of termination — not a per-round or per-call value, so it cannot be sliced into "early" vs "late" observations.
- No separate raw-response log exists in any results directory (directory search performed, none found), and every real run used `sanity=False`, so even the verbose per-call print statements present in the code were never written to any log during actual data generation.

**Consequence**: Analyses A (call-position label frequencies), B (adjacent-call agreement), C (early-vs-late distributions), and D (order sensitivity) as specified all require an ordered sequence of individual call outcomes per example. That sequence does not exist in any stored artifact, for any example, under any condition. Reconstructing one would mean fabricating data, which the task instructions explicitly forbid ("do not invent or reconstruct missing data"). **These four analyses are therefore reported as not computable from the existing data, rather than approximated.**

## 3. Call-position distributions

**Not computable** — see Section 2. `iidCallPositionTable.csv` contains a single row documenting this rather than fabricated frequencies.

## 4. Adjacent-call dependence

**Not computable** — see Section 2. `iidAdjacentAgreement.csv` contains a single row documenting this. No adjacent-call agreement statistic, and no reference/null construction, can be computed without a per-call sequence to compute it from.

## 5. Order sensitivity

**Not computable** — see Section 2. `iidOrderSensitivity.csv` contains a single row documenting this. Running-modal-label changes, first-half/second-half modal comparisons, and "when is the final label already established" all require the ordered per-call sequence that is not stored.

## 6. Relation to SE resolution

Per-call consistency cannot be compared between resolved and escalated examples for the reason above. What **is** genuinely stored and comparable is round-level convergence speed: `total_pulls` (how many individual calls were fired in total before the node stopped) and the elimination round (which round number the terminal event occurred in) — both already computed read-only in `analysis/seTrajectoryFeatures.csv` from a prior step. Reporting these here as the closest available proxy, **not** as a per-call consistency measure and **not** as a predictor of IID behavior (per instruction):


| Model | Dataset | Budget | N resolved | Mean pulls (resolved) | Mean elim. round (resolved) | N escalated | Mean pulls (escalated) | Mean elim. round (escalated) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3.1_8b | hatemoderate | B75 | 125 | 110.4 | 24.0 | 275 | 224.0 | 25.0 |
| llama3.1_8b | hatemoderate | B100 | 195 | 118.3 | 26.4 | 205 | 294.0 | 33.8 |
| llama3.1_8b | hatemoderate | B124 | 233 | 141.1 | 28.7 | 167 | 351.1 | 41.6 |
| llama3.1_8b | aegis | B75 | 103 | 109.1 | 24.0 | 297 | 219.5 | 25.1 |
| llama3.1_8b | aegis | B100 | 178 | 108.2 | 26.9 | 222 | 282.0 | 32.5 |
| llama3.1_8b | aegis | B124 | 199 | 116.9 | 28.9 | 201 | 335.2 | 38.3 |
| qwen2.5_7b | hatemoderate | B75 | 250 | 88.5 | 24.0 | 150 | 214.7 | 25.0 |
| qwen2.5_7b | hatemoderate | B100 | 284 | 90.3 | 25.1 | 116 | 259.1 | 32.6 |
| qwen2.5_7b | hatemoderate | B124 | 297 | 98.7 | 26.1 | 103 | 294.0 | 38.2 |
| qwen2.5_7b | aegis | B75 | 251 | 93.1 | 24.0 | 149 | 194.9 | 24.7 |
| qwen2.5_7b | aegis | B100 | 266 | 93.6 | 24.7 | 134 | 221.4 | 28.0 |
| qwen2.5_7b | aegis | B124 | 275 | 99.1 | 25.5 | 125 | 236.6 | 30.3 |
| mistral_7b | hatemoderate | B75 | 183 | 73.2 | 24.0 | 217 | 217.8 | 24.5 |
| mistral_7b | hatemoderate | B100 | 234 | 78.6 | 25.1 | 166 | 238.4 | 25.8 |
| mistral_7b | hatemoderate | B124 | 246 | 80.5 | 25.9 | 154 | 252.4 | 25.9 |

Escalated examples consistently use far more total pulls than resolved ones across every condition (consistent with escalation meaning "ran to budget exhaustion without converging") — this is a property of the stopping rule's mechanics, not evidence about whether the underlying repeated calls are IID.

## 7. Interpretation and limitations

- **These are diagnostics, not an IID test** — and in this case, the diagnostics could not even be attempted for four of the five requested analyses, because the prerequisite data (per-call outcome sequences) do not exist in any stored artifact.
- **Independence is an assumption of the theoretical analysis**, not something this audit can verify or falsify with the data currently available.
- **Absence of detectable dependence does not prove IID** — and here we do not even have absence-of-detectable-dependence; we have an inability to test for it at all with existing artifacts.
- **Any future dependence check would require re-instrumenting the pipeline** to persist per-call outcomes (e.g., adding the `votes` list, or per-pull labels in `ucb_node`, to the CSV schema) and re-running — which was explicitly out of scope for this audit ("do NOT run any new LLM inference"). Until that instrumentation exists, the theoretical guarantee's IID assumption should be understood as an assumption the existing experiments do not directly test, not as one that has been checked and passed.

