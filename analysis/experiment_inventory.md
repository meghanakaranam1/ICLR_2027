# ICLR 2027 Experiment Inventory

Read-only audit. No LLM/API calls were made, no result files were modified, deleted, or renamed, and the manuscript was not edited to produce this report. The one background process already running before this task started (isocompute calibration, `run_isocompute_final.py --calibrate-only` under `isocompute_calibrate_autoheal.sh`) was left untouched, per instruction — it is inventoried below as an in-progress item, not restarted or stopped.

## A. Executive summary

- **Distinct experiments/conditions discovered: 34** (counting each model×dataset×condition combination that has its own result directory or clearly distinct purpose)
- **COMPLETE: 12**
- **COMPLETE BUT STALE: 2**
- **PARTIAL: 9**
- **MISSING: 4**
- **OBSOLETE: 5**
- **INTENTIONALLY EXCLUDED: 0** (no evidence of a documented deliberate-exclusion decision anywhere in the repo)
- **UNKNOWN: 2**

The manuscript source available in this repo (`methods_section.tex`) contains **only Methods and Theory sections — no Results section, no tables, no figures with numbers**. This means nearly every empirical experiment below is technically "not yet represented in the manuscript," not because of a documented exclusion decision, but because the Results section doesn't exist yet in what's checked into this repo.

## B. Master experiment table

| Experiment | Model | Dataset | Condition/Budget | Expected N | Actual N | Schema | Status | Result path | Used in manuscript? | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Primary MV | llama3.1:8b | hatemoderate | graph_mv | 400 | 400 | current | A | `results_02.../llama3.1_8b/hatemoderate/graph_mv` | No (no Results section) | |
| Primary SE | llama3.1:8b | hatemoderate | graph_ucb_B75/100/124 | 400 each | 400 each | current | A | same tree | No | |
| Primary MV | llama3.1:8b | aegis | graph_mv | 400 | 400 | current | A | `results_02.../llama3.1_8b/aegis/graph_mv` | No | |
| Primary SE | llama3.1:8b | aegis | graph_ucb_B75/100/124 | 400 each | 400 each | current | A | same tree | No | |
| Primary MV | qwen2.5:7b | hatemoderate | graph_mv | 400 | 400 | current | A | `results_02.../qwen2.5_7b/hatemoderate/graph_mv` | No | |
| Primary SE | qwen2.5:7b | hatemoderate | graph_ucb_B75/100/124 | 400 each | 400 each | current | A | same tree | No | |
| Primary MV | qwen2.5:7b | aegis | graph_mv | 400 | 400 | current | A | `results_02.../qwen2.5_7b/aegis/graph_mv` | No | |
| Primary SE | qwen2.5:7b | aegis | graph_ucb_B75/100/124 | 400 each | 400 each | current | A | same tree | No | |
| Primary MV | mistral:7b | hatemoderate | graph_mv | 400 | 400 | current | A | `results_02.../mistral_7b/hatemoderate/graph_mv` | No | |
| Primary SE | mistral:7b | hatemoderate | graph_ucb_B75/100/124 | 400 each | 400 each | current | A | same tree | No | |
| MV isocompute (old) | llama3.1:8b | hatemoderate | graph_mv_isocompute, N_MV=140 | 400 | 400 | mixed (59.5% current-schema) | B | `results_02_n400_from_spark/llama3.1_8b/hatemoderate/graph_mv_isocompute` | No | Never re-calibrated per pair; see prior-session audit |
| MV isocompute (old) | llama3.1:8b | aegis | graph_mv_isocompute, N_MV=140 | 400 | 114 | mostly stale (8.8% current-schema) | C | `results_02_n400_from_spark/llama3.1_8b/aegis/graph_mv_isocompute` | No | |
| MV isocompute (old) | qwen2.5:7b | hatemoderate | graph_mv_isocompute, N_MV=140 | 400 | 100 | 0% current-schema | C | `results_02_n400_from_spark/qwen2.5_7b/hatemoderate/graph_mv_isocompute` | No | |
| MV isocompute (old) | qwen2.5:7b | aegis | graph_mv_isocompute, N_MV=140 | 400 | 0 | n/a | D | none found | No | |
| MV isocompute (old) | mistral:7b | hatemoderate | graph_mv_isocompute, N_MV=140 | 400 | 0 | n/a | D | none found | No | |
| MV isocompute calibration (this session, in progress) | llama3.1:8b/qwen2.5:7b/mistral:7b | hatemoderate/aegis | per-pair calibration | ~24 calib examples × 5 pairs | llama/hatemoderate: 14/~30 candidate-probes done; others not started | current | **In progress, actively running, untouched per this task's instructions** | `analysis/risk_coverage/isocompute_calib_checkpoint_*.jsonl` | No | See "in progress" note below |
| Isocompute calibration probes (abandoned, earlier vintage) | llama3.1:8b | hatemoderate | calib_nmv100/130/140/160 | 12 each | 12 each | unknown/old | E | `results_calibration_tmp/` | No | Superseded scratch runs, predate this session's redesign |
| Isocompute calibration probes (abandoned) | llama3.1:8b | aegis | calib_nmv60/80/95/100/130/160 | 12 each | 3-12 | unknown/old | E | `results_calibration_aegis_tmp/` | No | One (nmv160) only got 3/12 |
| Isocompute calibration probes (abandoned) | qwen2.5:7b | hatemoderate | calib_nmv80/100/110/140 | 12 each | 12 each | unknown/old | E | `results_calibration_qwen_tmp/` | No | |
| Isocompute calibration probes (abandoned) | qwen2.5:7b | aegis | calib_nmv60/70/90/120 | 12 each | 12 each | unknown/old | E | `results_calibration_qwen7b_aegis_tmp/` | No | |
| single_reasoning | REASONING_MODEL (deepseek-r1:8b, local) | any | one call, no DAG | — | 0 | n/a | D | none | No | Code exists (`run_single_reasoning`), explicitly documented as "never run for MODELS/graph conditions," no result directory anywhere |
| Frontier: Claude Haiku 4.5 (via HopGPT) | claude-haiku-4.5 | hatemoderate + aegis | single_call/graph_mv/SE B75/100/124 | 4000 | 3092 (77%) | current | C | `analysis/frontier_model/claude_haiku_4_5/` | No | Blocked by HopGPT $3000 budget cap this session |
| Frontier: Llama 70B (via HopGPT) | llama3-3-70b-instruct | hatemoderate + aegis | same 5 methods | 4000 | 2842 (71%) | current | C | `analysis/frontier_model/llama_70b/` | No | Same budget-cap block; HateModerate fully done (400/400×5), AEGIS incomplete |
| Frontier: Llama 3.3 70B (local, Spark) | llama3.3:70b (Ollama) | aegis | remaining ~1158 units | 1158 | user reports "finished" | current (code-level; schema unverified locally) | **UNKNOWN** | `llama_70b_spark_local/` on Spark, not yet synced to this machine | No | User states complete; no local file exists to verify row count/schema/duplicates from this machine as of this audit |
| Haiku B100 completion (targeted top-up) | claude-haiku-4-5 (direct Anthropic API) | hatemoderate | SE B100 only, 26 missing examples | 26 | 1 (8 attempted, 1 succeeded) | current | C | `analysis/frontier_model/haiku_hatemoderate_B100_completion/` | No | Stopped by the $5 self-imposed spend cap this session; net effect 375/400 for this one cell |
| GPT-5.6 frontier attempt | gpt-5.6-luna/sol/terra (via HopGPT) | aegis/hatemoderate | compat test only | — | 0 usable (content-filter blocked) | n/a | E | `results_TEST_frontier/` (empty file) | No | Confirmed Azure content-filter blocks on both datasets this session; abandoned in favor of Haiku/Llama-70B |
| Fixed-compute SE-vs-MV (B=3/5/10) | qwen2.5:7b, mistral:7b, llama3.1:8b | hatemoderate + aegis | pilot only | 300 (pilot) | 325 logged (pilot scope) | current | C | `analysis/fixed_compute/` | No | Pilot validated cleanly; full 400-example run never approved/launched (`FULL_RUN_APPROVED` marker absent) |
| Delta/confidence-level sensitivity | mistral:7b | hatemoderate | δ ∈ {0.01, 0.05, 0.1} | unclear (likely N=20 was the intended probe size) | 20 each | unknown vintage | C | `results_TEST_delta_sanity/` | No | Real robustness data exists but only N=20, one model, one dataset |
| Delta sensitivity retest | llama3.1:8b | aegis | δ=0.01, B75 | unclear | 0 (empty file) | n/a | D | `results_TEST_delta_retest/` | No | File exists, 0 usable rows |
| Temperature sweep | qwen2.5:7b | hatemoderate | T ∈ {0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 1.0}, graph_mv only | 100 each | 100 each | unknown vintage, MV-only (no SE) | C | `results_temp_sweep_qwen/` | No | Real, fairly complete 7-point sweep, but MV-only, one model/dataset pair, N=100 not 400 |
| Temperature T=0.8 probe | qwen2.5:7b | hatemoderate | T=0.8, graph_mv | 100 | 100 | unknown vintage | B (likely subsumed by the sweep above) | `results_temp08_check/` | No | Appears to be an earlier single-point check preceding the full sweep |
| Confidence repass | llama3.1:8b, qwen2.5:7b, mistral:7b | hatemoderate, aegis, xstest | graph_mv only, N=100 | 100 each | 100 each | unknown vintage | C | `results_confidence_repass/` | No | 7 files; purpose (per filename) is a confidence-field backfill/verification pass, not a new experimental condition |
| Second domain: XSTest | llama3.1:8b, qwen2.5:7b | xstest | graph_mv + SE B75/100/124 | 400 (to match primary N) | 100 (30 for qwen B100) | current-era but N=100 | C | `results_02/{llama3.1_8b,qwen2.5_7b}/xstest/` | No | Directly answers REVISION_PLAN.md's open question #2 ("second domain") — real attempt exists but at N=100, missing mistral entirely, no isocompute |
| ToxicChat | any | toxicchat | any | — | 0 | n/a | D | none | No | `build_toxicchat_node_systems()` and `DATASET_CONFIGS["toxicchat"]` exist in code; zero result files anywhere |
| Original run.py probes (pre-extension) | llama3.1:8b, qwen2.5:7b | aegis, hatemoderate | graph_mv, graph_ucb_B75 | small/unclear | 0-100, highly inconsistent | pre-current schema | E | `results/`, `results_01/` | No | Tiny, inconsistent early probes (one file is literally 0 rows); clearly superseded by `results_02` |
| Escalation leading-candidate reconstruction | — | — | — | — | 0 raw.csv found | n/a | UNKNOWN | `results_escalation_leading_candidate/` | No | Directory exists but contains no `raw.csv`; likely log-only or already fully superseded once `leading_candidate` was added to the main schema (2026-09-20 per code comments) |

## C. Model-scale / generalization inventory

### 7-8B open-weight models (primary)
llama3.1:8b, qwen2.5:7b, mistral:7b — all COMPLETE at N=400 for HateModerate/AEGIS graph_mv + SE B75/100/124 (mistral only has HateModerate; no mistral/aegis primary pair was found in `results_02`, worth flagging — checked `check_all.py`'s default ROOT and confirmed no `results_02/mistral_7b/aegis/` directory exists at all, so mistral/AEGIS was apparently never part of the primary 5-pair design, consistent with this session's own repeated "5 pairs" framing).

### Larger open-weight models
- **llama3.1:70b**: referenced in `run_extension_n400.py`'s PAIRS list (`("llama3.1:70b", "hatemoderate", None)`) and in a code comment about B150 data ("results_02/llama3.1_70b/{aegis,hatemoderate}/graph_ucb_B150... fully complete"). A `results_02/llama3.1_70b/` directory does exist with `aegis/`, `hatemoderate/` subdirectories. **This is a genuinely separate, real 70B open-weight result set from the main 8B/7B trio**, not yet cross-referenced against the primary 5-pair table above (out of scope to fully audit row-by-row here, but its existence should be flagged for the "model-scale" story the paper may want to tell — it's real evidence beyond 7-8B).
- **qwen2.5:32b**: `results_02/qwen2.5_32b/hatemoderate/` exists. Same flag as above — real mid-scale (32B) evidence exists on disk, unaudited in depth here, worth a dedicated pass before deciding how/whether to use it.
- **llama3.3:70b (local, Spark)**: see Frontier row above — reported complete by user, not yet verifiable from this machine.

### Reasoning models
- **deepseek-r1:8b** (REASONING_MODEL): code path (`run_single_reasoning`) exists and is fully implemented, explicitly documented as "never run for MODELS/graph conditions... single_reasoning condition currently exists in this pipeline" being **unused at runtime** — confirmed via zero result directories anywhere in the repo. **Status: MISSING (D)**, not merely incomplete — genuinely never executed as far as saved outputs show.

### Frontier/API models
- **Claude Haiku 4.5**: real data exists (77% of 4000 target via HopGPT + 1 additional example via direct Anthropic API), see table above. PARTIAL.
- **Llama 3.3 70B via HopGPT**: 71% of 4000 target, PARTIAL.
- **GPT-5.6 family (sol/terra/luna)**: attempted, blocked entirely by Azure content-filter on every tested example both datasets — **zero usable data**, OBSOLETE (abandoned this session in favor of Haiku/Llama-70B, not a configuration that can be salvaged without switching providers/routing).
- **No evidence found** of any OpenAI (o3/o4/GPT-4-class), Gemini, or other frontier provider ever being called — no result files, no compat-test logs, nothing beyond the GPT-5.6-via-HopGPT attempt above. Per the explicit instruction not to infer testing from API-client code alone: there is Anthropic SDK code (`run_haiku_hatemoderate_b100_completion.py`) and HopGPT routing code, but no evidence of any other provider being exercised.

## D. Robustness/ablation inventory

| Ablation | Evidence found | Completeness |
|---|---|---|
| Temperature sensitivity | `results_temp_sweep_qwen/` (7 points, T=0.0-1.0) + `results_temp08_check/` | Real but narrow: 1 model (qwen2.5:7b), 1 dataset (hatemoderate), MV-only, N=100 |
| Confidence-level (δ) sensitivity | `results_TEST_delta_sanity/` (δ=0.01/0.05/0.1) | Real but narrow: 1 model (mistral:7b), 1 dataset (hatemoderate), N=20 only |
| Confidence-level retest | `results_TEST_delta_retest/` | Empty (0 rows) — attempted, not completed |
| Graph architecture (node count) | No evidence found | Not attempted — no code path varies the 3-node Screener/Analyst/Adjudicator structure |
| MV call count sensitivity | Effectively covered by the isocompute calibration probes (varying N_MV) across multiple abandoned scratch directories, plus the in-progress recalibration this session | Scattered, none brought to a complete/final state prior to this session's in-progress attempt |
| SE budget sensitivity | This *is* the primary B75/100/124 sweep — already COMPLETE for the 5 primary pairs | Complete, already the paper's central comparison |
| Alternative concentration bounds / stopping rules / escalation rules | No evidence found (no alternate-bound code path, no flag/config for a different elimination rule) | Not attempted |
| Seed sensitivity | No evidence found — `SEED=42` is a single hardcoded constant throughout; no alternate-seed result directories | Not attempted |
| Dataset sampling sensitivity | No evidence found | Not attempted |
| Prompt/candidate-label sensitivity | No evidence found beyond the per-dataset bespoke prompts themselves (not a controlled ablation) | Not attempted |

## E. Incomplete/unfinished experiments

1. **Isocompute (per-pair compute-matched MV), all 5 primary pairs** — the old N=140-global-constant data is stale/incomplete for every pair (see table); a proper per-pair recalibration is **actively in progress right now** (this session, separate task, untouched here per instruction) but not yet complete for any of the 5 pairs at the time of this audit.
2. **Frontier Claude Haiku 4.5**: 908/4000 units missing, concentrated in AEGIS; blocked by HopGPT budget, a further $5 top-up attempt only recovered 1 more example (375/400 for the specific HateModerate×B100 cell).
3. **Frontier Llama 3.3 70B via HopGPT**: 1158/4000 units missing, entirely in AEGIS; a parallel local-Ollama-on-Spark effort is reported complete by the user but unverified from this machine.
4. **Fixed-compute (B=3/5/10) SE-vs-MV**: pilot-validated only; full N=400×5-pair run never approved/launched.
5. **XSTest (second domain)**: real MV+SE data exists for 2/3 models at N=100 (not 400), 0/3 for mistral, no isocompute condition.
6. **single_reasoning / deepseek-r1:8b**: fully coded, zero data — genuinely never run.
7. **llama3.1:70b, qwen2.5:32b**: real result directories exist but were not depth-audited in this pass (row counts/completeness/schema unverified here) — flagged as needing a dedicated follow-up audit before any claim is made about them.
8. **Robustness ablations (temperature, delta)**: real but narrow (single model/dataset, small N) — not close to a publishable ablation table as-is.

## F. Repository experiments not used in manuscript

Since the only manuscript source in this repo (`methods_section.tex`) has no Results section at all, this list is effectively "everything empirical" — reproduced here only for the items with genuinely *usable, complete* data, since those are the ones worth a deliberate decision about:

1. **Primary 5-pair MV+SE (B75/100/124)** — complete, current-schema, N=400 — the obvious Results-section backbone.
2. **llama3.1:70b and qwen2.5:32b result trees** — real, larger-scale evidence, not yet cross-checked for completeness in this pass.
3. **Temperature sweep (qwen/hatemoderate, 7 points)** — usable as a robustness appendix figure if the single-pair scope is disclosed.
4. **XSTest (2/3 models, N=100)** — usable as a limited second-domain data point if the N=100/missing-mistral/missing-isocompute limitations are disclosed.

## G. Planned-but-never-completed experiments

Evidence-graded per instruction (only counting items the repository shows were actually planned, not incidental code):

- **Clearly planned, not completed**: "Second/third domain for generalizability" (REVISION_PLAN.md, open question #2, explicitly unresolved) — XSTest is a real but incomplete attempt at this (Section F above).
- **Clearly planned, not completed**: "Add temperature ablation and an iso-compute MV condition... from the start this time" (REVISION_PLAN.md, sequencing item #4) — both exist as real but incomplete/stale data (temperature: narrow scope; isocompute: stale, currently being redone).
- **Clearly planned, not completed**: DKW/factor-of-2 theory correction and the O(log T)→O(√T) regret-claim correction (REVISION_PLAN.md's Tier-4 theory section) — the *current* `methods_section.tex` shows the O(√T) fix WAS made (Theorem "Quantile Estimation Across Episodes" / "Cumulative Excess Sample Cost" explicitly supersede an "incorrectly claimed" O(log T)), but the **independent-per-arm-query framing that DKW-avoidance rests on was never corrected** — and this session's own code audit (see chat transcript, "Three-Way Comparison" turn) found that framing does not match the actual implementation. This is a live, unresolved discrepancy between a planning document, two draft-text sources, and the code.
- **Possibly planned**: "Novelty vs. Even-Dar... pull one concrete case from real run data where identify-or-escalate diverges" (REVISION_PLAN.md) — `theory_draft_text.md` item 5 provides a *synthetic* worked example and explicitly says "swap in the actual p̂ values... later" — not yet done with real data.
- **Not actually planned / incidental**: ToxicChat dataset code (`build_toxicchat_node_systems`) — exists in code but is never mentioned in REVISION_PLAN.md or any draft text as an intended dataset; looks like boilerplate extended to match the other 3 bespoke-per-node builders, not a documented commitment.

## H. Conflicting/duplicate result versions

| Duplicate set | Rows/completeness | Schema | Verdict |
|---|---|---|---|
| `results_02/` | Current, most-complete for most conditions | Current (has `final_confidence`, `arm_estimates_json`, `leading_candidate` etc.) | **Authoritative / newest** for most pairs |
| `results_02_n400_from_spark/` | Partial for some conditions, but for a few specific ones (e.g., qwen2.5_7b/xstest, llama3.1_8b/xstest B100/B124/B75) has MORE or equal rows than `results_02` | Mixed | Used as a fallback by the existing `findBestSourceFile`/`check_all.py` pattern (picks whichever root has more rows per condition) — this is an intentional, already-established resolution rule in the codebase, not something this audit is overriding |
| `results_02_backup_pre_final_confidence_migration/` | Small, pre-dates the `final_confidence` schema field entirely | Old (pre-migration) | **Stale**, kept only as a historical backup per its own name; should never be used as a data source going forward |
| `results/`, `results_01/` | Tiny (0-100 rows, several genuinely 0), inconsistent | Pre-`results_02` schema | **Obsolete**, superseded by `results_02` per the codebase's own `RESULTS_DIR_NAME` versioning history (documented in `run_all_datasets.py`'s own comments) |
| `results_calibration_*_tmp/` (4 directories) | Small (3-12 rows), scratch calibration probes | Unknown/old | **Obsolete** — superseded by this session's in-progress, more rigorous per-pair recalibration |

## I. Recommended next decisions

- **Primary 5-pair MV+SE (B75/100/124)**: Finish before submission — this is already complete; the decision needed is *writing it up*, not running more.
- **Isocompute recalibration (all 5 pairs)**: Needs author decision — actively in progress this session (separate task); decide whether to let it finish before the deadline or scope it down.
- **Frontier Haiku/Llama-70B (HopGPT)**: Needs author decision — real budget constraint; decide whether partial-data disclosure or a further funded top-up is the path forward.
- **Fixed-compute (B=3/5/10)**: Needs author decision — pilot data alone already answers the reviewer's literal question (SE resolves ~0% at tiny budgets); full N=400 run may not be necessary to make that point.
- **XSTest (second domain)**: Needs author decision — real but incomplete (N=100, 2/3 models); either finish to N=400×3 models or present explicitly as a limited pilot.
- **single_reasoning/deepseek-r1:8b**: Needs author decision — coded but never run; either run it (answers a specific documented reviewer question per `run.py`'s own docstring) or drop the unused code path.
- **llama3.1:70b / qwen2.5:32b**: Needs author decision — real data exists, completeness unverified in this pass; audit in depth before deciding whether to include a "model scale" claim.
- **Temperature/delta ablations**: Needs author decision — real but narrow; either broaden (more models/datasets) or present explicitly as a single-pair robustness spot-check.
- **GPT-5.6 frontier attempt**: Obsolete; do not rerun — confirmed hard content-filter block, not a transient issue.
- **ToxicChat**: Can safely omit — no evidence it was ever a committed part of the plan, zero data exists.
- **Old `results/`, `results_01/`, `results_calibration_*_tmp/`, `results_02_backup_pre_final_confidence_migration/`**: Obsolete; do not rerun — superseded by `results_02`, kept only as historical record.
- **Independent-per-arm-query theory framing (DKW-avoidance argument)**: Needs author decision — this session's code audit found the framing doesn't match the implementation; this is a correctness issue in the theory section's own justification, not an experiment-completeness issue, and should be resolved before the theory section is finalized.
