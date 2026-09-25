# Mistral 7B × AEGIS — Focused Audit

Read-only. No inference run, no manuscript edits, no existing result files modified, no interaction with the currently-running isocompute calibration.

**Correction to the prior `experiment_inventory.md` report**: that report stated "mistral only has HateModerate; no mistral/aegis primary pair was found in `results_02`... apparently never part of the primary 5-pair design." **This was wrong** — `results_02/mistral_7b/aegis/` exists with substantial real data (see below). The correction is noted here rather than silently editing the prior file.

## 1. Result directories found

| Root | Condition | Rows | Unique IDs | mtime |
|---|---|---|---|---|
| `results_02/mistral_7b/aegis/` | `single_agent` | 400 | 400 | Sep 22 15:28 |
| `results_02/mistral_7b/aegis/` | `graph_mv` | 400 | 400 | Sep 22 01:44 |
| `results_02/mistral_7b/aegis/` | `graph_ucb_B75` | 155 | 155 | Sep 22 18:16 (most recent file in the tree) |
| `results_02/mistral_7b/aegis/` | `graph_ucb_B100` | 100 | 100 | Sep 21 20:32 |
| `results_02/mistral_7b/aegis/` | `graph_ucb_B124` | 100 | 100 | Sep 21 22:59 |
| `results_02/mistral_7b/aegis/` | `graph_ucb_B150` | — | — | not present |
| `results_02/mistral_7b/aegis/` | `graph_mv_isocompute` / `graph_mv_isocompute_final` | — | — | not present |
| `results_02_n400_from_spark/mistral_7b/aegis/` | all of single_agent/graph_mv/B75/B100/B124/**B150** | 100 each | 100 each | Sep 19 21:53 (all six files, same batch) |
| `results_02_backup_pre_final_confidence_migration/mistral_7b/aegis/` | graph_mv/B75/B100/B124/B150 (no single_agent) | 100 each | 100 each | Sep 20 21:28 |

No `mistral:7b`+`aegis` data exists under any other naming convention (`mistral_7b`, `mistral-7b`, etc. all searched) and no alternate directory hides it — the three roots above are exhaustive.

## 2. Explicit repository evidence

**Direct, unambiguous evidence it was planned** — `run_extension_n400.py:45`:
```python
("mistral:7b", "aegis", None),  # added 2026-09-21: extend after the prompt-fix refix job finishes at N=100
```
This is the exact same `PAIRS`-list mechanism used to drive extension of the other primary pairs to N=400 — mistral×aegis is listed alongside them, with a comment describing the specific plan ("extend after the prompt-fix refix job finishes at N=100").

Corroborating comment, `run_all_datasets.py:133-138` (discussing why B150 was dropped from `BUDGET_GRID`):
> "...Confirmed via disk audit: `results_02/llama3.1_70b/{aegis,hatemoderate}/graph_ucb_B150` and **`results_02/mistral_7b/{aegis,hatemoderate}/graph_ucb_B150`** are all fully complete..."

This confirms mistral×aegis was carried through the *same* budget-sweep design as every other pair, including the now-dropped B150 point (100/100 complete at that budget, per the `results_02_n400_from_spark` and `results_02_backup_pre_final_confidence_migration` snapshots — though not present at all in the current `results_02`, which only kept B75/B100/B124 per the active `BUDGET_GRID`).

No evidence found anywhere (code comments, `REVISION_PLAN.md`, `theory_draft_text.md`, or the manuscript) of a documented decision to **exclude** mistral×aegis, nor any "3×2 matrix"/"6 pairs"/"5 pairs only" language pinning the design to a specific count. The "five pairs" framing that recurs throughout this session's own work (frontier experiments, isocompute audit, SE-vs-MV agreement analysis) appears to be this session's own working assumption, not a decision documented anywhere in the repository itself.

## 3. Classification of the found data

**D — run under the same protocol, partially complete.** (Matching the audit's own option set: not A "never planned," not fully C either since `graph_mv`/`single_agent` are complete — it's a mix: those two conditions are COMPLETE, the three SE budgets are PARTIAL, at exactly the point a resumable extension job would stall if interrupted.)

Configuration-parity check, all pass:
- `graph_mv` and `single_agent` cover the **identical 400-example ID set** (verified directly, not assumed).
- `graph_ucb_B75/B100/B124`'s completed IDs are each a strict **subset** of that same 400-ID set (not a different sample).
- `B100` and `B124` are stuck at the **exact same 100 IDs** — consistent with a single resumable extension pass that advanced all three budgets together and was interrupted at the same point, with `B75` getting slightly further (155) before stopping.
- Global `TEMPERATURE=0.7`, `SEED=42` constants are process-wide, not overridden per-pair anywhere in the code — no evidence mistral×aegis would run under different settings than the other 5 pairs.
- Schema: current (`final_confidence`/`arm_estimates_json`/`leading_candidate` columns present), though only 232/400 `graph_mv` rows have `final_confidence` actually populated (58%) — same "mixed vintage within one file" pattern already flagged for several other pairs in the broader inventory, not unique to this one.

## 4. Can the current machinery reproduce N=400 for this pair without changing protocol?

**Yes.** `run_extension_n400.py` already has `("mistral:7b", "aegis", None)` in its `PAIRS` list, using the exact same `run_condition`-based, checkpoint/resume-safe machinery as every other pair (confirmed unmodified, not something this audit or any other task this session touched). Resuming it would pick up exactly where `graph_ucb_B75/B100/B124` left off (155/100/100), using the same 400-example set already fixed by `graph_mv`/`single_agent`'s existing 400 IDs. No isocompute condition exists for this pair yet, consistent with the broader isocompute gap already documented for other pairs.

## 5. Comparison against the 5 established primary pairs

| Property | Mistral×AEGIS | Other 5 primary pairs |
|---|---|---|
| N target | 400 (confirmed via `graph_mv`/`single_agent`) | 400 |
| Temperature | 0.7 (global constant, same code) | 0.7 |
| Seed | 42 (global constant, same code) | 42 |
| Prompts/node systems | `build_aegis_node_systems()`, same bespoke-per-node design | same |
| Graph | Screener→Analyst→Adjudicator, same `GRAPH` constant | same |
| Labels | `("unsafe", "safe")` + escalate, same `DATASET_CONFIGS["aegis"]` | same |
| SE budgets | 75/100/124 planned (partial), 150 once tried (100/100, later dropped) | 75/100/124 (complete) |
| Sampling procedure | Same `run_condition`/example-list machinery | same |

No configuration difference found anywhere — this pair was being run under **identical protocol** to the other five, just interrupted mid-extension.

## 6. Final answer

**Was Mistral × AEGIS originally intended to be part of the main experimental matrix? Yes, explicitly.**

The `run_extension_n400.py:45` comment ("added 2026-09-21: extend after the prompt-fix refix job finishes at N=100") is direct, first-party evidence of an active plan to bring this pair to N=400, using the identical mechanism and protocol as the other five pairs. `graph_mv` and `single_agent` already succeeded at reaching the full 400. The three SE budget conditions were mid-extension (advancing together, B75 slightly ahead) when work stopped — there is no evidence of an intentional exclusion decision anywhere in the repository, and the "five primary pairs" framing used elsewhere in this session's own work was this session's assumption, not a decision documented in the codebase.

## Terminal summary

```
CLASSIFICATION: PARTIAL (graph_mv/single_agent COMPLETE at 400/400; SE B75/B100/B124 PARTIAL at 155/100/100)
WAS IT PLANNED: YES -- explicit evidence in run_extension_n400.py:45
EVIDENCE: PAIRS list entry with dated comment; B150 cross-reference comment in run_all_datasets.py;
          identical 400-ID set shared across graph_mv/single_agent; SE budgets are strict subsets of that same set
CAN BE RESUMED WITHOUT PROTOCOL CHANGE: YES -- run_extension_n400.py already configured for this exact pair
SPARK DATA: user reports additional data exists on Spark; copy command provided (rsync into a separate
            inspection directory, not overwriting local results_02/mistral_7b/aegis/, since local B75 is
            already ahead at 155/400 and row counts need comparing before any merge decision)
```
