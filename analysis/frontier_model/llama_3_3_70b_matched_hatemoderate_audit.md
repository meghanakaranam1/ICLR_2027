# Llama 3.3 70B × HateModerate — Matched-to-Primary Audit

**Status: STOPPED at Step 4, blocked on network access.** Steps 1-3 (ID comparison, verification) are complete and reported below. Step 4 (running the missing examples) requires HopGPT (`api.ai.jh.edu`), which currently fails DNS resolution (`NXDOMAIN`) — confirmed to be a JHU-internal hostname that only resolves over the JHU VPN, not a general connectivity problem (general internet access confirmed working). No new inference has been attempted; no existing files were modified, deleted, or overwritten.

## 1. Canonical primary 400-example set

Verified across all three primary 7-8B HateModerate conditions — **all three use exactly the same 400 IDs** (llama3.1_8b, qwen2.5_7b, mistral_7b `graph_ucb_B124/raw.csv`, byte-identical ID sets, confirmed by direct set comparison, not assumed).

- **Canonical primary N = 400**
- **Label distribution: 200 hate / 200 not_hate** (exactly balanced, as expected from `balanced_sample_generic`)

## 2. Comparison with existing Llama 3.3 70B HateModerate IDs

```
Canonical primary N = 400
Existing 70B N = 400
Overlap = 16 (raw)  ->  15 (after excluding one ground-truth mismatch, see below)
Missing primary IDs = 384 (raw)  ->  385 (corrected)
Missing hate = 185 (raw)  ->  186 (corrected)
Missing not_hate = 199
```

The overlap is far smaller than a naive guess might suggest, and the reason is structural, not accidental: the existing 70B manifest was built from a plain prefix slice of the dataset loader's natural order (`loader()[:400]`, used by the frontier-experiment code), while the primary 7-8B experiments use `balanced_sample_generic(seed=42)` — a fundamentally different, class-balanced random sample. Two different sampling procedures over the same underlying 7702-example dataset will only incidentally agree on a small fraction of IDs, which is exactly what's observed (16/400 ≈ 4%).

## 3. Verification of existing 70B rows before reuse

For all 5 methods (single_call, graph_mv, graph_ucb_B75/100/124), every one of the (initially) 16 overlapping IDs was checked for: presence, no duplicates, non-null ground truth/final label/total_pulls, correct model tag (`llama3-3-70b-instruct`), and **ground-truth agreement with the canonical primary set**.

**One real problem found**: example ID `124` exists in both sets but disagrees on ground truth — canonical primary set says `hate`, the existing 70B row says `not_hate` for the identical ID. This is a genuine label inconsistency, most plausibly caused by dataset-loader drift between when each experiment was generated (this session independently confirmed elsewhere that the HateModerate loader's returned order/content is not perfectly stable run-to-run — see the earlier `balanced_sample_generic` reproducibility check in this session's history). **This ID was excluded from "safe to reuse" and moved into "needs fresh inference"** rather than silently accepting a mismatched ground truth.

All other 15 overlapping IDs passed every check cleanly across all 5 methods (75 row-checks total, 75/75 passed) — confirmed no duplicates within the existing 70B HateModerate data (400 unique IDs across 400 rows, every method).

**Corrected figures:**
```
Overlap (safe to reuse) = 15
Missing (needs new inference) = 385
Missing hate = 186
Missing not_hate = 199
```

## 4-10. Blocked

Steps 4 through 10 (running the 385 missing examples × 5 methods = 1925 new units, building the matched result set, validation, recomputed metrics, and the primary comparison) **cannot proceed until HopGPT connectivity is restored**. Per your explicit instruction ("If that is impossible, STOP and report the blocker rather than silently mixing backends"), no alternate backend was substituted and no examples were run.

**What's needed to unblock**: reconnect to the JHU VPN (confirmed via `nslookup api.ai.jh.edu` returning `NXDOMAIN` right now, while general internet access — e.g. `google.com` — works fine, isolating this to VPN/JHU-network access specifically, not a broader network problem).

Once connectivity is restored, this document will be updated (not replaced) with Steps 4-10.

## Terminal summary

```
PRIMARY HATE IDS: 200
PRIMARY NOT_HATE IDS: 200
EXISTING 70B OVERLAP: 15 (16 raw, 1 excluded for ground-truth mismatch on ID 124)
MISSING HATE IDS: 186
MISSING NOT_HATE IDS: 199
TOTAL NEW EXAMPLES REQUIRED: 385
TOTAL OVERLAPPING EXAMPLES RERUN: 0
FINAL MATCHED N: NOT YET COMPLETE -- blocked on HopGPT/VPN connectivity
FINAL LABEL BALANCE: NOT YET COMPLETE
BLOCKER: api.ai.jh.edu fails DNS resolution (NXDOMAIN) -- requires JHU VPN reconnection before Step 4 can run
```
