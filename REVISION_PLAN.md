# ICLR 2027 Paper Plan — Content-Moderation-Primary

**This is not a revise-and-resubmit of the self-harm paper.** Per the 2026-08-24 decision, the ICLR 2027 paper moves the primary domain from self-harm/behavioral-health screening to **content moderation**, and adds a **generalizability/distribution-shift angle** (same architecture tested across multiple domains). The self-harm paper (`Meghana_Final_ML4HC_Paper (4).pdf`, MLHC Submission 254, rejected — 6 reviews, ratings 2/2/2/3/5) is now the **source of known failure modes to fix in the shared architecture and theory**, not the paper being edited. Its AEGIS/SWMH results are candidate secondary datasets for the generalizability comparison, not necessarily in the final paper — open question, flagged below.

**Why the pivot:** MLHC reviewer pushback concentrated on false-negative risk being flat regardless of sampling strategy in the self-harm domain specifically — a critique tied to that domain's label ambiguity and stakes, not one that says anything general about the DAG/adaptive-sampling architecture. Content moderation (HateModerate, and other domains for the generalizability comparison) sidesteps that specific critique while keeping a real safety-classification task.

**Prompt design rule already in force:** don't write a custom prompt per dataset. Reuse a single, already-published policy/taxonomy as the fixed system prompt (see `ICLR_2027/hatemoderate.py` — Facebook's 41-policy community standards, imported verbatim, never edited). Apply the same rule to any additional domain added for generalizability.

---

## What's already built

- `ICLR_2027/hatemoderate.py` — HateModerate (NAACL 2024, 7,702 examples: 4,796 hate / 2,906 not_hate), Facebook's 4-tier/41-policy taxonomy as fixed `SYSTEM_PROMPT`, `load_hatemoderate()`, `balanced_sample()`. Binary ground truth (hate/not_hate); node action space stays 3-arm `{hate, not_hate, escalate}`, matching the self-harm paper's architecture.
- `ICLR_2027/run.py` — same Screener→Analyst→Adjudicator DAG, `graph_mv` (MV n=5) vs `graph_ucb` (successive elimination, B=124/node, cost-capped at $10) ported from `AAAI_2027/germain.py`'s `ucb_node()`. Only run as a 2-example sanity check so far.
- `ICLR_2027/analyze.py` — already ahead of what the self-harm paper shipped: computes accuracy/FNR/FPR/escalation-rate/avg-pulls/cost, excludes human-escalated examples from accuracy/FNR/FPR (matching `germain.py` convention), and **auto-flags degenerate results** (≥90% escalation-to-human, <5 resolved examples in a class, accuracy exactly 0 or 1). This already does more than several things MLHC reviewers had to explicitly request in Tier 2 below.

---

## What carries over unchanged from the MLHC reviews

The architecture (DAG of 3-arm categorical successive elimination with identify-or-escalate) and its theory are shared across domains — a reviewer objection to the algorithm/proof doesn't go away by changing datasets. These need to be fixed once, centrally, then apply to whichever domains ship in the final paper.

### ⚠️ Theory — needs a decision before drafting

**DKW factor-of-2 error.** The self-harm paper's Appendix A derives `|p̂(c)−p(c)| ≤ 2·sup|F̂ₙ−F|` from writing a category probability as a CDF difference, then states the final bound as `ε_n(δ)` — dropping the factor of 2 the line above it just established. This also breaks the "~220 fewer samples than Hoeffding" claim used to justify DKW over Hoeffding in Related Work. Separately, Algorithm 1's actual per-arm width (`w_c = sqrt(ln(4KT_c²/δ)/(2T_c))`) is a Hoeffding-with-union-bound-over-K construction, not the DKW bound at all — so DKW is invoked to motivate the method but isn't what the method uses. Two ways to resolve:
- **Option A (recommended):** Correct the factor of 2 honestly, recompute the sample-savings number even if it shrinks or flips sign for K=3, and demote DKW to "a tighter bound worth using in a future implementation" rather than what Algorithm 1 does. No re-running of any experiment needed — Algorithm 1's actual behavior is untouched.
- **Option B:** Rebuild Algorithm 1 to really use the corrected DKW width and re-run everything on whichever domains ship. More defensible theory, real re-run cost.

**O(log T) deployment-regret claim.** Each input induces its own fresh label distribution and nothing persists across episodes in the algorithm as described — three separate MLHC reviewers (DgKN, p52r, and p52r's post-rebuttal follow-up) converged on this, and the rebuttal's attempted save ("each node eventually identifies its modal label") was explicitly rejected as conflating a node-level modal label with the paper's own input-specific framing. Two ways to resolve:
- **Option A (recommended):** Drop the "system gets more reliable with patient/case volume" framing. Keep only what's provable: a per-episode anytime correctness guarantee (`P(wrong committed label) ≤ δ`), no cross-episode dependency claimed.
- **Option B:** Reframe what's bounded — not "error decreasing over time" but "extra total sample cost required to hold a fixed system-wide failure probability across T deployed episodes" (e.g. via per-episode `δ_t = O(1/t²))`. Real proof work, more defensible if it holds up, don't draft prose around it until it's actually rederived.

**Recommendation:** Option A for both, given this is a new paper with a new empirical section to build — spend the risk budget on the new experiments (below), not on relitigating proofs three reviewers already independently flagged as broken.

### 📊 Statistical methodology — apply to whatever domains ship

- Paired significance testing (McNemar) on the headline FPR/accuracy comparisons, not just overlapping Wilson CIs. The self-harm paper's own McNemar result (2 vs 0 discordant pairs, p=0.48) shows this matters — don't skip it for HateModerate.
- Multiplicity correction across however many conditions get compared (the self-harm paper compared 10 with none).
- Escalation-specific metrics: `analyze.py` already computes escalation rate; add auto-resolution rate and escalation precision (fraction of human-escalated cases that were genuinely ambiguous by ground truth) as named metrics, matching what MLHC reviewer p52r had to request by hand last time.
- Novelty vs. Even-Dar et al. (2006): pull one concrete case from real run data where identify-or-escalate diverges from forced-label action elimination. Same ask MLHC reviewers made; same fix, different dataset.

---

## Open questions before building further

1. **Does AEGIS/SWMH (self-harm) stay in the ICLR paper at all**, as one point in the multi-domain generalizability comparison, or does the paper drop self-harm entirely and stay content-moderation-only? The 2026-08-24 decision says "multiple domains/datasets" for generalizability but doesn't specify which ones beyond HateModerate.
2. **What's the second/third domain** for the generalizability angle if not self-harm? The self-harm paper's own Related Work already cites ToxicChat (Lin et al., 2023) and RealToxicityPrompts (Gehman et al., 2020) — both content-adjacent, both would need an equivalent "find the published policy doc, use it as the fixed prompt" treatment like `hatemoderate.py` did for Facebook's standards.
3. **Target venue deadline/page limit/template** — MLHC's format won't be ICLR's; haven't checked the ICLR 2027 CFP yet.

---

## Suggested sequencing

1. **Decide the open questions above** — particularly #1 and #2, since they determine how much of the self-harm paper's empirical section (Tables 3, Figures 2-3) is reusable vs. scrapped.
2. **Tier 4 theory decision** (Option A recommended for both DKW and regret) — fix once, centrally, before any new domain's results get written up against it.
3. **Finish the HateModerate run**: scale `run.py` from the n=2 sanity check to a real balanced sample (N=60 is what's currently configured; consider whether that's enough given `analyze.py`'s own `MIN_CLASS_N=5` flag — 60 balanced (30/30) clears it but budget/temperature sweeps analogous to the self-harm paper's B∈{10,50,75,100,124,150} sweep would need more).
4. **Port the missing conditions** from the self-harm paper's 10-condition design (single-agent, MV n∈{1,3,5}, budget sweep) into `ICLR_2027/run.py` — right now it only has `graph_mv` (n=5) and `graph_ucb` (B=124). Add temperature ablation and an iso-compute MV condition (matched total calls to whatever budget ends up primary) from the start this time, rather than adding them under review pressure.
5. **Second domain**, once #2 above is answered — same fixed-published-taxonomy pattern as `hatemoderate.py`.
6. **Reproducibility**: ship exact prompts/parsing/raw outputs as supplementary from the start (self-harm paper got dinged for gating this on acceptance).
7. **Draft.**
