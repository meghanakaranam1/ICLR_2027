# Theory section — drafted text (items 1, 2, 5, 6)

Ready-to-use prose for whenever the ICLR paper draft starts. All four are domain-agnostic — they describe the shared DAG/adaptive-sampling architecture, not AEGIS/SWMH/HateModerate specifics, so they carry over regardless of which dataset(s) end up in the paper. Verified against the actual implementation (`navigator_experiments/experiment.py:396` `ucb_node()`, `call_llm()` at line 311) before drafting, not just against the old PDF's prose.

---

## 1. Methods — clarifying what "pulling an arm" means

Insert immediately after Algorithm 1 (or wherever the algorithm is first described in the new draft):

> Pulling arm `c` re-issues the node's fixed classification prompt — the prompt does not vary by arm — and scores the draw as a Bernoulli outcome: reward 1 if the returned label equals `c`, else 0. Each active arm is pulled via its own independent query each round; a round with `|A_act|` active arms costs `|A_act|` LLM calls, not one. This is why total calls per input (Table 2, "Avg. pulls") can exceed the budget's face value early on, before elimination reduces the active set. Because each arm's pulls come from statistically independent draws, per-arm concentration bounds are the appropriate tool for confidence widths (see Appendix B), rather than a bound that assumes one shared sample update all arms' estimates simultaneously.

That last sentence does double duty — it states the mechanism and sets up why DKW (item 2) doesn't apply, without needing a separate transition paragraph later.

---

## 2. Cutting DKW as Algorithm 1's justification

**Related Work / Methods — replace** any "DKW avoids the `ln K` union-bound penalty" framing with:

> Prior work in this space has used the Dvoretzky–Kiefer–Wolfowitz inequality to bound categorical estimation error without a `ln K` penalty. That result requires a single shared multinomial sample stream, where one draw simultaneously updates every category's empirical count — the source of the correlated estimates DKW exploits. Our algorithm pulls each arm via an independent re-query (see Methods), so no such shared stream exists; a per-arm Hoeffding bound with a union bound over the `K` arms (Appendix B) is the applicable tool here, and that is what Algorithm 1 uses throughout.

**Limitations / Future Work — add:**

> The current design pulls each arm independently, so a response revealing information about all `K` labels only updates the tally of the one arm it was queried for — discarding roughly `(K-1)/K` of each call's information relative to a design where a single shared draw updated every arm's estimate at once. Such a design would also let a DKW-style simultaneous bound apply without the union-bound penalty over `K`, tightening the confidence widths and likely reducing the sample budget needed per input. We did not implement this here; it is a direct route to reducing the compute cost documented in Results.

This also gives Discussion something concrete to point to when explaining the 80-90 calls/input figure (see item 6).

---

## 5. Concrete divergence case: identify-or-escalate vs. Even-Dar et al. (2006)

Synthetic worked example — doesn't depend on a finished run. Swap in a real case from run data later if one shows the same structure; the mechanism illustrated is identical either way, so this can ship in a first draft without waiting on HateModerate scaling.

> To make the departure from standard action elimination concrete: consider a node with budget `B` exhausted while two arms remain active, `safe` and `escalate`, with empirical estimates `p̂(safe) = 0.55` and `p̂(escalate) = 0.45`, and confidence widths wide enough that neither dominates (`p̂(safe) − w(safe) < p̂(escalate) + w(escalate)`) — the elimination condition in Algorithm 1, line 8, is never satisfied for either arm. Standard action elimination (Even-Dar et al., 2006) forces a label at budget exhaustion by returning `arg max_c p̂(c)` — here, `safe` — despite the two arms being statistically indistinguishable at the achieved sample size. Our identify-or-escalate variant instead returns `escalate` whenever more than one arm survives budget exhaustion, regardless of which has the higher point estimate. The two algorithms produce different terminal outputs on the identical sequence of observations whenever budget is exhausted with more than one arm still active and neither dominant — which is exactly the condition the paper's B=10/B=50 degenerate-budget results (Table 3) show occurring at scale.

Once a real run has cases like this, swap in the actual `p̂` values and node/example identifiers — but the argument doesn't need to wait for that.

---

## 6. Discussion — tying compute cost to the pulling design

> The 80-90 average pulls per input observed in Results is a direct consequence of the per-arm-independent pulling design (Methods): each of the `K` arms is estimated from its own re-query stream, so a single LLM response only informs one arm's tally even though it reveals the full categorical outcome. A design using one shared draw per round to update all arms simultaneously (see Limitations) would use each call's information fully and would likely reduce this cost substantially — a direction we leave to future work rather than a fundamental limit of adaptive sampling itself.

---

## Still open (not drafted here)

- **Item 3** — Sample Complexity `n*` formula: needs your (a) rederive vs (b) relabel decision before drafting.
- **Item 4** — O(log T) regret narrowing: mechanically straightforward once you confirm Option A (per-episode-only guarantee) is still the call from last session; I can draft it on request, held back for now since it's a bigger structural edit to Appendix C than a drop-in paragraph.
