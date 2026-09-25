"""
ICLR_2027/run.py
-----------------
Majority Vote vs UCB successive elimination on HateModerate, through the
Screener -> Analyst -> Adjudicator graph.

Graph and node prompts are copied verbatim from AAAI_2027/exp8_graph/run.py
(graph_mv condition) for methodological continuity with that prior run.
The HateModerate taxonomy itself (hm.SYSTEM_PROMPT) is imported, never
re-typed, so it cannot drift between conditions or across runs.

Conditions:
  graph_mv             : n=5 votes/node, majority vote, budget-unconstrained
                          (worst case ~900 calls total, ~$3-4)
  graph_ucb            : Hoeffding-bound successive elimination per node,
                          B=124/node, hard cost cap at UCB_COST_CAP_USD
                          (worst case ~22k calls -- the cap stops the run
                          early and reports back rather than spending to
                          that worst case)
  graph_mv_isocompute  : majority vote at n=30/node -- same algorithm as
                          graph_mv, matched to roughly graph_ucb's per-node
                          call budget. Answers reviewers DgKN/57oy: isolates
                          whether graph_ucb's gain comes from the elimination
                          algorithm or just from spending more total calls.
  single_reasoning     : one call per input, no DAG routing, using
                          REASONING_MODEL (o3) instead of MODEL. Answers
                          reviewer vSZ4: why not one call with a large
                          reasoning budget instead of many small calls?
                          Deliberately the opposite kind of model from the
                          other three conditions -- see MODEL/REASONING_MODEL
                          comments below.

UCB algorithm ported from navigator_experiments/experiment.py (identical
file: AAAI_2027/germain.py) `ucb_node()`. All three nodes share the same
3-arm space {hate, not_hate, escalate}, matching the original germain.py
design where the final node's 'escalate' is a genuine terminal state --
here, sent to human review -- rather than a forced binary choice. On
budget exhaustion at any node the algorithm returns 'escalate', deferring
to the next agent (or, at Adjudicator, to a human), matching the original
algorithm's semantics.

Accuracy/FNR/FPR (computed in analyze.py) exclude examples that terminate
in human escalation, matching germain.py's original convention
(`correct = None for escalations`). Escalation rate is reported as its
own metric: fraction of inputs sent to human review (Adjudicator escalates),
tracked separately from fraction that merely pass Screener (agent handoff).

Adjudicator only ever runs when Screener AND Analyst both landed on
'escalate' -- so it is shown their underlying vote distributions (MV) or
empirical arm estimates (UCB), not just their collapsed 'escalate' labels,
via build_adjudicator_context_mv()/_ucb(). Without this, Adjudicator would
be reasoning blind on every case, unable to tell a split vote leaning
toward one label apart from a genuinely unanimous "nobody knows."

Usage:
    export HOPGPT_API_KEY=...       # preferred if set (routes to https://api.ai.jh.edu)
    # or: export OPENAI_API_KEY=... # falls back to direct OpenAI if HOPGPT_API_KEY unset
    python run.py --sanity          # 3 examples, both conditions, verbose
    python run.py                   # full run: 60 balanced examples
"""

import os
import sys
import csv
import json
import math
import time
import argparse
import collections

import openai

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MLHC_DIR = os.path.dirname(BASE_DIR)
AAAI_DIR = os.path.join(MLHC_DIR, "AAAI_2027")
sys.path.insert(0, AAAI_DIR)

import hatemoderate as hm  # fixed 41-policy prompt + loader -- never edited here

# ── Config ────────────────────────────────────────────────────────────────────

MODEL          = "gpt-5.6-luna"  # main conditions: cheap, non-reasoning, confirmed 0 reasoning
                                  # tokens at this project's 10-token output budget (see
                                  # ICLR_2027/REVISION_PLAN.md model-selection discussion).
                                  # Deliberately non-reasoning: the successive-elimination
                                  # algorithm needs real call-to-call output variability to
                                  # have something to eliminate between.
REASONING_MODEL = "o3"           # single_reasoning baseline only: deliberately the opposite
                                  # kind of model, to fairly represent "one call, reason hard"
                                  # as the alternative strategy reviewer vSZ4 asked about.
                                  # o3 does NOT accept a custom temperature (only its default,
                                  # 1, is supported) -- see _call_reasoning().
TEMPERATURE    = 0.7
N_EXAMPLES     = 60          # balanced: 30 hate / 30 not_hate
SEED           = 42
N_MV           = 5           # majority-vote calls per node
N_MV_ISOCOMPUTE = 30         # iso-compute MV: matches UCB_BUDGET's order of magnitude per
                              # node (124/K=3 arms ~= 41 pulls/arm worst case; 30 is a
                              # starting approximation -- refine once real UCB average-pulls
                              # data exists for this domain, per DgKN/57oy's iso-compute ask).
UCB_BUDGET     = 124         # max calls per node for UCB
UCB_DELTA      = 0.05

UCB_COST_CAP_USD       = 10.00  # hard stop for graph_ucb
ISOCOMPUTE_COST_CAP_USD = 10.00  # hard stop for graph_mv_isocompute
REASONING_COST_CAP_USD  = 5.00   # hard stop for single_reasoning; o3 reasoning tokens can
                                  # spike (observed 5,000-20,000 on hard prompts in the wild,
                                  # though our own test showed ~64 on a realistic classification
                                  # prompt) -- cap exists as a safety net, not an expected cost.

# Approximate pricing ($ per 1M tokens: input, output). Not fetched live --
# verify against provider pricing pages before treating cost totals as
# authoritative. gpt-4o kept for reference/fallback only.
PRICE = {
    "gpt-4o":       (2.50, 10.00),
    "gpt-5.6-luna": (1.00, 6.00),
    "o3":           (2.00, 8.00),
}

GRAPH = ["Screener", "Analyst", "Adjudicator"]

# ── Node prompts (verbatim from AAAI_2027/exp8_graph/run.py) ──────────────────

SCREENER_SYSTEM = (
    hm.SYSTEM_PROMPT + "\n\n"
    "You are the SCREENER node — the first reviewer in a three-stage pipeline. "
    "Your job is to handle clear-cut cases quickly. "
    "Label obvious hate speech as 'hate'. Label obvious non-violations as 'not_hate'. "
    "If the case is genuinely ambiguous and requires deeper policy analysis, "
    "label it 'escalate' to pass it to the next reviewer. "
    "Only escalate when you are genuinely uncertain — do not escalate as a default."
)

ANALYST_SYSTEM = (
    hm.SYSTEM_PROMPT + "\n\n"
    "You are the ANALYST node — the second reviewer in a three-stage pipeline. "
    "The previous reviewer was uncertain about this content. "
    "Your job is to conduct a deeper analysis: identify which specific policy tier "
    "applies, consider context and intent, and make a more informed determination. "
    "Label 'hate' or 'not_hate' if you can now make a confident call. "
    "Label 'escalate' only if the case still requires final adjudication."
)

SINGLE_REASONING_SYSTEM = (
    hm.SYSTEM_PROMPT + "\n\n"
    "You are a content moderation reviewer with unlimited time to reason carefully "
    "about this single case before responding. There is no second reviewer after you — "
    "this is your one opportunity to get it right. "
    "Label 'hate' or 'not_hate' if you can reach a confident, defensible determination. "
    "Label 'escalate' only if, after careful reasoning, the case remains genuinely "
    "ambiguous and requires human judgment — not as a default."
)

ADJUDICATOR_SYSTEM = (
    hm.SYSTEM_PROMPT + "\n\n"
    "You are the ADJUDICATOR node — the final AI reviewer in a three-stage pipeline. "
    "Two previous reviewers were uncertain about this content. "
    "Label 'hate' or 'not_hate' if you can now make a confident, defensible call. "
    "If the case remains genuinely ambiguous even after this deeper review, label it "
    "'escalate' to send it to a human moderator for final review. "
    "Only escalate when you are genuinely uncertain — do not escalate as a default."
)

NODE_SYSTEMS = {
    "Screener":    SCREENER_SYSTEM,
    "Analyst":     ANALYST_SYSTEM,
    "Adjudicator": ADJUDICATOR_SYSTEM,
}

# All three nodes share the same 3-arm space. Adjudicator's 'escalate' is now
# a genuine terminal state (sent to human review), not agent-to-agent handoff.
NODE_ARMS = {
    "Screener":    ["hate", "not_hate", "escalate"],
    "Analyst":     ["hate", "not_hate", "escalate"],
    "Adjudicator": ["hate", "not_hate", "escalate"],
}

MV_USER = "Classify the following text. Respond with exactly one word: hate, not_hate, or escalate.\n\n{text}"
UCB_USER = "Classify the following text. Respond with exactly one word: {arms}.\n\n{text}"


def build_adjudicator_context_mv(screener_r: dict, analyst_r: dict) -> str:
    """
    Adjudicator only ever runs when both Screener and Analyst's majority vote
    was 'escalate' (the graph breaks on any non-escalate label) -- so showing
    Adjudicator just their collapsed labels is vacuous, it already knows both
    said 'escalate'. Show the underlying vote distributions instead: a 3-2
    split leaning 'hate' looks very different from a unanimous 'escalate'.
    """
    def fmt(name, r):
        tally = collections.Counter(r["votes"])
        tally_str = ", ".join(f"{k}={v}" for k, v in tally.most_common())
        return f"{name} ({len(r['votes'])} independent votes): {r['votes']} → tally: {tally_str}"

    return (
        "Two previous automated reviewers assessed this content and both were uncertain "
        "enough to escalate. Here is what each of them actually voted, before their votes "
        "were collapsed to 'escalate':\n\n"
        f"{fmt('Screener', screener_r)}\n"
        f"{fmt('Analyst', analyst_r)}\n\n"
        "Use this to inform your own independent judgment — a split vote leaning toward "
        "one label is different from a unanimous 'escalate'. You are not bound by their "
        "votes; make your own determination."
    )


def build_adjudicator_context_ucb(screener_r: dict, analyst_r: dict) -> str:
    """UCB analogue of build_adjudicator_context_mv: shows empirical per-arm win-rate
    estimates at the point Screener/Analyst terminated, instead of just their collapsed
    'escalate' label."""
    def fmt(name, r):
        parts = [f"{arm}={est:.2f}" if est is not None else f"{arm}=?"
                 for arm, est in r["arm_estimates"].items()]
        return f"{name} ({r['pulls']} samples): estimated probabilities — " + ", ".join(parts)

    return (
        "Two previous automated reviewers assessed this content using adaptive sampling "
        "and both terminated on 'escalate' (budget exhausted or arms never separated). "
        "Here are their empirical probability estimates for each label:\n\n"
        f"{fmt('Screener', screener_r)}\n"
        f"{fmt('Analyst', analyst_r)}\n\n"
        "Use this to inform your own independent judgment — estimates leaning toward one "
        "label are different from genuinely flat, uninformative estimates. You are not "
        "bound by their estimates; make your own determination."
    )

# ── Cost tracking ────────────────────────────────────────────────────────────

_running_cost_usd = 0.0


def estimate_cost(model: str, in_tok: int, out_tok: int) -> float:
    p_in, p_out = PRICE[model]
    return (in_tok * p_in + out_tok * p_out) / 1_000_000


# ── LLM call ─────────────────────────────────────────────────────────────────

client = None


def _supports_custom_temperature(model: str) -> bool:
    """GPT-5-family models (confirmed: gpt-5, gpt-5.6-luna/terra/sol) reject any
    temperature other than their default (1) via this gateway -- empirically
    confirmed, not documented in advance. gpt-4o/4.1 and claude-* are assumed
    to support custom temperature (matches original paper's claude-sonnet-4-6
    setup); verify if adding a new model family here."""
    return not (model.startswith("gpt-5") or model.startswith("o3") or model.startswith("o4"))


def _call(system: str, user: str, max_tokens: int = 10) -> tuple[str, int, int]:
    kwargs = dict(
        model=MODEL,
        max_completion_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    if _supports_custom_temperature(MODEL):
        kwargs["temperature"] = TEMPERATURE
    for attempt in range(10):
        try:
            resp = client.chat.completions.create(**kwargs)
            text    = resp.choices[0].message.content.strip()
            in_tok  = resp.usage.prompt_tokens
            out_tok = resp.usage.completion_tokens
            return text, in_tok, out_tok
        except Exception as exc:
            wait = min(5 * (2 ** attempt), 120)
            print(f"  [retry {attempt+1}] {exc} — waiting {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("API failed after 10 retries")


def _call_reasoning(system: str, user: str, max_tokens: int = 1000) -> tuple[str, int, int]:
    """Separate from _call(): o3 (and reasoning models generally) reject a custom
    temperature -- only the API default (1) is supported, confirmed empirically.
    Also uses a much larger token budget than _call()'s 10-token default, since
    reasoning tokens are billed against the same completion budget and would
    otherwise leave no room for the visible answer."""
    for attempt in range(10):
        try:
            resp = client.chat.completions.create(
                model=REASONING_MODEL,
                max_completion_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            text_out = resp.choices[0].message.content.strip()
            in_tok   = resp.usage.prompt_tokens
            out_tok  = resp.usage.completion_tokens
            return text_out, in_tok, out_tok
        except Exception as exc:
            wait = min(5 * (2 ** attempt), 120)
            print(f"  [retry {attempt+1}] {exc} — waiting {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("Reasoning API failed after 10 retries")


def _parse_label(text: str, valid: set) -> str:
    t = text.strip().lower().replace(" ", "_").strip(".,;:")
    if t in valid:
        return t
    for word in text.lower().split():
        w = word.strip(".,;:")
        if w in valid:
            return w
    return "not_hate" if "not_hate" in valid else next(iter(valid))


# ── Majority vote ────────────────────────────────────────────────────────────

def run_node_mv(node: str, text: str, sanity: bool = False, context: str | None = None,
                 n_mv: int = N_MV) -> dict:
    """context, if given, is prepended to the user turn (used only for
    Adjudicator, to show it Screener/Analyst's vote distributions).
    n_mv, if given, overrides N_MV -- used by graph_mv_isocompute to match
    graph_ucb's per-node call budget instead of the default 5 votes/node."""
    system = NODE_SYSTEMS[node]
    valid  = set(NODE_ARMS[node])
    base_user = MV_USER.format(text=text)
    user = f"{context}\n\n{base_user}" if context else base_user
    labels, total_it, total_ot, cost = [], 0, 0, 0.0
    for _ in range(n_mv):
        resp, it, ot = _call(system, user, max_tokens=10)
        label = _parse_label(resp, valid)
        if label not in valid:
            label = "not_hate"
        labels.append(label)
        total_it += it
        total_ot += ot
        cost += estimate_cost(MODEL, it, ot)
    final = collections.Counter(labels).most_common(1)[0][0]
    if sanity:
        print(f"    [MV {node}] votes={labels} → {final!r}  cost=${cost:.4f}")
    return {"label": final, "pulls": n_mv, "input_tokens": total_it,
            "output_tokens": total_ot, "cost_usd": cost, "votes": labels}


def run_graph_mv(text: str, sanity: bool = False, n_mv: int = N_MV) -> dict:
    nodes_used, node_labels = [], []
    node_results = {}
    total_pulls = total_it = total_ot = 0
    cost = 0.0
    for node in GRAPH:
        nodes_used.append(node)
        if node == "Adjudicator":
            # Guaranteed reachable here: Screener and Analyst both voted
            # 'escalate' (the loop breaks below on any other label).
            context = build_adjudicator_context_mv(node_results["Screener"], node_results["Analyst"])
            r = run_node_mv(node, text, sanity, context=context, n_mv=n_mv)
        else:
            r = run_node_mv(node, text, sanity, n_mv=n_mv)
        node_results[node] = r
        node_labels.append(r["label"])
        total_pulls += r["pulls"]
        total_it    += r["input_tokens"]
        total_ot    += r["output_tokens"]
        cost        += r["cost_usd"]
        if r["label"] != "escalate":
            break
    # final may legitimately be 'escalate' if Adjudicator was reached and still
    # uncertain -- that is a terminal "sent to human review" outcome, not an error.
    final = node_labels[-1]
    return {
        "final_label": final, "nodes_used": nodes_used, "node_labels": node_labels,
        "escalation_count": sum(1 for l in node_labels if l == "escalate"),
        "total_pulls": total_pulls, "total_tokens_input": total_it,
        "total_tokens_output": total_ot, "cost_usd": cost,
    }


# ── UCB successive elimination ──────────────────────────────────────────────

def ucb_node(node: str, text: str, budget: int, sanity: bool = False,
             context: str | None = None) -> dict:
    """
    Hoeffding-bound successive elimination over this node's arm space.
    Ported from germain.py's ucb_node(), generalized to accept a per-node
    arm list (all three nodes currently share the same 3-arm space, but the
    function doesn't assume that).

    context, if given, is prepended to the user turn (used only for
    Adjudicator, to show it Screener/Analyst's empirical arm estimates).

    Budget-exhaustion fallback:
      - If 'escalate' is in this node's arm space: return 'escalate' —
        defers to the next agent, or (at Adjudicator) to a human reviewer.
        This matches the original algorithm's semantics.
      - Otherwise: return the arm with the highest empirical win rate so
        far. Not reachable with the current NODE_ARMS config (every node
        includes 'escalate'), kept for correctness if that ever changes.
    """
    system    = NODE_SYSTEMS[node]
    arms      = NODE_ARMS[node]
    K         = len(arms)
    base_user = UCB_USER.format(arms=", ".join(arms), text=text)
    user      = f"{context}\n\n{base_user}" if context else base_user

    active_arms = list(arms)
    arm_wins    = {a: 0 for a in arms}
    arm_pulls   = {a: 0 for a in arms}
    total_pulls = total_it = total_ot = 0
    cost        = 0.0
    trace       = []
    round_num   = 0

    def finalize(label):
        return {
            "label": label, "pulls": total_pulls, "input_tokens": total_it,
            "output_tokens": total_ot, "cost_usd": cost, "trace": trace,
            "arm_estimates": {
                a: (arm_wins[a] / arm_pulls[a]) if arm_pulls[a] else None
                for a in arms
            },
        }

    while len(active_arms) > 1:
        if total_pulls >= budget:
            if "escalate" in arms:
                return finalize("escalate")
            best = max(arms, key=lambda a: (arm_wins[a] / arm_pulls[a]) if arm_pulls[a] else -1)
            return finalize(best)

        round_num += 1
        round_log = {"round": round_num, "active_before": list(active_arms), "eliminated": []}

        for arm in list(active_arms):
            if total_pulls >= budget:
                break
            resp, it, ot = _call(system, user, max_tokens=10)
            label = _parse_label(resp, set(arms))
            arm_pulls[arm] += 1
            total_pulls    += 1
            total_it       += it
            total_ot       += ot
            cost           += estimate_cost(MODEL, it, ot)
            global _running_cost_usd
            _running_cost_usd += estimate_cost(MODEL, it, ot)
            if label == arm:
                arm_wins[arm] += 1

        stats = {}
        for arm in active_arms:
            T_c = arm_pulls[arm]
            if T_c == 0:
                continue
            est   = arm_wins[arm] / T_c
            inner = math.log(4.0 * K * (T_c ** 2) / UCB_DELTA) / (2.0 * T_c)
            width = math.sqrt(max(0.0, inner))
            stats[arm] = {"est": est, "lower": est - width, "upper": est + width}

        if not stats:
            break

        best_arm   = max(stats, key=lambda a: stats[a]["est"])
        best_lower = stats[best_arm]["lower"]
        new_active = []
        for arm in active_arms:
            if arm not in stats:
                new_active.append(arm)
                continue
            if stats[arm]["upper"] < best_lower:
                round_log["eliminated"].append(arm)
                if sanity:
                    print(f"      [UCB {node}] eliminated {arm!r} (upper={stats[arm]['upper']:.3f} < best_lower={best_lower:.3f})")
            else:
                new_active.append(arm)
        active_arms = new_active
        round_log["active_after"] = list(active_arms)
        trace.append(round_log)

        if sanity:
            print(f"    [UCB {node}] round={round_num} T={total_pulls}/{budget} active={active_arms}")

    if len(active_arms) == 1:
        return finalize(active_arms[0])
    if "escalate" in arms:
        return finalize("escalate")
    best = max(arms, key=lambda a: (arm_wins[a] / arm_pulls[a]) if arm_pulls[a] else -1)
    return finalize(best)


def run_graph_ucb(text: str, sanity: bool = False) -> dict:
    nodes_used, node_labels, traces = [], [], []
    node_results = {}
    total_pulls = total_it = total_ot = 0
    cost = 0.0
    for node in GRAPH:
        nodes_used.append(node)
        if node == "Adjudicator":
            # Guaranteed reachable here: Screener and Analyst both terminated
            # on 'escalate' (the loop breaks below on any other label).
            context = build_adjudicator_context_ucb(node_results["Screener"], node_results["Analyst"])
            r = ucb_node(node, text, UCB_BUDGET, sanity, context=context)
        else:
            r = ucb_node(node, text, UCB_BUDGET, sanity)
        node_results[node] = r
        node_labels.append(r["label"])
        traces.append({"node": node, "trace": r["trace"]})
        total_pulls += r["pulls"]
        total_it    += r["input_tokens"]
        total_ot    += r["output_tokens"]
        cost        += r["cost_usd"]
        if r["label"] != "escalate":
            break
    # final may legitimately be 'escalate' if Adjudicator's budget was
    # exhausted or its arms never separated -- terminal "sent to human
    # review" outcome, not an error.
    final = node_labels[-1]
    return {
        "final_label": final, "nodes_used": nodes_used, "node_labels": node_labels,
        "escalation_count": sum(1 for l in node_labels if l == "escalate"),
        "total_pulls": total_pulls, "total_tokens_input": total_it,
        "total_tokens_output": total_ot, "cost_usd": cost,
        "arm_elimination_trace": json.dumps(traces),
    }


# ── Single-call reasoning baseline ──────────────────────────────────────────

def run_single_reasoning(text: str, sanity: bool = False) -> dict:
    """Answers reviewer vSZ4's question: why not one call with a large reasoning
    budget instead of many small calls? No DAG routing -- one call to
    REASONING_MODEL (o3), same 3-way action space as the other conditions.
    Uses _call_reasoning(), not _call(): o3 needs a larger token budget (reasoning
    tokens share the completion budget) and rejects a custom temperature."""
    valid = {"hate", "not_hate", "escalate"}
    user  = MV_USER.format(text=text)
    resp, it, ot = _call_reasoning(SINGLE_REASONING_SYSTEM, user, max_tokens=1000)
    label = _parse_label(resp, valid)
    if label not in valid:
        label = "not_hate"
    cost = estimate_cost(REASONING_MODEL, it, ot)
    if sanity:
        print(f"    [single_reasoning] → {label!r}  cost=${cost:.4f}")
    return {
        "final_label": label, "nodes_used": ["single_reasoning"], "node_labels": [label],
        "escalation_count": int(label == "escalate"),
        "total_pulls": 1, "total_tokens_input": it,
        "total_tokens_output": ot, "cost_usd": cost,
    }


# ── Condition runner ─────────────────────────────────────────────────────────

COLUMNS = [
    "input_id", "condition", "final_label", "ground_truth", "correct",
    "escalated_to_human", "guideline",
    "nodes_used", "node_labels", "escalation_count",
    "total_pulls", "total_tokens_input", "total_tokens_output", "cost_usd",
    "arm_elimination_trace",
]


def run_condition(condition: str, examples: list, cost_cap: float | None, sanity: bool = False):
    results_dir = os.path.join(BASE_DIR, "results", condition)
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "raw.csv")

    completed    = set()
    prior_cost   = 0.0
    write_header = True
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                completed.add(row["input_id"])
                prior_cost += float(row.get("cost_usd", 0) or 0)
        write_header = False

    todo = [e for e in examples if e["id"] not in completed]
    print(f"\n  [{condition}] {len(completed)} done (${prior_cost:.4f} spent), {len(todo)} to run")

    run_cost = prior_cost
    stopped_early = False

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()

        for i, ex in enumerate(todo):
            if cost_cap is not None and run_cost >= cost_cap:
                print(f"  [{condition}] COST CAP HIT (${run_cost:.2f} >= ${cost_cap:.2f}) "
                      f"— stopping early at {len(completed) + i}/{len(examples)} examples")
                stopped_early = True
                break

            gt   = ex["label"]
            text = ex["text"]
            if condition == "graph_ucb":
                result = run_graph_ucb(text, sanity)
            elif condition == "graph_mv_isocompute":
                result = run_graph_mv(text, sanity, n_mv=N_MV_ISOCOMPUTE)
            elif condition == "single_reasoning":
                result = run_single_reasoning(text, sanity)
            else:
                result = run_graph_mv(text, sanity)
            run_cost += result["cost_usd"]

            escalated_to_human = (result["final_label"] == "escalate")
            # Matches germain.py convention: correct = None (blank) for
            # examples that terminate in human escalation -- excluded from
            # accuracy/FNR/FPR in analyze.py, not counted as either right or wrong.
            correct_val = "" if escalated_to_human else int(result["final_label"] == gt)

            row = {
                "input_id": ex["id"], "condition": condition,
                "ground_truth": gt, "correct": correct_val,
                "escalated_to_human": int(escalated_to_human),
                "guideline": ex.get("guideline", ""),
                "nodes_used": str(result["nodes_used"]), "node_labels": str(result["node_labels"]),
                "escalation_count": result["escalation_count"],
                "total_pulls": result["total_pulls"],
                "total_tokens_input": result["total_tokens_input"],
                "total_tokens_output": result["total_tokens_output"],
                "cost_usd": round(result["cost_usd"], 6),
                "arm_elimination_trace": result.get("arm_elimination_trace", ""),
                "final_label": result["final_label"],
            }
            writer.writerow(row)
            f.flush()

            done = len(completed) + i + 1
            if done % 5 == 0 or done == len(examples):
                print(f"  [{condition}] {done}/{len(examples)}  running_cost=${run_cost:.4f}", flush=True)

    return run_cost, stopped_early


def main(sanity: bool):
    global client
    hopgpt_key = os.environ.get("HOPGPT_API_KEY")
    if hopgpt_key:
        client = openai.OpenAI(api_key=hopgpt_key, base_url="https://api.ai.jh.edu")
        print("Using HopGPT (base_url=https://api.ai.jh.edu)")
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            sys.exit("ERROR: neither HOPGPT_API_KEY nor OPENAI_API_KEY set.")
        client = openai.OpenAI(api_key=api_key)
        print("Using direct OpenAI API")

    all_examples = hm.load_hatemoderate()
    n = 3 if sanity else N_EXAMPLES
    examples = hm.balanced_sample(all_examples, n=n, seed=SEED)
    print(f"\n{'Sanity' if sanity else 'Full'} run: {len(examples)} balanced examples")
    print(f"  graph_mv:            model={MODEL}  n={N_MV}/node, uncapped")
    print(f"  graph_ucb:           model={MODEL}  B={UCB_BUDGET}/node, cap=${UCB_COST_CAP_USD:.2f}")
    print(f"  graph_mv_isocompute: model={MODEL}  n={N_MV_ISOCOMPUTE}/node, cap=${ISOCOMPUTE_COST_CAP_USD:.2f}")
    print(f"  single_reasoning:    model={REASONING_MODEL}  1 call/input, cap=${REASONING_COST_CAP_USD:.2f}")

    costs = {}
    stopped = {}
    costs["graph_mv"], stopped["graph_mv"] = run_condition(
        "graph_mv", examples, cost_cap=None, sanity=sanity)
    costs["graph_ucb"], stopped["graph_ucb"] = run_condition(
        "graph_ucb", examples, cost_cap=UCB_COST_CAP_USD, sanity=sanity)
    costs["graph_mv_isocompute"], stopped["graph_mv_isocompute"] = run_condition(
        "graph_mv_isocompute", examples, cost_cap=ISOCOMPUTE_COST_CAP_USD, sanity=sanity)
    costs["single_reasoning"], stopped["single_reasoning"] = run_condition(
        "single_reasoning", examples, cost_cap=REASONING_COST_CAP_USD, sanity=sanity)

    print("\n" + "=" * 60)
    for cond, cost in costs.items():
        print(f"  {cond:<20s} cost=${cost:.4f}")
    print(f"  {'TOTAL':<20s} cost=${sum(costs.values()):.4f}")
    for cond, was_stopped in stopped.items():
        if was_stopped:
            print(f"NOTE: {cond} hit its cost cap before finishing all examples — "
                  f"run again with the same command to resume.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", action="store_true", help="3 examples, verbose")
    args = parser.parse_args()
    main(sanity=args.sanity)
