"""
ICLR_2027/analyze.py
---------------------
Computes accuracy / FNR / FPR / escalation rate / avg pulls per input /
total cost for all four conditions on HateModerate, from the raw per-example
CSVs written by run.py. Writes summary.csv and summary.md.

Conditions: graph_mv, graph_ucb (both on MODEL), graph_mv_isocompute (MODEL,
matched call budget to graph_ucb -- answers DgKN/57oy's iso-compute question),
single_reasoning (REASONING_MODEL, one call/input -- answers vSZ4's "why not
reason harder once" question). See run.py's module docstring for details.

Adjudicator can terminate in 'escalate' (sent to human review). Matching
germain.py's original convention, those examples are EXCLUDED from
accuracy/FNR/FPR (both numerator and denominator) -- they were never
actually resolved by the system. "Escalation to human" is reported as its
own rate. This is distinct from "escalated past Screener", which just means
the case didn't resolve on the very first node (normal multi-agent triage
behavior, not a failure).

Flags degenerate results explicitly rather than reporting them silently:
  - escalation-to-human rate >= 0.9 (system essentially never decides)
  - fewer than 5 RESOLVED examples in the hate or not_hate ground-truth
    class (FNR/FPR would be statistically meaningless)
  - accuracy == 0 or 1.0 exactly (suspicious for a real eval)
"""

import os
import csv
import ast

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
CONDITIONS = ["graph_mv", "graph_ucb", "graph_mv_isocompute", "single_reasoning"]

DEGENERATE_ESCALATION_THRESHOLD = 0.90
MIN_CLASS_N = 5


def load(condition: str) -> list[dict]:
    path = os.path.join(BASE_DIR, "results", condition, "raw.csv")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def compute_metrics(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return None

    is_escalated    = lambda r: r.get("escalated_to_human") == "1" or r["final_label"] == "escalate"
    escalated_human = [r for r in rows if is_escalated(r)]
    resolved        = [r for r in rows if not is_escalated(r)]

    esc_human_rate = len(escalated_human) / n
    esc_past_screener_rate = sum(1 for r in rows if len(ast.literal_eval(r["nodes_used"])) > 1) / n

    hate    = [r for r in resolved if r["ground_truth"] == "hate"]
    nothate = [r for r in resolved if r["ground_truth"] == "not_hate"]

    acc = sum(int(r["correct"]) for r in resolved) / len(resolved) if resolved else None
    fnr = sum(1 for r in hate    if r["final_label"] != "hate")     / len(hate)    if hate    else None
    fpr = sum(1 for r in nothate if r["final_label"] != "not_hate") / len(nothate) if nothate else None

    avg_pulls = sum(int(r["total_pulls"]) for r in rows) / n
    total_cost = sum(float(r["cost_usd"]) for r in rows)
    total_input_tok  = sum(int(r["total_tokens_input"])  for r in rows)
    total_output_tok = sum(int(r["total_tokens_output"]) for r in rows)

    flags = []
    if esc_human_rate >= DEGENERATE_ESCALATION_THRESHOLD:
        flags.append(f"escalation-to-human rate {esc_human_rate:.0%} >= {DEGENERATE_ESCALATION_THRESHOLD:.0%} — system essentially never decides, treat as degenerate")
    if not resolved:
        flags.append("zero resolved examples — accuracy/FNR/FPR undefined (all examples escalated to human)")
    else:
        if len(hate) < MIN_CLASS_N:
            flags.append(f"only {len(hate)} resolved hate-class examples — FNR not statistically meaningful")
        if len(nothate) < MIN_CLASS_N:
            flags.append(f"only {len(nothate)} resolved not_hate-class examples — FPR not statistically meaningful")
        if acc in (0.0, 1.0):
            flags.append(f"accuracy is exactly {acc:.0%} — verify this isn't a labeling/parsing bug")
    if n < 30:
        flags.append(f"n={n} is well below the target sample — results likely from a cost-cap early stop or --sanity run")

    return {
        "n": n, "n_resolved": len(resolved), "accuracy": acc, "fnr": fnr, "fpr": fpr,
        "escalation_rate_to_human": esc_human_rate,
        "escalation_rate_past_screener": esc_past_screener_rate,
        "avg_pulls_per_input": avg_pulls,
        "total_cost_usd": total_cost, "total_input_tokens": total_input_tok,
        "total_output_tokens": total_output_tok, "flags": flags,
    }


def fmt(x, pct=True):
    if x is None:
        return "n/a"
    return f"{x:.1%}" if pct else f"{x:.3f}"


def main():
    metrics = {}
    for cond in CONDITIONS:
        rows = load(cond)
        m = compute_metrics(rows)
        if m is None:
            print(f"MISSING: {cond} (no results file — run run.py first)")
            continue
        metrics[cond] = m

    if not metrics:
        print("No results found. Run run.py first.")
        return

    print("\n" + "=" * 100)
    print("MV vs UCB vs iso-compute MV vs single-call reasoning — HateModerate")
    print("(Accuracy/FNR/FPR computed over resolved examples only — human-escalated examples excluded)")
    print("=" * 100)
    header = (f"{'Condition':<20} {'n':>4} {'n_res':>6} {'Acc':>7} {'FNR':>7} {'FPR':>7} "
              f"{'Esc→human':>10} {'Esc>Scrnr':>10} {'Avg pulls':>10} {'Cost ($)':>9}")
    print(header)
    print("-" * len(header))
    for cond in CONDITIONS:
        if cond not in metrics:
            continue
        m = metrics[cond]
        print(f"{cond:<20} {m['n']:>4} {m['n_resolved']:>6} {fmt(m['accuracy']):>7} {fmt(m['fnr']):>7} {fmt(m['fpr']):>7} "
              f"{fmt(m['escalation_rate_to_human']):>10} {fmt(m['escalation_rate_past_screener']):>10} "
              f"{m['avg_pulls_per_input']:>10.2f} {m['total_cost_usd']:>9.4f}")

    print()
    for cond in CONDITIONS:
        if cond not in metrics:
            continue
        for flag in metrics[cond]["flags"]:
            print(f"  [FLAG:{cond}] {flag}")

    # ── Write CSV ────────────────────────────────────────────────────────────
    csv_path = os.path.join(BASE_DIR, "results", "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "n", "n_resolved", "accuracy", "fnr", "fpr",
                          "escalation_rate_to_human", "escalation_rate_past_screener",
                          "avg_pulls_per_input", "total_cost_usd", "total_input_tokens",
                          "total_output_tokens", "flags"])
        for cond in CONDITIONS:
            if cond not in metrics:
                continue
            m = metrics[cond]
            writer.writerow([cond, m["n"], m["n_resolved"], m["accuracy"], m["fnr"], m["fpr"],
                              m["escalation_rate_to_human"], m["escalation_rate_past_screener"],
                              round(m["avg_pulls_per_input"], 3), round(m["total_cost_usd"], 4),
                              m["total_input_tokens"], m["total_output_tokens"],
                              " | ".join(m["flags"])])
    print(f"\nSaved {csv_path}")

    # ── Write Markdown ───────────────────────────────────────────────────────
    md_path = os.path.join(BASE_DIR, "results", "summary.md")
    with open(md_path, "w") as f:
        f.write("# MV vs UCB vs iso-compute MV vs single-call reasoning — HateModerate\n\n")
        f.write("Screener → Analyst → Adjudicator graph (single_reasoning bypasses the graph — one call, "
                "no routing), fixed 41-policy Facebook taxonomy prompt.\n\n")
        f.write("Accuracy/FNR/FPR computed over resolved examples only (human-escalated examples excluded, "
                "matching the germain.py convention of `correct = None` for escalations).\n\n")
        f.write("| Condition | n | n resolved | Accuracy | FNR | FPR | Esc. → human | Esc. past Screener | Avg pulls/input | Cost ($) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for cond in CONDITIONS:
            if cond not in metrics:
                continue
            m = metrics[cond]
            f.write(f"| {cond} | {m['n']} | {m['n_resolved']} | {fmt(m['accuracy'])} | {fmt(m['fnr'])} | {fmt(m['fpr'])} | "
                     f"{fmt(m['escalation_rate_to_human'])} | {fmt(m['escalation_rate_past_screener'])} | "
                     f"{m['avg_pulls_per_input']:.2f} | {m['total_cost_usd']:.4f} |\n")
        any_flags = any(metrics[c]["flags"] for c in metrics)
        if any_flags:
            f.write("\n## Flags\n\n")
            for cond in CONDITIONS:
                if cond not in metrics:
                    continue
                for flag in metrics[cond]["flags"]:
                    f.write(f"- **{cond}**: {flag}\n")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
