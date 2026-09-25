"""
IID / repeated-sampling diagnostic audit.

Read-only over existing files. No LLM calls, no manuscript edit, no
experiment/raw file modified.

CENTRAL FINDING FROM SCHEMA INSPECTION (this determines everything below):
individual per-call outputs are NOT persisted anywhere in this codebase's
outputs, for either the majority-vote (`run_node_mv`) or successive-
elimination (`ucb_node`) code paths.

  - `run_node_mv` (run_all_datasets.py) builds a `votes` list (one parsed
    label per call) purely in memory, uses it once to build the
    Adjudicator's context string, and returns it in a dict -- but `votes`
    is never one of the ~26 columns `run_condition` writes to raw.csv (see
    the COLUMNS list / the row-building block in run_condition). It is
    discarded after the process that generated it exits.
  - `ucb_node` fires one identical request per currently-active arm each
    round (confirmed at run_all_datasets.py:1162-1173: "a 'pull for arm X'
    here is an IDENTICAL request regardless of X"), parses each response,
    and increments `arm_wins`/`arm_pulls` counters -- but the per-pull
    PARSED LABEL is never stored. What IS persisted per round, inside
    `arm_elimination_trace`, is only `{round, active_before, eliminated,
    active_after}` -- the surviving/eliminated ARM SET at each round, not
    which label each individual call produced.
  - `arm_estimates_json` (when present) is a single FINAL snapshot of each
    arm's cumulative win-rate at termination, not a per-round or per-call
    value.
  - No separate raw-response log file exists in any results directory
    (confirmed by directory search); `sanity=False` was used for every real
    run, so even the verbose per-call print statements in the code were
    never emitted to any log during actual data generation.

CONSEQUENCE: Analyses A, B, C, and D as specified in the task all require an
ordered sequence of individual call outcomes Y_1, Y_2, ..., Y_T per example.
That sequence does not exist in any stored artifact. Fabricating one (e.g.
by treating the final arm_estimates_json proportions as if they were a time
series, or by inventing a plausible-looking sequence consistent with the
final win-rate) would violate the explicit instruction not to invent or
reconstruct missing data, and would not be a diagnostic of the actual data
-- it would be a diagnostic of a fabrication. This script therefore does
NOT attempt A/B/C/D from invented per-call sequences. It documents the
finding, and produces the one analysis that IS honestly answerable from
existing artifacts (Analysis E: SE-resolution-status vs. the round-level
aggregates that genuinely are stored: total_pulls, elimination round).
"""

import os
import csv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ANALYSIS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

PAIRS = [
    ("llama3.1_8b", "hatemoderate"),
    ("llama3.1_8b", "aegis"),
    ("qwen2.5_7b", "hatemoderate"),
    ("qwen2.5_7b", "aegis"),
    ("mistral_7b", "hatemoderate"),
]


def loadFeatureRows():
    with open(os.path.join(ANALYSIS_DIR, "seTrajectoryFeatures.csv")) as f:
        return list(csv.DictReader(f))


def analysisE(rows):
    """The one honestly-answerable piece: does round-level convergence speed
    (total_pulls, eliminationRound -- both genuinely stored) differ between
    resolved and escalated examples? This is NOT a per-call consistency
    analysis and is NOT presented as one; it is the closest legitimate
    proxy available, and is explicitly NOT used as a predictor of IID
    behavior (per instruction) -- it simply reports what these aggregate,
    already-stored quantities look like for the two groups."""
    outRows = []
    for model, dataset in PAIRS:
        for budget in [75, 100, 124]:
            subset = [r for r in rows if r["model"] == model and r["dataset"] == dataset and r["budget"] == str(budget)]
            resolved = [r for r in subset if r["resolved"] == "True"]
            escalated = [r for r in subset if r["resolved"] == "False"]

            def summarize(group, field):
                vals = [float(r[field]) for r in group if r[field]]
                return (len(vals), sum(vals) / len(vals) if vals else None)

            nRes, meanPullsRes = summarize(resolved, "totalPulls")
            nEsc, meanPullsEsc = summarize(escalated, "totalPulls")
            _, meanRoundRes = summarize(resolved, "eliminationRound")
            _, meanRoundEsc = summarize(escalated, "eliminationRound")

            outRows.append({
                "model": model, "dataset": dataset, "budget": budget,
                "nResolved": nRes, "meanTotalPulls_resolved": meanPullsRes,
                "meanEliminationRound_resolved": meanRoundRes,
                "nEscalated": nEsc, "meanTotalPulls_escalated": meanPullsEsc,
                "meanEliminationRound_escalated": meanRoundEsc,
            })
    return outRows


def writeCsv(path, rows, fieldOrder=None):
    if not rows:
        with open(path, "w") as f:
            f.write("NOT_COMPUTABLE\n")
        return
    fields = fieldOrder or list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def writeNotComputableCsv(path, reason):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["status", "reason"])
        writer.writerow(["NOT_COMPUTABLE", reason])


REASON_TEXT = (
    "Individual per-call outcomes (the Y_1...Y_T sequence for a given "
    "example) are not persisted in any existing raw.csv or log file for "
    "either the majority-vote or successive-elimination code paths -- "
    "confirmed by direct inspection of run_all_datasets.py (run_node_mv's "
    "`votes` list is discarded after use, never written to a CSV column; "
    "ucb_node's arm_elimination_trace records only the per-round surviving/"
    "eliminated ARM SET, never the per-call parsed label; arm_estimates_json "
    "is a single final snapshot, not a time series). Computing this "
    "analysis would require fabricating a per-call sequence, which is "
    "explicitly disallowed. See computeIidAudit.py's module docstring and "
    "iidAudit.md Section 2 for the full evidence trail."
)


def writeMarkdown(analysisERows):
    lines = []
    lines.append("# IID / Repeated-Sampling Diagnostic Audit\n")

    lines.append("## 1. Purpose\n")
    lines.append(
        "The paper's theoretical analysis models repeated LLM outputs for a "
        "fixed example x as IID draws from a fixed categorical distribution "
        "p_c(x). This audit checks whether the existing repeated-call data "
        "provide obvious diagnostic evidence against that assumption "
        "(stationarity across call position, lack of serial dependence, "
        "stable label frequencies). It is explicitly a diagnostic sweep, "
        "not a formal statistical test of IID, and a failure to find "
        "evidence of dependence does not prove independence.\n"
    )

    lines.append("## 2. Data and representation\n")
    lines.append(
        "**Central finding: individual per-call outcomes are not stored "
        "anywhere in the existing outputs.** This was verified by direct "
        "inspection of `run_all_datasets.py`, not assumed:\n\n"
        "- `run_node_mv` (majority-vote path) builds an in-memory `votes` "
        "list — one parsed label per call — used once to build the "
        "Adjudicator's context string (`build_adjudicator_context_mv`), "
        "then discarded. `votes` is not among the ~26 columns "
        "`run_condition` writes to `raw.csv`.\n"
        "- `ucb_node` (successive-elimination path) fires one **identical** "
        "request per currently-active arm each round (confirmed at "
        "`run_all_datasets.py:1162-1173`, comment: \"a 'pull for arm X' "
        "here is an IDENTICAL request regardless of X\"), and increments "
        "per-arm win/pull counters — but the parsed label of each "
        "individual pull is never stored. The persisted "
        "`arm_elimination_trace` records only `{round, active_before, "
        "eliminated, active_after}` per round: the surviving/eliminated "
        "ARM SET, not which label each call produced.\n"
        "- `arm_estimates_json` (where present) is a single FINAL snapshot "
        "of each arm's cumulative win-rate at the moment of termination — "
        "not a per-round or per-call value, so it cannot be sliced into "
        "\"early\" vs \"late\" observations.\n"
        "- No separate raw-response log exists in any results directory "
        "(directory search performed, none found), and every real run used "
        "`sanity=False`, so even the verbose per-call print statements "
        "present in the code were never written to any log during actual "
        "data generation.\n\n"
        "**Consequence**: Analyses A (call-position label frequencies), B "
        "(adjacent-call agreement), C (early-vs-late distributions), and D "
        "(order sensitivity) as specified all require an ordered sequence "
        "of individual call outcomes per example. That sequence does not "
        "exist in any stored artifact, for any example, under any "
        "condition. Reconstructing one would mean fabricating data, which "
        "the task instructions explicitly forbid (\"do not invent or "
        "reconstruct missing data\"). **These four analyses are therefore "
        "reported as not computable from the existing data, rather than "
        "approximated.**\n"
    )

    lines.append("## 3. Call-position distributions\n")
    lines.append(
        "**Not computable** — see Section 2. `iidCallPositionTable.csv` "
        "contains a single row documenting this rather than fabricated "
        "frequencies.\n"
    )

    lines.append("## 4. Adjacent-call dependence\n")
    lines.append(
        "**Not computable** — see Section 2. `iidAdjacentAgreement.csv` "
        "contains a single row documenting this. No adjacent-call "
        "agreement statistic, and no reference/null construction, can be "
        "computed without a per-call sequence to compute it from.\n"
    )

    lines.append("## 5. Order sensitivity\n")
    lines.append(
        "**Not computable** — see Section 2. `iidOrderSensitivity.csv` "
        "contains a single row documenting this. Running-modal-label "
        "changes, first-half/second-half modal comparisons, and \"when is "
        "the final label already established\" all require the ordered "
        "per-call sequence that is not stored.\n"
    )

    lines.append("## 6. Relation to SE resolution\n")
    lines.append(
        "Per-call consistency cannot be compared between resolved and "
        "escalated examples for the reason above. What **is** genuinely "
        "stored and comparable is round-level convergence speed: "
        "`total_pulls` (how many individual calls were fired in total "
        "before the node stopped) and the elimination round (which round "
        "number the terminal event occurred in) — both already computed "
        "read-only in `analysis/seTrajectoryFeatures.csv` from a prior "
        "step. Reporting these here as the closest available proxy, "
        "**not** as a per-call consistency measure and **not** as a "
        "predictor of IID behavior (per instruction):\n\n"
    )
    lines.append("| Model | Dataset | Budget | N resolved | Mean pulls (resolved) | Mean elim. round (resolved) | N escalated | Mean pulls (escalated) | Mean elim. round (escalated) |")
    lines.append("| --- " * 9 + "|")
    for r in analysisERows:
        def fmt(v):
            return f"{v:.1f}" if v is not None else "n/a"
        lines.append(f"| {r['model']} | {r['dataset']} | B{r['budget']} | {r['nResolved']} | "
                      f"{fmt(r['meanTotalPulls_resolved'])} | {fmt(r['meanEliminationRound_resolved'])} | "
                      f"{r['nEscalated']} | {fmt(r['meanTotalPulls_escalated'])} | {fmt(r['meanEliminationRound_escalated'])} |")
    lines.append(
        "\nEscalated examples consistently use far more total pulls than "
        "resolved ones across every condition (consistent with escalation "
        "meaning \"ran to budget exhaustion without converging\") — this is "
        "a property of the stopping rule's mechanics, not evidence about "
        "whether the underlying repeated calls are IID.\n"
    )

    lines.append("## 7. Interpretation and limitations\n")
    lines.append(
        "- **These are diagnostics, not an IID test** — and in this case, "
        "the diagnostics could not even be attempted for four of the five "
        "requested analyses, because the prerequisite data (per-call "
        "outcome sequences) do not exist in any stored artifact.\n"
        "- **Independence is an assumption of the theoretical analysis**, "
        "not something this audit can verify or falsify with the data "
        "currently available.\n"
        "- **Absence of detectable dependence does not prove IID** — and "
        "here we do not even have absence-of-detectable-dependence; we "
        "have an inability to test for it at all with existing artifacts.\n"
        "- **Any future dependence check would require re-instrumenting "
        "the pipeline** to persist per-call outcomes (e.g., adding the "
        "`votes` list, or per-pull labels in `ucb_node`, to the CSV schema) "
        "and re-running — which was explicitly out of scope for this audit "
        "(\"do NOT run any new LLM inference\"). Until that instrumentation "
        "exists, the theoretical guarantee's IID assumption should be "
        "understood as an assumption the existing experiments do not "
        "directly test, not as one that has been checked and passed.\n"
    )

    with open(os.path.join(OUT_DIR, "iidAudit.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    rows = loadFeatureRows()

    writeNotComputableCsv(os.path.join(OUT_DIR, "iidCallPositionTable.csv"), REASON_TEXT)
    writeNotComputableCsv(os.path.join(OUT_DIR, "iidAdjacentAgreement.csv"), REASON_TEXT)
    writeNotComputableCsv(os.path.join(OUT_DIR, "iidOrderSensitivity.csv"), REASON_TEXT)

    eRows = analysisE(rows)
    with open(os.path.join(OUT_DIR, "iidRelationToSE.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(eRows[0].keys()))
        writer.writeheader()
        writer.writerows(eRows)

    writeMarkdown(eRows)

    print("=== KEY RESULTS ===")
    print("1. Number of examples with usable repeated-call SEQUENCES (ordered per-call outcomes): 0")
    print("   -- individual per-call outcomes are not persisted anywhere in the existing data")
    print("   (see iidAudit.md Section 2 for the full code-level evidence trail).")
    print("2. Calls per sequence / observed range: NOT APPLICABLE -- no sequences exist to measure.")
    print("   (total_pulls, the COUNT of calls per example, IS stored and ranges widely -- "
          f"resolved examples: mean {sum(r['meanTotalPulls_resolved'] for r in eRows if r['meanTotalPulls_resolved'])/len(eRows):.1f} pulls; "
          f"escalated: mean {sum(r['meanTotalPulls_escalated'] for r in eRows if r['meanTotalPulls_escalated'])/len(eRows):.1f} pulls -- "
          "but this is a total count, not an ordered sequence of outcomes.)")
    print("3. Early-vs-late distribution differences: NOT COMPUTABLE (no per-call labels stored).")
    print("4. Adjacent-call agreement vs independence reference: NOT COMPUTABLE (no per-call labels stored).")
    print("5. Fraction of sequences with running-majority changes: NOT COMPUTABLE (no per-call labels stored).")
    print("6. Fraction where first-half/second-half modal labels differ: NOT COMPUTABLE (no per-call labels stored).")
    print("7. Major schema/data-quality limitation: the raw pipeline never persists individual")
    print("   per-call outcomes for either the majority-vote or successive-elimination code paths --")
    print("   only round-level arm-survival sets (UCB) and a final cumulative win-rate snapshot are")
    print("   stored. This is a genuine gap for auditing the IID assumption from existing data alone;")
    print("   closing it would require adding per-call logging to the pipeline and re-running,")
    print("   which is out of scope here (no new LLM inference was permitted for this task).")
