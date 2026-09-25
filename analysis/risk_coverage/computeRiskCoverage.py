"""
Risk-coverage analysis for the successive-elimination (SE) LLM classification
experiment.

Does NOT modify, rerun, or reinterpret the original experiment. Reads only
the existing per-example raw.csv files already produced by run_all_datasets.py
/ run_extension_n400.py, and computes standard selective-prediction metrics
from them:

    coverage      = resolvedCount / totalCount
    accuracy      = correctResolved / resolvedCount   (over RESOLVED examples only)
    selectiveRisk = 1 - accuracy

A row is "resolved" unless its final_label is one of the four non-terminal
outcomes the pipeline itself never scores: escalate, self_refused,
prompt_injected, parse_failed. This matches the existing schema exactly
(those four are precisely the rows where the pipeline's own `correct` column
is left blank) -- it is not a new definition invented for this analysis.

Outputs (all under analysis/risk_coverage/, nothing elsewhere is touched):
  - riskCoverageSummary.csv / .json   (per model/dataset/method/budget row)
  - riskCoverageComparisons.csv       (SE-vs-MV paired comparisons)
  - riskCoverageAggregate.csv         (mean across the 5 pairs, per method)
  - figures/riskCoverage_<model>_<dataset>.png   (one panel per pair)
  - figures/riskCoverage_allPairs.png            (multi-panel figure)
  - SUMMARY.md                        (findings + assumptions + repro commands)

Run: python3 analysis/risk_coverage/computeRiskCoverage.py
(from the ICLR_2027 project root; no LLM calls, reads existing CSVs only)
"""

import os
import csv
import json
import math
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Locating the existing experiment outputs (read-only)
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULT_ROOTS = ["results_02_n400_from_spark", "results_02"]  # same precedence used throughout this project's own analysis scripts

# non-terminal outcomes -- taken verbatim from run_all_datasets.py's own
# NON_TERMINAL_LABELS set plus parse_failed (the pipeline's 3-strike safety
# net outcome), NOT invented here.
NON_TERMINAL_LABELS = {"escalate", "self_refused", "prompt_injected", "parse_failed"}

PAIRS = [
    ("llama3.1_8b", "hatemoderate"),
    ("llama3.1_8b", "aegis"),
    ("qwen2.5_7b", "hatemoderate"),
    ("qwen2.5_7b", "aegis"),
    ("mistral_7b", "hatemoderate"),
]

# method label -> (condition directory name, budget or None)
METHODS = [
    ("single_call", "single_agent", None),
    ("majority_voting", "graph_mv", None),
    ("successive_elimination", "graph_ucb_B75", 75),
    ("successive_elimination", "graph_ucb_B100", 100),
    ("successive_elimination", "graph_ucb_B124", 124),
]

EXPECTED_TOTAL = 400
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def findBestSourceFile(modelDir, dataset, condition):
    """Same 'pick whichever available copy has more rows' rule used
    throughout this project's own ad-hoc analysis scripts tonight -- not a
    new convention. Returns (path, rowCount) or (None, 0)."""
    bestPath, bestN = None, -1
    for root in RESULT_ROOTS:
        path = os.path.join(PROJECT_ROOT, root, modelDir, dataset, condition, "raw.csv")
        if os.path.isfile(path):
            with open(path) as f:
                n = sum(1 for _ in f) - 1
            if n > bestN:
                bestPath, bestN = path, n
    return bestPath, bestN


def loadRows(modelDir, dataset, condition):
    path, _ = findBestSourceFile(modelDir, dataset, condition)
    if path is None:
        return None, None
    with open(path) as f:
        return list(csv.DictReader(f)), path


def isResolved(row):
    return row["final_label"] not in NON_TERMINAL_LABELS


def wilsonInterval(successes, n, z=1.959963984540054):
    """95% Wilson score interval for a binomial proportion. Appropriate here
    because accuracy-on-resolved is exactly a binomial proportion (correct /
    resolved), and Wilson is well-behaved at small n and at p near 0 or 1
    (both of which occur in this data -- some resolved subsets are small)."""
    if n == 0:
        return (None, None)
    phat = successes / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    halfWidth = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    lo = (centre - halfWidth) / denom
    hi = (centre + halfWidth) / denom
    return (max(0.0, lo), min(1.0, hi))


def computeRiskCoverage(rows, expectedTotal=None):
    """Core reusable function requested: given the raw rows for ONE
    model/dataset/method combination, return the risk-coverage point plus
    every consistency-check quantity needed to audit it.

    Never treats an unresolved/escalated row as an incorrect prediction:
    correctResolved and resolvedCount are both computed strictly over the
    resolved subset."""
    totalCount = len(rows)
    resolved = [r for r in rows if isResolved(r)]
    escalated = [r for r in rows if not isResolved(r)]
    resolvedCount = len(resolved)
    escalatedCount = len(escalated)

    correctResolved = sum(1 for r in resolved if r["final_label"] == r["ground_truth"])

    coverage = resolvedCount / totalCount if totalCount else None
    accuracy = correctResolved / resolvedCount if resolvedCount else None
    selectiveRisk = (1 - accuracy) if accuracy is not None else None
    escalationRate = escalatedCount / totalCount if totalCount else None

    ciLo, ciHi = wilsonInterval(correctResolved, resolvedCount) if resolvedCount else (None, None)

    # --- consistency checks (assertions per the spec; fail loudly, not silently) ---
    assert resolvedCount + escalatedCount == totalCount, \
        f"resolvedCount+escalatedCount != totalCount ({resolvedCount}+{escalatedCount}!={totalCount})"
    assert correctResolved <= resolvedCount, "correctResolved exceeds resolvedCount"
    if expectedTotal is not None:
        assert totalCount == expectedTotal, f"totalCount {totalCount} != expected {expectedTotal}"
    if coverage is not None:
        assert 0.0 <= coverage <= 1.0, f"coverage out of range: {coverage}"
    if accuracy is not None:
        assert 0.0 <= accuracy <= 1.0, f"accuracy out of range: {accuracy}"
        assert abs(selectiveRisk - (1 - accuracy)) < 1e-9, "selectiveRisk != 1-accuracy"
    # no escalated row may have contributed to correctResolved: verify directly,
    # not just by construction, since this is the specific failure mode the
    # spec warns about.
    escalatedIds = {r["input_id"] for r in escalated}
    resolvedIds = {r["input_id"] for r in resolved}
    assert escalatedIds.isdisjoint(resolvedIds), "an id appears in both resolved and escalated sets"

    return {
        "totalCount": totalCount,
        "resolvedCount": resolvedCount,
        "escalatedCount": escalatedCount,
        "correctResolved": correctResolved,
        "coverage": coverage,
        "accuracy": accuracy,
        "selectiveRisk": selectiveRisk,
        "escalationRate": escalationRate,
        "accuracyCiLo": ciLo,
        "accuracyCiHi": ciHi,
    }


def mcnemarPaired(rowsA, rowsB):
    """Paired comparison restricted to examples BOTH methods resolved (the
    only way to preserve pairing when the two methods have different
    resolved subsets, since a wrong/right label doesn't exist for an
    escalated example). Exact binomial McNemar on the discordant pairs.
    This is the same construction already used and reported in the paper's
    own Section 6.1/6.2 McNemar tables -- reused here, not invented fresh."""
    from scipy.stats import binomtest
    aById = {r["input_id"]: r for r in rowsA}
    bById = {r["input_id"]: r for r in rowsB}
    common = set(aById) & set(bById)
    common = {i for i in common if isResolved(aById[i]) and isResolved(bById[i])}
    aOnly = bOnly = bothRight = bothWrong = 0
    for i in common:
        ra, rb = aById[i], bById[i]
        aCorrect = ra["final_label"] == ra["ground_truth"]
        bCorrect = rb["final_label"] == rb["ground_truth"]
        if aCorrect and not bCorrect:
            aOnly += 1
        elif bCorrect and not aCorrect:
            bOnly += 1
        elif aCorrect and bCorrect:
            bothRight += 1
        else:
            bothWrong += 1
    nDiscordant = aOnly + bOnly
    pValue = 1.0 if nDiscordant == 0 else binomtest(min(aOnly, bOnly), nDiscordant, 0.5).pvalue
    return {
        "nJointlyResolved": len(common),
        "aOnlyCorrect": aOnly,
        "bOnlyCorrect": bOnly,
        "pValue": pValue,
    }


def main():
    summaryRows = []
    rawByKey = {}  # (model,dataset,methodLabel,budget) -> rows, for later paired comparisons

    for modelDir, dataset in PAIRS:
        for methodLabel, condition, budget in METHODS:
            rows, sourcePath = loadRows(modelDir, dataset, condition)
            if rows is None:
                raise RuntimeError(
                    f"STOP: no data found for {modelDir}/{dataset}/{condition} in any of {RESULT_ROOTS}. "
                    "Per the task's own instruction, this is reported rather than silently skipped or fabricated."
                )
            metrics = computeRiskCoverage(rows, expectedTotal=EXPECTED_TOTAL)
            rawByKey[(modelDir, dataset, methodLabel, budget)] = rows

            summaryRows.append({
                "model": modelDir,
                "dataset": dataset,
                "method": methodLabel,
                "budget": budget if budget is not None else "",
                "sourceFile": os.path.relpath(sourcePath, PROJECT_ROOT),
                **metrics,
            })

    # every method evaluated on the same underlying 400 examples per pair (check 8)
    for modelDir, dataset in PAIRS:
        idSets = []
        for methodLabel, condition, budget in METHODS:
            rows = rawByKey[(modelDir, dataset, methodLabel, budget)]
            idSets.append(frozenset(r["input_id"] for r in rows))
        firstSet = idSets[0]
        for s in idSets[1:]:
            assert s == firstSet, (
                f"STOP: {modelDir}/{dataset} -- methods are not evaluated on the same 400 "
                f"example ids (symmetric difference size {len(s ^ firstSet)}). Reporting, not fixing."
            )

    # --- write main summary CSV/JSON ---
    fieldOrder = [
        "model", "dataset", "method", "budget", "totalCount", "resolvedCount",
        "coverage", "correctResolved", "accuracy", "selectiveRisk",
        "escalatedCount", "escalationRate", "accuracyCiLo", "accuracyCiHi", "sourceFile",
    ]
    csvPath = os.path.join(OUT_DIR, "riskCoverageSummary.csv")
    with open(csvPath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldOrder)
        writer.writeheader()
        for row in summaryRows:
            writer.writerow({k: row.get(k) for k in fieldOrder})
    with open(os.path.join(OUT_DIR, "riskCoverageSummary.json"), "w") as f:
        json.dump(summaryRows, f, indent=2)

    # --- SE vs majority-voting paired comparisons ---
    comparisonRows = []
    for modelDir, dataset in PAIRS:
        mvRows = rawByKey[(modelDir, dataset, "majority_voting", None)]
        mvMetrics = next(r for r in summaryRows if r["model"] == modelDir and r["dataset"] == dataset and r["method"] == "majority_voting")
        for budget in [75, 100, 124]:
            seRows = rawByKey[(modelDir, dataset, "successive_elimination", budget)]
            seMetrics = next(r for r in summaryRows if r["model"] == modelDir and r["dataset"] == dataset and r["method"] == "successive_elimination" and r["budget"] == budget)
            paired = mcnemarPaired(mvRows, seRows)
            comparisonRows.append({
                "model": modelDir, "dataset": dataset, "comparison": f"SE_B{budget}_vs_majorityVoting",
                "coverageDiff_SEminusMV": seMetrics["coverage"] - mvMetrics["coverage"],
                "accuracyDiff_SEminusMV": seMetrics["accuracy"] - mvMetrics["accuracy"],
                "selectiveRiskDiff_SEminusMV": seMetrics["selectiveRisk"] - mvMetrics["selectiveRisk"],
                "seCoverage": seMetrics["coverage"], "mvCoverage": mvMetrics["coverage"],
                "seRisk": seMetrics["selectiveRisk"], "mvRisk": mvMetrics["selectiveRisk"],
                "pairedMcNemar_nJointlyResolved": paired["nJointlyResolved"],
                "pairedMcNemar_mvOnlyCorrect": paired["aOnlyCorrect"],
                "pairedMcNemar_seOnlyCorrect": paired["bOnlyCorrect"],
                "pairedMcNemar_pValue": paired["pValue"],
            })
    with open(os.path.join(OUT_DIR, "riskCoverageComparisons.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparisonRows[0].keys()))
        writer.writeheader()
        writer.writerows(comparisonRows)

    # --- aggregate across the 5 pairs, per method (clearly labeled as a mean-of-pairs, not pooled examples) ---
    aggregateRows = []
    for methodLabel, condition, budget in METHODS:
        matching = [r for r in summaryRows if r["method"] == methodLabel and r["budget"] == (budget if budget is not None else "")]
        meanCoverage = sum(r["coverage"] for r in matching) / len(matching)
        meanAccuracy = sum(r["accuracy"] for r in matching) / len(matching)
        meanSelectiveRisk = sum(r["selectiveRisk"] for r in matching) / len(matching)
        aggregateRows.append({
            "method": methodLabel, "budget": budget if budget is not None else "",
            "nPairs": len(matching),
            "meanCoverage": meanCoverage, "meanAccuracy": meanAccuracy, "meanSelectiveRisk": meanSelectiveRisk,
            "note": "mean across the 5 model/dataset pairs, NOT pooled examples",
        })
    with open(os.path.join(OUT_DIR, "riskCoverageAggregate.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregateRows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregateRows)

    return summaryRows, comparisonRows, aggregateRows


def makeFigures(summaryRows):
    pairs = PAIRS
    methodStyle = {
        "single_call": dict(marker="s", color="#457b9d", label="Single call"),
        "majority_voting": dict(marker="D", color="#2a9d8f", label="Majority voting"),
    }
    seColor = "#e76f51"
    seMarkers = {75: "o", 100: "^", 124: "v"}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for idx, (modelDir, dataset) in enumerate(pairs):
        ax = axes[idx]
        rowsHere = [r for r in summaryRows if r["model"] == modelDir and r["dataset"] == dataset]
        for r in rowsHere:
            if r["method"] == "successive_elimination":
                ax.scatter(r["coverage"], r["selectiveRisk"], marker=seMarkers[r["budget"]],
                           color=seColor, s=70, zorder=3,
                           label=f"SE B={r['budget']}")
            else:
                st = methodStyle[r["method"]]
                ax.scatter(r["coverage"], r["selectiveRisk"], marker=st["marker"],
                           color=st["color"], s=70, zorder=3, label=st["label"])
        ax.set_title(f"{modelDir} × {dataset}")
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Selective risk (1 - accuracy)")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, max(0.5, max(r["selectiveRisk"] for r in rowsHere) * 1.15))
        ax.grid(alpha=0.3)
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            uniq = dict(zip(labels, handles))
            fig.legend(uniq.values(), uniq.keys(), loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
    axes[-1].axis("off")
    fig.suptitle("Risk-coverage: discrete operating points (no interpolation)", y=1.0)
    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig.savefig(os.path.join(OUT_DIR, "figures", "riskCoverage_allPairs.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "figures", "riskCoverage_allPairs.svg"), bbox_inches="tight")
    plt.close(fig)

    # one figure per pair too, as offered as an alternative in the spec
    for modelDir, dataset in pairs:
        fig, ax = plt.subplots(figsize=(6, 5))
        rowsHere = [r for r in summaryRows if r["model"] == modelDir and r["dataset"] == dataset]
        for r in rowsHere:
            if r["method"] == "successive_elimination":
                ax.scatter(r["coverage"], r["selectiveRisk"], marker=seMarkers[r["budget"]],
                           color=seColor, s=90, zorder=3, label=f"SE B={r['budget']}")
            else:
                st = methodStyle[r["method"]]
                ax.scatter(r["coverage"], r["selectiveRisk"], marker=st["marker"],
                           color=st["color"], s=90, zorder=3, label=st["label"])
        ax.set_title(f"Risk-coverage: {modelDir} × {dataset}")
        ax.set_xlabel("Coverage"); ax.set_ylabel("Selective risk (1 - accuracy)")
        ax.set_xlim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        base = f"riskCoverage_{modelDir}_{dataset}"
        fig.savefig(os.path.join(OUT_DIR, "figures", base + ".png"), dpi=200)
        fig.savefig(os.path.join(OUT_DIR, "figures", base + ".svg"))
        plt.close(fig)


if __name__ == "__main__":
    summaryRows, comparisonRows, aggregateRows = main()
    makeFigures(summaryRows)
    print(f"Wrote {len(summaryRows)} summary rows, {len(comparisonRows)} comparisons, {len(aggregateRows)} aggregate rows.")
    print("Outputs in:", OUT_DIR)
