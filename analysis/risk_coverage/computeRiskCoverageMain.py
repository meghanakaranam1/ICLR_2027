"""
Publication-quality risk-vs-coverage analysis: does SE's reduced coverage
correspond to lower selective risk among the examples it resolves?

Read-only over the already-computed analysis/risk_coverage/*.csv files
(riskCoverageSummary.csv). No new data, no LLM calls, no manuscript edit,
no existing experiment file touched.
"""

import os
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

PAIRS = [
    ("llama3.1_8b", "hatemoderate"),
    ("llama3.1_8b", "aegis"),
    ("qwen2.5_7b", "hatemoderate"),
    ("qwen2.5_7b", "aegis"),
    ("mistral_7b", "hatemoderate"),
]
BUDGETS = [75, 100, 124]


def loadSummary():
    with open(os.path.join(OUT_DIR, "riskCoverageSummary.csv")) as f:
        return list(csv.DictReader(f))


def getRow(summary, model, dataset, method, budget=""):
    return next(r for r in summary
                if r["model"] == model and r["dataset"] == dataset
                and r["method"] == method and r["budget"] == str(budget))


def buildMainTable(summary):
    rows = []
    for model, dataset in PAIRS:
        mv = getRow(summary, model, dataset, "majority_voting")
        mvCoverage, mvAccuracy = float(mv["coverage"]), float(mv["accuracy"])
        mvRisk = 1 - mvAccuracy
        for budget in BUDGETS:
            se = getRow(summary, model, dataset, "successive_elimination", budget)
            seCoverage, seAccuracy = float(se["coverage"]), float(se["accuracy"])
            seRisk = 1 - seAccuracy
            rows.append({
                "model": model, "dataset": dataset, "seBudget": budget,
                "mvCoverage": mvCoverage, "mvAccuracy": mvAccuracy, "mvSelectiveRisk": mvRisk,
                "seCoverage": seCoverage, "seAccuracy": seAccuracy, "seSelectiveRisk": seRisk,
                "coverageDiff_SEminusMV": seCoverage - mvCoverage,
                "riskDiff_SEminusMV": seRisk - mvRisk,
                "accuracyDiff_SEminusMV": seAccuracy - mvAccuracy,
            })
    return rows


def writeMainTableCsv(rows):
    fieldOrder = list(rows[0].keys())
    with open(os.path.join(OUT_DIR, "riskCoverageMainTable.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldOrder)
        writer.writeheader()
        writer.writerows(rows)


def classifyPair(rows, model, dataset):
    """Categorize a pair's risk trend across the 3 budgets, purely descriptively."""
    sub = [r for r in rows if r["model"] == model and r["dataset"] == dataset]
    diffs = [r["riskDiff_SEminusMV"] for r in sub]
    if all(d < -0.005 for d in diffs):
        return "SE risk lower than MV at all 3 budgets"
    if all(d > 0.005 for d in diffs):
        return "SE risk higher than MV at all 3 budgets"
    if all(abs(d) <= 0.005 for d in diffs):
        return "SE risk essentially unchanged vs MV"
    return "mixed across budgets"


def makeFigure(rows):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(PAIRS)))
    mvMarker = "D"
    seMarkers = {75: "o", 100: "^", 124: "v"}

    for pi, (model, dataset) in enumerate(PAIRS):
        sub = [r for r in rows if r["model"] == model and r["dataset"] == dataset]
        label = f"{model} × {dataset}"
        mvCov, mvRisk = sub[0]["mvCoverage"], sub[0]["mvSelectiveRisk"]
        ax.scatter(mvCov, mvRisk, marker=mvMarker, s=90, color=colors[pi],
                   edgecolor="black", linewidth=0.6, zorder=3, label=f"{label} (MV)")
        for r in sub:
            ax.scatter(r["seCoverage"], r["seSelectiveRisk"], marker=seMarkers[r["seBudget"]],
                       s=70, color=colors[pi], edgecolor="black", linewidth=0.4, zorder=3)
        # connect MV -> B75 -> B100 -> B124 with a thin line to show the trajectory, same color per pair
        xs = [mvCov] + [r["seCoverage"] for r in sorted(sub, key=lambda r: r["seBudget"])]
        ys = [mvRisk] + [r["seSelectiveRisk"] for r in sorted(sub, key=lambda r: r["seBudget"])]
        ax.plot(xs, ys, color=colors[pi], alpha=0.35, linewidth=1.2, zorder=2)

    # marker-shape legend (method), separate from the color legend (pair)
    from matplotlib.lines import Line2D
    shapeLegendHandles = [
        Line2D([0], [0], marker=mvMarker, color="w", markerfacecolor="gray", markeredgecolor="black", markersize=9, label="Majority voting"),
        Line2D([0], [0], marker=seMarkers[75], color="w", markerfacecolor="gray", markeredgecolor="black", markersize=8, label="SE B=75"),
        Line2D([0], [0], marker=seMarkers[100], color="w", markerfacecolor="gray", markeredgecolor="black", markersize=8, label="SE B=100"),
        Line2D([0], [0], marker=seMarkers[124], color="w", markerfacecolor="gray", markeredgecolor="black", markersize=8, label="SE B=124"),
    ]
    colorLegendHandles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=colors[pi], markersize=9, label=f"{m} × {d}")
        for pi, (m, d) in enumerate(PAIRS)
    ]

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Selective risk (1 − accuracy on resolved examples)")
    ax.set_title("Risk vs. coverage: majority voting and SE (B=75/100/124)\nby model × dataset")
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 1.02)

    legend1 = ax.legend(handles=colorLegendHandles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                         title="Model × dataset", fontsize=8, title_fontsize=9)
    ax.add_artist(legend1)
    ax.legend(handles=shapeLegendHandles, loc="lower left", bbox_to_anchor=(1.02, 0.0),
              title="Method / budget", fontsize=8, title_fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "riskCoverageMainFigure.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "riskCoverageMainFigure.svg"), bbox_inches="tight")
    plt.close(fig)


def writeMarkdown(rows):
    lines = []
    lines.append("# Risk vs. Coverage — Main Analysis\n")
    lines.append(
        "Does SE's reduced coverage correspond to lower selective risk among "
        "the examples it resolves? Read-only analysis over the existing "
        "`riskCoverageSummary.csv` / `riskCoverageComparisons.csv` outputs. "
        "No new data, no LLM calls, no manuscript or experiment file "
        "modified.\n"
    )

    lines.append("## 1. Method\n")
    lines.append(
        "For each of the 5 model×dataset pairs, majority voting (MV) gives "
        "one (coverage, selective risk) point; SE gives three, one per "
        "budget (B75/B100/B124). `coverage = resolvedCount/totalCount`, "
        "`selectiveRisk = 1 - accuracy`, where `accuracy` is computed only "
        "over resolved examples (escalated examples are never counted as "
        "incorrect). All values are read directly from the existing "
        "`riskCoverageSummary.csv` — nothing is recomputed from raw "
        "predictions here. Differences are reported as SE − MV at each "
        "budget.\n"
    )

    lines.append("## 2. Full numerical table\n")
    header = ("| Model | Dataset | SE budget | MV coverage | MV accuracy | MV risk | "
               "SE coverage | SE accuracy | SE risk | Cov. diff | Risk diff | Acc. diff |")
    sep = "| --- " * 12 + "|"
    lines.append(header)
    lines.append(sep)
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['dataset']} | B{r['seBudget']} | "
            f"{r['mvCoverage']:.3f} | {r['mvAccuracy']:.3f} | {r['mvSelectiveRisk']:.3f} | "
            f"{r['seCoverage']:.3f} | {r['seAccuracy']:.3f} | {r['seSelectiveRisk']:.3f} | "
            f"{r['coverageDiff_SEminusMV']:+.3f} | {r['riskDiff_SEminusMV']:+.3f} | {r['accuracyDiff_SEminusMV']:+.3f} |"
        )

    lines.append("\n## 3. Model-dataset observations (B75 → B100 → B124 trajectory)\n")
    narrativeByPair = {
        ("llama3.1_8b", "hatemoderate"): (
            "Lower SE coverage is accompanied by lower selective risk at all "
            "three budgets (risk diff negative throughout: {d75:+.3f}, "
            "{d100:+.3f}, {d124:+.3f}), with the gap narrowing as budget/"
            "coverage increases toward MV's level."
        ),
        ("llama3.1_8b", "aegis"): (
            "Same qualitative pattern as HateModerate, with a larger risk "
            "reduction at every budget ({d75:+.3f}, {d100:+.3f}, {d124:+.3f}) "
            "— the largest SE-favoring gap of any pair in this table."
        ),
        ("qwen2.5_7b", "hatemoderate"): (
            "Only modest risk improvement ({d75:+.3f}, {d100:+.3f}, "
            "{d124:+.3f}) — directionally the same as llama, but the "
            "magnitude is small relative to llama's pairs."
        ),
        ("qwen2.5_7b", "aegis"): (
            "Risk is essentially unchanged from MV at every budget "
            "({d75:+.3f}, {d100:+.3f}, {d124:+.3f}) — coverage drops "
            "modestly but selective risk does not meaningfully move."
        ),
        ("mistral_7b", "hatemoderate"): (
            "SE has lower coverage but HIGHER selective risk than MV at "
            "every budget ({d75:+.3f}, {d100:+.3f}, {d124:+.3f}) — the one "
            "pair where the coverage/risk relationship runs opposite to "
            "the usual selective-prediction expectation."
        ),
    }
    for model, dataset in PAIRS:
        sub = sorted([r for r in rows if r["model"] == model and r["dataset"] == dataset], key=lambda r: r["seBudget"])
        d = {f"d{r['seBudget']}": r["riskDiff_SEminusMV"] for r in sub}
        lines.append(f"**{model} × {dataset}**: " + narrativeByPair[(model, dataset)].format(**d) + "\n")

    lines.append("## 4. Cross-condition interpretation\n")
    lines.append(
        "The five pairs show genuinely heterogeneous behavior — this is not "
        "noise around one universal pattern. Two pairs (llama on both "
        "datasets) show SE trading coverage for meaningfully lower risk; one "
        "pair (qwen/HateModerate) shows the same direction but weakly; one "
        "pair (qwen/AEGIS) shows no meaningful risk change; and one pair "
        "(mistral/HateModerate) shows SE's lower coverage paired with "
        "*higher* risk, the opposite of the usual selective-prediction "
        "expectation.\n\n"
        "Critically, the paired agreement analysis "
        "(`seMvAgreement.md`) already established that SE and MV make the "
        "exact same decision on 99.67% of jointly-resolved examples, with "
        "only 11 discordant examples across all 3,310 example-comparisons. "
        "**This means the coverage/risk differences reported above should "
        "be read primarily as differences in *which* examples each method "
        "resolves, not as evidence that SE judges shared examples "
        "differently or better.** Where SE's resolved-set risk is lower "
        "than MV's, it is because SE's selection of which examples to "
        "resolve happens to exclude examples that would have been wrong — "
        "not because SE reaches a different, better verdict on the same "
        "examples.\n"
    )

    lines.append("## 5. What this does NOT establish\n")
    lines.append(
        "- **Not causal.** A negative risk-diff at lower coverage does not "
        "mean *lowering coverage causes* lower risk — both are downstream "
        "consequences of which examples SE's stopping rule happens to "
        "resolve at a given budget, not a controlled intervention on "
        "coverage itself.\n"
        "- **Not a universal difficulty detector.** The pattern holds for "
        "2 of 5 pairs clearly, weakly for 1, is absent for 1, and is "
        "*inverted* for 1 (mistral/HateModerate). Any claim that \"SE "
        "identifies easier examples\" must be scoped to the specific "
        "pairs where the data support it, not stated as a general property "
        "of the method.\n"
        "- **Selective risk, overall correctness, and coverage are three "
        "different quantities and are not interchangeable.** Selective "
        "risk is conditional on resolution (it says nothing about the "
        "escalated majority in some low-coverage conditions); overall "
        "correctness across *all* 400 examples would need to fold in the "
        "escalation rate itself (not computed here); coverage alone says "
        "nothing about accuracy. A pair can have simultaneously low "
        "coverage, low selective risk, AND low overall usefulness if "
        "escalation is too aggressive — this table does not adjudicate "
        "that tradeoff, it only reports the two axes requested.\n"
    )

    with open(os.path.join(OUT_DIR, "riskCoverageAnalysis.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    summary = loadSummary()
    rows = buildMainTable(summary)
    writeMainTableCsv(rows)
    makeFigure(rows)
    writeMarkdown(rows)

    lowerAtAll, higherAtAll, unchanged, mixed = [], [], [], []
    for model, dataset in PAIRS:
        category = classifyPair(rows, model, dataset)
        label = f"{model} × {dataset}"
        if category == "SE risk lower than MV at all 3 budgets":
            lowerAtAll.append(label)
        elif category == "SE risk higher than MV at all 3 budgets":
            higherAtAll.append(label)
        elif category == "SE risk essentially unchanged vs MV":
            unchanged.append(label)
        else:
            mixed.append(label)

    maxCovReduction = max(rows, key=lambda r: -r["coverageDiff_SEminusMV"])
    maxRiskReduction = min(rows, key=lambda r: r["riskDiff_SEminusMV"])
    maxRiskIncrease = max(rows, key=lambda r: r["riskDiff_SEminusMV"])

    print("=== KEY RESULTS ===")
    print(f"SE risk LOWER than MV at all 3 budgets: {lowerAtAll}")
    print(f"SE risk essentially UNCHANGED vs MV: {unchanged}")
    print(f"SE risk HIGHER than MV at all 3 budgets: {higherAtAll}")
    if mixed:
        print(f"Mixed/other pattern across budgets: {mixed}")
    print(f"\nLargest absolute coverage reduction: {maxCovReduction['model']}/{maxCovReduction['dataset']} B{maxCovReduction['seBudget']}: {maxCovReduction['coverageDiff_SEminusMV']:+.3f}")
    print(f"Largest absolute risk reduction: {maxRiskReduction['model']}/{maxRiskReduction['dataset']} B{maxRiskReduction['seBudget']}: {maxRiskReduction['riskDiff_SEminusMV']:+.3f}")
    print(f"Largest absolute risk increase: {maxRiskIncrease['model']}/{maxRiskIncrease['dataset']} B{maxRiskIncrease['seBudget']}: {maxRiskIncrease['riskDiff_SEminusMV']:+.3f}")
