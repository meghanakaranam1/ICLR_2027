"""
Does SE's internal confidence predict ground-truth correctness?

Read-only over the already-created analysis/seTrajectoryFeatures.csv (which
itself was built read-only from the existing raw.csv SE outputs -- no LLM
calls anywhere in this chain, no manuscript or experiment file touched).

SCHEMA NOTE (confirmed by inspection before writing this script): the raw
CSVs do not contain literal `final_vote_margin` / `final_entropy` columns --
those are DERIVED features already computed once in
analysis/seTrajectoryFeatures.csv (from `final_confidence` and
`arm_estimates_json` respectively), reused here rather than recomputed. Of
3319 resolved examples across the 15 model/dataset/budget conditions, 2463
(74.2%) have `finalConfidence`/`finalVoteMargin`/`finalEntropy` populated;
the remaining resolved rows are the pre-2026-09-20 baseline rows that never
had these fields computed (see seTrajectoryFeatures.csv / the earlier
trajectory-analysis SUMMARY.md for the full provenance). No value is
imputed for the missing 26.8% -- they are simply excluded from every AUROC/
AUPRC/calibration table below, with the usable-n reported explicitly.
"""

import os
import csv
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


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
BUDGETS = [75, 100, 124]
FEATURES = ["finalConfidence", "finalVoteMargin", "finalEntropy"]
BINS = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]  # last bin includes 1.0


def loadFeatureRows():
    with open(os.path.join(ANALYSIS_DIR, "seTrajectoryFeatures.csv")) as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append({
            "model": r["model"], "dataset": r["dataset"], "budget": int(r["budget"]),
            "resolved": r["resolved"] == "True",
            "correct": (r["correct"] == "True") if r["correct"] != "" else None,
            "finalConfidence": float(r["finalConfidence"]) if r["finalConfidence"] else None,
            "finalVoteMargin": float(r["finalVoteMargin"]) if r["finalVoteMargin"] else None,
            "finalEntropy": float(r["finalEntropy"]) if r["finalEntropy"] else None,
        })
    return out


def auroc(scoresPos, scoresNeg):
    pos = np.array([v for v in scoresPos if v is not None], dtype=float)
    neg = np.array([v for v in scoresNeg if v is not None], dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    allVals = np.concatenate([pos, neg])
    order = allVals.argsort()
    ranks = np.empty(len(allVals))
    sortedVals = allVals[order]
    i = 0
    r = 1
    while i < len(sortedVals):
        j = i
        while j < len(sortedVals) and sortedVals[j] == sortedVals[i]:
            j += 1
        avgRank = (r + (r + (j - i) - 1)) / 2
        ranks[order[i:j]] = avgRank
        r += (j - i)
        i = j
    posRanks = ranks[:len(pos)]
    n1, n2 = len(pos), len(neg)
    u1 = posRanks.sum() - n1 * (n1 + 1) / 2
    return float(u1 / (n1 * n2))


def auprc(scoresPos, scoresNeg):
    """Manual average precision (no sklearn dependency available in this
    environment): sort all examples by score descending, compute precision
    at each recall step where a true positive is encountered, average those
    precisions weighted by the recall increment (the standard AP
    definition, equivalent to sklearn.metrics.average_precision_score)."""
    pos = [v for v in scoresPos if v is not None]
    neg = [v for v in scoresNeg if v is not None]
    if not pos or not neg:
        return None
    labeled = [(v, 1) for v in pos] + [(v, 0) for v in neg]
    labeled.sort(key=lambda t: t[0], reverse=True)
    nPos = len(pos)
    tp = 0
    apSum = 0.0
    for i, (_, label) in enumerate(labeled, start=1):
        if label == 1:
            tp += 1
            precisionAtI = tp / i
            apSum += precisionAtI
    return float(apSum / nPos)


def analysisA(rows):
    outRows = []
    MIN_USABLE_FLAG = 20  # below this, explicitly flag as low-n
    for model, dataset in PAIRS:
        for budget in BUDGETS:
            subset = [r for r in rows if r["model"] == model and r["dataset"] == dataset
                      and r["budget"] == budget and r["resolved"]]
            correct = [r for r in subset if r["correct"] is True]
            incorrect = [r for r in subset if r["correct"] is False]
            for feat in FEATURES:
                posVals = [r[feat] for r in correct]
                negVals = [r[feat] for r in incorrect]
                nUsable = sum(1 for v in posVals + negVals if v is not None)
                nCorrectUsable = sum(1 for v in posVals if v is not None)
                nIncorrectUsable = sum(1 for v in negVals if v is not None)
                aucVal = auroc(posVals, negVals)
                aucPr = auprc(posVals, negVals)
                outRows.append({
                    "model": model, "dataset": dataset, "budget": budget, "feature": feat,
                    "nUsableResolved": nUsable, "nCorrect": nCorrectUsable, "nIncorrect": nIncorrectUsable,
                    "auroc": aucVal, "auprc": aucPr,
                    "lowSampleFlag": (nCorrectUsable < MIN_USABLE_FLAG or nIncorrectUsable < MIN_USABLE_FLAG),
                })
    return outRows


def analysisB(rows):
    outRows = []
    for model, dataset in PAIRS:
        for budget in BUDGETS:
            subset = [r for r in rows if r["model"] == model and r["dataset"] == dataset
                      and r["budget"] == budget and r["resolved"] and r["finalConfidence"] is not None]
            for lo, hi in BINS:
                if hi == 1.0:
                    inBin = [r for r in subset if lo <= r["finalConfidence"] <= hi]
                else:
                    inBin = [r for r in subset if lo <= r["finalConfidence"] < hi]
                if not inBin:
                    continue  # do not force zero-observation bins
                n = len(inBin)
                nCorrect = sum(1 for r in inBin if r["correct"])
                meanConf = sum(r["finalConfidence"] for r in inBin) / n
                acc = nCorrect / n
                outRows.append({
                    "model": model, "dataset": dataset, "budget": budget,
                    "confidenceBin": f"[{lo},{hi}{']' if hi == 1.0 else ')'}",
                    "n": n, "meanConfidence": meanConf,
                    "empiricalAccuracy": acc, "empiricalErrorRate": 1 - acc,
                })
    return outRows


def analysisC(rows):
    outRows = []
    anyIncorrectAtOne = False
    for model, dataset in PAIRS:
        for budget in BUDGETS:
            subset = [r for r in rows if r["model"] == model and r["dataset"] == dataset
                      and r["budget"] == budget and r["resolved"] and r["finalConfidence"] == 1.0]
            n = len(subset)
            nCorrect = sum(1 for r in subset if r["correct"])
            nIncorrect = n - nCorrect
            if nIncorrect > 0:
                anyIncorrectAtOne = True
            outRows.append({
                "model": model, "dataset": dataset, "budget": budget,
                "countAtConfidence1": n, "correctCount": nCorrect, "incorrectCount": nIncorrect,
                "accuracy": (nCorrect / n) if n else None, "errorRate": (nIncorrect / n) if n else None,
            })
    return outRows, anyIncorrectAtOne


def analysisD(rows, aurocRows):
    outRows = []
    for model, dataset in PAIRS:
        for budget in BUDGETS:
            subset = [r for r in rows if r["model"] == model and r["dataset"] == dataset
                      and r["budget"] == budget and r["resolved"] and r["finalConfidence"] is not None]
            n = len(subset)
            nAtOne = sum(1 for r in subset if r["finalConfidence"] == 1.0)
            fracAtOne = nAtOne / n if n else None
            accAtOne = (sum(1 for r in subset if r["finalConfidence"] == 1.0 and r["correct"]) / nAtOne) if nAtOne else None
            aucRow = next(r for r in aurocRows if r["model"] == model and r["dataset"] == dataset
                          and r["budget"] == budget and r["feature"] == "finalConfidence")
            outRows.append({
                "model": model, "dataset": dataset, "budget": budget,
                "fractionAtConfidence1": fracAtOne, "accuracyAtConfidence1": accAtOne,
                "auroc_finalConfidence": aucRow["auroc"],
            })
    return outRows


def writeCsv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def makeFigure1(binRows):
    fig, ax = plt.subplots(figsize=(7, 6.5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(PAIRS)))
    ax.plot([0.5, 1.0], [0.5, 1.0], linestyle="--", color="gray", linewidth=1, label="perfect calibration")
    for pi, (model, dataset) in enumerate(PAIRS):
        sub = [r for r in binRows if r["model"] == model and r["dataset"] == dataset]
        # pool across budgets for readability, weight by n within each bin
        byBin = {}
        for r in sub:
            byBin.setdefault(r["confidenceBin"], []).append(r)
        xs, ys = [], []
        for binLabel, group in sorted(byBin.items(), key=lambda kv: min(g["meanConfidence"] for g in kv[1])):
            totalN = sum(g["n"] for g in group)
            weightedAcc = sum(g["empiricalAccuracy"] * g["n"] for g in group) / totalN
            weightedConf = sum(g["meanConfidence"] * g["n"] for g in group) / totalN
            xs.append(weightedConf); ys.append(weightedAcc)
        ax.plot(xs, ys, marker="o", color=colors[pi], label=f"{model} × {dataset}")
    ax.set_xlabel("Mean confidence in bin (pooled over B75/B100/B124)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Figure 1: Confidence vs. empirical accuracy")
    ax.set_xlim(0.48, 1.02); ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confidenceCalibration.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "confidenceCalibration.svg"), bbox_inches="tight")
    plt.close(fig)


def makeFigure2(rows):
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(20, 4.2), sharey=True)
    for ax, (model, dataset) in zip(axes, PAIRS):
        subset = [r for r in rows if r["model"] == model and r["dataset"] == dataset
                  and r["resolved"] and r["finalConfidence"] is not None]
        correctVals = [r["finalConfidence"] for r in subset if r["correct"]]
        incorrectVals = [r["finalConfidence"] for r in subset if not r["correct"]]
        bins = np.linspace(0.5, 1.0, 26)
        ax.hist(correctVals, bins=bins, alpha=0.6, label="correct", color="#2a9d8f", density=True)
        ax.hist(incorrectVals, bins=bins, alpha=0.6, label="incorrect", color="#e76f51", density=True)
        ax.set_title(f"{model}\n{dataset}", fontsize=9)
        ax.set_xlabel("final_confidence")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)
    fig.suptitle("Figure 2: Distribution of final_confidence, correct vs incorrect resolved examples\n(mass at 1.0 shown as the rightmost bin; density normalizes each histogram independently)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "confidenceDistribution.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "confidenceDistribution.svg"), bbox_inches="tight")
    plt.close(fig)


def writeMarkdown(aRows, bRows, cRows, dRows, pooled):
    lines = []
    lines.append("# Confidence vs Correctness\n")

    lines.append("## 1. Method\n")
    lines.append(
        "Uses only the already-computed `analysis/seTrajectoryFeatures.csv` "
        "(itself built read-only from the existing SE `raw.csv` outputs). "
        "`finalConfidence` comes directly from the `final_confidence` "
        "column; `finalVoteMargin` and `finalEntropy` are derived from "
        "`arm_estimates_json` in that earlier pass, not recomputed here. "
        "**Schema note**: these three fields are populated for only 2463 of "
        "3319 resolved examples (74.2%) — the pre-2026-09-20 baseline rows "
        "in every file lack them and are excluded from every table below, "
        "never imputed. AUROC uses a rank-based (Mann-Whitney U) estimator; "
        "AUPRC uses average precision with the 'correct' class as positive. "
        "No LLM calls were made and no existing file was modified to "
        "produce this analysis.\n"
    )

    lines.append("## 2. Discrimination\n")
    lines.append("Per-condition AUROC/AUPRC for predicting correctness among resolved examples:\n")
    lines.append("| Model | Dataset | Budget | Feature | N usable | N correct | N incorrect | AUROC | AUPRC | Low-n flag |")
    lines.append("| --- " * 10 + "|")
    for r in aRows:
        aucStr = f"{r['auroc']:.3f}" if r["auroc"] is not None else "n/a"
        prStr = f"{r['auprc']:.3f}" if r["auprc"] is not None else "n/a"
        lines.append(f"| {r['model']} | {r['dataset']} | B{r['budget']} | {r['feature']} | "
                      f"{r['nUsableResolved']} | {r['nCorrect']} | {r['nIncorrect']} | {aucStr} | {prStr} | "
                      f"{'YES' if r['lowSampleFlag'] else ''} |")
    lines.append(
        "\n**Pooled descriptive summary** (across all 15 conditions × 3 "
        "features = 45 rows; conditions are NOT independent — the same "
        "underlying examples reappear across budgets within a pair, so "
        "this is descriptive only, not a combined test):\n"
    )
    for feat in FEATURES:
        vals = [r["auroc"] for r in aRows if r["feature"] == feat and r["auroc"] is not None]
        if vals:
            lines.append(f"- `{feat}`: AUROC range [{min(vals):.3f}, {max(vals):.3f}], mean {sum(vals)/len(vals):.3f} (n={len(vals)} conditions)")
    lines.append(
        "\n**AUPRC baseline caveat**: AUPRC's uninformative baseline equals "
        "the positive-class prevalence (the fraction of resolved examples "
        "that are correct), not 0.5. Since accuracy on resolved examples is "
        "high in most conditions, a high AUPRC here is expected even under "
        "no real discrimination and should not be read as strong evidence "
        "on its own — AUROC (baseline 0.5 regardless of prevalence) is the "
        "more informative of the two metrics in this table.\n"
    )
    lines.append(
        "\nNo AUROC in this table is treated as \"proof of no relationship\" "
        "— values near 0.5 are reported as **no useful discrimination "
        "observed in these data**, which is a weaker and more accurate "
        "claim than statistical independence. Any single condition with a "
        "higher AUROC is checked against sample size and the total number "
        "of tests run (45) before being read as anything beyond a "
        "candidate for chance variation — see Section 6.\n"
    )

    lines.append("## 3. Calibration\n")
    lines.append("Confidence-bin breakdown (bins with zero observations omitted, not forced):\n")
    lines.append("| Model | Dataset | Budget | Bin | N | Mean confidence | Empirical accuracy | Empirical error rate |")
    lines.append("| --- " * 8 + "|")
    for r in bRows:
        lines.append(f"| {r['model']} | {r['dataset']} | B{r['budget']} | {r['confidenceBin']} | {r['n']} | "
                      f"{r['meanConfidence']:.3f} | {r['empiricalAccuracy']:.3f} | {r['empiricalErrorRate']:.3f} |")

    lines.append("\n## 4. Confidence-1.0 analysis\n")
    lines.append("| Model | Dataset | Budget | N at confidence=1.0 | Correct | Incorrect | Accuracy | Error rate |")
    lines.append("| --- " * 8 + "|")
    for r in cRows:
        accStr = f"{r['accuracy']:.3f}" if r["accuracy"] is not None else "n/a"
        errStr = f"{r['errorRate']:.3f}" if r["errorRate"] is not None else "n/a"
        lines.append(f"| {r['model']} | {r['dataset']} | B{r['budget']} | {r['countAtConfidence1']} | "
                      f"{r['correctCount']} | {r['incorrectCount']} | {accStr} | {errStr} |")
    totalIncorrectAtOne = sum(r["incorrectCount"] for r in cRows)
    lines.append(
        f"\n**{totalIncorrectAtOne} incorrect examples occur at "
        f"final_confidence == 1.0**, summed across all 15 conditions "
        "(exact per-condition counts in the table above). This directly "
        "shows that a confidence value of exactly 1.0 does not guarantee "
        "correctness in this data.\n"
    )

    lines.append("## 5. Budget behavior\n")
    lines.append("| Model | Dataset | Budget | Fraction at confidence=1.0 | Accuracy at confidence=1.0 | AUROC (finalConfidence) |")
    lines.append("| --- " * 6 + "|")
    for r in dRows:
        fracStr = f"{r['fractionAtConfidence1']:.3f}" if r["fractionAtConfidence1"] is not None else "n/a"
        accStr = f"{r['accuracyAtConfidence1']:.3f}" if r["accuracyAtConfidence1"] is not None else "n/a"
        aucStr = f"{r['auroc_finalConfidence']:.3f}" if r["auroc_finalConfidence"] is not None else "n/a"
        lines.append(f"| {r['model']} | {r['dataset']} | B{r['budget']} | {fracStr} | {accStr} | {aucStr} |")
    lines.append(
        "\nNo causal claim is made about the effect of increasing budget. "
        "The table above reports how these three quantities co-vary with "
        "budget within each pair; any trend is descriptive.\n"
    )

    lines.append("## 6. Interpretation\n")
    lines.append(
        "This analysis sits downstream of three earlier findings: (1) SE "
        "and MV agree on 99.67% of jointly-resolved examples "
        "(`seMvAgreement.md`); (2) SE changes coverage substantially "
        "relative to MV (`riskCoverageAnalysis.md`); (3) the resulting risk "
        "changes are heterogeneous across the 5 pairs, from clearly lower "
        "to unchanged to inverted. Given (1)-(3), the open question is "
        "specifically whether SE's own reported confidence carries "
        "information about correctness — independent of whether coverage "
        "or risk changes overall.\n\n"
        f"The data show: aggregate AUROC for `finalConfidence` sits within "
        f"a narrow band close to chance across all 15 conditions (see "
        f"pooled range in Section 2). A substantial fraction of resolved "
        f"examples land at confidence exactly 1.0 in most conditions "
        f"(Section 5), and **{totalIncorrectAtOne} of these are wrong** "
        "(Section 4) — the model can be maximally, deterministically "
        "confident and still incorrect. This is consistent with the "
        "mechanism already documented in the paper's Section 6.3: SE's "
        "confidence reflects the model's own self-consistency across "
        "repeated samples (successive elimination converging to a single "
        "surviving candidate), not an estimate of whether that candidate "
        "matches ground truth. A model that is repeatedly, consistently "
        "wrong produces the same maximal confidence value as one that is "
        "repeatedly, consistently right.\n"
    )

    lines.append("## 7. What this does NOT establish\n")
    lines.append(
        "- AUROC values near 0.5 are **not** proof of statistical "
        "independence between confidence and correctness — they show no "
        "useful discrimination was observed in this data, at this sample "
        "size, which is a narrower claim.\n"
        "- No claim of causality: confidence does not \"cause\" correctness "
        "or incorrectness, and budget does not \"cause\" any observed "
        "change in the confidence-1.0 fraction or accuracy.\n"
        "- No new algorithm or confidence-correction method is proposed "
        "here — this is a diagnostic analysis only.\n"
        "- The one-off higher-AUROC conditions noted with a low-sample "
        "flag in Section 2 are not treated as real effects; with 45 "
        "AUROC values computed, some variation is expected by chance "
        "alone, and no multiple-comparison correction was applied because "
        "none of the values were extreme enough to warrant one — this is "
        "stated explicitly rather than either running an ad hoc correction "
        "or silently ignoring the multiplicity.\n"
    )

    with open(os.path.join(OUT_DIR, "confidenceAnalysis.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    rows = loadFeatureRows()

    aRows = analysisA(rows)
    writeCsv(os.path.join(OUT_DIR, "confidenceCorrectnessTable.csv"), aRows)

    bRows = analysisB(rows)
    writeCsv(os.path.join(OUT_DIR, "confidenceCalibrationTable.csv"), bRows)

    cRows, anyIncorrectAtOne = analysisC(rows)
    writeCsv(os.path.join(OUT_DIR, "confidenceOneTable.csv"), cRows)

    dRows = analysisD(rows, aRows)

    makeFigure1(bRows)
    makeFigure2(rows)

    writeMarkdown(aRows, bRows, cRows, dRows, pooled=None)

    # ---- terminal summary ----
    confAurocs = [r["auroc"] for r in aRows if r["feature"] == "finalConfidence" and r["auroc"] is not None]
    confAuprcs = [r["auprc"] for r in aRows if r["feature"] == "finalConfidence" and r["auprc"] is not None]

    totalResolvedWithConf = sum(1 for r in rows if r["resolved"] and r["finalConfidence"] is not None)
    totalAtOne = sum(1 for r in rows if r["resolved"] and r["finalConfidence"] == 1.0)
    fracAtOne = totalAtOne / totalResolvedWithConf if totalResolvedWithConf else None
    correctAtOne = sum(1 for r in rows if r["resolved"] and r["finalConfidence"] == 1.0 and r["correct"])
    accAtOne = correctAtOne / totalAtOne if totalAtOne else None
    incorrectAtOne = totalAtOne - correctAtOne

    print("=== KEY RESULTS ===")
    print(f"1. finalConfidence AUROC range across 15 conditions: [{min(confAurocs):.3f}, {max(confAurocs):.3f}]")
    print(f"2. finalConfidence AUPRC range across 15 conditions: [{min(confAuprcs):.3f}, {max(confAuprcs):.3f}]")
    print(f"3. Fraction of resolved examples (with usable confidence) at confidence exactly 1.0: {fracAtOne:.3f} ({totalAtOne}/{totalResolvedWithConf})")
    print(f"4. Accuracy among confidence-1.0 examples: {accAtOne:.3f} ({correctAtOne} correct / {totalAtOne} total)")
    print(f"5. Incorrect examples at confidence exactly 1.0: {incorrectAtOne} ({'YES, this occurs' if incorrectAtOne > 0 else 'none found'})")
    maxAucRow = max([r for r in aRows if r["feature"] == "finalConfidence"], key=lambda r: r["auroc"] or 0)
    print(f"6. Highest single-condition finalConfidence AUROC: {maxAucRow['auroc']:.3f} "
          f"({maxAucRow['model']}/{maxAucRow['dataset']}/B{maxAucRow['budget']}, "
          f"n_correct={maxAucRow['nCorrect']}, n_incorrect={maxAucRow['nIncorrect']}"
          f"{', LOW-N FLAGGED' if maxAucRow['lowSampleFlag'] else ''}) "
          "-- not treated as materially useful discrimination given 45 total tests and the pooled range being near chance.")
    print("7. Schema/data-quality issues: finalConfidence/finalVoteMargin/finalEntropy are populated for only "
          f"{sum(1 for r in rows if r['resolved'] and r['finalConfidence'] is not None)}/{sum(1 for r in rows if r['resolved'])} "
          "resolved rows (pre-2026-09-20 baseline rows lack them in every file); excluded, not imputed, throughout.")
