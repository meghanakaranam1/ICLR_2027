"""
SE vs majority-voting exact-agreement analysis on jointly-resolved examples.

Read-only: no LLM calls, no changes to any existing experiment file or to
riskCoverageComparisons.csv. Reuses the same jointly-resolved restriction
and McNemar construction already used in risk_coverage/computeRiskCoverage.py
and in the paper's own Section 6.1/6.2 tables.

Outputs:
  analysis/risk_coverage/seMvAgreement.csv          (15 rows, per model/dataset/budget)
  analysis/risk_coverage/seMvAgreementSummary.csv   (1 pooled descriptive row)
  analysis/risk_coverage/seMvAgreement.md           (methodology + tables + interpretation)
"""

import os
import csv
import math
from scipy.stats import binomtest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULT_ROOTS = ["results_02_n400_from_spark", "results_02"]
NON_TERMINAL = {"escalate", "self_refused", "prompt_injected", "parse_failed"}
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

PAIRS = [
    ("llama3.1_8b", "hatemoderate"),
    ("llama3.1_8b", "aegis"),
    ("qwen2.5_7b", "hatemoderate"),
    ("qwen2.5_7b", "aegis"),
    ("mistral_7b", "hatemoderate"),
]
BUDGETS = [75, 100, 124]


def findBestSourceFile(modelDir, dataset, condition):
    bestPath, bestN = None, -1
    for root in RESULT_ROOTS:
        path = os.path.join(PROJECT_ROOT, root, modelDir, dataset, condition, "raw.csv")
        if os.path.isfile(path):
            with open(path) as f:
                n = sum(1 for _ in f) - 1
            if n > bestN:
                bestPath, bestN = path, n
    return bestPath


def loadRows(modelDir, dataset, condition):
    path = findBestSourceFile(modelDir, dataset, condition)
    with open(path) as f:
        return list(csv.DictReader(f))


def isResolved(row):
    return row["final_label"] not in NON_TERMINAL


def analyzeOnePair(modelDir, dataset, budget):
    mvRows = loadRows(modelDir, dataset, "graph_mv")
    seRows = loadRows(modelDir, dataset, f"graph_ucb_B{budget}")
    mvById = {r["input_id"]: r for r in mvRows}
    seById = {r["input_id"]: r for r in seRows}

    common = set(mvById) & set(seById)
    jointlyResolved = {i for i in common if isResolved(mvById[i]) and isResolved(seById[i])}
    n = len(jointlyResolved)

    bothCorrect = bothIncorrect = 0
    mvCorrectSeIncorrect = 0  # MV right, SE wrong (discordant, "b")
    seCorrectMvIncorrect = 0  # SE right, MV wrong (discordant, "c")

    for i in jointlyResolved:
        mvRow, seRow = mvById[i], seById[i]
        mvCorrect = mvRow["final_label"] == mvRow["ground_truth"]
        seCorrect = seRow["final_label"] == seRow["ground_truth"]
        if mvCorrect and seCorrect:
            bothCorrect += 1
        elif (not mvCorrect) and (not seCorrect):
            bothIncorrect += 1
        elif mvCorrect and not seCorrect:
            mvCorrectSeIncorrect += 1
        else:
            seCorrectMvIncorrect += 1

    # "agree" here means EXACT SAME FINAL LABEL (not just same correctness),
    # per the task's own wording ("make the exact same final decision").
    agreeCount = sum(1 for i in jointlyResolved if mvById[i]["final_label"] == seById[i]["final_label"])
    disagreeCount = n - agreeCount
    exactAgreementRate = agreeCount / n if n else None

    seCorrectCount = bothCorrect + seCorrectMvIncorrect
    mvCorrectCount = bothCorrect + mvCorrectSeIncorrect
    pairedAccDiff = (seCorrectCount - mvCorrectCount) / n if n else None

    nDiscordant = mvCorrectSeIncorrect + seCorrectMvIncorrect
    pValue = 1.0 if nDiscordant == 0 else binomtest(
        min(mvCorrectSeIncorrect, seCorrectMvIncorrect), nDiscordant, 0.5
    ).pvalue

    # 95% CI on the paired accuracy difference (SE - MV), via the standard
    # Wald-type variance estimator for a difference of paired/matched binary
    # proportions, built from the discordant-pair counts b=mvCorrectSeIncorrect,
    # c=seCorrectMvIncorrect (same b/c McNemar itself uses):
    #   diff = (c - b) / n
    #   Var(diff) = [ n*(b+c) - (c-b)^2 ] / n^3
    #   CI = diff +/- 1.96 * sqrt(Var(diff))
    # This is the discordant-pair formulation explicitly permitted by the
    # task instructions when a dedicated paired-proportion CI routine isn't
    # already available in the codebase.
    ciLo = ciHi = None
    if n:
        b, c = mvCorrectSeIncorrect, seCorrectMvIncorrect
        variance = (n * (b + c) - (c - b) ** 2) / (n ** 3)
        variance = max(variance, 0.0)
        halfWidth = 1.959963984540054 * math.sqrt(variance)
        ciLo = pairedAccDiff - halfWidth
        ciHi = pairedAccDiff + halfWidth

    return {
        "model": modelDir, "dataset": dataset, "seBudget": budget,
        "jointlyResolvedN": n,
        "agreeCount": agreeCount, "disagreeCount": disagreeCount,
        "exactAgreementRate": exactAgreementRate,
        "bothCorrectCount": bothCorrect, "bothIncorrectCount": bothIncorrect,
        "mvCorrectSeIncorrectCount": mvCorrectSeIncorrect,
        "seCorrectMvIncorrectCount": seCorrectMvIncorrect,
        "pairedAccuracyDiff_SEminusMV": pairedAccDiff,
        "pairedAccDiffCiLo": ciLo, "pairedAccDiffCiHi": ciHi,
        "mcNemarExactPValue": pValue,
    }


def main():
    rows = []
    for modelDir, dataset in PAIRS:
        for budget in BUDGETS:
            rows.append(analyzeOnePair(modelDir, dataset, budget))

    fieldOrder = [
        "model", "dataset", "seBudget", "jointlyResolvedN",
        "agreeCount", "disagreeCount", "exactAgreementRate",
        "bothCorrectCount", "bothIncorrectCount",
        "mvCorrectSeIncorrectCount", "seCorrectMvIncorrectCount",
        "pairedAccuracyDiff_SEminusMV", "pairedAccDiffCiLo", "pairedAccDiffCiHi",
        "mcNemarExactPValue",
    ]
    with open(os.path.join(OUT_DIR, "seMvAgreement.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldOrder)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldOrder})

    # ---- pooled descriptive summary (explicitly NOT independent replicates) ----
    totalComparisons = sum(r["jointlyResolvedN"] for r in rows)  # example-COMPARISONS, not unique examples
    uniqueExamplesPerPair = {}
    for modelDir, dataset in PAIRS:
        mvRows = loadRows(modelDir, dataset, "graph_mv")
        uniqueExamplesPerPair[(modelDir, dataset)] = len(mvRows)  # 400 per pair, for reference
    totalUniqueExamplesAcrossPairs = sum(uniqueExamplesPerPair.values())  # 5*400=2000, each counted once per pair regardless of budget

    totalAgree = sum(r["agreeCount"] for r in rows)
    totalDisagree = sum(r["disagreeCount"] for r in rows)
    totalBothCorrect = sum(r["bothCorrectCount"] for r in rows)
    totalBothIncorrect = sum(r["bothIncorrectCount"] for r in rows)
    totalMvOnly = sum(r["mvCorrectSeIncorrectCount"] for r in rows)
    totalSeOnly = sum(r["seCorrectMvIncorrectCount"] for r in rows)
    totalDiscordant = totalMvOnly + totalSeOnly

    pooledAgreementRate = totalAgree / totalComparisons if totalComparisons else None
    rates = [r["exactAgreementRate"] for r in rows]
    diffs = [r["pairedAccuracyDiff_SEminusMV"] for r in rows]

    summary = {
        "totalExampleComparisons_15rows_pooled": totalComparisons,
        "note_uniqueExamplesPerPairIs400_countedOnceRegardlessOfBudget": totalUniqueExamplesAcrossPairs,
        "totalAgreeCount": totalAgree,
        "totalDisagreeCount": totalDisagree,
        "pooledExactAgreementRate": pooledAgreementRate,
        "totalBothCorrectCount": totalBothCorrect,
        "totalBothIncorrectCount": totalBothIncorrect,
        "totalMvCorrectSeIncorrectCount": totalMvOnly,
        "totalSeCorrectMvIncorrectCount": totalSeOnly,
        "totalDiscordantCorrectnessPairs": totalDiscordant,
        "minAgreementRateAcross15Rows": min(rates),
        "maxAgreementRateAcross15Rows": max(rates),
        "minPairedAccDiffAcross15Rows": min(diffs),
        "maxPairedAccDiffAcross15Rows": max(diffs),
    }
    with open(os.path.join(OUT_DIR, "seMvAgreementSummary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    return rows, summary


def writeMarkdown(rows, summary):
    lines = []
    lines.append("# SE vs Majority Voting — Exact Agreement Analysis\n")
    lines.append(
        "Descriptive analysis of how often successive elimination (SE) and "
        "majority voting (MV) produce the exact same final label on examples "
        "**both methods resolve** (jointly resolved: neither escalates, "
        "self-refuses, is prompt-injected, nor parse-fails). Read-only over "
        "existing `raw.csv` outputs; no LLM calls, no experiment file "
        "modified, no reruns.\n"
    )
    lines.append("## Methodology\n")
    lines.append(
        "- **Jointly resolved**: an example is included only if both MV's "
        "`graph_mv` and SE's `graph_ucb_B{budget}` produced a real task "
        "label for it (not one of `escalate`/`self_refused`/"
        "`prompt_injected`/`parse_failed`).\n"
        "- **Exact agreement**: MV's `final_label` equals SE's `final_label` "
        "on that example (a stricter criterion than \"both correct\" — two "
        "methods can both be wrong and still agree, or both be right and "
        "still agree; this row is agreement on the *decision*, not on "
        "correctness).\n"
        "- **McNemar's exact test**: binomial test on the discordant pairs "
        "only (`b` = MV-correct/SE-incorrect, `c` = SE-correct/MV-incorrect), "
        "identical construction to the paper's existing Section 6.1/6.2 "
        "tables and to `riskCoverageComparisons.csv`.\n"
        "- **95% CI on paired accuracy difference** (SE − MV): Wald-type CI "
        "for a difference of matched/paired binary proportions, built "
        "directly from the discordant-pair counts: "
        "`diff = (c - b) / n`, `Var(diff) = [n(b+c) - (c-b)^2] / n^3`, "
        "`CI = diff ± 1.96*sqrt(Var(diff))`. This is the discordant-pair "
        "formulation the task instructions permit when no dedicated paired-"
        "proportion routine already exists in the codebase — it is not a "
        "new statistical method invented for this analysis, just the "
        "standard McNemar-adjacent variance estimator applied directly.\n"
        "- **Non-significance ≠ equivalence**: a McNemar p-value of 1.0 "
        "means the data give no evidence of a difference; it does **not** "
        "prove SE and MV are equivalent, especially at the small discordant-"
        "pair counts seen here (many rows have 0–3 discordant pairs, so "
        "power to detect a real but modest difference is low).\n"
    )

    lines.append("## Full 15-row table\n")
    header = ("| Model | Dataset | Budget | N (jointly resolved) | Agree | Disagree | "
               "Agreement rate | Both correct | Both incorrect | MV-only correct | SE-only correct | "
               "Paired acc. diff (SE-MV) | 95% CI | McNemar p |")
    sep = "| --- " * 13 + "|"
    lines.append(header)
    lines.append(sep)
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['dataset']} | B{r['seBudget']} | {r['jointlyResolvedN']} | "
            f"{r['agreeCount']} | {r['disagreeCount']} | {r['exactAgreementRate']:.3f} | "
            f"{r['bothCorrectCount']} | {r['bothIncorrectCount']} | "
            f"{r['mvCorrectSeIncorrectCount']} | {r['seCorrectMvIncorrectCount']} | "
            f"{r['pairedAccuracyDiff_SEminusMV']:+.3f} | "
            f"[{r['pairedAccDiffCiLo']:+.3f}, {r['pairedAccDiffCiHi']:+.3f}] | "
            f"{r['mcNemarExactPValue']:.4f} |"
        )

    lines.append("\n## Pooled descriptive statistics\n")
    lines.append(
        "**Important**: the 15 rows are NOT independent statistical "
        "replicates — the same underlying 400 examples per pair reappear "
        "at all three budgets, so pooling across rows pools "
        "example-*comparisons*, not unique examples. The pooled numbers "
        "below are purely descriptive (a weighted overall rate), not a "
        "combined hypothesis test.\n"
    )
    lines.append(f"- Total example-comparisons pooled across all 15 rows: **{summary['totalExampleComparisons_15rows_pooled']}**")
    lines.append(f"- For reference, unique examples per pair (counted once, budget-independent): 400 × 5 pairs = **{summary['note_uniqueExamplesPerPairIs400_countedOnceRegardlessOfBudget']}**")
    lines.append(f"- Total agree: **{summary['totalAgreeCount']}**, total disagree: **{summary['totalDisagreeCount']}**")
    lines.append(f"- Pooled exact agreement rate: **{summary['pooledExactAgreementRate']:.4f}**")
    lines.append(f"- Both correct: {summary['totalBothCorrectCount']}, both incorrect: {summary['totalBothIncorrectCount']}")
    lines.append(f"- MV-correct/SE-incorrect: {summary['totalMvCorrectSeIncorrectCount']}, SE-correct/MV-incorrect: {summary['totalSeCorrectMvIncorrectCount']}")
    lines.append(f"- Total discordant correctness pairs: **{summary['totalDiscordantCorrectnessPairs']}** (out of {summary['totalExampleComparisons_15rows_pooled']} comparisons)")
    lines.append(f"- Agreement rate range across the 15 rows: [{summary['minAgreementRateAcross15Rows']:.3f}, {summary['maxAgreementRateAcross15Rows']:.3f}]")
    lines.append(f"- Paired accuracy difference range across the 15 rows: [{summary['minPairedAccDiffAcross15Rows']:+.3f}, {summary['maxPairedAccDiffAcross15Rows']:+.3f}]")

    lines.append("\n## Interpretation\n")
    lines.append(
        "SE and MV make the exact same decision on the large majority of "
        f"jointly-resolved examples (pooled agreement rate "
        f"{summary['pooledExactAgreementRate']:.1%}), and the total number "
        f"of examples where they disagree on correctness "
        f"({summary['totalDiscordantCorrectnessPairs']} out of "
        f"{summary['totalExampleComparisons_15rows_pooled']} comparisons) is "
        "small relative to the shared/agreeing population. This supports "
        "the descriptive claim that whatever correctness differences exist "
        "between SE and MV are concentrated in a small number of discordant "
        "examples, not spread broadly across the jointly-resolved set.\n\n"
        "This is **not** the same as concluding SE and MV are statistically "
        "equivalent. Every McNemar p-value in the 15-row table is "
        "non-significant, but several rows have only 0-3 discordant pairs — "
        "at that count, the test has very little power to detect anything "
        "but a large effect, so \"non-significant\" here mainly reflects "
        "\"too few discordant examples to say anything,\" not confirmed "
        "sameness. The 95% CIs on the paired accuracy difference are wide "
        "relative to the point estimates in most rows, which is the more "
        "honest way to see the same limitation: the data are compatible "
        "with a range of true differences, not pinned down to zero.\n"
    )

    with open(os.path.join(OUT_DIR, "seMvAgreement.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    rows, summary = main()
    writeMarkdown(rows, summary)

    print("=== KEY RESULTS ===")
    print(f"Overall (pooled) exact agreement rate across 15 comparisons: {summary['pooledExactAgreementRate']:.4f}")
    print(f"Total disagreements: {summary['totalDisagreeCount']} (of {summary['totalExampleComparisons_15rows_pooled']} example-comparisons)")
    print(f"Total discordant correctness pairs: {summary['totalDiscordantCorrectnessPairs']}")
    print(f"Range of agreement rates across the 15 rows: [{summary['minAgreementRateAcross15Rows']:.3f}, {summary['maxAgreementRateAcross15Rows']:.3f}]")
    print(f"Range of paired accuracy differences (SE-MV) across the 15 rows: [{summary['minPairedAccDiffAcross15Rows']:+.3f}, {summary['maxPairedAccDiffAcross15Rows']:+.3f}]")
