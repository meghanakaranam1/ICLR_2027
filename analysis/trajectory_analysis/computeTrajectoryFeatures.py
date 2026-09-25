"""
SE trajectory-feature analysis: does the successive-elimination sampling
trajectory carry information that distinguishes (A) correct vs incorrect
resolved predictions, and separately (B) resolved vs escalated examples?

Read-only over the existing N=400 SE (B=75/100/124) raw.csv outputs. No LLM
calls, no changes to the SE algorithm or to any existing result file.

--------------------------------------------------------------------------
SCHEMA FINDINGS FROM INSPECTION (read before trusting any feature below):

  arm_elimination_trace: JSON list of {"node": <name>, "trace": [...]}, one
  entry per graph node actually evaluated (Screener, then Analyst/
  Adjudicator only if the prior node(s) were non-terminal). Each node's
  "trace" is a list of {"round", "active_before", "eliminated",
  "active_after"} dicts, one per sampling round for that node.

  CONFIRMED BY DIRECT INSPECTION: for every node's trace checked, "eliminated"
  is empty ([]) at every round except (at most) the very last round, where
  either all-but-one candidate is cut at once (a resolved node) or nothing
  is ever cut (the node never converges within budget -> non-terminal for
  that node). There is no incremental, round-by-round narrowing recorded --
  elimination is a single terminal event, not a gradual process. This means
  several of the originally-requested features (early vote margin,
  trajectory stability, modal-label-change count, "did the leader ever
  change") are NOT reconstructable: no per-round vote proportions or
  per-round leading-candidate are stored anywhere in the trace, only the
  eliminated/active SET at each round. These features are explicitly
  SKIPPED below (not approximated, not guessed) and reported as such in
  SUMMARY.md, per instruction.

  arm_estimates_json / final_confidence / leading_estimate are POPULATED
  ONLY for rows generated after 2026-09-20 (confirmed: present for row index
  >= ~100-105 in every file checked, i.e. the N=400 extension rows; blank
  for the original N=100 baseline rows carried forward from before that
  schema addition). leading_candidate itself has much broader coverage
  (essentially 100% of resolved rows, ~97% of escalated rows) because it
  was backfilled separately in an earlier session. Any feature that depends
  on arm_estimates_json (finalVoteMargin, finalEntropy) is therefore only
  computable for a SUBSET of rows -- this is reported explicitly per
  pair/budget, not silently interpolated or imputed.

  numberOfSurvivingCandidates at termination is, by construction, ALWAYS
  exactly 1 for every RESOLVED row (confirmed: 0 exceptions across the rows
  checked) -- a resolved final_label can only arise from a node whose trace
  ends with a single survivor. This makes the feature CONSTANT (zero
  variance) within the resolved population, so it cannot discriminate
  correct vs incorrect (Question A) by definition, and is only meaningful
  for Question B (resolved vs escalated), where it is close to tautological
  with the resolved/escalated label itself -- reported as such, not
  presented as a "finding".
--------------------------------------------------------------------------

Outputs (new files only, nothing existing is modified):
  analysis/seTrajectoryFeatures.csv     (one row per resolved+escalated example, all pairs/budgets)
  analysis/seCorrectVsIncorrect.csv     (Question A summary stats + Mann-Whitney)
  analysis/seResolvedVsEscalated.csv    (Question B summary stats + Mann-Whitney)
  analysis/seAuroc.csv                  (AUROC/AUPRC for both questions, per pair + aggregate)
  analysis/trajectory_analysis/figures/*.png|svg
  analysis/trajectory_analysis/SUMMARY.md
"""

import os
import csv
import json
import math
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULT_ROOTS = ["results_02_n400_from_spark", "results_02"]
NON_TERMINAL = {"escalate", "self_refused", "prompt_injected", "parse_failed"}
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


def isResolved(row):
    return row["final_label"] not in NON_TERMINAL


def safeJson(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def safeFloat(text):
    if text is None or text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def extractFeatures(row):
    """Extract every reliably-reconstructable trajectory feature for one row.
    Returns a dict; a feature is set to None (not fabricated) whenever the
    underlying data isn't present for THIS row, per instruction."""
    feats = {}

    feats["totalPulls"] = safeFloat(row.get("total_pulls"))
    feats["finalConfidence"] = safeFloat(row.get("final_confidence"))
    feats["finalLeadingCandidate"] = row.get("leading_candidate") or None

    trace = safeJson(row.get("arm_elimination_trace"))
    if trace:
        terminatingNode = trace[-1]
        rounds = terminatingNode["trace"]
        feats["eliminationRound"] = rounds[-1]["round"]  # last round of the terminating node -- see module docstring: elimination is single-shot, this equals total rounds run for that node
        survivors = rounds[-1]["active_after"]
        feats["numberOfSurvivingCandidates"] = len(survivors)
        # numberOfEliminations: count of rounds with a non-empty 'eliminated' list (confirmed 0 or 1 across the data inspected -- reported as such, not treated as graded)
        feats["numberOfEliminations"] = sum(1 for r in rounds if r["eliminated"])
    else:
        feats["eliminationRound"] = None
        feats["numberOfSurvivingCandidates"] = None
        feats["numberOfEliminations"] = None

    aej = safeJson(row.get("arm_estimates_json"))
    if aej:
        realKeys = {k: v for k, v in aej.items() if k not in NON_TERMINAL}
        if len(realKeys) >= 1:
            sortedVals = sorted(realKeys.values(), reverse=True)
            top = sortedVals[0]
            second = sortedVals[1] if len(sortedVals) > 1 else 0.0
            feats["finalVoteMargin"] = top - second
            # entropy over the REAL-label distribution only (matches what
            # "final empirical label distribution" means for a 2-way task;
            # including escalate/self_refused/prompt_injected as extra
            # categories would conflate difficulty-signal with the
            # resolution decision itself, which Q6 already studies directly)
            probs = [v for v in realKeys.values() if v > 0]
            entropy = -sum(p * math.log2(p) for p in probs) if probs else None
            feats["finalEntropy"] = entropy
        else:
            feats["finalVoteMargin"] = None
            feats["finalEntropy"] = None
    else:
        feats["finalVoteMargin"] = None
        feats["finalEntropy"] = None

    # NOT reconstructable from stored data -- explicitly left absent, not
    # guessed (see module docstring): earlyVoteMargin, trajectoryStability,
    # modalLabelChanges, "did leader ever change after first becoming leader".

    return feats


def buildFeatureTable():
    allRows = []
    for modelDir, dataset in PAIRS:
        for budget in BUDGETS:
            path = findBestSourceFile(modelDir, dataset, f"graph_ucb_B{budget}")
            if path is None:
                raise RuntimeError(f"STOP: missing graph_ucb_B{budget} for {modelDir}/{dataset}")
            rows = list(csv.DictReader(open(path)))
            for r in rows:
                resolved = isResolved(r)
                correct = (r["final_label"] == r["ground_truth"]) if resolved else None
                feats = extractFeatures(r)
                allRows.append({
                    "model": modelDir, "dataset": dataset, "budget": budget,
                    "inputId": r["input_id"], "finalLabel": r["final_label"],
                    "groundTruth": r["ground_truth"], "resolved": resolved,
                    "correct": correct,
                    **feats,
                })
    return allRows


FEATURE_NAMES = [
    "finalConfidence", "totalPulls", "eliminationRound", "finalVoteMargin",
    "finalEntropy", "numberOfEliminations", "numberOfSurvivingCandidates",
]


def summarizeGroup(values):
    vals = np.array([v for v in values if v is not None], dtype=float)
    if len(vals) == 0:
        return dict(n=0, mean=None, median=None, std=None, q25=None, q75=None)
    return dict(
        n=len(vals), mean=float(np.mean(vals)), median=float(np.median(vals)),
        std=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        q25=float(np.percentile(vals, 25)), q75=float(np.percentile(vals, 75)),
    )


def mannWhitney(a, b):
    aVals = [v for v in a if v is not None]
    bVals = [v for v in b if v is not None]
    if len(aVals) < 2 or len(bVals) < 2:
        return dict(n1=len(aVals), n2=len(bVals), u=None, p=None, rankBiserial=None)
    u, p = mannwhitneyu(aVals, bVals, alternative="two-sided")
    # rank-biserial effect size: r = 1 - 2U/(n1*n2)
    rankBiserial = 1 - (2 * u) / (len(aVals) * len(bVals))
    return dict(n1=len(aVals), n2=len(bVals), u=float(u), p=float(p), rankBiserial=float(rankBiserial))


def auroc(scoresPos, scoresNeg):
    """Mann-Whitney U based AUROC (rank-sum method); handles ties via
    average ranks (scipy's rankdata under the hood through mannwhitneyu's
    U statistic is not directly what we want here, so compute directly)."""
    pos = np.array([v for v in scoresPos if v is not None], dtype=float)
    neg = np.array([v for v in scoresNeg if v is not None], dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return None, len(pos), len(neg)
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
    sumPosRanks = posRanks.sum()
    n1, n2 = len(pos), len(neg)
    u1 = sumPosRanks - n1 * (n1 + 1) / 2
    aucVal = u1 / (n1 * n2)
    return float(aucVal), n1, n2


def auprc(scoresPos, scoresNeg):
    """Simple AUPRC via sklearn if available, else manual trapezoid over
    thresholds (treat 'positive' class as the smaller/target group)."""
    try:
        from sklearn.metrics import average_precision_score
    except ImportError:
        return None
    pos = [v for v in scoresPos if v is not None]
    neg = [v for v in scoresNeg if v is not None]
    if not pos or not neg:
        return None
    y = [1] * len(pos) + [0] * len(neg)
    scores = pos + neg
    return float(average_precision_score(y, scores))


def main():
    allRows = buildFeatureTable()

    # write the full feature table
    fieldOrder = ["model", "dataset", "budget", "inputId", "finalLabel", "groundTruth",
                  "resolved", "correct"] + FEATURE_NAMES
    with open(os.path.join(ANALYSIS_DIR, "seTrajectoryFeatures.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldOrder)
        writer.writeheader()
        for row in allRows:
            writer.writerow({k: row.get(k) for k in fieldOrder})

    # ---- Question A: correct vs incorrect (resolved only) ----
    qaRows = []
    for modelDir, dataset in PAIRS:
        for budget in BUDGETS:
            subset = [r for r in allRows if r["model"] == modelDir and r["dataset"] == dataset and r["budget"] == budget and r["resolved"]]
            correctVals = {feat: [r[feat] for r in subset if r["correct"]] for feat in FEATURE_NAMES}
            incorrectVals = {feat: [r[feat] for r in subset if not r["correct"]] for feat in FEATURE_NAMES}
            for feat in FEATURE_NAMES:
                cSumm = summarizeGroup(correctVals[feat])
                iSumm = summarizeGroup(incorrectVals[feat])
                mw = mannWhitney(correctVals[feat], incorrectVals[feat])
                qaRows.append({
                    "model": modelDir, "dataset": dataset, "budget": budget, "feature": feat,
                    "nCorrect": cSumm["n"], "meanCorrect": cSumm["mean"], "medianCorrect": cSumm["median"],
                    "stdCorrect": cSumm["std"], "q25Correct": cSumm["q25"], "q75Correct": cSumm["q75"],
                    "nIncorrect": iSumm["n"], "meanIncorrect": iSumm["mean"], "medianIncorrect": iSumm["median"],
                    "stdIncorrect": iSumm["std"], "q25Incorrect": iSumm["q25"], "q75Incorrect": iSumm["q75"],
                    "mannWhitneyU": mw["u"], "mannWhitneyP": mw["p"], "rankBiserialEffectSize": mw["rankBiserial"],
                })
    with open(os.path.join(ANALYSIS_DIR, "seCorrectVsIncorrect.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(qaRows[0].keys()))
        writer.writeheader()
        writer.writerows(qaRows)

    # ---- Question B: resolved vs escalated (all examples, any budget) ----
    qbRows = []
    for modelDir, dataset in PAIRS:
        for budget in BUDGETS:
            subset = [r for r in allRows if r["model"] == modelDir and r["dataset"] == dataset and r["budget"] == budget]
            resolvedVals = {feat: [r[feat] for r in subset if r["resolved"]] for feat in FEATURE_NAMES}
            escalatedVals = {feat: [r[feat] for r in subset if not r["resolved"]] for feat in FEATURE_NAMES}
            for feat in FEATURE_NAMES:
                rSumm = summarizeGroup(resolvedVals[feat])
                eSumm = summarizeGroup(escalatedVals[feat])
                mw = mannWhitney(resolvedVals[feat], escalatedVals[feat])
                qbRows.append({
                    "model": modelDir, "dataset": dataset, "budget": budget, "feature": feat,
                    "nResolved": rSumm["n"], "meanResolved": rSumm["mean"], "medianResolved": rSumm["median"],
                    "stdResolved": rSumm["std"], "q25Resolved": rSumm["q25"], "q75Resolved": rSumm["q75"],
                    "nEscalated": eSumm["n"], "meanEscalated": eSumm["mean"], "medianEscalated": eSumm["median"],
                    "stdEscalated": eSumm["std"], "q25Escalated": eSumm["q25"], "q75Escalated": eSumm["q75"],
                    "mannWhitneyU": mw["u"], "mannWhitneyP": mw["p"], "rankBiserialEffectSize": mw["rankBiserial"],
                })
    with open(os.path.join(ANALYSIS_DIR, "seResolvedVsEscalated.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(qbRows[0].keys()))
        writer.writeheader()
        writer.writerows(qbRows)

    # ---- AUROC/AUPRC, per pair (pooled over 3 budgets) + aggregate ----
    aurocRows = []
    for taskName, groupKeyFn in [
        ("correctness", lambda r: r["correct"] if r["resolved"] else None),
        ("resolution", lambda r: r["resolved"]),
    ]:
        for modelDir, dataset in PAIRS:
            subset = [r for r in allRows if r["model"] == modelDir and r["dataset"] == dataset]
            if taskName == "correctness":
                subset = [r for r in subset if r["resolved"]]
                posGroup = lambda r: r["correct"] is True
                negGroup = lambda r: r["correct"] is False
            else:
                posGroup = lambda r: r["resolved"] is True
                negGroup = lambda r: r["resolved"] is False
            for feat in FEATURE_NAMES:
                posVals = [r[feat] for r in subset if posGroup(r)]
                negVals = [r[feat] for r in subset if negGroup(r)]
                aucVal, nPos, nNeg = auroc(posVals, negVals)
                aucPr = auprc(posVals, negVals)
                aurocRows.append({
                    "task": taskName, "scope": f"{modelDir}/{dataset}", "feature": feat,
                    "nPositive": nPos, "nNegative": nNeg, "auroc": aucVal, "auprc": aucPr,
                })
        # aggregate: pool feature values across all 5 pairs (labelled explicitly as pooled)
        allSubset = allRows if taskName == "resolution" else [r for r in allRows if r["resolved"]]
        for feat in FEATURE_NAMES:
            if taskName == "correctness":
                posVals = [r[feat] for r in allSubset if r["correct"] is True]
                negVals = [r[feat] for r in allSubset if r["correct"] is False]
            else:
                posVals = [r[feat] for r in allSubset if r["resolved"] is True]
                negVals = [r[feat] for r in allSubset if r["resolved"] is False]
            aucVal, nPos, nNeg = auroc(posVals, negVals)
            aucPr = auprc(posVals, negVals)
            aurocRows.append({
                "task": taskName, "scope": "AGGREGATE_POOLED_ACROSS_5_PAIRS", "feature": feat,
                "nPositive": nPos, "nNegative": nNeg, "auroc": aucVal, "auprc": aucPr,
            })
    with open(os.path.join(ANALYSIS_DIR, "seAuroc.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aurocRows[0].keys()))
        writer.writeheader()
        writer.writerows(aurocRows)

    return allRows, qaRows, qbRows, aurocRows


def makeFigures(allRows, aurocRows):
    pairs = PAIRS
    strongFeatures = ["finalConfidence", "totalPulls", "finalVoteMargin"]

    # Figure 1: correct vs incorrect, pooled across budgets, per pair
    fig, axes = plt.subplots(1, len(strongFeatures), figsize=(15, 5))
    for ax, feat in zip(axes, strongFeatures):
        data, labels = [], []
        for modelDir, dataset in pairs:
            resolved = [r for r in allRows if r["model"] == modelDir and r["dataset"] == dataset and r["resolved"]]
            correctVals = [r[feat] for r in resolved if r["correct"] and r[feat] is not None]
            incorrectVals = [r[feat] for r in resolved if not r["correct"] and r[feat] is not None]
            if correctVals:
                data.append(correctVals); labels.append(f"{modelDir[:6]}/{dataset[:4]}\ncorrect")
            if incorrectVals:
                data.append(incorrectVals); labels.append(f"{modelDir[:6]}/{dataset[:4]}\nwrong")
        if data:
            ax.boxplot(data, labels=labels)
            ax.set_title(feat)
            ax.tick_params(axis="x", rotation=90, labelsize=6)
    fig.suptitle("Figure 1: Correct vs Incorrect (resolved examples, pooled over budgets)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "figures", "figure1_correct_vs_incorrect.png"), dpi=200)
    fig.savefig(os.path.join(OUT_DIR, "figures", "figure1_correct_vs_incorrect.svg"))
    plt.close(fig)

    # Figure 2: resolved vs escalated
    fig, axes = plt.subplots(1, len(strongFeatures), figsize=(15, 5))
    for ax, feat in zip(axes, strongFeatures):
        data, labels = [], []
        for modelDir, dataset in pairs:
            subset = [r for r in allRows if r["model"] == modelDir and r["dataset"] == dataset]
            resolvedVals = [r[feat] for r in subset if r["resolved"] and r[feat] is not None]
            escalatedVals = [r[feat] for r in subset if not r["resolved"] and r[feat] is not None]
            if resolvedVals:
                data.append(resolvedVals); labels.append(f"{modelDir[:6]}/{dataset[:4]}\nresolved")
            if escalatedVals:
                data.append(escalatedVals); labels.append(f"{modelDir[:6]}/{dataset[:4]}\nescalated")
        if data:
            ax.boxplot(data, labels=labels)
            ax.set_title(feat)
            ax.tick_params(axis="x", rotation=90, labelsize=6)
    fig.suptitle("Figure 2: Resolved vs Escalated (all examples, pooled over budgets)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "figures", "figure2_resolved_vs_escalated.png"), dpi=200)
    fig.savefig(os.path.join(OUT_DIR, "figures", "figure2_resolved_vs_escalated.svg"))
    plt.close(fig)

    # Figure 3: AUROC comparison, correctness vs resolution, per feature (aggregate)
    aggRows = [r for r in aurocRows if r["scope"] == "AGGREGATE_POOLED_ACROSS_5_PAIRS"]
    feats = FEATURE_NAMES
    correctnessAuc = [next((r["auroc"] for r in aggRows if r["task"] == "correctness" and r["feature"] == f), None) for f in feats]
    resolutionAuc = [next((r["auroc"] for r in aggRows if r["task"] == "resolution" and r["feature"] == f), None) for f in feats]
    x = np.arange(len(feats))
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.35
    ax.bar(x - w/2, [v if v is not None else 0 for v in correctnessAuc], w, label="A: correctness (correct vs incorrect)")
    ax.bar(x + w/2, [v if v is not None else 0 for v in resolutionAuc], w, label="B: resolution (resolved vs escalated)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance (0.5)")
    ax.set_xticks(x); ax.set_xticklabels(feats, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("AUROC (aggregate, pooled across 5 pairs)")
    ax.set_title("Figure 3: AUROC, correctness prediction vs resolution/difficulty prediction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "figures", "figure3_auroc_comparison.png"), dpi=200)
    fig.savefig(os.path.join(OUT_DIR, "figures", "figure3_auroc_comparison.svg"))
    plt.close(fig)


if __name__ == "__main__":
    allRows, qaRows, qbRows, aurocRows = main()
    makeFigures(allRows, aurocRows)
    print(f"Wrote {len(allRows)} feature rows, {len(qaRows)} Q-A summary rows, "
          f"{len(qbRows)} Q-B summary rows, {len(aurocRows)} AUROC rows.")
