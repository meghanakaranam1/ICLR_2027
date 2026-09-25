"""
Proper per-pair compute-matched MV ("isocompute") calibration + final run.

Reuses run_all_datasets.py's _call/_post_chat/DATASET_CONFIGS/node-system
builders/run_condition COMPLETELY UNMODIFIED -- that file is never edited by
this script. The one thing this script does NOT reuse verbatim is
run_node_mv/run_graph_mv's ThreadPoolExecutor concurrency: those primary
functions hard-code max_workers=min(n_mv,20), which this session confirmed
overwhelms the local Ollama server (only ~4 real parallel slots) badly
enough to produce a near-total stall (0 completed candidates after 30
minutes, many requests failing after all 4 retries). Per explicit
instruction, this script instead carries a LOCAL, calibration-only
duplicate of run_node_mv/run_graph_mv (calibRunNodeMv/calibRunGraphMv below)
with a configurable, low-default max_workers -- run_all_datasets.py's own
run_node_mv/run_graph_mv are untouched and still used unmodified for the
final N=400 run (Step 9), since by then Ollama's queue depth is no longer
the bottleneck once max_workers is reduced at the SOURCE of the concurrency
(this script), not by patching the primary implementation.

_call/_post_chat themselves are reused byte-for-byte -- confirmed by audit
(see chat transcript) that every value _call returns already corresponds to
exactly one successful HTTP response; a fully-exhausted retry raises
RuntimeError rather than silently producing a fabricated/ambiguous result.
The fix needed was operational (concurrency, Step 2) and harness-level
(per-example try/except + incremental checkpointing, below), not a change
to _call/_post_chat's counting semantics.
"""
import os
import sys
import csv
import json
import time
import argparse
import statistics
import collections
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_all_datasets as m

PAIRS = [
    ("llama3.1_8b", "llama3.1:8b", "hatemoderate"),
    ("llama3.1_8b", "llama3.1:8b", "aegis"),
    ("qwen2.5_7b", "qwen2.5:7b", "hatemoderate"),
    ("qwen2.5_7b", "qwen2.5:7b", "aegis"),
    ("mistral_7b", "mistral:7b", "hatemoderate"),
]
ROOTS = [os.path.join(m.BASE_DIR, "results_02_n400_from_spark"),
         os.path.join(m.BASE_DIR, "results_02")]

CALIB_N_PER_CLASS = 12   # ~24 total, within the requested 20-30 range
CALIB_SEED = 20260925    # distinct from SEED=42 used for the primary 400
CALIB_MAX_WORKERS = 2     # wave size -- see waved-cooldown pulling below
COOLDOWN_SECONDS = 8      # pause before the NEXT wave, only when the
                           # PREVIOUS wave had any timeout/failure -- lets
                           # orphaned server-side generations (client
                           # timeout does not cancel server-side work, see
                           # the Step 3 audit) actually clear before firing
                           # more concurrent load, instead of compounding
                           # the backlog wave after wave.

OUT_DIR = os.path.join(m.BASE_DIR, "analysis", "risk_coverage")
FROZEN_BUDGETS_PATH = os.path.join(OUT_DIR, "isocompute_frozen_budgets.json")


# ── Calibration-only duplicate of run_node_mv/run_graph_mv, LOCAL to this ──
# ── script. run_all_datasets.py is never imported-and-monkeypatched here, ──
# ── never edited -- this is a parallel implementation for the calibration ──
# ── harness only, with the ONE deliberate difference being max_workers.  ──

def calibRunNodeMv(model, node, text, node_systems, node_arms, context=None,
                    n_mv=m.N_MV, max_workers=CALIB_MAX_WORKERS):
    system = node_systems[node]
    valid = set(node_arms[node])
    base_user = m.MV_USER.format(arms=m.format_arms(node_arms[node]), text=text)
    user = f"{context}\n\n{base_user}" if context else base_user

    successes, failures = 0, 0
    votes, total_it, total_ot = [], 0, 0
    injection_detected = False

    def onePull(_):
        return m._call(model, system, user, max_tokens=10)

    # Waved pulling: fire at most `max_workers` pulls at once, wait for that
    # WHOLE wave to finish, THEN decide whether to cool down before the next
    # wave -- request -> wait -> (success: continue immediately) / (timeout:
    # cooldown, then next wave). Never submits a new wave while the previous
    # one might still have orphaned server-side work in flight, unlike the
    # earlier all-at-once-submit design which let retried/timed-out
    # generations pile up faster than the server could clear them.
    remaining = n_mv
    responses = []
    while remaining > 0:
        waveSize = min(remaining, max_workers)
        with ThreadPoolExecutor(max_workers=waveSize) as executor:
            futures = [executor.submit(onePull, i) for i in range(waveSize)]
            waveHadFailure = False
            for fut in futures:
                try:
                    resp, it, ot = fut.result()
                    successes += 1
                    responses.append((resp, it, ot))
                except Exception:
                    failures += 1
                    waveHadFailure = True
        remaining -= waveSize
        if waveHadFailure and remaining > 0:
            time.sleep(COOLDOWN_SECONDS)

    for resp, it, ot in responses:
        total_it += it
        total_ot += ot
        if m._is_refusal(resp):
            votes.append("self_refused")
            continue
        if m._is_prompt_injected(resp):
            votes.append("prompt_injected")
            injection_detected = True
            continue
        try:
            label = m._parse_label(resp, valid)
        except m.LabelParseError as e:
            e.node = node
            raise
        votes.append(label)

    if not votes:
        raise RuntimeError(f"node {node}: all {n_mv} pulls failed (0 successful responses)")

    final = collections.Counter(votes).most_common(1)[0][0]
    return {"label": final, "votes": votes, "pulls": len(votes), "requestedPulls": n_mv,
            "successfulCalls": successes, "failedCalls": failures,
            "input_tokens": total_it, "output_tokens": total_ot,
            "injection_detected": injection_detected}


class PartialGraphFailure(RuntimeError):
    """Raised when a node fails entirely (0 successful pulls) partway
    through the graph. Carries how many nodes DID complete successfully
    before the failure, and their accumulated pulls/calls, so the caller can
    report 'partial graph result' (some nodes succeeded) vs 'no usable
    result at all' (failed on the very first node) instead of collapsing
    both into one undifferentiated failure -- per explicit instruction not
    to silently treat a failed example as equivalent to any other outcome."""
    def __init__(self, message, nodesCompletedBeforeFailure, pullsBeforeFailure,
                 successfulCallsBeforeFailure, failedNode):
        super().__init__(message)
        self.nodesCompletedBeforeFailure = nodesCompletedBeforeFailure
        self.pullsBeforeFailure = pullsBeforeFailure
        self.successfulCallsBeforeFailure = successfulCallsBeforeFailure
        self.failedNode = failedNode


def calibRunGraphMv(model, text, node_systems, node_arms, n_mv=m.N_MV, max_workers=CALIB_MAX_WORKERS):
    node_results = {}
    node_labels = []
    total_pulls = total_successes = total_failures = 0
    for node in m.GRAPH:
        try:
            if node == "Adjudicator":
                context = m.build_adjudicator_context_mv(node_results["Screener"], node_results["Analyst"])
                r = calibRunNodeMv(model, node, text, node_systems, node_arms, context=context,
                                    n_mv=n_mv, max_workers=max_workers)
            else:
                r = calibRunNodeMv(model, node, text, node_systems, node_arms,
                                    n_mv=n_mv, max_workers=max_workers)
        except Exception as e:
            raise PartialGraphFailure(
                f"node {node} failed entirely: {e}",
                nodesCompletedBeforeFailure=list(node_results.keys()),
                pullsBeforeFailure=total_pulls,
                successfulCallsBeforeFailure=total_successes,
                failedNode=node,
            ) from e
        node_results[node] = r
        node_labels.append(r["label"])
        total_pulls += r["pulls"]
        total_successes += r["successfulCalls"]
        total_failures += r["failedCalls"]
        if r["label"] not in m.NON_TERMINAL_LABELS:
            break
    if node_labels[-1] not in m.NON_TERMINAL_LABELS:
        final = node_labels[-1]
    elif len(set(node_labels)) == 1:
        final = node_labels[0]
    else:
        final = "escalate"
    return {"final_label": final, "total_pulls": total_pulls,
            "totalSuccessfulCalls": total_successes, "totalFailedCalls": total_failures,
            "nodesUsed": list(node_results.keys())}


# ── Shared helpers (final-400-ID resolution, calibration-set construction) ──

def findBestB124File(modelDir, dataset):
    best, bestN = None, -1
    for root in ROOTS:
        p = os.path.join(root, modelDir, dataset, "graph_ucb_B124", "raw.csv")
        if os.path.isfile(p):
            with open(p) as f:
                n = sum(1 for _ in f) - 1
            if n > bestN:
                best, bestN = p, n
    return best


def loadFinal400(modelDir, dataset):
    path = findBestB124File(modelDir, dataset)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    ids = [r["input_id"] for r in rows]
    pulls = [int(r["total_pulls"]) for r in rows]
    return ids, statistics.mean(pulls), statistics.median(pulls), path


def buildCalibrationSet(dataset, finalIds):
    import random
    cfg = m.DATASET_CONFIGS[dataset]
    labels = cfg["labels"]
    pos, neg = labels
    allEx = cfg["loader"]()
    finalIdSet = set(finalIds)
    candidates = [e for e in allEx if e["id"] not in finalIdSet]
    posEx = [e for e in candidates if e["label"] == pos]
    negEx = [e for e in candidates if e["label"] == neg]
    rng = random.Random(CALIB_SEED)
    k = min(CALIB_N_PER_CLASS, len(posEx), len(negEx))
    return rng.sample(posEx, k) + rng.sample(negEx, k)


def nodeSystemsFor(dataset):
    return m.build_hatemoderate_node_systems() if dataset == "hatemoderate" else m.build_aegis_node_systems()


# ── Smoke test (Step 4) ──────────────────────────────────────────────────

def runSmokeTest():
    modelDir, modelTag, dataset = "llama3.1_8b", "llama3.1:8b", "hatemoderate"
    finalIds, targetMean, targetMedian, b124Path = loadFinal400(modelDir, dataset)
    nodeSystems = nodeSystemsFor(dataset)
    cfg = m.DATASET_CONFIGS[dataset]
    nodeArms = m.build_node_arms(cfg["labels"])

    import random
    allEx = cfg["loader"]()
    finalIdSet = set(finalIds)
    candidates = [e for e in allEx if e["id"] not in finalIdSet]
    rng = random.Random(CALIB_SEED)
    smokeSet = rng.sample(candidates, 8)

    nMv = 180
    print(f"=== SMOKE TEST: {modelDir}/{dataset}, n_mv={nMv}, N={len(smokeSet)}, max_workers={CALIB_MAX_WORKERS} ===")
    print(f"(reference: SE B124 mean total_pulls = {targetMean:.1f}, median = {targetMedian:.1f}, from {b124Path})")

    pulls, successes, failures = [], 0, 0
    t0 = time.time()
    for i, ex in enumerate(smokeSet):
        exT0 = time.time()
        try:
            r = calibRunGraphMv(modelTag, ex["text"], nodeSystems, nodeArms, n_mv=nMv, max_workers=CALIB_MAX_WORKERS)
            pulls.append(r["total_pulls"])
            successes += r["totalSuccessfulCalls"]
            failures += r["totalFailedCalls"]
            print(f"  [{i+1}/{len(smokeSet)}] id={ex['id']} total_pulls={r['total_pulls']} "
                  f"successful_calls={r['totalSuccessfulCalls']} failed_calls={r['totalFailedCalls']} "
                  f"nodes={r['nodesUsed']} ({time.time()-exT0:.1f}s)", flush=True)
        except Exception as e:
            print(f"  [{i+1}/{len(smokeSet)}] id={ex['id']} FAILED ENTIRELY: {e} ({time.time()-exT0:.1f}s)", flush=True)
    elapsed = time.time() - t0

    print(f"\n=== SMOKE TEST SUMMARY ===")
    print(f"requested pulls per example (n_mv x up to 3 nodes, early-exit dependent): up to {nMv*3}")
    print(f"successful responses (total across all examples): {successes}")
    print(f"failed calls (exhausted all 4 retries, total): {failures}")
    print(f"examples completed: {len(pulls)}/{len(smokeSet)}")
    if pulls:
        print(f"mean total_pulls/example: {statistics.mean(pulls):.1f}")
        print(f"median total_pulls/example: {statistics.median(pulls):.1f}")
    print(f"wall-clock time: {elapsed:.1f}s")
    print(f"clean (0 failed calls, all examples completed): {failures == 0 and len(pulls) == len(smokeSet)}")


# ── Calibration (Steps 5-8) ──────────────────────────────────────────────

def calibCheckpointPath(modelDir, dataset):
    return os.path.join(OUT_DIR, f"isocompute_calib_checkpoint_{modelDir}_{dataset}.jsonl")


def loadCalibCheckpoint(modelDir, dataset):
    path = calibCheckpointPath(modelDir, dataset)
    done = {}
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                done[(rec["nMv"], rec["exampleId"])] = rec
    return done


def appendCalibCheckpoint(modelDir, dataset, rec):
    path = calibCheckpointPath(modelDir, dataset)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def calibErrorsPath(modelDir, dataset):
    return os.path.join(OUT_DIR, f"isocompute_calib_errors_{modelDir}_{dataset}.jsonl")


def appendCalibError(modelDir, dataset, rec):
    path = calibErrorsPath(modelDir, dataset)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def calibCandidateLogPath(modelDir, dataset):
    return os.path.join(OUT_DIR, f"isocompute_calib_candidates_{modelDir}_{dataset}.jsonl")


def appendCalibCandidateLog(modelDir, dataset, rec):
    """Per-candidate audit record, appended immediately after each candidate
    finishes -- NOT overwritten, so the full sequence of candidates tried
    for this pair survives even if the process is killed/restarted mid-pair.
    This is a separate, coarser-grained audit trail than the per-example
    checkpoint above."""
    path = calibCandidateLogPath(modelDir, dataset)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def runCandidateOnCalibSet(modelTag, dataset, modelDir, nodeSystems, nodeArms, calibSet, nMv, already):
    pulls, successes, failures, nFailedExamples = [], 0, 0, 0
    reused = 0
    attempted = len(calibSet)
    for ex in calibSet:
        key = (nMv, ex["id"])
        if key in already:
            rec = already[key]
            pulls.append(rec["totalPulls"])
            successes += rec["successfulCalls"]
            failures += rec["failedCalls"]
            reused += 1
            continue
        try:
            r = calibRunGraphMv(modelTag, ex["text"], nodeSystems, nodeArms, n_mv=nMv, max_workers=CALIB_MAX_WORKERS)
            rec = {"nMv": nMv, "exampleId": ex["id"], "totalPulls": r["total_pulls"],
                   "successfulCalls": r["totalSuccessfulCalls"], "failedCalls": r["totalFailedCalls"],
                   "finalLabel": r["final_label"]}
            appendCalibCheckpoint(modelDir, dataset, rec)
            pulls.append(r["total_pulls"])
            successes += r["totalSuccessfulCalls"]
            failures += r["totalFailedCalls"]
        except PartialGraphFailure as e:
            nFailedExamples += 1
            partial = len(e.nodesCompletedBeforeFailure) > 0
            errRec = {"nMv": nMv, "exampleId": ex["id"], "failedNode": e.failedNode,
                      "nodesCompletedBeforeFailure": e.nodesCompletedBeforeFailure,
                      "pullsBeforeFailure": e.pullsBeforeFailure,
                      "successfulCallsBeforeFailure": e.successfulCallsBeforeFailure,
                      "hasPartialResult": partial, "error": str(e)}
            appendCalibError(modelDir, dataset, errRec)
            kind = (f"PARTIAL result ({e.nodesCompletedBeforeFailure} completed before {e.failedNode} failed, "
                    f"{e.pullsBeforeFailure} pulls/{e.successfulCallsBeforeFailure} successful calls salvaged but discarded)"
                    if partial else f"NO usable result (failed on the very first node, {e.failedNode})")
            print(f"    WARNING: example {ex['id']} at n_mv={nMv} failed entirely -- {kind}. "
                  f"Excluded from mean/median, not imputed.", flush=True)
        except Exception as e:
            nFailedExamples += 1
            errRec = {"nMv": nMv, "exampleId": ex["id"], "error": str(e), "unexpectedErrorType": type(e).__name__}
            appendCalibError(modelDir, dataset, errRec)
            print(f"    WARNING: example {ex['id']} at n_mv={nMv} failed with unexpected error type "
                  f"{type(e).__name__}: {e}. Excluded from mean/median, not imputed.", flush=True)
    completed = len(pulls)
    return {"nMv": nMv, "pulls": pulls, "successes": successes, "failures": failures,
            "attempted": attempted, "completed": completed,
            "nFailedExamples": nFailedExamples, "reused": reused}


def runAndReportCandidate(modelTag, dataset, modelDir, nodeSystems, nodeArms, calibSet, nMv,
                           already, targetMean, allResults, stage):
    t0 = time.time()
    res = runCandidateOnCalibSet(modelTag, dataset, modelDir, nodeSystems, nodeArms, calibSet, nMv, already)
    meanP = statistics.mean(res["pulls"]) if res["pulls"] else None
    medP = statistics.median(res["pulls"]) if res["pulls"] else None
    diff = abs(meanP - targetMean) if meanP is not None else float("inf")
    elapsed = time.time() - t0

    print(f"    n_mv={nMv:4d}  attempted={res['attempted']} completed={res['completed']} "
          f"failedExamples={res['nFailedExamples']}  meanPulls(completed only)="
          f"{meanP if meanP is None else round(meanP,1)}  medianPulls={medP}  "
          f"|diff|={diff if diff==float('inf') else round(diff,1)}  "
          f"successfulCalls={res['successes']} connectionFailures={res['failures']} "
          f"reusedFromCheckpoint={res['reused']} ({elapsed:.0f}s)", flush=True)

    # Monotonicity check against the previous candidate actually observed so
    # far (by n_mv order, not insertion order) -- flagged, not explained.
    priorPoints = sorted((k, v) for k, v in allResults.items() if v["meanPulls"] is not None)
    if meanP is not None and priorPoints:
        lowerNeighbors = [(k, v) for k, v in priorPoints if k < nMv]
        higherNeighbors = [(k, v) for k, v in priorPoints if k > nMv]
        if lowerNeighbors:
            prevK, prevV = max(lowerNeighbors, key=lambda kv: kv[0])
            if meanP < prevV["meanPulls"]:
                print(f"    NON-MONOTONIC: n_mv={nMv} (mean={meanP:.1f}) < n_mv={prevK} (mean={prevV['meanPulls']:.1f}) "
                      f"-- increasing n_mv produced a LOWER realized mean. Recorded, not explained; "
                      f"needs investigation before this pair's budget is trusted.", flush=True)
        if higherNeighbors:
            nextK, nextV = min(higherNeighbors, key=lambda kv: kv[0])
            if meanP > nextV["meanPulls"]:
                print(f"    NON-MONOTONIC: n_mv={nMv} (mean={meanP:.1f}) > n_mv={nextK} (mean={nextV['meanPulls']:.1f}) "
                      f"-- a SMALLER n_mv produced a HIGHER realized mean. Recorded, not explained; "
                      f"needs investigation before this pair's budget is trusted.", flush=True)

    candRec = {"model": modelDir, "dataset": dataset, "stage": stage, "nMv": nMv,
               "attempted": res["attempted"], "completed": res["completed"],
               "failedExamples": res["nFailedExamples"], "successfulCalls": res["successes"],
               "connectionFailures": res["failures"], "reusedFromCheckpoint": res["reused"],
               "meanTotalPulls": meanP, "medianTotalPulls": medP,
               "targetSeB124Mean": targetMean, "absoluteDifference": diff if diff != float("inf") else None,
               "timestamp": time.time()}
    appendCalibCandidateLog(modelDir, dataset, candRec)

    allResults[nMv] = {"meanPulls": meanP, "medianPulls": medP, "diffFromTarget": diff, **res}
    return allResults[nMv]


def calibrateOnePair(modelDir, modelTag, dataset, finalIds, targetMean):
    calibSet = buildCalibrationSet(dataset, finalIds)
    nodeSystems = nodeSystemsFor(dataset)
    cfg = m.DATASET_CONFIGS[dataset]
    nodeArms = m.build_node_arms(cfg["labels"])
    already = loadCalibCheckpoint(modelDir, dataset)

    print(f"\n=== Calibrating {modelDir}/{dataset} (target B124 mean={targetMean:.1f}, "
          f"calib N={len(calibSet)}, max_workers={CALIB_MAX_WORKERS}) ===", flush=True)

    step = 20
    base = int(round(targetMean / step)) * step
    coarseGrid = sorted(set(max(20, base + d) for d in (-60, -20, 20, 60)))
    print(f"  Stage 1 (coarse): {coarseGrid}")
    allResults = {}
    for nMv in coarseGrid:
        runAndReportCandidate(modelTag, dataset, modelDir, nodeSystems, nodeArms, calibSet, nMv,
                               already, targetMean, allResults, stage="coarse")

    # Stage 1b: if the coarse grid's boundary point still overshoots/undershoots
    # in the SAME direction as its neighbor (confirmed monotonic, per Step 3),
    # the true crossing point lies OUTSIDE the tested range -- extend outward
    # in that direction until the mean brackets the target on both sides, or
    # n_mv=20 floor is hit. A fixed +-10 search around whichever single point
    # happens to have the smallest |diff| would silently miss this (as seen
    # for llama3.1_8b/hatemoderate: n_mv=160, the SMALLEST coarse candidate,
    # already overshoots the target of 228.8 at mean=260, and n_mv=200
    # overshoots further at mean=308.3 -- both monotonic, but the true
    # crossing point is below 160, not between the tested points).
    withMean = sorted((k, v) for k, v in allResults.items() if v["meanPulls"] is not None)
    extendAttempts = 0
    while withMean and extendAttempts < 6:
        lowestK, lowestV = withMean[0]
        highestK, highestV = withMean[-1]
        bracketsBelow = lowestV["meanPulls"] <= targetMean
        bracketsAbove = highestV["meanPulls"] >= targetMean
        if bracketsBelow and bracketsAbove:
            break  # target is already bracketed by tested points
        if not bracketsBelow and lowestK > 20:
            nextK = max(20, lowestK - 40)
            if nextK in allResults:
                break
            print(f"  Stage 1b: lowest tested n_mv={lowestK} still overshoots "
                  f"(mean={lowestV['meanPulls']:.1f} > target={targetMean:.1f}) -- extending downward to {nextK}")
            runAndReportCandidate(modelTag, dataset, modelDir, nodeSystems, nodeArms, calibSet, nextK,
                                   already, targetMean, allResults, stage="extend_down")
        elif not bracketsAbove:
            nextK = highestK + 40
            if nextK in allResults:
                break
            print(f"  Stage 1b: highest tested n_mv={highestK} still undershoots "
                  f"(mean={highestV['meanPulls']:.1f} < target={targetMean:.1f}) -- extending upward to {nextK}")
            runAndReportCandidate(modelTag, dataset, modelDir, nodeSystems, nodeArms, calibSet, nextK,
                                   already, targetMean, allResults, stage="extend_up")
        else:
            break
        withMean = sorted((k, v) for k, v in allResults.items() if v["meanPulls"] is not None)
        extendAttempts += 1

    bestCoarse = min(allResults, key=lambda k: allResults[k]["diffFromTarget"])
    fineGrid = sorted(set(v for v in (bestCoarse - 10, bestCoarse + 10) if v >= 20 and v not in allResults))
    print(f"  Stage 2 (fine, around best point so far {bestCoarse}): {fineGrid}")
    for nMv in fineGrid:
        runAndReportCandidate(modelTag, dataset, modelDir, nodeSystems, nodeArms, calibSet, nMv,
                               already, targetMean, allResults, stage="fine")

    chosenNMv = min(allResults, key=lambda k: (allResults[k]["diffFromTarget"], k))
    chosen = allResults[chosenNMv]

    print(f"\n  --- {modelDir}/{dataset}: all candidates tried (target={targetMean:.1f}) ---")
    print(f"  {'n_mv':>6} {'mean':>8} {'median':>8} {'|diff|':>8} {'completed':>10} {'failed':>7}")
    for nMv in sorted(allResults):
        v = allResults[nMv]
        meanStr = f"{v['meanPulls']:.1f}" if v["meanPulls"] is not None else "n/a"
        diffStr = f"{v['diffFromTarget']:.1f}" if v["diffFromTarget"] != float("inf") else "n/a"
        marker = " <== provisional pick" if nMv == chosenNMv else ""
        print(f"  {nMv:>6} {meanStr:>8} {str(v['medianPulls']):>8} {diffStr:>8} {v['completed']:>10} {v['nFailedExamples']:>7}{marker}")

    print(f"\n  --> PROVISIONAL n_mv={chosenNMv} for {modelDir}/{dataset} "
          f"(calibration mean={chosen['meanPulls']:.1f} over {chosen['completed']} completed examples, "
          f"{chosen['nFailedExamples']} excluded as failed, target={targetMean:.1f}, "
          f"|diff|={chosen['diffFromTarget']:.1f}). "
          f"NOT yet 'compute matched' -- that label is only earned after the full 400-example run "
          f"confirms the realized mean; this is 'calibrated to approximately match realized B124 "
          f"model-call volume' pending that confirmation.")
    return chosenNMv, chosen, allResults, calibSet


def runCalibrationPhase():
    frozen = {}
    for modelDir, modelTag, dataset in PAIRS:
        finalIds, targetMean, targetMedian, b124Path = loadFinal400(modelDir, dataset)
        chosenNMv, chosen, allResults, calibSet = calibrateOnePair(modelDir, modelTag, dataset, finalIds, targetMean)
        key = f"{modelDir}/{dataset}"
        relDiff = chosen["diffFromTarget"] / targetMean if targetMean else None
        frozen[key] = {
            "model": modelDir, "modelTag": modelTag, "dataset": dataset,
            "status": "PROVISIONAL",  # only becomes final once the full 400-example run
                                       # confirms the realized mean matches calibration (Step 8)
            "calibrationIds": [ex["id"] for ex in calibSet],
            "targetSeB124MeanTotalPulls": targetMean,
            "targetSeB124MedianTotalPulls": targetMedian,
            "candidateBudgetsTested": sorted(allResults.keys()),
            "chosenNMv": chosenNMv,
            "calibrationAttemptedExamples": chosen["attempted"],
            "calibrationCompletedExamples": chosen["completed"],
            "calibrationMeanTotalPulls": chosen["meanPulls"],
            "calibrationMedianTotalPulls": chosen["medianPulls"],
            "absoluteDifference": chosen["diffFromTarget"],
            "relativeDifference": relDiff,
            "calibSuccessfulCalls": chosen["successes"],
            "calibConnectionFailures": chosen["failures"],
            "calibFailedExamples": chosen["nFailedExamples"],
            "b124SourceFile": b124Path,
            "finalIdsCount": len(finalIds),
            "timestamp": time.time(),
            "codeContext": "run_isocompute_final.py, calibration-local concurrency helper, max_workers=" + str(CALIB_MAX_WORKERS),
        }
    with open(FROZEN_BUDGETS_PATH, "w") as f:
        json.dump(frozen, f, indent=2)
    print(f"\nFrozen budgets written to {FROZEN_BUDGETS_PATH}")
    return frozen


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--calibrate-only", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        runSmokeTest()
        sys.exit(0)

    if args.calibrate_only:
        if os.path.isfile(FROZEN_BUDGETS_PATH):
            print(f"Frozen budgets already exist at {FROZEN_BUDGETS_PATH} -- not recalibrating. "
                  f"Delete that file explicitly if you want to redo calibration.")
            sys.exit(0)
        runCalibrationPhase()
