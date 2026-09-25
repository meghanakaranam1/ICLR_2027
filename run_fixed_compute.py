"""
Fixed-compute SE vs MV comparison, addressing Reviewer DgKN22's request for
results under a small total-call budget (B=3,5,10) matched to majority vote.

Reuses the existing pipeline's primitives UNCHANGED:
  - ucb_node()      (single-node successive elimination, untouched)
  - run_node_mv()   (single-node majority vote, untouched -- already
                     accepts an arbitrary n_mv, so MV at budget B is just
                     run_node_mv(..., n_mv=B), no new MV code needed)
  - _call/_post_chat, _parse_label, _is_refusal, _is_prompt_injected
  - DATASET_CONFIGS, node_systems, node_arms, dataset loaders

NEW code in this file (not in run_all_datasets.py) is exactly one thing:
runGraphSeFixedBudget(), a graph-level orchestrator that enforces a TRUE
TOTAL cap of B calls across up to 3 nodes. This is necessary because the
existing run_graph_ucb() passes the SAME budget to every node independently
(confirmed by inspection: each of Screener/Analyst/Adjudicator can use up
to `budget` pulls, so the existing function can use up to 3xB total, which
would violate "no more than B total calls"). The fix here does not change
ucb_node's internal round logic at all -- it only changes what budget value
each node is CALLED WITH, passing the REMAINING budget (B minus total pulls
used by prior nodes so far) instead of the full B every time.

Does NOT modify run_all_datasets.py or any existing results directory.
All output lives under analysis/fixed_compute/.
"""

import os
import sys
import csv
import json
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_all_datasets as m
import run_extension_n400 as ext

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis", "fixed_compute")
os.makedirs(OUT_DIR, exist_ok=True)

PAIRS = [
    ("llama3.1:8b", "llama3.1_8b", "hatemoderate"),
    ("llama3.1:8b", "llama3.1_8b", "aegis"),
    ("qwen2.5:7b", "qwen2.5_7b", "hatemoderate"),
    ("qwen2.5:7b", "qwen2.5_7b", "aegis"),
    ("mistral:7b", "mistral_7b", "hatemoderate"),
]
BUDGETS = [3, 5, 10]
METHODS = ["MV", "SE"]

NON_TERMINAL = {"escalate", "self_refused", "prompt_injected", "parse_failed"}


# ---------------------------------------------------------------------------
# The one new function: a total-budget-capped SE graph orchestrator.
# ---------------------------------------------------------------------------

def runGraphSeFixedBudget(model, text, totalBudget, nodeSystems, nodeArms, sanity=False):
    """Graph-level SE with a HARD TOTAL cap of totalBudget calls across up
    to 3 nodes. Reuses ucb_node() completely unmodified -- the only change
    from run_graph_ucb() is that each node is given the REMAINING budget
    (totalBudget - pullsUsedSoFar), not the full totalBudget every time. If
    the remaining budget is already 0 before a node would run, that node is
    skipped entirely (0 extra calls) and treated as non-terminal (escalate),
    exactly matching what ucb_node itself returns when budget is exhausted
    immediately (assert 'escalate' in arms; return finalize('escalate')) --
    so this is consistent with the existing algorithm's own convention for
    "no budget left", not a new convention invented here."""
    nodesUsed, nodeLabels, traces = [], [], []
    nodeResults = {}
    totalPulls = totalInputTokens = totalOutputTokens = 0
    injectionDetected = False

    for node in m.GRAPH:
        remainingBudget = totalBudget - totalPulls
        if remainingBudget <= 0:
            # No budget left for this node at all -- treat as escalate
            # without spending any additional calls, per the module
            # docstring above.
            nodesUsed.append(node)
            nodeLabels.append("escalate")
            traces.append({"node": node, "trace": [], "note": "skipped: 0 remaining budget"})
            if node == "Adjudicator" or all(l in NON_TERMINAL for l in nodeLabels):
                pass
            continue

        nodesUsed.append(node)
        if node == "Adjudicator" and "Screener" in nodeResults and "Analyst" in nodeResults:
            context = m.build_adjudicator_context_ucb(nodeResults["Screener"], nodeResults["Analyst"])
            r = m.ucb_node(model, node, text, remainingBudget, nodeSystems, nodeArms, sanity, context=context)
        else:
            r = m.ucb_node(model, node, text, remainingBudget, nodeSystems, nodeArms, sanity)
        nodeResults[node] = r
        nodeLabels.append(r["label"])
        traces.append({"node": node, "trace": r["trace"]})
        totalPulls += r["pulls"]; totalInputTokens += r["input_tokens"]; totalOutputTokens += r["output_tokens"]
        injectionDetected = injectionDetected or r.get("injection_detected", False)
        if r["label"] not in NON_TERMINAL:
            break

    lastRealNode = nodeResults.get(nodesUsed[-1]) if nodesUsed and nodesUsed[-1] in nodeResults else None

    if nodeLabels[-1] not in NON_TERMINAL:
        final = nodeLabels[-1]
    elif len(set(nodeLabels)) == 1:
        final = nodeLabels[0]
    else:
        final = "escalate"

    finalConfidence = None if final in NON_TERMINAL else (lastRealNode.get("final_confidence") if lastRealNode else None)

    return {
        "final_label": final, "nodes_used": nodesUsed, "node_labels": nodeLabels,
        "total_pulls": totalPulls, "total_tokens_input": totalInputTokens,
        "total_tokens_output": totalOutputTokens, "cost_usd": 0.0,
        "arm_elimination_trace": json.dumps(traces),
        "injection_detected": injectionDetected,
        "final_confidence": finalConfidence,
        "leading_candidate": lastRealNode.get("leading_candidate") if lastRealNode else None,
        "leading_estimate": lastRealNode.get("leading_estimate") if lastRealNode else None,
        "arm_estimates_json": lastRealNode.get("arm_estimates_json", "") if lastRealNode else "",
    }


# ---------------------------------------------------------------------------
# Per-call logging: monkeypatch _post_chat once, tagging every call with
# whatever context is currently set. Single-threaded per unit (no outer
# concurrency needed at this tiny budget scale), so a plain mutable global
# is safe -- unlike the frontier-model script, there is only ONE model
# process active at a time here (each of the 3 Ollama models is run
# sequentially, one after another, all local).
# ---------------------------------------------------------------------------

_ctx = {"dataset": None, "exampleId": None, "method": None, "budget": None, "callIndex": 0}
_callsFile = None


def setContext(dataset, exampleId, method, budget):
    _ctx.update(dataset=dataset, exampleId=exampleId, method=method, budget=budget, callIndex=0)


_origPostChat = m._post_chat


def loggingPostChat(payload):
    _ctx["callIndex"] += 1
    t0 = time.time()
    record = {
        "model": payload.get("model"), "dataset": _ctx["dataset"], "example_id": _ctx["exampleId"],
        "method": _ctx["method"], "budget": _ctx["budget"], "call_index": _ctx["callIndex"],
        "timestamp": None, "raw_output": None, "parsed_label": None, "error": None,
        "latency_seconds": None,
    }
    try:
        data = _origPostChat(payload)
        record["timestamp"] = time.time()
        record["latency_seconds"] = round(time.time() - t0, 3)
        text = data["choices"][0]["message"]["content"]
        record["raw_output"] = text
        usage = data.get("usage", {})
        record["input_tokens"] = usage.get("prompt_tokens")
        record["output_tokens"] = usage.get("completion_tokens")
        _callsFile.write(json.dumps(record) + "\n")
        _callsFile.flush()
        return data
    except Exception as e:
        record["error"] = str(e)
        record["timestamp"] = time.time()
        _callsFile.write(json.dumps(record) + "\n")
        _callsFile.flush()
        raise


m._post_chat = loggingPostChat


# ---------------------------------------------------------------------------
# Manifest, resumability
# ---------------------------------------------------------------------------

def buildManifest(nPerPair=None):
    manifest = []
    for modelId, modelDir, dataset in PAIRS:
        cfg = m.DATASET_CONFIGS[dataset]
        examples = cfg["loader"]()
        if nPerPair is not None:
            examples = examples[:nPerPair]
        else:
            examples = examples[:400]
        for ex in examples:
            manifest.append({"modelId": modelId, "modelDir": modelDir, "dataset": dataset,
                              "exampleId": ex["id"], "groundTruth": ex["label"]})
    return manifest


def loadCompleted():
    path = os.path.join(OUT_DIR, "completed.jsonl")
    completed = set()
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    completed.add((rec["modelId"], rec["dataset"], rec["exampleId"], rec["method"], rec["budget"]))
    return completed


def appendResult(record):
    with open(os.path.join(OUT_DIR, "results.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")
    with open(os.path.join(OUT_DIR, "completed.jsonl"), "a") as f:
        f.write(json.dumps({"modelId": record["modelId"], "dataset": record["dataset"],
                             "exampleId": record["exampleId"], "method": record["method"],
                             "budget": record["budget"]}) + "\n")


def runOneUnit(modelId, dataset, ex, method, budget, nodeSystems, nodeArms):
    setContext(dataset, ex["id"], method, budget)
    t0 = time.time()
    if method == "MV":
        r = m.run_node_mv(modelId, "Screener", ex["text"], nodeSystems, nodeArms, sanity=False, n_mv=budget)
        finalLabel = r["label"]
        actualCalls = r["pulls"]
        result = {"final_label": finalLabel, "total_pulls": actualCalls,
                  "final_confidence": r.get("final_confidence"), "leading_candidate": r.get("leading_candidate")}
    else:
        result = runGraphSeFixedBudget(modelId, ex["text"], budget, nodeSystems, nodeArms, sanity=False)
        actualCalls = result["total_pulls"]
    elapsed = time.time() - t0

    resolved = result["final_label"] not in NON_TERMINAL
    return {
        "modelId": modelId, "dataset": dataset, "exampleId": ex["id"], "method": method, "budget": budget,
        "groundTruth": ex["label"], "finalLabel": result["final_label"], "resolved": resolved,
        "correct": (result["final_label"] == ex["label"]) if resolved else None,
        "actualCallsUsed": actualCalls, "maxBudget": budget, "elapsedSeconds": round(elapsed, 3),
        "finalConfidence": result.get("final_confidence"), "leadingCandidate": result.get("leading_candidate"),
    }


def runManifest(manifest, label):
    completed = loadCompleted()
    print(f"  {len(completed)} units already completed, skipping those.")
    todo = []
    for rec in manifest:
        for method in METHODS:
            for budget in BUDGETS:
                key = (rec["modelId"], rec["dataset"], rec["exampleId"], method, budget)
                if key in completed:
                    continue
                todo.append((rec, method, budget))
    print(f"  {len(todo)} units to run for {label}.")

    global _callsFile
    _callsFile = open(os.path.join(OUT_DIR, "calls.jsonl"), "a")

    nodeSystemsCache = {}
    nodeArmsCache = {}
    nOk = nErr = 0
    for rec, method, budget in todo:
        dataset = rec["dataset"]
        if dataset not in nodeSystemsCache:
            nodeSystemsCache[dataset] = ext.get_node_systems(dataset)
            cfg = m.DATASET_CONFIGS[dataset]
            nodeArmsCache[dataset] = m.build_node_arms(cfg["labels"])
        cfg = m.DATASET_CONFIGS[dataset]
        exById = {e["id"]: e for e in cfg["loader"]()}
        ex = exById[rec["exampleId"]]
        try:
            result = runOneUnit(rec["modelId"], dataset, ex, method, budget,
                                 nodeSystemsCache[dataset], nodeArmsCache[dataset])
            appendResult(result)
            nOk += 1
            print(f"  OK {rec['modelId']}/{dataset}/{ex['id']}/{method}/B{budget}: "
                  f"label={result['finalLabel']} calls={result['actualCallsUsed']}/{budget}")
        except Exception as e:
            nErr += 1
            with open(os.path.join(OUT_DIR, "errors.jsonl"), "a") as f:
                f.write(json.dumps({"modelId": rec["modelId"], "dataset": dataset, "exampleId": ex["id"],
                                     "method": method, "budget": budget, "error": str(e)}) + "\n")
            print(f"  ERROR {rec['modelId']}/{dataset}/{ex['id']}/{method}/B{budget}: {e}")

    _callsFile.close()
    print(f"  {label}: {nOk} succeeded, {nErr} errored this run.")
    return nOk, nErr


def validatePilot(manifest):
    """Print the validation report required before the full run."""
    completed = {}
    with open(os.path.join(OUT_DIR, "results.jsonl")) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            key = (rec["modelId"], rec["dataset"], rec["method"], rec["budget"])
            completed.setdefault(key, []).append(rec)

    print("\n=== PILOT VALIDATION REPORT ===")
    print(f"{'model':<14}{'dataset':<14}{'method':<5}{'budget':<7}{'n':<5}{'meanCalls':<11}{'maxCalls':<10}{'resolved':<10}{'escalated':<11}{'overBudget':<12}")
    anyOverBudget = 0
    for (modelId, dataset, method, budget), recs in sorted(completed.items()):
        calls = [r["actualCallsUsed"] for r in recs]
        overBudget = sum(1 for c in calls if c > budget)
        anyOverBudget += overBudget
        resolved = sum(1 for r in recs if r["resolved"])
        escalated = len(recs) - resolved
        print(f"{modelId:<14}{dataset:<14}{method:<5}{budget:<7}{len(recs):<5}"
              f"{sum(calls)/len(calls):<11.2f}{max(calls):<10}{resolved:<10}{escalated:<11}{overBudget:<12}")

    # duplicate check
    seen = set()
    dupes = 0
    allRecs = [r for recs in completed.values() for r in recs]
    for r in allRecs:
        key = (r["modelId"], r["dataset"], r["exampleId"], r["method"], r["budget"])
        if key in seen:
            dupes += 1
        seen.add(key)

    print(f"\nTotal pilot units: {len(allRecs)}")
    print(f"Units exceeding their max budget (SHOULD BE 0): {anyOverBudget}")
    print(f"Duplicate result units (SHOULD BE 0): {dupes}")
    parseFailures = sum(1 for r in allRecs if r["finalLabel"] == "parse_failed")
    print(f"Parse failures: {parseFailures}")

    if anyOverBudget > 0:
        print("\n*** VALIDATION FAILED: some units exceeded their max budget. DO NOT proceed to full run. ***")
        return False
    if dupes > 0:
        print("\n*** VALIDATION FAILED: duplicate results detected. DO NOT proceed to full run. ***")
        return False
    print("\nValidation PASSED: budget enforcement correct, no duplicates.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # BUG FIX (caught live): --resume alone used to always build the FULL
    # 400-example manifest regardless of whether the run being resumed was
    # the pilot or the full experiment, which accidentally launched full-
    # scale work when only a pilot retry was intended. Now --resume without
    # --pilot explicitly refuses, to force the caller to say which scope
    # they mean (--pilot --resume to retry only pilot units; --resume alone
    # only proceeds once the pilot has been explicitly approved, tracked
    # via a marker file written by a successful validated pilot run only
    # after the human operator confirms it).
    if args.resume and not args.pilot:
        approvalMarker = os.path.join(OUT_DIR, "FULL_RUN_APPROVED")
        if not os.path.isfile(approvalMarker):
            sys.exit(
                "REFUSING to resume the full run: no approval marker found at "
                f"{approvalMarker}. If you intended to retry pilot units only, "
                "use `python run_fixed_compute.py --pilot --resume`. If the "
                "full run has been explicitly approved, create the marker "
                "file (e.g. `touch analysis/fixed_compute/FULL_RUN_APPROVED`) "
                "and rerun `--resume`."
            )

    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump({"pairs": PAIRS, "budgets": BUDGETS, "methods": METHODS,
                    "temperature": m.TEMPERATURE, "ucbDelta": m.UCB_DELTA}, f, indent=2)

    if args.pilot:
        manifest = buildManifest(nPerPair=10)
        with open(os.path.join(OUT_DIR, "pilot_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        runManifest(manifest, "pilot")
        validatePilot(manifest)
    elif args.resume:
        manifest = buildManifest(nPerPair=None)
        runManifest(manifest, "full (resumed)")
    else:
        manifest = buildManifest(nPerPair=None)
        with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        runManifest(manifest, "full")
