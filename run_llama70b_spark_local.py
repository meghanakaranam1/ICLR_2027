"""
Completes the remaining llama_70b x AEGIS work (1158 units missing from
analysis/frontier_model/llama_70b/, all in AEGIS -- HateModerate is already
400/400 for this model) using a LOCAL Ollama instance instead of HopGPT.

Intended to run ON SPARK (or any machine with enough RAM/VRAM for a 70B
model), not on the M4 -- this machine's 16GB RAM cannot host a 70B model at
any usable quantization.

Reuses run_all_datasets.py's ucb_node/run_graph_mv/run_graph_ucb/_call
COMPLETELY UNMODIFIED -- no monkeypatching needed at all, since its default
_post_chat already targets local Ollama (OLLAMA_BASE_URL =
http://localhost:11434), which is correct as long as this script runs
directly on the machine serving Ollama.

Writes to a SEPARATE directory (llama_70b_spark_local/, not
llama_70b/) rather than merging into the original HopGPT-routed
llama_70b/completed.jsonl -- deliberate, not an oversight: Ollama's local
serving stack (a specific GGUF quantization) is not guaranteed to be
byte-identical to whatever HopGPT/Bedrock served for the original
llama3-3-70b-instruct calls, so keeping them in separate files preserves
the ability to check for/report any systematic difference later, rather
than silently conflating two different serving backends under one label.

Usage (run from the ICLR_2027 repo root, on Spark):
    python3 run_llama70b_spark_local.py --model llama3.3:70b --resume
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_all_datasets as m
import run_extension_n400 as ext

DATASET = "aegis"
METHODS = ["single_call", "graph_mv", "graph_ucb_B75", "graph_ucb_B100", "graph_ucb_B124"]
BUDGETS = {"graph_ucb_B75": 75, "graph_ucb_B100": 100, "graph_ucb_B124": 124}

ORIGINAL_COMPLETED_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "analysis", "frontier_model", "llama_70b", "completed.jsonl")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "analysis", "frontier_model", "llama_70b_spark_local")


def loadOriginalCompletedKeys():
    keys = set()
    if os.path.isfile(ORIGINAL_COMPLETED_PATH):
        with open(ORIGINAL_COMPLETED_PATH) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["dataset"] == DATASET:
                    keys.add((r["dataset"], r["exampleId"], r["method"], r.get("budget")))
    return keys


def loadOwnCompletedKeys():
    path = os.path.join(OUT_DIR, "completed.jsonl")
    keys = set()
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                keys.add((r["dataset"], r["exampleId"], r["method"], r.get("budget")))
    return keys


def buildManifest():
    originalDone = loadOriginalCompletedKeys()
    cfg = m.DATASET_CONFIGS[DATASET]
    examples = cfg["loader"]()[:400]
    manifest = []
    for ex in examples:
        for method in METHODS:
            budget = BUDGETS.get(method)
            key = (DATASET, ex["id"], method, budget)
            if key not in originalDone:
                manifest.append((ex, method, budget))
    return manifest


def appendJsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def runOneUnit(modelId, ex, method, budget, nodeSystems, nodeArms):
    if method == "single_call":
        validArms = set(nodeArms["Screener"])
        system = nodeSystems["Screener"]
        user = m.MV_USER.format(arms=m.format_arms(nodeArms["Screener"]), text=ex["text"])
        text, it, ot = m._call(modelId, system, user, max_tokens=10)
        try:
            label = m._parse_label(text, validArms)
        except m.LabelParseError:
            label = "parse_failed"
        return {"final_label": label, "total_pulls": 1}
    elif method == "graph_mv":
        return m.run_graph_mv(modelId, ex["text"], nodeSystems, nodeArms, sanity=False)
    else:
        return m.run_graph_ucb(modelId, ex["text"], budget, nodeSystems, nodeArms, sanity=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Exact Ollama model tag, e.g. llama3.3:70b")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many units this invocation (for a quick smoke test).")
    args = parser.parse_args()

    # Fail fast and loud if Ollama isn't reachable or the model isn't pulled --
    # same style check as run_all_datasets.py's own main() does.
    import requests
    try:
        requests.get(f"{m.OLLAMA_BASE_URL}/api/version", timeout=5).raise_for_status()
    except Exception:
        sys.exit(f"ERROR: Ollama doesn't appear to be running at {m.OLLAMA_BASE_URL}. Start it with `ollama serve`.")

    os.makedirs(OUT_DIR, exist_ok=True)

    manifest = buildManifest()
    ownDone = loadOwnCompletedKeys()
    todo = [(ex, method, budget) for (ex, method, budget) in manifest
            if (DATASET, ex["id"], method, budget) not in ownDone]

    print(f"Total missing from original llama_70b run (AEGIS only): {len(manifest)}")
    print(f"Already completed by this Spark-local run (resumed): {len(ownDone)}")
    print(f"Remaining to process now: {len(todo)}")

    if args.limit:
        todo = todo[:args.limit]
        print(f"--limit: processing only {len(todo)} unit(s) this invocation.")

    nodeSystems = ext.get_node_systems(DATASET)
    cfg = m.DATASET_CONFIGS[DATASET]
    nodeArms = m.build_node_arms(cfg["labels"])

    nOk = nErr = 0
    for ex, method, budget in todo:
        budgetLabel = f"B={budget}" if budget else method
        try:
            t0 = time.time()
            result = runOneUnit(args.model, ex, method, budget, nodeSystems, nodeArms)
            elapsed = time.time() - t0
            record = {
                "model": args.model, "servingBackend": "ollama_local_spark",
                "dataset": DATASET, "exampleId": ex["id"], "method": method, "budget": budget,
                "groundTruth": ex["label"], "finalLabel": result.get("final_label"),
                "totalPulls": result.get("total_pulls"), "elapsedSeconds": round(elapsed, 3),
                "finalConfidence": result.get("final_confidence"),
                "leadingCandidate": result.get("leading_candidate"),
                "timestamp": time.time(),
            }
            appendJsonl(os.path.join(OUT_DIR, "completed.jsonl"), record)
            nOk += 1
            print(f"  OK aegis/{ex['id']}/{method}/{budgetLabel}: {record['finalLabel']} "
                  f"(pulls={record['totalPulls']}, {elapsed:.1f}s)")
        except Exception as e:
            errRecord = {"model": args.model, "dataset": DATASET, "exampleId": ex["id"],
                         "method": method, "budget": budget, "error": str(e), "timestamp": time.time()}
            appendJsonl(os.path.join(OUT_DIR, "errors.jsonl"), errRecord)
            nErr += 1
            print(f"  ERROR aegis/{ex['id']}/{method}/{budgetLabel}: {e}")

    print(f"\n{nOk} succeeded, {nErr} errored this run.")
    finalDone = loadOwnCompletedKeys()
    print(f"Total completed by Spark-local run so far: {len(finalDone)}/{len(manifest)}")


if __name__ == "__main__":
    main()
