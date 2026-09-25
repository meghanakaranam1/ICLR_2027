"""
Completes the 385 HateModerate examples needed to make Llama 3.3 70B use
the EXACT SAME canonical 400-example set (200 hate / 200 not_hate) as the
primary 7-8B experiments (llama3.1_8b, qwen2.5_7b, mistral_7b -- confirmed
identical IDs across all three). The existing llama_70b HateModerate data
(analysis/frontier_model/llama_70b/) used a DIFFERENT 400-example set (a
plain prefix slice, not the balanced seeded sample), so only 15 of its 400
IDs are safely reusable (one more, ID "124", overlaps by ID but disagrees
on ground truth and is excluded -- see
analysis/frontier_model/llama_3_3_70b_matched_hatemoderate_audit.md).

Driven by an EXPLICIT manifest (llama70b_hatemoderate_missing_manifest.json,
385 entries: {"id":..., "groundTruth":...}) rather than a dataset diff, so
the exact set of IDs to run is fixed and auditable, not re-derived from a
possibly-drifted dataset loader on this machine.

Reuses run_all_datasets.py's ucb_node/run_graph_mv/run_graph_ucb/_call
COMPLETELY UNMODIFIED -- no monkeypatching, since its default _post_chat
targets local Ollama, correct when run directly on the Ollama-hosting
machine (Spark).

Writes to a SEPARATE directory (llama_70b_spark_hatemoderate_matched/) --
does NOT touch analysis/frontier_model/llama_70b/ (the original HopGPT
HateModerate data) or llama_70b_spark_local/ (the AEGIS gap-fill). Backend
note: this uses local Ollama (a GGUF quantization), NOT HopGPT/Bedrock --
a deliberate, explicit choice made because HopGPT was unreachable (JHU VPN
down) when this was needed; flagged, not hidden, per the audit doc's Step 8.

Usage (run from the ICLR_2027 repo root, on Spark):
    python3 run_llama70b_spark_hatemoderate_matched.py --model llama3.3:70b --resume
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_all_datasets as m
import run_extension_n400 as ext

DATASET = "hatemoderate"
METHODS = ["single_call", "graph_mv", "graph_ucb_B75", "graph_ucb_B100", "graph_ucb_B124"]
BUDGETS = {"graph_ucb_B75": 75, "graph_ucb_B100": 100, "graph_ucb_B124": 124}

MANIFEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "analysis", "frontier_model", "llama70b_hatemoderate_missing_manifest.json")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "analysis", "frontier_model", "llama_70b_spark_hatemoderate_matched")


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

    import requests
    try:
        requests.get(f"{m.OLLAMA_BASE_URL}/api/version", timeout=5).raise_for_status()
    except Exception:
        sys.exit(f"ERROR: Ollama doesn't appear to be running at {m.OLLAMA_BASE_URL}. Start it with `ollama serve`.")

    if not os.path.isfile(MANIFEST_PATH):
        sys.exit(f"ERROR: manifest not found at {MANIFEST_PATH} -- copy "
                  f"analysis/frontier_model/llama70b_hatemoderate_missing_manifest.json onto Spark first.")

    manifestIds = json.load(open(MANIFEST_PATH))
    groundTruthById = {e["id"]: e["groundTruth"] for e in manifestIds}

    cfg = m.DATASET_CONFIGS[DATASET]
    allExamples = cfg["loader"]()
    byId = {e["id"]: e for e in allExamples}

    missingFromLoader = [i for i in groundTruthById if i not in byId]
    if missingFromLoader:
        print(f"WARNING: {len(missingFromLoader)} manifest IDs not found in this machine's "
              f"HateModerate loader output -- dataset drift risk. First few: {missingFromLoader[:5]}")

    examples = [byId[i] for i in groundTruthById if i in byId]
    # Sanity: manifest's ground truth must match what THIS machine's loader says too,
    # not just the canonical primary source -- catch a SECOND drift point if any.
    mismatches = [e["id"] for e in examples if e["label"] != groundTruthById[e["id"]]]
    if mismatches:
        print(f"WARNING: {len(mismatches)} examples where this machine's loader ground truth "
              f"disagrees with the manifest's canonical ground truth: {mismatches[:5]} -- "
              f"using the LOADER's own label for consistency with how ground_truth is normally sourced.")

    os.makedirs(OUT_DIR, exist_ok=True)

    manifest = [(ex, method, BUDGETS.get(method)) for ex in examples for method in METHODS]
    ownDone = loadOwnCompletedKeys()
    todo = [(ex, method, budget) for (ex, method, budget) in manifest
            if (DATASET, ex["id"], method, budget) not in ownDone]

    print(f"Manifest examples: {len(examples)} (target 385)")
    print(f"Total units (examples x 5 methods): {len(manifest)}")
    print(f"Already completed by this run (resumed): {len(ownDone)}")
    print(f"Remaining to process now: {len(todo)}")

    if args.limit:
        todo = todo[:args.limit]
        print(f"--limit: processing only {len(todo)} unit(s) this invocation.")

    nodeSystems = ext.get_node_systems(DATASET)
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
            print(f"  OK hatemoderate/{ex['id']}/{method}/{budgetLabel}: {record['finalLabel']} "
                  f"(pulls={record['totalPulls']}, {elapsed:.1f}s)")
        except Exception as e:
            errRecord = {"model": args.model, "dataset": DATASET, "exampleId": ex["id"],
                         "method": method, "budget": budget, "error": str(e), "timestamp": time.time()}
            appendJsonl(os.path.join(OUT_DIR, "errors.jsonl"), errRecord)
            nErr += 1
            print(f"  ERROR hatemoderate/{ex['id']}/{method}/{budgetLabel}: {e}")

    print(f"\n{nOk} succeeded, {nErr} errored this run.")
    finalDone = loadOwnCompletedKeys()
    print(f"Total completed by this run so far: {len(finalDone)}/{len(manifest)}")


if __name__ == "__main__":
    main()
