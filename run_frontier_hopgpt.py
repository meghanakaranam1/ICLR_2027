"""
Resumable frontier-model runner via HopGPT, reusing the existing
run_all_datasets.py pipeline (run_graph_mv, run_graph_ucb, node_systems,
node_arms, dataset loaders, parsing, refusal/injection handling) UNCHANGED.
The only new thing is where model calls are sent: this module monkeypatches
run_all_datasets._post_chat to route to HopGPT instead of local Ollama, with
per-call logging for the IID audit.

Does NOT modify run_all_datasets.py, does NOT touch any existing results
directory. All new output lives under analysis/frontier_model/.

Usage:
    python3 run_frontier_hopgpt.py --compat-test
    python3 run_frontier_hopgpt.py --pilot
    python3 run_frontier_hopgpt.py --resume     (continues whatever --pilot or the full run left unfinished)
"""

import os
import sys
import csv
import json
import time
import argparse
import threading
import contextvars
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_all_datasets as m
import run_extension_n400 as ext

# ---------------------------------------------------------------------------
# Global fix (confirmed necessary by direct test: contextvars do NOT
# propagate into ThreadPoolExecutor worker threads by default in this
# environment -- a plain nested-pool test returned None for all values).
# This patches concurrent.futures.ThreadPoolExecutor.submit process-wide so
# any submitted callable runs inside a copy of the SUBMITTING thread's
# context. This is what makes it safe to raise maxWorkersPerModel above 1:
# ucb_node's/run_graph_mv's own internal ThreadPoolExecutor (used for
# concurrent pulls within a round, unmodified/untouched) now correctly
# inherits whichever unit's contextvar was set by the outer worker thread
# that invoked it, instead of the previous approach (a single shared mutable
# dict) which was only safe with exactly one unit in flight per model.
# Executor.map() is unaffected by name but calls self.submit() internally in
# every CPython version, so it is covered by this patch too.
# ---------------------------------------------------------------------------
_originalSubmit = concurrent.futures.ThreadPoolExecutor.submit


def _contextPropagatingSubmit(self, fn, *args, **kwargs):
    ctx = contextvars.copy_context()
    return _originalSubmit(self, ctx.run, fn, *args, **kwargs)


concurrent.futures.ThreadPoolExecutor.submit = _contextPropagatingSubmit

# Per-call unit context (dataset/exampleId/method/budget), now propagated
# correctly into nested thread pools by the patch above. Replaces the old
# RunnerState.context shared mutable dict, which was ambiguous under
# concurrent units for the same model.
_callContextVar = contextvars.ContextVar("callContext", default=None)

HOPGPT_URL = "https://api.ai.jh.edu/v1/chat/completions"
HOPGPT_KEY_PATH = os.path.expanduser("~/.hopgpt_key")

MODELS = {
    "claude_haiku_4_5": "claude-haiku-4.5",
    "llama_70b": "llama3-3-70b-instruct",
}

DATASETS = ["hatemoderate", "aegis"]
METHODS = ["single_call", "graph_mv", "graph_ucb_B75", "graph_ucb_B100", "graph_ucb_B124"]
BUDGETS = {"graph_ucb_B75": 75, "graph_ucb_B100": 100, "graph_ucb_B124": 124}

OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis", "frontier_model")


def loadHopgptKey():
    with open(HOPGPT_KEY_PATH) as f:
        return f.read().strip()


# ---------------------------------------------------------------------------
# Per-call logging: context now travels via _callContextVar (see above),
# which the global ThreadPoolExecutor.submit patch propagates correctly
# into ucb_node's/run_graph_mv's own internal pull-threads. This makes it
# safe to run MULTIPLE units concurrently for the same model
# (maxWorkersPerModel > 1) -- each outer worker thread sets its own
# contextvar value before calling runOneUnit, and that value (a distinct
# object per unit) is what nested threads inherit, so concurrent units for
# the same model no longer share or clobber a single mutable context.
#
# LIMITATION (documented, not hidden): the exact internal UCB round number
# is not exposed to _post_chat by the existing code (it lives inside
# ucb_node's own loop). callIndex below is a within-unit sequential counter
# (the order calls complete in), not the UCB algorithm's internal round
# label. Reconstructing the true round number would require modifying
# ucb_node's signature, which the task instructs not to do.
# ---------------------------------------------------------------------------

class RunnerState:
    """One instance per model -- isolates log-file/lock/counters so the two
    concurrently-running models never interfere with each other."""
    def __init__(self, safeModelName, modelId, logFile):
        self.safeModelName = safeModelName
        self.modelId = modelId
        self.logFile = logFile
        self.logLock = threading.Lock()
        self.callIndexCounters = {}
        self.callIndexLock = threading.Lock()

    def setCallContext(self, dataset, exampleId, method, budget):
        ctx = {"dataset": dataset, "exampleId": exampleId, "method": method, "budget": budget}
        key = (dataset, exampleId, method, budget)
        with self.callIndexLock:
            self.callIndexCounters[key] = 0
        _callContextVar.set(ctx)

    def nextCallIndex(self, ctx):
        key = (ctx["dataset"], ctx["exampleId"], ctx["method"], ctx["budget"])
        with self.callIndexLock:
            self.callIndexCounters[key] = self.callIndexCounters.get(key, 0) + 1
            return self.callIndexCounters[key]

    def logCall(self, record):
        with self.logLock:
            self.logFile.write(json.dumps(record) + "\n")
            self.logFile.flush()


_runnerRegistry = {}  # modelId (the HopGPT id string, e.g. "claude-haiku-4.5") -> RunnerState
_registryLock = threading.Lock()


def registerRunnerState(state: RunnerState):
    with _registryLock:
        _runnerRegistry[state.modelId] = state


def makeSharedHopgptPostChat(hopgptKey):
    """ONE dispatcher, set as m._post_chat a single time before either
    model's pilot loop starts. This is necessary (not just a stylistic
    choice) because m._post_chat is a single module-level attribute on
    run_all_datasets -- two models monkeypatching it independently while
    running concurrently would race, with whichever model patches last
    silently stealing the other's calls (confirmed as a real risk before
    this fix). Since run_graph_mv/run_graph_ucb already thread the correct
    model-id string through to _call and into payload["model"] unmodified,
    this dispatcher simply reads that string back out to decide which
    model's RunnerState to log against -- it never needs to overwrite it."""
    def hopgptPostChat(payload):
        modelId = payload["model"]
        state = _runnerRegistry[modelId]
        headers = {"Authorization": f"Bearer {hopgptKey}", "Content-Type": "application/json"}
        outgoing = payload
        ctx = _callContextVar.get() or {"dataset": None, "exampleId": None, "method": None, "budget": None}
        callIndex = state.nextCallIndex(ctx)
        t0 = time.time()
        record = {
            "model": modelId, "dataset": ctx["dataset"], "exampleId": ctx["exampleId"],
            "method": ctx["method"], "budget": ctx["budget"], "callIndex": callIndex,
            "timestamp": None, "httpStatus": None, "rawOutput": None,
            "parsedLabel": None, "error": None, "inputTokens": None,
            "outputTokens": None, "latencySeconds": None,
        }
        lastExc = None
        for attempt in range(4):
            try:
                resp = requests.post(HOPGPT_URL, json=outgoing, headers=headers, timeout=90)
                record["httpStatus"] = resp.status_code
                record["latencySeconds"] = round(time.time() - t0, 3)
                record["timestamp"] = time.time()
                if resp.status_code != 200:
                    record["error"] = resp.text[:800]
                    state.logCall(record)
                    raise RuntimeError(f"HopGPT call failed: {resp.status_code}: {resp.text[:300]}")
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                record["rawOutput"] = text
                record["inputTokens"] = usage.get("prompt_tokens")
                record["outputTokens"] = usage.get("completion_tokens")
                state.logCall(record)
                return data
            except requests.exceptions.RequestException as exc:
                lastExc = exc
                wait = min(5 * (2 ** attempt), 20)
                print(f"  [retry {attempt+1}] {exc} -- waiting {wait}s", flush=True)
                time.sleep(wait)
        record["error"] = f"network error after retries: {lastExc}"
        state.logCall(record)
        raise RuntimeError(f"HopGPT call failed after retries: {lastExc}")
    return hopgptPostChat


# ---------------------------------------------------------------------------
# Compatibility test
# ---------------------------------------------------------------------------

def runCompatibilityTest(safeModelName, modelId, hopgptKey):
    print(f"\n=== Compatibility test: {safeModelName} ({modelId}) ===")
    modelDir = os.path.join(OUT_ROOT, safeModelName)
    os.makedirs(modelDir, exist_ok=True)
    logPath = os.path.join(modelDir, "compat_calls.jsonl")

    logFile = open(logPath, "w")
    state = RunnerState(safeModelName, modelId, logFile)
    registerRunnerState(state)
    m._post_chat = makeSharedHopgptPostChat(hopgptKey)

    results = []
    tempRejected = False
    for datasetName in DATASETS:
        cfg = m.DATASET_CONFIGS[datasetName]
        examples = cfg["loader"]()[:2]
        nodeSystems = ext.get_node_systems(datasetName)
        nodeArms = m.build_node_arms(cfg["labels"])
        validArms = set(nodeArms["Screener"])
        for ex in examples:
            if tempRejected:
                results.append({"dataset": datasetName, "exampleId": ex["id"], "status": "SKIPPED"})
                continue
            state.setCallContext(datasetName, ex["id"], "compat", None)
            system = nodeSystems["Screener"]
            user = m.MV_USER.format(arms=m.format_arms(nodeArms["Screener"]), text=ex["text"])
            try:
                text, it, ot = m._call(modelId, system, user, max_tokens=10)
                record = {"dataset": datasetName, "exampleId": ex["id"], "status": "OK",
                          "rawOutput": text, "inputTokens": it, "outputTokens": ot}
                if m._is_refusal(text):
                    record["classification"] = "refusal"
                elif m._is_prompt_injected(text):
                    record["classification"] = "prompt_injected"
                else:
                    try:
                        record["parsedLabel"] = m._parse_label(text, validArms)
                        record["classification"] = "parsed"
                    except m.LabelParseError:
                        record["classification"] = "parse_failure"
            except RuntimeError as e:
                errText = str(e)
                record = {"dataset": datasetName, "exampleId": ex["id"], "status": "ERROR", "error": errText}
                if "temperature" in errText.lower() and ("unsupported" in errText.lower() or "not support" in errText.lower()):
                    record["classification"] = "temperature_rejected"
                    tempRejected = True
                elif "content_filter" in errText.lower() or "content management policy" in errText.lower():
                    record["classification"] = "content_filter_block"
                else:
                    record["classification"] = "other_error"
            results.append(record)
            print(f"  {datasetName}/{ex['id']}: {record.get('classification', record['status'])}")

    logFile.close()
    with open(os.path.join(modelDir, "compat_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# ---------------------------------------------------------------------------
# Pilot manifest + resumable unit execution
# ---------------------------------------------------------------------------

def buildPilotManifest():
    return buildManifestN(20)


def buildManifestN(n):
    """n=20 reproduces the exact original pilot manifest (same loader,
    same prefix-slice convention used everywhere else in this project
    tonight, e.g. run_extension_n400.py); n=400 is the full main-experiment
    sample size, using the SAME dataset loaders/ordering as the existing
    open-weight experiments, so example IDs match exactly for
    cross-model comparison. Since n=20's examples are always a PREFIX of
    n=400's, everything already completed in the pilot is automatically a
    subset of the full manifest and gets skipped by the existing
    loadCompletedUnits()-based resume logic -- no special-casing needed."""
    manifest = []
    for datasetName in DATASETS:
        cfg = m.DATASET_CONFIGS[datasetName]
        examples = cfg["loader"]()[:n]
        for ex in examples:
            manifest.append({"dataset": datasetName, "exampleId": ex["id"], "groundTruth": ex["label"]})
    return manifest


def printDryRunReport(manifest):
    print("\n=== DRY-RUN / MANIFEST VALIDATION ===")
    datasetCounts = {}
    for rec in manifest:
        datasetCounts[rec["dataset"]] = datasetCounts.get(rec["dataset"], 0) + 1
    for safeModelName, modelId in MODELS.items():
        modelDir = os.path.join(OUT_ROOT, safeModelName)
        completed = loadCompletedUnits(modelDir)
        expectedUnits = len(manifest) * len(METHODS)
        completedCount = sum(1 for rec in manifest for method in METHODS
                              if (rec["dataset"], rec["exampleId"], method, BUDGETS.get(method)) in completed)
        print(f"\n{safeModelName} ({modelId}):")
        for dataset, n in datasetCounts.items():
            print(f"  {dataset}: {n} target examples")
        print(f"  budgets: {list(BUDGETS.values())}")
        print(f"  expected work units (examples x methods): {expectedUnits}")
        print(f"  already completed: {completedCount}")
        print(f"  remaining: {expectedUnits - completedCount}")


def loadCompletedUnits(modelDir):
    path = os.path.join(modelDir, "completed.jsonl")
    completed = set()
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                completed.add((rec["dataset"], rec["exampleId"], rec["method"], rec.get("budget")))
    return completed


def appendCompleted(modelDir, record):
    path = os.path.join(modelDir, "completed.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def appendError(modelDir, record):
    path = os.path.join(modelDir, "errors.jsonl")
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def runOneUnit(state: RunnerState, modelId, dataset, ex, method, cfg, nodeSystems, nodeArms):
    """Runs exactly one (dataset, example, method[, budget]) unit using the
    EXISTING run_graph_mv / run_graph_ucb / _call functions unmodified.
    Safe to call concurrently for the same model now that context travels
    via _callContextVar (see note above) instead of shared mutable state."""
    budget = BUDGETS.get(method)
    state.setCallContext(dataset, ex["id"], method, budget)
    t0 = time.time()

    if method == "single_call":
        validArms = set(nodeArms["Screener"])
        system = nodeSystems["Screener"]
        user = m.MV_USER.format(arms=m.format_arms(nodeArms["Screener"]), text=ex["text"])
        text, it, ot = m._call(modelId, system, user, max_tokens=10)
        finalLabel = None
        if m._is_refusal(text):
            finalLabel = "self_refused"
        elif m._is_prompt_injected(text):
            finalLabel = "prompt_injected"
        else:
            try:
                finalLabel = m._parse_label(text, validArms)
            except m.LabelParseError:
                finalLabel = "parse_failed"
        result = {"final_label": finalLabel, "total_pulls": 1,
                  "total_tokens_input": it, "total_tokens_output": ot}
    elif method == "graph_mv":
        result = m.run_graph_mv(modelId, ex["text"], nodeSystems, nodeArms, sanity=False)
    else:
        result = m.run_graph_ucb(modelId, ex["text"], budget, nodeSystems, nodeArms, sanity=False)

    elapsed = time.time() - t0
    return result, elapsed


def runModelPilot(safeModelName, modelId, hopgptKey, manifest, maxWorkersPerModel=1):
    # maxWorkersPerModel > 1 is now safe: see the _callContextVar +
    # ThreadPoolExecutor.submit patch note above. Controlled, not unbounded
    # -- caller picks a small explicit number per model (see CONCURRENCY in
    # __main__).
    print(f"\n=== Pilot: {safeModelName} ({modelId}), concurrency={maxWorkersPerModel} ===")
    modelDir = os.path.join(OUT_ROOT, safeModelName)
    os.makedirs(modelDir, exist_ok=True)

    logFile = open(os.path.join(modelDir, "calls.jsonl"), "a")
    state = RunnerState(safeModelName, modelId, logFile)
    registerRunnerState(state)
    # m._post_chat is set to the SAME shared dispatcher every time this is
    # called; harmless to reassign redundantly when both models start it
    # around the same time, since the dispatcher itself is stateless and
    # reads _runnerRegistry fresh on every call.
    m._post_chat = makeSharedHopgptPostChat(hopgptKey)

    completed = loadCompletedUnits(modelDir)
    print(f"  {len(completed)} units already completed for this model, skipping those.")

    # PERFORMANCE FIX (caught live): the loader used to be called fresh
    # inside this loop, once per manifest record -- for an 800-record
    # manifest that meant reloading the full 7702-row HateModerate CSV or
    # 1964-row AEGIS dataset up to 800 times before any real API call
    # happened, adding minutes of pure overhead on every launch/resume.
    # Cache each dataset's examples-by-id dict once, outside the loop.
    exampleCacheByDataset = {}

    todo = []
    for rec in manifest:
        dataset, exampleId = rec["dataset"], rec["exampleId"]
        if dataset not in exampleCacheByDataset:
            cfg = m.DATASET_CONFIGS[dataset]
            exampleCacheByDataset[dataset] = {e["id"]: e for e in cfg["loader"]()}
        ex = exampleCacheByDataset[dataset][exampleId]
        for method in METHODS:
            budget = BUDGETS.get(method)
            key = (dataset, exampleId, method, budget)
            if key in completed:
                continue
            todo.append((dataset, ex, method, budget))

    print(f"  {len(todo)} units remaining to run for this model.")

    def worker(unit):
        dataset, ex, method, budget = unit
        cfg = m.DATASET_CONFIGS[dataset]
        nodeSystems = ext.get_node_systems(dataset)
        nodeArms = m.build_node_arms(cfg["labels"])
        try:
            result, elapsed = runOneUnit(state, modelId, dataset, ex, method, cfg, nodeSystems, nodeArms)
            record = {
                "model": modelId, "dataset": dataset, "exampleId": ex["id"], "method": method,
                "budget": budget, "groundTruth": ex["label"], "finalLabel": result.get("final_label"),
                "totalPulls": result.get("total_pulls"), "elapsedSeconds": round(elapsed, 3),
                "finalConfidence": result.get("final_confidence"),
                "leadingCandidate": result.get("leading_candidate"),
            }
            appendCompleted(modelDir, record)
            return ("ok", record)
        except Exception as e:
            errRecord = {"model": modelId, "dataset": dataset, "exampleId": ex["id"],
                         "method": method, "budget": budget, "error": str(e)}
            appendError(modelDir, errRecord)
            return ("error", errRecord)

    nOk = nErr = 0
    # per (dataset, method, budget) progress totals -- fixed for the run.
    groupTotals = {}
    for rec in manifest:
        for method in METHODS:
            key = (rec["dataset"], method, BUDGETS.get(method))
            groupTotals[key] = groupTotals.get(key, 0) + 1

    # BUG FIX (caught live during a real run -- an earlier version of this
    # counter incremented an in-memory dict once per "ok" future without
    # re-verifying against the actual completed set, and drifted from the
    # true on-disk state badly enough to print a false "400/400 complete"
    # while the file genuinely had only 386 unique examples for that group.
    # The exact mechanism wasn't pinned down with certainty, but the fix
    # that removes the whole class of risk is straightforward: make the
    # progress count authoritative by re-deriving it from the real
    # completed-units set on disk every time a line is due to print,
    # rather than trusting an incrementally-updated in-memory number.
    nSinceLastPrint = 0

    def recomputeGroupDone(key):
        dataset, method, budget = key
        currentCompleted = loadCompletedUnits(modelDir)
        return sum(1 for rec in manifest if rec["dataset"] == dataset
                   and (rec["dataset"], rec["exampleId"], method, budget) in currentCompleted)

    with ThreadPoolExecutor(max_workers=maxWorkersPerModel) as executor:
        futures = {executor.submit(worker, unit): unit for unit in todo}
        for future in as_completed(futures):
            status, record = future.result()
            key = (record["dataset"], record["method"], record.get("budget"))
            if status == "ok":
                nOk += 1
                nSinceLastPrint += 1
            else:
                nErr += 1
            # concise batch-progress line every 20 successful completions;
            # done/total always re-derived from disk, never accumulated
            # in-memory, so it cannot report a false 100%.
            if status == "ok" and nSinceLastPrint >= 20:
                nSinceLastPrint = 0
                total = groupTotals.get(key, 0)
                done = recomputeGroupDone(key)
                budgetLabel = f"B={key[2]}" if key[2] else key[1]
                print(f"{safeModelName} | {key[0]} | {budgetLabel} | {done}/{total} complete")

    # final authoritative progress line per group at the end of this run
    for key, total in sorted(groupTotals.items()):
        done = recomputeGroupDone(key)
        budgetLabel = f"B={key[2]}" if key[2] else key[1]
        print(f"{safeModelName} | {key[0]} | {budgetLabel} | {done}/{total} complete (final, this run)")

    logFile.close()
    print(f"  {safeModelName}: {nOk} succeeded, {nErr} errored this run.")

    # write/refresh summary.json
    completedNow = loadCompletedUnits(modelDir)
    with open(os.path.join(modelDir, "summary.json"), "w") as f:
        json.dump({
            "safeModelName": safeModelName, "modelId": modelId,
            "totalUnitsExpectedForPilot": len(manifest) * len(METHODS),
            "unitsCompleted": len(completedNow),
        }, f, indent=2)

    with open(os.path.join(modelDir, "checkpoint.json"), "w") as f:
        json.dump({"lastUpdated": time.time(), "unitsCompleted": len(completedNow)}, f, indent=2)

    return nOk, nErr


def writePilotSummaryCsv(manifest):
    rows = []
    for safeModelName in MODELS:
        modelDir = os.path.join(OUT_ROOT, safeModelName)
        path = os.path.join(modelDir, "completed.jsonl")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                nonTerminal = {"escalate", "self_refused", "prompt_injected", "parse_failed"}
                resolved = rec["finalLabel"] not in nonTerminal
                rows.append({
                    "model": safeModelName, "dataset": rec["dataset"], "example_id": rec["exampleId"],
                    "method": rec["method"], "budget": rec.get("budget") or "",
                    "resolved": resolved, "predicted_label": rec["finalLabel"], "ground_truth": rec["groundTruth"],
                    "correct": (rec["finalLabel"] == rec["groundTruth"]) if resolved else "",
                    "escalated": rec["finalLabel"] == "escalate",
                    "parse_failed": rec["finalLabel"] == "parse_failed",
                    "refused": rec["finalLabel"] == "self_refused",
                    "prompt_injected": rec["finalLabel"] == "prompt_injected",
                    "total_calls": rec.get("totalPulls"), "latency": rec.get("elapsedSeconds"),
                })
    if rows:
        with open(os.path.join(OUT_ROOT, "pilot_summary.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compat-test", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--resume", action="store_true")
    # --full is explicit and separate from --pilot/--resume (unlike the
    # fixed_compute script's earlier bug where bare --resume silently used
    # the full manifest) precisely to avoid that class of mistake: --resume
    # ALONE still means "resume the 20/dataset pilot"; the 400/dataset run
    # requires --full explicitly, every time.
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    hopgptKey = loadHopgptKey()

    with open(os.path.join(OUT_ROOT, "model_manifest.json"), "w") as f:
        json.dump(MODELS, f, indent=2)

    if args.compat_test:
        for safeModelName, modelId in MODELS.items():
            runCompatibilityTest(safeModelName, modelId, hopgptKey)

    manifest = None
    if args.full:
        manifest = buildManifestN(400)
        with open(os.path.join(OUT_ROOT, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        printDryRunReport(manifest)
    elif args.pilot or args.resume:
        manifest = buildPilotManifest()
        with open(os.path.join(OUT_ROOT, "pilot_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    # Per-model concurrency. claude_haiku_4_5 is the bottleneck (verbose
    # preamble -> LabelParseError -> whole SE unit wasted-and-redone
    # pattern, documented in memory); calls are network-bound (<1s each)
    # so raising this genuinely cuts wall-clock time. llama_70b is already
    # effectively done and left at 1 to avoid disturbing it. Controlled
    # (small, explicit), not unbounded -- see _callContextVar patch note
    # above for why this is now safe.
    CONCURRENCY = {"claude_haiku_4_5": 4, "llama_70b": 1}

    if manifest is not None:
        # run both models CONCURRENTLY (controlled: 2 model-level workers,
        # each model additionally runs up to CONCURRENCY[name] units at
        # once internally)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(runModelPilot, safeModelName, modelId, hopgptKey, manifest,
                                 CONCURRENCY.get(safeModelName, 1)): safeModelName
                for safeModelName, modelId in MODELS.items()
            }
            for future in as_completed(futures):
                safeModelName = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"MODEL {safeModelName} FAILED: {e}")

        writePilotSummaryCsv(manifest)
