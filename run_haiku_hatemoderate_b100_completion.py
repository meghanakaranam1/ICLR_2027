"""
Scoped completion runner: Claude Haiku 4.5 x HateModerate x SE (graph_ucb) x
B100 ONLY, for the 26 examples missing from the original frontier_model run.

HARD SCOPE (do not extend): model=claude-haiku-4-5, dataset=hatemoderate,
method=graph_ucb, budget=100. No other model/dataset/method/budget is
touched. The existing analysis/frontier_model/claude_haiku_4_5/ directory
and its files are read-only inputs here (to determine which example IDs are
already done) and are never written to.

Reuses run_all_datasets.py's ucb_node/run_graph_ucb/_call UNCHANGED -- the
only new code is a _post_chat replacement that routes to the real Anthropic
API instead of local Ollama, translating between the OpenAI-chat-style
payload/response shape _call() already expects and the Anthropic Messages
API shape. Same temperature (TEMPERATURE=0.7, already a module constant used
by _call), same max_tokens=10, same retry-on-transient-error pattern -- with
one deliberate addition: an "insufficient credit" response is detected and
raised as a distinct, non-retried exception that stops the whole run
cleanly, per explicit instruction.

$5 absolute spending ceiling (carried over from the still-standing earlier
instruction, never rescinded): tracked from REAL per-call usage returned by
the API on every response, checked before starting each new example using a
conservative worst-case-per-example projection (300 calls/example, the
theoretical max for 3 nodes x budget=100 with zero early resolution) against
a $4.50 target ceiling (leaving deliberate headroom under the true $5 limit,
per "do not intentionally spend right up to $5").

Usage:
    python3 run_haiku_hatemoderate_b100_completion.py --validate   (2 examples only)
    python3 run_haiku_hatemoderate_b100_completion.py --resume     (all remaining, up to budget)
"""

import os
import sys
import csv
import json
import time
import argparse
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_all_datasets as m
import run_extension_n400 as ext
import anthropic

MODEL_ID = "claude-haiku-4-5"
DATASET = "hatemoderate"
METHOD = "graph_ucb_B100"
BUDGET = 100

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "analysis", "frontier_model", "haiku_hatemoderate_B100_completion")
EXISTING_COMPLETED_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "analysis", "frontier_model", "claude_haiku_4_5", "completed.jsonl")

ANTHROPIC_KEY_PATH = os.path.expanduser("~/.anthropic_key")

# Official Anthropic pricing, verified directly from platform.claude.com/docs
# on 2026-09-24 -- no caching (confirmed not viable for this prompt size).
PRICE_IN_PER_TOKEN = 1.0 / 1_000_000
PRICE_OUT_PER_TOKEN = 5.0 / 1_000_000

SPEND_CEILING = 5.00
SPEND_TARGET = 4.50   # deliberate safety margin under the true $5 ceiling
WORST_CASE_CALLS_PER_EXAMPLE = 3 * BUDGET  # 3 nodes x full budget, zero early resolution


class InsufficientCreditError(RuntimeError):
    """Raised the moment the API reports a credit/balance problem -- never
    retried, always propagates up to stop the whole run cleanly, per
    explicit instruction to stop rather than modify the experiment."""
    pass


def loadAnthropicKey():
    with open(ANTHROPIC_KEY_PATH) as f:
        return f.read().strip()


class CostTracker:
    def __init__(self):
        self.lock = threading.Lock()
        self.totalInputTokens = 0
        self.totalOutputTokens = 0
        self.totalCalls = 0

    def add(self, inputTokens, outputTokens):
        with self.lock:
            self.totalInputTokens += inputTokens
            self.totalOutputTokens += outputTokens
            self.totalCalls += 1

    def costSoFar(self):
        with self.lock:
            return (self.totalInputTokens * PRICE_IN_PER_TOKEN
                     + self.totalOutputTokens * PRICE_OUT_PER_TOKEN)

    def meanTokensPerCall(self):
        with self.lock:
            if self.totalCalls == 0:
                return None, None
            return (self.totalInputTokens / self.totalCalls,
                    self.totalOutputTokens / self.totalCalls)

    def snapshot(self):
        with self.lock:
            return {
                "totalCalls": self.totalCalls,
                "totalInputTokens": self.totalInputTokens,
                "totalOutputTokens": self.totalOutputTokens,
                "costSoFar": round(self.totalInputTokens * PRICE_IN_PER_TOKEN
                                    + self.totalOutputTokens * PRICE_OUT_PER_TOKEN, 6),
            }


def makeAnthropicPostChat(client, costTracker, callLogFile, callLogLock, exampleContext):
    """Replaces run_all_datasets._post_chat. Translates the OpenAI-chat-style
    payload _call() builds into an Anthropic Messages API call, and the
    Anthropic response back into the {"choices":[{"message":{"content":...}}],
    "usage":{"prompt_tokens":...,"completion_tokens":...}} shape _call()
    expects -- _call/ucb_node/run_graph_ucb are otherwise completely
    unmodified. exampleContext is a mutable single-slot dict the caller
    updates before each run_graph_ucb() invocation (this script never runs
    more than one example concurrently, so this is unambiguous -- see
    processOneExample)."""

    def anthropicPostChat(payload):
        messages = payload["messages"]
        systemText = None
        chatMessages = []
        for msg in messages:
            if msg["role"] == "system":
                systemText = msg["content"]
            else:
                chatMessages.append({"role": msg["role"], "content": msg["content"]})
        maxTokens = payload["max_tokens"]
        temperature = payload["temperature"]

        callIndex = exampleContext.get("callIndex", 0) + 1
        exampleContext["callIndex"] = callIndex
        record = {
            "model": MODEL_ID, "dataset": DATASET, "exampleId": exampleContext.get("exampleId"),
            "method": METHOD, "budget": BUDGET, "callIndex": callIndex,
            "timestamp": None, "httpStatus": None, "rawOutput": None,
            "inputTokens": None, "outputTokens": None, "error": None,
        }

        lastExc = None
        for attempt in range(4):
            try:
                resp = client.messages.create(
                    model=MODEL_ID, system=systemText, messages=chatMessages,
                    max_tokens=maxTokens, temperature=temperature,
                )
                text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
                it, ot = resp.usage.input_tokens, resp.usage.output_tokens
                record["timestamp"] = time.time()
                record["httpStatus"] = 200
                record["rawOutput"] = text
                record["inputTokens"] = it
                record["outputTokens"] = ot
                costTracker.add(it, ot)
                with callLogLock:
                    callLogFile.write(json.dumps(record) + "\n")
                    callLogFile.flush()
                return {"choices": [{"message": {"content": text}}],
                        "usage": {"prompt_tokens": it, "completion_tokens": ot}}
            except anthropic.APIStatusError as e:
                msg = str(e)
                if "credit balance is too low" in msg.lower() or "insufficient" in msg.lower():
                    record["error"] = f"INSUFFICIENT_CREDIT: {msg[:300]}"
                    with callLogLock:
                        callLogFile.write(json.dumps(record) + "\n")
                        callLogFile.flush()
                    raise InsufficientCreditError(msg)
                lastExc = e
                wait = min(5 * (2 ** attempt), 15)
                print(f"    [retry {attempt+1}] {e} -- waiting {wait}s", flush=True)
                time.sleep(wait)
            except InsufficientCreditError:
                raise
            except Exception as e:
                lastExc = e
                wait = min(5 * (2 ** attempt), 15)
                print(f"    [retry {attempt+1}] {e} -- waiting {wait}s", flush=True)
                time.sleep(wait)

        record["error"] = f"failed after retries: {lastExc}"
        with callLogLock:
            callLogFile.write(json.dumps(record) + "\n")
            callLogFile.flush()
        raise RuntimeError(f"Anthropic call failed after retries: {lastExc}")

    return anthropicPostChat


def loadExistingCompletedIds():
    ids = set()
    if os.path.isfile(EXISTING_COMPLETED_PATH):
        with open(EXISTING_COMPLETED_PATH) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["dataset"] == DATASET and r["method"] == METHOD:
                    ids.add(r["exampleId"])
    return ids


def loadOwnCompletedIds():
    path = os.path.join(OUT_DIR, "completed.jsonl")
    ids = set()
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                ids.add(r["exampleId"])
    return ids


def buildMissingExampleList():
    existingDone = loadExistingCompletedIds()
    cfg = m.DATASET_CONFIGS[DATASET]
    examples = cfg["loader"]()[:400]
    missing = [ex for ex in examples if ex["id"] not in existingDone]
    return missing


def appendJsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def processOneExample(client, costTracker, callLogFile, callLogLock, ex, nodeSystems, nodeArms):
    exampleContext = {"exampleId": ex["id"], "callIndex": 0}
    m._post_chat = makeAnthropicPostChat(client, costTracker, callLogFile, callLogLock, exampleContext)
    t0 = time.time()
    result = m.run_graph_ucb(MODEL_ID, ex["text"], BUDGET, nodeSystems, nodeArms, sanity=False)
    elapsed = time.time() - t0
    return result, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true", help="Process only 2 examples, for cost validation.")
    parser.add_argument("--resume", action="store_true", help="Process all remaining missing examples, up to the spend ceiling.")
    args = parser.parse_args()

    if not args.validate and not args.resume:
        print("Specify --validate or --resume.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    key = loadAnthropicKey()
    client = anthropic.Anthropic(api_key=key)

    missing = buildMissingExampleList()
    ownDone = loadOwnCompletedIds()
    todo = [ex for ex in missing if ex["id"] not in ownDone]

    print(f"Existing frontier run already completed: {400 - len(missing)}/400")
    print(f"Missing (target for this completion run): {len(missing)}")
    print(f"Already done by THIS completion run (resumed): {len(ownDone)}")
    print(f"Remaining to process now: {len(todo)}")

    if args.validate:
        todo = todo[:2]
        print(f"--validate: limiting to {len(todo)} example(s) this invocation.")

    if not todo:
        print("Nothing to do.")
        return

    nodeSystems = ext.get_node_systems(DATASET)
    cfg = m.DATASET_CONFIGS[DATASET]
    nodeArms = m.build_node_arms(cfg["labels"])

    costTracker = CostTracker()
    callLogFile = open(os.path.join(OUT_DIR, "calls.jsonl"), "a")
    callLogLock = threading.Lock()

    nOk = nErr = 0
    stoppedReason = None

    for ex in todo:
        # Conservative pre-check: worst-case cost for a NEW example uses this
        # run's OWN observed mean tokens/call so far (falls back to a fixed
        # historical estimate -- 2216 in / 4.4 out per call, from the prior
        # completed 374 examples -- before this run has any calls of its own).
        meanIn, meanOut = costTracker.meanTokensPerCall()
        if meanIn is None:
            meanIn, meanOut = 2216.26, 4.41  # historical fallback, first example only
        worstCaseCost = WORST_CASE_CALLS_PER_EXAMPLE * (meanIn * PRICE_IN_PER_TOKEN + meanOut * PRICE_OUT_PER_TOKEN)
        costSoFar = costTracker.costSoFar()
        if costSoFar + worstCaseCost > SPEND_TARGET:
            stoppedReason = (f"Stopping before example {ex['id']}: cumulative spend "
                              f"${costSoFar:.4f} + worst-case ${worstCaseCost:.4f} would exceed "
                              f"the ${SPEND_TARGET:.2f} safety target.")
            print(stoppedReason)
            break

        print(f"Processing hatemoderate/{ex['id']}/graph_ucb_B100 ... (spend so far: ${costSoFar:.4f})")
        try:
            result, elapsed = processOneExample(client, costTracker, callLogFile, callLogLock,
                                                 ex, nodeSystems, nodeArms)
            record = {
                "model": MODEL_ID, "dataset": DATASET, "exampleId": ex["id"], "method": METHOD,
                "budget": BUDGET, "groundTruth": ex["label"], "finalLabel": result.get("final_label"),
                "totalPulls": result.get("total_pulls"), "elapsedSeconds": round(elapsed, 3),
                "finalConfidence": result.get("final_confidence"),
                "leadingCandidate": result.get("leading_candidate"),
                "timestamp": time.time(),
            }
            appendJsonl(os.path.join(OUT_DIR, "completed.jsonl"), record)
            appendJsonl(os.path.join(OUT_DIR, "results.jsonl"), record)
            nOk += 1
            print(f"  OK: label={record['finalLabel']} pulls={record['totalPulls']} "
                  f"(cumulative spend: ${costTracker.costSoFar():.4f})")
        except InsufficientCreditError as e:
            errRecord = {"model": MODEL_ID, "dataset": DATASET, "exampleId": ex["id"], "method": METHOD,
                         "budget": BUDGET, "error": f"INSUFFICIENT_CREDIT: {e}", "timestamp": time.time()}
            appendJsonl(os.path.join(OUT_DIR, "errors.jsonl"), errRecord)
            stoppedReason = f"API reported insufficient credit while processing {ex['id']} -- stopping cleanly."
            print(stoppedReason)
            nErr += 1
            break
        except Exception as e:
            errRecord = {"model": MODEL_ID, "dataset": DATASET, "exampleId": ex["id"], "method": METHOD,
                         "budget": BUDGET, "error": str(e), "timestamp": time.time()}
            appendJsonl(os.path.join(OUT_DIR, "errors.jsonl"), errRecord)
            nErr += 1
            print(f"  ERROR: {e}")

    callLogFile.close()

    finalOwnDone = loadOwnCompletedIds()
    snap = costTracker.snapshot()
    metadata = {
        "model": MODEL_ID, "dataset": DATASET, "method": METHOD, "budget": BUDGET,
        "targetN": 400,
        "completedByOriginalFrontierRun": 400 - len(missing),
        "missingAtStartOfThisRun": len(missing),
        "completedByThisRunSoFar": len(finalOwnDone),
        "remainingAfterThisRun": len(missing) - len(finalOwnDone),
        "thisInvocation": {"okCount": nOk, "errorCount": nErr, "stoppedReason": stoppedReason},
        "spend": snap,
        "spendCeiling": SPEND_CEILING, "spendTarget": SPEND_TARGET,
        "lastUpdated": time.time(),
    }
    with open(os.path.join(OUT_DIR, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(OUT_DIR, "cost_summary.json"), "w") as f:
        json.dump(snap, f, indent=2)

    print("\n=== RUN SUMMARY ===")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
