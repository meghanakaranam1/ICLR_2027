"""
Temperature/compatibility test for candidate frontier models via HopGPT:
claude-opus-5 and claude-haiku-4.5. Up to 4 calls per model (2 HateModerate
+ 2 AEGIS), stopping early for a model at the first temperature=0.7
rejection. No SE/MV, no existing results touched.
"""
import os, sys, json, time, csv
sys.path.insert(0, "/Users/meghanakarnam/Desktop/MLHC/ICLR_2027")
import requests
import run_all_datasets as m
import run_extension_n400 as ext

HOPGPT_KEY = open(os.path.expanduser("~/.hopgpt_key")).read().strip()
HOPGPT_URL = "https://api.ai.jh.edu/v1/chat/completions"
CANDIDATES = ["claude-opus-5", "claude-haiku-4.5"]

def hopgptPostChat(payload):
    headers = {"Authorization": f"Bearer {HOPGPT_KEY}", "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            return requests.post(HOPGPT_URL, json=payload, headers=headers, timeout=60)
        except requests.exceptions.RequestException as e:
            print(f"  [retry {attempt+1}] network error: {e}")
            time.sleep(3)
    return None

testExamples = []
for datasetName in ["hatemoderate", "aegis"]:
    cfg = m.DATASET_CONFIGS[datasetName]
    examples = cfg["loader"]()[:2]
    for ex in examples:
        testExamples.append((datasetName, ex))

allResults = []
for modelId in CANDIDATES:
    print(f"\n=== Testing {modelId} ===")
    stoppedEarly = False
    for datasetName, ex in testExamples:
        if stoppedEarly:
            record = {
                "model": modelId, "dataset": datasetName, "exampleId": ex["id"],
                "httpStatus": "SKIPPED", "tempAccepted": False, "successfulResponse": False,
                "parsedLabel": None, "refusal": False, "contentFilterBlock": False,
                "parseFailure": False, "errorDetail": "skipped -- model already marked incompatible on temperature",
            }
            allResults.append(record)
            continue

        nodeSystems = ext.get_node_systems(datasetName)
        cfg = m.DATASET_CONFIGS[datasetName]
        nodeArms = m.build_node_arms(cfg["labels"])
        system = nodeSystems["Screener"]
        user = m.MV_USER.format(arms=m.format_arms(nodeArms["Screener"]), text=ex["text"])
        payload = {
            "model": modelId,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": 10,
            "temperature": 0.7,
        }
        resp = hopgptPostChat(payload)
        record = {
            "model": modelId, "dataset": datasetName, "exampleId": ex["id"],
            "httpStatus": resp.status_code if resp is not None else "NETWORK_ERROR",
            "tempAccepted": None, "successfulResponse": False, "parsedLabel": None,
            "refusal": False, "contentFilterBlock": False, "parseFailure": False, "errorDetail": None,
        }
        if resp is None:
            record["errorDetail"] = "network error after retries"
            record["tempAccepted"] = False
        elif resp.status_code != 200:
            bodyText = resp.text[:500]
            record["errorDetail"] = bodyText
            bodyLower = bodyText.lower()
            if "temperature" in bodyLower and ("unsupported" in bodyLower or "not support" in bodyLower):
                record["tempAccepted"] = False
                stoppedEarly = True
                print(f"  {datasetName}/{ex['id']}: TEMPERATURE REJECTED -- {bodyText[:200]}")
            elif "content_filter" in bodyLower or "content management policy" in bodyLower:
                record["contentFilterBlock"] = True
                record["tempAccepted"] = True  # got past temp check, blocked on content instead
                print(f"  {datasetName}/{ex['id']}: status=400 CONTENT FILTER BLOCK")
            else:
                record["tempAccepted"] = None
                print(f"  {datasetName}/{ex['id']}: status=400 OTHER ERROR -- {bodyText[:200]}")
        else:
            record["tempAccepted"] = True
            data = resp.json()
            try:
                text = data["choices"][0]["message"]["content"].strip()
                record["rawResponseText"] = text
                validArms = set(nodeArms["Screener"])
                if m._is_refusal(text):
                    record["refusal"] = True
                elif m._is_prompt_injected(text):
                    record["errorDetail"] = "flagged as prompt-injected"
                else:
                    try:
                        label = m._parse_label(text, validArms)
                        record["parsedLabel"] = label
                        record["successfulResponse"] = True
                    except m.LabelParseError:
                        record["parseFailure"] = True
                print(f"  {datasetName}/{ex['id']}: status=200 temp_ok=True label={record['parsedLabel']} raw={text!r}")
            except (KeyError, IndexError) as e:
                record["errorDetail"] = f"unexpected shape: {e}"
        allResults.append(record)

fieldOrder = ["model", "dataset", "exampleId", "httpStatus", "tempAccepted",
              "successfulResponse", "parsedLabel", "refusal", "contentFilterBlock",
              "parseFailure", "errorDetail"]
with open("analysis/frontier_model/compatibilityTestCandidates.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldOrder)
    writer.writeheader()
    for r in allResults:
        writer.writerow({k: r.get(k) for k in fieldOrder})

print("\n=== COMPARISON TABLE ===")
print(f"{'model':<20}{'temp0.7 accepted':<20}{'successful':<14}{'parseable':<13}{'AEGIS blocks':<15}{'HM blocks':<12}")
for modelId in CANDIDATES:
    sub = [r for r in allResults if r["model"] == modelId]
    tempAccepted = any(r["tempAccepted"] for r in sub if r["httpStatus"] != "SKIPPED")
    tempEverRejected = any(r["tempAccepted"] is False and r["httpStatus"] != "NETWORK_ERROR" for r in sub)
    nSuccess = sum(1 for r in sub if r["successfulResponse"])
    nParseable = sum(1 for r in sub if r["parsedLabel"])
    aegisBlocks = sum(1 for r in sub if r["dataset"] == "aegis" and r["contentFilterBlock"])
    hmBlocks = sum(1 for r in sub if r["dataset"] == "hatemoderate" and r["contentFilterBlock"])
    tempStr = "REJECTED" if tempEverRejected else ("YES" if tempAccepted else "UNKNOWN/ERROR")
    print(f"{modelId:<20}{tempStr:<20}{nSuccess:<14}{nParseable:<13}{aegisBlocks:<15}{hmBlocks:<12}")
