"""
10-example compatibility test for claude-sonnet-5 via HopGPT: 5 HateModerate
+ 5 AEGIS examples, exact existing prompts/parsing/temperature, single calls
only (no SE/MV). Read-only over existing example loaders; writes only to
analysis/frontier_model/.
"""
import os, sys, json, time, csv
sys.path.insert(0, "/Users/meghanakarnam/Desktop/MLHC/ICLR_2027")
import requests
import run_all_datasets as m
import run_extension_n400 as ext

HOPGPT_KEY = open(os.path.expanduser("~/.hopgpt_key")).read().strip()
MODEL_ID = "claude-sonnet-5"
HOPGPT_URL = "https://api.ai.jh.edu/v1/chat/completions"

def hopgptPostChat(payload):
    headers = {"Authorization": f"Bearer {HOPGPT_KEY}", "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            resp = requests.post(HOPGPT_URL, json=payload, headers=headers, timeout=60)
            return resp
        except requests.exceptions.RequestException as e:
            print(f"  [retry {attempt+1}] network error: {e}")
            time.sleep(3)
    return None

results = []
for datasetName in ["hatemoderate", "aegis"]:
    cfg = m.DATASET_CONFIGS[datasetName]
    examples = cfg["loader"]()[:5]
    nodeSystems = ext.get_node_systems(datasetName)
    nodeArms = m.build_node_arms(cfg["labels"])
    system = nodeSystems["Screener"]
    validArms = set(nodeArms["Screener"])

    for ex in examples:
        user = m.MV_USER.format(arms=m.format_arms(nodeArms["Screener"]), text=ex["text"])
        payload = {
            "model": MODEL_ID,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": 10,
            "temperature": 0.7,
        }
        resp = hopgptPostChat(payload)
        record = {
            "exampleId": ex["id"], "dataset": datasetName, "model": MODEL_ID,
            "groundTruth": ex["label"],
            "httpStatus": resp.status_code if resp is not None else "NETWORK_ERROR",
            "rawResponseText": None, "parsedLabel": None,
            "validLabelReturned": False, "refusal": False,
            "contentFilterBlock": False, "parseFailure": False,
            "errorDetail": None,
        }
        if resp is None:
            record["errorDetail"] = "network error after retries"
        elif resp.status_code != 200:
            record["errorDetail"] = resp.text[:500]
            bodyLower = resp.text.lower()
            if "content_filter" in bodyLower or "content management policy" in bodyLower or "safety" in bodyLower:
                record["contentFilterBlock"] = True
        else:
            data = resp.json()
            try:
                text = data["choices"][0]["message"]["content"].strip()
                record["rawResponseText"] = text
                if m._is_refusal(text):
                    record["refusal"] = True
                elif m._is_prompt_injected(text):
                    record["errorDetail"] = "flagged as prompt-injected response"
                else:
                    try:
                        label = m._parse_label(text, validArms)
                        record["parsedLabel"] = label
                        record["validLabelReturned"] = True
                    except m.LabelParseError:
                        record["parseFailure"] = True
            except (KeyError, IndexError) as e:
                record["errorDetail"] = f"unexpected response shape: {e}; body={json.dumps(data)[:500]}"
        results.append(record)
        print(f"  {datasetName}/{ex['id']}: status={record['httpStatus']} valid={record['validLabelReturned']} "
              f"refusal={record['refusal']} contentFilter={record['contentFilterBlock']} parseFail={record['parseFailure']} "
              f"raw={record['rawResponseText']!r}")

fieldOrder = list(results[0].keys())
with open("analysis/frontier_model/compatibilityTestResults.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldOrder)
    writer.writeheader()
    writer.writerows(results)

nValid = sum(1 for r in results if r["validLabelReturned"])
nRefusal = sum(1 for r in results if r["refusal"])
nBlocked = sum(1 for r in results if r["contentFilterBlock"])
nParseFail = sum(1 for r in results if r["parseFailure"])
nOtherError = sum(1 for r in results if r["errorDetail"] and not r["contentFilterBlock"] and r["httpStatus"] != 200)

print("\n=== COMPATIBILITY TEST SUMMARY ===")
print(f"Total examples: {len(results)}")
print(f"Successful classifications: {nValid}")
print(f"Refusals: {nRefusal}")
print(f"Content-filter/provider blocks: {nBlocked}")
print(f"Parse failures: {nParseFail}")
print(f"Other errors: {nOtherError}")

aegisResults = [r for r in results if r["dataset"] == "aegis"]
aegisBlocked = sum(1 for r in aegisResults if r["contentFilterBlock"])
print(f"\nAEGIS-specific: {aegisBlocked}/{len(aegisResults)} blocked by content filter")
if aegisBlocked >= 3:
    print("*** SYSTEMATIC BLOCKING DETECTED ON AEGIS -- STOPPING PER INSTRUCTIONS ***")
