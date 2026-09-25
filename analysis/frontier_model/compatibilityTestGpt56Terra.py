import os, sys, csv, json
sys.path.insert(0, "/Users/meghanakarnam/Desktop/MLHC/ICLR_2027")
import requests
import run_all_datasets as m
import run_extension_n400 as ext

HOPGPT_KEY = open(os.path.expanduser("~/.hopgpt_key")).read().strip()
HOPGPT_URL = "https://api.ai.jh.edu/v1/chat/completions"
MODEL_ID = "gpt-5.6-terra"

testExamples = []
for datasetName in ["hatemoderate", "aegis"]:
    cfg = m.DATASET_CONFIGS[datasetName]
    for ex in cfg["loader"]()[:2]:
        testExamples.append((datasetName, ex))

results = []
stoppedOnTemp = False
for datasetName, ex in testExamples:
    if stoppedOnTemp:
        results.append({"model": MODEL_ID, "dataset": datasetName, "exampleId": ex["id"],
                         "httpStatus": "SKIPPED", "temperatureAccepted": False,
                         "successfulCall": False, "parseableCall": False,
                         "contentFilterBlock": False, "parseFailure": False,
                         "inputTokens": None, "outputTokens": None, "errorDetail": "skipped after temp rejection"})
        continue

    cfg = m.DATASET_CONFIGS[datasetName]
    nodeSystems = ext.get_node_systems(datasetName)
    nodeArms = m.build_node_arms(cfg["labels"])
    system = nodeSystems["Screener"]
    validArms = set(nodeArms["Screener"])
    user = m.MV_USER.format(arms=m.format_arms(nodeArms["Screener"]), text=ex["text"])

    payload = {"model": MODEL_ID, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
               "max_tokens": 10, "temperature": 0.7}
    resp = requests.post(HOPGPT_URL, json=payload,
                          headers={"Authorization": f"Bearer {HOPGPT_KEY}", "Content-Type": "application/json"},
                          timeout=60)

    record = {"model": MODEL_ID, "dataset": datasetName, "exampleId": ex["id"],
              "httpStatus": resp.status_code, "temperatureAccepted": None,
              "successfulCall": False, "parseableCall": False,
              "contentFilterBlock": False, "parseFailure": False,
              "inputTokens": None, "outputTokens": None, "errorDetail": None}

    if resp.status_code != 200:
        body = resp.text[:500]
        record["errorDetail"] = body
        bodyLower = body.lower()
        if "temperature" in bodyLower and ("unsupported" in bodyLower or "not support" in bodyLower):
            record["temperatureAccepted"] = False
            stoppedOnTemp = True
            print(f"*** TEMPERATURE REJECTED on {datasetName}/{ex['id']}: {body} ***")
        elif "content_filter" in bodyLower or "content management policy" in bodyLower:
            record["temperatureAccepted"] = True
            record["contentFilterBlock"] = True
        else:
            print(f"OTHER ERROR on {datasetName}/{ex['id']}: {body}")
    else:
        record["temperatureAccepted"] = True
        record["successfulCall"] = True
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        record["inputTokens"] = usage.get("prompt_tokens")
        record["outputTokens"] = usage.get("completion_tokens")
        if m._is_refusal(text):
            record["errorDetail"] = "refusal"
        elif m._is_prompt_injected(text):
            record["errorDetail"] = "prompt_injected"
        else:
            try:
                label = m._parse_label(text, validArms)
                record["parseableCall"] = True
                record["errorDetail"] = f"parsed={label}"
            except m.LabelParseError:
                record["parseFailure"] = True
        print(f"{datasetName}/{ex['id']}: status=200 raw={text!r} in={record['inputTokens']} out={record['outputTokens']}")

    results.append(record)

with open("analysis/frontier_model/compatibilityTestGpt56Terra.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

print("\n=== SUMMARY ===")
print("model:", MODEL_ID)
print("temperature accepted:", not stoppedOnTemp)
print("successful calls:", sum(1 for r in results if r["successfulCall"]))
print("parseable calls:", sum(1 for r in results if r["parseableCall"]))
print("AEGIS blocks:", sum(1 for r in results if r["dataset"]=="aegis" and r["contentFilterBlock"]))
print("HateModerate blocks:", sum(1 for r in results if r["dataset"]=="hatemoderate" and r["contentFilterBlock"]))
print("parse failures:", sum(1 for r in results if r["parseFailure"]))
print("provider errors (non-temp, non-block):", sum(1 for r in results if r["httpStatus"] not in (200,"SKIPPED") and not r["contentFilterBlock"] and r["temperatureAccepted"] is not False))
totalIn = sum(r["inputTokens"] for r in results if r["inputTokens"])
totalOut = sum(r["outputTokens"] for r in results if r["outputTokens"])
print(f"total input tokens: {totalIn}, total output tokens: {totalOut}")
