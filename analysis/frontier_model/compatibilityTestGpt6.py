import os, sys, csv, json
sys.path.insert(0, "/Users/meghanakarnam/Desktop/MLHC/ICLR_2027")
import requests
import run_all_datasets as m
import run_extension_n400 as ext

HOPGPT_KEY = open(os.path.expanduser("~/.hopgpt_key")).read().strip()
HOPGPT_URL = "https://api.ai.jh.edu/v1/chat/completions"
CANDIDATES = ["gpt-6-sol", "gpt-6-luna", "gpt-6-astra"]

cfg = m.DATASET_CONFIGS["hatemoderate"]
ex = cfg["loader"]()[0]
nodeSystems = ext.get_node_systems("hatemoderate")
nodeArms = m.build_node_arms(cfg["labels"])
system = nodeSystems["Screener"]
user = m.MV_USER.format(arms=m.format_arms(nodeArms["Screener"]), text=ex["text"])

results = []
for modelId in CANDIDATES:
    payload = {"model": modelId, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
               "max_tokens": 10, "temperature": 0.7}
    resp = requests.post(HOPGPT_URL, json=payload,
                          headers={"Authorization": f"Bearer {HOPGPT_KEY}", "Content-Type": "application/json"},
                          timeout=30)
    print(f"{modelId}: status={resp.status_code}")
    print(f"  body: {resp.text[:400]}")
    results.append({
        "model": modelId, "modelIdentifierAccepted": resp.status_code == 200,
        "temperatureAccepted": None, "successfulCalls": 0, "parseableCalls": 0,
        "AEGISBlocks": 0, "HateModerateBlocks": 0, "providerErrors": 1 if resp.status_code != 200 else 0,
        "rawErrorMessage": resp.text[:500] if resp.status_code != 200 else "",
        "callsSpent": 1,
    })

with open("analysis/frontier_model/compatibilityTestGpt6.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)
