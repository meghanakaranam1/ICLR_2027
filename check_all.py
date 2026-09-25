import csv, os, sys
from collections import Counter

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results_02_n400_from_spark"

for model in sorted(os.listdir(ROOT)):
    model_dir = os.path.join(ROOT, model)
    if not os.path.isdir(model_dir):
        continue
    for dataset in sorted(os.listdir(model_dir)):
        ds_dir = os.path.join(model_dir, dataset)
        if not os.path.isdir(ds_dir):
            continue
        for cond in sorted(os.listdir(ds_dir)):
            path = os.path.join(ds_dir, cond, "raw.csv")
            if not os.path.isfile(path):
                continue
            try:
                rows = list(csv.DictReader(open(path)))
            except Exception as e:
                print(f"{model}/{dataset}/{cond}: ERROR {e}")
                continue
            if not rows:
                continue
            n = len(rows)
            fl = Counter(r.get("final_label", "") for r in rows)
            scored = [r for r in rows if r.get("correct") not in ("", None)]
            acc = sum(1 for r in scored if r["correct"] == "1") / len(scored) if scored else None
            esc = fl.get("escalate", 0)
            acc_str = f"{acc:.3f}" if acc is not None else "n/a"
            print(f"{model}/{dataset}/{cond}: n={n} escalate={esc} ({esc/n:.1%}) acc={acc_str} ({len(scored)} scored)")
