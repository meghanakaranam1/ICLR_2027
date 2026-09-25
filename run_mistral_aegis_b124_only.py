"""
Runs ONLY mistral:7b/aegis's graph_ucb_B124 extension (100 -> 400), reusing
run_extension_n400.py's own get_old_examples/get_new_examples/run_condition
UNMODIFIED -- the only difference from run_extension_n400.py's run_pair()
is that this loops over exactly one condition (graph_ucb_B124) instead of
graph_mv + all of BUDGET_GRID.

Built because run_extension_n400.py has no per-condition selector, and B75
is being run concurrently elsewhere (this machine's own local job) --
touching B75 here too would duplicate/conflict with that work. graph_mv
and B100 are already complete for this pair (400/400 each) so there was
nothing to gain from including them either.

Usage:
    python3 run_mistral_aegis_b124_only.py
"""
import sys, os, time
sys.path.insert(0, ".")
import run_all_datasets as m
import run_extension_n400 as ext

MODEL = "mistral:7b"
DATASET = "aegis"
BUDGET = 124
NEW_N = 300  # 100 -> 400, matching NEW_TOTAL_DEFAULT

m._check_ollama_running()

model_dir = MODEL.replace(":", "_")
old_examples, old_ids = ext.get_old_examples(model_dir, DATASET, m.RESULTS_DIR_NAME)
new_examples = ext.get_new_examples(DATASET, old_ids, NEW_N)
combined = old_examples + new_examples
print(f"{MODEL}/{DATASET}/graph_ucb_B{BUDGET}: {len(old_examples)} old + {len(new_examples)} new = {len(combined)} total")

node_systems = ext.get_node_systems(DATASET)
node_arms = m.build_node_arms(m.DATASET_CONFIGS[DATASET]["labels"])

existing_path = os.path.join(m.BASE_DIR, m.RESULTS_DIR_NAME, model_dir, DATASET, f"graph_ucb_B{BUDGET}", "raw.csv")
if os.path.exists(existing_path):
    import csv
    with open(existing_path) as f:
        existing_n = sum(1 for _ in csv.DictReader(f))
    print(f"already on disk: {existing_n}/{len(combined)} rows")
    if existing_n >= len(combined):
        print("already complete, nothing to do.")
        sys.exit(0)

def fn(text, b=BUDGET):
    return m.run_graph_ucb(MODEL, text, b, node_systems, node_arms, sanity=False)

t0 = time.time()
m.run_condition(MODEL, DATASET, f"graph_ucb_B{BUDGET}", fn, combined, sanity=False)
print(f"\nDone in {(time.time()-t0)/60:.1f} min")
