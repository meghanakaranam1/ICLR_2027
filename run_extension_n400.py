"""
Production runner: extends 5 model/dataset pairs from N=100 to N=400 by
combining the existing 100 examples with 300 new ones (pre-verified
zero-overlap, zero-duplicate) and passing the combined 400-example list to
run_condition() for each of that pair's applicable conditions --
run_condition()'s own existing per-example checkpoint logic detects the
100 already-completed input_ids and skips them, appending only the 300
new ones. This exact mechanism was sanity-tested twice already
(new-only list, then combined-list) on llama3.1:8b/hatemoderate -- see
test_extension_append.py / test_extension_combined_list.py.

Resumability: this script carries NO separate checkpoint state of its
own. If interrupted and re-run, already-fully-done (pair, condition)
combinations are detected almost instantly by run_condition() (reads
"400 done, 0 to run" and returns), and whatever was mid-progress resumes
via run_condition()'s own per-example resume logic. Safe to re-run from
the top at any time.

Usage:
  Real run (all 5 pairs, 300 new each, writes to real results_02):
    python3 run_extension_n400.py

  Dry run (ONE pair, small N, isolated temp directory, real data untouched):
    python3 run_extension_n400.py --dry-run --pair qwen2.5:7b aegis --dry-run-n 8
"""
import sys, os, csv, time, random, shutil, argparse
sys.path.insert(0, ".")
import run_all_datasets as m

# (model, dataset, isocompute_n_mv or None) -- isocompute only included where
# a real calibrated n_mv exists for that specific pair (confirmed via the
# scratchpad calibration scripts; qwen2.5:7b/aegis and mistral:7b/hatemoderate
# were never calibrated for isocompute, so that condition is correctly
# excluded for them, matching each pair's actual existing condition set).
PAIRS = [
    ("llama3.1:8b", "hatemoderate", 140),  # already fully finished including isocompute, unaffected by this change
    # isocompute dropped (2026-09-21, time-saving decision) for the 2 pairs
    # still in progress -- was 95/100 respectively. llama3.1:8b/hatemoderate
    # above already completed its isocompute condition before this change,
    # so it's left as-is; these two simply won't run/extend isocompute now.
    ("llama3.1:8b", "aegis", None),
    ("qwen2.5:7b", "hatemoderate", None),
    ("qwen2.5:7b", "aegis", None),
    ("mistral:7b", "hatemoderate", None),
    ("mistral:7b", "aegis", None),  # added 2026-09-21: extend after the prompt-fix refix job finishes at N=100
    # added 2026-09-22: "big"/frontier-scale open-weight backbones, previously
    # only run at N=100 as a sanity check, not part of the N=400 primary
    # results. Run only once Spark has memory headroom (llama3.1:70b needs
    # ~42GB, qwen2.5:32b ~19GB) -- don't launch alongside another active
    # small-model job without checking `free -h` first.
    ("llama3.1:70b", "aegis", None),
    ("llama3.1:70b", "hatemoderate", None),
    ("qwen2.5:32b", "hatemoderate", None),
]

NEW_TOTAL_DEFAULT = 400
SEED_POS = m.SEED + 1000
SEED_NEG = m.SEED + 1001


def get_node_systems(dataset_name):
    if dataset_name == "hatemoderate":
        return m.build_hatemoderate_node_systems()
    elif dataset_name == "aegis":
        return m.build_aegis_node_systems()
    raise ValueError(f"unsupported dataset: {dataset_name}")


ORIGINAL_N = 100  # every pair's pre-extension baseline size, fixed for this script's scope


def get_old_examples(model_dir, dataset_name, results_dir_name):
    """Reconstructs the ORIGINAL 100 example objects by matching on
    (id, ground_truth) together, NOT id alone -- HateModerate specifically
    has 538 ids reused across two different (text, label) rows, confirmed
    to affect 15 of these exact 100 ids. Matching on the label too (read
    from the existing raw.csv, which is guaranteed internally consistent
    with the original text -- see the code-path verification done earlier)
    disambiguates all 15 cleanly.

    CRITICAL (2026-09-21 fix): only the FIRST ORIGINAL_N rows (by file
    order -- CSVs here are append-only, so this is insertion order) are
    ever treated as "old". graph_mv/raw.csv is the SAME file this script
    appends new rows into, so reading "whatever's currently in the file"
    without this cap is a real bug: after a crash mid-graph_mv, a wrapper
    restart would see the partial progress from the last attempt (e.g. 110
    rows) as the new "old" baseline and draw ANOTHER full new_n batch on
    top of it, instead of recognizing it only needs the remainder -- each
    crash compounds the total further with no fixed target ever reached.
    Confirmed happening live: a qwen2.5:7b/aegis test run drifted
    100->114->128->142 over two crashes before this fix. Capping to the
    first ORIGINAL_N rows makes old_ids (and therefore the new_examples
    drawn from get_new_examples()) fully deterministic and stable across
    any number of crash-restarts, so run_condition()'s own per-example
    checkpointing can correctly converge on exactly ORIGINAL_N + new_n
    every time."""
    existing_path = os.path.join(m.BASE_DIR, results_dir_name, model_dir, dataset_name, "graph_mv", "raw.csv")
    with open(existing_path) as f:
        original_rows = list(csv.DictReader(f))[:ORIGINAL_N]
    original_ids = set(r["input_id"] for r in original_rows)
    gt_by_id = {r["input_id"]: r["ground_truth"] for r in original_rows}

    cfg = m.DATASET_CONFIGS[dataset_name]
    all_examples = cfg["loader"]()
    old_examples = [e for e in all_examples
                    if e["id"] in original_ids and e["label"] == gt_by_id[e["id"]]]
    assert len(old_examples) == len(original_ids), (
        f"reconstructed {len(old_examples)} old examples, expected {len(original_ids)}"
    )
    return old_examples, original_ids


def get_new_examples(dataset_name, excluded_ids, new_n):
    cfg = m.DATASET_CONFIGS[dataset_name]
    labels = cfg["labels"]
    pos, neg = labels
    all_examples = cfg["loader"]()
    remaining_pool = [e for e in all_examples if e["id"] not in excluded_ids]
    pos_remaining = [e for e in remaining_pool if e["label"] == pos]
    neg_remaining = [e for e in remaining_pool if e["label"] == neg]

    k_new = new_n // 2
    rng_pos = random.Random(SEED_POS)
    rng_neg = random.Random(SEED_NEG)
    new_pos = rng_pos.sample(pos_remaining, k_new)
    new_neg = rng_neg.sample(neg_remaining, k_new)
    new_examples = new_pos + new_neg

    new_ids = set(e["id"] for e in new_examples)
    assert len(new_ids) == len(new_examples), "duplicate ids within the new sample"
    assert not (excluded_ids & new_ids), "overlap between new sample and excluded (old) ids"
    return new_examples


def run_pair(model, dataset_name, isocompute_n_mv, new_n, results_dir_name):
    pair_label = f"{model}/{dataset_name}"
    model_dir = model.replace(":", "_")
    pair_start = time.time()

    old_examples, old_ids = get_old_examples(model_dir, dataset_name, results_dir_name)
    new_examples = get_new_examples(dataset_name, old_ids, new_n)
    combined = old_examples + new_examples
    print(f"\n{'='*70}\n{pair_label}  (extending {len(old_examples)} -> {len(combined)})\n{'='*70}")
    print(f"{pair_label}: {len(old_examples)} old + {len(new_examples)} new = {len(combined)} total")

    node_systems = get_node_systems(dataset_name)
    node_arms = m.build_node_arms(m.DATASET_CONFIGS[dataset_name]["labels"])

    conditions = [("graph_mv", None)] + [(f"graph_ucb_B{b}", b) for b in m.BUDGET_GRID]
    if isocompute_n_mv is not None:
        conditions.append(("graph_mv_isocompute", "isocompute"))

    for cond_name, param in conditions:
        t0 = time.time()
        elapsed_in_pair = (t0 - pair_start) / 60
        print(f"\n--- {pair_label} / {cond_name} --- (elapsed in this pair so far: {elapsed_in_pair:.1f} min)")

        # Fast completeness check (2026-09-20): if this condition's raw.csv
        # already has exactly len(combined) rows, skip calling
        # run_condition() at all -- avoids the cost of re-opening/re-reading
        # the file just to discover "0 to run" again, and gives immediate,
        # clear visibility into which (pair, condition) combos are already
        # fully done when resuming after an interruption.
        existing_path = os.path.join(m.BASE_DIR, results_dir_name, model_dir, dataset_name, cond_name, "raw.csv")
        if os.path.exists(existing_path):
            with open(existing_path) as f:
                existing_n = sum(1 for _ in csv.DictReader(f))
            if existing_n >= len(combined):
                print(f"    already complete, skipping ({existing_n}/{len(combined)} rows present)")
                continue

        if cond_name == "graph_mv":
            def fn(text):
                return m.run_graph_mv(model, text, node_systems, node_arms, sanity=False)
        elif cond_name == "graph_mv_isocompute":
            def fn(text):
                return m.run_graph_mv(model, text, node_systems, node_arms, sanity=False, n_mv=isocompute_n_mv)
        else:
            budget = param
            def fn(text, b=budget):
                return m.run_graph_ucb(model, text, b, node_systems, node_arms, sanity=False)

        m.run_condition(model, dataset_name, cond_name, fn, combined, sanity=False)
        dt = time.time() - t0
        print(f"--- {pair_label} / {cond_name} finished in {dt/60:.1f} min ---")

    print(f"\n{pair_label}: all conditions done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pair", nargs=2, metavar=("MODEL", "DATASET"))
    parser.add_argument("--dry-run-n", type=int, default=8)
    parser.add_argument("--new-n", type=int, default=None,
                         help="override how many NEW examples to add on a real run "
                              "(default 300, i.e. 100->400). Useful for a small timed "
                              "test slice -- unlike --dry-run, this writes real rows "
                              "into results_02, so nothing is wasted; a later full run "
                              "just continues from wherever this left off.")
    args = parser.parse_args()

    m._check_ollama_running()

    if args.dry_run:
        assert args.pair, "--dry-run requires --pair MODEL DATASET"
        dry_model, dry_dataset = args.pair
        iso_n_mv = next((iso for mo, da, iso in PAIRS if mo == dry_model and da == dry_dataset), None)
        model_dir = dry_model.replace(":", "_")

        # Pair-specific tmp dir name (2026-09-20 fix): previously a single
        # fixed "results_TEST_extension_production_tmp" name shared across
        # ALL --dry-run invocations, which would race (one process's rmtree
        # deleting another's in-progress data) if two dry-runs for different
        # pairs were ever launched concurrently -- exactly the scenario this
        # script is now being used for (testing parallel Spark execution).
        tmp_name = f"results_TEST_extension_production_tmp_{model_dir}_{dry_dataset}"
        tmp_dir = os.path.join(m.BASE_DIR, tmp_name)
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        real_graph_mv = os.path.join(m.BASE_DIR, "results_02", model_dir, dry_dataset, "graph_mv")
        tmp_graph_mv = os.path.join(tmp_dir, model_dir, dry_dataset, "graph_mv")
        os.makedirs(tmp_graph_mv, exist_ok=True)
        shutil.copy(os.path.join(real_graph_mv, "raw.csv"), os.path.join(tmp_graph_mv, "raw.csv"))
        print(f"DRY RUN: copied real {dry_model}/{dry_dataset} baseline into isolated {tmp_name}, "
              f"new_n={args.dry_run_n} (real results_02 untouched)")

        m.RESULTS_DIR_NAME = tmp_name
        t_start = time.time()
        run_pair(dry_model, dry_dataset, iso_n_mv, args.dry_run_n, tmp_name)
        print(f"\nDRY RUN total elapsed: {(time.time()-t_start)/60:.1f} min")

        # Verify and clean up
        with open(os.path.join(tmp_graph_mv, "raw.csv")) as f:
            final_rows = list(csv.DictReader(f))
        print(f"DRY RUN verification: graph_mv final row count = {len(final_rows)} "
              f"(expected 100 + {args.dry_run_n})")
        shutil.rmtree(tmp_dir)
        print("Cleaned up temp directory.")
    else:
        # --pair on a REAL (non-dry) run (2026-09-20): runs just that one
        # pair against the real results_02, instead of all 5 sequentially.
        # Added to support running different pairs as separate concurrent
        # processes on Spark (its GB10 unified-memory architecture has
        # enough headroom to hold multiple small models loaded at once --
        # confirmed via nvidia-smi/ollama ps -- but real multi-model compute
        # parallelism on shared GPU cores is unverified, so pairs should be
        # tested 2-at-a-time on a small slice before committing all 5 to
        # simultaneous full runs; see the parallel-vs-sequential timing
        # comparison this was built for).
        run_pairs = PAIRS
        if args.pair:
            pair_model, pair_dataset = args.pair
            run_pairs = [p for p in PAIRS if p[0] == pair_model and p[1] == pair_dataset]
            assert run_pairs, f"no PAIRS entry matches {pair_model}/{pair_dataset}"

        new_n = args.new_n if args.new_n is not None else (NEW_TOTAL_DEFAULT - 100)

        t_start = time.time()
        for model, dataset_name, iso_n_mv in run_pairs:
            run_pair(model, dataset_name, iso_n_mv, new_n, m.RESULTS_DIR_NAME)
            print(f"\n[total elapsed so far: {(time.time()-t_start)/60:.1f} min]")
        print(f"\n{len(run_pairs)} PAIR(S) DONE. Total time: {(time.time()-t_start)/3600:.2f}h")
