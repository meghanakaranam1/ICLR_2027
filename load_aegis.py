"""
load_aegis.py
--------------
Load NVIDIA Aegis AI Content Safety Dataset 2.0, unfiltered, for general
content moderation. Binary labels: safe / unsafe.

Uses the 'prompt' column as input text and 'prompt_label' as ground truth
(prompt-level, human-annotated -- see the AEGIS 2.0 label-structure
inspection earlier in this project). No category/severity filtering is
applied: this is the full prompt_label-labeled set for the requested split,
matching the "AEGIS (unfiltered)" item in the settled four-dataset plan.

Perf note (2026-09-16): `import datasets` alone was taking ~460s on this
machine -- huggingface_hub does a network connectivity check at import
time, and that was hanging/retrying against this machine's flaky network
(same class of issue as the Ollama pull failures earlier tonight). Since
this dataset is already cached locally (~/.cache/huggingface/datasets/),
HF_HUB_OFFLINE=1 skips that network check entirely: import + full load
drops from ~600s to ~5s. Set as setdefault so an explicit override (e.g. a
real cache-miss scenario needing a fresh download) still works.
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def load_aegis(split: str = "test") -> list:
    """
    Load nvidia/Aegis-AI-Content-Safety-Dataset-2.0.

    split: "test"  → only the test split  (1,964 rows; prompt_label is
                      human-annotated on every row in this split)
           "train" → only the train split
           "all"   → train + validation + test

    Returns list of dicts: {id, text, label, category}.
    label is 'safe' or 'unsafe', from prompt_label.
    category is the raw violated_categories field (kept for reference; not
    used for filtering here).
    """
    from datasets import load_dataset

    print("\n" + "="*60)
    print(f"Loading: nvidia/Aegis-AI-Content-Safety-Dataset-2.0 (split={split}, unfiltered)")
    print("="*60)

    ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0")

    splits_to_use = ["train", "validation", "test"] if split == "all" else [split]

    # Diagnostic: confirm prompt_label's null rate on the actual loaded data
    # for these splits -- don't assume it matches a different column's
    # (e.g. response_label's) null rate from an earlier, separate inspection.
    total_rows, null_prompt_label = 0, 0
    for s in splits_to_use:
        for row in ds[s]:
            total_rows += 1
            if not (row.get("prompt_label") or "").strip():
                null_prompt_label += 1
    print(
        f"prompt_label null-rate check ({', '.join(splits_to_use)}): "
        f"{null_prompt_label}/{total_rows} rows "
        f"({100.0 * null_prompt_label / max(total_rows, 1):.1f}%) have empty prompt_label"
    )

    examples = []
    seen_ids = set()
    for s in splits_to_use:
        for row in ds[s]:
            gt   = (row.get("prompt_label") or "").strip().lower()
            text = (row.get("prompt") or "").strip().encode('ascii', errors='replace').decode('ascii')
            if gt not in ("safe", "unsafe") or not text:
                continue
            if row["id"] in seen_ids:
                continue
            seen_ids.add(row["id"])
            examples.append({
                "id":       row["id"],
                "text":     text,
                "label":    gt,
                "category": (row.get("violated_categories") or "").strip(),
            })

    n_safe = sum(1 for e in examples if e["label"] == "safe")
    print(f"Loaded {len(examples)} examples (safe={n_safe}, unsafe={len(examples) - n_safe})")
    return examples
