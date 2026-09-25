"""
load_toxicchat.py
------------------
Load ToxicChat (lmsys/toxic-chat) for content moderation.
Binary labels: toxic / not_toxic.

Verified against the existing, working loader at
Neurips2026/neurips/data/load_toxicchat.py -- same HF dataset id, config,
split, and column names. Also confirmed this dataset is already present in
the local HuggingFace cache (~/.cache/huggingface/datasets/lmsys___toxic-chat),
so it loads without requiring a fresh network fetch.

Perf note (2026-09-16): see load_aegis.py's matching note -- HF_HUB_OFFLINE=1
skips a slow/flaky network connectivity check at `import datasets` time
(~460s -> ~2s on this machine), safe here since the dataset is cached.
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def load_toxicchat(split: str = "test") -> list:
    """
    Load lmsys/toxic-chat, config 'toxicchat0124'.
    Returns list of dicts: {id, text, label}.
    label is 'toxic' or 'not_toxic', from the 'toxicity' column (0/1).
    Text comes from the 'user_input' column.
    """
    from datasets import load_dataset

    print("\n" + "="*60)
    print("Loading: ToxicChat (lmsys/toxic-chat, toxicchat0124)")
    print("="*60)

    token = os.environ.get("HF_TOKEN")
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split=split, token=token)

    print(f"Sample row: {ds[0]}")

    examples = []
    for i, row in enumerate(ds):
        text = (row.get("user_input") or "").strip()
        if not text:
            continue
        examples.append({
            "id":    f"toxicchat_{i}",
            "text":  text,
            "label": "toxic" if int(row["toxicity"]) == 1 else "not_toxic",
        })

    n_toxic = sum(1 for e in examples if e["label"] == "toxic")
    print(f"Loaded {len(examples)} examples (toxic={n_toxic}, not_toxic={len(examples) - n_toxic})")
    return examples
