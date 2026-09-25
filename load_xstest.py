"""
load_xstest.py
---------------
Load XSTest (v2) for content moderation. Binary labels: safe / unsafe.

Verified by an actual live load (no prior XSTest reference existed anywhere
in this repo to check against):

  natolambert/xstest-v2-copy has 6 splits (gpt4, llama2new, llama2orig,
  mistralguard, mistralinstruct, prompts) -- one per model's completions,
  plus a prompt-only 'prompts' split. `final_label` in the model splits is
  a completion-compliance annotation (e.g. '1_full_compliance'), NOT a
  prompt-level safety label -- do not use it as ground truth.

  The prompt-level ground truth is the 'type' field on the 'prompts' split:
  450 rows, 18 distinct types -- 8 prefixed 'contrast_' (200 rows, unsafe)
  and 10 not (250 rows, safe). NOT an even 225/225 split (corrected
  2026-09-19 -- confirmed via a live category count on the cached dataset,
  see below). The 10 safe types include 'nons_group_real_discr' and
  'real_group_nons_discr' (T6/T7), which are each other's contrast pair by
  construction and so don't get their own separate 'contrast_' entries --
  that's why there are only 8 contrast_ types against 10 non-contrast ones,
  not a symmetric 9/9. 'contrast_*' categories are the unsafe set
  (genuinely unsafe requests worded like the safe ones, used to test
  over-refusal); the rest are the safe set.

Perf note (2026-09-16): see load_aegis.py's matching note -- HF_HUB_OFFLINE=1
skips a slow/flaky network connectivity check at `import datasets` time
(~460s -> ~2s on this machine), safe here since the dataset is cached.
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def load_xstest() -> list:
    """
    Load natolambert/xstest-v2-copy, 'prompts' split.
    Returns list of dicts: {id, text, label, category}.
    label is 'safe' or 'unsafe', derived from whether `type` is prefixed
    'contrast_'. category is the raw `type` field.
    """
    from datasets import load_dataset

    print("\n" + "="*60)
    print("Loading: XSTest (natolambert/xstest-v2-copy, 'prompts' split)")
    print("="*60)

    ds = load_dataset("natolambert/xstest-v2-copy", split="prompts")
    print(f"Sample row: {ds[0]}")

    examples = []
    for row in ds:
        text = (row.get("prompt") or "").strip()
        if not text:
            continue
        rtype = row.get("type") or ""
        examples.append({
            "id":       row.get("id"),
            "text":     text,
            "label":    "unsafe" if rtype.startswith("contrast_") else "safe",
            "category": rtype,
        })

    n_safe = sum(1 for e in examples if e["label"] == "safe")
    print(f"Loaded {len(examples)} examples (safe={n_safe}, unsafe={len(examples) - n_safe})")
    return examples
