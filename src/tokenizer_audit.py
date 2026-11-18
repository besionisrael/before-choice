"""D-3 tokenizer audit (CPU, tokenizer + config files only; no weights).

Checks, under the official chat template (non-thinking, add_generation_prompt):
  * answer letters: which strings are single tokens; what the template emits last
  * token length per world; base/cf and pathcf pair alignment
    (same length, differing positions only inside manipulated label spans)
Writes outputs/metrics/tokenizer_audit.json.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"  # pinned 2026-11-13 (HF last_modified 2025-09-17)


def load(p):
    return [json.loads(x) for x in (ROOT / p).read_text(encoding="utf-8").splitlines()]


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    enc = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True,
                                            add_generation_prompt=True)
    rows = load("data/generated/pilot_v1.jsonl") + load("data/generated/pilot_v1_pathcf.jsonl")
    ids = {r["world_id"]: enc(r["prompt"]) for r in rows}
    by = {r["world_id"]: r for r in rows}
    res: dict = {"model_id": MODEL_ID, "revision": REVISION,
                 "transformers": __import__("transformers").__version__}
    res["answer_tokens"] = {s: tok.encode(s, add_special_tokens=False) for s in ("A", "B", " A", " B")}
    res["template_tail"] = tok.convert_ids_to_tokens(ids["F0000-base"][-6:])
    res["template_example"] = tok.apply_chat_template([{"role": "user", "content": "X"}], tokenize=False,
                                                     add_generation_prompt=True)
    lab_len = Counter()
    for r in rows:
        for e in r["transitions"]:
            for x in (e["source"], e["target"]):
                lab_len[len(tok.encode(" " + x, add_special_tokens=False))] += 1
    res["label_token_length_hist_with_leading_space"] = dict(sorted(lab_len.items()))
    lens = defaultdict(list)
    for r in rows:
        lens[(r["variant"], r["depth"])].append(len(ids[r["world_id"]]))
    res["prompt_tokens_by_variant_depth"] = {f"{v}|d{d}": [min(L), max(L)] for (v, d), L in sorted(lens.items())}

    def compare(a, b):
        x, y = ids[a], ids[b]
        if len(x) != len(y):
            return {"aligned": False, "len": [len(x), len(y)]}
        return {"aligned": True, "n_diff_positions": sum(i != j for i, j in zip(x, y))}

    pairs = []
    for r in rows:
        v = r["variant"]
        if v.endswith("_cf") or v == "cf":
            pairs.append(("cf", r["world_id"].replace("cf", "base") if v != "cf" else r["family_id"] + "-base", r["world_id"]))
        if "source_world_id" in r:
            pairs.append((v.split("_")[0], r["source_world_id"], r["world_id"]))
    summary = defaultdict(Counter)
    misaligned = []
    for kind, a, b in pairs:
        c = compare(a, b)
        summary[kind]["aligned" if c["aligned"] else "MISALIGNED"] += 1
        if not c["aligned"]:
            misaligned.append([kind, a, b, c["len"]])
    res["pair_alignment"] = {k: dict(v) for k, v in summary.items()}
    res["misaligned_examples"] = misaligned[:20]
    out = ROOT / "outputs" / "metrics" / "tokenizer_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "prompt_tokens_by_variant_depth"}, indent=1, ensure_ascii=True))
    del by


if __name__ == "__main__":
    sys.exit(main())
