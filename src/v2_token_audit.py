"""CPU tokenizer audit of the V2 pools under each candidate's official chat template (non-thinking).

Checks per model: A/B single-token ids; template tail (pre-answer position); prompt token lengths;
token-length equality of every minimal pair (base/cf-type pairs, swap, pathcf vs source).
Tokenizer/config/template files only (no weights). Gemma 4 is rendered with jinja2 + tokenizers
because the pinned transformers 4.57.1 does not support it.
Writes outputs/v2_audit/token_audit.json.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import jinja2
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "Qwen/Qwen3-32B": ("9216db5781bf21249d130ec9da846c4624c16137", {"enable_thinking": False}),
    "allenai/Olmo-3.1-32B-Instruct": ("ac0587e4a7744a551c059d8cd17ba220bc940dae", {}),
    "google/gemma-4-31B-it": ("842da3794eaa0b77d5f08bae87a17459d91ff475", {"enable_thinking": False}),
}


def encoder(model: str, rev: str, kw: dict):
    if model.startswith("google/gemma-4"):
        tok = Tokenizer.from_file(hf_hub_download(model, "tokenizer.json", revision=rev))
        tpl = Path(hf_hub_download(model, "chat_template.jinja", revision=rev)).read_text(encoding="utf-8")
        env = jinja2.Environment()
        env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
        t = env.from_string(tpl)
        render = lambda p: t.render(messages=[{"role": "user", "content": p}], add_generation_prompt=True,
                                    bos_token="<bos>", **kw)
        enc = lambda s: tok.encode(s, add_special_tokens=False).ids
        ids_of = lambda s: tok.encode(s, add_special_tokens=False).ids
        to_tok = lambda ids: [tok.id_to_token(i) for i in ids]
    else:
        tok = AutoTokenizer.from_pretrained(model, revision=rev)
        render = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                   add_generation_prompt=True, **kw)
        enc = lambda s: tok(s, add_special_tokens=False).input_ids
        ids_of = lambda s: tok.encode(s, add_special_tokens=False)
        to_tok = tok.convert_ids_to_tokens
    return render, enc, ids_of, to_tok


def pairs(rows):
    by = {r["world_id"]: r for r in rows}
    out = []
    for r in rows:
        v, f = r["variant"], r["family_id"]
        if v.endswith("cf") and not v.startswith(("upath", "cpath", "xswap")):
            src = f"{f}-{v[:-2]}base" if v != "cf" else f"{f}-base"
            out.append(("cf_pair", src, r["world_id"]))
        if v == "swap_base":
            out.append(("swap_letters", f"{f}-base", r["world_id"]))
        if "source_world_id" in r:
            out.append((v.split("_")[0], r["source_world_id"], r["world_id"]))
    return [(k, a, b) for k, a, b in out if a in by and b in by]


def main() -> None:
    rows = []
    for p in ("screen", "competence"):
        rows += [json.loads(x) for x in (ROOT / "data" / "v2" / f"{p}.jsonl").read_text(encoding="utf-8").splitlines()]
    res = {}
    for model, (rev, kw) in MODELS.items():
        render, enc, ids_of, to_tok = encoder(model, rev, kw)
        n = {}
        for r in rows:
            n[r["world_id"]] = len(enc(render(r["prompt"])))
        tail = to_tok(enc(render(rows[0]["prompt"]))[-6:])
        mis = Counter()
        tot = Counter()
        for k, a, b in pairs(rows):
            tot[k] += 1
            mis[k] += n[a] != n[b]
        res[model] = {"revision": rev, "template_kwargs": kw, "A_ids": ids_of("A"), "B_ids": ids_of("B"),
                      "pre_answer_tail_tokens": tail,
                      "prompt_tokens_min_max": [min(n.values()), max(n.values())],
                      "pairs_total": dict(tot), "pairs_length_mismatch": dict(mis)}
        print(model, json.dumps(res[model], ensure_ascii=True))
    out = ROOT / "outputs" / "v2_audit" / "token_audit.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
