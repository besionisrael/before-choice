"""CPU-only pre-flight checks for BTC-V2 runs (no GPU, no model weights downloaded).

Run in the environment that mirrors the target container image:
    .venv-cpucheck\\Scripts\\python src/v2_cpu_checks.py qwen3_32b olmo31_32b   (torch 2.5.1, transformers 4.57.1)
    .venv-gemma4\\Scripts\\python   src/v2_cpu_checks.py gemma4_31b             (torch 2.6.0, transformers 5.23.0)

Per model:
  1. tokenizer at pinned revision: A/B single-token ids; official template + frozen tail on all 1,560 V2 prompts;
     container-path token counts (tok(text, add_special_tokens=False)) written for cross-env comparison;
     for Gemma 4: rendered text of transformers 5.x == jinja2 rendering used locally (string equality).
  2. architecture from the real config.json on the META device (no weights): number of hidden states,
     vocabulary size of the logits, A/B ids inside the vocabulary.
  3. end-to-end execution of v2_remote.infer on CPU with a TINY RANDOM model of the same architecture
     family and the real tokenizer: all in-container assertions, full-vocabulary top-1 fields, replay,
     JSON-serialisability, and that a tampered payload is rejected.
Writes outputs/v2_audit/cpu_check_<model>.json.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from v2_common import MODELS, N_REPLAY, POOLS, prompt_digest, render  # noqa: E402


def v2_rows():
    rows = []
    for p in ("SCREEN", "COMPETENCE"):
        f = ROOT / "data" / "v2" / f"{POOLS[p]['file']}_model_inputs.jsonl"
        rows += [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()]
    return rows


def tiny_config(cfg):
    d = cfg.to_dict()
    if "per_layer_config" in d.get("text_config", {}):  # Gemma 4: heterogeneous per-layer config
        t = d["text_config"]
        plc = t["per_layer_config"]
        full_key = sorted(plc)[0]
        t.update(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                 num_key_value_heads=2, head_dim=16, layer_types=["sliding_attention", "full_attention"])
        fl = dict(plc[full_key])
        for k in list(fl):
            if k in ("head_dim", "global_head_dim"):
                fl[k] = 16
            if k in ("num_key_value_heads", "num_global_key_value_heads"):
                fl[k] = 2
            if k in ("num_attention_heads",):
                fl[k] = 4
            if k in ("intermediate_size",):
                fl[k] = 128
        t["per_layer_config"] = {"01": fl}
        v = d["vision_config"]
        v.update(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                 num_key_value_heads=2, head_dim=16)
        return type(cfg).from_dict(d)
    c = copy.deepcopy(cfg)
    tc = getattr(c, "text_config", c)
    for k, v in dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=16).items():
        if hasattr(tc, k):
            setattr(tc, k, v)
    if hasattr(tc, "layer_types") and tc.layer_types:
        tc.layer_types = tc.layer_types[:2]
    for k in ("global_head_dim", "num_global_key_value_heads"):
        if hasattr(tc, k) and getattr(tc, k):
            setattr(tc, k, 16 if "dim" in k else 2)
    if hasattr(c, "vision_config") and c.vision_config is not None:
        vc = c.vision_config
        for k, v in dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=16).items():
            if hasattr(vc, k):
                setattr(vc, k, v)
    return c


def check(model_key: str) -> dict:
    import torch
    import transformers
    from transformers import AutoConfig, AutoTokenizer
    m = MODELS[model_key]
    rep: dict = {"model_key": model_key, "repo": m["repo"], "revision": m["revision"],
                 "torch": torch.__version__, "transformers": transformers.__version__}
    tok = AutoTokenizer.from_pretrained(m["repo"], revision=m["revision"])
    assert tok.convert_tokens_to_ids("A") == m["A_id"] and tok.convert_tokens_to_ids("B") == m["B_id"]
    assert tok.encode("A", add_special_tokens=False) == [m["A_id"]] and tok.encode("B", add_special_tokens=False) == [m["B_id"]]
    rows = v2_rows()
    counts, n_tail = {}, 0
    jinja_equal = None
    if m["stack"] == "gemma4":
        from v2_token_audit import encoder
        jrender, _, _, _ = encoder(m["repo"], m["revision"], m["template_kwargs"])
        jinja_equal = 0
    for r in rows:
        text = render(tok, model_key, r["prompt"])  # asserts frozen tail
        n_tail += 1
        counts[r["world_id"]] = len(tok(text, add_special_tokens=False).input_ids)
        if jinja_equal is not None:
            jinja_equal += int(jrender(r["prompt"]) == text)
    rep["template_tail_ok"] = f"{n_tail}/{len(rows)}"
    if jinja_equal is not None:
        rep["gemma_jinja_equals_transformers5"] = f"{jinja_equal}/{len(rows)}"
        assert jinja_equal == len(rows)
    rep["token_counts_sha256"] = hashlib.sha256(json.dumps(counts, sort_keys=True).encode()).hexdigest()
    rep["pre_answer_last_token"] = tok.convert_ids_to_tokens(tok(render(tok, model_key, rows[0]["prompt"]), add_special_tokens=False).input_ids[-1])

    # ---- 2. architecture on meta device (real config, no weights)
    cfg = AutoConfig.from_pretrained(m["repo"], revision=m["revision"])
    if m["stack"] == "gemma4":
        from transformers import Gemma4ForConditionalGeneration as Cls
    else:
        from transformers import AutoModelForCausalLM as Cls
    with torch.device("meta"):
        big = Cls._from_config(cfg) if hasattr(Cls, "_from_config") else Cls.from_config(cfg)
    lm_head = big.get_output_embeddings()
    rep["meta_vocab_out"] = int(lm_head.weight.shape[0])
    tc = getattr(cfg, "text_config", cfg)
    rep["meta_num_hidden_layers"] = int(tc.num_hidden_layers)
    rep["n_hidden_states_expected"] = m["n_hidden_states"]
    assert tc.num_hidden_layers + 1 == m["n_hidden_states"]
    assert max(m["A_id"], m["B_id"]) < rep["meta_vocab_out"]
    rep["meta_params_b"] = round(sum(p.numel() for p in big.parameters()) / 1e9, 2)
    del big

    # ---- 3. end-to-end infer() on CPU with a tiny random model
    from v2_remote import infer
    torch.manual_seed(0)
    small = Cls._from_config(tiny_config(cfg), torch_dtype=torch.float32) if hasattr(Cls, "_from_config") else Cls.from_config(tiny_config(cfg))
    sub = [r for r in rows if r["pool"] == "SCREEN"][:25]
    payload = [{"world_id": r["world_id"], "prompt": r["prompt"], "prompt_sha256": r["prompt_sha256"],
                "expected_n_tokens": counts[r["world_id"]]} for r in sub]
    res = infer(model_key, payload, prompt_digest(payload), guard_s=600, device="cpu", _model_override=small)
    json.dumps(res)  # must be JSON-serialisable
    R = res["results"]
    assert len(R) == len(sub) + min(N_REPLAY, len(sub))
    assert all(x["vocab_size"] == rep["meta_vocab_out"] for x in R), "logits must span the full vocabulary"
    assert all(isinstance(x["top1_in_AB"], bool) and x["top1_id"] == x["top5"][0][0] for x in R)
    rep["e2e_tiny_model"] = {"n_results": len(R), "vocab_size": R[0]["vocab_size"],
                             "top1_in_AB_rate_random_model": sum(x["top1_in_AB"] for x in R) / len(R),
                             "replay_identical": all(R[i]["logp_A"] == R[len(sub) + i]["logp_A"] for i in range(min(N_REPLAY, len(sub))))}
    # tampered payloads must be rejected
    rejected = {}
    for name, bad in {"extra_field": [dict(payload[0], oracle={"x": 1})] + payload[1:],
                      "wrong_digest": payload, "wrong_tokens": [dict(payload[0], expected_n_tokens=1)] + payload[1:]}.items():
        dig = prompt_digest(bad) if name != "wrong_digest" else "0" * 64
        try:
            infer(model_key, bad, dig, guard_s=600, device="cpu", _model_override=small)
            rejected[name] = False
        except AssertionError:
            rejected[name] = True
    rep["tamper_rejected"] = rejected
    assert all(rejected.values())
    out = ROOT / "outputs" / "v2_audit" / f"cpu_check_{model_key}.json"
    out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    (ROOT / "outputs" / "v2_audit" / f"token_counts_{model_key}.json").write_text(json.dumps(counts, sort_keys=True), encoding="utf-8")
    return rep


if __name__ == "__main__":
    for mk in sys.argv[1:]:
        print(json.dumps(check(mk), indent=1))
