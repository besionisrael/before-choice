"""Inference body executed INSIDE the Modal container (BTC-V2 S/C runs). Behaviour only:
one forward pass per world, no generation, no gradients, no hidden states captured (that is D3).

Records, per world: log p(A), log p(B) (forced-choice inputs) AND the top-1 / top-5 tokens of the
FULL-VOCABULARY next-token distribution (S4 / C3b conformity is computed from the full vocabulary,
independently of the forced A/B choice). Returns JSON-serialisable Python types only.
"""

from __future__ import annotations

import hashlib
import time

from v2_common import MODELS, N_REPLAY, SEND_FIELDS, prompt_digest, render


def infer(model_key: str, rows: list[dict], expected_digest: str, guard_s: int,
          device: str = "cuda", _model_override=None) -> dict:
    """`device`/`_model_override` exist ONLY for CPU tests (tiny random model); Modal runs use defaults."""
    import torch
    import transformers
    from transformers import AutoTokenizer

    m = MODELS[model_key]
    t0 = time.time()
    tm = {"remote_start_epoch": t0}
    # ---- oracle isolation + dataset integrity
    for r in rows:
        assert tuple(sorted(r)) == tuple(sorted(SEND_FIELDS)), f"unexpected fields {sorted(r)}"
        assert hashlib.sha256(r["prompt"].encode("utf-8")).hexdigest() == r["prompt_sha256"]
    assert prompt_digest(rows) == expected_digest, "prompt digest mismatch (dataset not the frozen pool)"
    # ---- tokenizer, template, token ids
    tok = AutoTokenizer.from_pretrained(m["repo"], revision=m["revision"])
    assert tok.convert_tokens_to_ids("A") == m["A_id"] and tok.convert_tokens_to_ids("B") == m["B_id"]
    assert tok.encode("A", add_special_tokens=False) == [m["A_id"]]
    assert tok.encode("B", add_special_tokens=False) == [m["B_id"]]
    tm["tokenizer_s"] = time.time() - t0
    # ---- model (weights straight to GPU; bf16)
    t1 = time.time()
    if m["stack"] == "gemma4":
        from transformers import Gemma4ForConditionalGeneration as Cls
    else:
        from transformers import AutoModelForCausalLM as Cls
    if _model_override is not None:
        model = _model_override.eval()
    else:
        model = Cls.from_pretrained(m["repo"], revision=m["revision"], dtype=torch.bfloat16, device_map="cuda").eval()
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    sync()
    tm["load_s"] = time.time() - t1
    print(f"model ready at {time.time() - t0:.1f}s", flush=True)

    plan = [(0, r) for r in rows] + [(1, r) for r in rows[:N_REPLAY]]
    out = []
    t1 = time.time()
    with torch.no_grad():
        for i, (rep, r) in enumerate(plan):
            if time.time() - t0 > guard_s:
                raise RuntimeError(f"wall-clock guard {guard_s}s reached after {i} forwards; aborting")
            text = render(tok, model_key, r["prompt"])
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            assert ids.shape[1] == r["expected_n_tokens"], (r["world_id"], int(ids.shape[1]), r["expected_n_tokens"])
            logits = model(input_ids=ids).logits[0, -1]  # pre-answer position = last template token
            lp = torch.log_softmax(logits.float(), dim=-1)  # FULL vocabulary
            top = lp.topk(5)
            top1 = int(top.indices[0])
            out.append({"world_id": r["world_id"], "replay": rep, "n_tokens": int(ids.shape[1]),
                        "vocab_size": int(lp.shape[0]),
                        "logp_A": float(lp[m["A_id"]]), "logp_B": float(lp[m["B_id"]]),
                        "logit_A": float(logits[m["A_id"]]), "logit_B": float(logits[m["B_id"]]),
                        "top1_id": top1, "top1_token": tok.convert_ids_to_tokens(top1),
                        "top1_in_AB": top1 in (m["A_id"], m["B_id"]),
                        "top5": [[int(j), tok.convert_ids_to_tokens(int(j)), float(v)] for v, j in zip(top.values, top.indices)]})
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(plan)} forwards at {time.time() - t0:.1f}s", flush=True)
    sync()
    tm["forwards_s"] = time.time() - t1
    tm["n_forwards"] = len(plan)
    tm["remote_end_epoch"] = time.time()
    return {"results": out, "timing": tm, "wall_s": time.time() - t0,
            "env": {"torch": str(torch.__version__), "transformers": str(transformers.__version__),
                    "cuda": str(torch.version.cuda),
                    "gpu": str(torch.cuda.get_device_name()) if device == "cuda" else "cpu-test",
                    "max_mem_gb": float(torch.cuda.max_memory_allocated() / 1e9) if device == "cuda" else 0.0,
                    "dtype": "bfloat16", "test_mode": _model_override is not None,
                    "model": m["repo"], "revision": m["revision"]}}
