"""BTC-D2-001 — D2/D3 GPU smoke run (Modal). BILLABLE. REQUIRES EXPLICIT OWNER APPROVAL.

Exact command (PowerShell, project root, venv active, after `modal setup`):
    modal run src/d2_smoke.py

Scope (fixed, no CLI knobs that enlarge it):
  * 20 TRAIN-split worlds chosen deterministically by select_worlds():
    one train family per depth x {base, cf, swap_base, swap_cf}       = 12
    + depth-3 and depth-4 family x {upath_base, upath_cf, cpath_base, cpath_cf} = 8
  * each world run TWICE in the same container (determinism replay)
  * 1 x L4, max_containers=1, retries=0, timeout=900 s, internal wall guard 780 s
  * single forward pass per prompt, no generation, no gradient
  * returns: log-probs of tokens 'A'(32)/'B'(33) at the first answer position, top-5,
    token count, residual stream at the last prompt position for all 37
    hidden_states (embeddings + 36 layers), fp16  -> ~3.8 MB returned to local disk
  * nothing persisted on Modal (no Volume); weights downloaded into the ephemeral container
Only world_id / prompt / prompt_sha256 / expected token count leave the machine.
The oracle never leaves; scoring happens locally in src/score_d2.py.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path

import modal

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
TOKEN_A, TOKEN_B = 32, 33
TEMPLATE_TAIL = "<|im_start|>assistant\n"
WALL_GUARD_S = 780
N_EXPECTED = 20

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1", "transformers==4.57.1", "huggingface_hub==0.36.2", "numpy==2.1.3")
app = modal.App("btc-d2-smoke-001")

ROOT = Path(__file__).resolve().parents[1]


def select_worlds() -> list[dict]:
    rows = [json.loads(x) for x in (ROOT / "data/generated/pilot_v1.jsonl").read_text(encoding="utf-8").splitlines()]
    rows += [json.loads(x) for x in (ROOT / "data/generated/pilot_v1_pathcf.jsonl").read_text(encoding="utf-8").splitlines()]
    by = {r["world_id"]: r for r in rows}
    train = [r for r in rows if r["split"] == "train" and r["variant"] == "base"]
    fam = {d: sorted(r["family_id"] for r in train if r["depth"] == d)[0] for d in (2, 3, 4)}
    ids = [f"{fam[d]}-{v}" for d in (2, 3, 4) for v in ("base", "cf", "swap_base", "swap_cf")]
    ids += [f"{fam[d]}-{v}" for d in (3, 4) for v in ("upath_base", "upath_cf", "cpath_base", "cpath_cf")]
    sel = [by[i] for i in ids]
    assert len(sel) == N_EXPECTED and all(r["split"] == "train" for r in sel)
    return sel


@app.function(gpu="L4", image=image, timeout=900, retries=0, max_containers=1)
def run(rows: list[dict]) -> dict:
    """Never lets a torch object or torch exception cross the wire: the local side has no
    torch, and attempt 1 of BTC-D2-001 lost its results to exactly that (see D2 report)."""
    import traceback
    try:
        payload = _run_impl(rows)
        return {"ok": True, "json": json.dumps(payload["meta"]), "acts_npy": bytes(payload["acts_npy"])}
    except BaseException as exc:  # noqa: BLE001 - converted to plain strings on purpose
        return {"ok": False, "error": f"{type(exc).__module__}.{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()}


def _run_impl(rows: list[dict]) -> dict:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    log = []
    for r in rows:  # oracle isolation + integrity
        assert set(r) == {"world_id", "prompt", "prompt_sha256", "expected_n_tokens"}, "unexpected field"
        assert hashlib.sha256(r["prompt"].encode()).hexdigest() == r["prompt_sha256"]
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    assert tok.convert_tokens_to_ids("A") == TOKEN_A and tok.convert_tokens_to_ids("B") == TOKEN_B
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.bfloat16)
    model = model.to("cuda").eval()
    log.append(f"loaded in {time.time() - t0:.1f}s")
    print(log[-1], flush=True)
    torch.manual_seed(0)

    results, acts = [], []
    with torch.no_grad():
        for rep in (0, 1):
            for r in rows:
                if time.time() - t0 > WALL_GUARD_S:
                    raise RuntimeError("wall-clock guard reached; aborting (partial results discarded)")
                text = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                               tokenize=False, add_generation_prompt=True)
                assert text.endswith(TEMPLATE_TAIL)
                ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
                assert ids.shape[1] == r["expected_n_tokens"], (r["world_id"], ids.shape[1])
                out = model(input_ids=ids, output_hidden_states=True)
                lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
                top = lp.topk(5)
                results.append({"world_id": r["world_id"], "replay": rep, "n_tokens": int(ids.shape[1]),
                                "logp_A": lp[TOKEN_A].item(), "logp_B": lp[TOKEN_B].item(),
                                "top5": [[tok.convert_ids_to_tokens(int(i)), v.item()]
                                         for v, i in zip(top.values, top.indices)]})
                hs = torch.stack([h[0, -1] for h in out.hidden_states])  # (37, 2560)
                acts.append(hs.to(torch.float16).cpu().numpy())
            print(f"replay {rep} done at {time.time() - t0:.1f}s", flush=True)
    buf = io.BytesIO()
    np.save(buf, np.stack(acts))  # (40, 37, 2560): first 20 = replay 0, next 20 = replay 1
    meta = {"results": results, "log": log, "wall_s": float(time.time() - t0),
            "env": {"torch": str(torch.__version__), "transformers": str(__import__("transformers").__version__),
                    "cuda": str(torch.version.cuda), "gpu": str(torch.cuda.get_device_name()),
                    "max_mem_gb": float(torch.cuda.max_memory_allocated() / 1e9), "dtype": "bfloat16",
                    "hidden_states": "tuple of 37: [0]=embeddings, [i]=output of block i, last prompt position"}}
    return {"meta": meta, "acts_npy": buf.getvalue()}


@app.local_entrypoint()
def main() -> None:
    from transformers import AutoTokenizer

    sel = select_worlds()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    rows = [{"world_id": r["world_id"], "prompt": r["prompt"], "prompt_sha256": r["prompt_sha256"],
             "expected_n_tokens": len(tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                                              tokenize=True, add_generation_prompt=True))}
            for r in sel]
    t0 = time.time()
    wire = run.remote(rows)
    out = ROOT / "outputs" / "d2_smoke"
    out.mkdir(parents=True, exist_ok=True)
    if not wire["ok"]:
        (out / "remote_error.txt").write_text(wire["error"] + "\n\n" + wire["traceback"], encoding="utf-8")
        raise SystemExit(f"REMOTE FAILURE (no retry): {wire['error']}")
    res = json.loads(wire["json"])
    res["acts_npy"] = wire["acts_npy"]
    (out / "behavior_raw.jsonl").write_text("".join(json.dumps(x) + "\n" for x in res["results"]), encoding="utf-8")
    (out / "acts_last_pos_fp16.npy").write_bytes(res["acts_npy"])
    manifest = {"experiment_id": "BTC-D2-001", "model_id": MODEL_ID, "revision": MODEL_REVISION,
                "world_ids": [r["world_id"] for r in rows], "gpu_request": "L4 x1", "timeout_s": 900,
                "remote_wall_s": res["wall_s"], "local_wall_s": time.time() - t0, "env": res["env"],
                "log": res["log"], "acts_shape": [40, 37, 2560], "acts_order": "replay0 worlds, then replay1 worlds",
                "acts_sha256": hashlib.sha256(res["acts_npy"]).hexdigest(),
                "cost_note": "read actual cost from Modal Usage & Billing and record it here"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("remote_wall_s", "env", "acts_sha256")}, indent=1))
    print("Next (local, CPU): python src/score_d2.py outputs/d2_smoke")
