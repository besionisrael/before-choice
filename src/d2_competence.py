"""BTC-D2-002 — pre-registered D2 behavioural competence checkpoint (Modal). BILLABLE.
REQUIRES EXPLICIT OWNER APPROVAL. Behaviour only: NO activation capture (that is D3).

Exact command (PowerShell, project root, venv active):
    modal run src/d2_competence.py

Scope (fixed; no CLI knobs):
  * ALL train + val worlds = 352, read ONLY from
      data/splits/{train,val,pathcf_train,pathcf_val}.jsonl
    The locked test files (test.jsonl, pathcf_test.jsonl) are never opened.
  * one forward pass per world (no generation, no gradient, no hidden states),
    + determinism replay of the first 20 worlds of the run order
  * model / revision / dtype / chat template / prompts / token ids identical to BTC-D2-001b
  * 1 x L4, max_containers=1, retries=0, timeout=900 s, internal wall guard 780 s
  * returns JSON only (no torch objects); scoring is local (src/score_d2_competence.py)
Only world_id / prompt / prompt_sha256 / expected token count leave the machine.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import modal

MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
TOKEN_A, TOKEN_B = 32, 33
TEMPLATE_TAIL = "<|im_start|>assistant\n"
WALL_GUARD_S = 780
N_EXPECTED = 352
N_REPLAY = 20
SPLIT_FILES = ("train.jsonl", "val.jsonl", "pathcf_train.jsonl", "pathcf_val.jsonl")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1", "transformers==4.57.1", "huggingface_hub==0.36.2", "numpy==2.1.3")
app = modal.App("btc-d2-competence-002")

ROOT = Path(__file__).resolve().parents[1]


def select_worlds() -> list[dict]:
    rows = []
    for f in SPLIT_FILES:
        rows += [json.loads(x) for x in (ROOT / "data" / "splits" / f).read_text(encoding="utf-8").splitlines()]
    assert len(rows) == N_EXPECTED, len(rows)
    assert {r["split"] for r in rows} == {"train", "val"}, "locked test split must not be used"
    assert len({r["world_id"] for r in rows}) == N_EXPECTED
    rows.sort(key=lambda r: r["world_id"])
    return rows


@app.function(gpu="L4", image=image, timeout=900, retries=0, max_containers=1)
def run(rows: list[dict]) -> dict:
    import traceback
    try:
        return {"ok": True, "json": json.dumps(_run_impl(rows))}
    except BaseException as exc:  # noqa: BLE001 - plain strings only cross the wire
        return {"ok": False, "error": f"{type(exc).__module__}.{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()}


def _run_impl(rows: list[dict]) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tm = {"remote_start_epoch": t0}
    for r in rows:
        assert set(r) == {"world_id", "prompt", "prompt_sha256", "expected_n_tokens"}, "unexpected field"
        assert hashlib.sha256(r["prompt"].encode()).hexdigest() == r["prompt_sha256"]
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    assert tok.convert_tokens_to_ids("A") == TOKEN_A and tok.convert_tokens_to_ids("B") == TOKEN_B
    t1 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.bfloat16)
    tm["weights_download_and_cpu_load_s"] = time.time() - t1
    t1 = time.time()
    model = model.to("cuda").eval()
    torch.cuda.synchronize()
    tm["to_gpu_s"] = time.time() - t1
    print(f"model ready at {time.time() - t0:.1f}s", flush=True)

    plan = [(0, r) for r in rows] + [(1, r) for r in rows[:N_REPLAY]]
    results = []
    t1 = time.time()
    with torch.no_grad():
        for i, (rep, r) in enumerate(plan):
            if time.time() - t0 > WALL_GUARD_S:
                raise RuntimeError(f"wall-clock guard reached after {i} forwards; aborting")
            text = tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                           tokenize=False, add_generation_prompt=True)
            assert text.endswith(TEMPLATE_TAIL)
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            assert ids.shape[1] == r["expected_n_tokens"], (r["world_id"], int(ids.shape[1]))
            logits = model(input_ids=ids).logits[0, -1]
            lp = torch.log_softmax(logits.float(), dim=-1)
            top = lp.topk(5)
            results.append({"world_id": r["world_id"], "replay": rep, "n_tokens": int(ids.shape[1]),
                            "logit_A_bf16": float(logits[TOKEN_A].item()), "logit_B_bf16": float(logits[TOKEN_B].item()),
                            "logp_A": float(lp[TOKEN_A].item()), "logp_B": float(lp[TOKEN_B].item()),
                            "top5": [[tok.convert_ids_to_tokens(int(j)), float(v.item())]
                                     for v, j in zip(top.values, top.indices)]})
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(plan)} forwards at {time.time() - t0:.1f}s", flush=True)
    torch.cuda.synchronize()
    tm["forwards_s"] = time.time() - t1
    tm["n_forwards"] = len(plan)
    tm["remote_end_epoch"] = time.time()
    return {"results": results, "timing": tm, "wall_s": float(time.time() - t0),
            "env": {"torch": str(torch.__version__), "transformers": str(__import__("transformers").__version__),
                    "cuda": str(torch.version.cuda), "gpu": str(torch.cuda.get_device_name()),
                    "max_mem_gb": float(torch.cuda.max_memory_allocated() / 1e9), "dtype": "bfloat16"}}


@app.local_entrypoint()
def main() -> None:
    from transformers import AutoTokenizer

    sel = select_worlds()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    rows = [{"world_id": r["world_id"], "prompt": r["prompt"], "prompt_sha256": r["prompt_sha256"],
             "expected_n_tokens": len(tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                                              tokenize=True, add_generation_prompt=True))}
            for r in sel]
    out = ROOT / "outputs" / "d2_competence"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    wire = run.remote(rows)
    t_recv = time.time()
    if not wire["ok"]:
        (out / "remote_error.txt").write_text(wire["error"] + "\n\n" + wire["traceback"], encoding="utf-8")
        raise SystemExit(f"REMOTE FAILURE (no retry): {wire['error']}")
    res = json.loads(wire["json"])
    (out / "behavior_raw.jsonl").write_text("".join(json.dumps(x) + "\n" for x in res["results"]), encoding="utf-8")
    tm = res["timing"]
    tm.update({"local_call_to_remote_start_s": tm["remote_start_epoch"] - t0,
               "remote_end_to_local_receive_s": t_recv - tm["remote_end_epoch"],
               "local_wall_s": time.time() - t0})
    manifest = {"experiment_id": "BTC-D2-002", "model_id": MODEL_ID, "revision": MODEL_REVISION,
                "split_files": list(SPLIT_FILES), "n_worlds": len(rows), "n_replay": N_REPLAY,
                "gpu_request": "L4 x1", "timeout_s": 900, "timing": tm, "env": res["env"],
                "behavior_sha256": hashlib.sha256((out / "behavior_raw.jsonl").read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("n_worlds", "timing", "env")}, indent=1))
    print("Next (local, CPU): python src/score_d2_competence.py outputs/d2_competence")
