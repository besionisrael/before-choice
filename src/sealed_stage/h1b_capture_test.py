"""BTC-H1b v1.0 — capture of LOCKED-TEST-H1b (4,000 worlds). PREPARED, NOT RUN. BILLABLE: requires the owner's approval
AND the H1B-CAPTURE unlock.

    modal run src/sealed_stage/h1b_capture_test.py

* input: `model_inputs.jsonl` only, via sealed_access_h1b.staged_read(stage="H1B-CAPTURE"); positions computed locally
  from the prompt + pinned tokenizer (d3x_positions): slot 0 = ANS (`<channel|>`), slots 1-6 = arrival occurrence of the
  6 prompt-derived candidates (canonical order); the container asserts identical tokenisation (sha256 of ids)
* layers {32, 40, 48}; frozen D3 fidelity check, non-finite rejection, 20 bitwise replays (d3x_capture_core)
* 1 x A100-80GB, cpu (4,4), memory 16-96 GiB, retries=0, max_containers=1, timeout 1,500 s, guard 1,420 s; ephemeral
  Volume; shard hashes + shapes verified BEFORE deletion; activations sealed at once (LOCKED-TEST-H1B-ACT)
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 and (_HERE.parents[2] / "src" / "sealed.py").is_file() else None
if ROOT is not None:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "src" / "sealed_stage"))

import modal  # noqa: E402

from v2_common import CPU, GPU, MEMORY_MIB, MODELS  # noqa: E402

RUN_ID, STAGE = "BTC-H1b-CAPTURE", "H1B-CAPTURE"
TIMEOUT_S, GUARD_S, SHARD, N_REPLAY, COMMIT_WAIT_S = 1500, 1420, 500, 20, 300
LAYERS = (32, 40, 48)
N_WORLDS = 4000
SLOTS = (0, 1, 3, 5, 7, 9, 11)   # ANS + LAB_TGT of candidates 0..5 in the d3x_positions 22-slot layout

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.6.0", "transformers==5.23.0", "huggingface_hub==2.4.0", "tokenizers==0.23.3",
                      "safetensors==0.8.0", "accelerate==1.10.1", "numpy==2.1.3", "jinja2==3.1.6")
         .add_local_python_source("v2_common", "d3_capture_core", "d3x_capture_core"))
app = modal.App("btc-h1b-capture-gemma4")


@app.function(image=image, gpu=GPU, cpu=CPU, memory=MEMORY_MIB, retries=0, max_containers=1, timeout=TIMEOUT_S, scaledown_window=2)
def capture(rows: list[dict], digest: str) -> dict:
    import traceback
    try:
        return {"ok": True, "json": json.dumps(_capture(rows, digest))}
    except BaseException as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__module__}.{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _capture(rows, digest):
    import torch
    import transformers
    from d3x_capture_core import run_capture_multi
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
    m = MODELS["gemma4_31b"]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(m["repo"], revision=m["revision"])
    model = Gemma4ForConditionalGeneration.from_pretrained(m["repo"], revision=m["revision"], dtype=torch.bfloat16, device_map="cuda")
    torch.cuda.synchronize()
    t_load = time.time() - t0
    print(f"model ready at {t_load:.1f}s", flush=True)
    man = run_capture_multi(rows, LAYERS, digest, Path("/vol") / "xshards", "cuda", model, tok, GUARD_S - t_load, SHARD, N_REPLAY)
    man["env"] = {"torch": str(torch.__version__), "transformers": str(transformers.__version__), "cuda": str(torch.version.cuda),
                  "gpu": str(torch.cuda.get_device_name()), "max_mem_gb": float(torch.cuda.max_memory_allocated() / 1e9)}
    man["timing"] = {"load_s": t_load, "capture_s": man["wall_s"], "remote_total_s": time.time() - t0}
    return man


def preflight() -> dict:
    import sealed
    import sealed_access_h1b as SA
    frz = json.loads((ROOT / "outputs" / "h1b_freeze" / "V1_MANIFEST.json").read_text(encoding="utf-8"))
    sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
    chk = {"frozen_code": {p: sha(p) == h for p, h in frz["code_sha256"].items()},
           "pool_seal_intact": all(sealed.hash_only(SA.POOL_DIR).values()),
           "no_prior_capture": not SA.ACT_DIR.exists(),
           "no_analytical_access": SA.reads_outside_stage(SA.POOL_DIR, STAGE) == []}
    if not (all(chk["frozen_code"].values()) and all(v for k, v in chk.items() if k != "frozen_code")):
        raise SystemExit(f"PREFLIGHT FAILED (nothing read, nothing launched): {chk}")
    return chk


@app.local_entrypoint()
def main() -> None:
    import sealed
    import sealed_access_h1b as SA
    import d3x_positions as XP
    from d3x_capture_core import verify_xshards
    assert ROOT is not None
    chk = preflight()
    raw = SA.staged_read(STAGE, SA.POOL_DIR, "model_inputs.jsonl", f"{RUN_ID} forward passes (prompts only)")
    rows_in = [json.loads(x) for x in raw.decode("utf-8").splitlines()]
    assert len(rows_in) == N_WORLDS and all(hashlib.sha256(r["prompt"].encode()).hexdigest() == r["prompt_sha256"] for r in rows_in)
    go = XP.gemma_tokenise()
    rows = []
    for r in rows_in:
        text, ids, off = go(r["prompt"])
        p = XP.positions_for(r["prompt"], text, ids, off)
        assert len(p["cands"]) == 6, "depth-4 pool expected"
        rows.append({"world_id": r["world_id"], "prompt": r["prompt"], "prompt_sha256": r["prompt_sha256"],
                     "expected_n_tokens": p["n_tokens"], "tok": [p["tok"][s] for s in SLOTS], "ids_sha256": p["ids_sha256"]})
    digest = hashlib.sha256("\n".join(r["prompt_sha256"] for r in rows).encode()).hexdigest()
    out = SA.ACT_DIR
    out.mkdir(parents=True, exist_ok=False)
    t_call = time.time()
    with modal.Volume.ephemeral() as vol:
        wire = capture.with_options(volumes={"/vol": vol}).remote(rows, digest)
        t_ret = time.time()
        if not wire["ok"]:
            (ROOT / "outputs" / "d3" / "h1b_capture_remote_error.txt").write_text(wire["error"] + "\n\n" + wire["traceback"], encoding="utf-8")
            raise SystemExit(f"REMOTE FAILURE (no retry): {wire['error']}")
        rman = json.loads(wire["json"])
        expected = sorted({fn for sh in rman["shards"] for fn in sh["files"]} | {"MANIFEST.json"})
        deadline = time.time() + COMMIT_WAIT_S
        while True:
            present = {e.path.split("/")[-1] for e in vol.listdir("/xshards")}
            if set(expected) <= present or time.time() > deadline:
                break
            time.sleep(5)
        missing = sorted(set(expected) - present)
        if missing:
            raise SystemExit(f"TRANSFER FAILURE (no retry): files not visible after {COMMIT_WAIT_S}s: {missing[:5]}")
        t_vis = time.time()
        for name in expected:
            with open(out / name, "wb") as f:
                for chunk in vol.read_file(f"/xshards/{name}"):
                    f.write(chunk)
        t_dl = time.time()
        check = verify_xshards(out, len(SLOTS), len(LAYERS))
        ids = [json.loads(x)["world_id"] for sh in rman["shards"] for x in (out / f"{sh['name']}_index.jsonl").read_text(encoding="utf-8").splitlines()]
        complete = sorted(ids) == sorted(r["world_id"] for r in rows)
        if not check["hashes_and_shapes_ok"] or check["n_worlds"] != N_WORLDS or not complete:
            raise SystemExit(f"INTEGRITY FAILURE after download (no retry): {check} complete={complete}")
        run = {"run_id": RUN_ID, "stage": STAGE, "preflight": chk, "digest": digest, "verify": check, "world_ids_complete": complete,
               "fidelity": rman["fidelity_summary"], "env": rman["env"],
               "timing": dict(rman["timing"], call_to_return_s=t_ret - t_call, return_to_visible_s=t_vis - t_ret, download_s=t_dl - t_vis),
               "resources": {"gpu": GPU, "cpu": CPU, "memory_mib": MEMORY_MIB, "timeout_s": TIMEOUT_S, "guard_s": GUARD_S}}
        (out / "run_manifest.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    seal = sealed.seal_dir(out, SA.ACT_SEAL)
    summary = {k: run[k] for k in ("run_id", "verify", "world_ids_complete", "env", "timing")}
    summary["fidelity"] = {k: run["fidelity"][k] for k in ("max_abs_diff", "n_divergences_logged")}
    summary["seal"] = {"seal_id": seal["seal_id"], "SEAL_MANIFEST_sha256": hashlib.sha256((out / sealed.MANIFEST).read_bytes()).hexdigest()}
    (ROOT / "outputs" / "d3" / "h1b_capture_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
