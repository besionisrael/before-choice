"""BTC-H1b X3+X4 — multi-position activation capture on PROBE-DEV (3,480 training-rule worlds). PREPARED, NOT RUN.
BILLABLE: requires the owner's explicit approval of BTC-H1b-X3X4.

    modal run src/h1b_x3x4_run_gemma.py

* input: PROBE-DEV model inputs (hash-checked) restricted to the worlds of outputs/d3x/plan/positions.jsonl (hash-checked),
  with per-world token positions and the sha256 of the expected token ids (the container asserts identical tokenisation)
* capture: d3x_capture_core (frozen D3 fidelity check, non-finite rejection, 20 bitwise replays), 22 position slots x
  9 layers {0, 8, 16, 24, 32, 40, 48, 56, 61}
* same image / model revision / limits as D3-DEV-R1: 1 x A100-80GB, cpu (4,4), memory 16-96 GiB, retries=0,
  max_containers=1, timeout 1,500 s, guard 1,420 s; ephemeral Volume; shard hashes + shapes verified BEFORE the Volume
  is deleted; then a local replication check of the pre-answer vectors against the D3-DEV-R1 capture
* no LOCKED-TEST path, no oracle field in the payload
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[1] if len(_HERE.parents) > 1 and (_HERE.parents[1] / "src" / "sealed.py").is_file() else None
if ROOT is not None:
    sys.path.insert(0, str(ROOT / "src"))

import modal  # noqa: E402

from v2_common import CPU, GPU, MEMORY_MIB, MODELS  # noqa: E402

RUN_ID = "BTC-H1b-X3X4"
TIMEOUT_S, GUARD_S, SHARD, N_REPLAY, COMMIT_WAIT_S = 1500, 1420, 400, 20, 300
LAYERS = (0, 8, 16, 24, 32, 40, 48, 56, 61)
N_WORLDS = 3480
DEV_INPUTS_SHA = "a3c814363726493477b785397c5118035bd35b449cecc046d4fd1cc62ba47fa1"
POSITIONS_SHA = "2663054996281ca566bd48820ad77fe3702ac9c118fa162db6a3b5d0d9d44760"

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.6.0", "transformers==5.23.0", "huggingface_hub==2.4.0", "tokenizers==0.23.3",
                      "safetensors==0.8.0", "accelerate==1.10.1", "numpy==2.1.3", "jinja2==3.1.6")
         .add_local_python_source("v2_common", "d3_capture_core", "d3x_capture_core"))
app = modal.App("btc-h1b-x3x4-gemma4")


@app.function(image=image, gpu=GPU, cpu=CPU, memory=MEMORY_MIB, retries=0, max_containers=1, timeout=TIMEOUT_S,
              scaledown_window=2)
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


def replication_check(out: Path) -> dict:
    """Pre-answer vectors (slot 0) vs the D3-DEV-R1 capture at the same 9 layers (diagnostic; reported)."""
    import numpy as np
    old_dir = ROOT / "outputs" / "d3" / "probe_dev"
    om = json.loads((old_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    pos, base = {}, 0
    for s in om["shards"]:
        for k, x in enumerate((old_dir / f"{s['name']}_index.jsonl").read_text(encoding="utf-8").splitlines()):
            pos[json.loads(x)["world_id"]] = (s["name"], k)
    nm = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    n_equal, n, worst = 0, 0, 0.0
    cache = {}
    for s in nm["shards"]:
        a = np.load(out / f"{s['name']}_acts.npy", mmap_mode="r")
        for k, x in enumerate((out / f"{s['name']}_index.jsonl").read_text(encoding="utf-8").splitlines()):
            wid = json.loads(x)["world_id"]
            sh, j = pos[wid]
            if sh not in cache:
                cache.clear()
                cache[sh] = np.load(old_dir / f"{sh}_acts.npy", mmap_mode="r")
            old = np.asarray(cache[sh][j, list(LAYERS)])
            new = np.asarray(a[k, 0])
            n += 1
            n_equal += int(np.array_equal(old, new))
            f = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
            worst = max(worst, float(np.abs(f(old) - f(new)).max()))
    return {"n_worlds": n, "bitwise_equal": n_equal, "max_abs_diff": worst}


@app.local_entrypoint()
def main() -> None:
    from d3x_capture_core import verify_xshards
    from v2_local import local_encoder
    assert ROOT is not None
    dev = ROOT / "data" / "d5" / "probe_dev"
    p_in, p_pos = dev / "model_inputs.jsonl", ROOT / "outputs" / "d3x" / "plan" / "positions.jsonl"
    assert hashlib.sha256(p_in.read_bytes()).hexdigest() == DEV_INPUTS_SHA
    assert hashlib.sha256(p_pos.read_bytes()).hexdigest() == POSITIONS_SHA
    posd = {r["world_id"]: r for r in map(json.loads, p_pos.read_text(encoding="utf-8").splitlines())}
    rows_in = [r for r in map(json.loads, p_in.read_text(encoding="utf-8").splitlines()) if r["world_id"] in posd]
    assert len(rows_in) == N_WORLDS
    count = local_encoder("gemma4_31b")
    rows = []
    for r in rows_in:
        q = posd[r["world_id"]]
        assert q["prompt_sha256"] == r["prompt_sha256"] and count(r["prompt"]) == q["n_tokens"]
        rows.append({"world_id": r["world_id"], "prompt": r["prompt"], "prompt_sha256": r["prompt_sha256"],
                     "expected_n_tokens": q["n_tokens"], "tok": q["tok"], "ids_sha256": q["ids_sha256"]})
    digest = hashlib.sha256("\n".join(r["prompt_sha256"] for r in rows).encode()).hexdigest()
    out = ROOT / "outputs" / "d3x" / "probe_dev"
    assert not (out / "MANIFEST.json").exists(), "already captured: no second run without approval"
    out.mkdir(parents=True, exist_ok=True)
    t_call = time.time()
    with modal.Volume.ephemeral() as vol:
        wire = capture.with_options(volumes={"/vol": vol}).remote(rows, digest)
        t_ret = time.time()
        if not wire["ok"]:
            (out / "remote_error.txt").write_text(wire["error"] + "\n\n" + wire["traceback"], encoding="utf-8")
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
        check = verify_xshards(out, 22, len(LAYERS))  # BEFORE the Volume is deleted
        if not check["hashes_and_shapes_ok"] or check["n_worlds"] != N_WORLDS:
            raise SystemExit(f"INTEGRITY FAILURE after download (no retry): {check}")
        size = sum((out / n).stat().st_size for n in expected)
        run = {"run_id": RUN_ID, "pool": "PROBE-DEV (training-rule subset)", "inputs_sha256": DEV_INPUTS_SHA, "positions_sha256": POSITIONS_SHA,
               "digest": digest, "verify": check, "bytes_downloaded": size, "env": rman["env"], "fidelity": rman["fidelity_summary"],
               "timing": dict(rman["timing"], call_to_return_s=t_ret - t_call, return_to_visible_s=t_vis - t_ret,
                              download_s=t_dl - t_vis, local_wall_s=time.time() - t_call),
               "resources": {"gpu": GPU, "cpu": CPU, "memory_mib": MEMORY_MIB, "timeout_s": TIMEOUT_S, "guard_s": GUARD_S},
               "cost_note": "record actual cost from `modal billing report --show-resources` immediately after the run"}
        (out / "x_run_manifest.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    run["replication_vs_D3_DEV_R1"] = replication_check(out)
    (out / "x_run_manifest.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: run[k] for k in ("verify", "bytes_downloaded", "timing", "replication_vs_D3_DEV_R1")}, indent=1))
