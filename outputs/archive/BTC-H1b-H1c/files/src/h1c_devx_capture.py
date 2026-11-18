"""BTC-H1c DEV EXTENSION capture (3,000 worlds; 13 slots = ANS + 12 arrival keys; 3 layers). PREPARED, NOT RUN.
BILLABLE: requires owner approval.

    modal run src/h1c_devx_capture.py

* input: data/h1c/devx/model_inputs.jsonl (hash-checked) + outputs/h1c_devx/plan/positions.jsonl (hash-checked);
  the container asserts identical tokenisation (sha256 of ids) for every world
* capture core: d3x_capture_core (frozen D3 fidelity check at ANS, non-finite rejection, 20 bitwise replays)
* 1 x A100-80GB, cpu (4,4), memory 16-96 GiB, retries=0, max_containers=1, timeout 950 s, guard 900 s (cost cap)
* slots 0-12 of the 30-slot H1c layout (identical indices to the H1c DEV capture) at layers {32, 40, 48}
* local post-checks BEFORE any analysis (results in outputs/h1c_devx/capture/GATES.json):
    G1 shard hashes + shapes (before the ephemeral Volume is deleted)
    G2 BLOCKING bitwise identity: for every late pair, the two binary candidates' arrival-key vectors are bitwise
       identical between base_L and dswap_L at all 5 layers -> any mismatch = GATE FAILED, analysis refused
    G3 competence (C-4): forced-choice accuracy >= 0.70 overall and >= 0.60 on late-pair worlds
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

RUN_ID = "BTC-H1c-DEVX"
TIMEOUT_S, GUARD_S, SHARD, N_REPLAY, COMMIT_WAIT_S = 950, 900, 500, 20, 300
LAYERS = (32, 40, 48)
N_WORLDS, P_SLOTS = 3000, 13
INPUTS_SHA = "ab4ee61e5ffa37f60555de15d3193752ba690751d51b8085264875ac9ae5d832"
POSITIONS_SHA = "c4c58f0b80ef6ded38002c745ef250f71ecd12e878bcadfe724d6f1d8c252ed7"

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.6.0", "transformers==5.23.0", "huggingface_hub==2.4.0", "tokenizers==0.23.3",
                      "safetensors==0.8.0", "accelerate==1.10.1", "numpy==2.1.3", "jinja2==3.1.6")
         .add_local_python_source("v2_common", "d3_capture_core", "d3x_capture_core"))
app = modal.App("btc-h1c-devx-gemma4")


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


@app.local_entrypoint()
def main() -> None:
    from d3x_capture_core import verify_xshards
    from v2_local import local_encoder
    assert ROOT is not None
    dev, plan = ROOT / "data" / "h1c" / "devx", ROOT / "outputs" / "h1c_devx" / "plan" / "positions.jsonl"
    assert hashlib.sha256((dev / "model_inputs.jsonl").read_bytes()).hexdigest() == INPUTS_SHA
    assert hashlib.sha256(plan.read_bytes()).hexdigest() == POSITIONS_SHA
    posd = {r["world_id"]: r for r in map(json.loads, plan.read_text(encoding="utf-8").splitlines())}
    rows_in = [json.loads(x) for x in (dev / "model_inputs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows_in) == N_WORLDS
    count = local_encoder("gemma4_31b")
    rows = []
    for r in rows_in:
        q = posd[r["world_id"]]
        assert q["prompt_sha256"] == r["prompt_sha256"] and count(r["prompt"]) == q["n_tokens"]
        rows.append({"world_id": r["world_id"], "prompt": r["prompt"], "prompt_sha256": r["prompt_sha256"],
                     "expected_n_tokens": q["n_tokens"], "tok": q["tok"][:P_SLOTS], "ids_sha256": q["ids_sha256"]})
    digest = hashlib.sha256("\n".join(r["prompt_sha256"] for r in rows).encode()).hexdigest()
    out = ROOT / "outputs" / "h1c_devx" / "capture"
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
        check = verify_xshards(out, P_SLOTS, len(LAYERS))   # G1, before the Volume is deleted
        if not check["hashes_and_shapes_ok"] or check["n_worlds"] != N_WORLDS:
            raise SystemExit(f"INTEGRITY FAILURE after download (no retry): {check}")
        run = {"run_id": RUN_ID, "inputs_sha256": INPUTS_SHA, "positions_sha256": POSITIONS_SHA, "digest": digest, "G1": check,
               "fidelity": rman["fidelity_summary"], "env": rman["env"],
               "timing": dict(rman["timing"], call_to_return_s=t_ret - t_call, return_to_visible_s=t_vis - t_ret, download_s=t_dl - t_vis),
               "resources": {"gpu": GPU, "cpu": CPU, "memory_mib": MEMORY_MIB, "timeout_s": TIMEOUT_S, "guard_s": GUARD_S}}
        (out / "run_manifest.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    from h1c_gates import gates
    g = gates(out, dev, plan)
    (out / "GATES.json").write_text(json.dumps(g, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"G1": run["G1"], "G2": {k: g["G2_bitwise_identity"][k] for k in ("late_pairs", "identical", "passed")},
                      "G3": {k: g["G3_competence"][k] for k in ("overall", "late_pairs", "passed")}, "analysis_allowed": g["analysis_allowed"]}, indent=1))
