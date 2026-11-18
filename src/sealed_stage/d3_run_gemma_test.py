"""BTC-D3-TEST — activation capture of LOCKED-TEST for Gemma-4-31B-it. PREPARED, NOT RUN.
BILLABLE: requires the owner's explicit approval of BTC-D3-TEST AND an owner unlock file for stage D3-TEST-CAPTURE.

    modal run src/sealed_stage/d3_run_gemma_test.py

Identical to the approved D3-DEV-R1 capture (src/d3_run_gemma.py) except for the input source and the sealing:
  * capture core: the SAME frozen module `d3_capture_core` (fidelity FIDELITY_MAX_ABS = 0.25 over the full vocabulary,
    A/B decision rule with original-tie / near-tie tolerance 2d + tau, tau = 0.125, non-finite rejection, 20 replays,
    candidate embeddings) — its sha256 is checked against the freeze manifest before anything is read;
  * the SAME image, model revision, resources and limits (1 x A100-80GB, cpu=(4,4), memory=(16,96) GiB, retries=0,
    max_containers=1, timeout 1500 s, guard 1420 s, ephemeral Volume, commit wait 300 s, scaledown_window=2);
  * INPUT: only `model_inputs.jsonl` (oracle-free, model-visible prompts), read through
    sealed_access.staged_read(stage="D3-TEST-CAPTURE") -> owner unlock required, sha256 re-checked, access logged.
    the oracle world file is NOT readable in this stage (refused by sealed_access) and is never named here;
  * OUTPUT: shards verified (sha256 + shapes, 4,800 worlds, world ids == input ids) BEFORE the Volume is deleted, then
    the activation directory is SEALED at once (seal LOCKED-TEST-D3-ACT, read-only). Only an integrity summary without
    any per-world content (counts, fidelity summary, env, timing) is written outside the seal.
No analysis, no label, no oracle, no probe is touched by this script.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

# Fix BTC-D3-TEST attempt 1 (2026-11-14): inside the Modal container this file is mounted as /root/<name>.py, so
# `parents[2]` does not exist (IndexError at import -> startup crash loop). ROOT is resolved only on the local machine.
_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 and (_HERE.parents[2] / "src" / "sealed.py").is_file() else None
if ROOT is not None:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "src" / "sealed_stage"))

import modal  # noqa: E402

from v2_common import CPU, GPU, MEMORY_MIB, MODELS  # noqa: E402

RUN_ID = "BTC-D3-TEST"
STAGE = "D3-TEST-CAPTURE"
TIMEOUT_S, GUARD_S, SHARD = 1500, 1420, 400
COMMIT_WAIT_S = 300
N_REPLAY = 20
N_WORLDS = 4800
FROZEN_INPUTS_SHA = "8fbc920d44295a5d54749c7bf9913fa27c882b78c049cd38880895dbbd43bd4f"
FROZEN_CODE = ("src/d3_capture_core.py", "src/v2_common.py", "src/v2_local.py", "src/sealed.py")
LABEL_RE = re.compile(r"(?<= )([A-Z0-9]{4})(?=[.,:])")

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.6.0", "transformers==5.23.0", "huggingface_hub==2.4.0", "tokenizers==0.23.3",
                      "safetensors==0.8.0", "accelerate==1.10.1", "numpy==2.1.3", "jinja2==3.1.6")
         .add_local_python_source("v2_common", "d3_capture_core"))
app = modal.App("btc-d3-test-gemma4")


@app.function(image=image, gpu=GPU, cpu=CPU, memory=MEMORY_MIB, retries=0, max_containers=1, timeout=TIMEOUT_S,
              scaledown_window=2)  # stop (and final-commit the Volume) right after returning
def capture(rows: list[dict], labels: list[str], digest: str) -> dict:
    import traceback
    try:
        return {"ok": True, "json": json.dumps(_capture(rows, labels, digest))}
    except BaseException as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__module__}.{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _capture(rows, labels, digest):
    import torch
    import transformers
    from d3_capture_core import run_capture
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
    m = MODELS["gemma4_31b"]
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(m["repo"], revision=m["revision"])
    model = Gemma4ForConditionalGeneration.from_pretrained(m["repo"], revision=m["revision"],
                                                           dtype=torch.bfloat16, device_map="cuda")
    torch.cuda.synchronize()
    t_load = time.time() - t0
    print(f"model ready at {t_load:.1f}s", flush=True)
    man = run_capture(rows, labels, digest, Path("/vol") / "shards", "cuda", model, tok, GUARD_S - t_load, SHARD, N_REPLAY)
    man["env"] = {"torch": str(torch.__version__), "transformers": str(transformers.__version__),
                  "cuda": str(torch.version.cuda), "gpu": str(torch.cuda.get_device_name()),
                  "max_mem_gb": float(torch.cuda.max_memory_allocated() / 1e9)}
    man["timing"] = {"load_s": t_load, "capture_s": man["wall_s"], "remote_total_s": time.time() - t0}
    return man


def preflight() -> dict:
    """Checks that need NO sealed content: frozen code, seal integrity, expected input hash, no prior capture,
    no prior analytical access."""
    import sealed
    import sealed_access as SA
    frz = json.loads((ROOT / "reports" / "BTC-H1_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
    seal_man = json.loads((SA.POOL_DIR / sealed.MANIFEST).read_text(encoding="utf-8"))
    chk = {"frozen_code": {p: sha(p) == frz["code_sha256"][p] for p in FROZEN_CODE},
           "pool_seal_intact": all(sealed.hash_only(SA.POOL_DIR).values()),
           "inputs_hash_is_frozen": seal_man["files"]["model_inputs.jsonl"] == FROZEN_INPUTS_SHA,
           # attempt 1 (startup crash, no activations) left an EMPTY, unsealed directory; it is kept as a trace
           "no_prior_capture": (not SA.ACT_DIR.exists()) or (SA.ACT_DIR.is_dir() and not any(SA.ACT_DIR.iterdir())),
           "no_analytical_access_ever": SA.reads_outside_stage(SA.POOL_DIR, STAGE) == []}
    ok = all(chk["frozen_code"].values()) and all(v for k, v in chk.items() if k != "frozen_code")
    if not ok:
        raise SystemExit(f"PREFLIGHT FAILED (nothing read, nothing launched): {chk}")
    return chk


@app.local_entrypoint()
def main() -> None:
    import sealed
    import sealed_access as SA
    from d3_capture_core import verify_shards
    from v2_local import local_encoder
    assert ROOT is not None, "local entrypoint must run from the project checkout"
    chk = preflight()
    raw = SA.staged_read(STAGE, SA.POOL_DIR, "model_inputs.jsonl", f"{RUN_ID} forward passes (prompts only)")
    assert hashlib.sha256(raw).hexdigest() == FROZEN_INPUTS_SHA
    rows_in = [json.loads(x) for x in raw.decode("utf-8").splitlines()]
    assert len(rows_in) == N_WORLDS and len({r["world_id"] for r in rows_in}) == N_WORLDS
    # same oracle-free schema as the PROBE-DEV model-input file; only world_id / prompt / prompt_sha256 leave this machine
    assert all(set(r) == {"family_id", "pool", "prompt", "prompt_sha256", "variant", "world_id"} for r in rows_in), "unexpected fields"
    assert all(hashlib.sha256(r["prompt"].encode()).hexdigest() == r["prompt_sha256"] for r in rows_in)
    count = local_encoder("gemma4_31b")
    rows = [{"world_id": r["world_id"], "prompt": r["prompt"], "prompt_sha256": r["prompt_sha256"],
             "expected_n_tokens": count(r["prompt"])} for r in rows_in]
    labels = sorted({m.group(1) for r in rows for m in LABEL_RE.finditer(r["prompt"])})
    digest = hashlib.sha256("\n".join(r["prompt_sha256"] for r in rows).encode()).hexdigest()
    out = SA.ACT_DIR
    out.mkdir(parents=True, exist_ok=True)   # preflight guarantees: absent or empty
    summary_path = ROOT / "outputs" / "d3" / "d3_test_capture_summary.json"
    t_call = time.time()
    with modal.Volume.ephemeral() as vol:  # deleted automatically when this block exits (also on error)
        wire = capture.with_options(volumes={"/vol": vol}).remote(rows, labels, digest)
        t_ret = time.time()
        if not wire["ok"]:
            (ROOT / "outputs" / "d3" / "d3_test_remote_error.txt").write_text(wire["error"] + "\n\n" + wire["traceback"], encoding="utf-8")
            raise SystemExit(f"REMOTE FAILURE (no retry): {wire['error']}")
        rman = json.loads(wire["json"])
        expected = sorted({fn for sh in rman["shards"] for fn in sh["files"]}
                          | {"candidate_embeddings.npy", "candidate_labels.json", "MANIFEST.json"})
        deadline = time.time() + COMMIT_WAIT_S
        while True:
            present = {e.path.split("/")[-1] for e in vol.listdir("/shards")}
            if set(expected) <= present or time.time() > deadline:
                break
            time.sleep(5)
        missing = sorted(set(expected) - present)
        if missing:
            raise SystemExit(f"TRANSFER FAILURE (no retry): files not visible after {COMMIT_WAIT_S}s: {missing[:5]}")
        t_vis = time.time()
        for name in expected:
            with open(out / name, "wb") as f:
                for chunk in vol.read_file(f"/shards/{name}"):
                    f.write(chunk)
        t_dl = time.time()
        check = verify_shards(out)  # hashes + shapes verified BEFORE the Volume is deleted
        ids = [json.loads(x)["world_id"] for sh in rman["shards"]
               for x in (out / f"{sh['name']}_index.jsonl").read_text(encoding="utf-8").splitlines()]
        complete = sorted(ids) == sorted(r["world_id"] for r in rows)
        if not check["hashes_and_shapes_ok"] or check["n_worlds"] != N_WORLDS or not complete:
            raise SystemExit(f"INTEGRITY FAILURE after download (no retry): {check} complete={complete}")
        size = sum((out / n).stat().st_size for n in expected)
        run = {"run_id": RUN_ID, "pool": "LOCKED-TEST", "stage": STAGE, "inputs_sha256": FROZEN_INPUTS_SHA, "digest": digest,
               "n_labels": len(labels), "preflight": chk,
               "remote_manifest_sha256": hashlib.sha256((out / "MANIFEST.json").read_bytes()).hexdigest(),
               "verify": check, "world_ids_complete": complete, "bytes_downloaded": size, "env": rman["env"],
               "fidelity": rman.get("fidelity_summary"),
               "timing": dict(rman["timing"], call_to_return_s=t_ret - t_call, return_to_visible_s=t_vis - t_ret,
                              download_s=t_dl - t_vis, local_wall_s=time.time() - t_call),
               "resources": {"gpu": GPU, "cpu": CPU, "memory_mib": MEMORY_MIB, "timeout_s": TIMEOUT_S, "guard_s": GUARD_S},
               "cost_note": "record actual cost from `modal billing report --show-resources` immediately after the run"}
        (out / "d3_run_manifest.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    seal = sealed.seal_dir(out, SA.ACT_SEAL)  # activations sealed immediately; read only via stage H1-TEST-EVAL
    summary = {k: run[k] for k in ("run_id", "stage", "inputs_sha256", "n_labels", "verify", "world_ids_complete",
                                   "bytes_downloaded", "env", "fidelity", "timing", "resources")}
    summary["activation_seal"] = {"seal_id": seal["seal_id"], "sealed_utc": seal["sealed_utc"],
                                  "SEAL_MANIFEST_sha256": hashlib.sha256((out / sealed.MANIFEST).read_bytes()).hexdigest(),
                                  "n_files": len(seal["files"])}
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
