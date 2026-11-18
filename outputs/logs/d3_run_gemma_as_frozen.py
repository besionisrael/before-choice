"""D3 activation capture for Gemma-4-31B-it — PREPARED, NOT RUN. BILLABLE; requires its own owner approval.

    modal run src/d3_run_gemma.py --pool PROBE-DEV          (only pool supported until the H1 freeze is approved)

* oracle-free payload from the PROBE-DEV model-input file (hash-checked) + the sorted list of labels parsed from
  the prompts (for frozen candidate embeddings); same template / token-count / digest assertions as V2
* writes shards to an EPHEMERAL Modal Volume (auto-deleted when the local context exits, also on error),
  downloads them, verifies sha256 + shapes locally, then the Volume disappears
* 1 x A100-80GB, cpu=(4,4), memory=(16,96) GiB, retries=0, max_containers=1, timeout 1500 s, guard 1420 s
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import modal

from v2_common import CPU, GPU, MEMORY_MIB, MODELS, render

TIMEOUT_S, GUARD_S, SHARD = 1500, 1420, 400
N_REPLAY = 20
POOLS = {"PROBE-DEV": {"inputs": "data/d5/probe_dev/model_inputs.jsonl", "manifest": "data/d5/probe_dev/manifest.json", "n": 4800}}
LABEL_RE = re.compile(r"(?<= )([A-Z0-9]{4})(?=[.,:])")

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.6.0", "transformers==5.23.0", "huggingface_hub==2.4.0", "tokenizers==0.23.3",
                      "safetensors==0.8.0", "accelerate==1.10.1", "numpy==2.1.3", "jinja2==3.1.6")
         .add_local_python_source("v2_common", "d3_capture_core"))
app = modal.App("btc-d3-gemma4")


@app.function(image=image, gpu=GPU, cpu=CPU, memory=MEMORY_MIB, retries=0, max_containers=1, timeout=TIMEOUT_S)
def capture(rows: list[dict], labels: list[str], digest: str) -> dict:
    import traceback
    try:
        return {"ok": True, "json": json.dumps(_capture(rows, labels, digest))}
    except BaseException as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__module__}.{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _capture(rows, labels, digest):
    import torch
    from d3_capture_core import run_capture
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
    m = MODELS["gemma4_31b"]
    tok = AutoTokenizer.from_pretrained(m["repo"], revision=m["revision"])
    model = Gemma4ForConditionalGeneration.from_pretrained(m["repo"], revision=m["revision"],
                                                           dtype=torch.bfloat16, device_map="cuda")
    return run_capture(rows, labels, digest, Path("/vol") / "shards", "cuda", model, tok, GUARD_S, SHARD, N_REPLAY)


@app.local_entrypoint()
def main(pool: str = "PROBE-DEV") -> None:
    from v2_local import local_encoder
    from d3_capture_core import verify_shards
    root = Path(__file__).resolve().parents[1]
    assert pool in POOLS, "only PROBE-DEV is supported before the H1 freeze is approved"
    cfg = POOLS[pool]
    man = json.loads((root / cfg["manifest"]).read_text(encoding="utf-8"))
    p = root / cfg["inputs"]
    assert hashlib.sha256(p.read_bytes()).hexdigest() == man["sha256"]["model_inputs"]
    rows_in = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    assert len(rows_in) == cfg["n"]
    count = local_encoder("gemma4_31b")
    rows = [{"world_id": r["world_id"], "prompt": r["prompt"], "prompt_sha256": r["prompt_sha256"],
             "expected_n_tokens": count(r["prompt"])} for r in rows_in]
    labels = sorted({m.group(1) for r in rows for m in LABEL_RE.finditer(r["prompt"])})
    digest = hashlib.sha256("\n".join(r["prompt_sha256"] for r in rows).encode()).hexdigest()
    out = root / "outputs" / "d3" / pool.lower().replace("-", "_")
    assert not (out / "MANIFEST.json").exists(), "already captured: no second run without approval"
    out.mkdir(parents=True, exist_ok=True)
    with modal.Volume.ephemeral() as vol:  # deleted automatically when this block exits
        wire = capture.with_options(volumes={"/vol": vol}).remote(rows, labels, digest)
        if not wire["ok"]:
            (out / "remote_error.txt").write_text(wire["error"] + "\n\n" + wire["traceback"], encoding="utf-8")
            raise SystemExit(f"REMOTE FAILURE (no retry): {wire['error']}")
        for entry in vol.listdir("/shards"):
            name = entry.path.split("/")[-1]
            with open(out / name, "wb") as f:
                for chunk in vol.read_file(f"/shards/{name}"):
                    f.write(chunk)
    print(json.dumps(verify_shards(out), indent=1))
