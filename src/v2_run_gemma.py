"""BTC-V2 Modal runs for Gemma-4-31B-it on a SEPARATE stack (transformers 5.23.0, torch 2.6.0).
BILLABLE. Each run requires its own explicit owner authorisation (F-4 approved in principle only).

    modal run src/v2_run_gemma.py --run-id BTC-V2-S3
    modal run src/v2_run_gemma.py --run-id BTC-V2-C1 --model gemma4_31b   (only if selected)

Same limits as src/v2_run_std.py. Pins mirror the CPU-verified env (outputs/v2_audit/gemma4_env_freeze.txt).
"""

from __future__ import annotations

import json
import modal

from v2_common import CPU, GPU, MEMORY_MIB, STAGES

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.6.0", "transformers==5.23.0", "huggingface_hub==2.4.0", "tokenizers==0.23.3",
                      "safetensors==0.8.0", "accelerate==1.10.1", "numpy==2.1.3", "jinja2==3.1.6")
         .add_local_python_source("v2_common", "v2_remote"))
app = modal.App("btc-v2-gemma4")
_KW = dict(image=image, gpu=GPU, cpu=CPU, memory=MEMORY_MIB, retries=0, max_containers=1)


def _wrap(model_key, rows, digest, guard_s):
    import traceback
    try:
        from v2_remote import infer
        return {"ok": True, "json": json.dumps(infer(model_key, rows, digest, guard_s))}
    except BaseException as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__module__}.{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()}


@app.function(timeout=STAGES["S"]["timeout_s"], **_KW)
def screen(model_key: str, rows: list[dict], digest: str, guard_s: int) -> dict:
    assert guard_s == STAGES["S"]["guard_s"] and model_key == "gemma4_31b"
    return _wrap(model_key, rows, digest, guard_s)


@app.function(timeout=STAGES["C"]["timeout_s"], **_KW)
def competence(model_key: str, rows: list[dict], digest: str, guard_s: int) -> dict:
    assert guard_s == STAGES["C"]["guard_s"] and model_key == "gemma4_31b"
    return _wrap(model_key, rows, digest, guard_s)


@app.local_entrypoint()
def main(run_id: str, model: str = "") -> None:
    from v2_common import RUNS
    from v2_local import drive
    fn = screen if RUNS[run_id][0] == "S" else competence
    drive(run_id, model or None, fn, stack="gemma4")
