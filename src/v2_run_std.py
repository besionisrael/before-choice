"""BTC-V2 Modal runs for the transformers-4.57.1 stack: Qwen3-32B (S1), Olmo-3.1-32B (S2), and C1 if
one of them is selected. BILLABLE. Each run requires its own explicit owner authorisation.

    modal run src/v2_run_std.py --run-id BTC-V2-S1
    modal run src/v2_run_std.py --run-id BTC-V2-S2
    modal run src/v2_run_std.py --run-id BTC-V2-C1 --model <selected>   (only after recorded selection)

Limits (BTC-V2-PROTOCOL_FREEZE §7): 1 x A100-80GB, cpu=(4,4), memory=(16 GiB, 96 GiB), retries=0,
max_containers=1, timeout 780 s (S) / 1240 s (C), internal guard 720 s / 1160 s, no Volume.
"""

from __future__ import annotations

import json
import modal

from v2_common import CPU, GPU, MEMORY_MIB, STAGES

image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "transformers==4.57.1", "huggingface_hub==0.36.2", "accelerate==1.10.1",
                      "numpy==2.1.3", "jinja2==3.1.6")
         .add_local_python_source("v2_common", "v2_remote"))
app = modal.App("btc-v2-std")
_KW = dict(image=image, gpu=GPU, cpu=CPU, memory=MEMORY_MIB, retries=0, max_containers=1)


def _wrap(model_key, rows, digest, guard_s):
    import traceback
    try:
        from v2_remote import infer
        return {"ok": True, "json": json.dumps(infer(model_key, rows, digest, guard_s))}
    except BaseException as exc:  # noqa: BLE001 - plain strings only cross the wire
        return {"ok": False, "error": f"{type(exc).__module__}.{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()}


@app.function(timeout=STAGES["S"]["timeout_s"], **_KW)
def screen(model_key: str, rows: list[dict], digest: str, guard_s: int) -> dict:
    assert guard_s == STAGES["S"]["guard_s"]
    return _wrap(model_key, rows, digest, guard_s)


@app.function(timeout=STAGES["C"]["timeout_s"], **_KW)
def competence(model_key: str, rows: list[dict], digest: str, guard_s: int) -> dict:
    assert guard_s == STAGES["C"]["guard_s"]
    return _wrap(model_key, rows, digest, guard_s)


@app.local_entrypoint()
def main(run_id: str, model: str = "") -> None:
    from v2_common import RUNS
    from v2_local import drive
    fn = screen if RUNS[run_id][0] == "S" else competence
    drive(run_id, model or None, fn, stack="std")
