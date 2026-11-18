"""One short paid GPU smoke test. Requires Modal login and payment method."""

import subprocess
import modal

app = modal.App("before-choice-gpu-smoke")


@app.function(gpu="L4", timeout=180)
def inspect_gpu() -> str:
    return subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip()


@app.local_entrypoint()
def main() -> None:
    print("Modal GPU:", inspect_gpu.remote())
