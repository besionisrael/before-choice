"""Local (CPU) side of BTC-V2 runs: builds the oracle-free payload, calls the Modal function once,
saves raw outputs + manifest. Imported by src/v2_run_std.py and src/v2_run_gemma.py."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from v2_common import (CPU, GPU, MEMORY_MIB, MODELS, POOLS, RUNS, SEND_FIELDS, STAGES, load_model_inputs,
                       prompt_digest, render, worst_case_cost)

ROOT = Path(__file__).resolve().parents[1]


def local_encoder(model_key: str):
    """Returns prompt -> token count using the SAME rendering the container uses
    (std: transformers 4.57.1 template; gemma4: jinja2 + tokenizers, verified equal to transformers 5.x)."""
    m = MODELS[model_key]
    if m["stack"] == "gemma4":
        from v2_token_audit import encoder
        rend, enc, _, _ = encoder(m["repo"], m["revision"], m["template_kwargs"])
        return lambda p: len(enc(render(rend, model_key, p)))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(m["repo"], revision=m["revision"])
    return lambda p: len(tok(render(tok, model_key, p), add_special_tokens=False).input_ids)


def resolve(run_id: str, model_key: str | None) -> tuple[str, str]:
    assert run_id in RUNS, f"unknown run id {run_id}"
    stage, fixed = RUNS[run_id]
    if stage == "S":
        assert model_key in (None, fixed), "screening model is fixed by run id"
        return stage, fixed
    sel = ROOT / "outputs" / "v2" / "selection.json"
    assert sel.exists(), "C1 requires a recorded selection (outputs/v2/selection.json)"
    chosen = json.loads(sel.read_text(encoding="utf-8"))["selected_model"]
    assert model_key == chosen, f"C1 model must be the recorded selection ({chosen})"
    return stage, chosen


def drive(run_id: str, model_key: str | None, remote_fn, stack: str) -> None:
    stage, mk = resolve(run_id, model_key)
    assert MODELS[mk]["stack"] == stack, "wrong run file for this model's stack"
    st = STAGES[stage]
    pool = st["pool"]
    out = ROOT / "outputs" / "v2" / run_id
    assert not (out / "behavior_raw.jsonl").exists(), "run already executed: no second attempt without approval"
    rows = load_model_inputs(ROOT, pool)
    count = local_encoder(mk)
    payload = [{"world_id": r["world_id"], "prompt": r["prompt"], "prompt_sha256": r["prompt_sha256"],
                "expected_n_tokens": count(r["prompt"])} for r in rows]
    assert all(set(p) == set(SEND_FIELDS) for p in payload)
    digest = prompt_digest(payload)
    assert digest == POOLS[pool]["prompt_digest"]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    wire = remote_fn.remote(mk, payload, digest, st["guard_s"])
    t_recv = time.time()
    if not wire["ok"]:
        (out / "remote_error.txt").write_text(wire["error"] + "\n\n" + wire["traceback"], encoding="utf-8")
        raise SystemExit(f"REMOTE FAILURE (no retry): {wire['error']}")
    res = json.loads(wire["json"])
    (out / "behavior_raw.jsonl").write_text("".join(json.dumps(x) + "\n" for x in res["results"]), encoding="utf-8")
    tm = res["timing"]
    tm.update({"local_call_to_remote_start_s": tm["remote_start_epoch"] - t0,
               "remote_end_to_local_receive_s": t_recv - tm["remote_end_epoch"], "local_wall_s": time.time() - t0})
    man = {"run_id": run_id, "stage": stage, "pool": pool, "model_key": mk, "model": MODELS[mk],
           "pool_sha256": POOLS[pool]["sha256"], "inputs_sha256": POOLS[pool]["inputs_sha256"], "prompt_digest": digest,
           "gpu": GPU, "cpu": CPU, "memory_mib": MEMORY_MIB, "timeout_s": st["timeout_s"], "guard_s": st["guard_s"],
           "objective_usd": st["objective_usd"], "worst_case_usd": round(worst_case_cost(stage), 4),
           "timing": tm, "env": res["env"],
           "behavior_sha256": hashlib.sha256((out / "behavior_raw.jsonl").read_bytes()).hexdigest(),
           "cost_note": "record actual cost from `modal billing report --show-resources` immediately after the run"}
    (out / "manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: man[k] for k in ("run_id", "model_key", "timing", "env")}, indent=1))
    print(f"Next (local, CPU): python src/score_v2.py {run_id}")
