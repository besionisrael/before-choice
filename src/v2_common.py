"""Shared, framework-light definitions for BTC-V2 runs (imported locally AND inside Modal containers).

Everything here is frozen by reports/BTC-V2-PROTOCOL_FREEZE.md (v1.0, owner-validated F-1..F-5).
No torch import at module level (the local environment has no torch).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# ---------------------------------------------------------------- models (§1)
MODELS = {
    "qwen3_32b": {
        "repo": "Qwen/Qwen3-32B", "revision": "9216db5781bf21249d130ec9da846c4624c16137",
        "stack": "std", "template_kwargs": {"enable_thinking": False},
        "tail": "<|im_start|>assistant\n<think>\n\n</think>\n\n", "A_id": 32, "B_id": 33,
        "n_hidden_states": 65, "est_cost_rank": 1,
    },
    "olmo31_32b": {
        "repo": "allenai/Olmo-3.1-32B-Instruct", "revision": "ac0587e4a7744a551c059d8cd17ba220bc940dae",
        "stack": "std", "template_kwargs": {},
        "tail": "<|im_start|>assistant\n", "A_id": 32, "B_id": 33,
        "n_hidden_states": 65, "est_cost_rank": 2,
    },
    "gemma4_31b": {
        "repo": "google/gemma-4-31B-it", "revision": "842da3794eaa0b77d5f08bae87a17459d91ff475",
        "stack": "gemma4", "template_kwargs": {"enable_thinking": False},
        "tail": "<|turn>model\n<|channel>thought\n<channel|>", "A_id": 236776, "B_id": 236799,
        "n_hidden_states": 61, "est_cost_rank": 3,
    },
}

# ---------------------------------------------------------------- data (§2)
POOLS = {
    "SCREEN": {"file": "screen", "n": 240,
               "sha256": "cd3499d67f3be6fe03ee936f0c607134b5e14ae7b201ed26441b5c995eda34cd",
               "inputs_sha256": "42cc0517246c101e676d517066001e359fe83d301149970b8e940f6602b5dc20",
               "prompt_digest": "6e52d9dc17bf1fd7934ab7a686d6050d2819219e0829e1e66bb3e6257bb4de1b"},
    "COMPETENCE": {"file": "competence", "n": 1320,
                   "sha256": "b1df46cd67e87e938a9ace023f4c46dccac154968b160467113e9cecf4e245d6",
                   "inputs_sha256": "dc6c7115bcb2b15e21af952ca623562ded6708b5212d9f0df75a20d836fea0da",
                   "prompt_digest": "cf3c6e264b1fbc76a12051935b4e0d00e2fda7fe841e7ee77c25bfbe3f74dc14"},
}
SEND_FIELDS = ("world_id", "prompt", "prompt_sha256", "expected_n_tokens")

# ---------------------------------------------------------------- runs (§9) and budget (§7)
STAGES = {
    "S": {"pool": "SCREEN", "timeout_s": 780, "guard_s": 720, "objective_usd": 1.00},
    "C": {"pool": "COMPETENCE", "timeout_s": 1240, "guard_s": 1160, "objective_usd": 1.50},
}
RUNS = {"BTC-V2-S1": ("S", "qwen3_32b"), "BTC-V2-S2": ("S", "olmo31_32b"), "BTC-V2-S3": ("S", "gemma4_31b"),
        "BTC-V2-C1": ("C", None)}  # C1 model = recorded selection (outputs/v2/selection.json)
GPU = "A100-80GB"
CPU = (4.0, 4.0)
MEMORY_MIB = (16384, 98304)
N_REPLAY = 20
RATES = {"A100-80GB": 0.000694, "cpu_core": 0.0000131, "mem_gib": 0.00000222}  # $/s, modal.com/pricing 2026-11-13
OVERHEAD_S, EGRESS_USD, IMAGE_USD = 60, 0.02, 0.02
PROGRAMME_OBJECTIVE_USD = 4.50


def worst_case_rate() -> float:
    return RATES[GPU] + CPU[1] * RATES["cpu_core"] + (MEMORY_MIB[1] / 1024) * RATES["mem_gib"]


def worst_case_cost(stage: str) -> float:
    return (STAGES[stage]["timeout_s"] + OVERHEAD_S) * worst_case_rate() + EGRESS_USD + IMAGE_USD


# ---------------------------------------------------------------- helpers
def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def prompt_digest(rows) -> str:
    """Order-sensitive digest of the prompt hashes actually sent (checked inside the container)."""
    return hashlib.sha256("\n".join(r["prompt_sha256"] for r in rows).encode()).hexdigest()


def load_model_inputs(root: Path, pool: str) -> list[dict]:
    """Reads ONLY the oracle-free model-input file of a frozen pool and verifies both pool hashes."""
    meta = POOLS[pool]
    d = root / "data" / "v2"
    assert sha256_file(d / f"{meta['file']}.jsonl") == meta["sha256"], "pool file hash mismatch"
    p = d / f"{meta['file']}_model_inputs.jsonl"
    assert sha256_file(p) == meta["inputs_sha256"], "model-input file hash mismatch"
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == meta["n"]
    for r in rows:
        assert "oracle" not in r and "terminal_rewards" not in r and "transitions" not in r
        assert r["pool"] == pool
        assert hashlib.sha256(r["prompt"].encode("utf-8")).hexdigest() == r["prompt_sha256"]
    return rows


def render(tok_or_render, model_key: str, prompt: str) -> str:
    """Official chat template, non-thinking; asserts the frozen pre-answer tail."""
    m = MODELS[model_key]
    # NB: tokenizers are themselves callable, so test for apply_chat_template FIRST
    if hasattr(tok_or_render, "apply_chat_template"):
        text = tok_or_render.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                 add_generation_prompt=True, **m["template_kwargs"])
    else:
        text = tok_or_render(prompt)
    assert text.endswith(m["tail"]), f"{model_key}: template tail mismatch: {text[-60:]!r}"
    return text
