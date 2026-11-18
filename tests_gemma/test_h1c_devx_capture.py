"""BTC-H1c DEV-EXTENSION capture (13 slots) on a tiny random Gemma4 (real tokenizer/template, real H1c DEV prompts): 30 slots, fidelity, and
BITWISE identity of the binary candidates' arrival keys between the members of late pairs (the G2 gate) on a real
forward pass. No weights, no GPU."""

import hashlib
import json
import pathlib
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from transformers import AutoConfig, AutoTokenizer, Gemma4ForConditionalGeneration  # noqa: E402

import h1c_positions as HP  # noqa: E402
import h1c_token_checks as TC  # noqa: E402
from d3x_capture_core import capture_world_multi, run_capture_multi, verify_xshards  # noqa: E402
from v2_common import MODELS, render  # noqa: E402
from v2_cpu_checks import tiny_config  # noqa: E402

M = MODELS["gemma4_31b"]
TOK = AutoTokenizer.from_pretrained(M["repo"], revision=M["revision"])
torch.manual_seed(0)
MODEL = Gemma4ForConditionalGeneration._from_config(tiny_config(AutoConfig.from_pretrained(M["repo"], revision=M["revision"])), dtype=torch.bfloat16).eval()
NL = MODEL.config.get_text_config().num_hidden_layers
LAYERS = (0, 1, NL, NL + 1)
W = {r["world_id"]: r for r in map(json.loads, (ROOT / "data" / "h1c" / "devx" / "worlds.jsonl").read_text(encoding="utf-8").splitlines())}
GO = TC.gemma()


def payload(w):
    text, ids, off = GO(w["prompt"])
    p = HP.positions_for(w["prompt"], text, ids, off)
    return {"world_id": w["world_id"], "prompt": w["prompt"], "prompt_sha256": w["prompt_sha256"], "expected_n_tokens": p["n_tokens"],
            "tok": p["tok"][:13], "ids_sha256": p["ids_sha256"]}


class H1cDevxCapture(unittest.TestCase):
    def test_binary_keys_bitwise_identical_in_late_pairs(self):
        fams = sorted({w["family_id"] for w in W.values()})[:4]
        for f in fams:
            a, b = W[f"{f}-base_L"], W[f"{f}-dswap_L"]
            opt = a["oracle"]["optimal_action"]
            bs = HP.binary_slots(a["prompt"], opt)
            ra = capture_world_multi(MODEL, TOK, render(TOK, "gemma4_31b", a["prompt"]), payload(a)["tok"], LAYERS, M["A_id"], M["B_id"], "cpu", a["world_id"], payload(a)["ids_sha256"])
            rb = capture_world_multi(MODEL, TOK, render(TOK, "gemma4_31b", b["prompt"]), payload(b)["tok"], LAYERS, M["A_id"], M["B_id"], "cpu", b["world_id"], payload(b)["ids_sha256"])
            for s in bs:
                self.assertTrue(np.array_equal(ra["acts_u16"][s], rb["acts_u16"][s]), f"{f} slot {s}")
            self.assertFalse(np.array_equal(ra["acts_u16"][0], rb["acts_u16"][0]))   # the ANS query differs

    def test_run_capture_30_slots(self):
        rows = [payload(W[w]) for w in sorted(W)[:5]]
        dig = hashlib.sha256("\n".join(r["prompt_sha256"] for r in rows).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as t:
            man = run_capture_multi(rows, LAYERS, dig, pathlib.Path(t), "cpu", MODEL, TOK, 600, 2, 2)
            self.assertEqual(man["P"], 13)
            self.assertEqual(verify_xshards(pathlib.Path(t), 13, len(LAYERS)), {"hashes_and_shapes_ok": True, "n_worlds": 5})


if __name__ == "__main__":
    unittest.main()
