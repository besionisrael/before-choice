"""BTC-H1b X3+X4 capture tests (Gemma env): tiny random Gemma4 (real config structure), REAL tokenizer/template,
REAL PROBE-DEV prompts. No weights, no GPU.  .venv-gemma4\\Scripts\\python -m unittest discover -s tests_gemma"""

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

import d3x_positions as XP  # noqa: E402
from d3_capture_core import capture_world  # noqa: E402
from d3x_capture_core import capture_world_multi, run_capture_multi, verify_xshards  # noqa: E402
from v2_common import MODELS, render  # noqa: E402
from v2_cpu_checks import tiny_config  # noqa: E402

M = MODELS["gemma4_31b"]
TOK = AutoTokenizer.from_pretrained(M["repo"], revision=M["revision"])
torch.manual_seed(0)
MODEL = Gemma4ForConditionalGeneration._from_config(
    tiny_config(AutoConfig.from_pretrained(M["repo"], revision=M["revision"])), dtype=torch.bfloat16).eval()
NL = MODEL.config.get_text_config().num_hidden_layers
LAYERS = (0, 1, NL, NL + 1)          # tiny analogue of {0, ..., 60, 61}: last HF state and raw pre-norm
ROWS = [json.loads(x) for x in (ROOT / "data" / "d5" / "probe_dev" / "model_inputs.jsonl").read_text(encoding="utf-8").splitlines()]
ROWS = [r for r in ROWS if not r["variant"].startswith("decoy")][:10]
GO = XP.gemma_tokenise()


def payload(r):
    text, ids, off = GO(r["prompt"])
    p = XP.positions_for(r["prompt"], text, ids, off)
    return {"world_id": r["world_id"], "prompt": r["prompt"], "prompt_sha256": r["prompt_sha256"],
            "expected_n_tokens": p["n_tokens"], "tok": p["tok"], "ids_sha256": p["ids_sha256"]}


class XCapture(unittest.TestCase):
    def test_tokenisers_agree_and_positions_valid(self):
        for r in ROWS:
            text, ids, off = GO(r["prompt"])
            hf = TOK(render(TOK, "gemma4_31b", r["prompt"]), add_special_tokens=False).input_ids
            self.assertEqual(list(ids), list(hf))                         # container tokenisation == local positions
            p = XP.positions_for(r["prompt"], text, ids, off)
            self.assertEqual(len(p["tok"]), XP.P)
            self.assertEqual(p["tok"][0], len(ids) - 1)
            a = XP.audit_world(r["prompt"], text, ids, off, p, TOK.convert_ids_to_tokens)
            self.assertTrue(a["labels_ok"] and a["line_ends_ok"] and a["ans_is_channel"])

    def test_ans_slot_equals_frozen_capture_and_layout(self):
        r = ROWS[0]
        pl = payload(r)
        text = render(TOK, "gemma4_31b", r["prompt"])
        rec = capture_world_multi(MODEL, TOK, text, pl["tok"], LAYERS, M["A_id"], M["B_id"], "cpu", r["world_id"], pl["ids_sha256"])
        self.assertEqual(rec["acts_u16"].shape[:2], (XP.P, len(LAYERS)))
        ref = capture_world(MODEL, TOK, text, M["A_id"], M["B_id"], "cpu", world_id=r["world_id"])
        self.assertTrue(np.array_equal(rec["acts_u16"][0], ref["acts_u16"][list(LAYERS)]))   # same vectors as the frozen D3 path
        self.assertEqual(rec["meta"]["fidelity"], ref["meta"]["fidelity"])
        masked = [k for k, m in enumerate(rec["meta"]["mask"]) if not m]
        self.assertTrue(all((rec["acts_u16"][k] == 0).all() for k in masked))

    def test_tokenisation_mismatch_refused(self):
        r = ROWS[1]
        pl = payload(r)
        with self.assertRaises(AssertionError):
            capture_world_multi(MODEL, TOK, render(TOK, "gemma4_31b", r["prompt"]), pl["tok"], LAYERS, M["A_id"], M["B_id"],
                                "cpu", r["world_id"], "0" * 64)

    def test_run_capture_multi_shards_replay_and_verify(self):
        rows = [payload(r) for r in ROWS[:5]]
        dig = hashlib.sha256("\n".join(r["prompt_sha256"] for r in rows).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as t:
            man = run_capture_multi(rows, LAYERS, dig, pathlib.Path(t), "cpu", MODEL, TOK, 600, 2, 2)
            self.assertEqual(man["n_worlds"], 5)
            self.assertEqual(man["replay_bitwise_identical"], 2)
            self.assertEqual(verify_xshards(pathlib.Path(t), XP.P, len(LAYERS)), {"hashes_and_shapes_ok": True, "n_worlds": 5})


if __name__ == "__main__":
    unittest.main()
