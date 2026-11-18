"""D3 capture tests — run with the Gemma environment:  .venv-gemma4\\Scripts\\python -m unittest discover -s tests_gemma
Tiny random Gemma4 model (real config structure), real tokenizer/template, real PROBE-DEV prompts. No weights, no GPU."""

import hashlib
import json
import pathlib
import re
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoConfig, AutoTokenizer, Gemma4ForConditionalGeneration  # noqa: E402

from d3_capture_core import (ShardWriter, candidate_embeddings, capture_world, run_capture,  # noqa: E402
                             u16_to_bf16_f32, verify_shards)
from v2_common import MODELS, render  # noqa: E402
from v2_cpu_checks import tiny_config  # noqa: E402

M = MODELS["gemma4_31b"]
TOK = AutoTokenizer.from_pretrained(M["repo"], revision=M["revision"])
torch.manual_seed(0)
MODEL = Gemma4ForConditionalGeneration._from_config(
    tiny_config(AutoConfig.from_pretrained(M["repo"], revision=M["revision"])), dtype=torch.bfloat16).eval()
D = MODEL.config.get_text_config().hidden_size
L = MODEL.config.get_text_config().num_hidden_layers
ROWS_IN = [json.loads(x) for x in (ROOT / "data" / "d5" / "probe_dev" / "model_inputs.jsonl").read_text(encoding="utf-8").splitlines()][:12]


def payload(rows):
    out = []
    for r in rows:
        n = len(TOK(render(TOK, "gemma4_31b", r["prompt"]), add_special_tokens=False).input_ids)
        out.append({"world_id": r["world_id"], "prompt": r["prompt"], "prompt_sha256": r["prompt_sha256"], "expected_n_tokens": n})
    return out


def digest(rows):
    return hashlib.sha256("\n".join(r["prompt_sha256"] for r in rows).encode()).hexdigest()


class Capture(unittest.TestCase):
    def test_world_layout_and_fidelity(self):
        text = render(TOK, "gemma4_31b", ROWS_IN[0]["prompt"])
        rec = capture_world(MODEL, TOK, text, M["A_id"], M["B_id"], "cpu")
        self.assertEqual(rec["acts_u16"].shape, (L + 2, D))           # 61 HF states (L+1) + raw pre-norm
        self.assertEqual(rec["acts_u16"].dtype, np.uint16)
        self.assertEqual(rec["mean_emb"].shape, (D,))
        for k in ("post_norm", "pre_norm"):                          # BOTH reconstructions tested separately
            f = rec["meta"]["fidelity"][k]
            self.assertTrue(f["top1_ok"] and f["ab_decision_same"])
            self.assertLessEqual(f["max_abs_diff"], 0.25)
            self.assertEqual(f["reconstruction"], k)
        # conformity fields come from the ORIGINAL logits: recompute and compare
        with torch.no_grad():
            orig = MODEL(input_ids=TOK(text, return_tensors="pt", add_special_tokens=False).input_ids).logits[0, -1].float()
        lp = torch.log_softmax(orig, -1)
        self.assertEqual(rec["meta"]["top1_id"], int(lp.argmax()))
        self.assertAlmostEqual(rec["meta"]["logp_A"], float(lp[M["A_id"]]), places=6)
        ids = TOK(text, return_tensors="pt", add_special_tokens=False).input_ids
        with torch.no_grad():
            hs = MODEL(input_ids=ids, output_hidden_states=True).hidden_states
        ref = torch.stack([h[0, -1] for h in hs]).float().numpy()
        np.testing.assert_array_equal(u16_to_bf16_f32(rec["acts_u16"][:L + 1]), ref)   # lossless bf16
        np.testing.assert_allclose(rec["mean_emb"], hs[0][0].float().mean(0).numpy(), rtol=1e-6)
        self.assertFalse(np.array_equal(rec["acts_u16"][L], rec["acts_u16"][L + 1]))   # post-norm != pre-norm

    def test_candidate_embeddings_frozen_and_reproducible(self):
        labels = sorted({m.group(1) for r in ROWS_IN for m in re.finditer(r"(?<= )([A-Z0-9]{4})(?=[.,:])", r["prompt"])})
        e1, t1 = candidate_embeddings(MODEL, TOK, labels)
        e2, t2 = candidate_embeddings(MODEL, TOK, labels)
        np.testing.assert_array_equal(e1, e2)
        self.assertEqual(t1, t2)
        W = MODEL.model.language_model.embed_tokens
        with torch.no_grad():
            manual = W(torch.tensor([t1[0]]))[0].float().mean(0).numpy()
        np.testing.assert_allclose(e1[0], manual, rtol=1e-6)

    def test_run_capture_shards_hashes_replay(self):
        rows = payload(ROWS_IN)
        labels = sorted({m.group(1) for r in rows for m in re.finditer(r"(?<= )([A-Z0-9]{4})(?=[.,:])", r["prompt"])})
        with tempfile.TemporaryDirectory() as t:
            out = pathlib.Path(t) / "shards"
            man = run_capture(rows, labels, digest(rows), out, "cpu", MODEL, TOK, guard_s=600, shard=5, n_replay=4)
            fs = man["fidelity_summary"]
            self.assertEqual((fs["tau"], fs["limit"]), (0.125, 0.25))
            self.assertLessEqual(max(fs["max_abs_diff"].values()), 0.25)
            self.assertEqual(fs["n_divergences_logged"], len(fs["divergences"]))
            self.assertEqual([s["n"] for s in man["shards"]], [5, 5, 2])
            self.assertEqual(man["replay_bitwise_identical"], 4)
            v = verify_shards(out)
            self.assertEqual(v, {"hashes_and_shapes_ok": True, "n_worlds": 12})
            # tamper with one shard -> verification fails
            f = out / "shard_000_acts.npy"
            b = bytearray(f.read_bytes()); b[-1] ^= 1; f.write_bytes(bytes(b))
            self.assertFalse(verify_shards(out)["hashes_and_shapes_ok"])

    def test_payload_integrity_enforced(self):
        rows = payload(ROWS_IN[:3])
        with tempfile.TemporaryDirectory() as t:
            for bad, dig in ((rows, "0" * 64), ([dict(rows[0], oracle=1)] + rows[1:], None),
                             ([dict(rows[0], expected_n_tokens=1)] + rows[1:], None)):
                with self.assertRaises(AssertionError):
                    run_capture(bad, [], dig or digest(bad), pathlib.Path(t) / "x", "cpu", MODEL, TOK, 600, 5, 1)

    def test_shardwriter_roundtrip(self):
        with tempfile.TemporaryDirectory() as t:
            w = ShardWriter(pathlib.Path(t), 2)
            for i in range(3):
                w.add(f"w{i}", {"acts_u16": np.full((L + 2, D), i, np.uint16), "mean_emb": np.full(D, i, np.float32), "meta": {"i": i}})
            man = w.close({"n_worlds": 3})
            self.assertEqual(verify_shards(pathlib.Path(t)), {"hashes_and_shapes_ok": True, "n_worlds": 3})
            self.assertEqual(len(man["shards"]), 2)


if __name__ == "__main__":
    unittest.main()
