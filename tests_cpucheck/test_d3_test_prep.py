"""BTC-D3-TEST preparation tests: stage-scoped sealed access, single-evaluation guard, evaluator end-to-end.
Synthetic data only. Fake sealed pools are built from PROBE-DEV worlds in temporary directories; the real
LOCKED-TEST pool is never opened (only its seal is hash-checked by other tests)."""

import json
import os
import pathlib
import stat
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "sealed_stage"))
import h1_features as F  # noqa: E402
import h1_frozen as Z  # noqa: E402
import h1_train_dev as T  # noqa: E402
import sealed  # noqa: E402
import sealed_access as SA  # noqa: E402
import h1_eval_test as E  # noqa: E402

DEV = ROOT / "data" / "d5" / "probe_dev"
LABEL_RE = r"(?<= )([A-Z0-9]{4})(?=[.,:])"


def _unlock(root, stage, seal_id, files, scope=None):
    d = SA.unlock_dir(stage, root)
    d.mkdir(parents=True, exist_ok=True)
    sealed.unlock_path(d, seal_id).write_text(json.dumps({"owner_approval_id": "TEST", "approved_utc": "now",
                                                          "scope": scope or stage, "files": files}), encoding="utf-8")


def _rw(d):
    for p in pathlib.Path(d).rglob("*"):
        if p.is_file():
            os.chmod(p, stat.S_IWRITE | stat.S_IREAD)


def fake_pool(d, worlds, inputs):
    d.mkdir(parents=True)
    (d / "worlds.jsonl").write_text("".join(json.dumps(w) + "\n" for w in worlds), encoding="utf-8")
    (d / "model_inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in inputs), encoding="utf-8")
    sealed.seal_dir(d, SA.POOL_SEAL)


def fake_acts(d, worlds, inputs, dim=16, signal=3.0, seed=1):
    """Fake TEST capture: planted u2-label embedding at vector 12 (signal=0 -> null), new random label table."""
    import re
    rng = np.random.default_rng(seed)
    labels = sorted({m.group(1) for r in inputs for m in re.finditer(LABEL_RE, r["prompt"])})
    Tt = rng.standard_normal((len(labels), dim)).astype(np.float32)
    li = {l: i for i, l in enumerate(labels)}
    acts = rng.standard_normal((len(inputs), 62, dim)).astype(np.float32)
    W = {w["world_id"]: w for w in worlds}
    idx = []
    for k, r in enumerate(inputs):
        w = W[r["world_id"]]
        opt = w["oracle"]["optimal_action"]
        sup = F.supervision_targets(w, opt)
        if "u2" in sup:
            acts[k, 12] += signal * Tt[li[sup["u2"]]]
        idx.append({"world_id": r["world_id"], "logp_A": -0.1 if opt == "A" else -3.0, "logp_B": -3.0 if opt == "A" else -0.1})
    u16 = torch.from_numpy(acts).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    d.mkdir(parents=True)
    np.save(d / "shard_000_acts.npy", u16)
    np.save(d / "shard_000_mean_emb.npy", acts[:, 0])
    (d / "shard_000_index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in idx), encoding="utf-8")
    np.save(d / "candidate_embeddings.npy", Tt)
    (d / "candidate_labels.json").write_text(json.dumps({"labels": labels}), encoding="utf-8")
    (d / "MANIFEST.json").write_text(json.dumps({"shards": [{"name": "shard_000", "n": len(inputs), "files": {}}]}), encoding="utf-8")
    sealed.seal_dir(d, SA.ACT_SEAL)


class StageScopes(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        base = pathlib.Path(self.t.name)
        self.pool, self.root = base / "pool", base / "unlock"
        fake_pool(self.pool, [{"world_id": "w"}], [{"world_id": "w", "prompt": "p"}])

    def tearDown(self):
        _rw(self.t.name)
        self.t.cleanup()

    def test_capture_stage_reads_prompts_only(self):
        with self.assertRaises(PermissionError):           # no unlock at all
            SA.staged_read("D3-TEST-CAPTURE", self.pool, "model_inputs.jsonl", "x", self.root)
        _unlock(self.root, "D3-TEST-CAPTURE", SA.POOL_SEAL, ["model_inputs.jsonl"])
        self.assertIn(b'"prompt"', SA.staged_read("D3-TEST-CAPTURE", self.pool, "model_inputs.jsonl", "x", self.root))
        with self.assertRaises(PermissionError):           # oracle never readable in the capture stage
            SA.staged_read("D3-TEST-CAPTURE", self.pool, "worlds.jsonl", "x", self.root)
        with self.assertRaises(PermissionError):           # capture unlock does not open the analysis stage
            SA.staged_read("H1-TEST-EVAL", self.pool, "worlds.jsonl", "x", self.root)
        log = SA.access_log(self.pool)
        self.assertEqual([r["file"] for r in log], ["model_inputs.jsonl"])
        self.assertTrue(log[0]["purpose"].startswith("D3-TEST-CAPTURE:"))

    def test_capture_unlock_listing_worlds_is_still_refused(self):
        _unlock(self.root, "D3-TEST-CAPTURE", SA.POOL_SEAL, "*")
        with self.assertRaises(PermissionError):
            SA.staged_read("D3-TEST-CAPTURE", self.pool, "worlds.jsonl", "x", self.root)

    def test_scope_mismatch_and_unlisted_file_refused(self):
        _unlock(self.root, "H1-TEST-EVAL", SA.POOL_SEAL, ["model_inputs.jsonl"], scope="D3-TEST-CAPTURE")
        with self.assertRaises(PermissionError):
            SA.staged_read("H1-TEST-EVAL", self.pool, "model_inputs.jsonl", "x", self.root)
        _unlock(self.root, "H1-TEST-EVAL", SA.POOL_SEAL, ["model_inputs.jsonl"])
        with self.assertRaises(PermissionError):
            SA.staged_read("H1-TEST-EVAL", self.pool, "worlds.jsonl", "x", self.root)
        self.assertEqual(SA.reads_outside_stage(self.pool, "D3-TEST-CAPTURE"), [])
        _unlock(self.root, "H1-TEST-EVAL", SA.POOL_SEAL, ["model_inputs.jsonl", "worlds.jsonl"])
        self.assertTrue(SA.staged_read("H1-TEST-EVAL", self.pool, "worlds.jsonl", "x", self.root))
        # an oracle read is detectable afterwards by the capture preflight and the evaluator prechecks
        self.assertEqual(len(SA.reads_outside_stage(self.pool, "D3-TEST-CAPTURE")), 1)


class SingleEvaluation(unittest.TestCase):
    def test_second_evaluation_refused_before_any_read(self):
        with tempfile.TemporaryDirectory() as t:
            (pathlib.Path(t) / "EVAL_STARTED.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(SystemExit):
                E.run(pathlib.Path(t), pathlib.Path(t) / "no_unlock")


class EvaluatorEndToEnd(unittest.TestCase):
    """Synthetic predictors trained by the real runner; fake sealed TEST pool and activations read through staged_read."""

    @classmethod
    def setUpClass(cls):
        cls.t = tempfile.TemporaryDirectory()
        base = pathlib.Path(cls.t.name)
        cap, devdir = T.make_synthetic(base / "syn", n_worlds=960, d=16)
        dev = T.Dev(cap, devdir)
        st = T.run(dev, base / "out", layers=(0, 12), ranks=(8,), wds=(1e-3,), seeds=(0,), max_epochs=60)
        cls.L = Z.load_final(base / "out")
        cls.S_sha = Z.freeze_standardisation(dev, st, base / "out" / "final" / Z.STD_FILE)
        cls.S = Z.load_standardisation(base / "out" / "final" / Z.STD_FILE, cls.S_sha)
        inputs = [json.loads(x) for x in (DEV / "model_inputs.jsonl").read_text(encoding="utf-8").splitlines()][960:1920]
        W = {r["world_id"]: r for r in map(json.loads, (DEV / "worlds.jsonl").read_text(encoding="utf-8").splitlines())}
        cls.worlds = [W[r["world_id"]] for r in inputs]
        cls.inputs = inputs
        cls.base = base

    @classmethod
    def tearDownClass(cls):
        _rw(cls.t.name)
        cls.t.cleanup()

    def _run(self, tag, signal, min_pairs=10, min_fams=5):
        pool, act, root = self.base / f"pool_{tag}", self.base / f"act_{tag}", self.base / f"unlock_{tag}"
        fake_pool(pool, self.worlds, self.inputs)
        fake_acts(act, self.worlds, self.inputs, signal=signal)
        _unlock(root, "H1-TEST-EVAL", SA.POOL_SEAL, ["worlds.jsonl", "model_inputs.jsonl"])
        _unlock(root, "H1-TEST-EVAL", SA.ACT_SEAL, "*")
        P = self.L["predictors"]
        layers = sorted({p["input"] for p in P.values() if isinstance(p["input"], int)})
        idx, Zl, me, labels, Traw, inp, wor = E.load_test(pool, act, root, layers)
        rows, audit = E.build_rows(idx, inp, wor)
        res, ws = E.evaluate(rows, audit, P, self.S, Zl, me, labels, Traw, min_pairs, min_fams)
        return res, ws, rows, pool, act

    def test_planted_signal_met_and_logged(self):
        res, ws, rows, pool, act = self._run("sig", 3.0)
        pr = res["primary"]
        self.assertGreater(pr["n_pairs"], 10)
        self.assertGreater(pr["point"]["primary_u2"], pr["point"]["B3"])
        self.assertEqual(set(pr["delta_ci95"]), {"B1", "B2", "B3", "B6"})
        self.assertIn(res["H1_outcome"], ("MET + L2", "MET, L2 not attributed"))
        self.assertIn("a_order_same_layer", res["L2"])
        self.assertEqual(set(res["secondary"]["holm_family"]["members"]), {"S1_pooled_depth3_4_upath", "S2_xswap_depth4"})
        self.assertIn("not the whole plan", res["secondary"]["holm_family"]["note"])
        self.assertAlmostEqual(pr["point"]["B5"], pr["point"]["B5"])  # computed
        self.assertTrue(all(r["purpose"].startswith("H1-TEST-EVAL:") for r in SA.access_log(pool) + SA.access_log(act)))
        # decoys are excluded from every decoder (10 / 7 candidates), and counted
        self.assertFalse(any(r["variant"].startswith("decoy") for r in rows))
        self.assertTrue(any(k.startswith("not_evaluable_K") for k in res["admissibility"]))

    def test_null_signal_not_met_and_l2_not_evaluated(self):
        res, *_ = self._run("null", 0.0)
        self.assertEqual(res["H1_outcome"], "NOT MET")
        self.assertIn("NOT EVALUATED", res["L2"]["status"])

    def test_insufficient_rule_with_frozen_thresholds(self):
        res, *_ = self._run("ins", 3.0, min_pairs=E.MIN_PAIRS, min_fams=E.MIN_FAMILIES)
        self.assertEqual(res["H1_outcome"], "INSUFFICIENT")
        self.assertNotIn("delta_ci95", res["primary"])


class OwnerDecisions(unittest.TestCase):
    def _row_and_probs(self, b2_hits_true):
        cands = ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE", "FFFF"]
        tgt = {s: i for i, s in enumerate(("c1", "c2", "c3", "u1", "u2", "u3"))}
        row = {"family_id": "F99999", "variant": "upath_base", "depth": 4, "tgt": tgt, "cands": cands}
        names = ["primary_u2", "B1", "B5", "B3", "order_u1", "order_u3", "B3o_u1", "B3o_u2", "B3o_u3",
                 "B2_perm0", "B2_perm1", "B2_perm2"] + [f"anchor_{h}" for h in Z.ANCHORS]
        probs = {n: torch.full((1, 6), 1 / 6) for n in names}
        one = torch.zeros(1, 6)
        one[0, tgt["u2"] if b2_hits_true else tgt["c1"]] = 1.0
        for k in range(3):
            probs[f"B2_perm{k}"] = one.clone()
        return [row], probs, torch.arange(6).unsqueeze(0)

    def test_c1_b2_scored_against_true_u2(self):
        rows, probs, cand = self._row_and_probs(True)
        ws = E.world_scores(rows, probs, cand)
        self.assertEqual([ws[0][f"B2_perm{k}"] for k in range(3)], [1.0, 1.0, 1.0])
        rows, probs, cand = self._row_and_probs(False)
        self.assertEqual([E.world_scores(rows, probs, cand)[0][f"B2_perm{k}"] for k in range(3)], [0.0, 0.0, 0.0])

    def test_o2_every_preregistered_secondary_has_a_status(self):
        ids = {p["id"]: p["status"] for p in E.PLAN_STATUS}
        self.assertEqual(set(ids), {"S1", "S2", "S3", "S4", "H2", "B7", "B4"})
        self.assertTrue(ids["S1"].startswith("INFERENTIAL") and ids["S2"].startswith("INFERENTIAL"))
        self.assertEqual(ids["S4"], "NOT COMPUTABLE")
        self.assertEqual(ids["H2"], "NOT RUN")
        self.assertTrue(all("reason" in p for p in E.PLAN_STATUS if p["status"].startswith("NOT")))


class FrozenConstants(unittest.TestCase):
    def test_thresholds_and_bootstrap_are_v03(self):
        self.assertEqual((E.MIN_PAIRS, E.MIN_FAMILIES, E.B_RESAMPLES, E.BOOT_SEED), (180, 100, 10_000, 0))
        self.assertAlmostEqual(E.CHANCE_ORDER, 1 / 6)

    def test_only_sealed_stage_names_sealed_paths(self):
        for f in (ROOT / "src").glob("*.py"):
            if f.name not in ("build_d5.py", "sealed.py"):
                self.assertNotIn("locked_test", f.read_text(encoding="utf-8"), f.name)

    def test_capture_script_never_names_the_oracle_file(self):
        src = (ROOT / "src" / "sealed_stage" / "d3_run_gemma_test.py").read_text(encoding="utf-8")
        self.assertNotIn("worlds.jsonl", src)
        self.assertIn('STAGE = "D3-TEST-CAPTURE"', src)
        self.assertIn("from d3_capture_core import run_capture", src)


if __name__ == "__main__":
    unittest.main()
