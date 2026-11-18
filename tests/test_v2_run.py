"""CPU tests for BTC-V2 run scripts, budget and scorer (synthetic outputs in memory only)."""

import importlib.util
import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import v2_common as C  # noqa: E402
from score_v2 import load_oracle, score  # noqa: E402

HAS_MODAL = importlib.util.find_spec("modal") is not None


class Budget(unittest.TestCase):
    def test_limits_match_freeze(self):
        self.assertEqual((C.GPU, C.CPU, C.MEMORY_MIB), ("A100-80GB", (4.0, 4.0), (16384, 98304)))
        self.assertEqual((C.STAGES["S"]["timeout_s"], C.STAGES["S"]["guard_s"]), (780, 720))
        self.assertEqual((C.STAGES["C"]["timeout_s"], C.STAGES["C"]["guard_s"]), (1240, 1160))
        self.assertAlmostEqual(C.worst_case_rate(), 0.00095952, places=8)

    def test_worst_case_within_objectives_with_margin(self):
        s, c = C.worst_case_cost("S"), C.worst_case_cost("C")
        self.assertLessEqual(s, 0.85 * 1.00 + 0.001)
        self.assertLessEqual(c, 0.86 * 1.50)
        self.assertLessEqual(3 * s + c, 0.86 * C.PROGRAMME_OBJECTIVE_USD)

    @unittest.skipUnless(HAS_MODAL, "modal not installed")
    def test_modal_decorators(self):
        for f in ("v2_run_std.py", "v2_run_gemma.py"):
            src = (ROOT / "src" / f).read_text(encoding="utf-8")
            self.assertIn("_KW = dict(image=image, gpu=GPU, cpu=CPU, memory=MEMORY_MIB, retries=0, max_containers=1)", src)
            self.assertIn('@app.function(timeout=STAGES["S"]["timeout_s"], **_KW)', src)
            self.assertIn('@app.function(timeout=STAGES["C"]["timeout_s"], **_KW)', src)
            self.assertNotIn("modal.Volume", src)
            self.assertNotIn("volumes=", src)
            self.assertNotIn("output_hidden_states", src)
        from modal._resources import convert_fn_config_to_resources_config as conv
        r = conv(cpu=C.CPU, memory=C.MEMORY_MIB, gpu=C.GPU)
        self.assertEqual((r.milli_cpu, r.milli_cpu_max, r.memory_mb, r.memory_mb_max, r.gpu_config.gpu_type),
                         (4000, 4000, 16384, 98304, "A100-80GB"))

    def test_image_pins_match_cpu_check_envs(self):
        std = (ROOT / "src" / "v2_run_std.py").read_text(encoding="utf-8")
        for pin in ("torch==2.5.1", "transformers==4.57.1", "huggingface_hub==0.36.2", "accelerate==1.10.1"):
            self.assertIn(pin, std)
        frz = (ROOT / "outputs" / "v2_audit" / "gemma4_env_freeze.txt").read_text(encoding="utf-8")
        gem = (ROOT / "src" / "v2_run_gemma.py").read_text(encoding="utf-8")
        for pkg in ("transformers", "huggingface_hub", "tokenizers", "safetensors", "accelerate"):
            line = next(x for x in frz.splitlines() if x.lower().startswith(pkg + "=="))
            self.assertIn(f'"{line}"', gem, pkg)
        self.assertIn('"torch==2.6.0"', gem)


class Isolation(unittest.TestCase):
    def test_payload_source_is_frozen_inputs_only(self):
        for pool in C.POOLS:
            rows = C.load_model_inputs(ROOT, pool)
            self.assertEqual(C.prompt_digest(rows), C.POOLS[pool]["prompt_digest"])
        src = (ROOT / "src" / "v2_local.py").read_text(encoding="utf-8")
        self.assertNotIn("load_oracle", src)
        self.assertNotIn("splits", src)

    def test_run_registry(self):
        self.assertEqual(C.RUNS["BTC-V2-S1"], ("S", "qwen3_32b"))
        self.assertEqual(C.RUNS["BTC-V2-C1"], ("C", None))
        self.assertEqual(C.MODELS["qwen3_32b"]["revision"], "9216db5781bf21249d130ec9da846c4624c16137")
        self.assertEqual(C.MODELS["qwen3_32b"]["template_kwargs"], {"enable_thinking": False})


class CrossEnvTokens(unittest.TestCase):
    """Local (container-mirroring) counts must equal the counts produced in the CPU-check envs."""

    def test_counts_equal(self):
        from v2_local import local_encoder
        rows = C.load_model_inputs(ROOT, "SCREEN") + C.load_model_inputs(ROOT, "COMPETENCE")
        for mk in C.MODELS:
            f = ROOT / "outputs" / "v2_audit" / f"token_counts_{mk}.json"
            if not f.exists():
                self.skipTest(f"run src/v2_cpu_checks.py for {mk}")
            ref = json.loads(f.read_text(encoding="utf-8"))
            cnt = local_encoder(mk)
            bad = [r["world_id"] for r in rows if cnt(r["prompt"]) != ref[r["world_id"]]]
            self.assertEqual(bad, [], mk)

    def test_cpu_check_reports_pass(self):
        for mk in C.MODELS:
            rep = json.loads((ROOT / "outputs" / "v2_audit" / f"cpu_check_{mk}.json").read_text(encoding="utf-8"))
            self.assertEqual(rep["template_tail_ok"], "1560/1560")
            self.assertTrue(all(rep["tamper_rejected"].values()))
            self.assertEqual(rep["meta_num_hidden_layers"] + 1, C.MODELS[mk]["n_hidden_states"])
            self.assertEqual(rep["e2e_tiny_model"]["vocab_size"], rep["meta_vocab_out"])


def fake(pool, policy, top1=lambda o: True, tie=lambda o: False):
    oracle = load_oracle(pool)
    raw = []
    for rep, wids in ((0, sorted(oracle)), (1, sorted(oracle)[:20])):
        for w in wids:
            o = oracle[w]
            c = policy(o)
            a, b = (-0.7, -0.7) if tie(o) else ((-0.1, -3.0) if c == "A" else (-3.0, -0.1))
            raw.append({"world_id": w, "replay": rep, "logp_A": a, "logp_B": b,
                        "top1_in_AB": top1(o), "top1_token": c if top1(o) else "Let"})
    return raw


OPT = lambda o: o["oracle"]["optimal_action"]


class Scorer(unittest.TestCase):
    def test_screen_perfect_passes(self):
        m = score("BTC-V2-S1", fake("SCREEN", OPT))["metrics"]
        self.assertTrue(m["SCREEN_PASS"])
        self.assertEqual(m["depth2_base_cf"]["n"], 40)
        self.assertEqual(m["depth2_base_cf_pairs"]["n"], 20)

    def test_top1_conformity_independent_of_forced_choice(self):
        # forced A/B choice is always correct, but top-1 over the full vocabulary is "Let" in 10% of worlds
        bad = lambda o: int(o["family_id"][1:]) % 10 != 0
        m = score("BTC-V2-S1", fake("SCREEN", OPT, top1=bad))["metrics"]
        self.assertEqual(m["depth2_base_cf"]["acc_primary"], 1.0)
        self.assertFalse(m["criteria"]["S4"])
        self.assertFalse(m["SCREEN_PASS"])

    def test_constant_policy_fails_s1_s2(self):
        m = score("BTC-V2-S1", fake("SCREEN", lambda o: "A"))["metrics"]
        self.assertFalse(m["criteria"]["S1"])
        self.assertFalse(m["criteria"]["S2"])

    def test_competence_perfect(self):
        m = score("BTC-V2-C1", fake("COMPETENCE", OPT))["metrics"]
        self.assertEqual(m["COMPETENCE"], "PASS")
        self.assertTrue(m["D1prime_FEASIBLE"])
        self.assertEqual(m["upath_depth4"]["constructed"], 60)

    def test_competence_shallow_only(self):
        pol = lambda o: OPT(o) if o["depth"] < 4 else "A"
        m = score("BTC-V2-C1", fake("COMPETENCE", pol))["metrics"]
        self.assertEqual(m["COMPETENCE"], "PASS")
        self.assertFalse(m["criteria"]["C4"])
        self.assertFalse(m["D1prime_FEASIBLE"])

    def test_results_must_cover_pool(self):
        raw = fake("SCREEN", OPT)[:-30]
        with self.assertRaises(AssertionError):
            score("BTC-V2-S1", raw)


if __name__ == "__main__":
    unittest.main()
