"""Audit of BTC-V2 SCREEN / COMPETENCE pools (CPU)."""

import hashlib
import json
import pathlib
import sys
import unittest
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from build_dataset import write_jsonl  # noqa: E402
from build_v2 import MASTER_SEED, build, v1_labels  # noqa: E402
from independent_check import audit_world  # noqa: E402
from oracle import format_prompt  # noqa: E402

POOLS = build()
ALL = POOLS["SCREEN"] + POOLS["COMPETENCE"]
BY = {w["world_id"]: w for w in ALL}
DATA = ROOT / "data" / "v2"


def labels(w):
    return {x for e in w["transitions"] for x in (e["source"], e["target"])} - {"S"}


def other(a):
    return "B" if a == "A" else "A"


class Integrity(unittest.TestCase):
    def test_seed_and_determinism(self):
        self.assertEqual(MASTER_SEED, 2026111302)
        again = build()
        for p in POOLS:
            self.assertEqual(json.dumps(POOLS[p], sort_keys=True), json.dumps(again[p], sort_keys=True))

    def test_files_and_manifest_match_generator(self):
        man = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
        for p, rows in POOLS.items():
            f = DATA / f"{p.lower()}.jsonl"
            self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(), man["pools"][p]["sha256"])
            self.assertEqual([json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()], json.loads(json.dumps(rows)))
        self.assertEqual(man["not_generated"], ["PROBE-DEV", "LOCKED-TEST"])
        self.assertFalse((DATA / "probe_dev.jsonl").exists() or (DATA / "locked_test.jsonl").exists())

    def test_ground_truth_independent_oracle(self):
        self.assertEqual({w["world_id"]: audit_world(w) for w in ALL if audit_world(w)}, {})

    def test_prompts_are_v1_template(self):
        for w in ALL:
            self.assertEqual(w["prompt"], format_prompt(w))
            self.assertEqual(w["prompt_sha256"], hashlib.sha256(w["prompt"].encode("utf-8")).hexdigest())

    def test_model_inputs_have_no_oracle(self):
        for p in POOLS:
            for line in (DATA / f"{p.lower()}_model_inputs.jsonl").read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                self.assertEqual(set(r), {"world_id", "family_id", "variant", "pool", "prompt", "prompt_sha256"})

    def test_builder_never_reads_v1_files(self):
        src = (ROOT / "src" / "build_v2.py").read_text(encoding="utf-8")
        for s in ("splits/", "test.jsonl", "pilot_v1.jsonl", "open("):
            self.assertNotIn(s, src)


class Isolation(unittest.TestCase):
    def test_family_ids_and_pools(self):
        fam_pool = {}
        for w in ALL:
            n = int(w["family_id"][1:])
            self.assertTrue(10000 <= n < 20000)
            self.assertEqual(fam_pool.setdefault(w["family_id"], w["pool"]), w["pool"])
        self.assertEqual(Counter(fam_pool.values()), {"SCREEN": 60, "COMPETENCE": 90})

    def test_no_v1_label_reused(self):
        v1 = v1_labels()
        self.assertEqual(len(v1), 480)
        used = set().union(*(labels(w) for w in ALL))
        self.assertFalse(used & v1)

    def test_labels_private_to_family(self):
        owner = {}
        for w in ALL:
            for x in labels(w):
                self.assertEqual(owner.setdefault(x, w["family_id"]), w["family_id"], x)


class Balance(unittest.TestCase):
    def test_counts_and_balance(self):
        for p, per in (("SCREEN", 10), ("COMPETENCE", 15)):
            base = [w for w in POOLS[p] if w["variant"] == "base"]
            for d in (2, 3, 4):
                c = Counter(w["oracle"]["optimal_action"] for w in base if w["depth"] == d)
                self.assertEqual(c, {"A": per, "B": per})
        self.assertEqual(Counter(w["variant"] for w in POOLS["SCREEN"]),
                         {"base": 60, "cf": 60, "swap_base": 60, "swap_cf": 60})
        self.assertEqual(len(POOLS["COMPETENCE"]), 1320)


class Variants(unittest.TestCase):
    def setUp(self):
        self.fam = defaultdict(dict)
        for w in ALL:
            self.fam[w["family_id"]][w["variant"]] = w

    def test_cf_flips_and_minimal(self):
        for f, v in self.fam.items():
            for pre in ("", "alias_", "swap_", "shuffle_", "decoy_"):
                if pre + "base" not in v:
                    continue
                b, c = v[pre + "base"], v[pre + "cf"]
                self.assertNotEqual(b["oracle"]["optimal_action"], c["oracle"]["optimal_action"])
                self.assertEqual(b["terminal_rewards"], c["terminal_rewards"])
                self.assertEqual(len(b["prompt"]), len(c["prompt"]))
                self.assertEqual(sum(x != y for x, y in zip(b["prompt"].splitlines(), c["prompt"].splitlines())), 2)

    def test_swap_only_letters(self):
        for f, v in self.fam.items():
            b, s = v["base"], v["swap_base"]
            self.assertEqual(b["prompt"].replace("action A", "#").replace("action B", "action A").replace("#", "action B"), s["prompt"])

    def test_competence_controls(self):
        for f, v in self.fam.items():
            if v["base"]["pool"] != "COMPETENCE":
                continue
            b = v["base"]
            key = lambda w: sorted(json.dumps(e, sort_keys=True) for e in w["transitions"])
            self.assertEqual(key(b), key(v["shuffle_base"]))
            self.assertNotEqual(b["prompt"], v["shuffle_base"]["prompt"])
            self.assertFalse(labels(b) & labels(v["alias_base"]))
            self.assertEqual(b["oracle"]["optimal_action"], v["alias_base"]["oracle"]["optimal_action"])
            d, dc = v["decoy_base"], v["decoy_cf"]
            ch = d["oracle"]["decoy_chain"]
            self.assertFalse(set(ch) & set(sum(d["oracle"]["paths"].values(), [])))
            self.assertGreater(d["terminal_rewards"][ch[-1]], max(d["oracle"]["rewards"].values()))
            self.assertEqual(d["terminal_rewards"], dc["terminal_rewards"])

    def test_pathcf_choice_preserving(self):
        n = Counter()
        for f, v in self.fam.items():
            for name, w in v.items():
                kind = name.split("_")[0]
                if kind not in ("upath", "cpath", "xswap"):
                    continue
                s = v[name.split("_", 1)[1]]
                n[kind] += 1
                self.assertEqual(w["oracle"]["optimal_action"], s["oracle"]["optimal_action"])
                self.assertEqual(w["terminal_rewards"], s["terminal_rewards"])
                self.assertEqual(len(w["prompt"]), len(s["prompt"]))
                opt = s["oracle"]["optimal_action"]
                if kind == "upath":
                    self.assertEqual(w["oracle"]["paths"][opt], s["oracle"]["paths"][opt])
                    self.assertNotEqual(w["oracle"]["paths"][other(opt)][2], s["oracle"]["paths"][other(opt)][2])
                if kind == "cpath":
                    self.assertEqual(w["oracle"]["paths"][other(opt)], s["oracle"]["paths"][other(opt)])
        self.assertEqual(n, {"upath": 120, "cpath": 120, "xswap": 180})


class Writer(unittest.TestCase):
    def test_lf_hash_stable(self):
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            h = write_jsonl(POOLS["SCREEN"], pathlib.Path(t) / "s.jsonl")
        man = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(h, man["pools"]["SCREEN"]["sha256"])


if __name__ == "__main__":
    unittest.main()
