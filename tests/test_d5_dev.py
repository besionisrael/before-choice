"""Audit of PROBE-DEV (D5) and of the sealing mechanism. LOCKED-TEST is never generated here."""

import hashlib
import json
import os
import pathlib
import stat
import subprocess
import sys
import tempfile
import unittest
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from build_d5 import POOLS, build, exclusion_labels, labels_of  # noqa: E402
from independent_check import audit_world  # noqa: E402
from oracle import format_prompt  # noqa: E402
import sealed  # noqa: E402

DEV = build("PROBE-DEV")
D = ROOT / "data" / "d5" / "probe_dev"


class ProbeDev(unittest.TestCase):
    def test_files_match_generator_and_manifest(self):
        man = json.loads((D / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256((D / "worlds.jsonl").read_bytes()).hexdigest(), man["sha256"]["all"])
        rows = [json.loads(x) for x in (D / "worlds.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows, json.loads(json.dumps(DEV)))
        self.assertEqual(man["seed"], 2026111401)

    def test_composition_and_balance(self):
        fam = Counter(r["family_id"] for r in DEV)
        self.assertEqual(len(fam), 300)
        self.assertTrue(all(v == 16 for v in fam.values()))
        ids = sorted(int(f[1:]) for f in fam)
        self.assertEqual((ids[0], ids[-1]), (20000, 20299))
        base = [r for r in DEV if r["variant"] == "base"]
        self.assertEqual(Counter(r["depth"] for r in base), {4: 200, 3: 100})
        for d in (3, 4):
            for s in ("train", "val"):
                c = Counter(r["oracle"]["optimal_action"] for r in base if r["depth"] == d and r["split"] == s)
                self.assertEqual(c["A"], c["B"])
        self.assertEqual(Counter((r["depth"], r["split"]) for r in base),
                         {(4, "train"): 150, (4, "val"): 50, (3, "train"): 76, (3, "val"): 24})

    def test_independent_oracle_and_template(self):
        self.assertEqual([r["world_id"] for r in DEV if audit_world(r)], [])
        for r in DEV:
            self.assertEqual(r["prompt"], format_prompt(r))
            self.assertEqual(r["prompt_sha256"], hashlib.sha256(r["prompt"].encode()).hexdigest())

    def test_family_isolation_and_label_disjointness(self):
        split = {}
        for r in DEV:
            self.assertEqual(split.setdefault(r["family_id"], r["split"]), r["split"])
        owner = {}
        for r in DEV:
            for x in labels_of([r]):
                self.assertEqual(owner.setdefault(x, r["family_id"]), r["family_id"])
        self.assertFalse(labels_of(DEV) & exclusion_labels("PROBE-DEV"))

    def test_model_inputs_oracle_free(self):
        for line in (D / "model_inputs.jsonl").read_text(encoding="utf-8").splitlines():
            self.assertEqual(set(json.loads(line)), {"world_id", "family_id", "variant", "pool", "prompt", "prompt_sha256"})

    def test_primary_population_constructible(self):
        fam = defaultdict(dict)
        for r in DEV:
            fam[r["family_id"]][r["variant"]] = r
        n = 0
        for v in fam.values():
            if v["base"]["depth"] != 4:
                continue
            for src in ("base", "cf"):
                s, u = v[src], v[f"upath_{src}"]
                non = "B" if s["oracle"]["optimal_action"] == "A" else "A"
                self.assertNotEqual(s["oracle"]["paths"][non][2], u["oracle"]["paths"][non][2])
                n += 1
        self.assertEqual(n, 400)


class LockedTestSealing(unittest.TestCase):
    def test_locked_test_sealed_read_only(self):
        d = POOLS["LOCKED-TEST"]["out"]
        self.assertTrue((d / sealed.MANIFEST).exists())
        self.assertTrue(all(sealed.hash_only(d).values()))
        for f in d.iterdir():
            if f.name != sealed.ACCESS_LOG:
                self.assertFalse(os.stat(f).st_mode & stat.S_IWRITE, f.name)
        # Amendment D3-A4 (owner decision T-1, 2026-11-14): the access log is never deleted or reset. Until the
        # owner's H1-TEST-EVAL unlock it may contain ONLY capture-stage reads of the oracle-free prompt file;
        # any other entry (oracle file, other purpose) is a forbidden access and fails this test.
        log = d / sealed.ACCESS_LOG
        if log.exists():
            recs = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
            for r in recs:
                self.assertEqual(r["file"], "model_inputs.jsonl", r)
                self.assertTrue(r["purpose"].startswith("D3-TEST-CAPTURE:"), r)

    def test_generation_refused_without_approval(self):
        p = subprocess.run([sys.executable, str(ROOT / "src" / "build_d5.py"), "--pool", "LOCKED-TEST"],
                           capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("REFUSED", p.stdout + p.stderr)

    def test_no_dev_module_references_sealed_paths(self):
        allowed = {"build_d5.py", "sealed.py"}
        for f in (ROOT / "src").glob("*.py"):
            if f.name in allowed:
                continue
            src = f.read_text(encoding="utf-8")
            # the sealed DIRECTORY must not be referenced (mentioning the pool name in prose is fine)
            self.assertNotIn("locked_test", src, f.name)

    def test_seal_roundtrip(self):
        with tempfile.TemporaryDirectory() as t:
            d, u = pathlib.Path(t) / "pool", pathlib.Path(t) / "unlock"
            d.mkdir(); u.mkdir()
            (d / "worlds.jsonl").write_bytes(b'{"x": 1}\n')
            man = sealed.seal_dir(d, "TEST-SEAL")
            self.assertTrue(all(sealed.hash_only(d).values()))
            self.assertFalse(os.stat(d / "worlds.jsonl").st_mode & stat.S_IWRITE)
            with self.assertRaises(PermissionError):
                sealed.sealed_read(d, "worlds.jsonl", u, "test")
            sealed.unlock_path(u, "TEST-SEAL").write_text(json.dumps(
                {"owner_approval_id": "X", "approved_utc": "now", "scope": "test"}), encoding="utf-8")
            self.assertEqual(sealed.sealed_read(d, "worlds.jsonl", u, "test"), b'{"x": 1}\n')
            log = (d / sealed.ACCESS_LOG).read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(log), 1)
            os.chmod(d / "worlds.jsonl", stat.S_IWRITE | stat.S_IREAD)
            (d / "worlds.jsonl").write_bytes(b'{"x": 2}\n')
            self.assertFalse(sealed.hash_only(d)["worlds.jsonl"])
            with self.assertRaises(ValueError):
                sealed.sealed_read(d, "worlds.jsonl", u, "test")
            for p in d.iterdir():
                os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
            self.assertEqual(man["seal_id"], "TEST-SEAL")


if __name__ == "__main__":
    unittest.main()
