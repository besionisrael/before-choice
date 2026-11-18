"""PROPOSED post-evaluation seal check (closure of BTC-H1 v0.3). NOT part of any test suite until the owner approves it.
The historical test tests/test_d5_dev.py::test_locked_test_sealed_read_only (amendment D3-A4) is kept unchanged; it
fails by design since the authorised H1-TEST-EVAL reads.

This check reads ONLY seal manifests, access logs, unlock files and the freeze manifest (no sealed content). It accepts:
  * capture-scope reads of model_inputs.jsonl (D3-TEST-CAPTURE), each with the approval id of attempt 1 or R1;
  * H1-TEST-EVAL reads made under the owner's analytical unlock, at or after the EVAL_STARTED marker time;
and requires:
  * both seals intact, files read-only;
  * the logs to be APPEND-ONLY with respect to the closure snapshot (the archived log is a prefix of the current log),
    so no trace can be deleted or rewritten;
  * exactly one evaluation (marker + result present, hashes equal to the freeze manifest).
Run: python -m unittest tests_proposed.test_post_eval_seal
"""

import json
import os
import pathlib
import stat
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "sealed_stage"))
import sealed  # noqa: E402
import sealed_access as SA  # noqa: E402

ARCHIVE = ROOT / "outputs" / "archive" / "BTC-H1-v0.3"


def _log_bytes(d):
    p = d / sealed.ACCESS_LOG
    return p.read_bytes() if p.exists() else b""


class PostEvaluationSeals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frz = json.loads((ROOT / "reports" / "BTC-H1_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
        cls.started = json.loads((ROOT / "outputs" / "h1_test" / "EVAL_STARTED.json").read_text(encoding="utf-8"))["utc"]
        cls.eval_ids = {json.loads((SA.unlock_dir("H1-TEST-EVAL") / f"UNLOCK_{s}.json").read_text(encoding="utf-8"))["owner_approval_id"]
                        for s in (SA.POOL_SEAL, SA.ACT_SEAL)}
        cap = json.loads((SA.unlock_dir("D3-TEST-CAPTURE") / f"UNLOCK_{SA.POOL_SEAL}.json").read_text(encoding="utf-8"))
        cls.capture_ids = {cap["owner_approval_id"]} | {p["owner_approval_id"] for p in cap.get("previous_approvals", [])}

    def test_seals_intact_and_read_only(self):
        for d in (SA.POOL_DIR, SA.ACT_DIR):
            self.assertTrue(all(sealed.hash_only(d).values()), d)
            for f in d.iterdir():
                if f.name != sealed.ACCESS_LOG:
                    self.assertFalse(os.stat(f).st_mode & stat.S_IWRITE, f)

    def test_every_access_is_authorised(self):
        for r in SA.access_log(SA.POOL_DIR):
            if r["purpose"].startswith("D3-TEST-CAPTURE:"):
                self.assertEqual(r["file"], "model_inputs.jsonl", r)
                self.assertIn(r["approval"], self.capture_ids, r)
            else:
                self.assertTrue(r["purpose"].startswith("H1-TEST-EVAL:"), r)
                self.assertIn(r["file"], {"worlds.jsonl", "model_inputs.jsonl"}, r)
                self.assertIn(r["approval"], self.eval_ids, r)
                self.assertGreaterEqual(r["utc"], self.started, r)
        for r in SA.access_log(SA.ACT_DIR):
            self.assertTrue(r["purpose"].startswith("H1-TEST-EVAL:"), r)
            self.assertIn(r["approval"], self.eval_ids, r)
            self.assertGreaterEqual(r["utc"], self.started, r)

    def test_logs_append_only_since_closure(self):
        snap = json.loads((ARCHIVE / "ARCHIVE_MANIFEST.json").read_text(encoding="utf-8"))["access_log_snapshots"]
        for key, d in (("LOCKED-TEST-D5", SA.POOL_DIR), ("LOCKED-TEST-D3-ACT", SA.ACT_DIR)):
            old = (ARCHIVE / snap[key]["copy"]).read_bytes()
            self.assertTrue(_log_bytes(d).startswith(old), f"{key}: access log rewritten or truncated")

    def test_single_evaluation_hashes(self):
        import hashlib
        files = self.frz["artifacts"]["H1-TEST-EVAL"]["files"]
        for rel in ("outputs/h1_test/EVAL_RESULT.json", "outputs/h1_test/per_world.jsonl", "outputs/h1_test/EVAL_STARTED.json"):
            self.assertEqual(hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), files[rel], rel)


if __name__ == "__main__":
    unittest.main()
