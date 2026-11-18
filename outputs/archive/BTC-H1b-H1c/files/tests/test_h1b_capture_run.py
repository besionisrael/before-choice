"""Static checks of the (unexecuted) LOCKED-TEST-H1b capture script."""

import importlib.util
import pathlib
import shutil
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
P = ROOT / "src" / "sealed_stage" / "h1b_capture_test.py"
SRC = P.read_text(encoding="utf-8")
DEV_SRC = (ROOT / "src" / "d3_run_gemma.py").read_text(encoding="utf-8")


@unittest.skipUnless(importlib.util.find_spec("modal"), "modal not installed")
class H1bCapture(unittest.TestCase):
    def test_constants_and_mount_import(self):
        import modal.cli.import_refs as ir
        mod = ir.import_file_or_module(ir.parse_import_ref(str(P)))
        self.assertEqual((mod.TIMEOUT_S, mod.GUARD_S, mod.N_REPLAY, mod.N_WORLDS, mod.LAYERS, mod.STAGE),
                         (1500, 1420, 20, 4000, (32, 40, 48), "H1B-CAPTURE"))
        self.assertEqual(mod.SLOTS, (0, 1, 3, 5, 7, 9, 11))
        with tempfile.TemporaryDirectory() as t:
            q = pathlib.Path(t) / "h1b_capture_mount_copy.py"
            shutil.copy(P, q)
            self.assertIsNone(ir.import_file_or_module(ir.parse_import_ref(str(q))).ROOT)

    def test_limits_prompts_only_seal(self):
        for needle in ("gpu=GPU, cpu=CPU, memory=MEMORY_MIB, retries=0, max_containers=1, timeout=TIMEOUT_S",
                       "with modal.Volume.ephemeral() as vol:", "verify_xshards(out", "sealed.seal_dir(out, SA.ACT_SEAL)",
                       'SA.staged_read(STAGE, SA.POOL_DIR, "model_inputs.jsonl"'):
            self.assertIn(needle, SRC)
        pin = DEV_SRC[DEV_SRC.index(".pip_install("):DEV_SRC.index(".add_local_python_source")]
        self.assertIn(pin, SRC)
        self.assertNotIn("worlds.jsonl", SRC)


if __name__ == "__main__":
    unittest.main()
