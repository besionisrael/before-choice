"""Static checks of the (unexecuted) BTC-H1b X3+X4 Modal capture script."""

import importlib.util
import pathlib
import shutil
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
P = ROOT / "src" / "h1b_x3x4_run_gemma.py"
SRC = P.read_text(encoding="utf-8")
DEV_SRC = (ROOT / "src" / "d3_run_gemma.py").read_text(encoding="utf-8")


@unittest.skipUnless(importlib.util.find_spec("modal"), "modal not installed")
class X3X4Run(unittest.TestCase):
    def test_imports_and_constants(self):
        import modal.cli.import_refs as ir
        mod = ir.import_file_or_module(ir.parse_import_ref(str(P)))
        self.assertEqual(mod.app.name, "btc-h1b-x3x4-gemma4")
        self.assertEqual((mod.TIMEOUT_S, mod.GUARD_S, mod.N_REPLAY, mod.N_WORLDS), (1500, 1420, 20, 3480))
        self.assertEqual(mod.LAYERS, (0, 8, 16, 24, 32, 40, 48, 56, 61))

    def test_container_mount_import(self):
        import modal.cli.import_refs as ir
        with tempfile.TemporaryDirectory() as t:
            q = pathlib.Path(t) / "h1b_x3x4_mount_copy.py"
            shutil.copy(P, q)
            self.assertIsNone(ir.import_file_or_module(ir.parse_import_ref(str(q))).ROOT)

    def test_limits_image_transfer_and_no_sealed_path(self):
        for needle in ("gpu=GPU, cpu=CPU, memory=MEMORY_MIB, retries=0, max_containers=1, timeout=TIMEOUT_S",
                       "scaledown_window=2", "with modal.Volume.ephemeral() as vol:", '"max_mem_gb"'):
            self.assertIn(needle, SRC)
        pin = DEV_SRC[DEV_SRC.index(".pip_install("):DEV_SRC.index(".add_local_python_source")]
        self.assertIn(pin, SRC)
        self.assertNotIn("locked_test", SRC)
        self.assertNotIn("worlds.jsonl", SRC)
        blk = SRC[SRC.index("with modal.Volume.ephemeral() as vol:"):SRC.index("replication_check(out)\n    (out")]
        for needle in ("TRANSFER FAILURE", "verify_xshards(out", "INTEGRITY FAILURE"):
            self.assertIn(needle, blk)


if __name__ == "__main__":
    unittest.main()
