"""Static checks of the (unexecuted) H1c DEV-extension capture script."""

import importlib.util
import pathlib
import shutil
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
P = ROOT / "src" / "h1c_devx_capture.py"
SRC = P.read_text(encoding="utf-8")
DEV_SRC = (ROOT / "src" / "d3_run_gemma.py").read_text(encoding="utf-8")


@unittest.skipUnless(importlib.util.find_spec("modal"), "modal not installed")
class H1cDevxCapture(unittest.TestCase):
    def test_constants_mount_and_limits(self):
        import modal.cli.import_refs as ir
        mod = ir.import_file_or_module(ir.parse_import_ref(str(P)))
        self.assertEqual((mod.TIMEOUT_S, mod.GUARD_S, mod.N_WORLDS, mod.P_SLOTS, mod.LAYERS), (950, 900, 3000, 13, (32, 40, 48)))
        with tempfile.TemporaryDirectory() as t:
            q = pathlib.Path(t) / "h1c_devx_mount_copy.py"
            shutil.copy(P, q)
            self.assertIsNone(ir.import_file_or_module(ir.parse_import_ref(str(q))).ROOT)
        for needle in ("gpu=GPU, cpu=CPU, memory=MEMORY_MIB, retries=0, max_containers=1, timeout=TIMEOUT_S",
                       "with modal.Volume.ephemeral() as vol:", "verify_xshards(out", "from h1c_gates import gates"):
            self.assertIn(needle, SRC)
        self.assertIn(DEV_SRC[DEV_SRC.index(".pip_install("):DEV_SRC.index(".add_local_python_source")], SRC)
        self.assertNotIn("locked_test", SRC)


if __name__ == "__main__":
    unittest.main()
