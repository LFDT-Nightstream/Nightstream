from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/check_spec.py"
SPEC = ROOT / "SPEC.md"


def check(text):
    with TemporaryDirectory() as directory:
        spec = Path(directory) / "SPEC.md"
        spec.write_text(text, encoding="utf-8")
        return subprocess.run([sys.executable, "-B", str(SCRIPT), "--spec", str(spec)],
                              capture_output=True, text=True)


class CheckSpecTests(unittest.TestCase):
    def test_committed_spec_passes(self):
        result = check(SPEC.read_text(encoding="utf-8"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_dead_path_fails(self):
        result = check(SPEC.read_text(encoding="utf-8") + "\nSee `crates/no-such-crate/`.\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("`crates/no-such-crate/`", result.stdout)

    def test_hand_edited_table_fails(self):
        text = SPEC.read_text(encoding="utf-8")
        row = "| `productionProfile.global.expansionT` | `T` | 216 |"
        self.assertIn(row, text)
        result = check(text.replace(row, row.replace("216", "217")))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("parameters table is stale", result.stdout)

    def test_malformed_pin_fails(self):
        result = check(SPEC.read_text(encoding="utf-8") + "\n`Spec.production_parameter_values` (`12ab`)\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not 16 hex digits", result.stdout)


if __name__ == "__main__":
    unittest.main()
