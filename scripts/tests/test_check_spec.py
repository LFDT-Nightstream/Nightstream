import importlib.util
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/check_spec.py"
SPEC = ROOT / "SPEC.md"
LOADER = importlib.util.spec_from_file_location("check_spec", SCRIPT)
CHECK_SPEC = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(CHECK_SPEC)


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

    def test_specs_folder_fails(self):
        errors = list(CHECK_SPEC.layout_errors(["specs/nebula-superneo.md", "specs/vectors.json"]))
        self.assertEqual(len(errors), 2, errors)
        self.assertIn("`specs/nebula-superneo.md` is in a `specs/` folder", errors[0])

    def test_separate_spec_file_fails(self):
        errors = list(CHECK_SPEC.layout_errors(["formal/nightstream-fprime/NEBULA_SPEC.md"]))
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("is a separate spec file", errors[0])

    def test_known_documents_pass(self):
        paths = [*CHECK_SPEC.SPEC_FILES, "formal/nightstream-fprime/SECURITY_MODEL.md",
                 "decisions/nebula-memory-checking.md", "formal/nightstream-fprime/NightstreamFPrime/Spec/Nebula.lean"]
        self.assertEqual(list(CHECK_SPEC.layout_errors(paths)), [])

    def test_old_citation_fails(self):
        files = [("crates/a.rs", "/// Row O4.\n/// See spec §8.1.\n"), ("b.lean", "/-- SPEC.md section 7.3. -/")]
        errors = list(CHECK_SPEC.citation_errors(files))
        self.assertEqual(errors, ["crates/a.rs:2 cites `spec §`; cite a SPEC.md section or algorithm, or a Lean name"])


if __name__ == "__main__":
    unittest.main()
