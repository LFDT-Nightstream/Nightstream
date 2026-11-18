import importlib.util
from pathlib import Path
import re
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
        row = "| `productionProfile.global.expansionT` | `T` | 128 |"
        self.assertIn(row, text)
        result = check(text.replace(row, row.replace("128", "129")))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("parameters table is stale", result.stdout)

    def test_malformed_pin_fails(self):
        result = check(SPEC.read_text(encoding="utf-8") + "\n`Spec.production_parameter_values` (`12ab`)\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not 16 hex digits", result.stdout)

    def test_unpinned_claim_fails(self):
        text = re.sub(r"(`Claims\.Foundations\.chain_norm`) \(`[0-9a-f]{16}`\)", r"\1", SPEC.read_text(encoding="utf-8"))
        result = check(text)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("claim `Claims.Foundations.chain_norm` has no pin", result.stdout)

    def test_unlisted_paper_number_fails(self):
        result = check(SPEC.read_text(encoding="utf-8") + "\nHyperNova Lemma 99 has no row.\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cites HyperNova Lemma 99, which is not a row of the papers table", result.stdout)

    def test_citation_takes_the_last_paper_of_its_sentence(self):
        text = "SuperNeo v1.2 Theorems 12 and 13, Lemmas 1, 4 and 10, and Cyclo, ePrint 2026/359, Theorem 3. Theorem 5 names no paper."
        self.assertEqual(list(CHECK_SPEC.citations(text)), [
            ("SuperNeo", "Theorem 12"), ("SuperNeo", "Theorem 13"), ("SuperNeo", "Lemma 1"), ("SuperNeo", "Lemma 4"),
            ("SuperNeo", "Lemma 10"), ("ePrint 2026/359", "Theorem 3")])

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
