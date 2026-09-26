import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile


SPEC = importlib.util.spec_from_file_location(
    "check_golden_vectors", Path(__file__).with_name("check_golden_vectors.py"))
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)


class GoldenVectorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_retained_archive_has_both_folds_and_no_physical_witnesses(self):
        archive = Path(__file__).parent / "fixtures/golden-wide-v1.zip"
        check.extract_vectors(archive, self.root)
        for step in (1, 2):
            self.assertTrue((self.root / f"native/fold-{step}/proof.native").is_file())
            self.assertTrue((self.root / f"expected/step-{step}-nifs.json").is_file())
        self.assertFalse(list(self.root.rglob("physical.bin")))

    def test_archive_cannot_write_unlisted_paths(self):
        archive = self.root / "wrong.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("../escaped.json", "[]")
        with self.assertRaisesRegex(ValueError, "exactly the two-fold"):
            check.extract_vectors(archive, self.root / "output")
        self.assertFalse((self.root / "escaped.json").exists())

    def test_changed_expected_value_cannot_pass_after_verifier_acceptance(self):
        actual, expected = self.root / "actual", self.root / "expected"
        actual.mkdir()
        expected.mkdir()
        for kind in ("nifs", "caller"):
            name = f"step-2-{kind}.json"
            (actual / name).write_text("[1,2,3]\n")
            (expected / name).write_text("[1,2,3]\n")
        check.compare_expected(actual, expected, 2)
        (expected / "step-2-nifs.json").write_text("[1,2,4]\n")
        with self.assertRaisesRegex(ValueError, "retained Lean nifs vector for fold 2"):
            check.compare_expected(actual, expected, 2)


if __name__ == "__main__":
    unittest.main()
