import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile


SPEC = importlib.util.spec_from_file_location(
    "generate_lean_mutations", Path(__file__).with_name("generate_lean_mutations.py"))
mutations = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mutations)
ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT / "docs/reviews/nightstream-fprime-requirements/NIFS_DEC_AND_FINAL_OUTPUT_EVIDENCE.zip"


class LeanMutationGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.ccs = self.directory / "pi_ccs_input.json"
        self.children = self.directory / "children.json"
        with ZipFile(ARCHIVE) as archive:
            self.ccs.write_bytes(archive.read("fixtures/recursive_phase_input.json"))
            self.children.write_bytes(archive.read("fixtures/dec_running.json"))

    def test_every_case_matches_retained_archive_and_sources_stay_unchanged(self):
        originals = self.ccs.read_bytes(), self.children.read_bytes()
        output = self.directory / "generated"
        manifest = mutations.generate(self.ccs, self.children, output)
        with ZipFile(ARCHIVE) as archive:
            expected = json.loads(archive.read("fixtures/dec_mutation_manifest.json"))
            self.assertEqual(
                {case["case"]: case["expected_owner"] for case in manifest["cases"][:-1]},
                {case["name"]: case["expected_owner"] for case in expected},
            )
            for case in expected:
                name = case["name"] + ".json"
                self.assertEqual((output / "mutations" / name).read_bytes(),
                                 archive.read("fixtures/dec_mutations/" + name), name)
            self.assertEqual((output / manifest["changed_ccs_input"]).read_bytes(),
                             archive.read("fixtures/invalid_first_round_constant.json"))
        self.assertEqual(originals, (self.ccs.read_bytes(), self.children.read_bytes()))
        self.assertEqual(json.loads((output / "manifest.json").read_text()), manifest)
        self.assertEqual(manifest["cases"][-1]["expected_owner"], "upstream_pi_ccs")
        with self.assertRaises(FileExistsError):
            mutations.generate(self.ccs, self.children, output)

    def test_changed_source_words_are_used_and_field_increment_wraps(self):
        ccs, children = json.loads(self.ccs.read_text()), json.loads(self.children.read_text())
        ccs[3][0][0][0] = mutations.MODULUS - 1
        children[1][0][0] = mutations.MODULUS - 1
        self.ccs.write_text(mutations.numeric_json(ccs))
        self.children.write_text(mutations.numeric_json(children))
        originals = self.ccs.read_bytes(), self.children.read_bytes()
        output = self.directory / "fresh"
        manifest = mutations.generate(self.ccs, self.children, output)
        changed = json.loads((output / "mutations/public_child_0_commitment.json").read_text())
        self.assertEqual(changed[1][0][0], 0)
        changed[1][0][0] = mutations.MODULUS - 1
        self.assertEqual(changed, children)
        changed = json.loads((output / manifest["changed_ccs_input"]).read_text())
        self.assertEqual(changed[3][0][0][0], 0)
        changed[3][0][0][0] = mutations.MODULUS - 1
        self.assertEqual(changed, ccs)
        self.assertEqual(originals, (self.ccs.read_bytes(), self.children.read_bytes()))

    def test_invalid_source_shape_fails_before_output_creation(self):
        children = json.loads(self.children.read_text())
        children[4][0].pop()
        self.children.write_text(mutations.numeric_json(children))
        output = self.directory / "invalid"
        with self.assertRaisesRegex(ValueError, "expected vector width 14"):
            mutations.generate(self.ccs, self.children, output)
        self.assertFalse(output.exists())

    def test_numeric_input_accepts_only_one_optional_lf(self):
        path = self.directory / "numeric.json"
        for text in (b"[1]", b"[1]\n"):
            path.write_bytes(text)
            self.assertEqual(mutations.read_numeric(path), [1])
        for text in (b"[1]\r\n", b"[1]\n\n", b"[ 1]", b"[1e0]"):
            with self.subTest(text=text):
                path.write_bytes(text)
                with self.assertRaises(ValueError):
                    mutations.read_numeric(path)


if __name__ == "__main__":
    unittest.main()
