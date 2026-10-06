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
# The retained archive records the case table of the older 14-matrix format.
ARCHIVE = ROOT / "docs/reviews/nightstream-fprime-requirements/NIFS_DEC_AND_FINAL_OUTPUT_EVIDENCE.zip"
VECTORS = Path(__file__).with_name("fixtures") / "golden-v1.zip"


class LeanMutationGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.ccs = self.directory / "pi_ccs_input.json"
        self.children = self.directory / "children.json"
        with ZipFile(VECTORS) as vectors:
            self.ccs.write_bytes(vectors.read("native/fold-1/pi_ccs_input.json"))
            self.children.write_bytes(vectors.read("native/fold-1/children.json"))

    def test_every_case_matches_retained_owners_and_sources_stay_unchanged(self):
        originals = self.ccs.read_bytes(), self.children.read_bytes()
        output = self.directory / "generated"
        manifest = mutations.generate(self.ccs, self.children, output)
        with ZipFile(ARCHIVE) as archive:
            retained = json.loads(archive.read("fixtures/dec_mutation_manifest.json"))
        removed = {f"public_child_eval_A{matrix}" for matrix in range(mutations.MATRICES, 14)}
        self.assertEqual(
            {case["case"]: case["expected_owner"] for case in manifest["cases"][:-1]},
            {case["name"]: case["expected_owner"] for case in retained if case["name"] not in removed},
        )
        children = json.loads(originals[1])
        public_changes = {
            **{f"public_child_{child}_commitment": (1, child, 0) for child in range(16)},
            "public_child_public": (2, 0, 0),
            "public_shared_point": (0, 0, 0),
            "public_child_eval_K": (3, 0, 0, 0),
            **{f"public_child_eval_A{matrix}": (4, 0, matrix, 0, 0) for matrix in range(7)},
            "public_child_digit_range": (2, 0, 0),
        }
        for case in manifest["cases"][:-1]:
            name = case["case"]
            if name not in public_changes:
                continue
            with self.subTest(case=name):
                changed = json.loads((output / case["file"]).read_bytes())
                indices = public_changes[name]
                parent, original = changed, children
                for index in indices[:-1]:
                    parent, original = parent[index], original[index]
                field = indices[-1]
                expected = 2 if name == "public_child_digit_range" else (original[field] + 1) % mutations.MODULUS
                self.assertEqual(parent[field], expected)
                parent[field] = original[field]
                self.assertEqual(changed, children, "mutation changed a different field")
        changed = json.loads((output / manifest["changed_ccs_input"]).read_text())
        source = json.loads(originals[0])
        self.assertEqual(changed[3][0][0][0], (source[3][0][0][0] + 1) % mutations.MODULUS)
        changed[3][0][0][0] = source[3][0][0][0]
        self.assertEqual(changed, source)
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
        with self.assertRaisesRegex(ValueError, "expected vector width 7"):
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
