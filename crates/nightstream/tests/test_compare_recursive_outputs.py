import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "compare_recursive_outputs", Path(__file__).with_name("compare_recursive_outputs.py"))
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


class RecursiveComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.cpu, self.engine, self.reference = (root / name for name in ("cpu", "engine", "reference"))
        self.receipt = self.cpu / "comparison-fold-2.json"
        parent = {
            "structural_identifier": [3], "package_identity": [2],
            "transcript_state": [{"value": 4}], "transcript_absorbed": 0,
        }
        envelope = {
            "schema": 1, "package_identity": [2], "iteration": 3, "z0": [0],
            "current": [13], "child_witness_count": 16,
            "running_claims": list(range(16)), "running_parent": [3],
        }
        self.write(self.cpu / "step-2/envelope.json", dict(envelope, iteration=2))
        self.write(self.cpu / "step-3/envelope.json", envelope)
        archived = {key: envelope[key] for key in compare.ENVELOPE_FIELDS}
        archived.update({key: "record metadata" for key in compare.REFERENCE_METADATA})
        self.write(self.reference / "output/envelope/envelope.json", archived)
        for step in (2, 3):
            self.write(self.cpu / f"step-{step}/fresh-witness.json", [[step]])
            self.write(self.cpu / f"step-{step}/fresh-claim.json", {"x": [{"value": 12}]})
        for name in ("fresh-witness.json", "fresh-claim.json"):
            self.write(self.reference / "output/envelope" / name, compare.load(self.cpu / "step-3" / name))
        for child in range(16):
            for directory in (self.cpu / "step-2", self.cpu / "fold-2",
                              self.reference / "output/material"):
                self.write(directory / f"digit-{child}.json", [[child]])
        self.write(self.cpu / "fold-2/parent.json", parent)
        self.write(self.cpu / "fold-2/nifs.json", {"parent": parent, "children": [9]})
        (self.cpu / "fold-2/proof.native").write_bytes(b"complete canonical proof")
        native = {
            "schema": 1, "structural_identifier": [3], "package_identity": [2],
            "pi_ccs_input": [5], "pi_ccs_phase": [1], "pi_rlc_parent": [4, 5, 6, 7, 8, 1],
            "children": [9], "outgoing_state": [4], "absorbed": 0,
        }
        lean = [1, [5], [], [], [], [1], [0] * 6 + [[2]],
                [1, 0, 0, 4, 5, 6, 7, 8, 0, [4]], [],
                [1] + [0] * 13 + [[4], 0, [1, [9]]]]
        directory = self.reference / "nonzero-nifs"
        self.write(directory / "actual_result.json", native)
        self.write(directory / "nightstream-native-nonzero-nifs-1.lean.json", lean)
        self.write(directory / "pi_ccs_input.json", native["pi_ccs_input"])
        self.write(directory / "children.json", native["children"])
        (directory / "proof.bin").write_bytes(b"complete canonical proof")
        self.write(self.reference / "nonzero-successor/nightstream-native-nonzero-step-1.json",
                   [1, [], [10], [11], [[13], [], [12]]])
        shutil.copytree(self.cpu, self.engine)
        # The complete engine producer returns fold outputs, without a caller.
        shutil.rmtree(self.engine / "step-3")

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def run_comparison(self, engine=True, receipt=None):
        argv = ["compare_recursive_outputs.py", "--directory", str(self.cpu), "--fold", "2",
                "--reference", str(self.reference)]
        if engine:
            argv += ["--engine-directory", str(self.engine)]
        if receipt:
            argv += ["--receipt", str(receipt)]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            return compare.main()

    def test_one_receipt_connects_archive_cpu_and_engine_proof_bytes(self):
        self.assertEqual(self.run_comparison(), 0)
        later = compare.load(self.receipt)["later_fold"]
        engine = later["engine_comparisons"][0]
        self.assertEqual(later["proof"]["matched_sha256"], engine["proof"]["matched_sha256"])
        self.assertEqual(later["proof"]["actual"], engine["proof"]["reference"])
        self.assertEqual(len(engine["files"]), 37)
        self.assertIn("No engine successor caller", engine["scope"])

    def test_cpu_engine_agreement_does_not_hide_archive_mismatch(self):
        for directory in (self.cpu, self.engine):
            (directory / "fold-2/proof.native").write_bytes(b"a different complete proof")
        with self.assertRaisesRegex(ValueError, "every later proof byte"):
            self.run_comparison()
        self.assertFalse(self.receipt.exists())

    def test_new_engine_receipt_preserves_previous_cpu_comparison(self):
        self.run_comparison(engine=False)
        previous = self.receipt.read_bytes()
        receipt = self.engine / "lean-cpu-engine.json"
        self.run_comparison(receipt=receipt)
        self.assertEqual(self.receipt.read_bytes(), previous)
        self.assertEqual(len(compare.load(receipt)["later_fold"]["engine_comparisons"]), 1)

    def test_engine_must_match_the_checked_cpu_proof(self):
        (self.engine / "fold-2/proof.native").write_bytes(b"another engine proof")
        with self.assertRaisesRegex(ValueError, "every engine/CPU proof byte"):
            self.run_comparison()
        self.assertFalse(self.receipt.exists())

    def test_cpu_engine_witness_agreement_does_not_hide_archive_mismatch(self):
        for directory in (self.cpu, self.engine):
            self.write(directory / "fold-2/digit-0.json", [[42]])
        with self.assertRaisesRegex(ValueError, "complete child 0 matrix"):
            self.run_comparison()
        self.assertFalse(self.receipt.exists())

    def test_matching_proof_does_not_hide_different_engine_inputs(self):
        self.write(self.engine / "step-2/fresh-witness.json", [[42]])
        with self.assertRaisesRegex(ValueError, "complete fold input fresh-witness.json"):
            self.run_comparison()
        self.assertFalse(self.receipt.exists())

    def test_matching_proof_does_not_hide_different_returned_claims(self):
        nifs = compare.load(self.engine / "fold-2/nifs.json")
        nifs["children"] = [42]
        self.write(self.engine / "fold-2/nifs.json", nifs)
        with self.assertRaisesRegex(ValueError, "complete fold output nifs.json.children"):
            self.run_comparison()
        self.assertFalse(self.receipt.exists())

    def prepare_fold_three(self):
        shutil.copytree(self.cpu / "step-2", self.cpu / "step-3", dirs_exist_ok=True)
        shutil.copytree(self.cpu / "fold-2", self.cpu / "fold-3")
        shutil.copytree(self.cpu / "step-3", self.cpu / "step-4")
        self.write(self.cpu / "fold-3/caller-inputs.json", {"private_values": [1], "public_values": [2]})
        (self.cpu / "fold-3/physical.bin").write_bytes(b"complete physical assignment")
        for name in ("step-3", "fold-3", "step-4"):
            shutil.copytree(self.cpu / name, self.engine / name)
        lean_proof = self.reference / "fresh-lean-proof.bin"
        lean_proof.write_bytes((self.cpu / "fold-3/proof.native").read_bytes())
        return lean_proof

    def run_fold_three(self, lean_proof=None):
        argv = ["compare_recursive_outputs.py", "--directory", str(self.cpu), "--fold", "3",
                "--engine-directory", str(self.engine)]
        if lean_proof:
            argv += ["--lean-proof", str(lean_proof)]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            return compare.main()

    def test_fold_three_compares_complete_successor_and_explicit_lean_bytes(self):
        lean_proof = self.prepare_fold_three()
        self.assertEqual(self.run_fold_three(lean_proof), 0)
        receipt = compare.load(self.cpu / "comparison-fold-3.json")
        self.assertIsNone(receipt["reference_directory"])
        later = receipt["later_fold"]
        self.assertEqual(len(later["engine_comparisons"][0]["files"]), 42)
        self.assertEqual(later["proof"]["matched_sha256"], later["engine_comparisons"][0]["proof"]["matched_sha256"])
        self.assertIn("separately established independent generation", later["lean_proof_scope"])

    def test_fold_three_engine_comparison_alone_makes_no_lean_claim(self):
        self.prepare_fold_three()
        self.assertEqual(self.run_fold_three(), 0)
        receipt = compare.load(self.cpu / "comparison-fold-3.json")
        self.assertNotIn("proof", receipt["later_fold"])
        self.assertNotIn("lean_proof_scope", receipt["later_fold"])
        self.assertIn("No Lean provenance", receipt["scope"])

    def test_fold_three_cpu_engine_agreement_cannot_hide_lean_mismatch(self):
        lean_proof = self.prepare_fold_three()
        for directory in (self.cpu, self.engine):
            (directory / "fold-3/proof.native").write_bytes(b"other matching CPU and engine proof")
        with self.assertRaisesRegex(ValueError, "every fresh Lean-field/CPU proof byte"):
            self.run_fold_three(lean_proof)
        self.assertFalse((self.cpu / "comparison-fold-3.json").exists())

    def test_fold_three_rejects_changed_caller_and_physical_assignment(self):
        self.prepare_fold_three()
        caller = self.engine / "fold-3/caller-inputs.json"
        original = caller.read_bytes()
        self.write(caller, {"private_values": [42], "public_values": [2]})
        with self.assertRaisesRegex(ValueError, "complete successor caller.private_values"):
            self.run_fold_three()
        caller.write_bytes(original)
        (self.engine / "fold-3/physical.bin").write_bytes(b"changed assignment")
        with self.assertRaisesRegex(ValueError, "every physical assignment byte"):
            self.run_fold_three()
        self.assertFalse((self.cpu / "comparison-fold-3.json").exists())


if __name__ == "__main__":
    unittest.main()
