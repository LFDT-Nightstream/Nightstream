import contextlib
import importlib.util
import io
import hashlib
import json
from pathlib import Path
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
        root = Path(self.temporary.name).resolve()
        self.cpu, self.reference = (root / name for name in ("cpu", "reference"))
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

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def run_comparison(self, receipt=None):
        argv = ["compare_recursive_outputs.py", "--directory", str(self.cpu), "--fold", "2",
                "--reference", str(self.reference)]
        if receipt:
            argv += ["--receipt", str(receipt)]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            return compare.main()

    def test_complete_archive_cpu_proof_comparison_records_its_identity(self):
        self.assertEqual(self.run_comparison(), 0)
        proof = compare.load(self.receipt)["later_fold"]["proof"]
        self.assertEqual(proof["matched_sha256"], hashlib.sha256(b"complete canonical proof").hexdigest())
        self.assertEqual(proof["comparison"], "exact bytes")

    def test_changed_cpu_proof_cannot_match_archive(self):
        (self.cpu / "fold-2/proof.native").write_bytes(b"a different complete proof")
        with self.assertRaisesRegex(ValueError, "every later proof byte"):
            self.run_comparison()
        self.assertFalse(self.receipt.exists())

    def test_new_receipt_preserves_previous_comparison(self):
        self.run_comparison()
        previous = self.receipt.read_bytes()
        self.assertEqual(self.run_comparison(self.cpu / "another-comparison.json"), 0)
        self.assertEqual(self.receipt.read_bytes(), previous)

    def test_matching_proof_cannot_hide_changed_child_witness(self):
        self.write(self.cpu / "fold-2/digit-15.json", [[42]])
        with self.assertRaisesRegex(ValueError, "complete child 15 matrix"):
            self.run_comparison()
        self.assertFalse(self.receipt.exists())

    def test_matching_proof_cannot_hide_changed_fresh_witness(self):
        self.write(self.cpu / "step-3/fresh-witness.json", [[42]])
        with self.assertRaisesRegex(ValueError, "complete fresh-witness.json"):
            self.run_comparison()
        self.assertFalse(self.receipt.exists())


if __name__ == "__main__":
    unittest.main()
