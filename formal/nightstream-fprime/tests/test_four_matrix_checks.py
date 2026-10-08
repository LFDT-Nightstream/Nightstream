"""Exercise the current-profile comparison commands with retained golden data."""

import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile


TESTS = Path(__file__).resolve().parent
VECTORS = TESTS.parents[2] / "crates/nightstream/tests/fixtures/golden-v1.zip"


class FourMatrixChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with ZipFile(VECTORS) as vectors:
            actual = json.loads(vectors.read("native/fold-1/actual_result.json"))
            cls.proof = actual["pi_ccs_input"]
            cls.phase = actual["pi_ccs_phase"]
            cls.children = json.loads(vectors.read("native/fold-1/children.json"))
            cls.result = json.loads(vectors.read("expected/step-1-nifs.json"))
            cls.caller = json.loads(vectors.read("expected/step-1-caller.json"))

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.public = self.write("public.json", [self.proof[1], self.proof[2], self.proof[6]])
        self.proof_path = self.write("proof.json", self.proof)
        self.phase_path = self.write("phase.json", self.phase)
        # This tests comparison framing, not independent round production.
        self.round_path = self.write("round-27.json", [
            1, self.phase[1], self.phase[2], self.phase[5][26], self.proof[3][27],
            self.phase[4][27], self.phase[5][27], self.phase[8][26],
            self.phase[8][26], self.phase[8][27],
        ])

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value, separators=(",", ":")) + "\n")
        return path

    def run_check(self, script, *paths, error=None):
        result = subprocess.run(
            [sys.executable, "-B", str(TESTS / script), *map(str, paths)],
            capture_output=True, text=True, timeout=300,
        )
        if error is not None:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(error, result.stderr)
            return None
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def test_complete_evaluations_accept_current_vectors_and_reject_changed_last_matrix(self):
        merged = [1, self.phase[4], self.proof[4], copy.deepcopy(self.proof[5])]
        path = self.write("evaluations.json", merged)
        args = (self.public, self.round_path, path, self.proof_path, self.phase_path)
        report = self.run_check("check_piccs_original_complete.py", *args)
        self.assertEqual(report["matrix_K_values"], 3672)
        self.assertEqual(report["compared_K_values"], 4590)
        self.assertEqual(report["compared_field_words"], 9180)
        self.assertEqual(report["fresh_source_nonconstant_K_values"], 265)
        merged[3][16][3][53][1] = (merged[3][16][3][53][1] + 1) % (2**64 - 2**32 + 1)
        self.write(path.name, merged)
        self.run_check("check_piccs_original_complete.py", *args,
                       error="matrix[16][3][53][1] differs")

    def test_complete_evaluations_reject_a_missing_matrix_even_when_values_match(self):
        proof, phase = copy.deepcopy(self.proof), copy.deepcopy(self.phase)
        proof[5][16].pop()
        phase[13] = proof[5]
        self.write(self.proof_path.name, proof)
        self.write(self.phase_path.name, phase)
        merged = self.write("evaluations.json", [1, phase[4], proof[4], proof[5]])
        self.run_check("check_piccs_original_complete.py", self.public, self.round_path,
                       merged, self.proof_path, self.phase_path,
                       error="Lean matrix[16]: expected array length 4")

    def test_terminal_prefix_accepts_four_matrices_and_checks_both_mutations(self):
        def prefix(directory, kind, values):
            directory.mkdir(parents=True)
            (directory / "manifest.json").write_text(json.dumps(
                [1, 28, kind, len(values), 1, self.phase[4], [[0, 1]]]))
            (directory / "0-1.bin").write_bytes(
                b"".join(struct.pack("<QQ", *value) for value in values))

        fresh, norm = self.root / "fresh", self.root / "norm"
        prefix(fresh, 2, [matrix[0] for matrix in self.proof[5][0]])
        for source, coefficients in enumerate(self.proof[4]):
            prefix(norm / f"source-{source}", 3 + source, [coefficients[0]])
        report = self.run_check("check_piccs_terminal_prefix.py", self.public, self.round_path,
                                fresh, norm, self.proof_path, self.phase_path)
        self.assertEqual(report["fresh_matrices"], 4)
        self.assertEqual(report["compared_K_values"], 21)
        self.assertEqual(report["compared_field_words"], 42)
        self.assertEqual(report["canonical_field_bytes"], 336)
        self.assertEqual(report["norm_target_mutation"], "rejected")
        self.assertEqual(report["fresh_target_mutation"], "rejected")

    def test_nifs_caller_accepts_current_width_and_rejects_truncation(self):
        paths = [self.write(name, value) for name, value in (
            ("children.json", self.children), ("nifs.json", self.result),
            ("caller.json", self.caller),
        )]
        report = self.run_check("check_independent_nifs_bytes.py", *paths, *paths)
        self.assertEqual(report["caller_private_words"], 107070)
        self.assertEqual(report["changed_target"], "rejected")
        caller = copy.deepcopy(self.caller)
        caller[2].pop()
        self.write(paths[2].name, caller)
        self.run_check("check_independent_nifs_bytes.py", *paths, *paths,
                       error="wrong selected caller width")


if __name__ == "__main__":
    unittest.main()
