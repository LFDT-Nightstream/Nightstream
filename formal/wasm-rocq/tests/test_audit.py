import contextlib
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_audit


class AuditTests(unittest.TestCase):
    def test_names_and_wrapped_types(self):
        report = "theorem\n  : True\nAxioms:\nB.y :\n  forall x, x = x\nA.x : True\n"
        self.assertEqual(check_audit.axiom_names(report), {"A.x", "B.y"})

    def test_closed_context(self):
        self.assertEqual(check_audit.axiom_names("Closed under the global context\n"), set())

    def test_unrecognized_reports_fail(self):
        for report in (
            "", "theorem : True\n", "Axioms:\n", "Axioms:\n  orphan type\n",
            "Axioms:\nA.x :\n", "Axioms:\nA.x : True\nUnexpected output\n",
            "Axioms:\nA.x :\nB.y : True\n", "Axioms:\nA.x : True\nA.x : True\n",
            "Closed under the global context\nUnexpected output\n",
            "Closed under the global context\nAxioms:\nA.x : True\n",
        ):
            with self.subTest(report=report), self.assertRaises(ValueError):
                check_audit.axiom_names(report)

    def test_command_status_and_differences(self):
        for output, expected, rocq_status, status, message in (
            ("Axioms:\nB.y : True\nA.x : True\n", "A.x\nB.y\n", 0, 0, "Audit passed"),
            ("Axioms:\nA.x : True\nB.y : True\n", "A.x\n", 0, 1, "Added assumption: B.y"),
            ("Axioms:\nA.x : True\n", "A.x\nB.y\n", 0, 1, "Removed assumption: B.y"),
            ("Closed under the global context\n", "", 0, 0, "0 expected assumptions"),
            ("Closed under the global context\n", "A.x\n", 0, 1, "Removed assumption: A.x"),
            ("unrecognized\n", "A.x\n", 0, 1, "Audit failed"),
            ("Axioms:\nA.x : True\n", "A.x\nA.x\n", 0, 1, "duplicate name"),
            ("Axioms:\nA.x : True\n", "not a name\n", 0, 1, "one axiom name"),
            ("Axioms:\nA.x : True\n", "A.x\n", 1, 1, "Rocq error"),
        ):
            with self.subTest(output=output, expected=expected, rocq_status=rocq_status):
                captured = io.StringIO()
                result = subprocess.CompletedProcess([], rocq_status, output,
                                                     "Rocq error" if rocq_status else "")
                with patch.object(check_audit.subprocess, "run", return_value=result), \
                        patch.object(Path, "read_text", return_value=expected), \
                        contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                    self.assertEqual(check_audit.main(), status)
                self.assertIn(message, captured.getvalue())
                self.assertIn(output, captured.getvalue())


if __name__ == "__main__":
    unittest.main()
