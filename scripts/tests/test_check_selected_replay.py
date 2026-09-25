import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import check_selected_replay as check


class SelectedReplayTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.first, self.second, self.cpu = (self.root / name for name in ("first", "second", "current"))
        self.source = "formal/nightstream-fprime/Producer.lean"
        self.rename = "formal/nightstream-fprime/scripts/replay_recursive_loop.py"
        self.write(self.root / self.source, "unchanged Lean producer")
        self.write(self.root / self.rename, "cargo run -p neo-fold-legacy")
        self.sources = {self.source: self.digest("unchanged Lean producer"),
                        self.rename: self.digest("cargo run -p neo-fold-legacy")}
        self.write(self.root / "crates/nightstream/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json", {})
        self.write(self.root / "crates/nightstream/tests/fixtures/stage1_recursive_states/nonzero-running.json",
                   [2, [0], [1], [7]])
        for step, root in ((1, self.first), (2, self.second)):
            envelope = {"iteration": step, "z0": [0], "current": [1]}
            original = root / "original-sources"
            self.write(original / "next-message-input.json", [step, [0], [1], [7]])
            for name in ["envelope.json", "fresh-witness.json", "fresh-claim.json"] + [f"digit-{i}.json" for i in range(16)]:
                value = envelope if name == "envelope.json" else {"value": 1}
                self.write(original / name, value)
                self.write(self.cpu / f"cpu/step-{step}" / name, value)
            self.write(root / "original-package.json", {})
            self.write(root / "producer-sources.json", {"commit": "recorded-source", "files": {
                self.source: self.sources[self.source], self.rename: self.digest("cargo run -p neo-fold-clean"),
                "original-sources": check.identity(original),
                "original-package": check.identity(root / "original-package.json")}})
        self.write(self.cpu / "cpu/fold-1/nifs.json", {"parent": {}})
        self.enterContext(patch.object(check, "ROOT", self.root))
        self.enterContext(patch.object(check, "producer_sources", return_value=self.sources))
        self.enterContext(patch.object(check, "compare_envelope", return_value=({}, {})))

    @staticmethod
    def digest(text):
        return hashlib.sha256(text.encode()).hexdigest()

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value if isinstance(value, str) else json.dumps(value))

    def audit(self):
        return check.audit_inputs(self.first, self.second, self.cpu)

    def test_only_comparison_crate_rename_can_reuse_changed_producer_script(self):
        self.assertEqual(len(self.audit()), 2)
        self.write(self.root / self.rename, "changed computation")
        self.sources[self.rename] = self.digest("changed computation")
        with self.assertRaisesRegex(ValueError, "comparison crate rename only"):
            self.audit()

    def test_changed_or_added_lean_definition_requires_new_generation(self):
        self.sources[self.source] = self.digest("changed Lean producer")
        with self.assertRaisesRegex(ValueError, "retained Lean producer changed"):
            self.audit()
        self.sources[self.source] = self.digest("unchanged Lean producer")
        self.sources["formal/nightstream-fprime/Added.lean"] = self.digest("new definition")
        with self.assertRaisesRegex(ValueError, "complete Lean producer source set"):
            self.audit()

    def test_rehashed_retained_input_must_still_equal_current_cpu_input(self):
        self.write(self.first / "original-sources/digit-15.json", {"value": 2})
        path = self.first / "producer-sources.json"
        pin = json.loads(path.read_text())
        pin["files"]["original-sources"] = check.identity(self.first / "original-sources")
        self.write(path, pin)
        with self.assertRaisesRegex(ValueError, "complete original input"):
            self.audit()

    def test_failed_comparison_cannot_leave_a_success_record(self):
        output = self.root / "comparison"
        with patch.object(check, "source_identity", return_value={"commit": "test"}), \
                patch.object(check.subprocess, "run", return_value=SimpleNamespace(returncode=1)), \
                patch.object(check, "build") as build, self.assertRaisesRegex(ValueError, "input-audit failed"):
            check.execute(self.first, self.second, self.cpu, output)
        self.assertEqual(json.loads((output / "result.json").read_text())["outcome"], "failed")
        build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
