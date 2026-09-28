"""Checkpoint integrity and producer-input separation for the independent run."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import generate_independent_folds as runner


class Checkpoints(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.replay = object.__new__(runner.Replay)
        self.replay.directory = self.root
        self.replay.iteration = 2
        self.replay.logs = self.root / "logs"
        self.replay.logs.mkdir()
        self.replay.check = Mock()
        self.output = self.root / "values.bin"
        self.output.write_bytes(b"computed values")
        self.command = ["comparison", str(self.output)]
        runner.write_new(self.replay.logs / "check.command.json", {
            "command": self.command, "cwd": str(runner.REPO), "exit": 0, "outcome": "passed"})
        runner.write_new(self.replay.logs / "check.outputs.json", {
            str(self.output): runner.identity(self.output)})
        runner.write_new(self.replay.logs / "check.inputs.json", {})

    def test_resume_requires_same_complete_output(self):
        self.replay.run("check", "python", self.command, [self.output])
        self.replay.check.phase.assert_not_called()
        self.output.write_bytes(b"mutated values!")
        with self.assertRaisesRegex(ValueError, "changed checkpoint outputs"):
            self.replay.run("check", "python", self.command, [self.output])

    def test_changed_command_and_failed_record_cannot_resume(self):
        with self.assertRaisesRegex(ValueError, "changed checkpoint command"):
            self.replay.run("check", "python", ["different"], [self.output])
        record = self.replay.logs / "check.command.json"
        value = json.loads(record.read_text())
        value["exit"], value["outcome"] = 124, "timed-out"
        record.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "failed checkpoint"):
            self.replay.run("check", "python", self.command, [self.output])

    def test_directory_members_are_part_of_checkpoint(self):
        path = self.root / "blocks"
        path.mkdir()
        (path / "block-0").write_bytes(b"first")
        initial = runner.identity(path)
        (path / "unexpected").write_bytes(b"extra")
        self.assertNotEqual(initial, runner.identity(path))

    def test_changed_original_input_cannot_reuse_result(self):
        source = self.root / "original.bin"
        source.write_bytes(b"original")
        command = self.command + [str(source)]
        path = self.replay.logs / "check.command.json"
        record = json.loads(path.read_text())
        record["command"] = command
        path.write_text(json.dumps(record))
        inputs = runner.command_inputs(command, None, [self.output], runner.REPO)
        (self.replay.logs / "check.inputs.json").write_text(json.dumps(inputs))
        source.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "changed checkpoint inputs"):
            self.replay.run("check", "python", command, [self.output])

    def test_second_fold_reads_only_first_lean_outputs(self):
        self.replay.root = self.root
        self.replay.iteration = 3
        self.replay.public, self.replay.sources = self.root / "public", self.root / "sources"
        self.replay.binary, self.replay.native = self.root / "binary", self.root / "native"
        self.replay.run = Mock()
        self.replay.python = Mock()
        expected = self.root / "step-2-to-3"
        expected.mkdir()
        with self.assertRaisesRegex(ValueError, "first Lean handoff is incomplete"):
            self.replay.prepare()
        files = ("fresh-witness.json", "fresh-claim.json", "children.json", "digits-0.jsonl", "digits-1.jsonl")
        for name in files:
            (expected / name).write_bytes(b"complete checked output")
        runner.write_new(expected / "handoff.json", {
            "from_iteration": 2, "to_iteration": 3,
            "lean_feedback": {name: runner.identity(expected / name) for name in files}})
        self.replay.prepare()
        args = self.replay.python.call_args.args
        self.assertEqual(args[:3], ("source-projection", "scripts/project_replay_sources.py", "feedback"))
        self.assertEqual(args[3:6], tuple(expected / name for name in
                         ("fresh-witness.json", "fresh-claim.json", "children.json")))
        self.assertEqual(args[-2:], (expected / "digits-0.jsonl", expected / "digits-1.jsonl"))
        self.assertEqual(self.replay.python.call_args.kwargs["outputs"], [self.replay.out("sources")])
        self.assertEqual(self.replay.run.call_args.kwargs["request"]["phase"], "sources")
        (expected / "fresh-witness.json").write_bytes(b"changed after first handoff")
        self.replay.python.reset_mock()
        with self.assertRaisesRegex(ValueError, "first Lean handoff inputs changed"):
            self.replay.prepare()
        self.replay.python.assert_not_called()

    def test_ccs_first_components_read_original_sources(self):
        class Enough(Exception):
            pass
        self.replay.public, self.replay.sources = self.root / "public", self.root / "sources"
        self.replay.rounds = self.root / "rounds"
        calls = []
        def lean(*args, **kwargs):
            calls.append(args)
            if args[0] == "fresh-after1":
                raise Enough()
        self.replay.lean = lean
        with self.assertRaises(Enough):
            self.replay.ccs()
        self.assertEqual([call[2] for call in calls[:3]], ["norm", "fresh", "carried-matrix"])
        for call in calls[:3]:
            self.assertEqual(call[3:5], (self.replay.public, self.replay.sources))
        matrices = [call for call in calls if call[2] == "carried-matrix"]
        self.assertEqual(matrices[0][-2], 0)
        self.assertEqual(matrices[-1][-1], runner.ROWS)
        self.assertEqual([call[-1] for call in matrices[:-1]], [call[-2] for call in matrices[1:]])
        self.assertEqual(calls[1][-1], (runner.ROWS + 1) // 2)

    def test_schedule_has_no_gaps_or_unaligned_product_ranges(self):
        geometry = runner.matrix_geometry(runner.read(runner.ARTIFACT))
        self.replay.geometry = geometry
        ranges = self.replay.matrix_ranges()
        for block in geometry:
            selected = [(lo, hi) for index, lo, hi in ranges if index == block["block"]]
            self.assertEqual(selected[0][0], 0)
            self.assertEqual(selected[-1][1], block["count"])
            self.assertEqual([hi for lo, hi in selected[:-1]], [lo for lo, hi in selected[1:]])
            self.assertTrue(all(lo % block["alignment"] == hi % block["alignment"] == 0
                                for lo, hi in selected))

    def test_timed_batches_keep_complete_ordered_coverage_and_headroom(self):
        costs = [300, 300, 100, 100, 100, 100]
        batches = runner.timed_cuts(costs, 90, 750)
        self.assertEqual(batches, [(0, 2), (2, 6)])
        self.assertEqual([i for lo, hi in batches for i in range(lo, hi)], list(range(len(costs))))
        self.assertTrue(all(90 + sum(costs[lo:hi]) <= 750 for lo, hi in batches))
        self.assertEqual(runner.timed_cuts([700, 1, 1], 90, 750), [(0, 1), (1, 3)])
        for costs, overhead in (([float('nan')], 90), ([-1], 90), ([1], -1)):
            with self.assertRaises(ValueError):
                runner.timed_cuts(costs, overhead, 750)

    def test_matrix_timings_require_complete_unique_successful_ranges(self):
        ranges = [(0, 0, 4), (0, 4, 8), (1, 0, 2)]
        receipt = self.root / 'd-matrix-0.command.json'
        log = self.root / 'd-matrix-0.log'
        record = {'name': 'd-matrix-0', 'exit': 0, 'outcome': 'passed', 'elapsed_seconds': 6}
        events = [{'event': 'range_complete', 'block': block, 'first_local_row': lo,
                   'last_local_row_exclusive': hi, 'total_ns': 1_000_000_000}
                  for block, lo, hi in ranges]
        receipt.write_text(json.dumps(record))
        def write(events):
            log.write_text('\n'.join(json.dumps(event) for event in events))
        write(events)
        self.assertEqual(runner.measured_matrix_batches(ranges, self.root), [(0, 3)])
        for changed in (events[:-1], events + [events[0]]):
            write(changed)
            with self.assertRaises(ValueError):
                runner.measured_matrix_batches(ranges, self.root)
        write(events)
        record.update(exit=124, outcome='timed-out')
        receipt.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'completed first-fold'):
            runner.measured_matrix_batches(ranges, self.root)

    def test_native_completion_rejects_zero_tests_on_execution_and_reuse(self):
        for cached in (False, True):
            with self.subTest(cached=cached):
                name = f'native-{cached}'
                command = ['checker', 'selected::test', '--ignored', '--exact']
                record = {'command': command, 'cwd': str(runner.REPO), 'exit': 0, 'outcome': 'passed'}
                def phase(*args, **kwargs):
                    (self.replay.logs / f'{name}.log').write_text(
                        'running 0 tests\ntest result: ok. 0 passed; 0 failed; 0 ignored;\n')
                    (self.replay.logs / f'{name}.command.json').write_text(json.dumps(record))
                self.replay.check.phase.side_effect = phase
                if cached:
                    phase()
                    (self.replay.logs / f'{name}.inputs.json').write_text('{}')
                    (self.replay.logs / f'{name}.outputs.json').write_text('{}')
                with self.assertRaisesRegex(ValueError, 'native test did not complete'):
                    self.replay.run(name, 'rust', command)

    def test_native_completion_requires_the_selected_test_in_either_argument_order(self):
        log = 'test selected::test ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored;\n'
        for command in (['checker', 'selected::test', '--exact'], ['checker', '--exact', 'selected::test']):
            self.assertTrue(runner.native_test_completed(command, log))
            self.assertFalse(runner.native_test_completed(command, log.replace('selected::test', 'other::test')))
        self.assertFalse(runner.native_test_completed(['checker', 'selected::test'], log))


if __name__ == "__main__":
    unittest.main()
