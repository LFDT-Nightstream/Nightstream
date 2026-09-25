"""Focused projection tests. Execute separately under the existing 300-second cap.

Only carrier extent and logical width are reduced. The fixture keeps all five
public blocks and one partial final block, with the production nine-lane tail.
It retains 54 lanes, 16 children, 270 public words and 14 matrices.
"""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("projection", Path(__file__).parents[1] / "scripts/project_replay_sources.py")
projection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(projection)
D, P = projection.D, projection.P
LOGICAL = projection.PUBLIC + projection.LOGICAL % D
BLOCKS = (LOGICAL + D - 1) // D


def encoded(value):
    return json.dumps(value, separators=(",", ":")) + "\n"


def wrapped(value):
    return {"value": value}


def extension(value):
    return {"_phantom": None, "value": [wrapped(value[0]), wrapped(value[1])]}


def matrix(positive=None, negative=None):
    result = {"rows": D, "cols": BLOCKS, "data": [],
              "constant_hint": wrapped(0), "packed_signed_unit": None}
    if positive is not None:
        result["constant_hint"] = None
        result["packed_signed_unit"] = {
            "bits": {"ColumnMasks": {"positive": positive, "negative": negative}},
            "values": [wrapped(0), wrapped(1), wrapped(P - 1)], "cols": BLOCKS}
    return result


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "original"
        self.source.mkdir()
        self.enterContext(patch.object(projection, "BLOCKS", BLOCKS))
        self.enterContext(patch.object(projection, "LOGICAL", LOGICAL))
        self.x = [1] + [0] * 269
        self.publics = [[0] * 270 for _ in range(16)]
        self.publics[0][107] = P - 1
        self.running = [
            [[0, 0] for _ in range(28)],
            [[0] * 1188 for _ in range(16)], self.publics,
            [[[0, 0] for _ in range(54)] for _ in range(16)],
            [[[[0, 0] for _ in range(54)] for _ in range(14)] for _ in range(16)]]
        self.claim = {"c": {"d": 54, "kappa": 22, "data": [wrapped(0) for _ in range(1188)]},
                      "x": [wrapped(value) for value in self.x], "m_in": 270, "adv": None}
        positive, negative = [0] * BLOCKS, [0] * BLOCKS
        positive[0], positive[-1] = 1, 1 << 8
        self.fresh = matrix(positive, negative)
        self.save(self.source / "fresh-witness.json", self.fresh)
        self.save(self.source / "fresh-claim.json", self.claim)
        claims = []
        for child in range(16):
            data = [0] * 270
            for index, value in enumerate(self.publics[child]):
                data[(index % 54) * 5 + index // 54] = value
            claims.append({"c": self.claim["c"], "X": {
                "rows": 54, "cols": 5, "data": [wrapped(value) for value in data],
                "constant_hint": None, "packed_signed_unit": None},
                "m_in": 270, "adv": None, "r": [extension([0, 0]) for _ in range(28)],
                "eval_k": [extension([0, 0]) for _ in range(64)],
                "eval_a": [[extension([0, 0]) for _ in range(64)] for _ in range(14)]})
            pos, neg = [0] * BLOCKS, [0] * BLOCKS
            if child == 0:
                neg[1], pos[-1] = 1 << 53, 1 << 53
            if child == 15:
                pos[-1] = 1 << 52
            self.save(self.source / f"digit-{child}.json", matrix(pos, neg) if child in (0, 15) else matrix())
        self.save(self.source / "envelope.json", {
            "schema": 1, "iteration": 2, "z0": [0] * 4, "current": [0] * 4,
            "child_witness_count": 16, "running_claims": claims, "running_parent": {}})
        self.children = self.root / "children.json"
        self.save(self.children, self.running)
        self.first, self.second = self.root / "first.jsonl", self.root / "second.jsonl"
        self.first_rows = [[1, 54, 16, BLOCKS, 0, 3], [1, [[0, 0, 1 << 53]]], []]
        self.second_rows = [[1, 54, 16, BLOCKS, 3, BLOCKS],
                            [BLOCKS - 1, [[0, 1 << 53, 0], [15, 1 << 52, 0]]], []]
        self.range_file(self.first, self.first_rows)
        self.range_file(self.second, self.second_rows)
        self.expected_sources = "".join(encoded(value) for value in [
            [1, 54, 17, BLOCKS], [0, [[0, 1, 0]]], [1, [[1, 0, 1 << 53]]],
            [BLOCKS - 1, [[0, 1 << 8, 0], [1, 1 << 53, 0], [16, 1 << 52, 0]]], []])

    def save(self, path, value):
        path.write_text(encoded(value))

    def range_file(self, path, values):
        path.write_text("".join(encoded(value) for value in values))

    def invoke(self, mode, output):
        if mode == "original":
            arguments = [mode, str(self.source), str(output)]
        else:
            arguments = [mode, str(self.source / "fresh-witness.json"),
                         str(self.source / "fresh-claim.json"), str(self.children), str(output),
                         str(self.first), str(self.second)]
        with patch.object(sys, "argv", ["project_replay_sources.py", *arguments]), contextlib.redirect_stdout(io.StringIO()):
            projection.main()

    def reject(self, mode, marker):
        output = self.root / "rejected"
        with self.assertRaisesRegex(ValueError, marker):
            self.invoke(mode, output)
        self.assertFalse(output.exists())
        self.assertEqual(list(self.root.glob("source-projection-*")), [])

    def test_original_projection(self):
        output = self.root / "projected"
        self.invoke("original", output)
        self.assertEqual((output / "sources.jsonl").read_text(), self.expected_sources)
        self.assertEqual((output / "public.json").read_text(), encoded([[0] * 1188, self.x, self.running]))

    def test_feedback_projection(self):
        output = self.root / "projected"
        self.invoke("feedback", output)
        self.assertEqual((output / "sources.jsonl").read_text(), self.expected_sources)
        self.assertEqual((output / "public.json").read_text(), encoded([[0] * 1188, self.x, self.running]))

    def test_mask_overlap_and_high_bit(self):
        for field, index, value in [("negative", 0, 1), ("positive", 2, 1 << 54)]:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.fresh)
                changed["packed_signed_unit"]["bits"]["ColumnMasks"][field][index] = value
                self.save(self.source / "fresh-witness.json", changed)
                self.reject("original", "invalid signed-unit masks")

    def test_fresh_tail_both_modes(self):
        changed = copy.deepcopy(self.fresh)
        changed["packed_signed_unit"]["bits"]["ColumnMasks"]["positive"][-1] |= 1 << 53
        self.save(self.source / "fresh-witness.json", changed)
        for mode in ("original", "feedback"):
            with self.subTest(mode=mode):
                self.reject(mode, "nonzero fresh carrier tail")

    def test_public_mismatch_and_noncanonical(self):
        for value, marker in [(0, "source witness/public mismatch"), (P, "noncanonical field")]:
            with self.subTest(value=value):
                changed = copy.deepcopy(self.claim)
                changed["x"][0] = wrapped(value)
                self.save(self.source / "fresh-claim.json", changed)
                self.reject("feedback", marker)

    def test_d_child_order_and_duplicate(self):
        for entries in [[[15, 1 << 52, 0], [0, 1 << 53, 0]], [[0, 1, 0], [0, 2, 0]]]:
            with self.subTest(entries=entries):
                changed = copy.deepcopy(self.second_rows)
                changed[1][1] = entries
                self.range_file(self.second, changed)
                self.reject("feedback", "unordered, duplicate, or invalid D child")

    def test_d_gap_and_incomplete(self):
        changed = copy.deepcopy(self.second_rows)
        changed[0][4] = 4
        self.range_file(self.second, changed)
        self.reject("feedback", "D range gap/overlap")
        self.range_file(self.second, [[1, 54, 16, BLOCKS, 3, BLOCKS - 1], []])
        self.reject("feedback", "incomplete D range coverage")

    def test_d_terminator_and_trailing_data(self):
        self.range_file(self.second, self.second_rows[:-1])
        self.reject("feedback", "missing D range terminator")
        self.range_file(self.second, self.second_rows + [[]])
        self.reject("feedback", "extra data after D range terminator")

    def test_existing_output(self):
        output = self.root / "existing"
        output.mkdir()
        marker = output / "keep"
        marker.write_text("unchanged")
        with self.assertRaisesRegex(ValueError, "output directory already exists"):
            self.invoke("original", output)
        self.assertEqual(marker.read_text(), "unchanged")


if __name__ == "__main__":
    unittest.main()
