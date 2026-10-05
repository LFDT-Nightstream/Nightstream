import contextlib
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

# Pool workers import the checker by module name.
sys.path.insert(0, str(Path(__file__).parent))
import check_piccs_binary_fold as checker  # noqa: E402

P = checker.P
FIELD = struct.Struct("<QQ")
CHALLENGE = [5, 3]


def fold(low, high):
    """One binary fold of K values with the fixed challenge r = 5 + 3u, u² = 7."""
    (a, b), (c, d) = low, high
    delta0, delta1 = (c - a) % P, (d - b) % P
    r0, r1 = CHALLENGE
    return (a + r0 * delta0 + 7 * r1 * delta1) % P, (b + r0 * delta1 + r1 * delta0) % P


def write_prefix(directory, depth, rows, ranges, coins):
    directory.mkdir()
    for first, finish in ranges:
        (directory / f"{first}-{finish}.bin").write_bytes(
            b"".join(FIELD.pack(*value) for value in rows[first:finish]))
    (directory / "manifest.json").write_text(json.dumps(
        [1, depth, 1, 1, len(rows), coins, [list(extent) for extent in ranges]]))


def write_round(path):
    pair = [0, 0]
    path.write_text(json.dumps(
        [1, [pair] * 28, pair, [0], [pair] * 10, CHALLENGE, [0], pair, pair, pair]))


class BinaryFoldTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())
        rows = [((7 * index + 1) % P, (P - 3 * index) % P) for index in range(9)]
        folded = [fold(rows[2 * index], rows[2 * index + 1] if 2 * index + 1 < len(rows) else (0, 0))
                  for index in range(5)]
        coin = [1, 2]
        write_prefix(self.directory / "input", 1, rows, [(0, 3), (3, 9)], [coin])
        write_prefix(self.directory / "output", 2, folded, [(0, 2), (2, 5)], [coin, CHALLENGE])
        write_round(self.directory / "round.json")
        self.arguments = [str(self.directory / name) for name in ("input", "output", "round.json")]

    def run_checker(self, pool):
        saved = checker.POOL_INPUT_BYTES, sys.argv
        checker.POOL_INPUT_BYTES = 0 if pool else 2**62
        sys.argv = ["check_piccs_binary_fold.py", *self.arguments]
        try:
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                checker.main()
            return json.loads(stream.getvalue())
        finally:
            checker.POOL_INPUT_BYTES, sys.argv = saved

    def test_in_process_and_pool_accept_the_same_fold(self):
        in_process = self.run_checker(pool=False)
        self.assertEqual(in_process, self.run_checker(pool=True))
        self.assertEqual(in_process["compared_K_values"], 5)
        self.assertEqual(in_process["cross_file_pairs"], 1)
        self.assertEqual(in_process["zero_tail_pairs"], 1)

    def test_changed_output_word_is_rejected_by_both_paths(self):
        payload = self.directory / "output" / "0-2.bin"
        data = bytearray(payload.read_bytes())
        a, b = FIELD.unpack_from(data, FIELD.size)
        FIELD.pack_into(data, FIELD.size, (a + 1) % P, b)
        payload.write_bytes(bytes(data))
        for pool in (False, True):
            with self.subTest(pool=pool):
                with self.assertRaisesRegex(ValueError, "field bytes differ"):
                    self.run_checker(pool)


if __name__ == "__main__":
    unittest.main()
