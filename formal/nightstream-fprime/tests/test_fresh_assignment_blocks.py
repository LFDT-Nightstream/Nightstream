"""Empty assignment blocks are accepted only for block kinds without slots."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

FORMAL = Path(__file__).resolve().parents[1]
ASSEMBLE = FORMAL / "scripts/assemble_fresh_assignment.py"
CHECK = FORMAL / "tests/check_fresh_assignment_bytes.py"


def write_blocks(directory, empty_slots):
    """26 blocks of two coordinates each, except block 21, which is empty."""
    directory.mkdir()
    (directory / "public.bin").write_bytes(bytes(270))
    blocks, first = [], 270
    for ordinal in range(26):
        size = 0 if ordinal == 21 else 2
        name = f"block-{ordinal}.bin"
        (directory / name).write_bytes(bytes([1, 255][:size]))
        blocks.append({"ordinal": ordinal, "first": first, "finish": first + size,
                       "slots": empty_slots if ordinal == 21 else 1, "file": name})
        first += size
    (directory / "manifest.json").write_text(json.dumps({
        "schema": 1, "logical_width": first, "physical_fields": 0, "public_width": 270,
        "first_block": 0, "finish_block": 26, "blocks": blocks}))


class EmptyBlocks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def run_script(self, *args):
        return subprocess.run([sys.executable, "-B", *map(str, args)], capture_output=True, text=True)

    def test_block_without_slots_may_be_empty(self):
        write_blocks(self.root / "blocks", empty_slots=0)
        carrier, witness = self.root / "carrier.bin", self.root / "witness.json"
        assembled = self.run_script(ASSEMBLE, carrier, witness, self.root / "blocks")
        self.assertEqual(assembled.returncode, 0, assembled.stderr)
        checked = self.run_script(CHECK, witness, self.root / "blocks", "--complete")
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertEqual(json.loads(checked.stdout)["status"], "passed")

    def test_empty_block_with_slots_is_rejected(self):
        write_blocks(self.root / "valid", empty_slots=0)
        witness = self.root / "witness.json"
        self.assertEqual(self.run_script(ASSEMBLE, self.root / "carrier.bin", witness,
                                         self.root / "valid").returncode, 0)
        write_blocks(self.root / "blocks", empty_slots=1)
        assembled = self.run_script(ASSEMBLE, self.root / "other.bin", self.root / "other.json",
                                    self.root / "blocks")
        self.assertNotEqual(assembled.returncode, 0)
        self.assertIn("AssertionError", assembled.stderr)
        checked = self.run_script(CHECK, witness, self.root / "blocks", "--complete")
        self.assertNotEqual(checked.returncode, 0)
        self.assertIn("AssertionError", checked.stderr)


if __name__ == "__main__":
    unittest.main()
