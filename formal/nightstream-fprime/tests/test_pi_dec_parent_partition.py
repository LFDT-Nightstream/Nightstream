"""Check exact byte copying, zero ranges, and malformed range rejection."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/split_pidec_parent.py"
SPEC = importlib.util.spec_from_file_location("split_pidec_parent", SCRIPT)
PARTITION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PARTITION)


class ParentPartitionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "parent.jsonl"
        self.output = self.root / "parts"

    def test_exact_copy_and_empty_middle_range(self):
        first = b"[2,[ 0, 1 ]]\n"
        last = b"[10,[18446744069414584320]]\n"
        original = b"[1,11,2,11]\n" + first + last + b"[]\n"
        self.source.write_bytes(original)
        PARTITION.split_parent(self.source, self.output, 3)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["source"]["sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual([(x["start"], x["end"]) for x in manifest["outputs"]], [(2, 5), (5, 8), (8, 11)])
        payloads = [first, b"", last]
        for entry, payload in zip(manifest["outputs"], payloads):
            content = Path(entry["path"]).read_bytes()
            expected = f'[1,11,{entry["start"]},{entry["end"]}]\n'.encode() + payload + b"[]\n"
            self.assertEqual(content, expected)
            self.assertEqual(entry["sha256"], hashlib.sha256(content).hexdigest())

    def test_malformed_ranges_do_not_get_completed_manifests(self):
        cases = [
            b"[true,11,2,11]\n[]\n",
            b"[1,11,2,12]\n[]\n",
            b"[1,11,2,11]\n[2,[]]\n[2,[]]\n[]\n",
            b"[1,11,2,11]\n[10,[]]\n[3,[]]\n[]\n",
            b"[1,11,2,11]\n[11,[]]\n[]\n",
            b"[1,11,2,11]\n[2,[]]\n",
            b"[1,11,2,11]\n[]\n[2,[]]\n",
        ]
        for index, content in enumerate(cases):
            with self.subTest(index=index):
                self.source.write_bytes(content)
                output = self.root / str(index)
                with self.assertRaises(ValueError):
                    PARTITION.split_parent(self.source, output, 3)
                self.assertFalse((output / "manifest.json").exists())

    def test_existing_output_is_preserved(self):
        self.source.write_bytes(b"[1,11,2,11]\n[]\n")
        self.output.mkdir()
        marker = self.output / "marker"
        marker.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            PARTITION.split_parent(self.source, self.output, 3)
        self.assertEqual(marker.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
