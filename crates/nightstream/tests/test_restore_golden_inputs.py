import hashlib
import io
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

from restore_golden_inputs import checked_asset, restore_independent


class ArchiveRestorationTests(unittest.TestCase):
    def test_checked_archive_link_restores_actual_member_bytes(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            original = root / "source.tar"
            data = b"independent retained witness\n"
            with tarfile.open(original, "w") as archive:
                entry = tarfile.TarInfo("bundle/input.json")
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))
            subprocess.run(["zstd", "-q", str(original)], check=True)
            compressed = original.with_suffix(".tar.zst")
            link = root / "asset.tar.zst"
            link.symlink_to(compressed)
            digest = hashlib.sha256(compressed.read_bytes()).hexdigest()
            checked, _ = checked_asset(root, link.name, compressed.stat().st_size, digest)
            restore_independent(checked, root / "restored")
            self.assertEqual((root / "restored/bundle/input.json").read_bytes(), data)

    def test_changed_archive_content_cannot_use_retained_identity(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            original = b"original"
            (root / "asset.zip").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "differs from its retained source record"):
                checked_asset(root, "asset.zip", len(original), hashlib.sha256(original).hexdigest())


if __name__ == "__main__":
    unittest.main()
