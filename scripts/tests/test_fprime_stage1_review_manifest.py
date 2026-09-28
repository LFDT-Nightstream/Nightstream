import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import fprime_stage1_review_manifest as manifest


class ArtifactAliasTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.alias = next(iter(manifest.ARTIFACT_ALIASES))
        self.target = manifest.ARTIFACT_ALIASES[self.alias]
        for name in (self.alias, self.target):
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
        (self.root / self.target).write_bytes(b"canonical artifact\n")
        self.link = self.root / self.alias
        self.link.symlink_to(os.path.relpath(self.root / self.target, self.link.parent))
        for name, value in {
            "REPO_ROOT": self.root,
            "RECURSIVE_ROOTS": (Path("crates/nightstream"),),
            "EXPLICIT_FILES": (),
            "REQUIRED_ARTIFACTS": (self.target,),
            "ARTIFACT_ALIASES": {self.alias: self.target},
            "tracked_paths": lambda: [self.alias, self.target],
        }.items():
            holder = patch.object(manifest, name, value)
            holder.start()
            self.addCleanup(holder.stop)

    def test_capture_binds_both_the_link_and_canonical_target(self):
        self.assertEqual(set(manifest.collect_paths(None)), {self.alias, self.target})
        alias, _ = manifest.hash_entry(self.alias)
        before, _ = manifest.hash_entry(self.target)
        self.assertEqual(alias["mode"], "120000")
        self.assertEqual(alias["target"], self.target.as_posix())
        (self.root / self.target).write_bytes(b"changed artifact\n")
        after, _ = manifest.hash_entry(self.target)
        self.assertNotEqual(before["sha256"], after["sha256"])

    def test_retargeted_alias_is_rejected(self):
        self.link.unlink()
        self.link.symlink_to("unexpected-target.json")
        with self.assertRaisesRegex(manifest.ManifestError, "artifact alias changed"):
            manifest.hash_entry(self.alias)

    def test_unregistered_source_symlink_is_rejected(self):
        extra = self.link.parent / "unexpected.json"
        extra.symlink_to(self.link.name)
        with self.assertRaisesRegex(manifest.ManifestError, "unexpected source symlink"):
            manifest.collect_paths(None)


if __name__ == "__main__":
    unittest.main()
