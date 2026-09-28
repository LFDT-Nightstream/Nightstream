import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import check_piccs_norm_field_fold as checker


SPEC = importlib.util.spec_from_file_location(
    "projection", Path(__file__).parents[1] / "scripts/project_replay_sources.py")
projection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(projection)


class NormFieldFoldTests(unittest.TestCase):
    def test_accepts_complete_current_manifest_and_rejects_truncated_source(self):
        carrier = projection.D * projection.BLOCKS
        count = (carrier + 3) // 4
        sources = projection.CHILDREN + 1
        challenges = [[1, 2], [3, 4], [5, 6]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps(
                [1, 2, sources, carrier, 0, count, challenges[:2]]))
            # Sparse files exercise the complete profile's framing without
            # allocating or claiming to verify their arithmetic payloads.
            for source in range(sources):
                with (root / f"source-{source}.bin").open("wb") as output:
                    output.truncate(count)
            checker.check_sources(root, challenges)
            with (root / f"source-{sources - 1}.bin").open("r+b") as output:
                output.truncate(count - 1)
            with self.assertRaisesRegex(ValueError, "wrong source file type or byte count"):
                checker.check_sources(root, challenges)


if __name__ == "__main__":
    unittest.main()
