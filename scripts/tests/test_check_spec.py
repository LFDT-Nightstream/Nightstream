import importlib.util
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/check_spec.py"
SPEC = ROOT / "SPEC.md"
LOADER = importlib.util.spec_from_file_location("check_spec", SCRIPT)
CHECK_SPEC = importlib.util.module_from_spec(LOADER)
LOADER.loader.exec_module(CHECK_SPEC)


def check(text):
    with TemporaryDirectory() as directory:
        spec = Path(directory) / "SPEC.md"
        spec.write_text(text, encoding="utf-8")
        return subprocess.run([sys.executable, "-B", str(SCRIPT), "--spec", str(spec)],
                              capture_output=True, text=True)


class CheckSpecTests(unittest.TestCase):
    def test_committed_spec_passes(self):
        result = check(SPEC.read_text(encoding="utf-8"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_dead_path_fails(self):
        result = check(SPEC.read_text(encoding="utf-8") + "\nSee `crates/no-such-crate/`.\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("`crates/no-such-crate/`", result.stdout)

    def test_hand_edited_table_fails(self):
        text = SPEC.read_text(encoding="utf-8")
        row = "| `productionProfile.global.expansionT` | `T` | 128 |"
        self.assertIn(row, text)
        result = check(text.replace(row, row.replace("128", "129")))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("parameters table is stale", result.stdout)

    def test_stale_computed_value_fails(self):
        text = SPEC.read_text(encoding="utf-8")
        for old, new in [("B + cap·T·(b − 1) = 24,576`", "B + cap·T·(b − 1) = 73,728`"),
                         ("`W_ts ≤ 31` bits", "`W_ts ≤ 62` bits"),
                         ("already holds that minimum to 115 bits", "already holds that minimum to 109 bits"),
                         ("about `2^24.58`", "about `2^24.60`"),
                         ("`q = 2^64 − 59 = 18446744073709551557`", "`q = 2^64 − 59 = 18446744069414584321`")]:
            with self.subTest(new=new):
                self.assertEqual(text.count(old), 1, old)
                result = check(text.replace(old, new))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("where the sources give", result.stdout)

    def test_probability_bound_exponents(self):
        text = SPEC.read_text(encoding="utf-8")
        for prefix, exponent in [
            ("On these sets the exact sum is between `2^−", "119.0284"),
            ("`2^−119.0285` and `2^−", "119.0171"),
            ("`2^−119.0285` and `2^−", "119.0"),
            ("has probability at most `(Q_η + 1) · 2^−", "109.92"),
            ("`2(R + M + N · B_ops)/q²` is at most `2^−", "109.9"),
            ("`2(R + M + N · B_ops)/q²` is at most `2^−", "109.92"),
        ]:
            with self.subTest(prefix=prefix, exponent=exponent):
                pattern = re.escape(prefix) + r"[0-9.]+(?=`)"
                changed, count = re.subn(pattern, prefix + exponent, text)
                self.assertEqual(count, 1)
                result = check(changed)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"has {exponent} where the sources give", result.stdout)

    def test_nebula_cutoff_requires_last_accepted_and_first_rejected_tuple(self):
        lean = {key: value for _, key, value in CHECK_SPEC.rows()}
        artifact = CHECK_SPEC.artifact
        for cutoff in (139508, 139510):
            def with_cutoff(path, schema):
                value = artifact(path, schema)
                if path == CHECK_SPEC.NEBULA_GATE:
                    value[1] = cutoff
                return value

            with self.subTest(cutoff=cutoff), patch.object(CHECK_SPEC, "artifact", with_cutoff):
                with self.assertRaisesRegex(SystemExit, "cutoff"):
                    CHECK_SPEC.computed_claims({**lean, "secureTuples": cutoff})

    def test_changed_claim_text_fails(self):
        text = SPEC.read_text(encoding="utf-8")
        old = "So each recovered accumulator is at most"
        self.assertIn(old, text)
        result = check(text.replace(old, "So every recovered accumulator is at most"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no text matches the computed claim", result.stdout)

    def test_new_computed_value_needs_a_claim(self):
        result = check(SPEC.read_text(encoding="utf-8") + "\nThe light fold has 7 rounds (computed).\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("is not recomputed", result.stdout)

    def test_unchecked_values_are_listed(self):
        result = check(SPEC.read_text(encoding="utf-8"))
        for template, _ in CHECK_SPEC.UNCHECKED:
            self.assertIn(f"not recomputed: {template}", result.stdout)

    def test_malformed_pin_fails(self):
        result = check(SPEC.read_text(encoding="utf-8") + "\n`Spec.production_parameter_values` (`12ab`)\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not 16 hex digits", result.stdout)

    def test_unpinned_claim_fails(self):
        text = re.sub(r"(`Claims\.Foundations\.chain_norm`) \(`[0-9a-f]{16}`\)", r"\1", SPEC.read_text(encoding="utf-8"))
        result = check(text)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("claim `Claims.Foundations.chain_norm` has no pin", result.stdout)

    def test_unlisted_paper_number_fails(self):
        result = check(SPEC.read_text(encoding="utf-8") +
                       "\nHyperNova Lemma 99, Poseidon2 Section 99, Chiesa and Orrù Theorem 99, "
                       "SuperNeo v1.2 Lemmas 97 and 98, and SuperNeo Lemma 99.\n")
        self.assertNotEqual(result.returncode, 0)
        for citation in ("HyperNova Lemma 99", "Poseidon2 Section 99", "Chiesa and Orrù Theorem 99",
                         "SuperNeo v1.2 Lemma 97", "SuperNeo v1.2 Lemma 98", "SuperNeo Lemma 99"):
            self.assertIn(f"cites {citation}, which is not a row of the papers table", result.stdout)

    def test_named_citations_cover_their_rows(self):
        text = SPEC.read_text(encoding="utf-8").replace(
            "<!-- /papers -->",
            "| Sample Paper | 2026/99999 | 2026-01-01 | Lemma 3 | Example. |\n<!-- /papers -->")
        result = check(text + "\nSuperNeo v1.2 Theorems 12 and 13, SuperNeo Lemmas 1, 4 and 10, "
                       "and Sample Paper Lemma 3. Theorem 99 names no paper.\n")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_named_protocol_paper_fails_when_it_has_no_rows(self):
        text = re.sub(r"^\| HyperNova \|.*\n", "", SPEC.read_text(encoding="utf-8"), flags=re.M)
        result = check(text + "\nHyperNova Lemma 99 has no row.\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cites HyperNova Lemma 99, which is not a row", result.stdout)

    def test_direct_eprint_fails_when_the_paper_has_no_rows(self):
        text = SPEC.read_text(encoding="utf-8")
        text = re.sub(r"^\| Poseidon2 \|.*\n", "", text, flags=re.M)
        result = check(text)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cites ePrint 2023/323 Section 5.3, which is not a row of the papers table",
                      result.stdout)

    def test_specs_folder_fails(self):
        errors = list(CHECK_SPEC.layout_errors(["specs/nebula-superneo.md", "specs/vectors.json"]))
        self.assertEqual(len(errors), 2, errors)
        self.assertIn("`specs/nebula-superneo.md` is in a `specs/` folder", errors[0])

    def test_separate_spec_file_fails(self):
        errors = list(CHECK_SPEC.layout_errors(["formal/nightstream-fprime/NEBULA_SPEC.md"]))
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("is a separate spec file", errors[0])

    def test_known_documents_pass(self):
        paths = [*CHECK_SPEC.SPEC_FILES, "formal/nightstream-fprime/SECURITY_MODEL.md",
                 "decisions/nebula-memory-checking.md", "formal/nightstream-fprime/NightstreamFPrime/Spec/Nebula.lean"]
        self.assertEqual(list(CHECK_SPEC.layout_errors(paths)), [])

    def test_old_citation_fails(self):
        files = [("crates/a.rs", "/// Row O4.\n/// See spec §8.1.\n"), ("b.lean", "/-- SPEC.md section 7.3. -/")]
        errors = list(CHECK_SPEC.citation_errors(files))
        self.assertEqual(errors, ["crates/a.rs:2 cites `spec §`; cite a SPEC.md section or algorithm, or a Lean name"])


if __name__ == "__main__":
    unittest.main()
