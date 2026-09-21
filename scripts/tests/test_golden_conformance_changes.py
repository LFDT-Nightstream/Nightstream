from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.golden_conformance_changes import (
    ROOT, check_dependency_roots, checks_for_path, classify,
)


class GoldenConformanceChangesTests(unittest.TestCase):
    def test_current_production_and_reference_dependencies_are_registered(self):
        check_dependency_roots(ROOT)

    def test_new_dependency_cannot_silently_skip_checks(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Cargo.toml").write_text("[workspace]\n")
            crate = root / "crates/nightstream"
            crate.mkdir(parents=True)
            (crate / "Cargo.toml").write_text('[dependencies]\nnew = { path = "../new" }\n')
            with self.assertRaisesRegex(ValueError, "unregistered Cargo dependency: crates/new"):
                check_dependency_roots(root)

    def test_shared_implementation_and_build_inputs_select_both_engines(self):
        for path in (
            "crates/nightstream/src/lifecycle/verify.rs",
            "crates/neo-reductions/src/engines/optimized.rs",
            "crates/neo-transcript/src/lib.rs", "crates/neo-math/src/lib.rs",
            "crates/neo-params/src/lib.rs", "crates/neo-ccs/src/lib.rs",
            "crates/neo-ajtai/src/lib.rs", "crates/wip-spartan/src/lib.rs",
            "crates/neo-fold-clean/src/bin/generate_pi_ccs_fixture.rs",
            "crates/neo-fold-clean/tests/nifs/nifs_actual_mutations.rs",
            "crates/nightstream/tests/run_recursive_phase.py",
            "crates/nightstream/Cargo.toml", "Cargo.toml", "Cargo.lock",
            "rust-toolchain.toml", ".cargo/config.toml", ".gitattributes",
        ):
            with self.subTest(path=path):
                self.assertEqual(checks_for_path(path), ("native", "metal"))

    def test_lean_and_original_inputs_require_independent_regeneration(self):
        for path in (
            "formal/nightstream-fprime/NightstreamFPrime/Spec/PiCCS.lean",
            "formal/nightstream-fprime/scripts/replay_recursive_loop.py",
            "formal/nightstream-fprime/lean-toolchain",
            "formal/nightstream-fprime/lake-manifest.json",
            "formal/nightstream-fprime/lakefile.toml",
            "crates/nightstream/artifacts/shared-verifier-v1.json",
            "crates/nightstream-fprime/artifacts/shared-formulas-v1.json",
            "crates/nightstream/tests/fixtures/stage1_actual_nifs/proof.native",
            "scripts/lean_graph/obligations.json",
            "docs/reviews/nightstream-fprime-requirements/INDEPENDENT_REPLAY_ASSET.json",
            "docs/reviews/nightstream-fprime-requirements/EVIDENCE_RELEASE.json",
            "docs/reviews/nightstream-fprime-requirements/TERMINAL_REPLAY_INPUTS.json",
            "docs/reviews/nightstream-fprime-requirements/NONZERO_EXECUTION_RELEASE.json",
            "crates/nightstream/tests/restore_golden_inputs.py",
            "crates/neo-fold-clean/tests/nifs/fixtures/stage1_recursive_states/nonzero-running.json",
            ".github/workflows/ci.yml", "scripts/golden_conformance_changes.py",
        ):
            with self.subTest(path=path):
                self.assertEqual(checks_for_path(path), ("native", "lean_reference", "metal"))

    def test_metal_build_and_all_shader_includes_select_metal(self):
        for path in (
            "crates/neo-prover-metal/src/session.rs",
            "crates/neo-prover-metal/build.rs", "crates/neo-prover-metal/Cargo.toml",
            "crates/neo-prover-metal/shaders/include/goldilocks.metal",
            "scripts/build_metal_shaders.sh",
        ):
            with self.subTest(path=path):
                self.assertEqual(checks_for_path(path), ("metal",))

    def test_documentation_receipts_and_unrelated_projects_do_not_run_producers(self):
        for path in (
            "README.md", "crates/nightstream/VALIDATION.md",
            "formal/nightstream-fprime/README.md",
            "crates/nightstream/tests/evidence/nonzero-fold-20260921/metal-ccs.json",
            "docs/reviews/nightstream-fprime-requirements/NONZERO_NIFS_GATES.json",
            "crates/neo-wasm/src/lib.rs", "formal/nightstream-lean/Main.lean",
            "crates/neo-math/tests/arithmetic.rs",
        ):
            with self.subTest(path=path):
                self.assertEqual(checks_for_path(path), ())

    def test_embedded_documentation_is_a_build_input(self):
        self.assertEqual(checks_for_path(
            "crates/neo-fold-clean/tests/preprocessing_read_only.md"), ("native", "metal"))

    def test_unknown_dependency_files_and_invalid_paths_fail(self):
        for path in ("crates/neo-ajtai/setup/key.bin", "../Cargo.toml", "/Cargo.toml", "./Cargo.toml", ""):
            with self.subTest(path=path), self.assertRaises(ValueError):
                checks_for_path(path)

    def test_result_is_deterministic_and_reports_the_actual_triggers(self):
        changed = ["Cargo.lock", "crates/neo-prover-metal/build.rs", "Cargo.lock", "README.md"]
        result = classify(changed)
        self.assertTrue(result["native"])
        self.assertTrue(result["metal"])
        self.assertFalse(result["lean_reference"])
        self.assertEqual(result["triggers"]["native"], ["Cargo.lock"])
        self.assertEqual(result["triggers"]["metal"], sorted(set(changed) - {"README.md"}))
        self.assertEqual(classify([]), {
            "native": False, "lean_reference": False, "metal": False,
            "triggers": {"native": [], "lean_reference": [], "metal": []},
        })


if __name__ == "__main__":
    unittest.main()
