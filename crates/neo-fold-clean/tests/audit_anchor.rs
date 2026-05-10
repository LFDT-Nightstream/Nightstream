//! Audit anchor: this test file enforces the public shape promised by
//! `README.md`. If anything here breaks, the public lifecycle changed.
//!
//! Concretely, this test does *not* exercise the protocol math — that's what
//! the per-reduction conformance suites are for. It exercises the *shape*:
//! - The public lifecycle exists and has the names the README documents.
//! - The paper layer exposes the structures the glossary maps.
//! - The default base-case state has the right paper-symbol fields.

use neo_fold_clean::paper::construction2::{ProofState, State, TRIVIAL_PC};

#[test]
fn base_case_state_has_paper_fields() {
    let z_0 = [7u8; 32];
    let public_trace = [3u8; 32];
    let acc_digest = [5u8; 32];
    let s = State::base(z_0, public_trace, acc_digest);

    // chunk_count = 0, z_0 = z_i, pc = TRIVIAL_PC, ProofState = Initial.
    assert_eq!(s.chunk_count, 0);
    assert_eq!(s.step_count, 0);
    assert_eq!(s.z_0, z_0);
    assert_eq!(s.z_i, z_0);
    assert_eq!(s.pc, TRIVIAL_PC);
    assert_eq!(s.acc_digest, acc_digest);
    assert_eq!(s.public_trace, public_trace);
    assert!(matches!(s.proof, ProofState::Initial));
}

#[test]
fn lifecycle_surface_compiles() {
    // Compile-time anchor: if any of the lifecycle public names or types
    // disappear, the README is wrong. `preprocess` takes the Ajtai
    // homomorphism, the Π_RLC and Π_DEC mixers, and the optional
    // program-fixed `public_input_len`.
    fn _surface_check(
        _: fn(
            neo_fold_clean::Params,
            neo_fold_clean::Structure,
            neo_fold_clean::RlcMixer,
            neo_fold_clean::DecMixer,
            Option<usize>,
        ) -> Result<neo_fold_clean::Preprocessing, neo_fold_clean::Error>,
    ) {
    }
    _surface_check(neo_fold_clean::preprocess);
}

#[test]
fn verify_uncompressed_surface_compiles() {
    // Compile-time anchor: the finish + verify_uncompressed entrypoints exist
    // with the expected signatures, and the proof type carries the public
    // view that verification reads from.
    fn _finish(
        _: fn(
            &neo_fold_clean::Preprocessing,
            neo_fold_clean::Uncompressed,
        ) -> Result<neo_fold_clean::Uncompressed, neo_fold_clean::Error>,
    ) {
    }
    fn _surface(
        _: fn(&neo_fold_clean::Preprocessing, &neo_fold_clean::Uncompressed) -> Result<(), neo_fold_clean::Error>,
    ) {
    }
    _finish(neo_fold_clean::finish_uncompressed);
    _surface(neo_fold_clean::verify_uncompressed);
}

#[test]
fn fold_schedule_step_count_is_correct() {
    use neo_fold_clean::FoldSchedule;

    // RowsPerStep(0) is rejected.
    assert!(FoldSchedule::RowsPerStep(0).validate().is_err());
    assert!(FoldSchedule::RowsPerStep(1).validate().is_ok());

    // 0 rows under any schedule yields 0 steps.
    assert_eq!(FoldSchedule::WholeRun.step_count(0).unwrap(), 0);
    assert_eq!(FoldSchedule::RowsPerStep(1).step_count(0).unwrap(), 0);

    // WholeRun: any non-zero row count is one step.
    assert_eq!(FoldSchedule::WholeRun.step_count(1).unwrap(), 1);
    assert_eq!(FoldSchedule::WholeRun.step_count(7).unwrap(), 1);

    // RowsPerStep: ceiling division.
    assert_eq!(FoldSchedule::RowsPerStep(1).step_count(7).unwrap(), 7);
    assert_eq!(FoldSchedule::RowsPerStep(3).step_count(7).unwrap(), 3); // 3+3+1
    assert_eq!(FoldSchedule::RowsPerStep(7).step_count(7).unwrap(), 1);

    // Default is RowsPerStep(1).
    assert_eq!(FoldSchedule::default(), FoldSchedule::RowsPerStep(1));
}
