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
    // Compile-time anchor: both finalize variants + both verifier variants
    // exist with the expected signatures. Phase 1.7 split surfaces the
    // non-replay IVC verifier (`verify_uncompressed`) from the
    // chain-replay/audit verifier (`verify_uncompressed_audit`) at the
    // type level via `Uncompressed` vs `UncompressedAudit`.
    fn _finish_terminal(
        _: fn(
            &neo_fold_clean::Preprocessing,
            neo_fold_clean::UncompressedAudit,
        ) -> Result<neo_fold_clean::Uncompressed, neo_fold_clean::Error>,
    ) {
    }
    fn _finish_audit(
        _: fn(
            &neo_fold_clean::Preprocessing,
            neo_fold_clean::UncompressedAudit,
        ) -> Result<neo_fold_clean::UncompressedAudit, neo_fold_clean::Error>,
    ) {
    }
    fn _verify_terminal(
        _: fn(&neo_fold_clean::Preprocessing, &neo_fold_clean::Uncompressed) -> Result<(), neo_fold_clean::Error>,
    ) {
    }
    fn _verify_audit(
        _: fn(&neo_fold_clean::Preprocessing, &neo_fold_clean::UncompressedAudit) -> Result<(), neo_fold_clean::Error>,
    ) {
    }
    _finish_terminal(neo_fold_clean::finish_uncompressed);
    _finish_audit(neo_fold_clean::finish_uncompressed_with_audit);
    _verify_terminal(neo_fold_clean::verify_uncompressed);
    _verify_audit(neo_fold_clean::verify_uncompressed_audit);
}

/// Type-level anchor for the Phase 1.7 split: a terminal-only
/// [`Uncompressed`] **structurally cannot** expose the per-step audit
/// trail.
///
/// If you ever revert the type split (e.g. by moving `steps` /
/// `public_batches` back onto `Uncompressed`), the exhaustive
/// destructure below stops compiling. That's the point: a terminal-only
/// caller can never silently iterate audit fields, because those fields
/// only exist on `UncompressedAudit` — and reaching them requires
/// having an `UncompressedAudit` in hand, not just an `Uncompressed`.
#[test]
fn uncompressed_terminal_type_excludes_audit_trail_at_compile_time() {
    // `Uncompressed` is exhaustively `{ state, final_fold }`. Adding a
    // new field here without updating this anchor will fail the build.
    fn _terminal_destructure(p: &neo_fold_clean::Uncompressed) {
        let neo_fold_clean::Uncompressed {
            state: _,
            final_fold: _,
        } = p;
    }

    // `UncompressedAudit` is exhaustively `{ proof, steps, public_batches }`
    // — i.e. the terminal-only piece + the per-step audit trail. The two
    // halves are namespaced; `audit.proof` and `audit.steps` are not
    // siblings, so a caller that only sees `&Uncompressed` cannot get to
    // `steps` / `public_batches` without explicitly reconstructing an
    // `UncompressedAudit`.
    fn _audit_destructure(a: &neo_fold_clean::UncompressedAudit) {
        let neo_fold_clean::UncompressedAudit {
            proof: _,
            steps: _,
            public_batches: _,
        } = a;
    }

    let _ = _terminal_destructure;
    let _ = _audit_destructure;
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
