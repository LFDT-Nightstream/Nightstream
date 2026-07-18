//! Retained red-team regression for non-canonical serialized matrix encodings.

#[path = "../support/mod.rs"]
mod support;

use neo_ccs::Mat;
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::paper::construction2::{ProofState, RunningInstance};
use neo_fold_clean::CcsInstance;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use serde::Serialize;

/// Serialization-compatible representation of `Mat<T>`. `Mat` skips only
/// its private identity hint, so derived deserialization accepts these three
/// fields without running `Mat::from_row_major`'s length invariant.
#[derive(Serialize)]
struct MatWire<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

/// Safe dense-matrix constructors must reject dimensions whose element count
/// cannot be represented by `usize`. In optimized builds the unchecked
/// product wraps to zero, manufacturing a huge declared matrix with an empty
/// backing vector without serde or private-field access.
#[test]
fn dense_matrix_constructors_reject_dimension_product_overflow() {
    let rows = 1usize << (usize::BITS - 1);
    let from = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Mat::<F>::from_row_major(rows, 2, vec![])
    }));
    let zero = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| Mat::zero(rows, 2, F::ZERO)));
    let append = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut matrix = Mat::zero(1, 2, F::ZERO);
        matrix.append_zero_rows(rows, F::ZERO);
        matrix
    }));

    assert!(
        from.is_err() && zero.is_err() && append.is_err(),
        "input-validation failure: safe dense-matrix APIs accepted dimensions whose element count overflows usize (from_row_major={}, zero={}, append_zero_rows={})",
        from.is_ok(),
        zero.is_ok(),
        append.is_ok(),
    );
}

fn overlong_clone_with_suffix(matrix: &Mat<F>, suffix: F) -> Mat<F> {
    let mut data = matrix.as_slice().to_vec();
    data.push(suffix);
    let encoded = bincode::serialize(&MatWire {
        rows: matrix.rows(),
        cols: matrix.cols(),
        data,
    })
    .expect("serialize malformed Mat wire image");
    bincode::deserialize(&encoded).expect("derived Mat deserializer currently accepts overlong data")
}

fn overlong_clone(matrix: &Mat<F>) -> Mat<F> {
    overlong_clone_with_suffix(matrix, F::from_u64(9))
}

fn short_clone(matrix: &Mat<F>) -> Mat<F> {
    let mut data = matrix.as_slice().to_vec();
    let _ = data.pop().expect("fixture matrix is non-empty");
    let encoded = bincode::serialize(&MatWire {
        rows: matrix.rows(),
        cols: matrix.cols(),
        data,
    })
    .expect("serialize malformed Mat wire image");
    bincode::deserialize(&encoded).expect("derived Mat deserializer currently accepts short data")
}

/// The final running witness is verifier authority, not an ignored snapshot.
/// Both terminal verification modes must enforce canonical dense-matrix
/// backing length rather than committing and projecting only the declared
/// prefix while silently accepting a low-norm hidden suffix.
#[test]
fn terminal_verifiers_reject_overlong_deserialized_final_authority_witness() {
    let prep = support::toy_preprocessing();
    let audit =
        neo_fold_clean::prove(&prep, [vec![support::toy_instance(&prep, 33)]]).expect("construct one-batch audit");
    let mut finalized = neo_fold_clean::finish_uncompressed_with_audit(&prep, audit).expect("finalize one-batch audit");
    neo_fold_clean::verify_uncompressed(&prep, &finalized.proof).expect("honest terminal proof verifies");
    neo_fold_clean::verify_uncompressed_audit(&prep, &finalized).expect("honest audit proof verifies");

    let ProofState::Active { running, .. } = &mut finalized.proof.state.proof else {
        panic!("finalized proof must be active");
    };
    let running = running
        .as_materialized_mut()
        .expect("fixture uses a materialized final accumulator");
    let witness = &mut running.witnesses[0];
    let canonical_len = witness.rows() * witness.cols();
    *witness = overlong_clone_with_suffix(witness, F::ONE);
    assert_eq!(witness.as_slice().len(), canonical_len + 1);

    let terminal = neo_fold_clean::verify_uncompressed(&prep, &finalized.proof);
    let audit = neo_fold_clean::verify_uncompressed_audit(&prep, &finalized);
    assert!(
        terminal.is_err() && audit.is_err(),
        "proof-language failure: terminal verifiers accepted a final authority witness with hidden backing data outside its declared matrix shape (terminal_ok={}, audit_ok={})",
        terminal.is_ok(),
        audit.is_ok(),
    );
}

/// The short-backing counterpart must be rejected before Ajtai commitment
/// code indexes the declared final witness shape. Exercise both production
/// terminal verification and audit replay because both consume this untrusted
/// authority-bearing matrix.
#[test]
fn terminal_verifiers_reject_short_deserialized_final_witness_without_panicking() {
    let prep = support::toy_preprocessing();
    let audit =
        neo_fold_clean::prove(&prep, [vec![support::toy_instance(&prep, 34)]]).expect("construct one-batch audit");
    let mut finalized = neo_fold_clean::finish_uncompressed_with_audit(&prep, audit).expect("finalize one-batch audit");

    let ProofState::Active { running, .. } = &mut finalized.proof.state.proof else {
        panic!("finalized proof must be active");
    };
    let running = running
        .as_materialized_mut()
        .expect("fixture uses a materialized final accumulator");
    running.witnesses[0] = short_clone(&running.witnesses[0]);

    let terminal = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        neo_fold_clean::verify_uncompressed(&prep, &finalized.proof)
    }));
    let audit = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        neo_fold_clean::verify_uncompressed_audit(&prep, &finalized)
    }));
    let terminal_rejected = terminal.as_ref().is_ok_and(|result| result.is_err());
    let audit_rejected = audit.as_ref().is_ok_and(|result| result.is_err());
    assert!(
        terminal_rejected && audit_rejected,
        "verifier availability failure: short final-witness backing was not cleanly rejected (terminal_panicked={}, audit_panicked={}, terminal_rejected={terminal_rejected}, audit_rejected={audit_rejected})",
        terminal.is_err(),
        audit.is_err(),
    );
}

/// Inject one hidden element after a valid Pi_CCS output `X`. The output
/// digest and all verifier algebra address only `rows * cols` entries, so
/// the suffix is neither authenticated nor rejected and a completed NIFS
/// proof remains valid after this post-hoc mutation.
#[test]
fn nifs_rejects_overlong_deserialized_output_matrix() {
    let prep = support::toy_preprocessing();
    let fresh = support::toy_instance(&prep, 0);
    let fresh_claims = vec![fresh.claim.clone()];

    let mut prover_transcript = Transcript::session();
    let (_next_running, mut proof) = neo_fold_clean::paper::nifs::prove(
        &mut prover_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh],
        &RunningInstance::default(),
    )
    .expect("canonical NIFS proof");

    let output = proof.pi_ccs.outputs.first_mut().expect("one Pi_CCS output");
    let canonical_len = output.X.rows() * output.X.cols();
    output.X = overlong_clone(&output.X);
    assert_eq!(
        output.X.as_slice().len(),
        canonical_len + 1,
        "malformed suffix was injected"
    );

    let mut verifier_transcript = Transcript::session();
    let result = neo_fold_clean::paper::nifs::verify(
        &mut verifier_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        &fresh_claims,
        &RunningInstance::default(),
        &proof,
    );

    assert!(
        result.is_err(),
        "proof-language failure: NIFS.V accepted a post-hoc overlong Pi_CCS output matrix whose hidden suffix is outside every digest and algebra check"
    );
}

/// A short `Mat` wire image is accepted by derived deserialization while
/// retaining its larger declared shape. NIFS.V hashes Pi_CCS output `X` by
/// indexing every declared coordinate before validating backing length, so an
/// untrusted malformed proof must return `Err` rather than panic the verifier.
#[test]
fn nifs_verifier_rejects_short_deserialized_output_matrix_without_panicking() {
    let prep = support::toy_preprocessing();
    let fresh = support::toy_instance(&prep, 0);
    let fresh_claims = vec![fresh.claim.clone()];

    let mut prover_transcript = Transcript::session();
    let (_next_running, mut proof) = neo_fold_clean::paper::nifs::prove(
        &mut prover_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh],
        &RunningInstance::default(),
    )
    .expect("canonical NIFS proof");

    let output = proof.pi_ccs.outputs.first_mut().expect("one Pi_CCS output");
    let canonical_len = output.X.rows() * output.X.cols();
    output.X = short_clone(&output.X);
    assert_eq!(
        output.X.as_slice().len() + 1,
        canonical_len,
        "malformed short backing was injected"
    );

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut verifier_transcript = Transcript::session();
        neo_fold_clean::paper::nifs::verify(
            &mut verifier_transcript,
            &prep.params,
            prep.structure(),
            prep.optimized_cache(),
            prep.mix_rhos_commits(),
            prep.combine_b_pows(),
            &fresh_claims,
            &RunningInstance::default(),
            &proof,
        )
    }));

    assert!(
        result.is_ok(),
        "verifier availability failure: a short deserialized Pi_CCS output matrix panicked NIFS.V"
    );
    assert!(
        result.unwrap().is_err(),
        "soundness failure: NIFS.V accepted a short deserialized Pi_CCS output matrix"
    );
}

/// `m_in = 0` is a supported CCS statement shape: no public inputs and the
/// full assignment is private. Π_CCS accepts it, but Π_RLC passes the empty
/// `X` width to Rayon's `par_chunks_exact_mut(0)`, which panics instead of
/// producing a valid fold (or even returning a protocol error).
#[test]
fn nifs_prover_does_not_panic_for_zero_public_inputs() {
    let prep = support::toy_preprocessing_unfixed_public_input_len();
    let fresh = CcsInstance::from_low_norm_assignment(&prep.params, &prep.log, prep.structure(), &[F::ZERO], 0)
        .expect("zero-public-input CCS instance");

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut prover_transcript = Transcript::session();
        neo_fold_clean::paper::nifs::prove(
            &mut prover_transcript,
            &prep.params,
            prep.structure(),
            prep.optimized_cache(),
            &prep.log,
            None,
            prep.mix_rhos_commits(),
            prep.combine_b_pows(),
            vec![fresh],
            &RunningInstance::default(),
        )
    }));

    assert!(
        result.is_ok(),
        "completeness failure: NIFS.P panicked while folding a valid CCS instance with m_in=0"
    );
    assert!(result.unwrap().is_ok(), "valid zero-public-input fold was rejected");
}
