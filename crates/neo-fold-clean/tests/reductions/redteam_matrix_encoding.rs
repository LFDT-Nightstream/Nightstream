//! Retained red-team regression for non-canonical serialized matrix encodings.

#[path = "../support/mod.rs"]
mod support;

use neo_ccs::Mat;
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::CcsInstance;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use serde::Serialize;

/// Serialization-compatible dense representation of `Mat<T>`.
#[derive(Serialize)]
struct MatWire<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
    constant_hint: Option<T>,
    packed_signed_unit: Option<()>,
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

fn malformed_dense_encoding(matrix: &Mat<F>, data: Vec<F>) -> Vec<u8> {
    bincode::serialize(&MatWire {
        rows: matrix.rows(),
        cols: matrix.cols(),
        data,
        constant_hint: None,
        packed_signed_unit: None,
    })
    .expect("serialize malformed Mat wire image")
}

#[test]
fn mat_deserialization_rejects_overlong_dense_backing() {
    let matrix = Mat::zero(2, 3, F::ZERO);
    let mut data = matrix.as_slice().to_vec();
    data.push(F::ONE);
    let encoded = malformed_dense_encoding(&matrix, data);
    assert!(
        bincode::deserialize::<Mat<F>>(&encoded).is_err(),
        "matrix deserialization accepted hidden dense backing data"
    );
}

#[test]
fn mat_deserialization_rejects_short_dense_backing() {
    let matrix = Mat::zero(2, 3, F::ZERO);
    let mut data = matrix.as_slice().to_vec();
    data.pop();
    let encoded = malformed_dense_encoding(&matrix, data);
    assert!(
        bincode::deserialize::<Mat<F>>(&encoded).is_err(),
        "matrix deserialization accepted a short dense backing"
    );
}

/// `m_in = 0` is a supported CCS statement shape: no public inputs and the
/// full assignment is private. Π_CCS accepts it, but Π_RLC passes the empty
/// `X` width to Rayon's `par_chunks_exact_mut(0)`, which panics instead of
/// producing a valid fold (or even returning a protocol error).
#[test]
fn nifs_prover_does_not_panic_for_zero_public_inputs() {
    let prep = support::toy_preprocessing_unfixed_public_input_len();
    let assignment = vec![F::ZERO; neo_math::D];
    let fresh = CcsInstance::from_low_norm_assignment(&prep.params, &prep.log, prep.structure(), &assignment, 0)
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
