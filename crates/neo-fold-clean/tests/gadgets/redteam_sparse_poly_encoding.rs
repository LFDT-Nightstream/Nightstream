//! Retained red-team regression for sparse-polynomial structure validation.

#[path = "../support/mod.rs"]
mod support;

use std::panic::{catch_unwind, AssertUnwindSafe};

use neo_ccs::{check_ccs_rowwise_zero, CcsMatrix, CcsStructure, CscMat, Mat, SparsePoly, Term};
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::reductions::pi_ccs;
use neo_fold_clean::{config, preprocess, CcsInstance};
use neo_math::{D, F};
use neo_params::{goldilocks_paper_b2 as b2, NeoParams};
use p3_field::PrimeCharacteristicRing;

/// A term must have exactly one exponent for each CCS matrix.
#[test]
fn ccs_constructor_rejects_term_arity_mismatch() {
    let result = CcsStructure::new(
        vec![Mat::identity(1)],
        SparsePoly::new(
            1,
            vec![
                Term {
                    coeff: F::ONE,
                    exps: vec![1, 1],
                },
                Term {
                    coeff: -F::ONE,
                    exps: vec![1, 0],
                },
            ],
        ),
    );

    assert!(
        result.is_err(),
        "the CCS constructor accepted a term whose exponent count does not match the polynomial arity"
    );
}

/// Each exponent fits in `u32`, but their total degree does not. The public
/// constructor accepts this encoding and `SparsePoly::max_degree()` wraps the
/// true degree `2^32` to zero in release builds. That wrapped value drives
/// parameter selection and the verifier's sumcheck degree policy.
#[test]
fn sparse_polynomial_constructor_rejects_total_degree_overflow() {
    let result = CcsStructure::new(
        vec![Mat::identity(1), Mat::identity(1)],
        SparsePoly::new(
            2,
            vec![Term {
                coeff: F::ONE,
                exps: vec![u32::MAX, 1],
            }],
        ),
    );

    assert!(
        result.is_err(),
        "soundness-policy failure: the public CCS constructor accepted a polynomial whose true total degree exceeds u32; release max_degree() reports {}",
        result.as_ref().expect("assertion is reporting an accepted structure").max_degree()
    );
}

/// Even when the polynomial's total degree fits its advertised `u32` type,
/// the sumcheck round degree is `degree + 1`.  That bound is held as `usize`
/// by the reduction engine and must not be truncated back to `u32` when the
/// extension-field soundness policy is checked.
#[test]
fn sumcheck_policy_rejects_round_degree_that_exceeds_u32() {
    let structure = CcsStructure::new(
        vec![Mat::identity(1)],
        SparsePoly::new(
            1,
            vec![Term {
                coeff: F::ONE,
                exps: vec![u32::MAX],
            }],
        ),
    )
    .expect("single-exponent u32::MAX polynomial is accepted");
    assert_eq!(structure.max_degree(), u32::MAX);

    let params = neo_fold_clean::Params::production();
    let result = neo_reductions::engines::pi_ccs_joint::build_joint_dims(params.inner(), &structure, 1, 0);

    assert!(
        result.is_err(),
        "soundness-policy failure: true sumcheck round degree 2^32 was truncated to zero for extension checking; returned d_sc={}",
        result.expect("assertion is reporting an accepted dimension policy").degree
    );
}

/// `usize::next_power_of_two()` wraps to zero in optimized builds when the
/// declared domain is larger than the greatest representable power of two.
/// The sparse identity sentinel makes that extreme shape constructible
/// without allocating a matrix. Dimension derivation must reject the shape,
/// not reinterpret its 64-bit row and column domains as one-bit domains and
/// approve an extension field that fails the true soundness calculation.
#[test]
fn sumcheck_policy_rejects_wrapped_extreme_domain_dimensions() {
    let structure = CcsStructure::new_sparse(vec![CcsMatrix::Identity { n: usize::MAX }], SparsePoly::new(1, vec![]))
        .expect("sparse identity represents the extreme shape without allocation");
    assert_eq!(structure.n, usize::MAX);
    assert_eq!(structure.m, usize::MAX);

    let params = NeoParams::new(
        b2::Q,
        b2::ETA as u32,
        b2::D as u32,
        b2::KAPPA,
        b2::M,
        b2::B_BASE,
        b2::K_RHO,
        b2::T,
        b2::EXTENSION_DEGREE,
        122,
    )
    .expect("validated production-core parameters at lambda=122");
    let true_domain_bits = usize::BITS;
    let ell_d = neo_math::D.next_power_of_two().trailing_zeros();
    let true_ell_max = ell_d + true_domain_bits;
    assert!(
        params.extension_check(true_ell_max, 4).is_err(),
        "fixture requires production s=2 to be insufficient for the true domain"
    );

    let result = neo_reductions::engines::pi_ccs_joint::build_joint_dims(&params, &structure, 1, 0);
    assert!(
        result.is_err(),
        "soundness-policy failure: usize::MAX domain dimensions wrapped through next_power_of_two and were accepted as {:?}",
        result.expect("assertion is reporting an accepted dimension policy")
    );
}

/// A public sparse structure can be assembled from raw CSC arrays. Either the
/// constructor or preprocessing must reject inconsistent backing arrays as a
/// normal error; verifier-owned setup must not index malformed storage and
/// panic.
#[test]
fn preprocessing_rejects_malformed_csc_storage_without_panicking() {
    let structure = match CcsStructure::new_sparse(
        vec![CcsMatrix::Csc(CscMat {
            nrows: 1,
            ncols: 1,
            col_ptr: vec![],
            row_idx: vec![],
            vals: vec![],
        })],
        SparsePoly::new(1, vec![]),
    ) {
        Ok(structure) => structure,
        Err(_) => return,
    };
    let params = config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("shape-specific params");
    support::install_ajtai_module(&params, &structure);

    let preprocess_result = catch_unwind(AssertUnwindSafe(|| preprocess(params, structure, Some(0))));

    assert!(
        preprocess_result.is_ok(),
        "completeness/availability failure: public sparse-CCS construction accepted malformed CSC backing arrays, then public preprocessing panicked"
    );
    assert!(
        preprocess_result.expect("panic checked above").is_err(),
        "validation failure: public preprocessing accepted malformed CSC backing arrays"
    );
}

/// Public preprocessing must recheck the declared shape because callers can
/// mutate the public structure fields after construction.
#[test]
fn preprocessing_rejects_structure_whose_declared_shape_exceeds_matrix_shape() {
    let mut structure = CcsStructure::new_sparse(
        vec![CcsMatrix::Identity { n: 1 }],
        SparsePoly::new(
            1,
            vec![Term {
                coeff: F::ONE,
                exps: vec![1],
            }],
        ),
    )
    .expect("canonical one-variable identity CCS");
    structure.n = 2;
    structure.m = 2;

    let params = config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("shape-specific params");
    let result = preprocess(params, structure, Some(0));

    assert!(
        result.is_err(),
        "preprocessing accepted a declared 2x2 CCS backed by a 1x1 matrix"
    );
}

/// The selected zero-row padding specialization requires `f(0)=0`.
#[test]
fn pi_ccs_rejects_zero_row_padding_when_f_zero_is_nonzero() {
    let mut identity_prefix = Mat::zero(3, D, F::ZERO);
    for row in 0..3 {
        identity_prefix[(row, row)] = F::ONE;
    }
    let structure = CcsStructure::new(
        vec![identity_prefix],
        SparsePoly::new(
            1,
            vec![
                Term {
                    coeff: F::ONE,
                    exps: vec![1],
                },
                Term {
                    coeff: -F::ONE,
                    exps: vec![0],
                },
            ],
        ),
    )
    .expect("three-row CCS");
    let mut assignment = vec![F::ZERO; D];
    assignment[..3].fill(F::ONE);
    check_ccs_rowwise_zero(&structure, &assignment, &[])
        .expect("the public CCS relation checker accepts every real row");
    let params = config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("shape-specific params");
    support::install_ajtai_module(&params, &structure);
    let prep = preprocess(params, structure, Some(D)).expect("preprocessing");
    let fresh = CcsInstance::from_low_norm_assignment(&prep.params, &prep.log, prep.structure(), &assignment, D)
        .expect("valid assignment on all three real rows");
    let mut prover_transcript = Transcript::session();
    let result = pi_ccs::prove(
        &mut prover_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        vec![fresh],
        &RunningInstance::default(),
    );
    assert!(result.is_err(), "zero-row padding accepted a polynomial with f(0) != 0");
}
