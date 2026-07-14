//! Retained red-team regression for sparse-polynomial structure validation.

#[path = "../support/mod.rs"]
mod support;

use std::panic::{catch_unwind, AssertUnwindSafe};

use neo_ccs::{
    check_ccs_rowwise_zero, direct_sum, direct_sum_mixed, CcsMatrix, CcsStructure, CscMat, Mat, SparsePoly, Term,
};
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::nifs;
use neo_fold_clean::paper::reductions::pi_ccs;
use neo_fold_clean::{config, preprocess, CcsInstance};
use neo_math::F;
use neo_params::{goldilocks_paper_b2 as b2, NeoParams};
use p3_field::PrimeCharacteristicRing;

/// The public structure constructor and preprocessing API accept a term whose
/// exponent count differs from the polynomial arity. A normal optimized prover
/// later treats the same malformed verifier-owned structure as impossible and
/// panics. Successful public preprocessing must not create a prover-crashing
/// protocol context.
#[test]
fn pi_ccs_prover_does_not_panic_after_successful_public_preprocessing() {
    let structure = CcsStructure::new(
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
    )
    .expect("public CCS constructor accepts malformed term exponent counts");
    let params = config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("shape-specific params");
    support::install_ajtai_module(&params, &structure);
    let prep =
        preprocess(params, structure, Some(1)).expect("public preprocessing accepts the malformed sparse polynomial");
    let instance = CcsInstance::from_low_norm_assignment(&prep.params, &prep.log, prep.structure(), &[F::ZERO], 1)
        .expect("zero low-norm assignment");

    let prove_result = catch_unwind(AssertUnwindSafe(|| {
        let mut transcript = Transcript::session();
        pi_ccs::prove(
            &mut transcript,
            &prep.params,
            prep.structure(),
            prep.optimized_cache(),
            &prep.log,
            vec![instance],
            &RunningInstance::default(),
        )
    }));

    assert!(
        prove_result.is_ok(),
        "completeness/availability failure: public preprocessing accepted a malformed sparse polynomial that makes the optimized Pi_CCS prover panic"
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
    let result = neo_reductions::engines::utils::build_dims_and_policy(params.inner(), &structure);

    assert!(
        result.is_err(),
        "soundness-policy failure: true sumcheck round degree 2^32 was truncated to zero for extension checking; returned d_sc={}",
        result.expect("assertion is reporting an accepted dimension policy").d_sc
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

    let result = neo_reductions::engines::utils::build_dims_and_policy(&params, &structure);
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

    let preprocess_result = catch_unwind(AssertUnwindSafe(|| preprocess(params, structure, Some(1))));

    assert!(
        preprocess_result.is_ok(),
        "completeness/availability failure: public sparse-CCS construction accepted malformed CSC backing arrays, then public preprocessing panicked"
    );
    assert!(
        preprocess_result.expect("panic checked above").is_err(),
        "validation failure: public preprocessing accepted malformed CSC backing arrays"
    );
}

/// `CcsStructure::{n,m}` are public authority fields independent from each
/// matrix's real dimensions. Inflate a 1x1 identity relation into a declared
/// 2x2 program and prove the public assignment `[0, 1]`. For the declared
/// identity relation, row 1 should require the second coordinate to be zero;
/// the current engine instead treats the missing matrix row/column as zero and
/// accepts the underconstrained statement end to end.
#[test]
fn nifs_rejects_structure_whose_declared_shape_exceeds_matrix_shape() {
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
    support::install_ajtai_module(&params, &structure);
    let prep = preprocess(params, structure, Some(2))
        .expect("current preprocessing accepts contradictory declared and matrix shapes");
    let fresh = CcsInstance::from_low_norm_assignment(&prep.params, &prep.log, prep.structure(), &[F::ZERO, F::ONE], 2)
        .expect("declared-width assignment");
    let fresh_claims = vec![fresh.claim.clone()];

    let mut prover_transcript = Transcript::session();
    let (_next_running, proof) = nifs::prove(
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
    .expect("current prover accepts the underconstrained declared program");

    let mut verifier_transcript = Transcript::session();
    let result = nifs::verify(
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
        "soundness failure: NIFS.V accepted a declared 2x2 identity CCS whose real 1x1 matrix silently leaves the nonzero second public coordinate unconstrained"
    );
}

/// For `n=m=3`, the multilinear row domain pads to four points. The fourth
/// point is structural padding and must not be checked as a real CCS row,
/// even when `f(0) != 0`. This otherwise follows SuperNeo's normalization:
/// `M_1=I`, and every real row has `M_1 z = 1`, satisfying `f(X)=X-1`.
#[test]
fn pi_ccs_accepts_non_power_of_two_rows_when_f_zero_is_nonzero() {
    let structure = CcsStructure::new(
        vec![Mat::identity(3)],
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
    check_ccs_rowwise_zero(&structure, &[F::ONE, F::ONE, F::ONE], &[])
        .expect("the public CCS relation checker accepts every real row");
    let params = config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("shape-specific params");
    support::install_ajtai_module(&params, &structure);
    let prep = preprocess(params, structure, Some(3)).expect("preprocessing");
    let fresh =
        CcsInstance::from_low_norm_assignment(&prep.params, &prep.log, prep.structure(), &[F::ONE, F::ONE, F::ONE], 3)
            .expect("valid assignment on all three real rows");
    let fresh_claims = vec![fresh.claim.clone()];

    let mut prover_transcript = Transcript::session();
    let proof = pi_ccs::prove(
        &mut prover_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        vec![fresh],
        &RunningInstance::default(),
    )
    .expect("valid non-power-of-two relation must prove");

    let mut verifier_transcript = Transcript::session();
    pi_ccs::verify(
        &mut verifier_transcript,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &fresh_claims,
        &RunningInstance::default(),
        &proof,
    )
    .expect("padding row must not create a false completeness rejection");
}

/// A block-diagonal direct sum must preserve each component relation on the
/// rows where the other component is inactive.  Adding `f1 + beta*f2`
/// without compensating for `f1(0)`/`f2(0)` instead injects constants on
/// those rows, letting two false component statements cancel while rejecting
/// the honest concatenation.
#[test]
fn mixed_direct_sum_preserves_nonhomogeneous_component_relations() {
    fn affine_identity(offset: F) -> CcsStructure<F> {
        CcsStructure::new(
            vec![Mat::identity(1)],
            SparsePoly::new(
                1,
                vec![
                    Term {
                        coeff: F::ONE,
                        exps: vec![1],
                    },
                    Term {
                        coeff: offset,
                        exps: vec![0],
                    },
                ],
            ),
        )
        .expect("shifted identity relation")
    }

    let left = affine_identity(F::from_u64(2));
    let right = affine_identity(-F::ONE);
    let beta = F::from_u64(2);
    let combined = direct_sum_mixed(&left, &right, beta).expect("recommended mixed direct-sum construction");

    let honest = [-F::from_u64(2), F::ONE];
    check_ccs_rowwise_zero(&left, &honest[..1], &[]).expect("left honest assignment satisfies y+2=0");
    check_ccs_rowwise_zero(&right, &honest[1..], &[]).expect("right honest assignment satisfies y-1=0");
    let honest_combined_result = check_ccs_rowwise_zero(&combined, &honest, &[]);

    let forged = [F::ZERO, F::ZERO];
    assert!(
        check_ccs_rowwise_zero(&left, &forged[..1], &[]).is_err()
            && check_ccs_rowwise_zero(&right, &forged[1..], &[]).is_err(),
        "attack precondition: both component statements are false"
    );
    check_ccs_rowwise_zero(&combined, &forged, &[])
        .expect("buggy mixed polynomial cancels the two false component rows");

    let params = config::ccs_params(combined.n, combined.m, combined.t(), combined.max_degree())
        .expect("combined shape parameters");
    support::install_ajtai_module(&params, &combined);
    let prep = preprocess(params, combined, Some(2)).expect("preprocess vulnerable combined relation");
    let instance = CcsInstance::from_low_norm_assignment(&prep.params, &prep.log, prep.structure(), &forged, 2)
        .expect("low-norm forged combined assignment");
    let audit = neo_fold_clean::prove(&prep, [vec![instance]]).expect("prove forged combined statement");
    let proof = neo_fold_clean::finish_uncompressed(&prep, audit).expect("finalize forged combined statement");
    let result = neo_fold_clean::verify_uncompressed(&prep, &proof);

    assert!(
        honest_combined_result.is_ok() && result.is_err(),
        "mixed direct-sum relation failure: two honest components were rejected ({honest_combined_result:?}), while the full NIFS lifecycle accepted an assignment that violates both components ({result:?})"
    );
}

/// Even for homogeneous component polynomials, `beta=0` erases the entire
/// right relation. The public mixed direct-sum constructor returns `Result`
/// and must reject that degenerate mixer rather than relying on every caller
/// to have used the transcript-derived convenience wrapper.
#[test]
fn mixed_direct_sum_rejects_zero_mixing_scalar() {
    let identity_zero = || {
        CcsStructure::new(
            vec![Mat::identity(1)],
            SparsePoly::new(
                1,
                vec![Term {
                    coeff: F::ONE,
                    exps: vec![1],
                }],
            ),
        )
        .expect("homogeneous identity relation")
    };
    let left = identity_zero();
    let right = identity_zero();
    let result = direct_sum_mixed(&left, &right, F::ZERO);
    let forged = [F::ZERO, F::ONE];
    assert!(
        check_ccs_rowwise_zero(&right, &forged[1..], &[]).is_err(),
        "attack precondition: right component rejects one"
    );
    let accepted_forgery = result
        .as_ref()
        .is_ok_and(|combined| check_ccs_rowwise_zero(combined, &forged, &[]).is_ok());

    assert!(
        result.is_err(),
        "direct-sum soundness failure: beta=0 was accepted and erased the false right component (forgery accepted={accepted_forgery})"
    );
}

/// Both direct-sum constructors return `Result`, so dimension overflow in two
/// valid compact sparse structures must be rejected as an error rather than
/// wrapping the output row domain and panicking while embedding entries.
#[test]
fn direct_sum_rejects_dimension_overflow_without_panicking() {
    fn sparse_rows(nrows: usize, row_idx: Vec<usize>) -> CcsStructure<F> {
        let nnz = row_idx.len();
        CcsStructure::new_sparse(
            vec![CcsMatrix::Csc(CscMat {
                nrows,
                ncols: 1,
                col_ptr: vec![0, nnz],
                row_idx,
                vals: vec![F::ONE; nnz],
            })],
            SparsePoly::new(1, vec![]),
        )
        .expect("internally valid compact CSC structure")
    }

    let left = sparse_rows(usize::MAX, vec![]);
    let right = sparse_rows(2, vec![0]);

    let plain = catch_unwind(AssertUnwindSafe(|| direct_sum(&left, &right)));
    let mixed = catch_unwind(AssertUnwindSafe(|| direct_sum_mixed(&left, &right, F::from_u64(2))));

    assert!(
        plain.is_ok() && mixed.is_ok(),
        "setup availability failure: valid compact sparse dimensions overflowed and made a public direct-sum constructor panic (plain_panicked={}, mixed_panicked={})",
        plain.is_err(),
        mixed.is_err(),
    );
    assert!(
        plain.expect("panic checked above").is_err() && mixed.expect("panic checked above").is_err(),
        "input-validation failure: direct-sum constructors accepted wrapped aggregate dimensions"
    );
}
