//! Retained red-team regression for verifier-owned parameter/structure binding.

#[path = "../support/mod.rs"]
mod support;

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use neo_ajtai::{get_global_pp_for_dims, set_global_pp, AjtaiSModule, PP};
use neo_ccs::{CcsMatrix, CcsStructure, Mat, SModuleHomomorphism, SparsePoly, Term};
use neo_fold_clean::lifecycle::preprocess_with_test_log;
use neo_fold_clean::{config, finish_uncompressed, preprocess, prove, verify_uncompressed, CcsInstance};
use neo_math::{Rq, D, F};
use p3_field::PrimeCharacteristicRing;

const ACTUAL_MATRIX_COUNT: usize = 131_072;

/// SuperNeo D.5 loses `(K + k + 1) / |C|` in the Π_RLC coordinate
/// extractor. Shape-selected executable parameters must charge that loss at
/// the largest fold width they advertise.
#[test]
fn shape_selected_params_charge_pi_rlc_coordinate_extraction_loss() {
    let maximum_assignment_width = (config::M as usize / D) * D;
    let params = config::r1cs_params(1 << 26, maximum_assignment_width).expect("maximum-geometry parameters");
    let challenge_set_size = 5u128.pow(params.d());
    let extraction_queries = params.max_fresh_count() as u128 + params.k_rho() as u128 + 1;
    let effective_bits = (challenge_set_size / extraction_queries).ilog2();
    let summary = params
        .validate_ccs_shape(1 << 26, maximum_assignment_width, 3, 2)
        .expect("maximum-geometry combined census");

    assert!(
        effective_bits >= params.lambda() && summary.security_bits == params.lambda() && summary.slack_bits == 0,
        "soundness-policy failure: Π_RLC's (K+k+1)/|C| extractor loss leaves {effective_bits} whole bits at {extraction_queries} queries, while the combined census is {summary:?} for lambda={}",
        params.lambda(),
    );
}

/// Appendix B.2 fixes the padded row domain at `m = 2^30`. The largest
/// complete field vector is `n_F = D * floor(m / D)`. Shape selection must
/// reject a field vector above that complete-ring bound.
#[cfg(target_pointer_width = "64")]
#[test]
fn ccs_params_reject_shape_above_paper_profile() {
    let paper_m = 1usize << 30;
    let paper_n_f = (paper_m / D) * D;
    let production = config::production_params();
    let over_cap = paper_n_f + 1;
    let column_result = config::ccs_params(1, over_cap, 1, 0);
    let row_result = config::ccs_params(paper_m + 1, 1, 1, 0);
    let profile_matches_paper = production.m() == paper_m as u64;

    assert!(
        profile_matches_paper && column_result.is_err() && row_result.is_err(),
        "soundness-policy failure: Appendix B.2 m={paper_m} gives n_F={paper_n_f}, but production params.m={} and shape selection returned column={column_result:?}, row={row_result:?}",
        production.m(),
    );
}

/// Optional public-input arity is verifier policy. `None` must have a tagged
/// encoding distinct from the valid fixed-zero policy.
#[test]
fn verifier_key_distinguishes_unbounded_from_fixed_zero_public_input_arity() {
    let structure = CcsStructure::new(vec![Mat::zero(1, D, F::ZERO)], SparsePoly::<F>::new(1, Vec::new()))
        .expect("one-ring zero relation");
    let params = config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("shape-specific params");
    support::install_ajtai_module(&params, &structure);

    let unbounded = preprocess(params.clone(), structure.clone(), None).expect("unbounded public-input policy");
    let fixed_zero = preprocess(params, structure, Some(0)).expect("fixed-zero public-input policy");

    let assignment = vec![F::ZERO; D];
    let instance =
        CcsInstance::from_low_norm_assignment(&unbounded.params, &unbounded.log, unbounded.structure(), &assignment, D)
            .expect("one-ring instance");
    let audit = prove(&unbounded, [vec![instance]]).expect("prove under unbounded policy");
    let proof = finish_uncompressed(&unbounded, audit).expect("finish under unbounded policy");
    let unbounded_result = verify_uncompressed(&unbounded, &proof);
    let fixed_zero_result = verify_uncompressed(&fixed_zero, &proof);
    assert!(
        unbounded_result.is_ok() && fixed_zero_result.is_err(),
        "fixture requires different verifier languages (unbounded={unbounded_result:?}, fixed_zero={fixed_zero_result:?})"
    );

    let same_vk = unbounded.vk.digest() == fixed_zero.vk.digest();
    let same_boundary = neo_fold_clean::paper::digest::initial_boundary_digest(unbounded.structure_digest(), None)
        == neo_fold_clean::paper::digest::initial_boundary_digest(fixed_zero.structure_digest(), Some(0));
    assert!(
        !same_vk && !same_boundary,
        "verifier-policy encoding failure: None and Some(0) define different accepted languages but collide (same_vk={same_vk}, same_initial_boundary={same_boundary})"
    );
}

#[cfg(target_pointer_width = "64")]
#[test]
fn preprocessing_rejects_public_input_arity_above_structure_width() {
    let structure = CcsStructure::new(vec![Mat::zero(1, D, F::ZERO)], SparsePoly::<F>::new(1, Vec::new()))
        .expect("one-ring zero relation");
    let params = config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("shape-specific params");
    support::install_ajtai_module(&params, &structure);

    assert!(matches!(
        preprocess(params, structure, Some(usize::MAX)),
        Err(neo_fold_clean::Error::PreprocessingPublicInputTooLarge { m_in: usize::MAX, m: D })
    ));
}

/// An explicit parameter-selection margin is verifier policy. A public `u32`
/// margin that cannot be represented by the internal signed slack type must
/// be rejected, not narrowed to a negative number that makes the comparison
/// vacuously succeed.
#[test]
fn ccs_parameter_selection_rejects_unrepresentable_safety_margin() {
    let result = neo_fold_clean::Params::for_ccs_shape_with(1, 1, 1, 0, 100, u32::MAX);
    assert!(
        result.is_err(),
        "soundness-policy failure: u32 safety margin wrapped negative during signed comparison and was accepted as {result:?}"
    );
}

/// The joint verifier accepts CCS degree `D_f + 1`. Parameter selection must
/// charge that degree in the one physical SumCheck budget.
#[test]
fn ccs_parameter_selection_charges_strict_polynomial_degree_bound() {
    const DEGREE: u32 = 4_036;

    let structure = CcsStructure::new(
        vec![Mat::identity(1)],
        SparsePoly::new(
            1,
            vec![Term {
                coeff: F::ONE,
                exps: vec![DEGREE],
            }],
        ),
    )
    .expect("shape-valid degree-bound fixture");
    let params = config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("current selector accepts fixture");

    let true_policy = params.validate_ccs_shape(structure.n, structure.m, structure.t(), structure.max_degree());

    assert!(
        true_policy
            .as_ref()
            .is_ok_and(|summary| summary.verifier_degree == DEGREE + 1 && summary.slack_bits == 0),
        "soundness-policy failure: actual degree {DEGREE} requires verifier degree {}, but selected lambda={} has {true_policy:?}",
        DEGREE + 1,
        params.inner().lambda,
    );
}

/// The commitment public parameters are part of HyperNova's committed
/// relation and therefore part of the verifier language. Two same-shaped
/// Ajtai matrices that accept different proofs must not advertise the same
/// verifier-key digest.
#[test]
fn verifier_key_digest_binds_same_shaped_ajtai_public_parameters() {
    let structure = CcsStructure::new(vec![Mat::zero(1, D, F::ZERO)], SparsePoly::<F>::new(1, Vec::new()))
        .expect("toy zero relation");
    let params = config::r1cs_params(structure.n, structure.m).expect("shape params");
    let cols = structure.m.div_ceil(D);
    let kappa = params.kappa() as usize;
    let zero_ring = Rq([F::ZERO; D]);

    let weak_log = AjtaiSModule::new(Arc::new(PP::<Rq> {
        kappa,
        m: cols,
        d: D,
        m_rows: vec![vec![zero_ring; cols]; kappa],
    }));
    let mut strong_rows = vec![vec![zero_ring; cols]; kappa];
    strong_rows[0][0].0[0] = F::ONE;
    let strong_log = AjtaiSModule::new(Arc::new(PP::<Rq> {
        kappa,
        m: cols,
        d: D,
        m_rows: strong_rows,
    }));

    let weak =
        preprocess_with_test_log(params.clone(), structure.clone(), weak_log, Some(D)).expect("weak verifier context");
    let strong = preprocess_with_test_log(params, structure, strong_log, Some(D)).expect("strong verifier context");
    let mut assignment = vec![F::ZERO; D];
    assignment[0] = F::ONE;
    let instance = CcsInstance::from_low_norm_assignment(&weak.params, &weak.log, weak.structure(), &assignment, D)
        .expect("nonzero instance under weak setup");
    let proof = prove(&weak, vec![vec![instance]]).expect("prove under weak setup");
    let finished = finish_uncompressed(&weak, proof).expect("finish under weak setup");
    verify_uncompressed(&weak, &finished).expect("matching weak verifier accepts");
    assert!(
        verify_uncompressed(&strong, &finished).is_err(),
        "fixture must establish that the two Ajtai setups define different verifier languages"
    );

    assert_ne!(
        weak.vk.digest(),
        strong.vk.digest(),
        "verifier-key identity failure: different same-shaped Ajtai public parameters produce the same vk_fs_digest"
    );
}

#[test]
fn preprocessing_rejects_malformed_explicit_ajtai_matrix_shape() {
    let structure =
        CcsStructure::new(vec![Mat::identity(1)], SparsePoly::<F>::new(1, Vec::new())).expect("toy zero relation");
    let params = config::r1cs_params(structure.n, structure.m).expect("shape params");
    let cols = structure.m.div_ceil(D);
    let kappa = params.kappa() as usize;
    let malformed_log = AjtaiSModule::new(Arc::new(PP::<Rq> {
        kappa,
        m: cols,
        d: D,
        m_rows: vec![vec![Rq([F::ZERO; D]); cols]; kappa - 1],
    }));

    assert!(
        preprocess_with_test_log(params, structure, malformed_log, Some(0)).is_err(),
        "preprocessing must reject an explicit Ajtai matrix whose physical row count disagrees with κ"
    );
}

/// Global setup is a verifier-authority boundary. If an attacker registers a
/// well-shaped PP first, a later attempt to install a different verifier-owned
/// PP for the same `(d, m)` must fail loudly; returning `Ok(())` while keeping
/// the attacker's matrix makes the caller believe it selected a setup it did
/// not actually get.
#[test]
fn ajtai_registry_rejects_conflicting_well_formed_public_parameters() {
    const COLS: usize = 139;
    const KAPPA: usize = 2;

    let zero_ring = Rq([F::ZERO; D]);
    set_global_pp(PP::<Rq> {
        kappa: KAPPA,
        m: COLS,
        d: D,
        m_rows: vec![vec![zero_ring; COLS]; KAPPA],
    })
    .expect("attacker installs a well-shaped all-zero PP first");

    let mut nonzero_ring = zero_ring;
    nonzero_ring.0[0] = F::ONE;
    let replacement = set_global_pp(PP::<Rq> {
        kappa: KAPPA,
        m: COLS,
        d: D,
        m_rows: vec![vec![nonzero_ring; COLS]; KAPPA],
    });

    let installed = get_global_pp_for_dims(D, COLS).expect("registered PP");
    let silently_kept_attacker_pp = installed.m_rows[0][0] == zero_ring;

    let log = AjtaiSModule::from_global_for_dims(D, COLS).expect("Ajtai module from global registry");
    let zero_message = Mat::zero(D, COLS, F::ZERO);
    let mut distinct_message = zero_message.clone();
    distinct_message[(0, 0)] = F::ONE;
    let binding_collision = log.commit(&zero_message) == log.commit(&distinct_message);

    assert!(
        replacement.is_err() && silently_kept_attacker_pp && binding_collision,
        "setup-authority failure: the registry did not reject and preserve its first PP atomically (replacement={replacement:?}, preserved_first={silently_kept_attacker_pp}, first_pp_collision={binding_collision})"
    );
}

/// Shape-derived parameters are a soundness contract, not a caller hint. A
/// parameter set derived for `t = 1` must not be accepted with a structure
/// whose actual `t` supports only a lower shape-derived lambda.
#[test]
fn preprocessing_rejects_params_derived_for_a_different_ccs_shape() {
    let structure = CcsStructure::new_sparse(
        vec![CcsMatrix::Identity { n: 1 }; ACTUAL_MATRIX_COUNT],
        SparsePoly::<F>::new(ACTUAL_MATRIX_COUNT, Vec::new()),
    )
    .expect("large-arity zero CCS structure");

    let understated = config::ccs_params(
        structure.n,
        structure.m,
        /* matrix_count = */ 1,
        /* poly_degree = */ 0,
    )
    .expect("parameters derived for the understated shape");
    let actual = config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("parameters derived for the actual shape");

    assert!(
        actual.lambda() < understated.lambda(),
        "the larger matrix census must derive a lower lambda (actual={}, understated={})",
        actual.lambda(),
        understated.lambda(),
    );
    assert!(
        neo_reductions::engines::pi_ccs_joint::build_joint_dims(understated.inner(), &structure, 1, 0).is_err(),
        "the selected PiCCS dimension check accepted parameters for the wrong matrix count"
    );

    support::install_ajtai_module(&understated, &structure);
    let result = preprocess(understated, structure, Some(0));
    assert!(
        result.is_err(),
        "soundness-parameter failure: preprocessing accepted params derived for t=1 with an actual t={ACTUAL_MATRIX_COUNT} structure whose padded-row field/fork budget is unsupported"
    );
}

/// `PP` is serializable and its backing rows are public, but the global
/// registry and lifecycle preprocessing validate only its header dimensions
/// and kappa. A malformed verifier-owned setup must be rejected before an
/// ordinary instance constructor reaches an infallible commitment indexing
/// path.
#[test]
fn preprocessing_rejects_ajtai_pp_with_missing_matrix_rows() {
    const AJTAI_COLS: usize = 137;
    let width = D * AJTAI_COLS;
    let structure = CcsStructure::new(vec![Mat::zero(1, width, F::ZERO)], SparsePoly::<F>::new(1, Vec::new()))
        .expect("wide zero CCS structure");
    let params = config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("shape-specific params");

    set_global_pp(PP::<Rq> {
        kappa: params.kappa() as usize,
        m: AJTAI_COLS,
        d: D,
        m_rows: Vec::new(),
    })
    .expect("current PP registry accepts a header-only malformed setup");

    let prep = match preprocess(params, structure, Some(D)) {
        Err(_) => return,
        Ok(prep) => prep,
    };
    let assignment = vec![F::ZERO; width];
    let result = catch_unwind(AssertUnwindSafe(|| {
        CcsInstance::from_low_norm_assignment(&prep.params, &prep.log, prep.structure(), &assignment, D)
    }));

    assert!(
        result.is_ok(),
        "setup-validation failure: preprocessing accepted an Ajtai PP with no matrix rows, then instance construction panicked while committing"
    );
    assert!(
        result.unwrap().is_err(),
        "setup-validation failure: a malformed Ajtai PP with no matrix rows was accepted through instance construction"
    );
}

/// `public_input_len` is part of the verifier-key identity, so changing the
/// live policy after preprocessing must not leave the old key usable.  A key
/// advertised for one public ring must never prove and verify a two-public-ring
/// language merely through safe mutation of its context.
#[test]
fn verifier_rejects_public_input_policy_drift_after_key_derivation() {
    let structure = CcsStructure::new(vec![Mat::zero(1, 2 * D, F::ZERO)], SparsePoly::<F>::new(1, Vec::new()))
        .expect("two-ring zero relation");
    let params = config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("shape-specific params");
    support::install_ajtai_module(&params, &structure);

    let mut stale =
        preprocess(params.clone(), structure.clone(), Some(D)).expect("context advertised for one public ring");
    let honest_two =
        preprocess(params, structure, Some(2 * D)).expect("reference context advertised for two public rings");
    let advertised_one = stale.vk.digest();
    assert_ne!(
        advertised_one,
        honest_two.vk.digest(),
        "fixture precondition: verifier-key identity normally binds public-input arity"
    );

    let mut assignment = vec![F::ZERO; 2 * D];
    assignment[D] = F::ONE;
    let instance =
        CcsInstance::from_low_norm_assignment(&stale.params, &stale.log, stale.structure(), &assignment, 2 * D)
            .expect("valid two-public-ring instance");
    assert!(
        prove(&stale, [vec![instance.clone()]]).is_err(),
        "fixture precondition: the originally advertised arity-one policy rejects this arity-two statement"
    );

    stale.public_input_len = Some(2 * D);
    assert_eq!(
        stale.vk.digest(),
        advertised_one,
        "safe policy mutation leaves the once-derived verifier key stale"
    );
    let result = prove(&stale, [vec![instance]]);

    assert!(
        result.is_err(),
        "verifier-authority failure: key {advertised_one:?}, derived for one public ring, accepted a two-public-ring proof after its live policy drifted ({result:?})"
    );
}
