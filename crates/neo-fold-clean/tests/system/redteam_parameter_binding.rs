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

const ACTUAL_MATRIX_COUNT: usize = 4_096;

/// SuperNeo D.5 loses `(K + k + 1) / |C|` in the Π_RLC coordinate
/// extractor.  The production profile must charge that loss at the largest
/// fold width it advertises, not equate `lambda` with `log2(|C|)`.
#[test]
fn production_params_charge_pi_rlc_coordinate_extraction_loss() {
    let params = config::production_params();
    let challenge_set_size = 5u128.pow(params.d());
    let extraction_queries = params.max_fresh_count() as u128 + params.k_rho() as u128 + 1;
    let effective_bits = (challenge_set_size / extraction_queries).ilog2();

    assert!(
        effective_bits >= params.lambda(),
        "soundness-policy failure: Π_RLC's (K+k+1)/|C| extractor loss leaves only {effective_bits} whole bits at {extraction_queries} queries, below advertised lambda={}",
        params.lambda(),
    );
}

/// Appendix B.2 fixes the field-vector width `n_F = 2^30`; the SuperNeo
/// embedding packs that vector into `ceil(n_F / D)` ring elements, and the
/// D.8 estimator expands those ring elements back to `m_sis = n_F` scalar
/// columns. The runtime parameter's Ajtai `m` must therefore be the packed
/// ring width, and shape selection must reject field vectors above `n_F`.
#[cfg(target_pointer_width = "64")]
#[test]
fn ccs_params_reject_packed_width_above_ajtai_profile_cap() {
    let paper_n_f = 1usize << 30;
    let paper_ring_width = paper_n_f.div_ceil(D);
    let production = config::production_params();
    let over_cap = paper_n_f + 1;
    let result = config::ccs_params(1, over_cap, 1, 0);
    let profile_matches_paper = production.m() == paper_ring_width as u64;

    assert!(
        profile_matches_paper && result.is_err(),
        "soundness-policy failure: Appendix B.2 n_F={paper_n_f} packs to {paper_ring_width} ring columns, but production params.m={} and shape selection above n_F returned {result:?}",
        production.m(),
    );
}

/// Optional public-input arity is verifier policy. `None` must have a tagged
/// encoding distinct from every `Some(n)`, including the maximum machine word.
#[cfg(target_pointer_width = "64")]
#[test]
fn verifier_key_distinguishes_unbounded_from_maximum_public_input_arity() {
    let structure = CcsStructure::new(vec![Mat::identity(1)], SparsePoly::<F>::new(1, Vec::new()))
        .expect("one-coordinate zero relation");
    let params = config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("shape-specific params");
    support::install_ajtai_module(&params, &structure);

    let unbounded = preprocess(params.clone(), structure.clone(), None).expect("unbounded public-input policy");
    let maximum = preprocess(params, structure, Some(usize::MAX))
        .expect("current preprocessing accepts the maximum arity policy");

    let instance =
        CcsInstance::from_low_norm_assignment(&unbounded.params, &unbounded.log, unbounded.structure(), &[F::ZERO], 1)
            .expect("one-coordinate instance");
    let audit = prove(&unbounded, [vec![instance]]).expect("prove under unbounded policy");
    let proof = finish_uncompressed(&unbounded, audit).expect("finish under unbounded policy");
    let unbounded_result = verify_uncompressed(&unbounded, &proof);
    let maximum_result = verify_uncompressed(&maximum, &proof);
    assert!(
        unbounded_result.is_ok() && maximum_result.is_err(),
        "fixture requires different verifier languages (unbounded={unbounded_result:?}, maximum={maximum_result:?})"
    );

    let same_vk = unbounded.vk.digest() == maximum.vk.digest();
    let same_boundary = neo_fold_clean::paper::digest::initial_boundary_digest(unbounded.structure_digest(), None)
        == neo_fold_clean::paper::digest::initial_boundary_digest(maximum.structure_digest(), Some(usize::MAX));
    assert!(
        !same_vk && !same_boundary,
        "verifier-policy encoding failure: None and Some(usize::MAX) define different accepted languages but collide (same_vk={same_vk}, same_initial_boundary={same_boundary})"
    );
}

/// Parameter-selection margins are verifier security policy. A public `u32`
/// margin that cannot be represented by the internal signed slack type must
/// be rejected, not narrowed to a negative number that makes the comparison
/// vacuously succeed.
#[test]
fn ccs_parameter_selection_rejects_unrepresentable_safety_margin() {
    let result = neo_fold_clean::Params::for_ccs_shape_with(1, 1, 0, 100, u32::MAX);
    assert!(
        result.is_err(),
        "soundness-policy failure: u32 safety margin wrapped negative during signed comparison and was accepted as {result:?}"
    );
}

/// SuperNeo's polynomial parameter is the strict degree bound `u`, so a
/// relation of actual degree `d` must be budgeted with `u=d+1`. Parameter
/// selection must match the runtime sumcheck degree rather than underpricing
/// every nonconstant relation by one at a security boundary.
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

    let ell = 7u128;
    let u = DEGREE as u128 + 1;
    let fresh = 61u128;
    let k = 14u128;
    let d = 54u128;
    let t = 1u128;
    let true_factor = ell * u + (2 * fresh + k) * ell.max(k * t * d);
    let true_policy = params.inner().extension_check_factor(true_factor);

    assert!(
        true_policy
            .as_ref()
            .is_ok_and(|summary| summary.slack_bits >= 2),
        "soundness-policy failure: actual degree {DEGREE} requires strict bound u={}, but selected lambda={} has {true_policy:?}",
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
    let structure =
        CcsStructure::new(vec![Mat::identity(1)], SparsePoly::<F>::new(1, Vec::new())).expect("toy zero relation");
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
        preprocess_with_test_log(params.clone(), structure.clone(), weak_log, Some(1)).expect("weak verifier context");
    let strong = preprocess_with_test_log(params, structure, strong_log, Some(1)).expect("strong verifier context");
    let instance = CcsInstance::from_low_norm_assignment(&weak.params, &weak.log, weak.structure(), &[F::ONE], 1)
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
        replacement.is_err() && !silently_kept_attacker_pp && !binding_collision,
        "setup-authority failure: a conflicting verifier PP returned {replacement:?}, the registry silently kept the attacker PP={silently_kept_attacker_pp}, and two distinct low-norm messages collide={binding_collision}"
    );
}

/// Shape-derived parameters are a soundness contract, not a caller hint. A
/// parameter set derived for `t = 1` must not be accepted with a structure
/// whose actual `t` makes SuperNeo D.4 unsupported even at the configured
/// minimum lambda. Otherwise the runtime's smaller `ell * d_sc` check silently
/// approves a verifier context below the advertised security floor.
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

    assert!(
        config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree(),).is_err(),
        "the actual shape should exceed the configured s=2/min-lambda D.4 budget"
    );
    neo_reductions::engines::utils::build_dims_and_policy(understated.inner(), &structure)
        .expect("the runtime's incomplete shape check currently accepts the understated params");

    support::install_ajtai_module(&understated, &structure);
    let result = preprocess(understated, structure, Some(0));
    assert!(
        result.is_err(),
        "soundness-parameter failure: preprocessing accepted params derived for t=1 with an actual t=4096 structure whose full SuperNeo D.4 budget is unsupported"
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

    let prep = match preprocess(params, structure, Some(1)) {
        Err(_) => return,
        Ok(prep) => prep,
    };
    let assignment = vec![F::ZERO; width];
    let result = catch_unwind(AssertUnwindSafe(|| {
        CcsInstance::from_low_norm_assignment(&prep.params, &prep.log, prep.structure(), &assignment, 1)
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
/// advertised for one public coordinate must never prove and verify a
/// two-coordinate language merely through safe mutation of its context.
#[test]
fn verifier_rejects_public_input_policy_drift_after_key_derivation() {
    let structure = CcsStructure::new(vec![Mat::zero(1, 2, F::ZERO)], SparsePoly::<F>::new(1, Vec::new()))
        .expect("two-coordinate zero relation");
    let params = config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("shape-specific params");
    support::install_ajtai_module(&params, &structure);

    let mut stale =
        preprocess(params.clone(), structure.clone(), Some(1)).expect("context advertised for one public coordinate");
    let honest_two =
        preprocess(params, structure, Some(2)).expect("reference context advertised for two public coordinates");
    let advertised_one = stale.vk.digest();
    assert_ne!(
        advertised_one,
        honest_two.vk.digest(),
        "fixture precondition: verifier-key identity normally binds public-input arity"
    );

    let instance =
        CcsInstance::from_low_norm_assignment(&stale.params, &stale.log, stale.structure(), &[F::ZERO, F::ONE], 2)
            .expect("valid two-public-input instance");
    assert!(
        prove(&stale, [vec![instance.clone()]]).is_err(),
        "fixture precondition: the originally advertised arity-one policy rejects this arity-two statement"
    );

    stale.public_input_len = Some(2);
    assert_eq!(
        stale.vk.digest(),
        advertised_one,
        "safe policy mutation leaves the once-derived verifier key stale"
    );
    let audit = prove(&stale, [vec![instance]]).expect("prover follows the mutated live policy");
    let proof = finish_uncompressed(&stale, audit).expect("finalize under the stale-key context");
    let result = verify_uncompressed(&stale, &proof);

    assert!(
        result.is_err(),
        "verifier-authority failure: key {advertised_one:?}, derived for public-input arity 1, accepted an arity-2 proof after its public mutable policy drifted ({result:?})"
    );
}
