//! Cache-hygiene gates for [`Preprocessing`].
//!
//! `Preprocessing` carries verifier-derived `structure_digest` +
//! `optimized_cache` fields, both memoized from `structure` at
//! construction time. These tests pin two API-hygiene properties:
//!
//! 1. Right after `preprocess`, the cached fields agree with what
//!    recomputation from `structure` would produce. Catches a
//!    constructor that silently writes the wrong digest or skips the
//!    cache build.
//! 2. The cache-coupled fields are exposed through read-only accessors,
//!    so an integration-test caller cannot mutate `prep.structure` and
//!    leave the memoized digest/cache stale.
//!
//! These are not protocol-soundness tests — protocol authority is the
//! verifier-owned structure digest, not the optimized execution cache.
//! They're guardrails against shipping a stale cache into a code path
//! that would consume it.

use neo_ccs::matrix::Mat as NeoMat;
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsStructure, SparsePoly};
use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::nifs;
use neo_fold_clean::paper::relations::{CcsClaim, CcsInstance, CcsWitness};
use neo_fold_clean::{config, preprocess};
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

#[path = "../support/mod.rs"]
mod support;

/// Tiny R1CS shape — one constraint `z[0] = z[1] * z[2]`, padded to
/// `neo_math::D` columns. Keeps the cache build cheap so this test
/// stays cap-friendly.
fn one_product_r1cs() -> R1cs {
    let m = neo_math::D;
    let mut a = NeoMat::zero(1, m, F::default());
    a[(0, 1)] = F::ONE;
    let mut b = NeoMat::zero(1, m, F::default());
    b[(0, 2)] = F::ONE;
    let mut c = NeoMat::zero(1, m, F::default());
    c[(0, 0)] = F::ONE;
    R1cs { a, b, c, m_in: D }
}

#[test]
fn preprocessing_cached_structure_digest_matches_structure() {
    let r1cs = one_product_r1cs();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 0xCACE_0001).expect("preprocess");

    let recomputed = neo_fold_clean::paper::digest::structure_digest(prep.structure());
    assert_eq!(
        *prep.structure_digest(),
        recomputed,
        "preprocess must memoize the same digest the function would recompute"
    );
    let shape = prep.optimized_cache().shape();
    assert_eq!(
        shape,
        (prep.structure().n, prep.structure().m, prep.structure().t()),
        "optimized cache shape must match the structure it was built from"
    );
    prep.validate_cached_structure()
        .expect("freshly built preprocessing must validate");
}

#[test]
fn preprocessing_cache_accessors_expose_read_only_views() {
    let r1cs = one_product_r1cs();
    let prep = direct_ccs::preprocess_seeded(&r1cs, 0xCACE_0002).expect("preprocess");

    // These accessors intentionally return shared references. A stale
    // cache used to be constructible by mutating public fields after
    // `preprocess`; now that code does not compile from integration
    // tests or downstream crates.
    let _structure = prep.structure();
    let _digest = prep.structure_digest();
    let _cache = prep.optimized_cache();
    let _params: &neo_fold_clean::paper::params::Params = prep.params();
    let _setup: &neo_ajtai::AjtaiSModule = prep.commitment_scheme();
    let _key: &neo_fold_clean::paper::construction2::VerifierKey = prep.verifier_key();
    assert_eq!(prep.public_input_len(), Some(r1cs.m_in));

    prep.validate_cached_structure()
        .expect("read-only accessor use must leave caches valid");
}

#[test]
fn seeded_preprocessing_does_not_share_equal_shaped_setup_state() {
    let r1cs = one_product_r1cs();
    let seed_a = 0xCACE_1001;
    let seed_b = 0xCACE_1002;
    let prep_a = direct_ccs::preprocess_seeded(&r1cs, seed_a).expect("seed A preprocessing");
    let prep_b = direct_ccs::preprocess_seeded(&r1cs, seed_b).expect("seed B preprocessing");
    let prep_a_again = direct_ccs::preprocess_seeded(&r1cs, seed_a).expect("second seed A preprocessing");

    assert_ne!(prep_a.verifier_key().digest(), prep_b.verifier_key().digest());
    assert_eq!(prep_a.verifier_key().digest(), prep_a_again.verifier_key().digest());
    assert_eq!(
        prep_a
            .commitment_scheme()
            .seeded_params()
            .expect("seed A descriptor")
            .1[..8],
        seed_a.to_le_bytes()
    );
    assert_eq!(
        prep_b
            .commitment_scheme()
            .seeded_params()
            .expect("seed B descriptor")
            .1[..8],
        seed_b.to_le_bytes()
    );
}

#[test]
fn nifs_rejects_high_norm_fresh_witness_even_when_digits_are_low_norm() {
    let first = NeoMat::identity(2);
    let structure = CcsStructure::new(vec![first], SparsePoly::new(1, vec![])).expect("test structure shape is valid");
    assert_eq!(structure.n, structure.m, "test precondition: square CCS");
    assert!(
        structure.matrices[0].is_identity(),
        "test precondition: M0 is identity; this isolates raw-Z NC checking from the paper's M0=WLOG assumption"
    );

    let params = config::r1cs_params(structure.n, structure.m).expect("test params");
    support::install_ajtai_module(&params, &structure);
    let result = preprocess(params, structure, Some(0));
    let Ok(prep) = result else {
        return;
    };

    let mut z_mat = NeoMat::zero(D, prep.structure().m.div_ceil(D), F::ZERO);
    z_mat[(1, 0)] = F::from_u64(prep.params().b() as u64);
    let fresh = CcsInstance {
        claim: CcsClaim {
            adv: None,
            c: prep.commitment_scheme().commit(&z_mat),
            x: Vec::new(),
            m_in: 0,
        },
        witness: CcsWitness {
            w: vec![z_mat[(1, 0)]],
            Z: z_mat,
        },
    };

    let mut prove_tr = Transcript::session();
    let proof_result = nifs::prove(
        &mut prove_tr,
        prep.params(),
        prep.structure(),
        prep.optimized_cache(),
        prep.commitment_scheme(),
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        vec![fresh.clone()],
        &RunningInstance::default(),
    );

    if let Ok((_, proof)) = proof_result {
        let mut verify_tr = Transcript::session();
        let verified = nifs::verify(
            &mut verify_tr,
            prep.params(),
            prep.structure(),
            prep.optimized_cache(),
            prep.mix_rhos_commits(),
            prep.combine_b_pows(),
            &[fresh.claim],
            &RunningInstance::default(),
            &proof,
        );
        assert!(
            verified.is_err(),
            "Π_CCS/NC accepted a fresh witness with an out-of-alphabet Z entry whose base-b digits are individually low-norm"
        );
    }
}
