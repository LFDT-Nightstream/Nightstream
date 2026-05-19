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
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

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
    R1cs { a, b, c, m_in: 1 }
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

    prep.validate_cached_structure()
        .expect("read-only accessor use must leave caches valid");
}
