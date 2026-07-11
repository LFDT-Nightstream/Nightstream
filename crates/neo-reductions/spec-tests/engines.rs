//! Spec-derived invariant tests for Engines.spec.md
//!
//! Each test corresponds to a row in the Invariant Obligations table.

#[path = "common/mod.rs"]
mod common;

use neo_ccs::{CcsMatrix, CcsStructure, SparsePoly, Term};
use neo_math::{Fq, K};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

// ---------------------------------------------------------------------------
// 1. eq_points symmetry: eq(p,q) == eq(q,p)
// ---------------------------------------------------------------------------

/// Engines.spec.md: eq_points symmetry
#[test]
fn eq_points_symmetry() {
    let p = vec![
        K::from(Fq::from_u64(3)),
        K::from(Fq::from_u64(7)),
        K::from(Fq::from_u64(11)),
    ];
    let q = vec![
        K::from(Fq::from_u64(5)),
        K::from(Fq::from_u64(2)),
        K::from(Fq::from_u64(9)),
    ];

    let eq_pq = eq_points_manual(&p, &q);
    let eq_qp = eq_points_manual(&q, &p);
    assert_eq!(eq_pq, eq_qp, "eq(p,q) should equal eq(q,p)");
}

// ---------------------------------------------------------------------------
// 2. eq_points identity: eq(p,p) for boolean points
// ---------------------------------------------------------------------------

/// Engines.spec.md: eq_points identity for boolean points
#[test]
fn eq_points_identity_boolean() {
    let booleans: Vec<Vec<K>> = vec![
        vec![K::ZERO, K::ZERO, K::ZERO],
        vec![K::ONE, K::ZERO, K::ONE],
        vec![K::ONE, K::ONE, K::ONE],
        vec![K::ZERO, K::ONE, K::ZERO],
    ];

    for p in &booleans {
        let result = eq_points_manual(p, p);
        assert_eq!(result, K::ONE, "eq(p,p) should be 1 for boolean point");
    }
}

// ---------------------------------------------------------------------------
// 3. eq_points orthogonality
// ---------------------------------------------------------------------------

/// Engines.spec.md: eq(p,q) = 0 for distinct boolean points
#[test]
fn eq_points_orthogonality_boolean() {
    let p = vec![K::ONE, K::ZERO];
    let q = vec![K::ZERO, K::ONE];

    let result = eq_points_manual(&p, &q);
    assert_eq!(result, K::ZERO, "eq(p,q) should be 0 for distinct boolean points");
}

// ---------------------------------------------------------------------------
// 4. PiCcsEngine trait dispatch
// ---------------------------------------------------------------------------

/// Engines.spec.md: PiCcsEngine trait compiles and dispatches
#[test]
fn engine_trait_dispatch() {
    use neo_reductions::engines::pi_ccs::{OptimizedEngine, PiCcsEngine};

    let _engine = OptimizedEngine;
    fn _assert_engine<E: PiCcsEngine>(_e: &E) {}
    _assert_engine(&OptimizedEngine);
}

// ---------------------------------------------------------------------------
// 5. FoldingMode construction
// ---------------------------------------------------------------------------

/// Engines.spec.md: FoldingMode enum variants
#[test]
fn folding_mode_construction() {
    let mode = neo_reductions::pi_ccs::FoldingMode::Optimized;
    let _debug = format!("{mode:?}");
}

// ---------------------------------------------------------------------------
// 6. Challenges struct field access
// ---------------------------------------------------------------------------

/// Engines.spec.md: Challenges struct field access
#[test]
fn challenges_field_access() {
    let c = neo_reductions::Challenges {
        alpha: vec![K::from(Fq::from_u64(1)); 3],
        beta_a: vec![K::from(Fq::from_u64(2)); 3],
        beta_r: vec![K::from(Fq::from_u64(3)); 4],
        beta_m: vec![K::from(Fq::from_u64(4)); 2],
        gamma: K::from(Fq::from_u64(5)),
    };

    assert_eq!(c.alpha.len(), 3);
    assert_eq!(c.beta_a.len(), 3);
    assert_eq!(c.beta_r.len(), 4);
    assert_eq!(c.beta_m.len(), 2);
    assert_ne!(c.gamma, K::ZERO);
}

// ---------------------------------------------------------------------------
// 7. Concrete CCS shape drives the extension policy
// ---------------------------------------------------------------------------

#[test]
fn engine_rejects_parameters_selected_for_fewer_matrices() {
    const SHAPE: usize = 1 << 24;
    const ACTUAL_T: usize = 1_000;

    let params = NeoParams::goldilocks_auto_ccs_with(SHAPE, 1, 8, 96, 2).expect("one-matrix parameter profile");
    let matrices = (0..ACTUAL_T)
        .map(|_| CcsMatrix::Identity { n: SHAPE })
        .collect();
    let mut exps = vec![0; ACTUAL_T];
    exps[0] = 8;
    let structure = CcsStructure::new_sparse(matrices, SparsePoly::new(ACTUAL_T, vec![Term { coeff: Fq::ONE, exps }]))
        .expect("sparse synthetic CCS");

    assert!(
        neo_reductions::engines::utils::build_dims_and_policy(&params, &structure).is_err(),
        "engine preprocessing must charge the concrete structure's matrix count"
    );
}

// ---------------------------------------------------------------------------
// Helper: manual eq_points implementation for testing
// ---------------------------------------------------------------------------

fn eq_points_manual(p: &[K], q: &[K]) -> K {
    assert_eq!(p.len(), q.len());
    let mut result = K::ONE;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        let term = (K::ONE - pi) * (K::ONE - qi) + pi * qi;
        result *= term;
    }
    result
}
