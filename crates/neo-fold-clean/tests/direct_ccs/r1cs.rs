//! Happy-path tests for the direct-CCS frontend.
//!
//! Covers:
//! - R1CS shape validation accepts well-formed (A, B, C, m_in).
//! - R1CS satisfaction check accepts a satisfying assignment and rejects
//!   a non-satisfying one with the offending row index.
//! - `preprocess_seeded` produces a working `Preprocessing`.
//! - `build_instance` succeeds for a satisfying low-norm assignment.
//! - The instance can be folded through the IVC chain end-to-end.

use neo_ccs::matrix::Mat as NeoMat;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use neo_fold_clean::config;
use neo_fold_clean::frontends::direct_ccs::{self, FrontendError, R1cs};
use neo_fold_clean::{finish_uncompressed, prove, verify_uncompressed, FoldSchedule};

/// Minimal R1CS: one constraint `(z[1] + z[2]) · z[0] = z[3]`.
/// `z[0] = 1` (constant), `z[1] = a`, `z[2] = b`, `z[3] = c`.
/// Public input is z[..3] = [1, a, b]; witness is z[3..] = [c].
///
/// The matrix width is padded to `neo_math::D` because the engine's
/// SuperNeo packed-witness shape expects a Z matrix of D rows. Smaller
/// structure widths trigger downstream validation issues we'd rather
/// rule out at frontend-level — pad here once.
fn three_term_addition() -> R1cs {
    let m = neo_math::D;
    let mut a = NeoMat::zero(1, m, F::default());
    a[(0, 1)] = F::ONE;
    a[(0, 2)] = F::ONE;

    let mut b = NeoMat::zero(1, m, F::default());
    b[(0, 0)] = F::ONE;

    let mut c = NeoMat::zero(1, m, F::default());
    c[(0, 3)] = F::ONE;

    R1cs { a, b, c, m_in: 3 }
}

#[test]
fn r1cs_shape_validation_accepts_well_formed() {
    let r1cs = three_term_addition();
    r1cs.validate_shape().expect("shape valid");
}

#[test]
fn r1cs_shape_validation_rejects_mismatch() {
    let r1cs = three_term_addition();
    let mut bad = r1cs;
    bad.b = NeoMat::zero(1, 5, F::default()); // wrong column count
    match bad.validate_shape() {
        Err(FrontendError::ShapeMismatch { .. }) => {}
        other => panic!("expected ShapeMismatch, got {:?}", other),
    }
}

#[test]
fn r1cs_satisfaction_accepts_valid_assignment() {
    let r1cs = three_term_addition();
    // (a, b, c) = (1, 0, 1): (a + b) · 1 = c. ✓
    let z = pad_z(vec![F::ONE, F::ONE, F::ZERO, F::ONE], r1cs.m());
    r1cs.is_satisfied_by(&z).expect("satisfying assignment");
}

#[test]
fn r1cs_satisfaction_rejects_invalid_assignment() {
    let r1cs = three_term_addition();
    // (a, b, c) = (1, 1, 0): (1 + 1) · 1 = 2 ≠ 0 = c.
    let z = pad_z(vec![F::ONE, F::ONE, F::ONE, F::ZERO], r1cs.m());
    match r1cs.is_satisfied_by(&z) {
        Err(FrontendError::Unsatisfied { row: 0 }) => {}
        other => panic!("expected Unsatisfied row 0, got {:?}", other),
    }
}

#[test]
fn build_instance_succeeds_for_satisfying_low_norm_assignment() {
    let r1cs = three_term_addition();
    let prep = direct_ccs::preprocess_seeded(&r1cs, /* seed = */ 1).expect("preprocess");
    assert!(prep.params.has_production_core());
    assert_eq!(prep.params.k_rho(), config::K_RHO);

    // (a, b, c) = (1, 0, 1) — pad to structure.m with zeros.
    let z = pad_z(vec![F::ONE, F::ONE, F::ZERO, F::ONE], prep.structure().m);
    let _instance = direct_ccs::build_instance(&prep, &r1cs, &z).expect("build_instance");
}

#[test]
fn end_to_end_chain_proof_for_three_term_addition() {
    let r1cs = three_term_addition();
    let prep = direct_ccs::preprocess_seeded(&r1cs, /* seed = */ 2).expect("preprocess");

    // Two valid (a, b, c) triples.
    let z1 = pad_z(vec![F::ONE, F::ZERO, F::ONE, F::ONE], prep.structure().m); // (a=0, b=1, c=1)
    let z2 = pad_z(vec![F::ONE, F::ONE, F::ZERO, F::ONE], prep.structure().m); // (a=1, b=0, c=1)

    let i1 = direct_ccs::build_instance(&prep, &r1cs, &z1).expect("instance 1");
    let i2 = direct_ccs::build_instance(&prep, &r1cs, &z2).expect("instance 2");

    let batches = FoldSchedule::RowsPerStep(1)
        .partition(vec![i1, i2])
        .expect("partition");

    let proof = prove(&prep, batches).expect("prove");
    let finished = finish_uncompressed(&prep, proof).expect("finish_uncompressed");
    verify_uncompressed(&prep, &finished).expect("verify_uncompressed");
}

/// SuperNeo steady-state max-fresh path: `MAX_FRESH_K = 61` fresh + 14
/// running claims = 75 RLC claims, just under the `(K + k_rho) · T · (b-1)
/// < B = 2^14` bound (`75 · 216 = 16200`). 122 simple direct-CCS instances
/// partitioned as two batches of 61 prove, finalize, and verify.
///
/// The first `extend` deposits 61 fresh into `latest` without folding; the
/// second folds those 61 fresh into the (empty) running, producing the
/// `k_rho = 14`-sized running accumulator; `finish_uncompressed` then
/// flushes the trailing latest (61 fresh) against that 14-running, which
/// is exactly the `61 + 14 = 75` RLC-claim shape this test exists to
/// exercise.
#[test]
fn end_to_end_chain_proof_for_max_fresh_k61_steady_state() {
    let r1cs = three_term_addition();
    let prep = direct_ccs::preprocess_seeded(&r1cs, /* seed = */ 7).expect("preprocess");

    let k = neo_params::goldilocks_paper_b2::MAX_FRESH_K as usize;
    let total = 2 * k;
    let instances: Vec<_> = (0..total)
        .map(|_| {
            let z = pad_z(vec![F::ONE, F::ONE, F::ZERO, F::ONE], prep.structure().m);
            direct_ccs::build_instance(&prep, &r1cs, &z).expect("instance")
        })
        .collect();

    let batches = FoldSchedule::RowsPerStep(k)
        .partition(instances)
        .expect("partition");
    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0].len(), k);
    assert_eq!(batches[1].len(), k);

    let proof = prove(&prep, batches).expect("prove K=MAX_FRESH_K");
    let finished = finish_uncompressed(&prep, proof).expect("finish at K=61, k_rho=14 must pass RLC bound");
    verify_uncompressed(&prep, &finished).expect("verify_uncompressed");
}

/// Steady-state RLC bound violation. First batch = 61 (passes Π_RLC at
/// `61 + 0 = 61` claims), second batch = 62 ends up as the trailing
/// `latest`; finalization folds it against the 14-claim running, giving
/// `62 + 14 = 76` RLC claims and `76 · 216 = 16416 ≥ 2^14 = 16384` —
/// `finish_uncompressed` must reject.
#[test]
fn finish_uncompressed_rejects_steady_state_above_max_fresh() {
    let r1cs = three_term_addition();
    let prep = direct_ccs::preprocess_seeded(&r1cs, /* seed = */ 11).expect("preprocess");

    let k = neo_params::goldilocks_paper_b2::MAX_FRESH_K as usize;
    let build = || {
        let z = pad_z(vec![F::ONE, F::ONE, F::ZERO, F::ONE], prep.structure().m);
        direct_ccs::build_instance(&prep, &r1cs, &z).expect("instance")
    };

    let first: Vec<_> = (0..k).map(|_| build()).collect();
    let second: Vec<_> = (0..(k + 1)).map(|_| build()).collect();

    let proof = prove(&prep, [first, second]).expect("prove (RLC bound only checked on next fold)");
    let err = finish_uncompressed(&prep, proof)
        .err()
        .expect("finish_uncompressed must reject 62 fresh + 14 running (steady-state \u{03A0}_RLC bound)");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("RlcBoundViolated") || msg.to_lowercase().contains("rlc"),
        "expected steady-state \u{03A0}_RLC bound violation, got {msg}",
    );
}

// ── helpers ───────────────────────────────────────────────────────────────

fn pad_z(mut z: Vec<F>, new_len: usize) -> Vec<F> {
    z.resize(new_len, F::ZERO);
    z
}
