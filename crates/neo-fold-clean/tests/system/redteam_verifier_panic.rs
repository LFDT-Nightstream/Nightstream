//! Retained red-team regressions for verifier panic boundaries.

#[path = "../support/mod.rs"]
mod support;

use std::panic::{catch_unwind, AssertUnwindSafe};

use neo_ccs::Mat;
use neo_fold_clean::engine::r1cs_circuit::{Lc, R1csBuilder, Var};
use neo_fold_clean::frontends::direct_ccs::{self, R1cs};
use neo_fold_clean::frontends::f_prime::FPrimeImageLayout;
use neo_fold_clean::paper::construction2::ProofState;
use neo_fold_clean::paper::terminal_ce::merkle::{
    enforce_terminal_ce_merkle_root_from_leaf, terminal_ce_merkle_node, terminal_ce_merkle_root_from_leaf,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

/// The audit/decider preflight receives the final witness matrix from the
/// prover. It must validate its shape before calling the infallible Ajtai
/// commitment API; a malformed proof is an ordinary rejection, not license
/// to trip an internal assertion in the verifier process.
#[test]
fn audit_verifier_rejects_malformed_final_witness_without_panicking() {
    let prep = support::toy_preprocessing();
    let audit =
        neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 0)]]).expect("one-step audit proof");
    let mut finalized = neo_fold_clean::finish_uncompressed_with_audit(&prep, audit).expect("finalized audit proof");

    let ProofState::Active { running, .. } = &mut finalized.proof.state.proof else {
        panic!("finalized proof must be active");
    };
    let running = running
        .as_materialized_mut()
        .expect("fixture uses a materialized running accumulator");
    let original_cols = running.witnesses[0].cols();
    running.witnesses[0] = Mat::zero(neo_math::D - 1, original_cols, F::ZERO);

    let result = catch_unwind(AssertUnwindSafe(|| {
        neo_fold_clean::verify_uncompressed_audit(&prep, &finalized)
    }));

    assert!(
        result.is_ok(),
        "verifier availability failure: a malformed proof witness reached Ajtai's internal shape assertion"
    );
    assert!(
        result.unwrap().is_err(),
        "soundness failure: audit verifier accepted a malformed final witness matrix"
    );
}

/// A CE claim may carry a canonically shaped matrix whose declared column
/// count disagrees with its separate `m_in` metadata.  The terminal verifier
/// must reject that cross-field mismatch before hashing active X columns;
/// otherwise ordinary `Mat` indexing turns malformed proof data into a panic.
#[test]
fn terminal_verifier_rejects_m_in_x_column_mismatch_without_panicking() {
    let prep = support::toy_preprocessing();
    let audit = neo_fold_clean::prove(&prep, vec![vec![support::toy_instance(&prep, 30)]]).expect("one-step proof");
    let mut finished = neo_fold_clean::finish_uncompressed(&prep, audit).expect("finalized proof");

    let ProofState::Active { running, .. } = &finished.state.proof else {
        panic!("finalized proof must be active");
    };
    let mut forged_pre_running = running.materialize().expect("materialized final running");
    forged_pre_running.witnesses.clear();
    assert_eq!(forged_pre_running.claims[0].m_in, neo_math::D);
    forged_pre_running.claims[0].X = Mat::zero(neo_math::D, 0, F::ZERO);
    finished
        .final_fold
        .as_mut()
        .expect("one-step proof carries a terminal fold")
        .terminal_inputs
        .pre_final_running = forged_pre_running;

    let result = catch_unwind(AssertUnwindSafe(|| {
        neo_fold_clean::verify_uncompressed(&prep, &finished)
    }));

    assert!(
        result.is_ok(),
        "verifier availability failure: a canonical 54x0 matrix with m_in=54 reached unchecked active-column indexing"
    );
    assert!(
        result.unwrap().is_err(),
        "soundness failure: terminal verifier accepted a CE claim whose m_in disagrees with X.cols"
    );
}

/// `public_input_len=None` permits a generic program to choose its public
/// arity at the instance boundary, but one Π_RLC batch must still contain
/// claims from a single fixed relation. Locally valid arity-one and arity-zero
/// instances must be rejected before their differently-strided X matrices
/// reach the optimized combiner.
#[test]
fn lifecycle_rejects_mixed_public_input_relations_without_panicking() {
    let prep = support::toy_preprocessing_unfixed_public_input_len();
    let wide = support::toy_instance(&prep, 0);
    let narrow = neo_fold_clean::CcsInstance::from_low_norm_assignment(
        &prep.params,
        &prep.log,
        prep.structure(),
        &vec![F::ZERO; prep.structure().m],
        0,
    )
    .expect("valid arity-zero instance for the same CCS structure");
    assert_eq!((wide.claim.m_in, narrow.claim.m_in), (neo_math::D, 0));

    let audit = neo_fold_clean::prove(&prep, [vec![wide, narrow]])
        .expect("the first lifecycle step currently accepts the mixed batch");
    let result = catch_unwind(AssertUnwindSafe(|| neo_fold_clean::finish_uncompressed(&prep, audit)));

    assert!(
        matches!(result, Ok(Err(_))),
        "lifecycle must reject mixed public-input relations instead of panicking in Π_RLC"
    );
}

/// A relation with variables but no constraints is mathematically valid: its
/// empty row set is satisfied vacuously. The public R1CS validator and
/// row-wise checker both accept it, so preprocessing must either support the
/// relation or reject it as a normal frontend error. It must not reach the
/// infallible R1CS-to-CCS conversion and panic.
#[test]
fn direct_r1cs_preprocessing_rejects_zero_constraint_relation_without_panicking() {
    let zero_rows = Mat::zero(0, 1, F::ZERO);
    let r1cs = R1cs {
        a: zero_rows.clone(),
        b: zero_rows.clone(),
        c: zero_rows,
        m_in: 0,
    };
    r1cs.is_satisfied_by(&[F::ZERO])
        .expect("empty constraint set is satisfied vacuously");

    let result = catch_unwind(AssertUnwindSafe(|| direct_ccs::preprocess_seeded(&r1cs, 0x0BAD_CC50)));

    assert!(
        result.is_ok(),
        "frontend availability failure: a validator-accepted zero-constraint R1CS panicked during public preprocessing"
    );
    assert!(
        result.expect("panic checked above").is_err(),
        "validation failure: direct-CCS preprocessing silently accepted a zero-row relation that the reduction engine forbids"
    );
}

/// Public F' layout construction must reject a region-size sum that cannot be
/// represented by `usize`. The checked public constructor must reject the
/// configuration before any region cursor wraps.
#[test]
fn f_prime_image_layout_rejects_region_size_overflow_without_panicking() {
    let mut config = support::empty_f_prime_image_config();

    FPrimeImageLayout::try_new(config.clone()).expect("production shell layout");
    config.boundary_bits = usize::MAX;
    let result = catch_unwind(AssertUnwindSafe(|| FPrimeImageLayout::try_new(config)));
    assert!(
        result.is_ok(),
        "F' setup availability failure: malformed region sizes caused a panic"
    );
    assert!(result.unwrap().is_err(), "overflowing region sizes must reject");
}

fn merkle_digest(seed: u64) -> [F; 4] {
    [
        F::from_u64(seed),
        F::from_u64(seed + 1),
        F::from_u64(seed + 2),
        F::from_u64(seed + 3),
    ]
}

fn canonical_merkle_root(leaf: [F; 4], path: &[[F; 4]], index: usize) -> [F; 4] {
    let mut acc = leaf;
    for (level, sibling) in path.iter().copied().enumerate() {
        let bit = if level < usize::BITS as usize {
            (index >> level) & 1
        } else {
            0
        };
        acc = if bit == 0 {
            terminal_ce_merkle_node(acc, sibling)
        } else {
            terminal_ce_merkle_node(sibling, acc)
        };
    }
    acc
}

fn alloc_merkle_digest(builder: &mut R1csBuilder, digest: [F; 4]) -> [Var; 4] {
    digest.map(|value| builder.alloc(value))
}

/// A `usize` leaf index has exactly `usize::BITS` bits. For a deeper path,
/// every higher direction bit is mathematically zero; optimized Rust instead
/// masks an oversized shift count and reuses low bits unless the verifier
/// handles those levels explicitly.
#[test]
fn terminal_ce_merkle_native_treats_index_bits_above_usize_as_zero() {
    let leaf = merkle_digest(1);
    let path = (0..=usize::BITS)
        .map(|level| merkle_digest(100 + u64::from(level) * 10))
        .collect::<Vec<_>>();
    let index = 1usize;
    let expected = canonical_merkle_root(leaf, &path, index);
    let actual =
        terminal_ce_merkle_root_from_leaf(leaf, &path, index).expect("the index fits every finite usize path prefix");

    assert_eq!(
        actual, expected,
        "terminal CE Merkle verification reused a low index bit above usize::BITS"
    );
}

#[test]
fn terminal_ce_merkle_circuit_treats_index_bits_above_usize_as_zero() {
    let leaf = merkle_digest(1_000);
    let path = (0..=usize::BITS)
        .map(|level| merkle_digest(2_000 + u64::from(level) * 10))
        .collect::<Vec<_>>();
    let index = 1usize;
    let expected = canonical_merkle_root(leaf, &path, index);

    let mut builder = R1csBuilder::new();
    let leaf_vars = alloc_merkle_digest(&mut builder, leaf);
    let path_vars = path
        .iter()
        .copied()
        .map(|node| alloc_merkle_digest(&mut builder, node))
        .collect::<Vec<_>>();
    let root = enforce_terminal_ce_merkle_root_from_leaf(&mut builder, leaf_vars, &path_vars, index)
        .expect("the index fits every finite usize path prefix");
    for (wire, value) in root.into_iter().zip(expected) {
        builder.enforce_eq(&Lc::from_var(wire), &Lc::from_const(value));
    }

    assert!(
        builder.is_satisfied(),
        "recursive terminal CE Merkle verifier reused a low index bit above usize::BITS"
    );
}
