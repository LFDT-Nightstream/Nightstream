//! Spec-derived invariant tests for BarTransform.spec.md
//!
//! Each test corresponds to a row in the Invariant Obligations table.
//! The ct(bar(a)*b) = <a,b> identity is covered by the Lean theorem surface
//! (the exported-oracle conformance lane was removed; see git history).

#[path = "common/mod.rs"]
mod common;

use common::{random_fq_array, seeded_rng};
use neo_math::ring::{cf_inv, ct, superneo_bar_block, superneo_bar_vec, D};
use neo_math::Fq;
use p3_field::PrimeCharacteristicRing;
use rand::Rng;

/// BarTransform.spec.md: bar(a + b) = bar(a) + bar(b) (linearity — addition)
#[test]
fn bar_block_linear_add() {
    let mut rng = seeded_rng(0xBA20);
    for _ in 0..10 {
        let a = random_fq_array(&mut rng);
        let b = random_fq_array(&mut rng);

        let sum: [Fq; D] = std::array::from_fn(|i| a[i] + b[i]);
        let bar_sum = superneo_bar_block(sum);
        let bar_a = superneo_bar_block(a);
        let bar_b = superneo_bar_block(b);
        let bar_a_plus_b: [Fq; D] = std::array::from_fn(|i| bar_a[i] + bar_b[i]);

        assert_eq!(bar_sum, bar_a_plus_b);
    }
}

/// BarTransform.spec.md: bar(s * a) = s * bar(a) (linearity — scalar)
#[test]
fn bar_block_linear_scale() {
    let mut rng = seeded_rng(0xBA20);
    for _ in 0..10 {
        let a = random_fq_array(&mut rng);
        let s = Fq::from_u64(rng.random::<u64>());

        let scaled: [Fq; D] = std::array::from_fn(|i| s * a[i]);
        let bar_scaled = superneo_bar_block(scaled);
        let bar_a = superneo_bar_block(a);
        let s_bar_a: [Fq; D] = std::array::from_fn(|i| s * bar_a[i]);

        assert_eq!(bar_scaled, s_bar_a);
    }
}

/// BarTransform.spec.md: superneo_bar_vec is block-wise superneo_bar_block
#[test]
fn bar_vec_is_blockwise() {
    let mut rng = seeded_rng(0xBA20);
    // 3 blocks
    let v: Vec<Fq> = (0..3 * D)
        .map(|_| Fq::from_u64(rng.random::<u64>()))
        .collect();

    let bar_v = superneo_bar_vec(&v);

    // Manually apply block-wise
    for (blk_idx, chunk) in v.chunks_exact(D).enumerate() {
        let block: [Fq; D] = chunk.try_into().unwrap();
        let bar_block = superneo_bar_block(block);
        assert_eq!(&bar_v[blk_idx * D..(blk_idx + 1) * D], &bar_block);
    }
}

/// BarTransform.spec.md: superneo_bar_vec panics on misaligned input
#[test]
#[should_panic(expected = "superneo_bar_vec expects length multiple of D")]
fn bar_vec_panics_on_misaligned() {
    let v = vec![Fq::ZERO; D + 1];
    superneo_bar_vec(&v);
}

/// BarTransform.spec.md: ct(cf_inv(bar(a)) * cf_inv(b)) = <a, b> (Theorem 3)
/// Direct test without lean oracle vectors
#[test]
fn theorem_3_inner_product() {
    let mut rng = seeded_rng(0xBA20);
    for _ in 0..10 {
        let a = random_fq_array(&mut rng);
        let b = random_fq_array(&mut rng);

        let bar_a = superneo_bar_block(a);
        let ct_product = ct(&cf_inv(bar_a).mul(&cf_inv(b)));
        let dot: Fq = a
            .iter()
            .zip(b.iter())
            .fold(Fq::ZERO, |acc, (&x, &y)| acc + x * y);

        assert_eq!(ct_product, dot);
    }
}

/// BarTransform.spec.md: Bar matrix M^T * G = I (sanity check runs at first access)
/// This just forces the matrix to be built, which triggers the internal assertion.
#[test]
fn bar_matrix_sanity_check() {
    let _ = neo_math::superneo_bar_matrix();
}
