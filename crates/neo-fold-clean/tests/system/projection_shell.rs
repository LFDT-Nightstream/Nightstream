//! Road A Unit 2 — the projection trace primitive and its image
//! regions (Phase A: layout + native fill + parity; semantic rows are
//! Phase B, pinned red by
//! `ivc_invariants::projection_shell_semantic_rows_must_be_enforced`).

use neo_fold_clean::frontends::f_prime::image::FPrimeImageLayout;
use neo_fold_clean::frontends::f_prime::structure::{
    production_kmul_d2_ring_action_shell_image_config, production_kmul_ring_action_shell_image_config,
};
use neo_fold_clean::paper::f_prime::projection_trace::{
    encode_projection_identity, encode_projection_pair, encode_projection_shared, PROJECTION_IDENTITY_BITS,
    PROJECTION_IDENTITY_LANES, PROJECTION_PAIR_BITS, PROJECTION_PAIR_LANES, PROJECTION_SHARED_BITS,
    PROJECTION_SHARED_LANES,
};
use neo_math::field::KExtensions;
use neo_math::ring::D;
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

/// Deterministic SplitMix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn rho(&mut self) -> [F; D] {
        std::array::from_fn(|_| {
            let v = (self.next() % 5) as i64 - 2;
            if v >= 0 {
                F::from_u64(v as u64)
            } else {
                F::ZERO - F::from_u64((-v) as u64)
            }
        })
    }

    fn coeffs(&mut self) -> [F; D] {
        std::array::from_fn(|_| F::from_u64(self.next()))
    }
}

/// The width constants the image layout consumes, pinned. A change to
/// any of these is a protocol-surface change to the F' image.
#[test]
fn projection_region_widths_are_pinned() {
    assert_eq!(PROJECTION_PAIR_LANES, 329);
    assert_eq!(PROJECTION_IDENTITY_LANES, 326);
    assert_eq!(PROJECTION_SHARED_LANES, 274);
    assert_eq!(PROJECTION_PAIR_BITS, 329 * 64);
    assert_eq!(PROJECTION_IDENTITY_BITS, 326 * 64);
    assert_eq!(PROJECTION_SHARED_BITS, 274 * 64);
}

/// Honest fills produce exactly the declared lane counts, and the
/// batched identity residual is zero — the value the Phase B semantic
/// rows will enforce in-circuit.
#[test]
fn honest_fill_has_zero_identity_residual() {
    let mut rng = Rng(9);
    let beta = K::from_coeffs([F::from_u64(rng.next()), F::from_u64(rng.next())]);
    let (shared, powers) = encode_projection_shared(beta);
    assert_eq!(shared.len(), PROJECTION_SHARED_LANES);
    assert_eq!(powers.len(), D + 1);

    let pairs: Vec<([F; D], [F; D])> = (0..3).map(|_| (rng.rho(), rng.coeffs())).collect();
    let mut terms = Vec::new();
    for (rho, c) in &pairs {
        let (lanes, term) = encode_projection_pair(rho, c, &powers);
        assert_eq!(lanes.len(), PROJECTION_PAIR_LANES);
        terms.push(term);
    }

    let (lanes, residual) = encode_projection_identity(&pairs, &powers, &terms);
    assert_eq!(lanes.len(), PROJECTION_IDENTITY_LANES);
    assert_eq!(residual, K::ZERO, "honest identity residual must vanish");

    // A wrong pair term (e.g., a lying evaluation) leaves a nonzero
    // residual — what Phase B's final-identity row rejects.
    let mut bad_terms = terms.clone();
    bad_terms[1] += K::ONE;
    let (_, bad_residual) = encode_projection_identity(&pairs, &powers, &bad_terms);
    assert_ne!(bad_residual, K::ZERO, "a forged term must leave a residual");
}

/// The Road A production shell: projection regions in, D² pairs out,
/// integrated width measured and under the budget — with the D²
/// reference shell preserved for comparison.
#[test]
fn road_a_shell_is_measured_and_under_budget() {
    let road_a = FPrimeImageLayout::new(production_kmul_ring_action_shell_image_config());
    let d2 = FPrimeImageLayout::new(production_kmul_d2_ring_action_shell_image_config());

    println!("Road A shell: {} bits; D² reference: {} bits", road_a.end, d2.end);
    assert_eq!(road_a.end, 14_040_452, "integrated Road A shell width (pinned)");
    assert_eq!(d2.end, 94_330_948, "D² reference width (pinned)");
    assert!(road_a.projection.bits > 0, "projection regions present");
    assert_eq!(road_a.ring_action.bits, 0, "no D² pairs in the Road A shell");
}
