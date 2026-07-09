//! Road A projection trace primitive, image regions, and semantic rows.

use neo_fold_clean::frontends::f_prime::image::{FPrimeImage, FPrimeImageLayout};
use neo_fold_clean::frontends::f_prime::structure::{
    build_f_prime_structure, production_kmul_d2_ring_action_shell_image_config,
    production_kmul_ring_action_shell_image_config,
};
use neo_fold_clean::paper::f_prime::projection_trace::{
    encode_projection_identity, encode_projection_pair, encode_projection_shared, PROJECTION_IDENTITY_BITS,
    PROJECTION_IDENTITY_LANES, PROJECTION_PAIR_BITS, PROJECTION_PAIR_LANES, PROJECTION_SHARED_BITS,
    PROJECTION_SHARED_LANES,
};
use neo_math::field::KExtensions;
use neo_math::ring::D;
use neo_math::{F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

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

fn small_projection_layout(batch_len: usize) -> FPrimeImageLayout {
    let mut config = production_kmul_d2_ring_action_shell_image_config();
    config.kmul_count = 0;
    config.ring_action_pair_count = 0;
    config.projection_batches = vec![batch_len];
    FPrimeImageLayout::new(config)
}

fn write_lane_bits(values: &mut [F], bit_start: usize, value: F) {
    let v = value.as_canonical_u64();
    for i in 0..64 {
        values[bit_start + i] = F::from_u64((v >> i) & 1);
    }
}

fn write_projection_lanes(values: &mut [F], bit_start: usize, lanes: &[F]) {
    for (lane, &value) in lanes.iter().enumerate() {
        write_lane_bits(values, bit_start + lane * 64, value);
    }
}

fn honest_projection_image(batch_len: usize) -> (FPrimeImageLayout, FPrimeImage) {
    let layout = small_projection_layout(batch_len);
    let mut image = FPrimeImage::new(layout.clone());
    let mut rng = Rng(0x706a_7368_656c_6c31);
    let beta = K::from_coeffs([F::from_u64(rng.next()), F::from_u64(rng.next())]);
    let (shared, powers) = encode_projection_shared(beta);
    write_projection_lanes(&mut image.values, layout.projection_shared_splice, &shared);

    let pairs: Vec<([F; D], [F; D])> = (0..batch_len).map(|_| (rng.rho(), rng.coeffs())).collect();
    let mut terms = Vec::with_capacity(batch_len);
    for (idx, (rho, c)) in pairs.iter().enumerate() {
        let (lanes, term) = encode_projection_pair(rho, c, &powers);
        write_projection_lanes(&mut image.values, layout.projection_pair_splices[idx], &lanes);
        terms.push(term);
    }

    let (identity, residual) = encode_projection_identity(&pairs, &powers, &terms);
    assert_eq!(residual, K::ZERO);
    write_projection_lanes(&mut image.values, layout.projection_identity_splices[0], &identity);
    (layout, image)
}

#[test]
fn projection_semantic_rows_accept_honest_image_and_reject_tamper() {
    let (layout, mut image) = honest_projection_image(3);
    let structure = build_f_prime_structure(layout);
    let baseline = structure.extend_witness_from_image(&image);
    assert!(
        structure.is_satisfied(&baseline),
        "honest projection image must satisfy semantic rows (first failing row: {:?})",
        structure.first_unsatisfied_row(&baseline)
    );

    let tamper_bit = structure.layout.projection_identity_splices[0];
    image.values[tamper_bit] = F::ONE - image.values[tamper_bit];
    let tampered = structure.extend_witness_from_image(&image);
    assert!(
        !structure.is_satisfied(&tampered),
        "bit-valid tamper to projection identity output must be rejected"
    );
}

/// Phase B: an honestly-filled projection region satisfies the
/// structure's semantic rows, and tampering any load-bearing lane —
/// an evaluation partial, the claimed output, the quotient — breaks a
/// row. (The row-count gate in `ivc_invariants` pins that rows exist;
/// this pins that they enforce the right algebra.)
#[test]
fn projection_rows_accept_honest_fill_and_reject_tampers() {
    use neo_fold_clean::frontends::f_prime::image::{FPrimeImage, FPrimeImageConfig};
    use neo_fold_clean::frontends::f_prime::structure::build_f_prime_structure;
    use neo_fold_clean::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};
    use p3_field::PrimeField64;

    let config = FPrimeImageConfig {
        limbs: 3,
        app_private_var_widths: Vec::new(),
        boundary_bits: 0,
        nifs_payload_shapes: vec![],
        kmul_count: 0,
        ring_action_pair_count: 0,
        projection_batches: vec![2], // one identity consuming two pairs
        ring_action_pair_layout: RingActionTraceLayout::new(
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
        ),
        poseidon_one_shot_preimage_lens: vec![],
        sponge_transcript_permutes: 0,
        one_shot_digest_to_state_out_bindings: vec![],
        one_shot_digest_to_state_in_bindings: vec![],
        one_shot_digest_to_public_x_out_bindings: vec![],
        poseidon_transition_enforcements: vec![],
        unified_accumulator_selector: None,
        initial_semantic_state_digest_anchor: None,
    };
    let layout = FPrimeImageLayout::new(config);
    let structure = build_f_prime_structure(layout.clone());
    let mut image = FPrimeImage::new(layout);

    // Native fill via the trace primitive, bit-decomposed into the image.
    let splice_lanes = |image: &mut FPrimeImage, bit_offset: usize, lanes: &[F]| {
        for (i, lane) in lanes.iter().enumerate() {
            let v = lane.as_canonical_u64();
            for b in 0..64 {
                image.values[bit_offset + i * 64 + b] = F::from_u64((v >> b) & 1);
            }
        }
    };
    let shared_splice = image.layout.projection_shared_splice;
    let pair_splices = image.layout.projection_pair_splices.clone();
    let identity_splices = image.layout.projection_identity_splices.clone();

    let mut rng = Rng(77);
    let beta = K::from_coeffs([F::from_u64(rng.next()), F::from_u64(rng.next())]);
    let (shared, powers) = encode_projection_shared(beta);
    splice_lanes(&mut image, shared_splice, &shared);

    let pairs: Vec<([F; D], [F; D])> = (0..2).map(|_| (rng.rho(), rng.coeffs())).collect();
    let mut terms = Vec::new();
    for (i, (rho, c)) in pairs.iter().enumerate() {
        let (lanes, term) = encode_projection_pair(rho, c, &powers);
        splice_lanes(&mut image, pair_splices[i], &lanes);
        terms.push(term);
    }
    let (identity_lanes, residual) = encode_projection_identity(&pairs, &powers, &terms);
    assert_eq!(residual, K::ZERO);
    splice_lanes(&mut image, identity_splices[0], &identity_lanes);

    let z = structure.extend_witness_from_image(&image);
    assert!(
        structure.is_satisfied(&z),
        "honest projection fill must satisfy the semantic rows (first failing row: {:?})",
        structure.first_unsatisfied_row(&z)
    );

    // Tamper sweep: flip the low bit of a load-bearing lane, re-extend,
    // and require rejection; restore and require re-acceptance.
    let expect_reject = |image: &mut FPrimeImage, bit: usize, what: &str| {
        let old = image.values[bit];
        image.values[bit] = F::ONE - old;
        let z = structure.extend_witness_from_image(image);
        assert!(!structure.is_satisfied(&z), "{what} must be rejected");
        image.values[bit] = old;
        let z = structure.extend_witness_from_image(image);
        assert!(structure.is_satisfied(&z), "restore after {what}");
    };
    // (a) an evaluation partial product inside pair 0 (ρ-eval, first term).
    let pair0 = pair_splices[0];
    expect_reject(&mut image, pair0 + (2 * D) * 64, "a forged evaluation partial");
    // (b) the identity's claimed output coefficient 3.
    let identity0 = identity_splices[0];
    expect_reject(&mut image, identity0 + 3 * 64, "a forged output coefficient");
    // (c) the quotient's coefficient 10.
    expect_reject(&mut image, identity0 + (D + 10) * 64, "a forged quotient coefficient");
    // (d) a ladder rung's product lane in the shared region.
    let shared0 = shared_splice;
    expect_reject(&mut image, shared0 + (4 + 5 * 3 + 3) * 64, "a forged ladder power");
}
