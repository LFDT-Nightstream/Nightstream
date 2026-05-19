//! Phase 1.3d-mini-4 — ring_action ring-action product/output wire parity.
//!
//! Couples the bit-backed `RingActionTraceImage` ring_action slot against
//! `enforce_ring_mul`'s actual internal wires. Uses the new production
//! variant `enforce_ring_mul_with_products` — same constraints as
//! `enforce_ring_mul`, plus the `D²` intermediate product wires
//! returned for audit/parity (mirrors the mini-3 pattern for K-mul).
//!
//! Per pair, four three-way parity assertions (native ↔ wire ↔ image):
//!
//! - ρ limbs (`D = 54` F values).
//! - c limbs.
//! - product wires `prod[i][j] = ρ[i]·c[j]` (`D² = 2916` F values).
//! - output lanes (`D` F values; equals `Rq(ρ).mul(&Rq(c)).0`).
//!
//! Out of scope:
//! - CCS structure (Phase 1.4).
//! - Lifecycle migration / Spartan / generic AppStep.
//! - "Full F' ring_action coverage accounting" (council's separate ask before
//!   Phase 1.4 — counts/order-labels every ring_mul produced by the
//!   actual NIFS verifier). This slice proves the GADGET parity
//!   surface; the coverage gate is a separate slice.

use neo_fold_clean::engine::r1cs_circuit::builder::Var;
use neo_fold_clean::engine::r1cs_circuit::ring_action::{enforce_ring_mul_with_products, RingMulProducts};
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::frontends::f_prime::image::{FPrimeImage, FPrimeImageConfig, FPrimeImageLayout};
use neo_fold_clean::paper::f_prime::poseidon_trace::assert_committed_coords_are_bits;
use neo_fold_clean::paper::f_prime::ring_action_trace::{
    encode_ring_action_trace, LowNormEncoding, RingActionTraceLayout,
};
use neo_math::ring::{Rq, D};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

// ── Helpers ──────────────────────────────────────────────────────────────

fn signed_to_field(x: i64) -> F {
    if x >= 0 {
        F::from_u64(x as u64)
    } else {
        F::ZERO - F::from_u64((-x) as u64)
    }
}

/// ρ values in `[-3, 3]` — small alphabet-sampling regime.
fn make_rho_values() -> [F; D] {
    let raw: [i64; D] = std::array::from_fn(|i| ((i as i64 * 7 + 3) % 7) - 3);
    raw.map(signed_to_field)
}

/// c values in `[-50, 50]` — commitment-data regime under small norm.
fn make_c_values() -> [F; D] {
    let raw: [i64; D] = std::array::from_fn(|i| ((i as i64 * 13 + 1).rem_euclid(101)) - 50);
    raw.map(signed_to_field)
}

fn signed_digit_ring_layout() -> RingActionTraceLayout {
    RingActionTraceLayout::new(
        LowNormEncoding::SignedDigit { bits: 5 },
        LowNormEncoding::SignedDigit { bits: 8 },
        LowNormEncoding::SignedDigit { bits: 12 },
        LowNormEncoding::SignedDigit { bits: 20 },
    )
}

/// Allocate `[Var; D]` from F values.
fn alloc_d(builder: &mut R1csBuilder, vals: &[F; D]) -> [Var; D] {
    std::array::from_fn(|i| builder.alloc(vals[i]))
}

/// Image with a single ring_action pair slot — all other regions empty.
fn ring_action_only_image_config() -> FPrimeImageConfig {
    FPrimeImageConfig {
        limbs: 3,
        boundary_bits: 0,
        nifs_payload_shapes: vec![],
        kmul_count: 0,
        ring_action_pair_count: 1,
        ring_action_pair_layout: signed_digit_ring_layout(),
        poseidon_one_shot_preimage_lens: vec![],
        sponge_transcript_permutes: 0,
        one_shot_digest_to_state_out_bindings: vec![],
        one_shot_digest_to_public_x_out_bindings: vec![],
        poseidon_transition_enforcements: vec![],
        unified_accumulator_selector: None,
    }
}

/// Native ring multiplication: compute ρ, c, all `D²` products, and the
/// output via `Rq::mul`.
fn native_ring_mul(rho_vals: &[F; D], c_vals: &[F; D]) -> ([F; D], [F; D], Vec<Vec<F>>, [F; D]) {
    let mut prods = vec![vec![F::ZERO; D]; D];
    for i in 0..D {
        for j in 0..D {
            prods[i][j] = rho_vals[i] * c_vals[j];
        }
    }
    let output = Rq(*rho_vals).mul(&Rq(*c_vals)).0;
    (*rho_vals, *c_vals, prods, output)
}

/// Wire-side view: read the witness values for ρ, c, every product, and
/// every output lane via the `RingMulProducts` struct and the output
/// wires.
fn wire_view(
    rho_wires: &[Var; D],
    c_wires: &[Var; D],
    out_wires: &[Var; D],
    products: &RingMulProducts,
    witness: &[F],
) -> ([F; D], [F; D], Vec<Vec<F>>, [F; D]) {
    let rho_vals: [F; D] = std::array::from_fn(|i| witness[rho_wires[i].col()]);
    let c_vals: [F; D] = std::array::from_fn(|j| witness[c_wires[j].col()]);
    let prods: Vec<Vec<F>> = (0..D)
        .map(|i| {
            (0..D)
                .map(|j| witness[products.prods[i][j].col()])
                .collect()
        })
        .collect();
    let out: [F; D] = std::array::from_fn(|m| witness[out_wires[m].col()]);
    (rho_vals, c_vals, prods, out)
}

/// Decode ρ, c, products, output from the ring_action region of a
/// `FPrimeImage`. Translates the pair-local
/// `RingActionTraceLayout` offsets to the image's `values` frame
/// (primitive `z[k]` for `k ≥ 1` lives at `splice_offset + k - 1`).
fn decode_ring_action_pair(image: &FPrimeImage, pair_index: usize) -> ([F; D], [F; D], Vec<Vec<F>>, [F; D]) {
    let layout = image.layout.config.ring_action_pair_layout;
    let splice = image.layout.ring_action_pair_splices[pair_index];

    let decode_lane = |enc: LowNormEncoding, lane_start_primitive: usize| -> F {
        let lane_start = splice + lane_start_primitive - 1;
        let mut acc = F::ZERO;
        for i in 0..enc.limb_count() {
            let bit = image.values[lane_start + i];
            assert!(bit == F::ZERO || bit == F::ONE);
            if bit == F::ONE {
                acc += enc.limb_coef(i);
            }
        }
        acc
    };

    let rho: [F; D] = std::array::from_fn(|i| decode_lane(layout.rho_enc, layout.rho_limb_start(i)));
    let c: [F; D] = std::array::from_fn(|j| decode_lane(layout.c_enc, layout.c_limb_start(j)));
    let prods: Vec<Vec<F>> = (0..D)
        .map(|i| {
            (0..D)
                .map(|j| decode_lane(layout.prod_enc, layout.prod_limb_start(i, j)))
                .collect()
        })
        .collect();
    let out: [F; D] = std::array::from_fn(|m| decode_lane(layout.out_enc, layout.out_lane_start(m)));
    (rho, c, prods, out)
}

fn assert_four_way_match(
    label: &str,
    rho_a: &[F; D],
    c_a: &[F; D],
    prods_a: &[Vec<F>],
    out_a: &[F; D],
    rho_b: &[F; D],
    c_b: &[F; D],
    prods_b: &[Vec<F>],
    out_b: &[F; D],
) {
    assert_eq!(rho_a, rho_b, "{label}: ρ");
    assert_eq!(c_a, c_b, "{label}: c");
    assert_eq!(prods_a.len(), prods_b.len(), "{label}: prods outer len");
    for (i, (row_a, row_b)) in prods_a.iter().zip(prods_b.iter()).enumerate() {
        assert_eq!(row_a, row_b, "{label}: prods row {i}");
    }
    assert_eq!(out_a, out_b, "{label}: output");
}

// ── Tests ────────────────────────────────────────────────────────────────

#[test]
fn phase_1_3d_ring_action_three_way_parity() {
    let rho_vals = make_rho_values();
    let c_vals = make_c_values();

    // ── 1. Native. ──────────────────────────────────────────────────────
    let (native_rho, native_c, native_prods, native_out) = native_ring_mul(&rho_vals, &c_vals);

    // ── 2. Wire. ────────────────────────────────────────────────────────
    let mut builder = R1csBuilder::new();
    let rho_wires = alloc_d(&mut builder, &rho_vals);
    let c_wires = alloc_d(&mut builder, &c_vals);
    let (out_wires, products) = enforce_ring_mul_with_products(&mut builder, &rho_wires, &c_wires);
    assert!(
        builder.is_satisfied(),
        "ring_mul gadget must satisfy honest witness (first bad row {:?})",
        builder.first_unsatisfied_row()
    );
    let (wire_rho, wire_c, wire_prods, wire_out) =
        wire_view(&rho_wires, &c_wires, &out_wires, &products, builder.witness());

    assert_four_way_match(
        "native ↔ wire",
        &native_rho,
        &native_c,
        &native_prods,
        &native_out,
        &wire_rho,
        &wire_c,
        &wire_prods,
        &wire_out,
    );

    // ── 3. Image (build + splice + decode). ─────────────────────────────
    let layout = signed_digit_ring_layout();
    let ring_trace = encode_ring_action_trace(&rho_vals, &c_vals, layout);

    let image_layout = FPrimeImageLayout::new(ring_action_only_image_config());
    let mut image = FPrimeImage::new(image_layout);
    image.splice_ring_action_pair(0, &ring_trace);
    assert_committed_coords_are_bits(&image.values);

    let (img_rho, img_c, img_prods, img_out) = decode_ring_action_pair(&image, 0);

    assert_four_way_match(
        "image ↔ native",
        &img_rho,
        &img_c,
        &img_prods,
        &img_out,
        &native_rho,
        &native_c,
        &native_prods,
        &native_out,
    );
    assert_four_way_match(
        "image ↔ wire",
        &img_rho,
        &img_c,
        &img_prods,
        &img_out,
        &wire_rho,
        &wire_c,
        &wire_prods,
        &wire_out,
    );

    // Also verify the image's recorded native output (computed inside
    // `encode_ring_action_trace`) agrees.
    assert_eq!(
        ring_trace.output_native, native_out,
        "encode_ring_action_trace.output_native ↔ Rq::mul"
    );
}

#[test]
fn phase_1_3d_ring_action_products_match_pairwise_definition() {
    // Tiny hand-checkable case to confirm the wire side really returns
    // `ρ[i] · c[j]` per product, not some other layout.
    let rho_vals: [F; D] = std::array::from_fn(|i| F::from_u64(i as u64 + 1));
    let c_vals: [F; D] = std::array::from_fn(|j| F::from_u64(2 * (j as u64) + 3));

    let mut builder = R1csBuilder::new();
    let rho_wires = alloc_d(&mut builder, &rho_vals);
    let c_wires = alloc_d(&mut builder, &c_vals);
    let (_out, products) = enforce_ring_mul_with_products(&mut builder, &rho_wires, &c_wires);
    assert!(builder.is_satisfied());

    let witness = builder.witness();
    // Sample a few cells: (0,0), (3,7), (D-1, D-1).
    for &(i, j) in &[(0usize, 0usize), (3, 7), (D - 1, D - 1)] {
        let wire_prod = witness[products.prods[i][j].col()];
        let expected = rho_vals[i] * c_vals[j];
        assert_eq!(wire_prod, expected, "ρ[{i}]·c[{j}] mismatch (wire vs spec)");
    }
}

#[test]
fn phase_1_3d_ring_action_wire_output_matches_rq_mul_per_lane() {
    let rho_vals = make_rho_values();
    let c_vals = make_c_values();

    let mut builder = R1csBuilder::new();
    let rho_wires = alloc_d(&mut builder, &rho_vals);
    let c_wires = alloc_d(&mut builder, &c_vals);
    let (out_wires, _products) = enforce_ring_mul_with_products(&mut builder, &rho_wires, &c_wires);
    assert!(builder.is_satisfied());

    let expected = Rq(rho_vals).mul(&Rq(c_vals)).0;
    for m in 0..D {
        let wire_val = builder.witness()[out_wires[m].col()];
        assert_eq!(wire_val, expected[m], "out[{m}] mismatch (wire vs Rq::mul)");
    }
}

#[test]
fn phase_1_3d_ring_action_image_decode_lossless_for_full_product_matrix() {
    // Bit-exact: every product cell, encoded then decoded, round-trips
    // through the SignedDigit{12} encoding chosen for the image's prod
    // subregion. This is the load-bearing invariant for ring_action — all D²
    // products survive the bit-decomposition.
    let rho_vals = make_rho_values();
    let c_vals = make_c_values();
    let layout = signed_digit_ring_layout();
    let ring_trace = encode_ring_action_trace(&rho_vals, &c_vals, layout);
    let image_layout = FPrimeImageLayout::new(ring_action_only_image_config());
    let mut image = FPrimeImage::new(image_layout);
    image.splice_ring_action_pair(0, &ring_trace);

    let (_, _, img_prods, _) = decode_ring_action_pair(&image, 0);
    for i in 0..D {
        for j in 0..D {
            let expected = rho_vals[i] * c_vals[j];
            assert_eq!(img_prods[i][j], expected, "image prod[{i}][{j}] decode");
        }
    }
}
