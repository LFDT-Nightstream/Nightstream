//! Smoke tests for the F' low-norm source-image seed + R1CS view.

use neo_ajtai::{has_global_pp_for_dims, set_global_pp_seeded, AjtaiSModule};
use neo_ccs::{CcsStructure, Mat, SparsePoly};
use neo_fold_clean::config;
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::paper::f_prime::source_image::{FPrimeSourceImage, KWordImage};
use neo_fold_clean::paper::f_prime::source_image_circuit::{
    enforce_goldilocks_word_canonical, enforce_k_word_add, enforce_k_word_affine2, enforce_k_word_eq,
    enforce_k_word_mul, enforce_k_word_sub, enforce_word64_mul, SourceImageWires,
};
use neo_fold_clean::paper::relations::RelationError;
use neo_fold_clean::{CcsInstance, Params, Structure};
use neo_math::{KExtensions, D, F, K};
use p3_field::PrimeCharacteristicRing;

#[test]
fn f_prime_source_image_encodes_digest_as_binary_bits() {
    let x_out = [
        F::from_u64(0),
        F::from_u64(1),
        F::from_u64(0xffff_ffff),
        F::from_u64(0x1234_5678_9abc_def0),
    ];

    let mut image = FPrimeSourceImage::new();
    image.push_bit(true);
    image.push_enc_inst(x_out);

    assert!(image.is_binary(), "every appended value must be a bit");
    assert_eq!(image.len(), 1 + 256, "1 leading bit + 256 enc_inst bits");
    assert_eq!(image.values()[0], F::ONE);
    // Lane 0 is 0 → first 64 bits after the leading slot are all 0.
    assert!(image.values()[1..1 + 64].iter().all(|v| *v == F::ZERO));
    // Lane 1 is 1 → first bit of that lane is 1, rest 0.
    assert_eq!(image.values()[1 + 64], F::ONE);
    assert!(image.values()[1 + 65..1 + 128]
        .iter()
        .all(|v| *v == F::ZERO));
}

#[test]
fn f_prime_source_image_push_u64_le_matches_canonical_bits() {
    let v: u64 = 0xA5A5_A5A5_A5A5_A5A5;
    let mut image = FPrimeSourceImage::new();
    image.push_u64_le(v);
    assert!(image.is_binary());
    assert_eq!(image.len(), 64);
    for i in 0..64 {
        let expected = if (v >> i) & 1 == 1 { F::ONE } else { F::ZERO };
        assert_eq!(image.values()[i], expected, "bit {i} mismatch");
    }
}

// ── R1CS view of the source image ─────────────────────────────────────────

#[test]
fn source_image_wires_decode_word64_as_linear_combination() {
    let value = 0x1234_5678_9abc_def0;

    let mut image = FPrimeSourceImage::new();
    let word = image.push_u64_le(value);

    let mut builder = R1csBuilder::new();
    let wires = SourceImageWires::alloc(&mut builder, &image);
    enforce_goldilocks_word_canonical(&mut builder, &wires, word);

    let decoded = wires.word64_lc(word);
    assert_eq!(builder.eval(&decoded), F::from_u64(value));
    assert!(builder.is_satisfied(), "honest source image must satisfy");
}

#[test]
fn source_image_wires_reject_non_binary_bit_tamper() {
    let mut image = FPrimeSourceImage::new();
    let word = image.push_u64_le(5);

    let mut builder = R1csBuilder::new();
    let wires = SourceImageWires::alloc(&mut builder, &image);
    enforce_goldilocks_word_canonical(&mut builder, &wires, word);

    let first_bit_col = wires.bits()[word.bits().start()].col();
    builder.tamper_witness(first_bit_col, F::from_u64(2));

    assert!(!builder.is_satisfied(), "source-image bits must be binary");
}

#[test]
fn source_image_wires_reject_noncanonical_goldilocks_word() {
    // p = 0xffff_ffff_0000_0001 — the smallest 64-bit value that's NOT a
    // canonical Goldilocks element (hi == 0xffff_ffff AND lo == 1).
    let noncanonical = 0xffff_ffff_0000_0001;

    let mut image = FPrimeSourceImage::new();
    let word = image.push_u64_le(noncanonical);

    let mut builder = R1csBuilder::new();
    let wires = SourceImageWires::alloc(&mut builder, &image);
    enforce_goldilocks_word_canonical(&mut builder, &wires, word);

    assert!(
        !builder.is_satisfied(),
        "Goldilocks word canonicality must reject encodings >= p"
    );
}

#[test]
fn source_image_digest_lanes_decode_to_expected_fields() {
    let lanes = [
        F::from_u64(0),
        F::from_u64(1),
        F::from_u64(0xffff_ffff),
        F::from_u64(0x1234_5678_9abc_def0),
    ];

    let mut image = FPrimeSourceImage::new();
    let digest = image.push_digest_lanes(lanes);

    let mut builder = R1csBuilder::new();
    let wires = SourceImageWires::alloc(&mut builder, &image);

    for lane in digest.lanes() {
        enforce_goldilocks_word_canonical(&mut builder, &wires, lane);
    }

    let decoded = wires.digest_lcs(digest);
    for i in 0..4 {
        assert_eq!(builder.eval(&decoded[i]), lanes[i], "lane {i} mismatch");
    }
    assert!(builder.is_satisfied(), "honest digest source image must satisfy");
}

// ── `enc(F')` arithmetic primitive: Word64 multiplication ─────────────────

#[test]
fn source_image_word64_mul_accepts_honest_product() {
    let a = F::from_u64(123_456_789);
    let b = F::from_u64(987_654_321);
    let out = a * b;

    let mut image = FPrimeSourceImage::new();
    let a_word = image.push_goldilocks(a);
    let b_word = image.push_goldilocks(b);
    let out_word = image.push_goldilocks(out);

    let mut builder = R1csBuilder::new();
    let wires = SourceImageWires::alloc(&mut builder, &image);
    enforce_word64_mul(&mut builder, &wires, a_word, b_word, out_word);

    assert!(builder.is_satisfied());
}

#[test]
fn source_image_word64_mul_accepts_modular_wraparound_product() {
    // (-1) * (-1) = 1 in Goldilocks. `out` is committed as the canonical
    // bits of `1`, not the bits of `(-1)^2` over the integers — the
    // canonicality check on `out` pins it to the unique representative.
    let a = -F::ONE;
    let b = -F::ONE;
    let out = F::ONE;

    let mut image = FPrimeSourceImage::new();
    let a_word = image.push_goldilocks(a);
    let b_word = image.push_goldilocks(b);
    let out_word = image.push_goldilocks(out);

    let mut builder = R1csBuilder::new();
    let wires = SourceImageWires::alloc(&mut builder, &image);
    enforce_word64_mul(&mut builder, &wires, a_word, b_word, out_word);

    assert!(builder.is_satisfied(), "modular wraparound (-1)*(-1) = 1 must satisfy");
}

#[test]
fn source_image_word64_mul_rejects_tampered_output_bit() {
    let a = F::from_u64(17);
    let b = F::from_u64(19);
    let out = a * b;

    let mut image = FPrimeSourceImage::new();
    let a_word = image.push_goldilocks(a);
    let b_word = image.push_goldilocks(b);
    let out_word = image.push_goldilocks(out);

    let idx = out_word.bits().start();
    let old = image.values()[idx];
    image.set_bit(idx, old == F::ZERO);

    let mut builder = R1csBuilder::new();
    let wires = SourceImageWires::alloc(&mut builder, &image);
    enforce_word64_mul(&mut builder, &wires, a_word, b_word, out_word);

    assert!(
        !builder.is_satisfied(),
        "tampered output bit must break Word64 multiplication"
    );
}

#[test]
fn source_image_word64_mul_rejects_noncanonical_input_word() {
    // `a` is `p = 0xFFFF_FFFF_0000_0001` as a raw 64-bit pattern — not a
    // canonical Goldilocks element. The canonicality check inside
    // `enforce_word64_mul` must reject it.
    let mut image = FPrimeSourceImage::new();
    let a_word = image.push_u64_le(0xFFFF_FFFF_0000_0001);
    let b_word = image.push_goldilocks(F::from_u64(3));
    let out_word = image.push_goldilocks(F::ZERO);

    let mut builder = R1csBuilder::new();
    let wires = SourceImageWires::alloc(&mut builder, &image);
    enforce_word64_mul(&mut builder, &wires, a_word, b_word, out_word);

    assert!(!builder.is_satisfied(), "noncanonical input word must be rejected");
}

#[test]
fn source_image_word64_mul_allocates_no_raw_output_var() {
    // Verifies the discipline rule: the multiplication output is *not* a
    // freshly-allocated field witness var — it lives only as `out` bits
    // in the source image. After source-image alloc the column count
    // grows by exactly `image.len()`, and the mul itself must add nothing
    // beyond auxiliary canonicality bits (which are not "the output Var").
    let a = F::from_u64(7);
    let b = F::from_u64(11);
    let out = a * b;

    let mut image = FPrimeSourceImage::new();
    let a_word = image.push_goldilocks(a);
    let b_word = image.push_goldilocks(b);
    let out_word = image.push_goldilocks(out);

    let mut builder = R1csBuilder::new();
    let before = builder.cols();
    let wires = SourceImageWires::alloc(&mut builder, &image);
    let after_source_alloc = builder.cols();
    enforce_word64_mul(&mut builder, &wires, a_word, b_word, out_word);

    assert_eq!(
        after_source_alloc - before,
        image.len(),
        "source image should allocate exactly one wire per source bit"
    );
    // `enforce_goldilocks_word_canonical` does allocate aux bits (the
    // hi-is-max indicator + its inverse witness per word, 3 per word, 3
    // words). That's bookkeeping for canonicality, not a raw `out` Var.
    let mul_aux_cols = builder.cols() - after_source_alloc;
    assert!(
        mul_aux_cols <= 3 * 3,
        "Word64 mul should only add canonicality aux cols (≤ 3 per word × 3 words = 9); got {mul_aux_cols}"
    );
    assert!(builder.is_satisfied());
}

// ── `enc(F')` arithmetic primitive #2: K-extension multiplication ─────────

#[test]
fn source_image_k_word_mul_accepts_honest_product() {
    let a = K::from_coeffs([F::from_u64(3), F::from_u64(5)]);
    let b = K::from_coeffs([F::from_u64(7), F::from_u64(11)]);
    let out = a * b;

    let mut image = FPrimeSourceImage::new();
    let a_word = image.push_k(a);
    let b_word = image.push_k(b);
    let out_word = image.push_k(out);
    let mul = image.push_k_mul_witness(a, b);

    let mut builder = R1csBuilder::new();
    let wires = SourceImageWires::alloc(&mut builder, &image);
    enforce_k_word_mul(&mut builder, &wires, a_word, b_word, out_word, mul);

    assert!(builder.is_satisfied(), "honest K product must satisfy");
}

#[test]
fn source_image_k_word_mul_rejects_tampered_output_limb() {
    let a = K::from_coeffs([F::from_u64(7), F::from_u64(11)]);
    let b = K::from_coeffs([F::from_u64(13), F::from_u64(17)]);
    let out = a * b;

    let mut image = FPrimeSourceImage::new();
    let a_word = image.push_k(a);
    let b_word = image.push_k(b);
    let out_word = image.push_k(out);
    let mul = image.push_k_mul_witness(a, b);

    // Flip the lowest bit of out.c0. The closing equality
    // `out.c0 = p + W·q` over bit-decoded LCs must reject it.
    let idx = out_word.c0().bits().start();
    let old = image.values()[idx];
    image.set_bit(idx, old == F::ZERO);

    let mut builder = R1csBuilder::new();
    let wires = SourceImageWires::alloc(&mut builder, &image);
    enforce_k_word_mul(&mut builder, &wires, a_word, b_word, out_word, mul);

    assert!(
        !builder.is_satisfied(),
        "tampered K output limb must break the multiplication"
    );
}

#[test]
fn source_image_k_word_mul_rejects_tampered_intermediate_p() {
    let a = K::from_coeffs([F::from_u64(13), F::from_u64(17)]);
    let b = K::from_coeffs([F::from_u64(19), F::from_u64(23)]);
    let out = a * b;

    let mut image = FPrimeSourceImage::new();
    let a_word = image.push_k(a);
    let b_word = image.push_k(b);
    let out_word = image.push_k(out);
    let mul = image.push_k_mul_witness(a, b);

    // Flip the lowest bit of the Karatsuba intermediate `p = a0·b0`.
    // The product constraint `a0 · b0 = p` (bit-decoded LCs) must
    // reject it.
    let idx = mul.p().bits().start();
    let old = image.values()[idx];
    image.set_bit(idx, old == F::ZERO);

    let mut builder = R1csBuilder::new();
    let wires = SourceImageWires::alloc(&mut builder, &image);
    enforce_k_word_mul(&mut builder, &wires, a_word, b_word, out_word, mul);

    assert!(
        !builder.is_satisfied(),
        "tampered Karatsuba intermediate p must break the multiplication"
    );
}

#[test]
fn source_image_k_word_mul_rejects_noncanonical_limb() {
    // a.c0 is the raw 64-bit pattern p = 0xFFFF_FFFF_0000_0001 — exactly
    // the smallest noncanonical Goldilocks encoding. The canonicality
    // check inside `enforce_k_word_mul` must reject it before the
    // product law is even evaluated.
    let a_native = K::from_coeffs([F::ZERO, F::ZERO]); // for the Karatsuba witness
    let b_native = K::from_coeffs([F::from_u64(2), F::from_u64(3)]);

    let mut image = FPrimeSourceImage::new();
    let a_c0 = image.push_u64_le(0xFFFF_FFFF_0000_0001);
    let a_c1 = image.push_goldilocks(F::ZERO);
    let a_word = KWordImage::from_limbs(a_c0, a_c1);
    let b_word = image.push_k(b_native);
    let out_word = image.push_k(K::ZERO);
    let mul = image.push_k_mul_witness(a_native, b_native);

    let mut builder = R1csBuilder::new();
    let wires = SourceImageWires::alloc(&mut builder, &image);
    enforce_k_word_mul(&mut builder, &wires, a_word, b_word, out_word, mul);

    assert!(
        !builder.is_satisfied(),
        "noncanonical K limb must be rejected by canonicality"
    );
}

#[test]
fn source_image_k_word_mul_allocates_no_raw_product_vars() {
    // Strict discipline test: every coordinate of the K-mul lives in
    // the source image (a, b, out, plus Karatsuba p, q, r). The
    // gadget allocates *only* the canonicality helpers
    // (`hi_is_max` + its `inv` per word) and nothing else — no raw
    // product Vars, no raw output Vars.
    //
    // Canonicality aux count: 2 helpers × 9 source-image Word64Images
    // (a.c0, a.c1, b.c0, b.c1, out.c0, out.c1, p, q, r) = 18.
    let a = K::from_coeffs([F::from_u64(2), F::from_u64(3)]);
    let b = K::from_coeffs([F::from_u64(5), F::from_u64(7)]);
    let out = a * b;

    let mut image = FPrimeSourceImage::new();
    let a_word = image.push_k(a);
    let b_word = image.push_k(b);
    let out_word = image.push_k(out);
    let mul = image.push_k_mul_witness(a, b);

    let mut builder = R1csBuilder::new();
    let before = builder.cols();
    let wires = SourceImageWires::alloc(&mut builder, &image);
    let after_source_alloc = builder.cols();
    enforce_k_word_mul(&mut builder, &wires, a_word, b_word, out_word, mul);
    let after_mul = builder.cols();

    assert_eq!(
        after_source_alloc - before,
        image.len(),
        "source image should allocate exactly one wire per source bit"
    );
    assert_eq!(
        after_mul - after_source_alloc,
        18,
        "K-mul must add only canonicality helpers (2 × 9 words); any raw \
         product Var would push this past 18"
    );
    assert!(builder.is_satisfied(), "honest K product must satisfy");
}

// ── `enc(F')` audit: source image vs the real SuperNeo CCS boundary ───────

/// Minimal CCS fixture for proving that source-image values clear the
/// `CcsInstance::from_low_norm_assignment` boundary. The structure is
/// `Mat::identity(m)` with an empty polynomial, so the constructor only
/// runs its length and `‖z‖_∞ < b` validators — exactly what we want
/// to audit against.
struct AuditFixture {
    params: Params,
    log: AjtaiSModule,
    structure: Structure,
}

fn small_ccs_audit_fixture(target_m: usize) -> AuditFixture {
    let m = target_m.max(1);
    let structure = CcsStructure::new(vec![Mat::identity(m)], SparsePoly::new(1, vec![])).expect("audit CCS structure");
    let params =
        config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree()).expect("audit params");

    let cols = structure.m.div_ceil(D);
    if !has_global_pp_for_dims(D, cols) {
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&0xA00D_1700_5006_F0DEu64.to_le_bytes());
        match set_global_pp_seeded(D, params.kappa() as usize, cols, seed) {
            Ok(()) => {}
            Err(_err) if has_global_pp_for_dims(D, cols) => {}
            Err(err) => panic!("Ajtai PP install for audit: {err}"),
        }
    }
    let log = AjtaiSModule::from_global_for_dims(D, cols).expect("audit Ajtai log");

    AuditFixture { params, log, structure }
}

#[test]
fn source_image_values_can_be_committed_as_low_norm_ccs_assignment() {
    // Build a source image with realistic content: three canonical
    // Goldilocks words. Every coordinate is a bit, so the full
    // `z = [1 || image || zero-pad]` must clear `‖z‖_∞ < b` at b=2.
    let mut image = FPrimeSourceImage::new();
    let a = F::from_u64(7);
    let b = F::from_u64(11);
    let _ = image.push_goldilocks(a);
    let _ = image.push_goldilocks(b);
    let _ = image.push_goldilocks(a * b);
    assert!(image.is_binary(), "source image must be bit-valued");

    let fixture = small_ccs_audit_fixture(1 + image.len());

    let mut z = Vec::with_capacity(fixture.structure.m);
    z.push(F::ONE); // public constant-one slot
    z.extend_from_slice(image.values());
    z.resize(fixture.structure.m, F::ZERO);

    let m_in = D;
    let instance = CcsInstance::from_low_norm_assignment(&fixture.params, &fixture.log, &fixture.structure, &z, m_in);

    assert!(
        instance.is_ok(),
        "source-image assignment must pass the SuperNeo low-norm constructor; got {:?}",
        instance.err()
    );
}

#[test]
fn source_image_assignment_rejects_non_binary_coordinate_at_ccs_boundary() {
    // Same shape as the positive test, but a single coordinate is
    // overwritten to `2` — outside the centered b=2 window {-1, 0, 1}.
    // The real CCS constructor must refuse it; this is what proves the
    // audit is biting on the SuperNeo side, not just the bitness gadgets.
    let mut image = FPrimeSourceImage::new();
    let _ = image.push_goldilocks(F::from_u64(5));
    image.set_raw(0, F::from_u64(2));

    let fixture = small_ccs_audit_fixture(1 + image.len());

    let mut z = Vec::with_capacity(fixture.structure.m);
    z.push(F::ONE);
    z.extend_from_slice(image.values());
    z.resize(fixture.structure.m, F::ZERO);

    let m_in = D;
    let result = CcsInstance::from_low_norm_assignment(&fixture.params, &fixture.log, &fixture.structure, &z, m_in);

    assert!(
        result.is_err(),
        "non-binary source-image coordinate must fail SuperNeo low-norm CCS construction"
    );
}

#[test]
fn source_image_assignment_rejects_partial_public_ring_at_ccs_boundary() {
    let fixture = small_ccs_audit_fixture(D);
    let z = vec![F::ZERO; fixture.structure.m];

    assert!(matches!(
        CcsInstance::from_low_norm_assignment(&fixture.params, &fixture.log, &fixture.structure, &z, 1),
        Err(RelationError::PublicInputNotWholeRing { m_in: 1, d: D })
    ));
}

// ── `enc(F')` linear glue: K addition / subtraction / affine ──────────────

#[test]
fn source_image_k_word_eq_accepts_matching_values() {
    let v = K::from_coeffs([F::from_u64(42), F::from_u64(99)]);

    let mut image = FPrimeSourceImage::new();
    let lhs = image.push_k(v);
    let rhs = image.push_k(v);

    let mut builder = R1csBuilder::new();
    let wires = SourceImageWires::alloc(&mut builder, &image);
    enforce_k_word_eq(&mut builder, &wires, lhs, rhs);

    assert!(builder.is_satisfied(), "equal K values must satisfy K-eq");
}

#[test]
fn source_image_k_word_add_accepts_honest_sum() {
    let a = K::from_coeffs([F::from_u64(3), F::from_u64(5)]);
    let b = K::from_coeffs([F::from_u64(7), F::from_u64(11)]);
    let out = a + b;

    let mut image = FPrimeSourceImage::new();
    let a_word = image.push_k(a);
    let b_word = image.push_k(b);
    let out_word = image.push_k(out);

    let mut builder = R1csBuilder::new();
    let wires = SourceImageWires::alloc(&mut builder, &image);
    enforce_k_word_add(&mut builder, &wires, out_word, a_word, b_word);

    assert!(builder.is_satisfied(), "honest K sum must satisfy");
}

#[test]
fn source_image_k_word_add_rejects_tampered_sum() {
    let a = K::from_coeffs([F::from_u64(3), F::from_u64(5)]);
    let b = K::from_coeffs([F::from_u64(7), F::from_u64(11)]);
    let out = a + b;

    let mut image = FPrimeSourceImage::new();
    let a_word = image.push_k(a);
    let b_word = image.push_k(b);
    let out_word = image.push_k(out);

    let idx = out_word.c0().bits().start();
    let old = image.values()[idx];
    image.set_bit(idx, old == F::ZERO);

    let mut builder = R1csBuilder::new();
    let wires = SourceImageWires::alloc(&mut builder, &image);
    enforce_k_word_add(&mut builder, &wires, out_word, a_word, b_word);

    assert!(!builder.is_satisfied(), "tampered K sum must violate the limb-equality");
}

#[test]
fn source_image_k_word_sub_accepts_honest_difference() {
    let a = K::from_coeffs([F::from_u64(13), F::from_u64(17)]);
    let b = K::from_coeffs([F::from_u64(19), F::from_u64(23)]);
    let out = a - b;

    let mut image = FPrimeSourceImage::new();
    let a_word = image.push_k(a);
    let b_word = image.push_k(b);
    let out_word = image.push_k(out);

    let mut builder = R1csBuilder::new();
    let wires = SourceImageWires::alloc(&mut builder, &image);
    enforce_k_word_sub(&mut builder, &wires, out_word, a_word, b_word);

    assert!(builder.is_satisfied(), "honest K difference must satisfy");
}

#[test]
fn source_image_k_word_affine2_accepts_constants() {
    let a = K::from_coeffs([F::from_u64(2), F::from_u64(3)]);
    let b = K::from_coeffs([F::from_u64(5), F::from_u64(7)]);
    let ac = K::from_coeffs([F::from_u64(11), F::from_u64(13)]);
    let bc = K::from_coeffs([F::from_u64(17), F::from_u64(19)]);
    let constant = K::from_coeffs([F::from_u64(23), F::from_u64(29)]);
    let out = ac * a + bc * b + constant;

    let mut image = FPrimeSourceImage::new();
    let a_word = image.push_k(a);
    let b_word = image.push_k(b);
    let out_word = image.push_k(out);

    let mut builder = R1csBuilder::new();
    let wires = SourceImageWires::alloc(&mut builder, &image);
    enforce_k_word_affine2(&mut builder, &wires, out_word, a_word, ac, b_word, bc, constant);

    assert!(
        builder.is_satisfied(),
        "affine K combination must match native K arithmetic"
    );
}

#[test]
fn source_image_k_word_add_allocates_only_canonicality_helpers() {
    // Discipline: K-add is a pair of pure linear equalities. The only
    // columns the gadget should add are canonicality helpers for the
    // 6 limbs (a.c0, a.c1, b.c0, b.c1, out.c0, out.c1) = 2 × 6 = 12.
    // No raw output Var, no product witness, no aux scratch.
    let a = K::from_coeffs([F::from_u64(1), F::from_u64(2)]);
    let b = K::from_coeffs([F::from_u64(3), F::from_u64(4)]);
    let out = a + b;

    let mut image = FPrimeSourceImage::new();
    let a_word = image.push_k(a);
    let b_word = image.push_k(b);
    let out_word = image.push_k(out);

    let mut builder = R1csBuilder::new();
    let before = builder.cols();
    let wires = SourceImageWires::alloc(&mut builder, &image);
    let after_source_alloc = builder.cols();
    enforce_k_word_add(&mut builder, &wires, out_word, a_word, b_word);
    let after_add = builder.cols();

    assert_eq!(after_source_alloc - before, image.len());
    assert_eq!(
        after_add - after_source_alloc,
        12,
        "K-add must add only canonicality helpers (2 × 6 words); any raw \
         linear output Var or scratch would push this past 12"
    );
    assert!(builder.is_satisfied(), "honest K sum must satisfy");
}
