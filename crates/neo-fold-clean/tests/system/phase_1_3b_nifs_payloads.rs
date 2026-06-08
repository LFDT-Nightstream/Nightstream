//! Phase 1.3b — fill nifs_payloads NIFS payloads.
//!
//! Adds typed `NifsCcsClaimView` / `NifsCeClaimView` round-trip coverage on
//! the Phase 1.2 image. Drives encoding/decoding with a real toy NIFS
//! fixture (one fresh CcsClaim + the resulting parent_authority CeClaim
//! produced by `nifs::prove`). Tests check:
//!
//! - Round-trip parity: fill → decode returns the original view.
//! - Low-norm bit invariant after both claims are filled.
//! - Disjointness: nifs_payloads fill does not perturb boundary..app_private or kmul..poseidon.
//! - Wrong-shape inputs panic (overflow, mismatched lengths).
//!
//! Out of scope (council instructions):
//! - kmul K-mul fill (Phase 1.3c).
//! - Per-step parity vs F' R1CS emitter (Phase 1.3d).
//! - CCS structure / lifecycle / Spartan / generic AppStep.
//! - Any change that turns an `ivc_invariants` test green.

#[path = "../support/mod.rs"]
mod support;

use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::frontends::f_prime::image::{
    FPrimeImage, FPrimeImageConfig, FPrimeImageLayout, NifsCcsClaimShape, NifsCcsClaimView, NifsCeClaimShape,
    NifsCeClaimView, NifsPayloadShape,
};
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::digest::digest32_as_fields;
use neo_fold_clean::paper::f_prime::poseidon_trace::assert_committed_coords_are_bits;
use neo_fold_clean::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};
use neo_fold_clean::paper::nifs;
use neo_fold_clean::paper::relations::superneo_public_x_cols;
use neo_fold_clean::paper::relations::CcsClaim;
use neo_fold_clean::CeClaim;
use neo_math::F;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};

// ── Fixture: one fresh CcsClaim + one parent_authority CeClaim ───────────

struct NifsFixture {
    fresh: CcsClaim,
    parent: CeClaim,
}

fn build_nifs_fixture() -> NifsFixture {
    let prep = support::toy_preprocessing();
    let fresh_inst = vec![support::toy_instance(&prep, 17), support::toy_instance(&prep, 23)];
    let fresh = fresh_inst[0].claim.clone();
    let mut prover_tr = Transcript::session();
    let (running, _proof) = nifs::prove(
        &mut prover_tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        fresh_inst,
        &RunningInstance::default(),
    )
    .expect("NIFS.P");
    let parent = running
        .parent_authority
        .expect("parent_authority present")
        .clone();
    NifsFixture { fresh, parent }
}

// ── Converters from production types to nifs_payloads views ─────────────────────────

fn ccs_claim_to_view(claim: &CcsClaim) -> NifsCcsClaimView {
    NifsCcsClaimView {
        d: claim.c.d as u64,
        kappa: claim.c.kappa as u64,
        c_data: claim.c.data.clone(),
        x: claim.x.clone(),
        m_in: claim.m_in as u64,
    }
}

fn k_to_pair(k: &neo_math::K) -> [F; 2] {
    let limbs = k.as_basis_coefficients_slice();
    [limbs[0], limbs[1]]
}

fn ce_claim_to_view(claim: &CeClaim) -> NifsCeClaimView {
    let x_rows = claim.X.rows();
    let x_cols = claim.X.cols();
    let x_active = superneo_public_x_cols(claim.m_in);
    let x_active_flat: Vec<F> = (0..x_rows)
        .flat_map(|r| (0..x_active).map(move |c| claim.X[(r, c)]))
        .collect();
    let r: Vec<[F; 2]> = claim.r.iter().map(k_to_pair).collect();
    let y_ring: Vec<Vec<[F; 2]>> = claim
        .y_ring
        .iter()
        .map(|row| row.iter().map(k_to_pair).collect())
        .collect();
    let y_zcol: Vec<[F; 2]> = claim.y_zcol.iter().map(k_to_pair).collect();
    let s_col: Vec<[F; 2]> = claim.s_col.iter().map(k_to_pair).collect();
    NifsCeClaimView {
        d: claim.c.d as u64,
        kappa: claim.c.kappa as u64,
        c_data: claim.c.data.clone(),
        x_rows: x_rows as u64,
        x_cols: x_cols as u64,
        x_active_cols: x_active as u64,
        x_active_flat,
        r,
        y_ring,
        y_zcol,
        s_col,
        m_in: claim.m_in as u64,
        fold_digest_fields: digest32_as_fields(claim.fold_digest),
    }
}

fn ccs_view_shape(view: &NifsCcsClaimView) -> NifsCcsClaimShape {
    NifsCcsClaimShape {
        c_data_entries: view.c_data.len(),
        x_entries: view.x.len(),
    }
}

fn ce_view_shape(view: &NifsCeClaimView) -> NifsCeClaimShape {
    NifsCeClaimShape {
        c_data_entries: view.c_data.len(),
        x_rows: view.x_rows as usize,
        x_active_cols: view.x_active_cols as usize,
        r_len: view.r.len(),
        y_ring_inner_lens: view.y_ring.iter().map(|row| row.len()).collect(),
        y_zcol_len: view.y_zcol.len(),
        s_col_len: view.s_col.len(),
    }
}

// ── Config sized to the fixture's actual claim sizes ─────────────────────

fn skeleton_config_for(fresh_shape: &NifsCcsClaimShape, ce_shape: &NifsCeClaimShape) -> FPrimeImageConfig {
    FPrimeImageConfig {
        limbs: 3,
        app_private_var_widths: Vec::new(),
        boundary_bits: 704,
        nifs_payload_shapes: vec![
            NifsPayloadShape::CcsClaim(*fresh_shape),
            NifsPayloadShape::CeClaim(ce_shape.clone()),
        ],
        kmul_count: 8,
        ring_action_pair_count: 2,
        ring_action_pair_layout: RingActionTraceLayout::new(
            LowNormEncoding::SignedDigit { bits: 5 },
            LowNormEncoding::SignedDigit { bits: 8 },
            LowNormEncoding::SignedDigit { bits: 12 },
            LowNormEncoding::SignedDigit { bits: 20 },
        ),
        poseidon_one_shot_preimage_lens: vec![13, 40],
        sponge_transcript_permutes: 16,
        one_shot_digest_to_state_out_bindings: vec![],
        one_shot_digest_to_state_in_bindings: vec![],
        one_shot_digest_to_public_x_out_bindings: vec![],
        poseidon_transition_enforcements: vec![],
        unified_accumulator_selector: None,
        initial_semantic_state_digest_anchor: None,
    }
}

fn fresh_image(fresh_shape: &NifsCcsClaimShape, ce_shape: &NifsCeClaimShape) -> FPrimeImage {
    FPrimeImage::new(FPrimeImageLayout::new(skeleton_config_for(fresh_shape, ce_shape)))
}

// ── Round-trip tests ─────────────────────────────────────────────────────

#[test]
fn phase_1_3b_fresh_ccs_claim_round_trips() {
    let fixture = build_nifs_fixture();
    let view = ccs_claim_to_view(&fixture.fresh);
    let shape = ccs_view_shape(&view);

    // For this test we don't care about the CE region; pick a placeholder
    // CE shape with zero entries.
    let ce_placeholder = NifsCeClaimShape {
        c_data_entries: 0,
        x_rows: 0,
        x_active_cols: 0,
        r_len: 0,
        y_ring_inner_lens: vec![],
        y_zcol_len: 0,
        s_col_len: 0,
    };
    let mut image = fresh_image(&shape, &ce_placeholder);

    let next = image.fill_nifs_ccs_claim_at(0, &view);
    assert_eq!(next, shape.bits());

    let decoded = image.decode_nifs_ccs_claim_at(0, &shape);
    assert_eq!(decoded, view);
    assert_committed_coords_are_bits(&image.values);

    eprintln!(
        "phase_1_3b CcsClaim: {} bits ({} c_data, {} x)",
        shape.bits(),
        shape.c_data_entries,
        shape.x_entries
    );
}

#[test]
fn phase_1_3b_parent_authority_ce_claim_round_trips() {
    let fixture = build_nifs_fixture();
    let view = ce_claim_to_view(&fixture.parent);
    let shape = ce_view_shape(&view);

    let fresh_placeholder = NifsCcsClaimShape {
        c_data_entries: 0,
        x_entries: 0,
    };
    let mut image = fresh_image(&fresh_placeholder, &shape);

    // Skip past the (zero-bit) fresh placeholder and fill CE at offset 0
    // of nifs_payloads.
    let next = image.fill_nifs_ce_claim_at(0, &view);
    assert_eq!(next, shape.bits());

    let decoded = image.decode_nifs_ce_claim_at(0, &shape);
    assert_eq!(decoded, view);
    assert_committed_coords_are_bits(&image.values);

    eprintln!(
        "phase_1_3b CeClaim: {} bits ({} c_data, x {}×{}/{}, r {}, y_ring {} rows, y_zcol {}, s_col {})",
        shape.bits(),
        shape.c_data_entries,
        shape.x_rows,
        view.x_cols,
        shape.x_active_cols,
        shape.r_len,
        shape.y_ring_inner_lens.len(),
        shape.y_zcol_len,
        shape.s_col_len
    );
}

#[test]
fn phase_1_3b_two_claims_back_to_back_round_trip() {
    let fixture = build_nifs_fixture();
    let fresh_view = ccs_claim_to_view(&fixture.fresh);
    let ce_view = ce_claim_to_view(&fixture.parent);
    let fresh_shape = ccs_view_shape(&fresh_view);
    let ce_shape = ce_view_shape(&ce_view);

    let mut image = fresh_image(&fresh_shape, &ce_shape);

    let after_fresh = image.fill_nifs_ccs_claim_at(0, &fresh_view);
    assert_eq!(after_fresh, fresh_shape.bits());
    let after_ce = image.fill_nifs_ce_claim_at(after_fresh, &ce_view);
    assert_eq!(after_ce, fresh_shape.bits() + ce_shape.bits());
    assert_eq!(after_ce, image.layout.nifs_payloads.bits);

    let decoded_fresh = image.decode_nifs_ccs_claim_at(0, &fresh_shape);
    let decoded_ce = image.decode_nifs_ce_claim_at(after_fresh, &ce_shape);

    assert_eq!(decoded_fresh, fresh_view);
    assert_eq!(decoded_ce, ce_view);
    assert_committed_coords_are_bits(&image.values);
}

// ── Disjointness: nifs_payloads fill leaves other regions zero ──────────────────────

#[test]
fn phase_1_3b_nifs_fill_leaves_other_regions_zero() {
    let fixture = build_nifs_fixture();
    let fresh_view = ccs_claim_to_view(&fixture.fresh);
    let ce_view = ce_claim_to_view(&fixture.parent);
    let fresh_shape = ccs_view_shape(&fresh_view);
    let ce_shape = ce_view_shape(&ce_view);

    let mut image = fresh_image(&fresh_shape, &ce_shape);
    let _ = image.fill_nifs_ccs_claim_at(0, &fresh_view);
    let _ = image.fill_nifs_ce_claim_at(fresh_shape.bits(), &ce_view);

    for region in [
        image.layout.boundary,
        image.layout.state_in,
        image.layout.state_out,
        image.layout.chunk_digest,
        image.layout.app_private,
        image.layout.kmul,
        image.layout.ring_action,
        image.layout.poseidon,
    ] {
        for v in &image.values[region.offset..region.end()] {
            assert_eq!(
                *v,
                F::ZERO,
                "nifs_payloads fills must not perturb non-nifs_payloads region {region:?}"
            );
        }
    }
}

// ── Wrong-shape / overflow panics ────────────────────────────────────────

#[test]
#[should_panic(expected = "nifs_payloads CcsClaim payload at offset")]
fn phase_1_3b_nifs_overflow_panics() {
    let fixture = build_nifs_fixture();
    let view = ccs_claim_to_view(&fixture.fresh);

    // Configure nifs_payloads with an empty CcsClaim shape, then attempt to fill
    // it with the real (non-empty) view. The fill computes the actual
    // view shape and asserts it fits in the configured region — the
    // mismatch panics with "nifs_payloads CcsClaim payload at offset".
    let too_small = NifsCcsClaimShape {
        c_data_entries: 0,
        x_entries: 0,
    };
    let config = skeleton_config_for(
        &too_small,
        &NifsCeClaimShape {
            c_data_entries: 0,
            x_rows: 0,
            x_active_cols: 0,
            r_len: 0,
            y_ring_inner_lens: vec![],
            y_zcol_len: 0,
            s_col_len: 0,
        },
    );

    let layout = FPrimeImageLayout::new(config);
    let mut image = FPrimeImage::new(layout);
    image.fill_nifs_ccs_claim_at(0, &view);
}

#[test]
#[should_panic(expected = "nifs_payloads CeClaim x_active_flat length must match")]
fn phase_1_3b_nifs_ce_x_flat_length_mismatch_panics() {
    let fixture = build_nifs_fixture();
    let mut view = ce_claim_to_view(&fixture.parent);
    // Truncate x_active_flat so its length disagrees with x_rows * x_active_cols.
    if !view.x_active_flat.is_empty() {
        view.x_active_flat.pop();
    } else {
        // If the toy fixture has empty x_active_flat, fabricate the mismatch.
        view.x_rows = 1;
        view.x_active_cols = 4;
        // x_active_flat is still empty → 0 != 4
    }
    let shape = ce_view_shape(&view);
    let fresh_placeholder = NifsCcsClaimShape {
        c_data_entries: 0,
        x_entries: 0,
    };

    // Need a config that doesn't recompute shape from the (broken) view.
    // Build a fake shape with the unbroken size so the layout has room.
    let fake_shape = NifsCeClaimShape {
        c_data_entries: shape.c_data_entries,
        x_rows: view.x_rows.max(1) as usize,
        x_active_cols: view.x_active_cols.max(1) as usize,
        r_len: shape.r_len,
        y_ring_inner_lens: shape.y_ring_inner_lens.clone(),
        y_zcol_len: shape.y_zcol_len,
        s_col_len: shape.s_col_len,
    };
    let mut image = fresh_image(&fresh_placeholder, &fake_shape);
    image.fill_nifs_ce_claim_at(0, &view);
}

// ── FS-bound prefix parity vs production `ce_claim_digest` ───────────────

/// Build the FS-bound prefix that `paper::digest::ce_claim_digest`
/// hashes, MINUS the leading tag (which has no analogue inside the nifs_payloads
/// source-image — nifs_payloads starts at `d`, `kappa`, ...). The returned vector
/// is the F-sequence whose canonical 64-bit decomposition MUST equal
/// the first N bits of `fill_nifs_ce_claim_at`'s output.
fn ce_claim_fs_bound_prefix_f_sequence(view: &NifsCeClaimView) -> Vec<F> {
    let mut p = Vec::new();
    p.push(F::from_u64(view.d));
    p.push(F::from_u64(view.kappa));
    p.push(F::from_u64(view.c_data.len() as u64));
    p.extend_from_slice(&view.c_data);
    p.push(F::from_u64(view.x_rows));
    p.push(F::from_u64(view.x_cols));
    p.push(F::from_u64(view.x_active_cols));
    p.extend_from_slice(&view.x_active_flat);
    p.push(F::from_u64(view.r.len() as u64));
    for k in &view.r {
        p.push(k[0]);
        p.push(k[1]);
    }
    p.push(F::from_u64(view.y_ring.len() as u64));
    for row in &view.y_ring {
        p.push(F::from_u64(row.len() as u64));
        for k in row {
            p.push(k[0]);
            p.push(k[1]);
        }
    }
    p.push(F::from_u64(view.m_in));
    p.extend_from_slice(&view.fold_digest_fields);
    p
}

#[test]
fn phase_1_3b_nifs_ce_claim_fs_bound_prefix_matches_ce_claim_digest() {
    let fixture = build_nifs_fixture();
    let view = ce_claim_to_view(&fixture.parent);
    let shape = ce_view_shape(&view);
    let fresh_placeholder = NifsCcsClaimShape {
        c_data_entries: 0,
        x_entries: 0,
    };
    let mut image = fresh_image(&fresh_placeholder, &shape);
    image.fill_nifs_ce_claim_at(0, &view);

    let prefix_fields = ce_claim_fs_bound_prefix_f_sequence(&view);
    let prefix_bits = prefix_fields.len() * 64;
    let start = image.layout.nifs_payloads.offset;

    // Each F in the prefix must be the canonical 64-bit decomposition
    // stored contiguously at the start of nifs_payloads. This is the invariant the
    // P2 review pointed out: nifs_payloads's prefix bits = production
    // ce_claim_digest preimage prefix (tag aside).
    use neo_fold_clean::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
    use p3_field::PrimeField64;
    for (lane_idx, expected) in prefix_fields.iter().enumerate() {
        let lane_start = start + lane_idx * POSEIDON2_GOLDILOCKS_BITS;
        let mut decoded = F::ZERO;
        let mut pow = F::ONE;
        for bit in 0..POSEIDON2_GOLDILOCKS_BITS {
            let v = image.values[lane_start + bit];
            if v == F::ONE {
                decoded += pow;
            }
            pow *= F::from_u64(2);
        }
        assert_eq!(
            decoded.as_canonical_u64(),
            expected.as_canonical_u64(),
            "FS-bound prefix lane {lane_idx} (nifs_payloads bits {lane_start}..{}) must decode to ce_claim_digest preimage F",
            lane_start + POSEIDON2_GOLDILOCKS_BITS,
        );
    }

    // Sanity: the prefix occupies the FIRST prefix_bits of nifs_payloads, and the
    // remaining bits hold y_zcol/s_col (the non-FS-bound tail).
    let remaining_bits = image.layout.nifs_payloads.bits - prefix_bits;
    eprintln!(
        "phase_1_3b prefix-vs-ce_claim_digest: {} F → {} prefix bits, {} non-FS-bound tail bits",
        prefix_fields.len(),
        prefix_bits,
        remaining_bits,
    );
}

// ── P3 guard-rail: SignedDigit invalid `bits` panics on use ──────────────

#[test]
#[should_panic(expected = "SignedDigit::bits must be in 1..=64")]
fn phase_1_3b_signed_digit_bits_zero_panics_on_use() {
    let bad = LowNormEncoding::SignedDigit { bits: 0 };
    // First use triggers `assert_valid`; bits=0 must panic before
    // shifting `1u64 << (0 - 1)` underflows.
    let _ = bad.limb_count();
}

#[test]
#[should_panic(expected = "SignedDigit::bits must be in 1..=64")]
fn phase_1_3b_signed_digit_bits_over_64_panics_on_use() {
    let bad = LowNormEncoding::SignedDigit { bits: 65 };
    let _ = bad.limb_count();
}
