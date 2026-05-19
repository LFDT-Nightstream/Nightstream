//! Phase 1.1-mini-3a — one-shot Poseidon digest paths inside NIFS.V.
//!
//! This covers the digest-output paths that fit the existing one-shot
//! `encode_poseidon_trace` builder. The five paths and their production
//! mirrors:
//!
//! - R-04 boundary update      → `paper::digest::boundary_update_digest`
//! - R-04 public_trace update  → `paper::digest::public_trace_update_digest`
//! - R-16 fresh CCS digest     → `paper::digest::ccs_claim_digest`
//! - R-19 acc handle (children) → `paper::digest::accumulator_digest_from_claims`
//! - R-36 acc digest output    → `paper::digest::accumulator_digest_from_parent_c_data`
//!
//! Out of scope (mini-3b territory): the sponge-mode transcript paths
//! (R-11, R-12, R-20, R-22, R-23, R-25) and the ring action (mini-4).
//! Lifecycle migration, Spartan, generic AppStep, and any change that
//! turns an `ivc_invariants` test green are also out of scope.

#[path = "../support/mod.rs"]
mod support;

use neo_fold_clean::engine::transcript::Transcript;
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::digest::{
    accumulator_digest_from_claims, accumulator_digest_from_parent_c_data, boundary_update_digest, ccs_claim_digest,
    digest32_as_fields, public_trace_update_digest,
};
use neo_fold_clean::paper::f_prime::poseidon_trace::{
    assert_committed_coords_are_bits, decode_digest_lanes, encode_poseidon_trace,
};
use neo_fold_clean::paper::nifs;
use neo_fold_clean::paper::relations::CcsClaim;
use neo_fold_clean::CeClaim;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

// ── Test-local helpers ───────────────────────────────────────────────────

/// Mirror of `paper::digest::pack_bytes_as_fields` (pub(crate) in production).
fn pack_bytes_as_fields(bytes: &[u8]) -> Vec<F> {
    const BYTES_PER_LIMB: usize = 7;
    let mut out = Vec::with_capacity(1 + bytes.len().div_ceil(BYTES_PER_LIMB));
    out.push(F::from_u64(bytes.len() as u64));
    for chunk in bytes.chunks(BYTES_PER_LIMB) {
        let mut limb = [0u8; 8];
        limb[..chunk.len()].copy_from_slice(chunk);
        out.push(F::from_u64(u64::from_le_bytes(limb)));
    }
    out
}

// ── Preimage builders (byte-for-byte mirrors of paper::digest::*) ────────

fn boundary_update_preimage(prev: [u8; 32], chunk_digest: [F; 4]) -> Vec<F> {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/boundary_update/v1");
    preimage.extend(digest32_as_fields(prev));
    preimage.extend(chunk_digest);
    preimage
}

fn public_trace_update_preimage(prev: [u8; 32], chunk_digest: [F; 4]) -> Vec<F> {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/public_trace_update/v1");
    preimage.extend(digest32_as_fields(prev));
    preimage.extend(chunk_digest);
    preimage
}

fn ccs_claim_digest_preimage(claim: &CcsClaim) -> Vec<F> {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/ccs_claim_digest/v1");
    preimage.push(F::from_u64(claim.c.d as u64));
    preimage.push(F::from_u64(claim.c.kappa as u64));
    preimage.push(F::from_u64(claim.c.data.len() as u64));
    preimage.extend_from_slice(&claim.c.data);
    preimage.push(F::from_u64(claim.x.len() as u64));
    preimage.extend_from_slice(&claim.x);
    preimage.push(F::from_u64(claim.m_in as u64));
    preimage
}

fn accumulator_from_claims_preimage(base: u32, claims: &[CeClaim]) -> Vec<F> {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.next/direct_ccs/accumulator_phi_dec_parent/v1");
    preimage.push(F::from_u64(claims.len() as u64));
    if let Some(first) = claims.first() {
        let parent_len = first.c.data.len();
        preimage.push(F::from_u64(parent_len as u64));
        let base_f = F::from_u64(base as u64);
        let mut powers = Vec::with_capacity(claims.len());
        let mut pow = F::ONE;
        for claim in claims {
            if claim.c.data.len() != parent_len {
                preimage.push(F::from_u64(u64::MAX));
                return preimage;
            }
            powers.push(pow);
            pow *= base_f;
        }
        for lane_idx in 0..parent_len {
            let mut value = F::ZERO;
            for (claim, pow) in claims.iter().zip(powers.iter().copied()) {
                value += claim.c.data[lane_idx] * pow;
            }
            preimage.push(value);
        }
    }
    preimage
}

fn accumulator_from_parent_c_data_preimage(child_count: usize, parent_c_data: &[F]) -> Vec<F> {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.next/direct_ccs/accumulator_phi_dec_parent/v1");
    preimage.push(F::from_u64(child_count as u64));
    if child_count > 0 {
        preimage.push(F::from_u64(parent_c_data.len() as u64));
        preimage.extend_from_slice(parent_c_data);
    }
    preimage
}

// ── NIFS fixture ─────────────────────────────────────────────────────────

struct NifsFixture {
    /// `params.b()` — the b-ary base for accumulator recomposition.
    base: u32,
    /// One fresh CCS claim from `toy_instance`; used for R-16.
    fresh_claim: CcsClaim,
    /// The new running accumulator's children claims; used for R-19.
    running_claims: Vec<CeClaim>,
    /// The Π_RLC parent's commitment data; used for R-36.
    parent_c_data: Vec<F>,
    /// Number of Π_DEC children backing the parent; absorbed into R-36's preimage.
    parent_child_count: usize,
}

fn build_nifs_fixture() -> NifsFixture {
    let prep = support::toy_preprocessing();
    let fresh_inst = vec![support::toy_instance(&prep, 7), support::toy_instance(&prep, 11)];
    let fresh_claim = fresh_inst[0].claim.clone();
    let mut prover_tr = Transcript::session();
    let (running, _proof) = nifs::prove(
        &mut prover_tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        prep.mix_rhos_commits,
        prep.combine_b_pows,
        fresh_inst,
        &RunningInstance::default(),
    )
    .expect("NIFS.P should produce a running accumulator + parent_authority");
    let parent_authority = running
        .parent_authority
        .as_ref()
        .expect("non-empty running must carry the Pi_RLC parent_authority")
        .clone();
    NifsFixture {
        base: prep.params.b(),
        fresh_claim,
        running_claims: running.claims.clone(),
        parent_c_data: parent_authority.c.data.clone(),
        parent_child_count: running.claims.len(),
    }
}

// ── Deterministic synthetic inputs (boundary / public_trace) ─────────────

fn deterministic_prev_digest(seed: u8) -> [u8; 32] {
    let mut d = [0u8; 32];
    for (i, slot) in d.iter_mut().enumerate() {
        *slot = seed.wrapping_add(i as u8);
    }
    d
}

fn deterministic_chunk_digest() -> [F; 4] {
    [
        F::from_u64(0x0123_4567_89ab_cdef),
        F::from_u64(0xfedc_ba98_7654_3210),
        F::from_u64(0xaabb_ccdd_1122_3344),
        F::from_u64(0x5566_7788_99aa_bbcc),
    ]
}

// ── Tests (one per Table A path) ─────────────────────────────────────────

#[test]
fn phase_1_mini_3a_boundary_update_digest() {
    let prev = deterministic_prev_digest(0x80);
    let chunk = deterministic_chunk_digest();
    let reference = digest32_as_fields(boundary_update_digest(prev, chunk));

    let preimage = boundary_update_preimage(prev, chunk);
    let image = encode_poseidon_trace(&preimage);
    let decoded = decode_digest_lanes(&image);

    assert_committed_coords_are_bits(&image.values);
    assert_eq!(decoded, image.digest_native, "decode ↔ builder");
    assert_eq!(decoded, reference, "decode ↔ paper::digest::boundary_update_digest");
    eprintln!(
        "boundary_update: {} F, {} absorbs, {} trace bits",
        preimage.len(),
        image.layout.absorbs,
        image.layout.trace_len,
    );
}

#[test]
fn phase_1_mini_3a_public_trace_update_digest() {
    let prev = deterministic_prev_digest(0x40);
    let chunk = deterministic_chunk_digest();
    let reference = digest32_as_fields(public_trace_update_digest(prev, chunk));

    let preimage = public_trace_update_preimage(prev, chunk);
    let image = encode_poseidon_trace(&preimage);
    let decoded = decode_digest_lanes(&image);

    assert_committed_coords_are_bits(&image.values);
    assert_eq!(decoded, image.digest_native, "decode ↔ builder");
    assert_eq!(decoded, reference, "decode ↔ paper::digest::public_trace_update_digest");
    eprintln!(
        "public_trace_update: {} F, {} absorbs, {} trace bits",
        preimage.len(),
        image.layout.absorbs,
        image.layout.trace_len,
    );
}

#[test]
fn phase_1_mini_3a_fresh_ccs_claim_digest() {
    let fixture = build_nifs_fixture();
    let reference = ccs_claim_digest(&fixture.fresh_claim);

    let preimage = ccs_claim_digest_preimage(&fixture.fresh_claim);
    let image = encode_poseidon_trace(&preimage);
    let decoded = decode_digest_lanes(&image);

    assert_committed_coords_are_bits(&image.values);
    assert_eq!(decoded, image.digest_native, "decode ↔ builder");
    assert_eq!(decoded, reference, "decode ↔ paper::digest::ccs_claim_digest");
    eprintln!(
        "fresh ccs_claim_digest: {} F, {} absorbs, {} trace bits (~{:.2} KiB)",
        preimage.len(),
        image.layout.absorbs,
        image.layout.trace_len,
        image.layout.trace_len as f64 / 8.0 / 1024.0,
    );
}

#[test]
fn phase_1_mini_3a_accumulator_from_claims_digest() {
    let fixture = build_nifs_fixture();
    let reference = digest32_as_fields(accumulator_digest_from_claims(fixture.base, &fixture.running_claims));

    let preimage = accumulator_from_claims_preimage(fixture.base, &fixture.running_claims);
    let image = encode_poseidon_trace(&preimage);
    let decoded = decode_digest_lanes(&image);

    assert_committed_coords_are_bits(&image.values);
    assert_eq!(decoded, image.digest_native, "decode ↔ builder");
    assert_eq!(
        decoded, reference,
        "decode ↔ paper::digest::accumulator_digest_from_claims"
    );
    eprintln!(
        "accumulator_from_claims: {} F, {} absorbs, {} trace bits (~{:.2} KiB)",
        preimage.len(),
        image.layout.absorbs,
        image.layout.trace_len,
        image.layout.trace_len as f64 / 8.0 / 1024.0,
    );
}

#[test]
fn phase_1_mini_3a_accumulator_from_parent_c_data_digest() {
    let fixture = build_nifs_fixture();
    let reference = digest32_as_fields(accumulator_digest_from_parent_c_data(
        fixture.parent_child_count,
        &fixture.parent_c_data,
    ));

    let preimage = accumulator_from_parent_c_data_preimage(fixture.parent_child_count, &fixture.parent_c_data);
    let image = encode_poseidon_trace(&preimage);
    let decoded = decode_digest_lanes(&image);

    assert_committed_coords_are_bits(&image.values);
    assert_eq!(decoded, image.digest_native, "decode ↔ builder");
    assert_eq!(
        decoded, reference,
        "decode ↔ paper::digest::accumulator_digest_from_parent_c_data"
    );
    eprintln!(
        "accumulator_from_parent_c_data: {} F, {} absorbs, {} trace bits (~{:.2} KiB)",
        preimage.len(),
        image.layout.absorbs,
        image.layout.trace_len,
        image.layout.trace_len as f64 / 8.0 / 1024.0,
    );
}
