//! SplitNcV1 — engine-parity transcript driving.
//!
//! Mirrors the binding-and-sampling phases of
//! `optimized_verify_with_cache_and_public_instance_digest_impl`:
//!
//! 1. `bind_header_and_instance_digest_with_digest` (raw absorbs of
//!    `[11, hb…]` and `[12, id…]`).
//! 2. `bind_me_inputs_accumulator_handle` (raw absorbs of `[4]`,
//!    `[5, count]`, and the full-running accumulator handle with leading
//!    tag `[6, …]`).
//! 3. `sample_challenges` (raw `[2]` then K-batch squeeze for α/β_a/β_r/γ).
//! 4. `sample_beta_m` (raw `[3]` then K-batch squeeze for β_m).
//!
//! All raw-domain tag IDs come straight from `neo-reductions` so any future
//! renumbering is a compile error here.

use neo_math::F;
use neo_reductions::engines::utils::{
    PI_CCS_HEADER_BUNDLE_RAW_TAG, PI_CCS_INSTANCE_DIGEST_RAW_TAG, PI_CCS_ME_ACCUMULATOR_HANDLE_RAW_TAG,
    PI_CCS_ME_COUNT_RAW_TAG, PI_CCS_ME_INPUTS_RAW_DOMAIN_TAG,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::{alloc_constant_var, Error};
use crate::engine::r1cs_circuit::builder::{Lc, Var};
use crate::engine::r1cs_circuit::field_ext::KVar;
use crate::engine::r1cs_circuit::transcript::TranscriptGadget;
use crate::engine::r1cs_circuit::R1csBuilder;

/// Engine-side challenge bundle squeezed out of one `sample_challenges` call.
///
/// Mirrors `neo_reductions::engines::optimized_engine::Challenges` but holds
/// in-circuit `KVar` wires instead of native `K` values. `beta_m` is sampled
/// separately via [`sample_engine_beta_m`] and not present here.
#[derive(Clone, Debug)]
pub struct EngineChallenges {
    /// `α ∈ K^{ell_d}` — the Ajtai-domain challenge for χ_α / chi tabulation.
    pub alpha: Vec<KVar>,
    /// `β_a ∈ K^{ell_d}` — the row-domain mixing challenge for FE/NC terminal
    /// identities (`eq(α', β_a)`).
    pub beta_a: Vec<KVar>,
    /// `β_r ∈ K^{ell_n}` — the column-domain mixing challenge for FE terminal
    /// identity (`eq(r', β_r)`).
    pub beta_r: Vec<KVar>,
    /// `γ ∈ K` — geometric weight separating per-instance contributions.
    pub gamma: KVar,
}

/// Mirror of `neo_reductions::engines::utils::sample_challenges`.
///
/// Drives the transcript:
/// 1. Absorbs the raw `[F::from_u64(2)]` domain tag.
/// 2. Squeezes `2·(ell_d + ell + 1)` field lanes via
///    [`TranscriptGadget::challenge_fields_raw`], where `ell = ell_d + ell_n`.
/// 3. Packs the lanes pairwise into `ell_d + ell + 1` `KVar`s.
/// 4. Slices the batch as `[α | β_a | β_r | γ]`.
pub fn sample_engine_challenges(
    builder: &mut R1csBuilder,
    transcript: &mut TranscriptGadget,
    ell_d: usize,
    ell_n: usize,
) -> EngineChallenges {
    let ell = ell_d + ell_n;
    transcript.append_fields_raw_const(builder, &[F::from_u64(2)]);
    let total_k = ell_d + ell + 1;
    let batch = challenge_k_batch_raw(builder, transcript, total_k);

    let mut iter = batch.into_iter();
    let alpha: Vec<KVar> = (&mut iter).take(ell_d).collect();
    let beta_a: Vec<KVar> = (&mut iter).take(ell_d).collect();
    let beta_r: Vec<KVar> = (&mut iter).take(ell_n).collect();
    let gamma = iter
        .next()
        .expect("sample_engine_challenges: γ slot missing");
    debug_assert!(iter.next().is_none(), "sample_engine_challenges: batch over-drawn");

    EngineChallenges {
        alpha,
        beta_a,
        beta_r,
        gamma,
    }
}

/// Mirror of `neo_reductions::engines::utils::sample_beta_m`.
///
/// 1. Absorbs the raw `[F::from_u64(3)]` domain tag.
/// 2. Squeezes `2 · ell_m` lanes raw and packs as `ell_m` `KVar`s.
pub fn sample_engine_beta_m(builder: &mut R1csBuilder, transcript: &mut TranscriptGadget, ell_m: usize) -> Vec<KVar> {
    transcript.append_fields_raw_const(builder, &[F::from_u64(3)]);
    challenge_k_batch_raw(builder, transcript, ell_m)
}

/// Raw K-batch squeeze: mirrors `sample_k_batch(tr, count)` in
/// `neo_reductions::engines::utils`. Squeezes `2 · count` field lanes via
/// [`TranscriptGadget::challenge_fields_raw`] and packs them pairwise as
/// `KVar { c0, c1 }`.
pub(super) fn challenge_k_batch_raw(
    builder: &mut R1csBuilder,
    transcript: &mut TranscriptGadget,
    count: usize,
) -> Vec<KVar> {
    if count == 0 {
        return Vec::new();
    }
    let lanes = transcript.challenge_fields_raw(builder, 2 * count);
    lanes
        .chunks_exact(2)
        .map(|p| KVar::new(p[0], p[1]))
        .collect()
}

/// Mirror of `bind_header_and_instance_digest_with_digest` in
/// `neo_reductions::engines::utils`. This is the variant `neo-fold-clean`
/// always uses (via [`crate::engine::optimized::verify_pi_ccs`]) — the
/// raw-instances variant is dead code from our perspective.
///
/// The native call performs two raw absorbs, each 5 fields wide:
/// 1. `[PI_CCS_HEADER_BUNDLE_RAW_TAG, hb[0..4]]` — the header bundle is a
///    pure function of `(params, s, dims, mat_digest)`, all of which are
///    static at F'-build time, so we accept `header_bundle` as a const
///    `[F; 4]` and bind it on the wire side via `append_fields_raw_vars`
///    after a one-time const allocation.
/// 2. `[PI_CCS_INSTANCE_DIGEST_RAW_TAG, id[0..4]]` — `id` is a witness
///    digest the caller must derive from the actual fresh/running claim
///    wires (e.g. via [`super::enforce_pi_ccs_instance_digest`]). The four
///    digest lanes are passed in as `Var`s that the caller already pinned
///    to the recomputed value.
pub fn absorb_engine_header_bundle_and_instance_digest(
    builder: &mut R1csBuilder,
    transcript: &mut TranscriptGadget,
    header_bundle: [F; 4],
    instance_digest_wires: [Var; 4],
) {
    let tag_hb = alloc_constant_var(builder, F::from_u64(PI_CCS_HEADER_BUNDLE_RAW_TAG));
    let hb0 = alloc_constant_var(builder, header_bundle[0]);
    let hb1 = alloc_constant_var(builder, header_bundle[1]);
    let hb2 = alloc_constant_var(builder, header_bundle[2]);
    let hb3 = alloc_constant_var(builder, header_bundle[3]);
    transcript.append_fields_raw_vars(builder, &[tag_hb, hb0, hb1, hb2, hb3]);

    let tag_id = alloc_constant_var(builder, F::from_u64(PI_CCS_INSTANCE_DIGEST_RAW_TAG));
    transcript.append_fields_raw_vars(
        builder,
        &[
            tag_id,
            instance_digest_wires[0],
            instance_digest_wires[1],
            instance_digest_wires[2],
            instance_digest_wires[3],
        ],
    );
}

/// Mirror of `bind_me_inputs_accumulator_handle` in
/// `neo_reductions::engines::utils`. Performs three raw absorbs:
/// 1. `[PI_CCS_ME_INPUTS_RAW_DOMAIN_TAG]` — domain tag (1 field).
/// 2. `[PI_CCS_ME_COUNT_RAW_TAG, count]` — count header (2 fields).
/// 3. `[PI_CCS_ME_ACCUMULATOR_HANDLE_RAW_TAG, h[0], h[1], h[2], h[3]]` —
///    the single 4-lane handle replaces the per-claim ME-input
///    projection digest stream the retired bind variant used.
///
/// The handle wires must already be bound by the caller to a digest
/// recomputed from authoritative running-CE-claim data (e.g. the
/// accumulator-digest gadget in
/// [`crate::paper::reductions::accumulator_digest_circuit`]). It is not
/// a prover-supplied value.
pub fn absorb_engine_me_inputs_accumulator_handle(
    builder: &mut R1csBuilder,
    transcript: &mut TranscriptGadget,
    me_input_count: usize,
    handle: [Var; 4],
) {
    // 1. Domain tag.
    let dom_tag = alloc_constant_var(builder, F::from_u64(PI_CCS_ME_INPUTS_RAW_DOMAIN_TAG));
    transcript.append_fields_raw_vars(builder, &[dom_tag]);

    // 2. Count header.
    let count_tag = alloc_constant_var(builder, F::from_u64(PI_CCS_ME_COUNT_RAW_TAG));
    let count_val = alloc_constant_var(builder, F::from_u64(me_input_count as u64));
    transcript.append_fields_raw_vars(builder, &[count_tag, count_val]);

    // 3. Single accumulator-handle absorb (5 fields: tag + 4 digest lanes).
    let handle_tag = alloc_constant_var(builder, F::from_u64(PI_CCS_ME_ACCUMULATOR_HANDLE_RAW_TAG));
    transcript.append_fields_raw_vars(builder, &[handle_tag, handle[0], handle[1], handle[2], handle[3]]);
}

// ── Header-digest catch-up (sub-step I) ───────────────────────────────────

/// Decode native `PiCcsProof.header_digest` (32 bytes) into the four
/// Goldilocks lanes that [`TranscriptGadget::digest_fields`] (and the native
/// `Poseidon2Transcript::digest32`) emit.
///
/// Each lane is 8 little-endian bytes read as a canonical Goldilocks element,
/// matching the native `digest32` byte layout. Reject noncanonical limbs
/// instead of reducing them with `F::from_u64`; otherwise a proof byte string
/// containing `p + x` would alias to the transcript lane `x` in-circuit while
/// the native verifier rejects the raw bytes.
pub fn header_digest_bytes_to_fields(bytes: &[u8]) -> Result<[F; 4], Error> {
    if bytes.len() != 32 {
        return Err(Error::Shape(format!(
            "Π_CCS header_digest must be 32 bytes, got {}",
            bytes.len()
        )));
    }
    let mut out = [F::ZERO; 4];
    for (i, slot) in out.iter_mut().enumerate() {
        let mut limb = [0u8; 8];
        limb.copy_from_slice(&bytes[i * 8..(i + 1) * 8]);
        let value = u64::from_le_bytes(limb);
        if value >= F::ORDER_U64 {
            return Err(Error::Shape(format!(
                "Π_CCS header_digest limb {i} is noncanonical: {value} >= field modulus {}",
                F::ORDER_U64
            )));
        }
        *slot = F::from_u64(value);
    }
    Ok(out)
}

/// Native `crate::engine::optimized::verify_pi_ccs` performs a catch-up
/// `tr.digest32()` after the SplitNc engine verifier returns, then compares
/// the squeezed digest against `proof.header_digest` before any downstream
/// Π_RLC ρ sampling.
///
/// This helper mirrors that exactly:
/// 1. `digest_fields()` advances the in-circuit transcript by the same
///    `absorb_const_elem(F::ONE) → permute → take first 4 lanes` squeeze the
///    native sponge applies.
/// 2. Each observed digest lane is constrained equal to the proof's recorded
///    header-digest lane.
///
/// After this call, downstream Π_RLC.V ρ-sampling sees the same transcript
/// state as the native verifier does, so the two-side challenges agree.
pub fn enforce_header_digest_catch_up(
    builder: &mut R1csBuilder,
    transcript: &mut TranscriptGadget,
    expected_header_digest: [F; 4],
) {
    let observed = transcript.digest_fields(builder);
    for (wire, expected) in observed.into_iter().zip(expected_header_digest) {
        builder.enforce_eq(&Lc::from_var(wire), &Lc::from_const(expected));
    }
}
