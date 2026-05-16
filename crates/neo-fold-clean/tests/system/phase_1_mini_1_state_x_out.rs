//! Phase 1.1-mini-1 — state_x_out Poseidon trace layout + encoder + parity.
//!
//! This lays out the bit-backed Poseidon2 trace for the `state_x_out`
//! digest only, encodes it from a deterministic F' step input, decodes
//! the digest bits back to `[F; 4]`, and asserts equality against
//! `paper::digest::state_x_out_digest` and the bit-backed builder's own
//! reported digest.
//!
//! Out of scope (per council instructions):
//! - parent_authority digest, full NIFS, ring action (Phase 1.1-mini-2..4).
//! - lifecycle migration (Phase 1.5).
//! - generic `AppStep` trait.
//! - Spartan.
//! - any change to the existing F' R1CS emitter or the direct-CCS
//!   audit lifecycle.
//! - changes to the three failing `ivc_invariants` tests.
//!
//! The acceptance gate (Phase 1.1-mini-1):
//! 1. Layout offsets monotonic and non-overlapping.
//! 2. All committed coords in `{0, 1}` (the b=2 invariant) except the
//!    constant-one slot.
//! 3. Serialization round-trip works.
//! 4. Decoded `state_x_out` digest equals native `state_x_out_digest()`
//!    and the bit-backed builder's reported digest.

use neo_fold_clean::engine::ccs_native::poseidon2::{
    POSEIDON2_DIGEST_LEN, POSEIDON2_GOLDILOCKS_BITS, POSEIDON2_RATE, POSEIDON2_WIDTH,
};
use neo_fold_clean::paper::digest::{digest32_as_fields, state_x_out_digest};
use neo_fold_clean::paper::f_prime::poseidon_trace::{
    assert_committed_coords_are_bits, decode_digest_lanes, encode_poseidon_trace, PoseidonTraceLayout,
    BITS_PER_PERMUTATION,
};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

// ── Preimage helpers (mirror `paper::digest::state_x_out_digest`) ────────

/// Mirror of the private `paper::digest::pack_bytes_as_fields`. The first
/// field carries `bytes.len()`; subsequent fields pack 7 bytes each
/// little-endian. Reimplemented here so the test can construct the same
/// preimage the production `state_x_out_digest` builds, without exposing
/// `pack_bytes_as_fields` as `pub`.
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

#[inline]
fn u64_halves(value: u64) -> [F; 2] {
    [F::from_u64(value & 0xffff_ffff), F::from_u64(value >> 32)]
}

#[allow(clippy::too_many_arguments)]
fn build_state_x_out_preimage(
    vk_fs_digest: [u8; 32],
    structure_digest: &[F; 4],
    chunk_count: u64,
    step_count: u64,
    initial_boundary: [u8; 32],
    current_boundary: [u8; 32],
    pc: u64,
    semantic_acc: [u8; 32],
    construction2_acc: [u8; 32],
    public_trace: [u8; 32],
) -> Vec<F> {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/state_x_out/v1");
    preimage.extend(digest32_as_fields(vk_fs_digest));
    preimage.extend(structure_digest.iter().copied());
    preimage.extend(u64_halves(chunk_count));
    preimage.extend(u64_halves(step_count));
    preimage.extend(digest32_as_fields(initial_boundary));
    preimage.extend(digest32_as_fields(current_boundary));
    preimage.extend(u64_halves(pc));
    preimage.extend(digest32_as_fields(semantic_acc));
    preimage.extend(digest32_as_fields(construction2_acc));
    preimage.extend(digest32_as_fields(public_trace));
    preimage
}

/// Deterministic test inputs mirroring a recursive F' step's
/// state_x_out call. Values are chosen to be unique and non-zero so a
/// regression on any single field is detectable in the digest output.
fn deterministic_state_x_out_inputs() -> (
    [u8; 32],
    [F; 4],
    u64,
    u64,
    [u8; 32],
    [u8; 32],
    u64,
    [u8; 32],
    [u8; 32],
    [u8; 32],
) {
    let mk_digest = |seed: u8| {
        let mut d = [0u8; 32];
        for (i, slot) in d.iter_mut().enumerate() {
            *slot = seed.wrapping_add(i as u8);
        }
        d
    };
    let vk_fs_digest = mk_digest(0x10);
    let structure_digest = [
        F::from_u64(0x0123456789abcdef),
        F::from_u64(0xfedcba9876543210),
        F::from_u64(0xaabbccdd11223344),
        F::from_u64(0x55667788_99aabbcc),
    ];
    let chunk_count: u64 = 3;
    let step_count: u64 = 5;
    let initial_boundary = mk_digest(0x20);
    let current_boundary = mk_digest(0x30);
    let pc: u64 = 0; // TRIVIAL_PC
    let semantic_acc = mk_digest(0x40);
    let construction2_acc = mk_digest(0x50);
    let public_trace = mk_digest(0x60);
    (
        vk_fs_digest,
        structure_digest,
        chunk_count,
        step_count,
        initial_boundary,
        current_boundary,
        pc,
        semantic_acc,
        construction2_acc,
        public_trace,
    )
}

// ── Acceptance tests ─────────────────────────────────────────────────────

#[test]
fn phase_1_mini_1_layout_offsets_monotonic_and_non_overlapping() {
    // Cover the typical state_x_out preimage length (~40 F values),
    // plus boundary cases that exercise the absorb-rounding path.
    for preimage_len in [1, 4, 5, 7, 8, 40, 41, 100] {
        let layout = PoseidonTraceLayout::from_preimage_len(preimage_len);

        assert_eq!(layout.constant_slot, 0);
        assert!(layout.trace_start > layout.constant_slot);
        assert!(layout.end() > layout.trace_start);
        assert_eq!(
            layout.absorbs,
            preimage_len.div_ceil(POSEIDON2_RATE) + 1,
            "absorb count must include the padding permutation"
        );
        assert_eq!(layout.trace_len, layout.absorbs * BITS_PER_PERMUTATION);

        let final_state = layout.final_state_start();
        assert!(final_state >= layout.trace_start);
        assert!(final_state + POSEIDON2_WIDTH * POSEIDON2_GOLDILOCKS_BITS == layout.end());

        // Lanes monotonic non-overlapping
        for lane in 0..POSEIDON2_DIGEST_LEN {
            let start = layout.digest_lane_start(lane);
            assert!(start >= final_state);
            assert!(start + POSEIDON2_GOLDILOCKS_BITS <= layout.end());
            if lane > 0 {
                let prev = layout.digest_lane_start(lane - 1);
                assert_eq!(
                    start,
                    prev + POSEIDON2_GOLDILOCKS_BITS,
                    "digest lanes must be contiguous"
                );
            }
        }
    }
}

#[test]
fn phase_1_mini_1_committed_coords_are_bits() {
    let (vk_fs, structure, cc, sc, init_b, curr_b, pc, sem_acc, c2_acc, pub_trace) = deterministic_state_x_out_inputs();
    let preimage = build_state_x_out_preimage(
        vk_fs, &structure, cc, sc, init_b, curr_b, pc, sem_acc, c2_acc, pub_trace,
    );
    let image = encode_poseidon_trace(&preimage);
    assert_committed_coords_are_bits(&image.values);
}

#[test]
fn phase_1_mini_1_serialization_round_trip() {
    let (vk_fs, structure, cc, sc, init_b, curr_b, pc, sem_acc, c2_acc, pub_trace) = deterministic_state_x_out_inputs();
    let preimage = build_state_x_out_preimage(
        vk_fs, &structure, cc, sc, init_b, curr_b, pc, sem_acc, c2_acc, pub_trace,
    );
    let image = encode_poseidon_trace(&preimage);

    let bytes: Vec<u64> = image.values.iter().map(|v| v.as_canonical_u64()).collect();
    let recovered: Vec<F> = bytes.iter().map(|&u| F::from_u64(u)).collect();
    assert_eq!(image.values, recovered);

    // Also verify the layout descriptor itself round-trips (Clone + Eq).
    let layout_copy = image.layout;
    assert_eq!(image.layout, layout_copy);
}

#[test]
fn phase_1_mini_1_decoded_state_x_out_matches_native_and_builder() {
    let (vk_fs, structure, cc, sc, init_b, curr_b, pc, sem_acc, c2_acc, pub_trace) = deterministic_state_x_out_inputs();
    let preimage = build_state_x_out_preimage(
        vk_fs, &structure, cc, sc, init_b, curr_b, pc, sem_acc, c2_acc, pub_trace,
    );

    // Reference path: production `state_x_out_digest`.
    let reference_bytes = state_x_out_digest(
        vk_fs, &structure, cc, sc, init_b, curr_b, pc, sem_acc, c2_acc, pub_trace,
    );
    let reference_fields = digest32_as_fields(reference_bytes);

    // Encoded path: bit-backed trace.
    let image = encode_poseidon_trace(&preimage);
    let decoded = decode_digest_lanes(&image);

    // Triple parity: decoded bits ↔ builder's native digest ↔ production reference.
    assert_eq!(
        decoded, image.digest_native,
        "decoded digest bits must match the bit-backed builder's reported digest"
    );
    assert_eq!(
        image.digest_native, reference_fields,
        "bit-backed builder's digest must match production state_x_out_digest"
    );
    assert_eq!(
        decoded, reference_fields,
        "decoded digest must match production state_x_out_digest (transitive parity)"
    );
}
