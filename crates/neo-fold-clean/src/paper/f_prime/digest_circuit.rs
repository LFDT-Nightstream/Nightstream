//! In-circuit mirrors of the Construction-2 hash-chain digests.
//!
//! Each gadget here is the R1CS twin of a function in
//! [`crate::paper::digest`]. The native side hashes a specific preimage
//! layout via Poseidon2; the in-circuit side builds the same layout from
//! wire-allocated inputs and constants and invokes
//! [`enforce_poseidon2_hash`] to produce a `[Var; 4]` digest.
//!
//! **Soundness Invariant I-5**: any change here must move in lockstep with
//! the native `digest::*` it mirrors. Parity is enforced by the tests in
//! `tests/f_prime/digest_circuit.rs`.
//!
//! ## Domain constants
//!
//! The hot canonical `state_x_out` hash uses a compact single-field
//! domain ID instead of a byte-packed ASCII tag. The legacy
//! `boundary_update` helper uses the same compact-domain convention, even
//! though the canonical F' image no longer emits that trace.
//!
//! Legacy helpers that are no longer emitted in the canonical F' image
//! still mirror native `pack_bytes_as_fields(bytes)`, where the static tag
//! is laid out as:
//!
//!   - `F[0] = bytes.len() as u64`
//!   - `F[i+1] = u64::from_le_bytes([bytes[7i..7i+7], 0])` for i = 0..ceil(len/7)
//!
//! Since those tags are `&'static [u8]`, we precompute this F-sequence at
//! gadget-emit time and bind each entry to a constant wire.

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::builder::{Lc, R1csBuilder, Var};
use crate::engine::r1cs_circuit::poseidon2::{enforce_poseidon2_hash, DIGEST_LEN};
use crate::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;
use crate::paper::digest::{
    StateXOutDigestMode, F_PRIME_BOUNDARY_UPDATE_DOMAIN, F_PRIME_STATE_X_OUT_DOMAIN, NEBULA_ADV_PRESENT_MARKER,
};

/// Tag for `public_trace_update_digest`.
pub const PUBLIC_TRACE_UPDATE_TAG: &[u8] = b"neo.fold.clean/public_trace_update/v1";

/// Legacy `z_{i+1} = H(prev_z_i ‖ chunk_digest)`, mirrors
/// [`crate::paper::digest::boundary_update_digest`].
///
/// Canonical F' now uses `new_z_i = chunk_digest` plus a linear mirror row.
///
/// Both inputs are 4-limb digests (each limb is one Goldilocks F value
/// interpreted as 8 LE bytes natively). Output is the 4-limb result.
pub fn enforce_boundary_update_digest_circuit(
    builder: &mut R1csBuilder,
    prev: [Var; DIGEST_LEN],
    chunk_digest: [Var; DIGEST_LEN],
) -> [Var; DIGEST_LEN] {
    let mut input = vec![alloc_constant(builder, F::from_u64(F_PRIME_BOUNDARY_UPDATE_DOMAIN))];
    input.extend_from_slice(&prev);
    input.extend_from_slice(&chunk_digest);
    enforce_poseidon2_hash(builder, &input)
}

/// `public_trace_{i+1} = H(prev_trace ‖ chunk_digest)`, mirrors
/// [`crate::paper::digest::public_trace_update_digest`].
pub fn enforce_public_trace_update_digest_circuit(
    builder: &mut R1csBuilder,
    prev: [Var; DIGEST_LEN],
    chunk_digest: [Var; DIGEST_LEN],
) -> [Var; DIGEST_LEN] {
    let mut input = alloc_const_tag(builder, PUBLIC_TRACE_UPDATE_TAG);
    input.extend_from_slice(&prev);
    input.extend_from_slice(&chunk_digest);
    enforce_poseidon2_hash(builder, &input)
}

/// Inputs to [`enforce_state_x_out_digest_circuit`]. Mirrors the argument
/// list of native [`crate::paper::digest::state_x_out_digest`].
pub struct StateXOutDigestInputs {
    pub mode: StateXOutDigestMode,
    /// `vk_fs_digest` — 4 limbs (from 32 LE bytes natively).
    pub vk_fs_digest: [Var; DIGEST_LEN],
    /// SplitNc verifier header carried as part of `vk_fs`. These are
    /// witness wires so folded F' does not embed a hash of its own matrices.
    pub pi_ccs_header_bundle: [Var; DIGEST_LEN],
    /// CCS structure digest — 4 native F limbs.
    ///
    /// Retained on the input struct because callers already carry it,
    /// but not absorbed directly: `vk_fs_digest` is verifier-derived
    /// from the structure digest, params, public-input length, and
    /// semantic-state seed.
    pub structure_digest: [Var; DIGEST_LEN],
    /// Chunk counter carried by Construction-2 state. It is absorbed by
    /// `state_x_out`; unlike `z_0`, the O(1) verifier cannot rederive it
    /// from preprocessing alone.
    pub chunk_count: Var,
    pub step_count: Var,
    /// `z_0` — retained for call-site parity, but not absorbed directly.
    /// It is verifier-derived from `structure_digest` and
    /// `public_input_len`, both already absorbed by `vk_fs_digest`.
    pub initial_boundary: [Var; DIGEST_LEN],
    pub current_boundary: [Var; DIGEST_LEN],
    /// Program counter in HyperNova's recursive-link preimage. This build
    /// has a single `F'_j`, so callers also pin it to `TRIVIAL_PC`, but
    /// the digest still absorbs it directly.
    pub pc: Var,
    pub semantic_acc: [Var; DIGEST_LEN],
    pub construction2_acc: [Var; DIGEST_LEN],
    /// Retained for call-site shape parity with native `state_x_out_digest`.
    /// Canonical F' constrains `public_trace == z_i` separately, so the
    /// digest does not absorb this duplicate lane.
    pub public_trace: [Var; DIGEST_LEN],
}

/// `x_out` — the Construction-2 hash-chain output. Mirrors
/// [`crate::paper::digest::state_x_out_digest`] byte-for-byte (modulo the
/// digest32↔[F;4] conversion at the IO boundary).
pub fn enforce_state_x_out_digest_circuit(
    builder: &mut R1csBuilder,
    inputs: &StateXOutDigestInputs,
) -> [Var; DIGEST_LEN] {
    enforce_state_x_out_digest_inner(builder, inputs, None)
}

/// Nebula-chain variant of [`enforce_state_x_out_digest_circuit`]. The
/// lane digest extension is present at both open and closed segment states;
/// plain chains continue to use the original entrypoint unchanged.
pub fn enforce_state_x_out_digest_with_nebula_circuit(
    builder: &mut R1csBuilder,
    inputs: &StateXOutDigestInputs,
    nebula_lane_digest: [Var; DIGEST_LEN],
) -> [Var; DIGEST_LEN] {
    enforce_state_x_out_digest_inner(builder, inputs, Some(nebula_lane_digest))
}

fn enforce_state_x_out_digest_inner(
    builder: &mut R1csBuilder,
    inputs: &StateXOutDigestInputs,
    nebula_lane_digest: Option<[Var; DIGEST_LEN]>,
) -> [Var; DIGEST_LEN] {
    let mut preimage = vec![alloc_constant(builder, F::from_u64(F_PRIME_STATE_X_OUT_DOMAIN))];
    preimage.extend_from_slice(&inputs.vk_fs_digest);
    preimage.extend_from_slice(&inputs.pi_ccs_header_bundle);

    let [chunk_lo, chunk_hi] = enforce_u64_halves_from_var(builder, inputs.chunk_count);
    preimage.push(chunk_lo);
    preimage.push(chunk_hi);

    let [step_lo, step_hi] = enforce_u64_halves_from_var(builder, inputs.step_count);
    preimage.push(step_lo);
    preimage.push(step_hi);

    let [pc_lo, pc_hi] = enforce_u64_halves_from_var(builder, inputs.pc);
    preimage.push(pc_lo);
    preimage.push(pc_hi);

    preimage.extend_from_slice(&inputs.current_boundary);

    if matches!(inputs.mode, StateXOutDigestMode::Stateful) {
        preimage.extend_from_slice(&inputs.semantic_acc);
    }
    preimage.extend_from_slice(&inputs.construction2_acc);

    if let Some(lane) = nebula_lane_digest {
        preimage.push(alloc_constant(builder, F::from_u64(NEBULA_ADV_PRESENT_MARKER)));
        preimage.extend_from_slice(&lane);
    }

    enforce_poseidon2_hash(builder, &preimage)
}

// Accumulator-digest circuit lives in
// `crate::paper::reductions::accumulator_digest_circuit` — see that module
// for the full-running Construction-2 handle that binds HyperNova's `U_i`
// as child CE-claim digests plus the Π_RLC parent-authority digest.

// ── Internal helpers ──────────────────────────────────────────────────────

/// Allocate constant wires for a packed domain tag — `pack_bytes_as_fields`
/// at gadget-emit time. Output length is `1 + ceil(len/7)`. Shared with
/// the Nebula lane mirrors (`nebula_lane_circuit`).
pub(crate) fn alloc_const_tag(builder: &mut R1csBuilder, tag: &'static [u8]) -> Vec<Var> {
    const BYTES_PER_LIMB: usize = 7;
    let mut out = Vec::with_capacity(1 + tag.len().div_ceil(BYTES_PER_LIMB));
    out.push(alloc_constant(builder, F::from_u64(tag.len() as u64)));
    for chunk in tag.chunks(BYTES_PER_LIMB) {
        let mut limb = [0u8; 8];
        limb[..chunk.len()].copy_from_slice(chunk);
        out.push(alloc_constant(builder, F::from_u64(u64::from_le_bytes(limb))));
    }
    out
}

pub(crate) fn alloc_constant(builder: &mut R1csBuilder, c: F) -> Var {
    let v = builder.alloc(c);
    builder.enforce_eq(&Lc::from_var(v), &Lc::from_const(c));
    v
}

/// Split an F-valued Var (canonical u64 < p) into `(lo, hi)` 32-bit halves
/// matching native `u64_halves`. Uses the canonical bit-decomposition
/// helper so non-canonical witnesses are rejected by the canonicity gate.
fn enforce_u64_halves_from_var(builder: &mut R1csBuilder, var: Var) -> [Var; 2] {
    let bits = decompose_var_to_u64_bits(builder, var);
    let lo = compose_bits(builder, &bits[..32]);
    let hi = compose_bits(builder, &bits[32..]);
    [lo, hi]
}

fn compose_bits(builder: &mut R1csBuilder, bits: &[Var]) -> Var {
    let mut lc = Lc::zero();
    let mut pow2 = F::ONE;
    for &b in bits {
        lc.add_term(b, pow2);
        pow2 = pow2 + pow2;
    }
    let v = builder.alloc(builder.eval(&lc));
    builder.enforce_eq(&Lc::from_var(v), &lc);
    v
}
