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
//! ## Domain tag encoding
//!
//! Native `pack_bytes_as_fields(bytes)` lays the static tag out as:
//!   - `F[0] = bytes.len() as u64`
//!   - `F[i+1] = u64::from_le_bytes([bytes[7i..7i+7], 0])` for i = 0..ceil(len/7)
//!
//! Since tags are `&'static [u8]`, we precompute this F-sequence at
//! gadget-emit time and bind each entry to a constant wire (mirrors the
//! pattern used by [`crate::engine::r1cs_circuit::transcript::TranscriptGadget`]).

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::builder::{Lc, R1csBuilder, Var};
use crate::engine::r1cs_circuit::poseidon2::{enforce_poseidon2_hash, DIGEST_LEN};
use crate::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;

/// Tag for `state_x_out_digest` — must match `paper::digest::state_x_out_digest`.
pub const STATE_X_OUT_TAG: &[u8] = b"neo.fold.clean/state_x_out/v1";

/// Tag for `boundary_update_digest`.
pub const BOUNDARY_UPDATE_TAG: &[u8] = b"neo.fold.clean/boundary_update/v1";

/// Tag for `public_trace_update_digest`.
pub const PUBLIC_TRACE_UPDATE_TAG: &[u8] = b"neo.fold.clean/public_trace_update/v1";

/// `z_{i+1} = H(prev_z_i ‖ chunk_digest)`, mirrors
/// [`crate::paper::digest::boundary_update_digest`].
///
/// Both inputs are 4-limb digests (each limb is one Goldilocks F value
/// interpreted as 8 LE bytes natively). Output is the 4-limb result.
pub fn enforce_boundary_update_digest_circuit(
    builder: &mut R1csBuilder,
    prev: [Var; DIGEST_LEN],
    chunk_digest: [Var; DIGEST_LEN],
) -> [Var; DIGEST_LEN] {
    let mut input = alloc_const_tag(builder, BOUNDARY_UPDATE_TAG);
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
    /// `vk_fs_digest` — 4 limbs (from 32 LE bytes natively).
    pub vk_fs_digest: [Var; DIGEST_LEN],
    /// CCS structure digest — 4 native F limbs.
    pub structure_digest: [Var; DIGEST_LEN],
    /// `chunk_count` as a single F wire. The gadget bit-decomposes it
    /// into lo/hi 32-bit halves for the absorb (matching native
    /// `u64_halves`).
    pub chunk_count: Var,
    pub step_count: Var,
    pub initial_boundary: [Var; DIGEST_LEN],
    pub current_boundary: [Var; DIGEST_LEN],
    pub pc: Var,
    pub semantic_acc: [Var; DIGEST_LEN],
    pub construction2_acc: [Var; DIGEST_LEN],
    pub public_trace: [Var; DIGEST_LEN],
}

/// `x_out` — the Construction-2 hash-chain output. Mirrors
/// [`crate::paper::digest::state_x_out_digest`] byte-for-byte (modulo the
/// digest32↔[F;4] conversion at the IO boundary).
pub fn enforce_state_x_out_digest_circuit(
    builder: &mut R1csBuilder,
    inputs: &StateXOutDigestInputs,
) -> [Var; DIGEST_LEN] {
    let mut preimage = alloc_const_tag(builder, STATE_X_OUT_TAG);
    preimage.extend_from_slice(&inputs.vk_fs_digest);
    preimage.extend_from_slice(&inputs.structure_digest);

    let [chunk_lo, chunk_hi] = enforce_u64_halves_from_var(builder, inputs.chunk_count);
    preimage.push(chunk_lo);
    preimage.push(chunk_hi);

    let [step_lo, step_hi] = enforce_u64_halves_from_var(builder, inputs.step_count);
    preimage.push(step_lo);
    preimage.push(step_hi);

    preimage.extend_from_slice(&inputs.initial_boundary);
    preimage.extend_from_slice(&inputs.current_boundary);

    let [pc_lo, pc_hi] = enforce_u64_halves_from_var(builder, inputs.pc);
    preimage.push(pc_lo);
    preimage.push(pc_hi);

    preimage.extend_from_slice(&inputs.semantic_acc);
    preimage.extend_from_slice(&inputs.construction2_acc);
    preimage.extend_from_slice(&inputs.public_trace);

    enforce_poseidon2_hash(builder, &preimage)
}

// Accumulator-digest circuit lives in
// `crate::paper::reductions::accumulator_digest_circuit` — see that module
// for `enforce_accumulator_digest_from_parent_circuit`,
// `enforce_accumulator_digest_from_children_circuit`, and
// `AccumulatorDigestError`. It is shared between F' and SplitNc Π_CCS.V.

// ── Internal helpers ──────────────────────────────────────────────────────

/// Allocate constant wires for a packed domain tag — `pack_bytes_as_fields`
/// at gadget-emit time. Output length is `1 + ceil(len/7)`.
fn alloc_const_tag(builder: &mut R1csBuilder, tag: &'static [u8]) -> Vec<Var> {
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

fn alloc_constant(builder: &mut R1csBuilder, c: F) -> Var {
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
