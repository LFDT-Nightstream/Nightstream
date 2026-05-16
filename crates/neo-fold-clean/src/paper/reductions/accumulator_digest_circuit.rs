//! Neutral in-circuit mirror of `paper::digest::accumulator_digest_from_claims`.
//!
//! Owned here (not under `paper::f_prime::digest_circuit`) because two
//! reduction-layer call sites need it:
//!
//! - **F' R1CS**: binds `acc_digest_in` to the digest of the actual
//!   `running` accumulator (`enforce_f_prime_recursive_step_circuit`).
//! - **SplitNc Π_CCS.V**: derives the ME-input accumulator handle absorbed
//!   into the transcript instead of per-claim ME-input projection digests
//!   (matches the `bind_me_inputs_accumulator_handle` mode in
//!   `neo_reductions::engines::utils`).
//!
//! **Soundness Invariant I-5**: any change here must move in lockstep
//! with the native `paper::digest::accumulator_digest_from_claims` it
//! mirrors. The static `ACCUMULATOR_TAG` is the legacy
//! `neo.fold.next/...` string, preserved so absorbed values stay
//! reproducible across the refactor.

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::builder::{Lc, R1csBuilder, Var};
use crate::engine::r1cs_circuit::poseidon2::{enforce_poseidon2_hash, DIGEST_LEN};

/// Tag for `accumulator_digest_from_claims`. The `neo.fold.next/...`
/// string is **legacy stable** — kept identical to the native
/// counterpart so existing absorbed values remain reproducible across
/// the migration. The auditor reads this as "the accumulator
/// commitment hash" and compares the absorb against the native gadget;
/// the tag string is opaque from the soundness side.
pub const ACCUMULATOR_TAG: &[u8] = b"neo.fold.next/direct_ccs/accumulator_phi_dec_parent/v1";

/// Hash the b-ary-recomposed running-accumulator commitment data into a
/// 4-lane Poseidon2 digest, given the parent commitment wires directly.
///
/// Preimage layout (must stay byte-identical to native
/// `accumulator_digest_from_claims`):
/// ```text
///   preimage = pack(ACCUMULATOR_TAG) ‖ k ‖ (if k>0: parent_len ‖ parent.c_data)
///   acc_digest = poseidon2_hash(preimage)
/// ```
pub fn enforce_accumulator_digest_from_parent_circuit(
    builder: &mut R1csBuilder,
    k_children: usize,
    parent_c_data: &[Var],
) -> [Var; DIGEST_LEN] {
    let mut preimage = alloc_const_tag(builder, ACCUMULATOR_TAG);
    preimage.push(alloc_constant(builder, F::from_u64(k_children as u64)));
    if k_children > 0 {
        preimage.push(alloc_constant(builder, F::from_u64(parent_c_data.len() as u64)));
        preimage.extend_from_slice(parent_c_data);
    }
    enforce_poseidon2_hash(builder, &preimage)
}

/// Malformed-children error.
#[derive(Debug, thiserror::Error)]
pub enum AccumulatorDigestError {
    #[error("child commitment {idx} has length {got}, expected {expected}")]
    ChildLengthMismatch {
        idx: usize,
        got: usize,
        expected: usize,
    },
}

/// Compute the accumulator digest from children commitment wires
/// directly. For each lane `j` builds `parent_c[j] = Σ_i b^i ·
/// children[i].c[j]` via linear constraints, then hashes via
/// [`enforce_accumulator_digest_from_parent_circuit`]. Mirrors the
/// native chain
/// `parent = Π_DEC.combine_b_pows(children) → accumulator_digest_from_claims`.
pub fn enforce_accumulator_digest_from_children_circuit(
    builder: &mut R1csBuilder,
    b_norm: u32,
    children_c_data: &[Vec<Var>],
) -> Result<[Var; DIGEST_LEN], AccumulatorDigestError> {
    if children_c_data.is_empty() {
        return Ok(enforce_accumulator_digest_from_parent_circuit(builder, 0, &[]));
    }
    let parent_len = children_c_data[0].len();
    for (idx, child) in children_c_data.iter().enumerate() {
        if child.len() != parent_len {
            return Err(AccumulatorDigestError::ChildLengthMismatch {
                idx,
                got: child.len(),
                expected: parent_len,
            });
        }
    }

    let b_f = F::from_u64(b_norm as u64);
    let mut b_pows = Vec::with_capacity(children_c_data.len());
    let mut pow = F::ONE;
    for _ in 0..children_c_data.len() {
        b_pows.push(pow);
        pow = pow * b_f;
    }

    let mut parent_c = Vec::with_capacity(parent_len);
    for j in 0..parent_len {
        let mut sum_lc = Lc::zero();
        for (i, child) in children_c_data.iter().enumerate() {
            sum_lc.add_term(child[j], b_pows[i]);
        }
        let val = builder.eval(&sum_lc);
        let var = builder.alloc(val);
        builder.enforce_eq(&Lc::from_var(var), &sum_lc);
        parent_c.push(var);
    }

    Ok(enforce_accumulator_digest_from_parent_circuit(
        builder,
        children_c_data.len(),
        &parent_c,
    ))
}

// ── Internal helpers (kept private; the public surface is the two
//    `enforce_*` functions above) ─────────────────────────────────────────

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
