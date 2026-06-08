//! In-circuit mirrors of `paper::digest` accumulator handles.
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
//! **Soundness invariant**: Construction-2 state must bind HyperNova's
//! running instance `U_i`, not only a commitment projection. The
//! authority-bearing running-accumulator helper below mirrors
//! `paper::digest::accumulator_digest_from_running_parts`.

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::builder::{Lc, R1csBuilder, Var};
use crate::engine::r1cs_circuit::poseidon2::{enforce_poseidon2_hash, DIGEST_LEN};

pub const FULL_RUNNING_ACCUMULATOR_TAG: &[u8] = b"neo.fold.clean/accumulator/full_running/v1";

/// Hash the running accumulator from precomputed authority CE-claim digests.
///
/// Mirrors native `accumulator_digest_from_running_parts`.
pub fn enforce_accumulator_digest_from_running_circuit(
    builder: &mut R1csBuilder,
    child_digests: &[[Var; DIGEST_LEN]],
    parent_digest: Option<[Var; DIGEST_LEN]>,
) -> [Var; DIGEST_LEN] {
    let mut preimage = alloc_const_tag(builder, FULL_RUNNING_ACCUMULATOR_TAG);
    preimage.push(alloc_constant(builder, F::from_u64(child_digests.len() as u64)));
    for digest in child_digests {
        preimage.extend_from_slice(digest);
    }
    match parent_digest {
        Some(digest) => {
            preimage.push(alloc_constant(builder, F::ONE));
            preimage.extend_from_slice(&digest);
        }
        None => preimage.push(alloc_constant(builder, F::ZERO)),
    }
    let malformed = child_digests.is_empty() != parent_digest.is_none();
    if malformed {
        preimage.push(alloc_constant(builder, F::from_u64(u64::MAX)));
    }
    enforce_poseidon2_hash(builder, &preimage)
}

// ── Internal helpers ─────────────────────────────────────────────────────

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
