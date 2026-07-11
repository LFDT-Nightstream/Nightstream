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
//! **Soundness invariant**: the caller must first verify that the running
//! children are a strict Pi_DEC reduction of `parent_authority`. Native and
//! in-circuit NIFS.V both do so before consuming this handle. Under that
//! precondition the parent CE digest is the authority for the weak-reduction
//! class; hashing every child again is duplicate transcript work.

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::builder::{Lc, R1csBuilder, Var};
use crate::engine::r1cs_circuit::poseidon2::{enforce_poseidon2_hash, DIGEST_LEN};

pub const PARENT_AUTHORITY_ACCUMULATOR_TAG: &[u8] = b"neo.fold.clean/accumulator/parent_authority/v2";

/// Hash a strict Pi_DEC running accumulator from its parent CE digest.
///
/// Mirrors native `accumulator_digest_from_running_parts`. A non-empty
/// accumulator is represented by the CE parent that the caller has already
/// checked against all `k_rho` children. Empty and malformed shapes have
/// distinct encodings.
pub fn enforce_accumulator_digest_from_parent_circuit(
    builder: &mut R1csBuilder,
    child_count: usize,
    parent_digest: Option<[Var; DIGEST_LEN]>,
) -> [Var; DIGEST_LEN] {
    let mut preimage = alloc_const_tag(builder, PARENT_AUTHORITY_ACCUMULATOR_TAG);
    preimage.push(alloc_constant(builder, F::from_u64(child_count as u64)));
    match parent_digest {
        Some(digest) => {
            preimage.push(alloc_constant(builder, F::ONE));
            preimage.extend_from_slice(&digest);
        }
        None => preimage.push(alloc_constant(builder, F::ZERO)),
    }
    let malformed = (child_count == 0) != parent_digest.is_none();
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
