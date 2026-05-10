//! Paper-layer digest helpers for the Construction-2 hash chain.
//!
//! Owns: every Poseidon2 absorb that is part of a Soundness Invariant. This is
//! the central place where domain tags, absorb orders, and field/byte
//! conversions live; nothing else in the paper layer should call Poseidon2
//! directly except through helpers here.
//!
//! Each absorb is part of a Soundness Invariant: the order of fields, the
//! domain tag, and the field/byte conversions are all part of the protocol
//! binding. Any change here must move in lockstep with the in-circuit
//! gadget that recomputes the same digest in PR5's `engine::decider`.

use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CcsStructure, CeClaim};
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField64};

// ── Field/byte plumbing ───────────────────────────────────────────────────

/// 32-byte digest → 4 Goldilocks field limbs (little-endian).
pub fn digest32_as_fields(digest: [u8; 32]) -> [F; 4] {
    [
        F::from_u64(u64::from_le_bytes(digest[0..8].try_into().expect("limb 0"))),
        F::from_u64(u64::from_le_bytes(digest[8..16].try_into().expect("limb 1"))),
        F::from_u64(u64::from_le_bytes(digest[16..24].try_into().expect("limb 2"))),
        F::from_u64(u64::from_le_bytes(digest[24..32].try_into().expect("limb 3"))),
    ]
}

/// 4 Goldilocks limbs → 32 bytes (inverse of `digest32_as_fields`).
pub fn digest_fields_as_digest32(fields: [F; 4]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, field) in fields.into_iter().enumerate() {
        out[i * 8..(i + 1) * 8].copy_from_slice(&field.as_canonical_u64().to_le_bytes());
    }
    out
}

/// Pack a `&[u8]` domain tag into Goldilocks fields. The first field carries
/// the byte length; subsequent fields carry 7 bytes each (so we never overflow
/// the 64-bit modulus).
pub(crate) fn pack_bytes_as_fields(bytes: &[u8]) -> Vec<F> {
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
fn poseidon_digest_fields(input: &[F]) -> [F; 4] {
    neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash(input)
}

// ── CCS structure digests ──────────────────────────────────────────────────

/// 4-limb digest of the CCS structure's matrices. Forwarded from the engine
/// so paper-layer code has one entry point.
pub fn mat_digest(structure: &CcsStructure<F>) -> [F; 4] {
    let raw = neo_reductions::engines::utils::digest_ccs_matrices_with_sparse_cache(structure, None);
    [raw[0], raw[1], raw[2], raw[3]]
}

/// 4-limb digest of the full CCS structure `s = ({M_j}, f)`.
///
/// SuperNeo Definition 11 makes the polynomial `f` part of the public
/// structure, not an implementation detail. `mat_digest` remains available
/// for engine seams that only accept matrix digests, but protocol binding
/// paths use this digest so changing `f` changes `vk_fs`, `z_0`, public trace
/// seed, and `x_out`.
pub fn structure_digest(structure: &CcsStructure<F>) -> [F; 4] {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/structure_digest/v1");
    preimage.push(F::from_u64(structure.n as u64));
    preimage.push(F::from_u64(structure.m as u64));
    preimage.push(F::from_u64(structure.t() as u64));
    preimage.extend_from_slice(&mat_digest(structure));

    preimage.push(F::from_u64(structure.f.arity() as u64));
    preimage.push(F::from_u64(structure.f.max_degree() as u64));
    preimage.push(F::from_u64(structure.f.terms().len() as u64));
    for term in structure.f.terms() {
        preimage.push(term.coeff);
        preimage.push(F::from_u64(term.exps.len() as u64));
        for exp in &term.exps {
            preimage.push(F::from_u64(*exp as u64));
        }
    }
    poseidon_digest_fields(&preimage)
}

// ── Per-claim and per-chunk digests (Soundness Invariant I-5 inputs) ──────

/// Digest of one `CcsClaim`: domain tag + commitment header + commitment
/// data + public input + m_in.
pub(crate) fn ccs_claim_digest(claim: &CcsClaim<Commitment, F>) -> [F; 4] {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/ccs_claim_digest/v1");
    preimage.push(F::from_u64(claim.c.d as u64));
    preimage.push(F::from_u64(claim.c.kappa as u64));
    preimage.push(F::from_u64(claim.c.data.len() as u64));
    preimage.extend_from_slice(&claim.c.data);
    preimage.push(F::from_u64(claim.x.len() as u64));
    preimage.extend_from_slice(&claim.x);
    preimage.push(F::from_u64(claim.m_in as u64));
    poseidon_digest_fields(&preimage)
}

/// Digest of one chunk's public-instance data: domain tag + start_index +
/// fresh.len() + per-claim digests. This is the value that gets chained into
/// `z_i` and `public_trace_digest`.
pub(crate) fn chunk_public_digest(start_index: u64, fresh: &[CcsClaim<Commitment, F>]) -> [F; 4] {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/chunk_public_digest/v1");
    preimage.push(F::from_u64(start_index));
    preimage.push(F::from_u64(fresh.len() as u64));
    for claim in fresh {
        preimage.extend_from_slice(&ccs_claim_digest(claim));
    }
    poseidon_digest_fields(&preimage)
}

/// Digest of one CE claim's public fields: commitment, X (public input
/// matrix shape + values), evaluation point r, y_ring evaluations, m_in,
/// fold_digest. Mirrors `ccs_claim_digest` for the running-side claims.
pub(crate) fn ce_claim_digest(claim: &CeClaim<Commitment, F, K>) -> [F; 4] {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/ce_claim_digest/v1");
    // Commitment
    preimage.push(F::from_u64(claim.c.d as u64));
    preimage.push(F::from_u64(claim.c.kappa as u64));
    preimage.push(F::from_u64(claim.c.data.len() as u64));
    preimage.extend_from_slice(&claim.c.data);
    // X public-input matrix: hash shape + entries.
    preimage.push(F::from_u64(claim.X.rows() as u64));
    preimage.push(F::from_u64(claim.X.cols() as u64));
    for r in 0..claim.X.rows() {
        for c in 0..claim.X.cols() {
            preimage.push(claim.X[(r, c)]);
        }
    }
    // r evaluation point (extension-field elements).
    preimage.push(F::from_u64(claim.r.len() as u64));
    for r in &claim.r {
        // K elements split into base-field limbs via the public conversion that
        // the engine itself uses.
        for limb in r.as_basis_coefficients_slice() {
            preimage.push(*limb);
        }
    }
    // y_ring evaluations: shape + flattened.
    preimage.push(F::from_u64(claim.y_ring.len() as u64));
    for row in &claim.y_ring {
        preimage.push(F::from_u64(row.len() as u64));
        for v in row {
            for limb in v.as_basis_coefficients_slice() {
                preimage.push(*limb);
            }
        }
    }
    preimage.push(F::from_u64(claim.m_in as u64));
    preimage.extend(digest32_as_fields(claim.fold_digest));
    poseidon_digest_fields(&preimage)
}

/// Public-instance digest absorbed by Π_CCS prove and verify so the two
/// sides bind the same chunk context into their transcripts.
///
/// **Soundness boundary**: this is *not* a prover-supplied value. Both
/// prover and verifier compute it independently from the public claims
/// they already hold. The verifier is given `fresh_claims` (claims-only
/// view of the K fresh CCS instances) and `running_claims` (the running
/// accumulator's CE claims) — exactly the same data the prover uses.
/// Per CLAUDE.md: digests across trust boundaries must be recomputable
/// from authoritative inputs, never carried as authority.
pub fn pi_ccs_instance_digest(
    fresh_claims: &[CcsClaim<Commitment, F>],
    running_claims: &[CeClaim<Commitment, F, K>],
) -> [F; 4] {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/pi_ccs_instance_digest/v1");
    preimage.push(F::from_u64(fresh_claims.len() as u64));
    for claim in fresh_claims {
        preimage.extend_from_slice(&ccs_claim_digest(claim));
    }
    preimage.push(F::from_u64(running_claims.len() as u64));
    for claim in running_claims {
        preimage.extend_from_slice(&ce_claim_digest(claim));
    }
    poseidon_digest_fields(&preimage)
}

// ── Accumulator digest (semantic_acc_digest in x_out) ──────────────────────

/// `Σ b^i · c_i` over the running accumulator's commitment data, hashed.
///
/// **Domain tag** is the legacy `neo.fold.next/...` string — kept stable so
/// existing absorbed values remain reproducible. The auditor reads this as
/// "the accumulator commitment hash" and compares the absorb against the
/// in-circuit gadget; the tag string is opaque from the soundness side.
pub fn accumulator_digest_from_claims(base: u32, claims: &[CeClaim<Commitment, F, K>]) -> [u8; 32] {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.next/direct_ccs/accumulator_phi_dec_parent/v1");
    preimage.push(F::from_u64(claims.len() as u64));
    if let Some(first) = claims.first() {
        let parent_len = first.c.data.len();
        preimage.push(F::from_u64(parent_len as u64));
        let base = F::from_u64(base as u64);
        let mut powers = Vec::with_capacity(claims.len());
        let mut pow = F::ONE;
        for claim in claims {
            if claim.c.data.len() != parent_len {
                preimage.push(F::from_u64(u64::MAX));
                return digest_fields_as_digest32(poseidon_digest_fields(&preimage));
            }
            powers.push(pow);
            pow *= base;
        }
        for lane_idx in 0..parent_len {
            let mut value = F::ZERO;
            for (claim, pow) in claims.iter().zip(powers.iter().copied()) {
                value += claim.c.data[lane_idx] * pow;
            }
            preimage.push(value);
        }
    }
    digest_fields_as_digest32(poseidon_digest_fields(&preimage))
}

// ── Boundary + public-trace chains ────────────────────────────────────────

/// Initial `z_0`. Pure function of the full structure digest and
/// `public_input_len`.
pub fn initial_boundary_digest(structure_digest: &[F; 4], public_input_len: Option<usize>) -> [u8; 32] {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/initial_boundary/v1");
    preimage.extend(structure_digest.iter().copied());
    preimage.push(F::from_u64(public_input_len.map_or(u64::MAX, |n| n as u64)));
    digest_fields_as_digest32(poseidon_digest_fields(&preimage))
}

/// `z_{i+1} = H(prev_z_i || chunk_public_digest)`.
pub(crate) fn boundary_update_digest(prev: [u8; 32], chunk_digest: [F; 4]) -> [u8; 32] {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/boundary_update/v1");
    preimage.extend(digest32_as_fields(prev));
    preimage.extend(chunk_digest);
    digest_fields_as_digest32(poseidon_digest_fields(&preimage))
}

/// Initial `public_trace_digest`. Pure function of the full structure digest.
pub fn public_trace_seed_digest(structure_digest: &[F; 4]) -> [u8; 32] {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/public_trace_seed/v1");
    preimage.extend(structure_digest.iter().copied());
    digest_fields_as_digest32(poseidon_digest_fields(&preimage))
}

/// `public_trace_{i+1} = H(prev_public_trace || chunk_public_digest)`.
pub(crate) fn public_trace_update_digest(prev: [u8; 32], chunk_digest: [F; 4]) -> [u8; 32] {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/public_trace_update/v1");
    preimage.extend(digest32_as_fields(prev));
    preimage.extend(chunk_digest);
    digest_fields_as_digest32(poseidon_digest_fields(&preimage))
}

// ── vk_fs and x_out ────────────────────────────────────────────────────────

/// `vk_fs_digest` — Definition 14 + full CCS structure + program-fixed
/// `public_input_len`.
///
/// Absorbs the full 11-field `NeoParams` view plus the optional
/// `public_input_len` (encoded as `u64::MAX` when absent).
pub fn vk_fs_digest(params: &NeoParams, structure_digest: &[F; 4], public_input_len: Option<usize>) -> [u8; 32] {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/vk_fs/v1");
    preimage.extend(structure_digest.iter().copied());
    preimage.extend([
        F::from_u64(params.q),
        F::from_u64(params.eta as u64),
        F::from_u64(params.d as u64),
        F::from_u64(params.kappa as u64),
        F::from_u64(params.m),
        F::from_u64(params.b as u64),
        F::from_u64(params.k_rho as u64),
        F::from_u64(params.B),
        F::from_u64(params.T as u64),
        F::from_u64(params.s as u64),
        F::from_u64(params.lambda as u64),
        F::from_u64(public_input_len.map_or(u64::MAX, |n| n as u64)),
    ]);
    digest_fields_as_digest32(poseidon_digest_fields(&preimage))
}

/// `x_out` — the Construction-2 hash-chain output (11-input absorb).
///
/// **Soundness Invariant I-5**: this absorb sequence and the in-circuit
/// gadget that recomputes it must move in lockstep.
#[allow(clippy::too_many_arguments)]
pub(crate) fn state_x_out_digest(
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
) -> [u8; 32] {
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
    digest_fields_as_digest32(poseidon_digest_fields(&preimage))
}

#[inline]
fn u64_halves(value: u64) -> [F; 2] {
    [F::from_u64(value & 0xffff_ffff), F::from_u64(value >> 32)]
}

// ── Light Poseidon2 absorb wrapper for legacy `Transcript`-based digests ──

/// Poseidon2 absorb of a labelled byte payload, returning a 32-byte digest.
pub fn label_digest(label: &'static [u8], payload: &[&[u8]]) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(label);
    for slice in payload {
        tr.append_message(label, slice);
    }
    tr.digest32()
}
