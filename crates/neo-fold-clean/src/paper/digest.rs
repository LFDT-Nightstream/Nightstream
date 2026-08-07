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

use crate::paper::reductions::accumulator_sis_circuit::{
    accumulator_digest as sis_accumulator_digest, ACCUMULATOR_CE_CLAIM_SIS_CONFIG, CCS_CLAIM_SIS_CONFIG,
    CE_CLAIM_SIS_CONFIG, NEBULA_LEAF_SIS_CONFIG, PI_CCS_OUTPUTS_SIS_CONFIG, PROTOCOL_BINDING_MAX_FIELDS,
};
use crate::paper::reductions::pi_ccs_output_message::{OUTPUTS_DOMAIN, OUTPUT_MESSAGE_DOMAIN};
use neo_ajtai::{AjtaiError, AjtaiSModule, Commitment};
use neo_ccs::{CcsClaim, CcsStructure, CeClaim, LaneCommitments};
use neo_math::{F, K};
use neo_params::NeoParams;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField64};

pub(crate) const F_PRIME_CHUNK_CLAIM_DIGEST_TAG: &[u8] = b"neo.fold.clean/f_prime_chunk_claim_digest/v2";
pub(crate) const F_PRIME_CHUNK_PUBLIC_DIGEST_TAG: &[u8] = b"neo.fold.clean/f_prime_chunk_public_digest/v1";
pub(crate) const ACCUMULATOR_FAMILY_DIGEST_TAG: &[u8] = b"neo.fold.clean/accumulator/strict_binary_family/v5";
pub(crate) const ACCUMULATOR_FAMILY_AGGREGATE_TAG: &[u8] =
    b"neo.fold.clean/accumulator/strict_binary_family/aggregate/v1";

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

/// Return the first noncanonical 8-byte Goldilocks digest limb, if any.
///
/// Protocol byte digests that are interpreted as four in-circuit
/// Goldilocks lanes must be canonical. Otherwise `p` aliases to zero
/// through `F::from_u64`, letting two byte strings describe the same
/// field statement.
pub fn noncanonical_digest32_lane(digest: [u8; 32]) -> Option<usize> {
    for (lane, chunk) in digest.chunks_exact(8).enumerate() {
        let value = u64::from_le_bytes(chunk.try_into().expect("8-byte digest limb"));
        if value >= F::ORDER_U64 {
            return Some(lane);
        }
    }
    None
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

/// Compact domain ID for the legacy F' boundary-update helper.
///
/// Canonical F' no longer emits this trace, but parity tests and older
/// helper paths still mirror the native digest.
pub const F_PRIME_BOUNDARY_UPDATE_DOMAIN: u64 = 0x4e46_0001;

/// Compact domain ID for the hot F' `state_x_out` hash.
pub const F_PRIME_STATE_X_OUT_DOMAIN: u64 = 0x4e46_0002;

// ── CCS structure digests ──────────────────────────────────────────────────

/// 4-limb digest of the CCS structure's matrices. Forwarded from the engine
/// so paper-layer code has one entry point.
pub fn mat_digest(structure: &CcsStructure<F>) -> [F; 4] {
    let raw = neo_reductions::engines::utils::digest_ccs_matrices(structure);
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
    let matrix_digest = mat_digest(structure);
    structure_digest_from_mat_digest(structure, &matrix_digest)
}

/// Digest of the full validated SuperNeo parameter set.
///
/// This is the parameter component of the future terminal-CE proof public
/// statement. It is binding material only: a verifier must still prove/check
/// the CE relation against these parameters.
pub fn params_digest(params: &NeoParams) -> [F; 4] {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/params_digest/v1");
    preimage.extend(u64_halves(params.q));
    preimage.push(F::from_u64(params.eta as u64));
    preimage.push(F::from_u64(params.d as u64));
    preimage.push(F::from_u64(params.kappa as u64));
    preimage.extend(u64_halves(params.m));
    preimage.push(F::from_u64(params.b as u64));
    preimage.push(F::from_u64(params.k_rho as u64));
    preimage.extend(u64_halves(params.B));
    preimage.push(F::from_u64(params.T as u64));
    preimage.push(F::from_u64(params.s as u64));
    preimage.push(F::from_u64(params.lambda as u64));
    poseidon_digest_fields(&preimage)
}

/// Poseidon2 identity of the verifier-owned Ajtai public parameters.
///
/// Seeded parameters bind the canonical setup descriptor without
/// materializing the potentially very large matrix. Explicit parameters bind
/// the matrix itself through bounded-size row chunks. The two encodings are
/// deliberately domain-separated: changing either the setup seed or an
/// explicit matrix coefficient changes the verifier language.
pub fn ajtai_public_parameters_digest(log: &AjtaiSModule) -> Result<[F; 4], AjtaiError> {
    let (d, m) = log.dims();
    let kappa = log.kappa();
    if let Some((seeded_kappa, seed)) = log.seeded_params() {
        if seeded_kappa != kappa {
            return Err(AjtaiError::InvalidInput(format!(
                "Ajtai seeded PP kappa mismatch while deriving verifier identity (module κ={kappa}, seed κ={seeded_kappa})"
            )));
        }
        let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/ajtai_pp/seeded/v1");
        preimage.extend(u64_halves(d as u64));
        preimage.extend(u64_halves(kappa as u64));
        preimage.extend(u64_halves(m as u64));
        preimage.extend(pack_bytes_as_fields(b"neo-ajtai/chacha8/setup_par/v1"));
        preimage.extend(pack_bytes_as_fields(&seed));
        return Ok(poseidon_digest_fields(&preimage));
    }

    let pp = log.verification_pp()?;
    if pp.m_rows.len() != pp.kappa {
        return Err(AjtaiError::InvalidInput(format!(
            "Ajtai explicit PP row count {} does not match κ={}",
            pp.m_rows.len(),
            pp.kappa
        )));
    }
    if let Some((row, got)) = pp
        .m_rows
        .iter()
        .enumerate()
        .find_map(|(row, values)| (values.len() != pp.m).then_some((row, values.len())))
    {
        return Err(AjtaiError::InvalidInput(format!(
            "Ajtai explicit PP row {row} has {got} columns, expected {}",
            pp.m
        )));
    }
    let mut matrix_preimage = pack_bytes_as_fields(b"neo.fold.clean/ajtai_pp/matrix/v1");
    matrix_preimage.extend(u64_halves(pp.d as u64));
    matrix_preimage.extend(u64_halves(pp.kappa as u64));
    matrix_preimage.extend(u64_halves(pp.m as u64));
    matrix_preimage.extend(u64_halves(pp.m_rows.len() as u64));
    for (row_index, row) in pp.m_rows.iter().enumerate() {
        let mut row_preimage = pack_bytes_as_fields(b"neo.fold.clean/ajtai_pp/matrix_row/v1");
        row_preimage.extend(u64_halves(row_index as u64));
        row_preimage.extend(u64_halves(row.len() as u64));
        for (chunk_index, chunk) in row.chunks(256).enumerate() {
            let mut chunk_preimage = pack_bytes_as_fields(b"neo.fold.clean/ajtai_pp/matrix_chunk/v1");
            chunk_preimage.extend(u64_halves(row_index as u64));
            chunk_preimage.extend(u64_halves(chunk_index as u64));
            chunk_preimage.extend(u64_halves(chunk.len() as u64));
            for ring in chunk {
                chunk_preimage.extend_from_slice(&ring.0);
            }
            row_preimage.extend_from_slice(&poseidon_digest_fields(&chunk_preimage));
        }
        matrix_preimage.extend_from_slice(&poseidon_digest_fields(&row_preimage));
    }
    Ok(poseidon_digest_fields(&matrix_preimage))
}

/// Digest of the terminal-CE relation contract a compact proof must prove.
///
/// This is not proof material and not a backend choice. It is a public
/// statement guard against replaying a proof for a weaker relation against the
/// same terminal children. The direct reference rows enforce the same
/// obligation set in `paper::decider_ce_relation`.
pub fn terminal_ce_relation_digest() -> [F; 4] {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/terminal_ce_relation/v2");
    preimage.push(F::from_u64(1)); // Ajtai opening: commit(Z) == c.
    preimage.push(F::from_u64(1)); // Public projection: X == L_in(Z).
    preimage.push(F::from_u64(1)); // Low norm: ||Z||_infinity < b.
    preimage.push(F::from_u64(1)); // Evaluation: y_ring[j] == M_j · Z(r).
    preimage.push(F::from_u64(1)); // ct is lane zero of y_ring.
    poseidon_digest_fields(&preimage)
}

/// Same digest as [`structure_digest`], using a caller-supplied matrix digest.
///
/// This is useful at preprocessing boundaries that already built the
/// optimized-engine matrix digest for Π_CCS. The supplied digest must be
/// `mat_digest(structure)`; this helper only avoids recomputing it.
pub(crate) fn structure_digest_from_mat_digest(structure: &CcsStructure<F>, matrix_digest: &[F; 4]) -> [F; 4] {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/structure_digest/v1");
    preimage.extend(u64_halves(structure.n as u64));
    preimage.extend(u64_halves(structure.m as u64));
    preimage.extend(u64_halves(structure.t() as u64));
    preimage.extend_from_slice(matrix_digest);

    preimage.extend(u64_halves(structure.f.arity() as u64));
    preimage.extend(u64_halves(structure.f.max_degree() as u64));
    preimage.extend(u64_halves(structure.f.terms().len() as u64));
    for term in structure.f.terms() {
        preimage.push(term.coeff);
        preimage.extend(u64_halves(term.exps.len() as u64));
        for exp in &term.exps {
            preimage.extend(u64_halves(*exp as u64));
        }
    }
    poseidon_digest_fields(&preimage)
}

// ── Per-claim and per-chunk digests (Soundness Invariant I-5 inputs) ──────

/// Digest of one `CcsClaim`: domain tag + commitment header + commitment
/// data + public input + m_in.
///
/// The in-circuit verifier and parity tests recompute this value from the
/// same authoritative inputs.
pub fn ccs_claim_digest(claim: &CcsClaim<Commitment, F>) -> [F; 4] {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/ccs_claim_digest/v1");
    preimage.push(F::from_u64(claim.c.d as u64));
    preimage.push(F::from_u64(claim.c.kappa as u64));
    preimage.push(F::from_u64(claim.c.data.len() as u64));
    preimage.extend_from_slice(&claim.c.data);
    preimage.push(F::from_u64(claim.x.len() as u64));
    preimage.extend_from_slice(&claim.x);
    preimage.push(F::from_u64(claim.m_in as u64));
    append_adv_leaves(&mut preimage, &claim.adv);
    sis_accumulator_digest(CCS_CLAIM_SIS_CONFIG, &preimage).expect("nonempty CCS-claim SIS preimage")
}

// ── Nebula lane-commitment leaves ─────────────────────────────────────────

/// Leaf-digest tag for the ops lane.
pub const NEBULA_LEAF_OPS_TAG: &[u8] = b"neo.fold.clean/nebula/leaf/ops/v4";
/// Leaf-digest tag shared by the `is` and `fs` lanes. Lane-NEUTRAL by
/// design: the memory-boundary chains of consecutive segments compare
/// `fs`-side digests against `is`-side digests, which is only meaningful
/// if both sides hash with an identical formula. Lane identity is positional,
/// never tag-borne.
pub const NEBULA_LEAF_MEM_TAG: &[u8] = b"neo.fold.clean/nebula/leaf/mem/v4";

/// Nonzero marker prefixing the `adv` extension of a claim-digest
/// preimage, so a `Some(adv)` preimage can never alias a `None` one.
pub(crate) const NEBULA_ADV_PRESENT_MARKER: u64 = 0x4e42_4c41; // "NBLA"

/// Bind one lane commitment with a seeded Ajtai compression followed by one
/// Poseidon2 digest. Every chain link and transcript absorb downstream
/// consumes the four-field Poseidon2 leaf, never raw SIS output.
fn nebula_leaf_digest(tag: &'static [u8], c: &Commitment) -> [F; 4] {
    let mut preimage = pack_bytes_as_fields(tag);
    preimage.push(F::from_u64(c.d as u64));
    preimage.push(F::from_u64(c.kappa as u64));
    preimage.push(F::from_u64(c.data.len() as u64));
    preimage.extend_from_slice(&c.data);
    sis_accumulator_digest(NEBULA_LEAF_SIS_CONFIG, &preimage).expect("nonempty Nebula-leaf SIS preimage")
}

/// Per-lane leaf digests of a claim's `adv` tuple, ordered (ops, is, fs).
///
/// `pub` so the F′ `NebulaLane` chains and their tests
/// recompute leaves from the same authority-bearing definition.
pub fn nebula_lane_leaf_digests(adv: &LaneCommitments<Commitment>) -> [[F; 4]; 3] {
    [
        nebula_leaf_digest(NEBULA_LEAF_OPS_TAG, &adv.ops),
        nebula_leaf_digest(NEBULA_LEAF_MEM_TAG, &adv.is),
        nebula_leaf_digest(NEBULA_LEAF_MEM_TAG, &adv.fs),
    ]
}

/// Link tag of the ops `D` chain.
pub const NEBULA_CHAIN_OPS_TAG: &[u8] = b"neo.fold.clean/nebula/chain/ops/v3";
/// Link tag shared by the `is` and `fs` chains — lane-NEUTRAL so segment
/// k's FS chain and segment k+1's IS chain are formula-identical.
pub const NEBULA_CHAIN_MEM_TAG: &[u8] = b"neo.fold.clean/nebula/chain/mem/v3";

/// Header (initial value) of the ops chain.
pub fn nebula_chain_ops_header() -> [F; 4] {
    poseidon_digest_fields(&pack_bytes_as_fields(b"neo.fold.clean/nebula/chain/ops/header/v3"))
}

/// Header shared by the `is` and `fs` chains. Shared on purpose — header
/// symmetry is half of the formula identity the boundary equality needs
/// (the other half is the shared link tag above).
pub fn nebula_chain_mem_header() -> [F; 4] {
    poseidon_digest_fields(&pack_bytes_as_fields(b"neo.fold.clean/nebula/chain/mem/header/v3"))
}

/// One `D ← Poseidon2(D_prev, tag, leaf)` chain link (the carried-lane lifecycle) —
/// the paper's `C_i ← hash(C_{i−1}, C_ω)` with the leaf hop.
pub fn nebula_chain_link(prev: &[F; 4], link_tag: &'static [u8], leaf: &[F; 4]) -> [F; 4] {
    let mut preimage = pack_bytes_as_fields(link_tag);
    preimage.extend_from_slice(prev);
    preimage.extend_from_slice(leaf);
    poseidon_digest_fields(&preimage)
}

/// Compact digest of the carried `NebulaLane` — the value the
/// F′ state hash and step transcript absorb (present-only, like the
/// claim-digest `adv` extension). Field order is part of the protocol
/// binding; `gamma: None` (`⊥` before the segment's squeeze) absorbs as a
/// zero flag with zeroed slots, `Some` as a one flag plus coefficients.
#[allow(clippy::too_many_arguments)]
pub fn nebula_lane_digest(
    seg_idx: u64,
    idx: u64,
    ts: u64,
    gamma: Option<&[K; 2]>,
    h: &[K; 4],
    sp: &[u64; 2],
    d_pre: &[[F; 4]; 3],
    d_seen: &[[F; 4]; 3],
    d_mem: &[F; 4],
) -> [F; 4] {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/nebula/lane_digest/v3");
    preimage.push(F::from_u64(seg_idx));
    preimage.push(F::from_u64(idx));
    preimage.push(F::from_u64(ts));
    preimage.extend(sp.iter().map(|&s| F::from_u64(s)));
    match gamma {
        None => {
            preimage.push(F::ZERO);
            preimage.extend(std::iter::repeat_n(F::ZERO, 4));
        }
        Some(gamma) => {
            preimage.push(F::ONE);
            append_k_slice(&mut preimage, gamma.as_slice());
        }
    }
    append_k_slice(&mut preimage, h.as_slice());
    for chain in d_pre.iter().chain(d_seen.iter()) {
        preimage.extend_from_slice(chain);
    }
    preimage.extend_from_slice(d_mem);
    poseidon_digest_fields(&preimage)
}

/// Mem-domain leaf of a single commitment — the plan generator's `D_init`
/// path uses this for initial-memory lane commitments; identical
/// to the `is`/`fs` leaves of [`nebula_lane_leaf_digests`] by shared tag.
pub fn nebula_mem_leaf(c: &Commitment) -> [F; 4] {
    nebula_leaf_digest(NEBULA_LEAF_MEM_TAG, c)
}

/// The three per-lane chains over a segment's tuple sequence — the
/// prover's `D_pre` computation and the reference for every
/// `D_seen` comparison: headers, then one link per tuple per lane, with
/// the carried-lane tag discipline (ops-domain; shared mem-domain for is/fs).
pub fn nebula_lane_chains<'a>(advs: impl IntoIterator<Item = &'a LaneCommitments<Commitment>>) -> [[F; 4]; 3] {
    let mem = nebula_chain_mem_header();
    let mut chains = [nebula_chain_ops_header(), mem, mem];
    let tags: [&'static [u8]; 3] = [NEBULA_CHAIN_OPS_TAG, NEBULA_CHAIN_MEM_TAG, NEBULA_CHAIN_MEM_TAG];
    for adv in advs {
        let leaves = nebula_lane_leaf_digests(adv);
        for lane_id in 0..3 {
            chains[lane_id] = nebula_chain_link(&chains[lane_id], tags[lane_id], &leaves[lane_id]);
        }
    }
    chains
}

/// Wherever a claim digest binds `c.data`, a
/// present `adv` tuple is bound too, as its three leaves.
///
/// Present-only on purpose: a `None` claim's preimage stays byte-identical
/// to the pre-Nebula format. The one-joint in-circuit digest mirrors append
/// the same conditional extension.
/// Unambiguous despite the sponge's zero-fill final chunk: the extension
/// is a nonzero marker plus 12 leaf elements, always > RATE.
fn append_adv_leaves(preimage: &mut Vec<F>, adv: &Option<LaneCommitments<Commitment>>) {
    if let Some(adv) = adv {
        preimage.push(F::from_u64(NEBULA_ADV_PRESENT_MARKER));
        for leaf in nebula_lane_leaf_digests(adv) {
            preimage.extend_from_slice(&leaf);
        }
    }
}

/// F'-specific digest of one `CcsClaim`. Deliberately **does not** absorb
/// `claim.x` *nor* `claim.c.data`.
///
/// Rationale: in F', a fresh CCS instance's `x` is the recursive link
/// (`x = enc_inst(prior_x_out)`, where `prior_x_out` is computed from
/// state-in, which itself depends on the previous step's `chunk_digest`).
/// In neo-fold-clean's direct-CCS interim, the Ajtai log commitment binds
/// the **full** assignment `z = [x | w]`, so `claim.c.data` also depends
/// on `x`. Folding either `x` or `c.data` into the chunk digest creates
/// a hash fixed point
///   `x_i = enc_inst(state_x_out(state_{i+1}(chunk_digest(x_i))))`
/// that no F' frontend could solve. This digest absorbs only the
/// commitment *shape* (`d`, `kappa`) and `m_in`, all of which are
/// x-independent.
///
/// Soundness rationale for dropping `c.data` here: commitments are still
/// bound to the chain through the **running accumulator authority** path
/// (`AccumulatorHandle::from_running_parts`, which binds the exact ordered
/// child claims). Each fresh claim's commitment
/// also re-enters the F'-step transcript through NIFS.V's
/// algebraic checks (sumcheck on y_ring evaluations, ρ, β_m, …), which
/// reject any inconsistent `(c, x, witness)` triple. The chunk digest's
/// remaining role here is domain separation between consecutive F' steps,
/// for which `start_index` and `(d, kappa, m_in)` suffice.
///
/// Ordinary CCS-identity callers (running-accumulator digesting, etc.)
/// continue to use [`ccs_claim_digest`], which still binds both `c.data`
/// and `x`.
pub fn f_prime_chunk_claim_digest(claim: &CcsClaim<Commitment, F>) -> [F; 4] {
    f_prime_chunk_claim_shape_digest(claim.c.d, claim.c.kappa, claim.m_in)
}

fn f_prime_chunk_claim_shape_digest(d: usize, kappa: usize, m_in: usize) -> [F; 4] {
    let mut preimage = pack_bytes_as_fields(F_PRIME_CHUNK_CLAIM_DIGEST_TAG);
    preimage.extend(u64_halves(d as u64));
    preimage.extend(u64_halves(kappa as u64));
    preimage.extend(u64_halves(m_in as u64));
    // Deliberately do NOT absorb claim.x or claim.c.data: both depend on
    // the recursive-link x in direct-CCS (commitment covers full z).
    poseidon_digest_fields(&preimage)
}

/// F'-specific chunk digest when every fresh claim has the same verifier-owned
/// `(d, kappa, m_in)` shape.
pub fn f_prime_chunk_public_digest_for_uniform_shape(
    start_index: u64,
    fresh_len: usize,
    d: usize,
    kappa: usize,
    m_in: usize,
) -> [F; 4] {
    let claim_digest = f_prime_chunk_claim_shape_digest(d, kappa, m_in);
    let mut preimage = pack_bytes_as_fields(F_PRIME_CHUNK_PUBLIC_DIGEST_TAG);
    preimage.push(F::from_u64(start_index));
    preimage.push(F::from_u64(fresh_len as u64));
    for _ in 0..fresh_len {
        preimage.extend_from_slice(&claim_digest);
    }
    poseidon_digest_fields(&preimage)
}

/// F'-specific per-chunk digest. **This is a step/shape digest, not a
/// chunk-content digest**: it binds `(start_index, fresh.len())` plus, for
/// each claim, only the commitment *shape* (`d`, `kappa`) and `m_in` —
/// not `claim.x`, not `claim.c.data`. See [`f_prime_chunk_claim_digest`]
/// for the fixed-point rationale that forces this shape.
///
/// Consequence: two chunks with the same protocol shape (same number of
/// fresh claims, same `(d, kappa, m_in)`) at the same `start_index`
/// produce the same digest even if their claims' contents differ. The
/// chain coordinates that absorb this digest (`z_i`, `public_trace`,
/// F'-step transcript prefix) therefore do **not** authenticate
/// chunk-content equality; they are domain separators per step.
///
/// **Where content binding lives**: per-step NIFS.V's algebraic checks
/// fail under any tamper to `(claim.c.data, claim.x, witness)`. After
/// finalization, `acc_digest` equals the digest of the final running CE
/// claims and is independently recomputed by the verifier from
/// `Uncompressed.public_batches` walked through the full reduction
/// stack. Use that path — not the F' chunk digest — to argue
/// "this proof commits to these specific chunk contents."
pub fn f_prime_chunk_public_digest(start_index: u64, fresh: &[CcsClaim<Commitment, F>]) -> [F; 4] {
    let mut preimage = pack_bytes_as_fields(F_PRIME_CHUNK_PUBLIC_DIGEST_TAG);
    preimage.push(F::from_u64(start_index));
    preimage.push(F::from_u64(fresh.len() as u64));
    for claim in fresh {
        preimage.extend_from_slice(&f_prime_chunk_claim_digest(claim));
    }
    poseidon_digest_fields(&preimage)
}

/// Digest of one chunk's public-instance data: domain tag + start_index +
/// fresh.len() + per-claim digests. Ordinary CCS-identity variant that
/// binds `claim.x`; used outside the F' state-advance path (e.g. for
/// running-accumulator digesting). F' state advance uses
/// [`f_prime_chunk_public_digest`] instead, which excludes `x` to avoid
/// the recursive-link fixed point.
pub fn chunk_public_digest(start_index: u64, fresh: &[CcsClaim<Commitment, F>]) -> [F; 4] {
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
///
/// `pub` (not `pub(crate)`) so the one-joint in-circuit verifier and its
/// parity tests can recompute this from authoritative inputs.
pub fn ce_claim_digest(claim: &CeClaim<Commitment, F, K>) -> [F; 4] {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/ce_claim_digest/v3");
    // Commitment
    preimage.push(F::from_u64(claim.c.d as u64));
    preimage.push(F::from_u64(claim.c.kappa as u64));
    preimage.push(F::from_u64(claim.c.data.len() as u64));
    preimage.extend_from_slice(&claim.c.data);
    // X is the exact compact coefficient embedding. Valid SuperNeo claims
    // have shape `D × (m_in / D)` and require `m_in % D == 0`. Hash the
    // stored shape and all stored entries so malformed claims remain safe to
    // inspect and cannot hide data outside the canonical width.
    let active_x_cols = crate::paper::relations::superneo_public_x_cols(claim.m_in);
    preimage.push(F::from_u64(claim.X.rows() as u64));
    preimage.push(F::from_u64(claim.X.cols() as u64));
    preimage.push(F::from_u64(active_x_cols as u64));
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
    append_adv_leaves(&mut preimage, &claim.adv);
    sis_accumulator_digest(CE_CLAIM_SIS_CONFIG, &preimage).expect("nonempty CE-claim SIS preimage")
}

/// Digest of the paper-level CE claim carried in the Construction-2
/// accumulator.
///
/// This is intentionally separate from [`ce_claim_digest`]. The latter is the
/// paper-layer Π_CCS transcript digest and historically omits fields whose
/// consistency is enforced elsewhere. The accumulator handle is different: it
/// stands in for one child statement in HyperNova's `U_i`, so it binds the CE
/// relation fields rather than only commitment coordinates.
///
/// This is the current conservative Rust CE-core encoding, not yet the
/// Lean-minimal Phi81 family payload. Replacing repeated shared/derived fields
/// requires the concrete 270-coordinate serializer refinement first.
pub fn accumulator_ce_claim_digest(claim: &CeClaim<Commitment, F, K>) -> [F; 4] {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/accumulator_ce_claim_digest/v3");
    append_ce_claim_public_fields(&mut preimage, claim);
    sis_accumulator_digest(ACCUMULATOR_CE_CLAIM_SIS_CONFIG, &preimage)
        .expect("nonempty accumulator CE-claim SIS preimage")
}

/// Ordered digest of the exact `CE(b)^k` Construction-2 accumulator.
///
/// The checked Pi_RLC parent is deliberately absent. Strict Pi_DEC
/// recomposition does not uniquely determine its child vector, so a
/// parent-only handle would alias distinct paper accumulators before hashing.
pub fn accumulator_claims_digest(claims: &[CeClaim<Commitment, F, K>]) -> [F; 4] {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/accumulator/children/v4");
    preimage.push(F::from_u64(claims.len() as u64));
    for claim in claims {
        preimage.extend_from_slice(&accumulator_ce_claim_digest(claim));
    }
    poseidon_digest_fields(&preimage)
}

/// Canonical compact binding for a nonempty strict-binary PiDEC family.
///
/// Strict PiDEC makes the child `X` matrices the unique binary split of the
/// parent `X`, makes `r` and `fold_digest` common, derives `ct` from
/// `y_ring`, and forces padded `y_ring` lanes to zero. This serializer binds
/// those fields once. It still binds every ordered child commitment, every
/// active child `y_ring` coordinate, and every Nebula commitment coordinate.
/// The result therefore binds the exact child vector under the strict PiDEC
/// checks without repeating derived data for every child.
///
/// Returns `None` when the supplied family is not in that canonical profile.
/// Callers then use the conservative exact-child codec.
pub fn strict_binary_accumulator_family_digest(
    claims: &[CeClaim<Commitment, F, K>],
    parent: &CeClaim<Commitment, F, K>,
) -> Option<[F; 4]> {
    let preimage = strict_binary_accumulator_family_preimage(claims, parent)?;
    let chunk_digests = preimage
        .chunks(PROTOCOL_BINDING_MAX_FIELDS)
        .map(|chunk| {
            sis_accumulator_digest(ACCUMULATOR_CE_CLAIM_SIS_CONFIG, chunk)
                .expect("bounded nonempty accumulator-family SIS chunk")
        })
        .collect::<Vec<_>>();

    let mut aggregate = pack_bytes_as_fields(ACCUMULATOR_FAMILY_AGGREGATE_TAG);
    aggregate.push(F::from_u64(preimage.len() as u64));
    aggregate.push(F::from_u64(chunk_digests.len() as u64));
    for digest in chunk_digests {
        aggregate.extend_from_slice(&digest);
    }
    Some(poseidon_digest_fields(&aggregate))
}

fn strict_binary_accumulator_family_preimage(
    claims: &[CeClaim<Commitment, F, K>],
    parent: &CeClaim<Commitment, F, K>,
) -> Option<Vec<F>> {
    let first = claims.first()?;
    let active_x_cols = crate::paper::relations::superneo_public_x_cols(first.m_in);
    if first.m_in % neo_math::D != 0
        || first.c.d != neo_math::D
        || first.c.kappa == 0
        || first.c.data.len() != first.c.d * first.c.kappa
        || first.X.rows() != neo_math::D
        || first.X.cols() != active_x_cols
        || first.y_ring.is_empty()
        || first.ct.len() != first.y_ring.len()
        || first
            .y_ring
            .iter()
            .any(|row| row.len() != neo_math::D.next_power_of_two())
        || !claim_has_canonical_derived_fields(first)
        || !adv_has_shape(&first.adv, first.c.d, first.c.kappa)
    {
        return None;
    }

    let expected_child_x = neo_reductions::common::split_b_matrix_k(&parent.X, claims.len(), 2).ok()?;
    if parent.m_in != first.m_in
        || parent.X.rows() != first.X.rows()
        || parent.X.cols() != first.X.cols()
        || expected_child_x.len() != claims.len()
    {
        return None;
    }

    let has_adv = first.adv.is_some();
    for (claim, expected_x) in claims.iter().zip(expected_child_x.iter()) {
        if claim.c.d != first.c.d
            || claim.c.kappa != first.c.kappa
            || claim.c.data.len() != first.c.data.len()
            || claim.X.rows() != first.X.rows()
            || claim.X.cols() != first.X.cols()
            || claim.X != *expected_x
            || claim.r != first.r
            || claim.y_ring.len() != first.y_ring.len()
            || claim
                .y_ring
                .iter()
                .zip(first.y_ring.iter())
                .any(|(row, first_row)| row.len() != first_row.len())
            || claim.ct.len() != claim.y_ring.len()
            || claim.m_in != first.m_in
            || claim.fold_digest != first.fold_digest
            || claim.adv.is_some() != has_adv
            || !claim_has_canonical_derived_fields(claim)
            || !adv_has_shape(&claim.adv, first.c.d, first.c.kappa)
        {
            return None;
        }
    }

    let mut preimage = pack_bytes_as_fields(ACCUMULATOR_FAMILY_DIGEST_TAG);
    preimage.push(F::from_u64(claims.len() as u64));
    preimage.push(F::from_u64(first.c.d as u64));
    preimage.push(F::from_u64(first.c.kappa as u64));
    preimage.push(F::from_u64(first.c.data.len() as u64));
    preimage.push(F::from_u64(first.X.rows() as u64));
    preimage.push(F::from_u64(first.X.cols() as u64));
    preimage.push(F::from_u64(active_x_cols as u64));
    for row in 0..parent.X.rows() {
        for column in 0..active_x_cols {
            preimage.push(parent.X[(row, column)]);
        }
    }
    append_k_slice(&mut preimage, &first.r);
    preimage.push(F::from_u64(first.y_ring.len() as u64));
    preimage.push(F::from_u64(neo_math::D as u64));
    preimage.push(F::from_u64(first.m_in as u64));
    preimage.extend(digest32_as_fields(first.fold_digest));
    preimage.push(if has_adv { F::ONE } else { F::ZERO });

    for (index, claim) in claims.iter().enumerate() {
        preimage.push(F::from_u64(index as u64));
        preimage.extend_from_slice(&claim.c.data);
        for row in &claim.y_ring {
            for value in row.iter().take(neo_math::D) {
                preimage.extend_from_slice(value.as_basis_coefficients_slice());
            }
        }
        if let Some(adv) = &claim.adv {
            preimage.extend_from_slice(&adv.ops.data);
            preimage.extend_from_slice(&adv.is.data);
            preimage.extend_from_slice(&adv.fs.data);
        }
    }
    Some(preimage)
}

fn claim_has_canonical_derived_fields(claim: &CeClaim<Commitment, F, K>) -> bool {
    claim
        .ct
        .iter()
        .zip(&claim.y_ring)
        .all(|(ct, row)| row.first() == Some(ct))
        && claim
            .y_ring
            .iter()
            .all(|row| row.iter().skip(neo_math::D).all(|value| *value == K::ZERO))
}

fn adv_has_shape(adv: &Option<LaneCommitments<Commitment>>, d: usize, kappa: usize) -> bool {
    let Some(adv) = adv else {
        return true;
    };
    [&adv.ops, &adv.is, &adv.fs]
        .into_iter()
        .all(|commitment| commitment.d == d && commitment.kappa == kappa && commitment.data.len() == d * kappa)
}

/// Digest the terminal NIFS children for a compact terminal-CE proof statement.
///
/// Unlike [`ce_claim_digest`] and [`accumulator_ce_claim_digest`], this absorbs
/// every public CE field carried at the terminal boundary. The digest is never
/// authority by itself; the compact proof must still prove the terminal CE
/// relation against the children it binds.
pub fn terminal_children_digest(claims: &[CeClaim<Commitment, F, K>]) -> [F; 4] {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/terminal_children_digest/v2");
    preimage.push(F::from_u64(claims.len() as u64));
    for claim in claims {
        preimage.extend_from_slice(&terminal_ce_claim_digest(claim));
    }
    poseidon_digest_fields(&preimage)
}

/// Serialize the prover-chosen Π_CCS output message before Π_RLC samples `ρ`.
///
/// SuperNeo's interactive order is "Π_CCS sends output CE claims, then Π_RLC
/// samples random linear-combination coefficients." In the Fiat-Shamir
/// transcript, the output message therefore needs an explicit, verifier-
/// recomputable absorb before `ρ` is derived. Only active `y_ring` and
/// identity-first `y_ring` lanes are committed here: commitment/X are inherited from the
/// inputs, `r` is a verifier challenge, `ct` is the `y_ring` constant
/// term, `fold_digest` is the checked header, and padded lanes/unsupported
/// sidecars are canonical. Native and recursive verifiers enforce those
/// reconstruction equations before relying on this projection.
pub fn pi_ccs_outputs_preimage(claims: &[CeClaim<Commitment, F, K>]) -> Vec<F> {
    let mut preimage = pack_bytes_as_fields(OUTPUTS_DOMAIN);
    preimage.push(F::from_u64(claims.len() as u64));
    for claim in claims {
        preimage.extend(pack_bytes_as_fields(OUTPUT_MESSAGE_DOMAIN));
        append_active_k_rows(&mut preimage, &claim.y_ring);
    }
    preimage
}

/// SIS-compress [`pi_ccs_outputs_preimage`] for the Fiat-Shamir transcript.
pub fn pi_ccs_outputs_digest(claims: &[CeClaim<Commitment, F, K>]) -> [F; 4] {
    let preimage = pi_ccs_outputs_preimage(claims);
    sis_accumulator_digest(PI_CCS_OUTPUTS_SIS_CONFIG, &preimage).expect("nonempty PiCCS-output SIS preimage")
}

/// Digest of the compact terminal-CE proof's public statement.
///
/// This is the single backend-neutral public input a future compact proof
/// verifier should bind. It is still only binding material: the proof must
/// prove the terminal CE relation behind `terminal_children_digest`.
pub fn terminal_ce_public_digest(
    relation_digest: [F; 4],
    structure_digest: [F; 4],
    params_digest: [F; 4],
    terminal_children_digest: [F; 4],
    claim_count: usize,
) -> [F; 4] {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/terminal_ce_public/v2");
    preimage.extend_from_slice(&relation_digest);
    preimage.extend_from_slice(&structure_digest);
    preimage.extend_from_slice(&params_digest);
    preimage.extend_from_slice(&terminal_children_digest);
    preimage.extend(u64_halves(claim_count as u64));
    poseidon_digest_fields(&preimage)
}

fn terminal_ce_claim_digest(claim: &CeClaim<Commitment, F, K>) -> [F; 4] {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/terminal_ce_claim_digest/v2");
    append_terminal_ce_claim_public_fields(&mut preimage, claim);
    poseidon_digest_fields(&preimage)
}

fn append_active_k_rows(preimage: &mut Vec<F>, rows: &[Vec<K>]) {
    preimage.push(F::from_u64(rows.len() as u64));
    for row in rows {
        append_active_k_slice(preimage, row);
    }
}

fn append_active_k_slice(preimage: &mut Vec<F>, values: &[K]) {
    append_k_slice(preimage, &values[..values.len().min(neo_math::ring::D)]);
}

fn append_ce_claim_public_fields(preimage: &mut Vec<F>, claim: &CeClaim<Commitment, F, K>) {
    preimage.push(F::from_u64(claim.c.d as u64));
    preimage.push(F::from_u64(claim.c.kappa as u64));
    preimage.push(F::from_u64(claim.c.data.len() as u64));
    preimage.extend_from_slice(&claim.c.data);

    let active_x_cols = crate::paper::relations::superneo_public_x_cols(claim.m_in);
    preimage.push(F::from_u64(claim.X.rows() as u64));
    preimage.push(F::from_u64(claim.X.cols() as u64));
    preimage.push(F::from_u64(active_x_cols as u64));
    for r in 0..claim.X.rows() {
        for c in 0..claim.X.cols() {
            preimage.push(claim.X[(r, c)]);
        }
    }

    append_k_slice(preimage, &claim.r);
    append_k_rows(preimage, &claim.y_ring);
    append_k_slice(preimage, &claim.ct);
    preimage.push(F::from_u64(claim.m_in as u64));
    preimage.extend(digest32_as_fields(claim.fold_digest));
    append_adv_leaves(preimage, &claim.adv);
}

fn append_k_slice(preimage: &mut Vec<F>, values: &[K]) {
    preimage.push(F::from_u64(values.len() as u64));
    for value in values {
        for limb in value.as_basis_coefficients_slice() {
            preimage.push(*limb);
        }
    }
}

fn append_k_rows(preimage: &mut Vec<F>, rows: &[Vec<K>]) {
    preimage.push(F::from_u64(rows.len() as u64));
    for row in rows {
        append_k_slice(preimage, row);
    }
}

fn append_terminal_ce_claim_public_fields(preimage: &mut Vec<F>, claim: &CeClaim<Commitment, F, K>) {
    append_ce_claim_public_fields(preimage, claim);
}

/// Public-instance digest absorbed by Π_CCS prove and verify so the two
/// sides bind the same chunk context into their transcripts.
///
/// **Soundness boundary**: this is *not* a prover-supplied value. Both
/// prover and verifier compute it independently from the public claims
/// they already hold. The verifier is given `fresh_claims` (claims-only
/// view of the K fresh CCS instances) and `running_claims` (the running
/// accumulator's CE claims) — exactly the same data the prover uses.
/// Project soundness rule: digests across trust boundaries must be
/// recomputable from authoritative inputs, never carried as authority.
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

/// Π_CCS public-instance digest under Π_RLC-parent authority.
///
/// Fresh CCS claims are still hashed individually because they are the new
/// public instances being folded in this step. The running side is bound by the
/// single Π_RLC parent whose Π_DEC children are the running CE claims. The
/// children remain the algebraic inputs to Π_CCS and are checked against the
/// parent by Π_DEC. Their exact ordered accumulator handle is absorbed
/// separately before challenges; this digest owns only the parent-cache view
/// of the public-instance header.
pub fn pi_ccs_instance_digest_parent_authority(
    fresh_claims: &[CcsClaim<Commitment, F>],
    running_count: usize,
    running_parent_authority: Option<&CeClaim<Commitment, F, K>>,
) -> [F; 4] {
    let parent_digest = running_parent_authority.map(accumulator_ce_claim_digest);
    pi_ccs_instance_digest_from_parent_digest(fresh_claims, running_count, parent_digest)
}

/// Prover-side scheduling form of [`pi_ccs_instance_digest_parent_authority`].
/// The verifier never accepts this digest as authority; deferred backends must
/// reconstruct the parent claim and check its digest before proof egress.
pub(crate) fn pi_ccs_instance_digest_from_parent_digest(
    fresh_claims: &[CcsClaim<Commitment, F>],
    running_count: usize,
    running_parent_digest: Option<[F; 4]>,
) -> [F; 4] {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/pi_ccs_instance_digest/parent_authority/v1");
    preimage.push(F::from_u64(fresh_claims.len() as u64));
    for claim in fresh_claims {
        preimage.extend_from_slice(&ccs_claim_digest(claim));
    }
    preimage.push(F::from_u64(running_count as u64));
    match (running_count, running_parent_digest) {
        (0, None) => preimage.push(F::ZERO),
        (_, Some(parent_digest)) => {
            preimage.push(F::ONE);
            preimage.extend_from_slice(&parent_digest);
        }
        (_, None) => preimage.push(F::from_u64(u64::MAX)),
    }
    poseidon_digest_fields(&preimage)
}

// ── Accumulator digest (semantic_acc_digest in x_out) ──────────────────────

/// Compact handle for the running accumulator carried in Construction-2 state.
///
/// HyperNova's recursive link hashes the running instance `U_i`. Here `U_i` is
/// the exact ordered `CE(b)^k` child vector. The checked Pi_RLC parent is a
/// recomposition cache and is not a substitute for that vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccumulatorHandle {
    digest: [u8; 32],
}

impl AccumulatorHandle {
    /// Handle for an empty running accumulator.
    pub fn empty() -> Self {
        Self::from_claims(&[])
    }

    /// Handle for the exact ordered Construction-2 accumulator.
    pub fn from_claims(claims: &[CeClaim<Commitment, F, K>]) -> Self {
        Self {
            digest: digest_fields_as_digest32(accumulator_claims_digest(claims)),
        }
    }

    /// Shape-checking adapter for the running transport envelope.
    ///
    /// Canonical strict-binary families use the compact family codec. The
    /// parent supplies only the `X` value whose unique PiDEC split is checked
    /// against every child; it is not a second accumulator authority. Other
    /// shapes retain the conservative exact-child codec. Malformed
    /// empty/parent combinations receive a distinct fail-closed digest.
    pub fn from_running_parts(
        claims: &[CeClaim<Commitment, F, K>],
        parent_authority: Option<&CeClaim<Commitment, F, K>>,
    ) -> Self {
        let shape_is_valid =
            (claims.is_empty() && parent_authority.is_none()) || (!claims.is_empty() && parent_authority.is_some());
        if claims.is_empty() && parent_authority.is_none() {
            return Self::empty();
        }
        if !claims.is_empty() {
            if let Some(parent) = parent_authority {
                if let Some(digest) = strict_binary_accumulator_family_digest(claims, parent) {
                    return Self {
                        digest: digest_fields_as_digest32(digest),
                    };
                }
            }
        }
        if shape_is_valid {
            return Self::from_claims(claims);
        }
        Self {
            digest: malformed_accumulator_digest(claims, parent_authority),
        }
    }

    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    pub fn digest_fields(&self) -> [F; 4] {
        digest32_as_fields(self.digest)
    }
}

/// Compatibility wrapper for [`AccumulatorHandle::from_running_parts`].
pub fn accumulator_digest_from_running_parts(
    claims: &[CeClaim<Commitment, F, K>],
    parent_authority: Option<&CeClaim<Commitment, F, K>>,
) -> [u8; 32] {
    AccumulatorHandle::from_running_parts(claims, parent_authority).digest()
}

fn malformed_accumulator_digest(
    claims: &[CeClaim<Commitment, F, K>],
    parent_authority: Option<&CeClaim<Commitment, F, K>>,
) -> [u8; 32] {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/accumulator/malformed/v3");
    preimage.push(F::from_u64(claims.len() as u64));
    preimage.extend_from_slice(&accumulator_claims_digest(claims));
    match parent_authority {
        Some(parent) => {
            preimage.push(F::ONE);
            preimage.extend_from_slice(&accumulator_ce_claim_digest(parent));
        }
        None => preimage.push(F::ZERO),
    }
    digest_fields_as_digest32(poseidon_digest_fields(&preimage))
}

/// Canonical "no app state" seed for stateless chains. This is the
/// fixed digest of the empty Construction-2 running accumulator.
///
/// Used as the `initial_semantic_state_digest` for stateless
/// [`crate::lifecycle::Preprocessing`] so `vk_fs_digest` absorbs a
/// deterministic seed instead of being chain-state-aware. Stateful
/// frontends set their own seed via
/// [`crate::lifecycle::Preprocessing::with_initial_semantic_state_digest`].
pub fn empty_semantic_state_digest() -> [u8; 32] {
    AccumulatorHandle::empty().digest()
}

// ── Boundary + public-trace chains ────────────────────────────────────────

/// Initial `z_0`. Pure function of the full structure digest and
/// `public_input_len`.
pub fn initial_boundary_digest(structure_digest: &[F; 4], public_input_len: Option<usize>) -> [u8; 32] {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/initial_boundary/v2");
    preimage.extend(structure_digest.iter().copied());
    extend_optional_usize(&mut preimage, public_input_len);
    digest_fields_as_digest32(poseidon_digest_fields(&preimage))
}

/// Legacy helper: `z_{i+1} = H(prev_z_i || chunk_public_digest)`.
///
/// Canonical F' now carries `new_z_i = chunk_digest` directly and enforces
/// that equality in the F' structure, avoiding this bit-backed trace.
pub fn boundary_update_digest(prev: [u8; 32], chunk_digest: [F; 4]) -> [u8; 32] {
    let mut preimage = Vec::with_capacity(1 + 4 + 4);
    preimage.push(F::from_u64(F_PRIME_BOUNDARY_UPDATE_DOMAIN));
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
pub fn public_trace_update_digest(prev: [u8; 32], chunk_digest: [F; 4]) -> [u8; 32] {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/public_trace_update/v1");
    preimage.extend(digest32_as_fields(prev));
    preimage.extend(chunk_digest);
    digest_fields_as_digest32(poseidon_digest_fields(&preimage))
}

// ── vk_fs and x_out ────────────────────────────────────────────────────────

/// `vk_fs_digest` — Definition 14 + full CCS structure + canonical Pi_CCS
/// header bundle + Ajtai public-parameter identity + program-fixed
/// `public_input_len` + **chain initial semantic-state digest**.
///
/// The header bundle binds the matrix-dependent one-joint transcript identity;
/// the circuit recomputes this same digest from witness key fields and feeds
/// the same header wires to Pi_CCS.V. The Ajtai digest binds the exact
/// commitment map used by that relation. The remaining preimage absorbs the
/// full 11-field `NeoParams` view, the tagged optional `public_input_len`, and
/// `initial_semantic_state_digest` — the chain's
/// claimed starting application state.
///
/// Absorbing the initial app-state digest into `vk_fs` (rather than
/// adding a separate `state_x_out` slot) gives every step's chain
/// digest a transitive binding to the initial state without a layout
/// change in the F' image. The verifier-owned [`Preprocessing`] holds
/// the value at preprocess time (set by the frontend that knows what
/// the chain should start from); a malicious prover cannot relabel it
/// after-the-fact because every `vk_fs_digest`-bearing absorb
/// (including `state_x_out`) would diverge from the verifier's
/// preprocessing-pinned digest.
pub fn vk_fs_digest(
    params: &NeoParams,
    structure_digest: &[F; 4],
    pi_ccs_header_bundle: &[F; 4],
    ajtai_pp_digest: &[F; 4],
    public_input_len: Option<usize>,
    initial_semantic_state_digest: [u8; 32],
) -> [u8; 32] {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/vk_fs/v4");
    preimage.extend(structure_digest.iter().copied());
    preimage.extend(pi_ccs_header_bundle.iter().copied());
    preimage.extend(ajtai_pp_digest.iter().copied());
    preimage.extend(u64_halves(params.q));
    preimage.push(F::from_u64(params.eta as u64));
    preimage.push(F::from_u64(params.d as u64));
    preimage.push(F::from_u64(params.kappa as u64));
    preimage.extend(u64_halves(params.m));
    preimage.push(F::from_u64(params.b as u64));
    preimage.push(F::from_u64(params.k_rho as u64));
    preimage.extend(u64_halves(params.B));
    preimage.push(F::from_u64(params.T as u64));
    preimage.push(F::from_u64(params.s as u64));
    preimage.push(F::from_u64(params.lambda as u64));
    extend_optional_usize(&mut preimage, public_input_len);
    preimage.extend(digest32_as_fields(initial_semantic_state_digest));
    digest_fields_as_digest32(poseidon_digest_fields(&preimage))
}

/// Bind verifier policy that changes the accepted language to the base
/// Construction-2 verifier-key digest.
///
/// The structure digest alone cannot distinguish a generic CCS verifier from
/// an F' verifier that applies recursive-link or terminal-induction checks
/// outside the CCS relation. These bits are verifier authority and therefore
/// must be part of the key identity.
pub fn vk_fs_policy_digest(
    base_digest: [u8; 32],
    stateful: bool,
    f_prime_recursive_link: bool,
    terminal_induction: bool,
) -> [u8; 32] {
    let mut preimage = pack_bytes_as_fields(b"neo.fold.clean/vk_fs_policy/v1");
    preimage.extend(digest32_as_fields(base_digest));
    preimage.push(F::from_bool(stateful));
    preimage.push(F::from_bool(f_prime_recursive_link));
    preimage.push(F::from_bool(terminal_induction));
    digest_fields_as_digest32(poseidon_digest_fields(&preimage))
}

/// `x_out` — the Construction-2 hash-chain output.
///
/// **Soundness Invariant I-5**: this absorb sequence and the in-circuit
/// gadget that recomputes it must move in lockstep.
///
/// The initial app-state seed (`initial_semantic_state_digest`) is bound
/// into the chain transitively through `vk_fs_digest`: the verifier-owned
/// [`vk_fs_digest`] absorbs it as part of preprocessing, so every step's
/// `x_out` inherits the binding without needing a separate slot here.
///
/// This compact preimage is the canonical local F' link for stateful chains.
/// Fields not absorbed directly here must be verifier-owned or pinned
/// elsewhere:
///
/// - `structure_digest` is absorbed transitively by `vk_fs_digest`.
/// - `initial_boundary` (`z_0`) is also absorbed transitively: it is the
///   verifier-derived `initial_boundary_digest(structure_digest,
///   public_input_len)`, and both inputs are already in `vk_fs_digest`.
/// - `pc` is absorbed directly, matching HyperNova Construction 2's
///   `hash(vk_fs, i, z_0, z_i, U_i, pc_i)` recursive link. This build
///   still pins it to the single-program `TRIVIAL_PC`, but the binding
///   remains explicit so future multi-program variants do not inherit a
///   commitment-only selector.
/// - `public_trace` is constrained to match the boundary chain separately.
#[allow(clippy::too_many_arguments)]
pub fn state_x_out_digest(
    vk_fs_digest: [u8; 32],
    pi_ccs_header_bundle: [F; 4],
    _structure_digest: &[F; 4],
    chunk_count: u64,
    step_count: u64,
    _initial_boundary: [u8; 32],
    current_boundary: [u8; 32],
    pc: u64,
    semantic_acc: [u8; 32],
    construction2_acc: [u8; 32],
    _public_trace: [u8; 32],
) -> [u8; 32] {
    state_x_out_digest_with_mode(
        StateXOutDigestMode::Stateful,
        vk_fs_digest,
        pi_ccs_header_bundle,
        _structure_digest,
        chunk_count,
        step_count,
        _initial_boundary,
        current_boundary,
        pc,
        semantic_acc,
        construction2_acc,
        _public_trace,
        None,
    )
}

/// Which semantic-state coordinate the `state_x_out` hash absorbs.
///
/// Stateful chains carry an independently authenticated application-state
/// digest, so `state_x_out` absorbs both `semantic_acc` and
/// `construction2_acc`. Stateless chains require those two coordinates to be
/// equal; the F' circuit and native verifier enforce that equality, so the
/// stateless digest omits the duplicate semantic lanes and saves one
/// Poseidon2 permutation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateXOutDigestMode {
    Stateless,
    Stateful,
}

/// Typed source schedule for the exact `state_x_out` Poseidon2 preimage.
///
/// Production evaluation and the Rust-to-Lean drift gate consume the same
/// value. Digest and header instructions each contribute four field lanes;
/// counter instructions contribute low/high 32-bit limbs.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateXOutPreimageInstruction {
    Domain { value: u64 },
    VerifierDigest,
    PiCcsHeader,
    ChunkCountHalves,
    StepCountHalves,
    PcHalves,
    CurrentBoundary,
    SemanticState,
    Construction2Accumulator,
    NebulaPresentMarker { value: u64 },
    NebulaDigest,
}

impl StateXOutPreimageInstruction {
    #[doc(hidden)]
    pub const fn field_count(self) -> usize {
        match self {
            Self::Domain { .. } | Self::NebulaPresentMarker { .. } => 1,
            Self::ChunkCountHalves | Self::StepCountHalves | Self::PcHalves => 2,
            Self::VerifierDigest
            | Self::PiCcsHeader
            | Self::CurrentBoundary
            | Self::SemanticState
            | Self::Construction2Accumulator
            | Self::NebulaDigest => 4,
        }
    }
}

/// Exact ordered source program interpreted by
/// [`state_x_out_preimage_with_mode`].
#[doc(hidden)]
pub fn state_x_out_preimage_program(
    mode: StateXOutDigestMode,
    nebula_present: bool,
) -> Vec<StateXOutPreimageInstruction> {
    use StateXOutPreimageInstruction as Instruction;

    let mut program = vec![
        Instruction::Domain {
            value: F_PRIME_STATE_X_OUT_DOMAIN,
        },
        Instruction::VerifierDigest,
        Instruction::PiCcsHeader,
        Instruction::ChunkCountHalves,
        Instruction::StepCountHalves,
        Instruction::PcHalves,
        Instruction::CurrentBoundary,
    ];
    if matches!(mode, StateXOutDigestMode::Stateful) {
        program.push(Instruction::SemanticState);
    }
    program.push(Instruction::Construction2Accumulator);
    if nebula_present {
        program.extend([
            Instruction::NebulaPresentMarker {
                value: NEBULA_ADV_PRESENT_MARKER,
            },
            Instruction::NebulaDigest,
        ]);
    }
    program
}

#[allow(clippy::too_many_arguments)]
pub fn state_x_out_digest_with_mode(
    mode: StateXOutDigestMode,
    vk_fs_digest: [u8; 32],
    pi_ccs_header_bundle: [F; 4],
    _structure_digest: &[F; 4],
    chunk_count: u64,
    step_count: u64,
    _initial_boundary: [u8; 32],
    current_boundary: [u8; 32],
    pc: u64,
    semantic_acc: [u8; 32],
    construction2_acc: [u8; 32],
    _public_trace: [u8; 32],
    nebula_lane: Option<[F; 4]>,
) -> [u8; 32] {
    let preimage = state_x_out_preimage_with_mode(
        mode,
        vk_fs_digest,
        pi_ccs_header_bundle,
        chunk_count,
        step_count,
        current_boundary,
        pc,
        semantic_acc,
        construction2_acc,
        nebula_lane,
    );
    state_x_out_digest_from_preimage(&preimage)
}

/// Exact Poseidon2 field preimage used by [`state_x_out_digest_with_mode`].
///
/// This is crate-visible so the native verifier's execution receipt can
/// record the actual preimage passed to the hash, without reconstructing it
/// after verification.
#[allow(clippy::too_many_arguments)]
pub(crate) fn state_x_out_preimage_with_mode(
    mode: StateXOutDigestMode,
    vk_fs_digest: [u8; 32],
    pi_ccs_header_bundle: [F; 4],
    chunk_count: u64,
    step_count: u64,
    current_boundary: [u8; 32],
    pc: u64,
    semantic_acc: [u8; 32],
    construction2_acc: [u8; 32],
    nebula_lane: Option<[F; 4]>,
) -> Vec<F> {
    use StateXOutPreimageInstruction as Instruction;

    let program = state_x_out_preimage_program(mode, nebula_lane.is_some());
    let capacity = program.iter().copied().map(Instruction::field_count).sum();
    let mut preimage = Vec::with_capacity(capacity);
    for instruction in program {
        match instruction {
            Instruction::Domain { value } | Instruction::NebulaPresentMarker { value } => {
                preimage.push(F::from_u64(value));
            }
            Instruction::VerifierDigest => preimage.extend(digest32_as_fields(vk_fs_digest)),
            Instruction::PiCcsHeader => preimage.extend(pi_ccs_header_bundle),
            Instruction::ChunkCountHalves => preimage.extend(u64_halves(chunk_count)),
            Instruction::StepCountHalves => preimage.extend(u64_halves(step_count)),
            Instruction::PcHalves => preimage.extend(u64_halves(pc)),
            Instruction::CurrentBoundary => preimage.extend(digest32_as_fields(current_boundary)),
            Instruction::SemanticState => preimage.extend(digest32_as_fields(semantic_acc)),
            Instruction::Construction2Accumulator => {
                preimage.extend(digest32_as_fields(construction2_acc));
            }
            Instruction::NebulaDigest => {
                if let Some(lane) = nebula_lane {
                    preimage.extend_from_slice(&lane);
                }
            }
        }
    }
    preimage
}

/// Hash an already constructed canonical `state_x_out` preimage.
pub(crate) fn state_x_out_digest_from_preimage(preimage: &[F]) -> [u8; 32] {
    digest_fields_as_digest32(poseidon_digest_fields(preimage))
}

#[inline]
fn u64_halves(value: u64) -> [F; 2] {
    [F::from_u64(value & 0xffff_ffff), F::from_u64(value >> 32)]
}

fn extend_optional_usize(out: &mut Vec<F>, value: Option<usize>) {
    match value {
        None => out.extend([F::ZERO, F::ZERO]),
        Some(value) => {
            let [low, high] = u64_halves(value as u64);
            out.extend([low + F::ONE, high]);
        }
    }
}
