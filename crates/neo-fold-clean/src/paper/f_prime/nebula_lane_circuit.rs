//! In-circuit mirrors of the Nebula lane transition — spec §13 step 9's
//! circuit content, ahead of the `enc(F′)` regime decision.
//!
//! Owns: the R1CS twins of the lane's Poseidon2 material
//! ([`crate::paper::digest::nebula_lane_leaf_digests`],
//! [`crate::paper::digest::nebula_chain_link`],
//! [`crate::paper::digest::nebula_lane_digest`]) and of the §6.3
//! transition itself — [`enforce_nebula_advance_circuit`] (the per-claim
//! equalities and `D_seen` chain updates) and
//! [`enforce_nebula_close_circuit`] (the segment-close equalities,
//! including the product equation as two in-circuit K-mults, and the
//! reset that never touches `ts`).
//!
//! Does not own: the segment-open γ squeeze (that composes the sponge
//! transcript circuit and lands with the F′ transcript wiring), lane
//! *commitment* semantics (`relations/lanes.rs`), or any native
//! authority — today the lifecycle enforces §6.3 natively
//! (`construction2::nebula_lane`); these gadgets are the rows that
//! obligation transfers onto when F′ instances become foldable.
//!
//! **Soundness Invariant I-5** (same as `digest_circuit`): every mirror
//! here moves in lockstep with its native twin, enforced byte-for-byte
//! by the parity tests in `tests/f_prime/nebula_lane_circuit.rs`.

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::builder::{Lc, R1csBuilder, Var};
use crate::engine::r1cs_circuit::field_ext::{enforce_k_mul, KLc, KVar};
use crate::engine::r1cs_circuit::poseidon2::{enforce_poseidon2_hash, DIGEST_LEN};
use crate::paper::digest::{
    nebula_chain_mem_header, nebula_chain_ops_header, NEBULA_CHAIN_MEM_TAG, NEBULA_CHAIN_OPS_TAG, NEBULA_LEAF_MEM_TAG,
    NEBULA_LEAF_OPS_TAG,
};
use crate::paper::f_prime::digest_circuit::{alloc_const_tag, alloc_constant};

/// Mirror of the native lane-digest domain tag (private in
/// `paper::digest`; the parity test is the lockstep guard).
const NEBULA_LANE_DIGEST_TAG: &[u8] = b"neo.fold.clean/nebula/lane_digest/v3";

/// Wire view of the carried [`crate::paper::construction2::NebulaLane`]
/// (spec §6.1). `gamma` is meaningful only while a segment is open —
/// a closed lane's digest uses [`GammaWires::Absent`].
#[derive(Clone)]
pub struct NebulaLaneWires {
    pub seg_idx: Var,
    pub idx: Var,
    pub ts: Var,
    /// `(γ1, γ2)` of the open segment.
    pub gamma: [KVar; 2],
    /// Running `(h_rs, h_ws, h_is, h_fs)`, order per
    /// [`crate::paper::construction2::nebula_lane::H_RS`].
    pub h: [KVar; 4],
    /// Running stack pointers (v3.1).
    pub sp: [Var; 2],
    pub d_pre: [[Var; DIGEST_LEN]; 3],
    pub d_seen: [[Var; DIGEST_LEN]; 3],
    pub d_mem: [Var; DIGEST_LEN],
}

/// Wire view of a deposited claim's decoded step input
/// ([`crate::paper::construction2::NebulaStepX`], spec §4.4).
#[derive(Clone)]
pub struct NebulaStepXWires {
    pub seg_idx: Var,
    pub idx: Var,
    pub ts_in: Var,
    pub ts_out: Var,
    pub gamma: [KVar; 2],
    pub h_in: [KVar; 4],
    pub h_out: [KVar; 4],
    pub sp_in: [Var; 2],
    pub sp_out: [Var; 2],
}

/// γ slot of the lane digest preimage: the native `Option<[K; 2]>` is a
/// *shape* choice, so it is fixed at gadget-emit time, not by a witness
/// flag (an R1CS preimage cannot change length under a wire).
pub enum GammaWires {
    /// `⊥` — closed lane. Absorbs the native zero flag + four zero slots.
    Absent,
    /// Open segment. Absorbs the native one flag + the K-slice encoding.
    Present([KVar; 2]),
}

// ── Poseidon2 mirrors ─────────────────────────────────────────────────────

/// Mirror of the private native `nebula_leaf_digest(tag, c)` — one lane
/// commitment crosses Poseidon2 exactly once (spec §6.1, L0a). The
/// commitment's shape (`d`, `kappa`, data length) is structural and
/// enters as constants; the data enters as wires.
pub fn enforce_nebula_leaf_digest_circuit(
    builder: &mut R1csBuilder,
    tag: &'static [u8],
    d: usize,
    kappa: usize,
    c_data: &[Var],
) -> [Var; DIGEST_LEN] {
    let mut preimage = alloc_const_tag(builder, tag);
    preimage.push(alloc_constant(builder, F::from_u64(d as u64)));
    preimage.push(alloc_constant(builder, F::from_u64(kappa as u64)));
    preimage.push(alloc_constant(builder, F::from_u64(c_data.len() as u64)));
    preimage.extend_from_slice(c_data);
    enforce_poseidon2_hash(builder, &preimage)
}

/// Mirror of [`crate::paper::digest::nebula_lane_leaf_digests`]: the
/// (ops, is, fs) leaves with the §6.1 tag discipline — ops-domain tag
/// for ops, the shared lane-NEUTRAL mem-domain tag for is and fs.
pub fn enforce_nebula_lane_leaf_digests_circuit(
    builder: &mut R1csBuilder,
    d: usize,
    kappa: usize,
    ops: &[Var],
    is: &[Var],
    fs: &[Var],
) -> [[Var; DIGEST_LEN]; 3] {
    [
        enforce_nebula_leaf_digest_circuit(builder, NEBULA_LEAF_OPS_TAG, d, kappa, ops),
        enforce_nebula_leaf_digest_circuit(builder, NEBULA_LEAF_MEM_TAG, d, kappa, is),
        enforce_nebula_leaf_digest_circuit(builder, NEBULA_LEAF_MEM_TAG, d, kappa, fs),
    ]
}

/// Mirror of [`crate::paper::digest::nebula_chain_link`]:
/// `D ← Poseidon2(tag, D_prev, leaf)`.
pub fn enforce_nebula_chain_link_circuit(
    builder: &mut R1csBuilder,
    prev: [Var; DIGEST_LEN],
    link_tag: &'static [u8],
    leaf: [Var; DIGEST_LEN],
) -> [Var; DIGEST_LEN] {
    let mut preimage = alloc_const_tag(builder, link_tag);
    preimage.extend_from_slice(&prev);
    preimage.extend_from_slice(&leaf);
    enforce_poseidon2_hash(builder, &preimage)
}

/// Mirror of [`crate::paper::digest::nebula_lane_digest`] — the compact
/// lane handle the F′ state hash absorbs. Field order is protocol
/// binding; the parity test pins it against the native function for
/// both γ shapes.
pub fn enforce_nebula_lane_digest_circuit(
    builder: &mut R1csBuilder,
    lane: &NebulaLaneWires,
    gamma: GammaWires,
) -> [Var; DIGEST_LEN] {
    let mut preimage = alloc_const_tag(builder, NEBULA_LANE_DIGEST_TAG);
    preimage.push(lane.seg_idx);
    preimage.push(lane.idx);
    preimage.push(lane.ts);
    preimage.extend_from_slice(&lane.sp);
    match gamma {
        GammaWires::Absent => {
            // Native: zero flag + four zeroed slots.
            for _ in 0..5 {
                preimage.push(alloc_constant(builder, F::ZERO));
            }
        }
        GammaWires::Present(gamma) => {
            // Native: one flag + `append_k_slice` (length, then coeffs).
            preimage.push(alloc_constant(builder, F::ONE));
            preimage.push(alloc_constant(builder, F::from_u64(2)));
            for g in gamma {
                preimage.push(g.c0);
                preimage.push(g.c1);
            }
        }
    }
    // `append_k_slice(h)`: length, then coefficient pairs.
    preimage.push(alloc_constant(builder, F::from_u64(4)));
    for h in &lane.h {
        preimage.push(h.c0);
        preimage.push(h.c1);
    }
    for chain in lane.d_pre.iter().chain(lane.d_seen.iter()) {
        preimage.extend_from_slice(chain);
    }
    preimage.extend_from_slice(&lane.d_mem);
    enforce_poseidon2_hash(builder, &preimage)
}

// ── The §6.3 transition ───────────────────────────────────────────────────

/// One `advance_nebula` (spec §6.3) as rows, **excluding** the close —
/// mirror of the open-segment body of
/// [`crate::paper::construction2::NebulaLane::advance`]:
///
/// - the six per-claim equalities (`seg_idx`, `idx`, `ts_in`, γ, `h_in`,
///   `sp_in`) against the lane;
/// - the three `D_seen` chain links over the supplied leaves (§6.1 tag
///   discipline);
/// - the carried-state update (`h ← h_out`, `sp ← sp_out`,
///   `ts ← ts_out`, `idx ← idx + 1`).
///
/// The caller composes [`enforce_nebula_close_circuit`] when this claim
/// is the segment's `N`-th (`N` is a plan constant, so which step closes
/// is emit-time knowledge, exactly like the base/recursive split).
pub fn enforce_nebula_advance_circuit(
    builder: &mut R1csBuilder,
    lane: &NebulaLaneWires,
    x: &NebulaStepXWires,
    leaves: [[Var; DIGEST_LEN]; 3],
) -> NebulaLaneWires {
    enforce_var_eq(builder, x.seg_idx, lane.seg_idx);
    enforce_var_eq(builder, x.idx, lane.idx);
    enforce_var_eq(builder, x.ts_in, lane.ts);
    for (a, b) in x.gamma.iter().zip(lane.gamma.iter()) {
        enforce_k_eq(builder, a, b);
    }
    for (a, b) in x.h_in.iter().zip(lane.h.iter()) {
        enforce_k_eq(builder, a, b);
    }
    for (a, b) in x.sp_in.iter().zip(lane.sp.iter()) {
        enforce_var_eq(builder, *a, *b);
    }

    let link_tags: [&'static [u8]; 3] = [NEBULA_CHAIN_OPS_TAG, NEBULA_CHAIN_MEM_TAG, NEBULA_CHAIN_MEM_TAG];
    let mut d_seen = lane.d_seen;
    for lane_id in 0..3 {
        d_seen[lane_id] =
            enforce_nebula_chain_link_circuit(builder, d_seen[lane_id], link_tags[lane_id], leaves[lane_id]);
    }

    let idx_out = {
        let mut next = Lc::from_var(lane.idx);
        next.add_constant(F::ONE);
        let v = builder.alloc(builder.eval(&next));
        builder.enforce_eq(&Lc::from_var(v), &next);
        v
    };

    NebulaLaneWires {
        seg_idx: lane.seg_idx,
        idx: idx_out,
        ts: x.ts_out,
        gamma: x.gamma.clone(),
        h: x.h_out.clone(),
        sp: x.sp_out,
        d_pre: lane.d_pre,
        d_seen,
        d_mem: lane.d_mem,
    }
}

/// Segment close (spec §6.3) as rows — mirror of the native `close`:
///
/// - `sp = 0` (v3.1 segment-local stacks, the deterministic companion
///   to the product equation);
/// - `D_seen == D_pre` per lane (the L0b retroactive authority);
/// - the Nebula product equation `h_is · h_ws == h_rs · h_fs` — two
///   in-circuit K-mults plus one K-equality;
/// - the boundary handoff `D_seen[is] == D_mem`, then
///   `D_mem ← D_seen[fs]`;
/// - the reset: `seg_idx + 1`, `idx = 0`, `h = 1_K`, chains at their
///   headers — and `ts` is NOT reset.
///
/// The returned lane is the closed state; its digest uses
/// [`GammaWires::Absent`] (γ back to `⊥`).
pub fn enforce_nebula_close_circuit(builder: &mut R1csBuilder, lane: &NebulaLaneWires) -> NebulaLaneWires {
    for sp in lane.sp {
        builder.enforce_zero(&Lc::from_var(sp));
    }
    for (seen, pre) in lane.d_seen.iter().zip(lane.d_pre.iter()) {
        for (a, b) in seen.iter().zip(pre.iter()) {
            enforce_var_eq(builder, *a, *b);
        }
    }
    // Product equation, order per `nebula_lane::H_RS`: h[2]·h[1] == h[0]·h[3].
    let lhs = enforce_k_mul(builder, &KLc::from_var(lane.h[2]), &KLc::from_var(lane.h[1]));
    let rhs = enforce_k_mul(builder, &KLc::from_var(lane.h[0]), &KLc::from_var(lane.h[3]));
    enforce_k_eq(builder, &lhs, &rhs);
    for (a, b) in lane.d_seen[1].iter().zip(lane.d_mem.iter()) {
        enforce_var_eq(builder, *a, *b);
    }

    let seg_idx_out = {
        let mut next = Lc::from_var(lane.seg_idx);
        next.add_constant(F::ONE);
        let v = builder.alloc(builder.eval(&next));
        builder.enforce_eq(&Lc::from_var(v), &next);
        v
    };
    let zero = alloc_constant(builder, F::ZERO);
    let one_k = KVar::new(alloc_constant(builder, F::ONE), zero);
    let ops_header = alloc_const_digest(builder, nebula_chain_ops_header());
    let mem_header = alloc_const_digest(builder, nebula_chain_mem_header());

    NebulaLaneWires {
        seg_idx: seg_idx_out,
        idx: zero,
        ts: lane.ts,
        gamma: [one_k, one_k], // dead wires post-close; digest uses Absent
        h: [one_k; 4],
        sp: [zero, zero],
        d_pre: [ops_header, mem_header, mem_header],
        d_seen: [ops_header, mem_header, mem_header],
        d_mem: lane.d_seen[2],
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────

fn enforce_var_eq(builder: &mut R1csBuilder, a: Var, b: Var) {
    builder.enforce_eq(&Lc::from_var(a), &Lc::from_var(b));
}

fn enforce_k_eq(builder: &mut R1csBuilder, a: &KVar, b: &KVar) {
    enforce_var_eq(builder, a.c0, b.c0);
    enforce_var_eq(builder, a.c1, b.c1);
}

fn alloc_const_digest(builder: &mut R1csBuilder, digest: [F; DIGEST_LEN]) -> [Var; DIGEST_LEN] {
    digest.map(|value| alloc_constant(builder, value))
}
