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

use neo_math::field::KExtensions;
use neo_math::{F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::engine::r1cs_circuit::boolean::enforce_bit;
use crate::engine::r1cs_circuit::builder::{Lc, R1csBuilder, Var};
use crate::engine::r1cs_circuit::field_ext::{enforce_k_mul, KLc, KVar};
use crate::engine::r1cs_circuit::mux::enforce_mux_var;
use crate::engine::r1cs_circuit::poseidon2::{enforce_poseidon2_hash, DIGEST_LEN};
use crate::engine::r1cs_circuit::transcript::TranscriptGadget;
use crate::paper::construction2::nebula_lane::{NebulaLane, NEBULA_GAMMA_TRANSCRIPT_LABEL};
use crate::paper::construction2::nebula_lane::{K_LIMB_BITS, MAX_STACKS, SEG_IDX_BITS, STEP_IDX_BITS, TS_BITS};
use crate::paper::construction2::StackShape;
use crate::paper::digest::{
    nebula_chain_mem_header, nebula_chain_ops_header, NEBULA_CHAIN_MEM_TAG, NEBULA_CHAIN_OPS_TAG, NEBULA_LEAF_MEM_TAG,
    NEBULA_LEAF_OPS_TAG,
};
use crate::paper::f_prime::digest_circuit::{alloc_const_tag, alloc_constant};
use crate::paper::reductions::accumulator_sis_circuit::{
    enforce_accumulator_digest as enforce_sis_accumulator_digest, NEBULA_LEAF_SIS_CONFIG,
};
use crate::paper::relations::product_commitment_circuit::{validate_adv_shape, AdvCommitmentWires};

/// Mirror of the native lane-digest domain tag (private in
/// `paper::digest`; the parity test is the lockstep guard).
const NEBULA_LANE_DIGEST_TAG: &[u8] = b"neo.fold.clean/nebula/lane_digest/v3";

/// Wire view of the carried [`crate::paper::construction2::NebulaLane`]
/// (spec §6.1). `gamma` is meaningful only while a segment is open —
/// a closed lane's digest uses [`GammaWires::Absent`].
#[derive(Clone, Copy)]
pub struct NebulaLaneWires {
    /// `1` exactly while a segment is open (`gamma = Some`).
    pub open: Var,
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

/// Allocate one native carried lane as witness wires.
///
/// A closed lane has no native gamma value. Its two gamma wire slots use the
/// canonical dead value `1_K`; every authoritative consumer selects the
/// absent-gamma encoding when `open = 0`.
pub fn alloc_nebula_lane_wires(builder: &mut R1csBuilder, lane: &NebulaLane) -> NebulaLaneWires {
    let alloc_k = |builder: &mut R1csBuilder, value: K| {
        let (c0, c1) = value.to_limbs_u64();
        KVar::alloc(builder, F::from_u64(c0), F::from_u64(c1))
    };
    let alloc_digest = |builder: &mut R1csBuilder, digest: [F; DIGEST_LEN]| digest.map(|value| builder.alloc(value));
    let gamma = lane.gamma.unwrap_or([K::ONE; 2]);
    NebulaLaneWires {
        open: builder.alloc(if lane.gamma.is_some() { F::ONE } else { F::ZERO }),
        seg_idx: builder.alloc(F::from_u64(lane.seg_idx)),
        idx: builder.alloc(F::from_u64(lane.idx)),
        ts: builder.alloc(F::from_u64(lane.ts)),
        gamma: gamma.map(|value| alloc_k(builder, value)),
        h: lane.h.map(|value| alloc_k(builder, value)),
        sp: lane.sp.map(|value| builder.alloc(F::from_u64(value))),
        d_pre: lane.d_pre.map(|digest| alloc_digest(builder, digest)),
        d_seen: lane.d_seen.map(|digest| alloc_digest(builder, digest)),
        d_mem: alloc_digest(builder, lane.d_mem),
    }
}

/// Enforce equality of two carried lane bundles, including closed-state dead
/// gamma slots. This is the state-link relation used between adjacent F' steps.
pub fn enforce_nebula_lane_equality_circuit(builder: &mut R1csBuilder, lhs: &NebulaLaneWires, rhs: &NebulaLaneWires) {
    enforce_var_eq(builder, lhs.open, rhs.open);
    enforce_var_eq(builder, lhs.seg_idx, rhs.seg_idx);
    enforce_var_eq(builder, lhs.idx, rhs.idx);
    enforce_var_eq(builder, lhs.ts, rhs.ts);
    for (lhs, rhs) in lhs.gamma.iter().zip(rhs.gamma.iter()) {
        enforce_k_eq(builder, lhs, rhs);
    }
    for (lhs, rhs) in lhs.h.iter().zip(rhs.h.iter()) {
        enforce_k_eq(builder, lhs, rhs);
    }
    for (lhs, rhs) in lhs.sp.iter().zip(rhs.sp.iter()) {
        enforce_var_eq(builder, *lhs, *rhs);
    }
    for (lhs, rhs) in lhs.d_pre.iter().flatten().zip(rhs.d_pre.iter().flatten()) {
        enforce_var_eq(builder, *lhs, *rhs);
    }
    for (lhs, rhs) in lhs.d_seen.iter().flatten().zip(rhs.d_seen.iter().flatten()) {
        enforce_var_eq(builder, *lhs, *rhs);
    }
    for (lhs, rhs) in lhs.d_mem.iter().zip(rhs.d_mem.iter()) {
        enforce_var_eq(builder, *lhs, *rhs);
    }
}

/// Pin a carried lane bundle to one verifier-known native value.
pub fn enforce_nebula_lane_constant_circuit(builder: &mut R1csBuilder, wires: &NebulaLaneWires, expected: &NebulaLane) {
    let gamma = expected.gamma.unwrap_or([K::ONE; 2]);
    builder.enforce_eq(
        &Lc::from_var(wires.open),
        &Lc::from_const(if expected.gamma.is_some() { F::ONE } else { F::ZERO }),
    );
    builder.enforce_eq(
        &Lc::from_var(wires.seg_idx),
        &Lc::from_const(F::from_u64(expected.seg_idx)),
    );
    builder.enforce_eq(&Lc::from_var(wires.idx), &Lc::from_const(F::from_u64(expected.idx)));
    builder.enforce_eq(&Lc::from_var(wires.ts), &Lc::from_const(F::from_u64(expected.ts)));
    for (wire, value) in wires.gamma.iter().zip(gamma) {
        enforce_k_constant(builder, wire, value);
    }
    for (wire, value) in wires.h.iter().zip(expected.h) {
        enforce_k_constant(builder, wire, value);
    }
    for (wire, value) in wires.sp.iter().zip(expected.sp) {
        builder.enforce_eq(&Lc::from_var(*wire), &Lc::from_const(F::from_u64(value)));
    }
    for (wire, value) in wires
        .d_pre
        .iter()
        .flatten()
        .zip(expected.d_pre.iter().flatten())
    {
        builder.enforce_eq(&Lc::from_var(*wire), &Lc::from_const(*value));
    }
    for (wire, value) in wires
        .d_seen
        .iter()
        .flatten()
        .zip(expected.d_seen.iter().flatten())
    {
        builder.enforce_eq(&Lc::from_var(*wire), &Lc::from_const(*value));
    }
    for (wire, value) in wires.d_mem.iter().zip(expected.d_mem) {
        builder.enforce_eq(&Lc::from_var(*wire), &Lc::from_const(value));
    }
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

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum NebulaStepXDecodeError {
    #[error("Nebula step-x circuit expected {expected} bits, got {got}")]
    Length { expected: usize, got: usize },
}

pub const NEBULA_D_PRE_BITS: usize = 3 * DIGEST_LEN * K_LIMB_BITS;

pub const fn delayed_nebula_public_suffix_len(stacks: StackShape) -> usize {
    stacks.x_bits() + 1 + NEBULA_D_PRE_BITS
}

/// Previous fresh claim data needed by the delayed Nebula transition.
pub struct DelayedNebulaInputWires {
    pub step: NebulaStepXWires,
    pub open: Var,
    pub d_pre: [[Var; DIGEST_LEN]; 3],
}

pub struct NebulaOpenContextWires {
    pub vk_fs: [Var; DIGEST_LEN],
    pub z_i: [Var; DIGEST_LEN],
    pub acc_digest: [Var; DIGEST_LEN],
    pub plan_digest: [Var; DIGEST_LEN],
}

/// Decode the canonical `NebulaStepX` bit suffix into typed field wires.
///
/// Every input coordinate is constrained Boolean here. Multi-bit fields are
/// recomposed little-endian in the base field, matching the native decode;
/// 64-bit limb aliases therefore have the same modulo-q interpretation on
/// both sides.
pub fn decode_nebula_step_x_bits_circuit(
    builder: &mut R1csBuilder,
    bits: &[Var],
    stacks: StackShape,
) -> Result<NebulaStepXWires, NebulaStepXDecodeError> {
    let expected = stacks.x_bits();
    if bits.len() != expected {
        return Err(NebulaStepXDecodeError::Length {
            expected,
            got: bits.len(),
        });
    }
    for bit in bits {
        enforce_bit(builder, *bit);
    }

    let mut at = 0usize;
    let seg_idx = read_word(builder, bits, &mut at, SEG_IDX_BITS);
    let idx = read_word(builder, bits, &mut at, STEP_IDX_BITS);
    let ts_in = read_word(builder, bits, &mut at, TS_BITS);
    let ts_out = read_word(builder, bits, &mut at, TS_BITS);
    let gamma = [read_k(builder, bits, &mut at), read_k(builder, bits, &mut at)];
    let h_in = read_k4(builder, bits, &mut at);
    let h_out = read_k4(builder, bits, &mut at);

    let zero = alloc_constant(builder, F::ZERO);
    let mut sp_in = [zero; MAX_STACKS];
    let mut sp_out = [zero; MAX_STACKS];
    for stack in 0..stacks.count {
        sp_in[stack] = read_word(builder, bits, &mut at, stacks.sigma);
        sp_out[stack] = read_word(builder, bits, &mut at, stacks.sigma);
    }
    debug_assert_eq!(at, bits.len());

    Ok(NebulaStepXWires {
        seg_idx,
        idx,
        ts_in,
        ts_out,
        gamma,
        h_in,
        h_out,
        sp_in,
        sp_out,
    })
}

/// Decode `[step_x_bits || open || bits(D_pre)]` from one consumed fresh
/// claim. `D_pre` is canonical zero when `open = 0`, preventing unused
/// metadata from creating alternate public encodings.
pub fn decode_delayed_nebula_public_suffix_circuit(
    builder: &mut R1csBuilder,
    suffix: &[Var],
    stacks: StackShape,
) -> Result<DelayedNebulaInputWires, NebulaStepXDecodeError> {
    let expected = delayed_nebula_public_suffix_len(stacks);
    if suffix.len() != expected {
        return Err(NebulaStepXDecodeError::Length {
            expected,
            got: suffix.len(),
        });
    }

    let step_bits = stacks.x_bits();
    let step = decode_nebula_step_x_bits_circuit(builder, &suffix[..step_bits], stacks)?;
    let open = suffix[step_bits];
    enforce_bit(builder, open);
    let d_pre_bits = &suffix[step_bits + 1..];
    for bit in d_pre_bits {
        enforce_bit(builder, *bit);
    }

    let mut not_open = Lc::from_const(F::ONE);
    not_open.add_term(open, -F::ONE);
    for bit in d_pre_bits {
        builder.enforce(&not_open, &Lc::from_var(*bit), &Lc::zero());
    }

    let mut at = step_bits + 1;
    let mut d_pre = [[Var::ONE; DIGEST_LEN]; 3];
    for digest in &mut d_pre {
        for lane in digest {
            *lane = read_word(builder, suffix, &mut at, K_LIMB_BITS);
        }
    }
    debug_assert_eq!(at, suffix.len());
    Ok(DelayedNebulaInputWires { step, open, d_pre })
}

fn read_word(builder: &mut R1csBuilder, bits: &[Var], at: &mut usize, width: usize) -> Var {
    debug_assert!(width <= 64);
    let mut value = Lc::zero();
    for (power, bit) in bits[*at..*at + width].iter().enumerate() {
        value.add_term(*bit, F::from_u64(1u64 << power));
    }
    *at += width;
    let out = builder.alloc(builder.eval(&value));
    builder.enforce_eq(&Lc::from_var(out), &value);
    out
}

fn read_k(builder: &mut R1csBuilder, bits: &[Var], at: &mut usize) -> KVar {
    KVar::new(
        read_word(builder, bits, at, K_LIMB_BITS),
        read_word(builder, bits, at, K_LIMB_BITS),
    )
}

fn read_k4(builder: &mut R1csBuilder, bits: &[Var], at: &mut usize) -> [KVar; 4] {
    [
        read_k(builder, bits, at),
        read_k(builder, bits, at),
        read_k(builder, bits, at),
        read_k(builder, bits, at),
    ]
}

// ── Poseidon2 mirrors ─────────────────────────────────────────────────────

/// Mirror of the private native `nebula_leaf_digest(tag, c)`: a seeded
/// Ajtai compression over the exact tagged commitment preimage, followed by
/// one Poseidon2 digest. SIS output never enters Fiat-Shamir directly.
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
    enforce_sis_accumulator_digest(builder, NEBULA_LEAF_SIS_CONFIG, &preimage)
        .expect("fixed nonempty Nebula-leaf SIS preimage")
        .digest
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

/// Fixed-shape lane digest for a witness-time open/closed state.
///
/// Native encoding has different Poseidon preimage lengths for `gamma =
/// None` and `Some`. Emit both exact mirrors and select their digest, which
/// preserves native byte parity without a protocol encoding change.
pub fn enforce_nebula_lane_digest_selected_circuit(
    builder: &mut R1csBuilder,
    lane: &NebulaLaneWires,
) -> [Var; DIGEST_LEN] {
    enforce_bit(builder, lane.open);
    let absent = enforce_nebula_lane_digest_circuit(builder, lane, GammaWires::Absent);
    let present = enforce_nebula_lane_digest_circuit(builder, lane, GammaWires::Present(lane.gamma));
    std::array::from_fn(|idx| enforce_mux_var(builder, lane.open, present[idx], absent[idx]))
}

/// Open a closed segment or carry an already-open segment, in one fixed
/// relation shape.
///
/// `input.open + lane.open = 1` enforces the native lifecycle rule: the
/// first claim of a segment opens it, and later claims cannot reopen it.
/// The γ candidate is always computed for fixed shape but selected only on
/// the open branch.
pub fn enforce_nebula_maybe_open_circuit(
    builder: &mut R1csBuilder,
    lane: &NebulaLaneWires,
    input: &DelayedNebulaInputWires,
    context: &NebulaOpenContextWires,
    seg_max: u64,
) -> NebulaLaneWires {
    enforce_segment_index_bound(builder, lane.seg_idx, seg_max);
    enforce_bit(builder, lane.open);
    enforce_bit(builder, input.open);
    let mut open_sum = Lc::from_var(lane.open);
    open_sum.add_term(input.open, F::ONE);
    builder.enforce_eq(&open_sum, &Lc::from_const(F::ONE));
    builder.enforce(&Lc::from_var(input.open), &Lc::from_var(lane.idx), &Lc::zero());

    let mut staged = lane.clone();
    staged.d_pre = input.d_pre;
    let staged_digest = enforce_nebula_lane_digest_circuit(builder, &staged, GammaWires::Absent);

    let mut transcript = TranscriptGadget::new(builder, NEBULA_GAMMA_TRANSCRIPT_LABEL);
    transcript.append_fields(builder, b"nebula/vk_fs", &context.vk_fs);
    transcript.append_fields(builder, b"nebula/z_i", &context.z_i);
    transcript.append_fields(builder, b"nebula/acc_digest", &context.acc_digest);
    transcript.append_fields(builder, b"nebula/lane", &staged_digest);
    transcript.append_fields(builder, b"nebula/plan", &context.plan_digest);
    transcript.append_fields(builder, b"nebula/seg_idx", &[lane.seg_idx]);
    transcript.append_fields(builder, b"nebula/ts", &[lane.ts]);
    transcript.append_fields(builder, b"nebula/d_pre_ops", &input.d_pre[0]);
    transcript.append_fields(builder, b"nebula/d_pre_is", &input.d_pre[1]);
    transcript.append_fields(builder, b"nebula/d_pre_fs", &input.d_pre[2]);
    let gamma1 = transcript.challenge_fields(builder, b"nebula/gamma1", 2);
    let gamma2 = transcript.challenge_fields(builder, b"nebula/gamma2", 2);
    let candidate_gamma = [KVar::new(gamma1[0], gamma1[1]), KVar::new(gamma2[0], gamma2[1])];

    let select_k = |builder: &mut R1csBuilder, opened: KVar, carried: KVar| {
        KVar::new(
            enforce_mux_var(builder, input.open, opened.c0, carried.c0),
            enforce_mux_var(builder, input.open, opened.c1, carried.c1),
        )
    };
    let gamma = std::array::from_fn(|idx| select_k(builder, candidate_gamma[idx], lane.gamma[idx]));
    let d_pre = std::array::from_fn(|digest| {
        std::array::from_fn(|limb| {
            enforce_mux_var(builder, input.open, input.d_pre[digest][limb], lane.d_pre[digest][limb])
        })
    });

    NebulaLaneWires {
        open: Var::ONE,
        seg_idx: lane.seg_idx,
        idx: lane.idx,
        ts: lane.ts,
        gamma,
        h: lane.h,
        sp: lane.sp,
        d_pre,
        d_seen: lane.d_seen,
        d_mem: lane.d_mem,
    }
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
    enforce_var_eq(builder, lane.open, Var::ONE);
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
        open: lane.open,
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
    enforce_var_eq(builder, lane.open, Var::ONE);
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
        open: zero,
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

pub struct NebulaMaybeCloseOutput {
    pub lane: NebulaLaneWires,
    pub closed: Var,
}

/// Consume one previous fresh claim's verifier-bound `(suffix, adv)` pair.
pub fn enforce_delayed_nebula_claim_circuit(
    builder: &mut R1csBuilder,
    lane: &NebulaLaneWires,
    input: &DelayedNebulaInputWires,
    adv: &AdvCommitmentWires,
    context: &NebulaOpenContextWires,
    steps_per_segment: u64,
    seg_max: u64,
) -> Result<NebulaMaybeCloseOutput, String> {
    validate_adv_shape(Some(adv), adv.ops.d, adv.ops.kappa, "delayed Nebula fresh claim")?;
    let opened = enforce_nebula_maybe_open_circuit(builder, lane, input, context, seg_max);
    let leaves = enforce_nebula_lane_leaf_digests_circuit(
        builder,
        adv.ops.d,
        adv.ops.kappa,
        &adv.ops.data,
        &adv.is.data,
        &adv.fs.data,
    );
    let advanced = enforce_nebula_advance_circuit(builder, &opened, &input.step, leaves);
    Ok(enforce_nebula_maybe_close_circuit(
        builder,
        &advanced,
        steps_per_segment,
    ))
}

fn enforce_segment_index_bound(builder: &mut R1csBuilder, seg_idx: Var, seg_max: u64) {
    let counter_domain = 1u64 << SEG_IDX_BITS;
    assert!((1..=counter_domain).contains(&seg_max));
    if seg_max == counter_domain {
        return;
    }

    let raw = builder.witness()[seg_idx.col()].as_canonical_u64();
    let mut bits = [Var::ONE; SEG_IDX_BITS];
    let mut recomposed = Lc::zero();
    for (index, bit) in bits.iter_mut().enumerate() {
        *bit = builder.alloc(F::from_u64((raw >> index) & 1));
        enforce_bit(builder, *bit);
        recomposed.add_term(*bit, F::from_u64(1u64 << index));
    }
    builder.enforce_eq(&Lc::from_var(seg_idx), &recomposed);

    // Lexicographic comparison from the most-significant bit. `equal`
    // remains one only while the witnessed prefix equals the constant prefix.
    // A one where the constant has zero is forbidden on an equal prefix; the
    // final equality is forbidden, yielding seg_idx < seg_max.
    let mut equal = Var::ONE;
    for index in (0..SEG_IDX_BITS).rev() {
        let bit = bits[index];
        let bound_bit = (seg_max >> index) & 1;
        let factor = if bound_bit == 1 {
            Lc::from_var(bit)
        } else {
            builder.enforce(&Lc::from_var(equal), &Lc::from_var(bit), &Lc::zero());
            let mut one_minus_bit = Lc::from_const(F::ONE);
            one_minus_bit.add_term(bit, -F::ONE);
            one_minus_bit
        };
        equal = builder.alloc_mul(&Lc::from_var(equal), &factor);
    }
    builder.enforce_eq(&Lc::from_var(equal), &Lc::zero());
}

/// Enforce and apply segment close exactly when `lane.idx == N`.
///
/// The zero-test makes `closed` an equivalence, not prover advice. Every
/// close-only obligation is multiplied by that bit; the returned state is a
/// mux between the live lane and the native reset state.
pub fn enforce_nebula_maybe_close_circuit(
    builder: &mut R1csBuilder,
    lane: &NebulaLaneWires,
    steps_per_segment: u64,
) -> NebulaMaybeCloseOutput {
    enforce_var_eq(builder, lane.open, Var::ONE);
    let mut distance = Lc::from_var(lane.idx);
    distance.add_constant(-F::from_u64(steps_per_segment));
    let distance_value = builder.eval(&distance);
    let closed = builder.alloc(if distance_value == F::ZERO { F::ONE } else { F::ZERO });
    enforce_bit(builder, closed);
    builder.enforce(&distance, &Lc::from_var(closed), &Lc::zero());

    use p3_field::Field;
    let inverse = builder.alloc(if distance_value == F::ZERO {
        F::ZERO
    } else {
        distance_value.inverse()
    });
    let mut not_closed = Lc::from_const(F::ONE);
    not_closed.add_term(closed, -F::ONE);
    builder.enforce(&distance, &Lc::from_var(inverse), &not_closed);

    for sp in lane.sp {
        enforce_zero_when(builder, closed, Lc::from_var(sp));
    }
    for (seen, pre) in lane.d_seen.iter().zip(lane.d_pre.iter()) {
        for (a, b) in seen.iter().zip(pre.iter()) {
            enforce_zero_when(builder, closed, Lc::from_var(*a).add_scaled(&Lc::from_var(*b), -F::ONE));
        }
    }
    let lhs = enforce_k_mul(builder, &KLc::from_var(lane.h[2]), &KLc::from_var(lane.h[1]));
    let rhs = enforce_k_mul(builder, &KLc::from_var(lane.h[0]), &KLc::from_var(lane.h[3]));
    enforce_zero_when(
        builder,
        closed,
        Lc::from_var(lhs.c0).add_scaled(&Lc::from_var(rhs.c0), -F::ONE),
    );
    enforce_zero_when(
        builder,
        closed,
        Lc::from_var(lhs.c1).add_scaled(&Lc::from_var(rhs.c1), -F::ONE),
    );
    for (seen, memory) in lane.d_seen[1].iter().zip(lane.d_mem.iter()) {
        enforce_zero_when(
            builder,
            closed,
            Lc::from_var(*seen).add_scaled(&Lc::from_var(*memory), -F::ONE),
        );
    }

    let seg_idx_reset = {
        let mut next = Lc::from_var(lane.seg_idx);
        next.add_constant(F::ONE);
        let var = builder.alloc(builder.eval(&next));
        builder.enforce_eq(&Lc::from_var(var), &next);
        var
    };
    let zero = alloc_constant(builder, F::ZERO);
    let one = alloc_constant(builder, F::ONE);
    let one_k = KVar::new(one, zero);
    let ops_header = alloc_const_digest(builder, nebula_chain_ops_header());
    let mem_header = alloc_const_digest(builder, nebula_chain_mem_header());
    let select = |builder: &mut R1csBuilder, reset: Var, live: Var| enforce_mux_var(builder, closed, reset, live);
    let select_k = |builder: &mut R1csBuilder, reset: KVar, live: KVar| {
        KVar::new(select(builder, reset.c0, live.c0), select(builder, reset.c1, live.c1))
    };

    let out = NebulaLaneWires {
        open: select(builder, zero, lane.open),
        seg_idx: select(builder, seg_idx_reset, lane.seg_idx),
        idx: select(builder, zero, lane.idx),
        ts: lane.ts,
        gamma: std::array::from_fn(|idx| select_k(builder, one_k, lane.gamma[idx])),
        h: std::array::from_fn(|idx| select_k(builder, one_k, lane.h[idx])),
        sp: std::array::from_fn(|idx| select(builder, zero, lane.sp[idx])),
        d_pre: std::array::from_fn(|idx| {
            let reset = if idx == 0 { ops_header } else { mem_header };
            std::array::from_fn(|lane_idx| select(builder, reset[lane_idx], lane.d_pre[idx][lane_idx]))
        }),
        d_seen: std::array::from_fn(|idx| {
            let reset = if idx == 0 { ops_header } else { mem_header };
            std::array::from_fn(|lane_idx| select(builder, reset[lane_idx], lane.d_seen[idx][lane_idx]))
        }),
        d_mem: std::array::from_fn(|idx| select(builder, lane.d_seen[2][idx], lane.d_mem[idx])),
    };
    NebulaMaybeCloseOutput { lane: out, closed }
}

// ── Internal helpers ──────────────────────────────────────────────────────

fn enforce_var_eq(builder: &mut R1csBuilder, a: Var, b: Var) {
    builder.enforce_eq(&Lc::from_var(a), &Lc::from_var(b));
}

fn enforce_k_eq(builder: &mut R1csBuilder, a: &KVar, b: &KVar) {
    enforce_var_eq(builder, a.c0, b.c0);
    enforce_var_eq(builder, a.c1, b.c1);
}

fn enforce_k_constant(builder: &mut R1csBuilder, wire: &KVar, value: K) {
    let (c0, c1) = value.to_limbs_u64();
    builder.enforce_eq(&Lc::from_var(wire.c0), &Lc::from_const(F::from_u64(c0)));
    builder.enforce_eq(&Lc::from_var(wire.c1), &Lc::from_const(F::from_u64(c1)));
}

fn enforce_zero_when(builder: &mut R1csBuilder, selector: Var, value: Lc) {
    builder.enforce(&Lc::from_var(selector), &value, &Lc::zero());
}

fn alloc_const_digest(builder: &mut R1csBuilder, digest: [F; DIGEST_LEN]) -> [Var; DIGEST_LEN] {
    digest.map(|value| alloc_constant(builder, value))
}
