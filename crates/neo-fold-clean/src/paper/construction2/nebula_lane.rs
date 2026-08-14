//! `NebulaLane` — the commitment-carrying memory state inside F′ and the
//! CC-IVC realization of Nebula Construction 2.
//!
//! Owns: the carried lane struct, its per-step transition
//! (`open_segment` / `advance` / `advance_for_batch` with every guard,
//! close check, and reset), the segment γ transcript, the public-input contract
//! (`NebulaStepX` and its verifier-side decode), and the
//! finalization predicate. One typed error per check so every rejection
//! test lands on the specific assert it targets.
//!
//! Does not own: the prover-side x bit-*encode*
//! (`frontends/nebula/layout.rs`, beside its lane encoders — the layout
//! owner mirrors this module's field order), lane commitments
//! (`relations/lanes.rs`), or the absorb formulas (`paper/digest.rs`).
//!
//! These transitions run natively in the lifecycle and have a matching F′ R1CS
//! implementation.

use neo_ajtai::Commitment;
use neo_ccs::LaneCommitments;
use neo_math::field::KExtensions;
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

use crate::engine::transcript::Transcript;
use crate::paper::digest;
use crate::paper::relations::LaneScheme;

/// Order of the four running products wherever `[K; 4]` appears in the
/// Nebula protocol: `h[0] = h_rs`, `h[1] = h_ws`, `h[2] = h_is`,
/// `h[3] = h_fs` in public-input layout order.
pub const H_RS: usize = 0;
/// See [`H_RS`].
pub const H_WS: usize = 1;
/// See [`H_RS`].
pub const H_IS: usize = 2;
/// See [`H_RS`].
pub const H_FS: usize = 3;

/// Label of the per-segment γ transcript.
pub const NEBULA_GAMMA_TRANSCRIPT_LABEL: &[u8] = b"neo.fold.clean/nebula/gamma/v3";

/// Width of the segment-counter slot in `x`.
pub const SEG_IDX_BITS: usize = 16;
/// Width of the step-counter slot in `x`.
pub const STEP_IDX_BITS: usize = 16;
/// Timestamp width.
pub const TS_BITS: usize = 44;
/// Bits per `K` coefficient (canonical Goldilocks limb).
pub const K_LIMB_BITS: usize = 64;
/// Bits per `K` element (two limbs: real, then imaginary).
pub const K_BITS: usize = 2 * K_LIMB_BITS;
/// Bits of the stack-less step public input (`1,400`). The full width is
/// [`StackShape::x_bits`].
pub const X_BASE_BITS: usize = SEG_IDX_BITS + STEP_IDX_BITS + 2 * TS_BITS + 2 * K_BITS + 8 * K_BITS;
/// Maximum stacks per plan. Fixed-size `sp` arrays are
/// sized by this; unused entries stay 0 everywhere.
pub const MAX_STACKS: usize = 2;

/// Stack geometry of a plan: how many segment-local
/// stacks and the σ-bit stack-pointer width. [`Self::NONE`] is the v3
/// shape. The step-x width derives from this, so it rides
/// [`NebulaConfig`] to every verifier-side decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StackShape {
    /// `S`: number of stacks (`≤ MAX_STACKS`).
    pub count: usize,
    /// `σ`: stack-pointer width in bits; capacity is `2^σ − 1` cells.
    pub sigma: usize,
}

impl StackShape {
    /// The v3 shape: no stacks, 1,400-bit x.
    pub const NONE: Self = Self { count: 0, sigma: 0 };

    /// Bits of the step public input: the 1,400 base slots
    /// plus `sp_in`/`sp_out` per stack, appended.
    pub const fn x_bits(&self) -> usize {
        X_BASE_BITS + 2 * self.count * self.sigma
    }
}

/// The decoded `S_mem` step public input. The canonical
/// struct and the **verifier-side decode** live here because the F′
/// transition consumes them; the prover-side bit *encode* lives with the
/// layout owner (`frontends/nebula/layout.rs`), which re-exports this
/// type and mirrors the same field order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaStepX {
    /// Segment counter `k`.
    pub seg_idx: u64,
    /// Step counter within the segment.
    pub idx: u64,
    /// Global timestamp entering this step.
    pub ts_in: u64,
    /// Global timestamp leaving this step.
    pub ts_out: u64,
    /// Segment challenges `(γ1, γ2)`.
    pub gamma: [K; 2],
    /// Running products entering this step (order: [`H_RS`]).
    pub h_in: [K; 4],
    /// Running products leaving this step (order: [`H_RS`]).
    pub h_out: [K; 4],
    /// Stack pointers entering this step (v3.1); unused stacks stay 0.
    /// (Length is [`MAX_STACKS`], written literally — rustc 1.94 ICEs on
    /// re-exported consts in array lengths under struct-update syntax.)
    pub sp_in: [u64; 2],
    /// Stack pointers leaving this step (v3.1); unused stacks stay 0.
    pub sp_out: [u64; 2],
}

const _: () = assert!(MAX_STACKS == 2, "sp arrays above are written as [u64; 2]");

/// Rejections of a claim's public input as an `S_mem` step x.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum NebulaXError {
    #[error("nebula x: expected {want} slots ({label}), got {got}")]
    Length {
        label: &'static str,
        want: usize,
        got: usize,
    },
    #[error("nebula x: slot {0} is not a bit")]
    NonBit(usize),
    #[error("nebula x: leading public slot must be the constant 1")]
    MissingConstantOne,
    #[error("nebula x: ring-completion slot {0} must be zero")]
    NonZeroCompletion(usize),
    #[error("nebula x: delayed suffix with open = 0 must carry an all-zero D_pre encoding")]
    ClosedDPreNonZero,
}

/// Previous claim's public Nebula payload in the one-step-delayed F'
/// relation. `d_pre` is present exactly when this claim opens a segment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DelayedNebulaStepX {
    pub step: NebulaStepX,
    pub d_pre: Option<[[F; 4]; 3]>,
}

impl NebulaStepX {
    /// Decode a claim's complete public ring
    /// `x = [1 ‖ bits ‖ zero completion]`. The logical prefix has length
    /// `1 + stacks.x_bits()`. The claim length is its unique multiple-of-`D`
    /// completion. The decoder validates the leading constant, every bit, and
    /// every zero completion coefficient.
    pub fn decode_claim_x(x: &[F], stacks: StackShape) -> Result<Self, NebulaXError> {
        let logical_len = 1 + stacks.x_bits();
        let complete_len = logical_len.div_ceil(D) * D;
        if x.len() != complete_len {
            return Err(NebulaXError::Length {
                label: "complete public ring",
                want: complete_len,
                got: x.len(),
            });
        }
        if x[0] != F::ONE {
            return Err(NebulaXError::MissingConstantOne);
        }
        if let Some((offset, _)) = x[logical_len..]
            .iter()
            .enumerate()
            .find(|(_, value)| **value != F::ZERO)
        {
            return Err(NebulaXError::NonZeroCompletion(logical_len + offset));
        }
        Self::decode_bits(&x[1..logical_len], stacks)
    }

    /// Decode the canonical bit body without the CCS constant-one slot.
    /// This is the verifier-side mirror used by delayed F' suffixes.
    pub fn decode_bits(bits: &[F], stacks: StackShape) -> Result<Self, NebulaXError> {
        if bits.len() != stacks.x_bits() {
            return Err(NebulaXError::Length {
                label: "step-x bits",
                want: stacks.x_bits(),
                got: bits.len(),
            });
        }
        let mut reader = BitReader { bits, at: 0 };
        let seg_idx = reader.read_u64(SEG_IDX_BITS)?;
        let idx = reader.read_u64(STEP_IDX_BITS)?;
        let ts_in = reader.read_u64(TS_BITS)?;
        let ts_out = reader.read_u64(TS_BITS)?;
        let gamma = [reader.read_k()?, reader.read_k()?];
        let mut h = || -> Result<[K; 4], NebulaXError> {
            Ok([reader.read_k()?, reader.read_k()?, reader.read_k()?, reader.read_k()?])
        };
        let h_in = h()?;
        let h_out = h()?;
        let mut sp_in = [0u64; MAX_STACKS];
        let mut sp_out = [0u64; MAX_STACKS];
        for s in 0..stacks.count {
            sp_in[s] = reader.read_u64(stacks.sigma)?;
            sp_out[s] = reader.read_u64(stacks.sigma)?;
        }
        Ok(Self {
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
}

impl DelayedNebulaStepX {
    /// Exact logical width of `[step_x_bits || open || bits(D_pre)]`.
    pub const fn encoded_len(stacks: StackShape) -> usize {
        const D_PRE_BITS: usize = 3 * 4 * K_LIMB_BITS;
        stacks.x_bits() + 1 + D_PRE_BITS
    }

    /// Decode `[step_x_bits || open || bits(D_pre)]`. All coordinates are
    /// canonical bits, and an inactive `D_pre` must be all zero, matching the
    /// authoritative circuit decoder.
    pub fn decode_suffix(suffix: &[F], stacks: StackShape) -> Result<Self, NebulaXError> {
        let expected = Self::encoded_len(stacks);
        if suffix.len() != expected {
            return Err(NebulaXError::Length {
                label: "delayed F' suffix",
                want: expected,
                got: suffix.len(),
            });
        }

        let step_end = stacks.x_bits();
        let step = NebulaStepX::decode_bits(&suffix[..step_end], stacks)?;
        let open = suffix[step_end];
        if open != F::ZERO && open != F::ONE {
            return Err(NebulaXError::NonBit(step_end));
        }
        let d_pre_bits = &suffix[step_end + 1..];
        if open == F::ZERO && d_pre_bits.iter().any(|bit| *bit != F::ZERO) {
            return Err(NebulaXError::ClosedDPreNonZero);
        }

        let d_pre = if open == F::ONE {
            let mut reader = BitReader {
                bits: d_pre_bits,
                at: 0,
            };
            let mut digests = [[F::ZERO; 4]; 3];
            for digest in &mut digests {
                for lane in digest {
                    *lane = F::from_u64(reader.read_u64(K_LIMB_BITS)?);
                }
            }
            Some(digests)
        } else {
            // Validate bitness even when the canonical value is all zero.
            for (index, bit) in d_pre_bits.iter().enumerate() {
                if *bit != F::ZERO && *bit != F::ONE {
                    return Err(NebulaXError::NonBit(step_end + 1 + index));
                }
            }
            None
        };
        Ok(Self { step, d_pre })
    }
}

/// Minimal little-endian bit reader over `{0,1}`-valued field slots.
struct BitReader<'a> {
    bits: &'a [F],
    at: usize,
}

impl BitReader<'_> {
    fn read_u64(&mut self, nbits: usize) -> Result<u64, NebulaXError> {
        let mut value = 0u64;
        for k in 0..nbits {
            let slot = self.at + k;
            let bit = self.bits[slot];
            if bit == F::ONE {
                value |= 1 << k;
            } else if bit != F::ZERO {
                return Err(NebulaXError::NonBit(slot + 1));
            }
        }
        self.at += nbits;
        Ok(value)
    }

    fn read_k(&mut self) -> Result<K, NebulaXError> {
        let c0 = F::from_u64(self.read_u64(K_LIMB_BITS)?);
        let c1 = F::from_u64(self.read_u64(K_LIMB_BITS)?);
        Ok(K::from_coeffs([c0, c1]))
    }
}

/// One F′ step's Nebula payload, computed by the lifecycle (which owns
/// the decode-and-advance loop) and consumed by `f_prime::{prove,
/// verify}` — the same shape-parameter pattern as `SemanticStateAdvance`.
#[derive(Clone, Debug)]
pub struct NebulaAdvance {
    /// The lane after advancing over the deposited batch;
    /// installed on the next `State` and bound by `x_out`.
    pub lane_out: NebulaLane,
    /// The segment-open `D_pre` claim, present exactly when this step
    /// opened a segment; recorded on `StepProof.nebula_open` so the
    /// verifier replays the same open.
    pub open: Option<[[F; 4]; 3]>,
}

/// Program-specific values the transition needs. `Preprocessing` owns
/// the exact values, while one Poseidon2 digest carries their binding.
#[derive(Clone, Debug)]
pub struct NebulaConfig {
    /// The lane-commitment context, also threaded to Π_DEC.
    pub scheme: LaneScheme,
    /// `N` — steps per segment under exact cover.
    pub steps_per_segment: u64,
    /// Maximum number of closed segments accepted under this plan.
    pub seg_max: u64,
    /// Stack geometry; fixes the public-input decode width and the
    /// `sp` carry. [`StackShape::NONE`] for stack-less plans.
    pub stacks: StackShape,
    /// Verifier-owned initial application-state digest. The carried lane
    /// binds this value without baking it into the reusable relation.
    pub initial_semantic_state_digest: [F; 4],
    /// Poseidon2 digest of the canonical plan serialization.
    pub plan_digest: [F; 4],
    /// The verifier's ROM handle: mem-domain chain over the initial
    /// memory's lane commitments, independent of γ.
    pub d_init: [F; 4],
}

impl NebulaConfig {
    pub fn program_binding_digest(&self) -> [F; 4] {
        digest::nebula_program_binding_digest(&self.initial_semantic_state_digest, &self.plan_digest, &self.d_init)
    }
}

/// One lane-transition check per variant; rejection tests target these by name.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum NebulaError {
    #[error("nebula: carried program binding does not match verifier preprocessing")]
    ProgramBindingMismatch,
    #[error("nebula: open_segment requires idx == 0 and no open γ (segment already open or mid-segment)")]
    SegmentAlreadyOpen,
    #[error("nebula: segment index {seg_idx} reached plan limit {seg_max}")]
    SegmentLimit { seg_idx: u64, seg_max: u64 },
    #[error("nebula: advance requires an open segment (γ squeezed, idx < N)")]
    SegmentNotOpen,
    #[error("nebula: claim x counters (seg {x_seg}, idx {x_idx}) do not match lane (seg {lane_seg}, idx {lane_idx})")]
    CounterMismatch {
        x_seg: u64,
        x_idx: u64,
        lane_seg: u64,
        lane_idx: u64,
    },
    #[error("nebula: claim ts_in {x_ts} does not match lane ts {lane_ts}")]
    TsMismatch { x_ts: u64, lane_ts: u64 },
    #[error("nebula: claim γ does not match the segment's squeezed γ")]
    GammaMismatch,
    #[error("nebula: claim h_in does not match the lane's running products")]
    ProductThreadMismatch,
    #[error("nebula: claim sp_in does not match the lane's running stack pointers")]
    StackPointerMismatch,
    #[error("nebula: segment close — stacks must end empty (segment-local discipline, sp != 0)")]
    StackNotEmptyAtClose,
    #[error("nebula: deposited claim carries no adv tuple inside an open Nebula segment")]
    MissingAdv,
    #[error("nebula: segment close — folded lane chains do not match the pre-committed chains (D_seen != D_pre)")]
    PreSeenMismatch,
    #[error("nebula: segment close — multiset product equation failed (h_is·h_ws != h_rs·h_fs)")]
    ProductEquation,
    #[error("nebula: segment close — memory continuity failed (D_seen[is] != D_mem)")]
    BoundaryMismatch,
}

/// The constant-size carried memory state. Rides
/// `State.nebula`; its [`Self::digest`] is absorbed into `state_x_out`
/// and the F′ step transcript every step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaLane {
    /// Poseidon2 compression of the verifier-owned initial semantic state,
    /// plan digest, and initial-memory handle. The verifier recomputes it
    /// from those authoritative values.
    pub program_binding_digest: [F; 4],
    /// Current segment `k`.
    pub seg_idx: u64,
    /// Step within the segment; `0` between segments.
    pub idx: u64,
    /// Global timestamp — never resets across segments.
    pub ts: u64,
    /// `(γ1, γ2)` of the open segment; `None` is the spec's `⊥`.
    pub gamma: Option<[K; 2]>,
    /// Running `(h_rs, h_ws, h_is, h_fs)` (order: [`H_RS`]).
    pub h: [K; 4],
    /// Running stack pointers (v3.1); `0` at every segment boundary —
    /// stacks are segment-local. Unused stacks stay 0.
    pub sp: [u64; MAX_STACKS],
    /// Per-lane pre-committed chain digests (ops, is, fs), claimed at
    /// open and given authority retroactively by the close check.
    pub d_pre: [[F; 4]; 3],
    /// Per-lane running chains over the folded claims' leaves.
    pub d_seen: [[F; 4]; 3],
    /// Boundary handle: previous segment's final fs chain (`D_init` at
    /// chain start).
    pub d_mem: [F; 4],
}

/// Chain headers, ordered (ops, is, fs): `is` and `fs` share the
/// mem-domain header — formula identity for the boundary equality.
fn chain_headers() -> [[F; 4]; 3] {
    let mem = digest::nebula_chain_mem_header();
    [digest::nebula_chain_ops_header(), mem, mem]
}

/// Link tags per lane, ordered (ops, is, fs) — `is`/`fs` share `"mem"`.
const LINK_TAGS: [&[u8]; 3] = [
    digest::NEBULA_CHAIN_OPS_TAG,
    digest::NEBULA_CHAIN_MEM_TAG,
    digest::NEBULA_CHAIN_MEM_TAG,
];

impl NebulaLane {
    /// Chain start: counters and timestamp at zero,
    /// products at `1_K`, chains at headers, memory bound to the plan's
    /// `D_init`.
    pub fn base(cfg: &NebulaConfig) -> Self {
        Self {
            program_binding_digest: cfg.program_binding_digest(),
            seg_idx: 0,
            idx: 0,
            ts: 0,
            gamma: None,
            h: [K::ONE; 4],
            sp: [0; MAX_STACKS],
            d_pre: chain_headers(),
            d_seen: chain_headers(),
            d_mem: cfg.d_init,
        }
    }

    /// Segment open and its γ transcript.
    ///
    /// `d_pre` is the prover's **claim** about the segment's forthcoming
    /// lane-leaf chains; its authority is retroactive via the
    /// close equality. γ is squeezed from a fresh Poseidon2 transcript
    /// seeded by the F′ carried state at open (`vk_fs`, `z_i`,
    /// `acc_digest`, this lane) and absorbing the program binding, the
    /// segment counters, and the three `D_pre` digests — never the raw
    /// commitment list (constant-size replay in the eventual F′ R1CS).
    pub fn open_segment(
        &mut self,
        cfg: &NebulaConfig,
        vk_digest: [u8; 32],
        z_i: [u8; 32],
        acc_digest: [u8; 32],
        d_pre: [[F; 4]; 3],
    ) -> Result<(), NebulaError> {
        self.validate_program_binding(cfg)?;
        if self.idx != 0 || self.gamma.is_some() {
            return Err(NebulaError::SegmentAlreadyOpen);
        }
        if self.seg_idx >= cfg.seg_max {
            return Err(NebulaError::SegmentLimit {
                seg_idx: self.seg_idx,
                seg_max: cfg.seg_max,
            });
        }
        self.d_pre = d_pre;
        let mut tr = Transcript::with_label(NEBULA_GAMMA_TRANSCRIPT_LABEL);
        tr.append_fields(b"nebula/vk_fs", &digest::digest32_as_fields(vk_digest));
        tr.append_fields(b"nebula/z_i", &digest::digest32_as_fields(z_i));
        tr.append_fields(b"nebula/acc_digest", &digest::digest32_as_fields(acc_digest));
        tr.append_fields(b"nebula/lane", &self.digest());
        tr.append_fields(b"nebula/program_binding", &self.program_binding_digest);
        tr.append_fields(b"nebula/seg_idx", &[F::from_u64(self.seg_idx)]);
        tr.append_fields(b"nebula/ts", &[F::from_u64(self.ts)]);
        tr.append_fields(b"nebula/d_pre_ops", &d_pre[0]);
        tr.append_fields(b"nebula/d_pre_is", &d_pre[1]);
        tr.append_fields(b"nebula/d_pre_fs", &d_pre[2]);
        let squeeze_k = |tr: &mut Transcript, label: &'static [u8]| {
            let cf = tr.challenge_fields(label, 2);
            K::from_coeffs([cf[0], cf[1]])
        };
        self.gamma = Some([
            squeeze_k(&mut tr, b"nebula/gamma1"),
            squeeze_k(&mut tr, b"nebula/gamma2"),
        ]);
        Ok(())
    }

    /// Per-step transition, run for each
    /// deposited claim in order — both sides compute it identically.
    /// Closes the segment (three equalities, `D_mem` handoff, reset
    /// without `ts`) when the step is the segment's `N`-th.
    pub fn advance(
        &mut self,
        cfg: &NebulaConfig,
        x: &NebulaStepX,
        adv: Option<&LaneCommitments<Commitment>>,
    ) -> Result<(), NebulaError> {
        self.validate_program_binding(cfg)?;
        let Some(gamma) = self.gamma else {
            return Err(NebulaError::SegmentNotOpen);
        };
        if self.idx >= cfg.steps_per_segment {
            return Err(NebulaError::SegmentNotOpen);
        }
        let Some(adv) = adv else {
            return Err(NebulaError::MissingAdv);
        };
        if x.seg_idx != self.seg_idx || x.idx != self.idx {
            return Err(NebulaError::CounterMismatch {
                x_seg: x.seg_idx,
                x_idx: x.idx,
                lane_seg: self.seg_idx,
                lane_idx: self.idx,
            });
        }
        if x.ts_in != self.ts {
            return Err(NebulaError::TsMismatch {
                x_ts: x.ts_in,
                lane_ts: self.ts,
            });
        }
        if x.gamma != gamma {
            return Err(NebulaError::GammaMismatch);
        }
        if x.h_in != self.h {
            return Err(NebulaError::ProductThreadMismatch);
        }
        if x.sp_in != self.sp {
            return Err(NebulaError::StackPointerMismatch);
        }

        let leaves = digest::nebula_lane_leaf_digests(adv);
        for lane_id in 0..3 {
            self.d_seen[lane_id] =
                digest::nebula_chain_link(&self.d_seen[lane_id], LINK_TAGS[lane_id], &leaves[lane_id]);
        }
        self.h = x.h_out;
        self.sp = x.sp_out;
        self.ts = x.ts_out;
        self.idx += 1;

        if self.idx == cfg.steps_per_segment {
            self.close(cfg)?;
        }
        Ok(())
    }

    /// Segment close: the three equalities, the boundary
    /// handoff, and the reset that never touches `ts`. The `sp == 0`
    /// check is the deterministic companion to the product equation,
    /// which already rejects an unpopped push w.h.p. (segment-local
    /// stack discipline).
    fn close(&mut self, _cfg: &NebulaConfig) -> Result<(), NebulaError> {
        if self.sp != [0; MAX_STACKS] {
            return Err(NebulaError::StackNotEmptyAtClose);
        }
        if self.d_seen != self.d_pre {
            return Err(NebulaError::PreSeenMismatch);
        }
        if self.h[H_IS] * self.h[H_WS] != self.h[H_RS] * self.h[H_FS] {
            return Err(NebulaError::ProductEquation);
        }
        if self.d_seen[1] != self.d_mem {
            return Err(NebulaError::BoundaryMismatch);
        }
        self.d_mem = self.d_seen[2];
        self.seg_idx += 1;
        self.idx = 0;
        self.gamma = None;
        self.h = [K::ONE; 4];
        self.d_pre = chain_headers();
        self.d_seen = chain_headers();
        Ok(())
    }

    /// An externally accepted proof must end at a closed segment: `idx == 0`,
    /// `γ == ⊥`, and chains at
    /// headers. A trailing open segment has folded op rows whose product
    /// equation and `D_seen == D_pre` binding were never checked.
    pub fn is_closed(&self) -> bool {
        self.idx == 0 && self.gamma.is_none() && self.d_pre == chain_headers() && self.d_seen == chain_headers()
    }

    pub fn matches_program_binding(&self, cfg: &NebulaConfig) -> bool {
        self.program_binding_digest == cfg.program_binding_digest()
    }

    fn validate_program_binding(&self, cfg: &NebulaConfig) -> Result<(), NebulaError> {
        if self.matches_program_binding(cfg) {
            Ok(())
        } else {
            Err(NebulaError::ProgramBindingMismatch)
        }
    }

    /// Advance over one deposited batch through the shared prove/verify
    /// transition: an optional segment open followed by one advance per
    /// deposited claim, in order. Both sides
    /// call exactly this, so a divergence is impossible by construction.
    ///
    /// `vk_digest`, `z_i`, and `acc_digest` are the F′ carried state at
    /// this step's input — the γ-transcript seed when the batch
    /// opens a segment.
    pub fn advance_for_batch(
        &mut self,
        cfg: &NebulaConfig,
        vk_digest: [u8; 32],
        z_i: [u8; 32],
        acc_digest: [u8; 32],
        open: Option<[[F; 4]; 3]>,
        claims: &[neo_ccs::CcsClaim<Commitment, F>],
    ) -> Result<(), crate::paper::construction2::Error> {
        if let Some(d_pre) = open {
            self.open_segment(cfg, vk_digest, z_i, acc_digest, d_pre)?;
        }
        for claim in claims {
            let x = NebulaStepX::decode_claim_x(&claim.x, cfg.stacks)?;
            self.advance(cfg, &x, claim.adv.as_ref())?;
        }
        Ok(())
    }

    /// Advance over claims of the authoritative folded F' relation. Each
    /// claim carries `[1 || enc_inst || delayed Nebula suffix]`; the suffix
    /// belongs to the application execution embedded in that claim and is
    /// consumed one recursion step later.
    pub fn advance_for_delayed_claims(
        &mut self,
        cfg: &NebulaConfig,
        vk_digest: [u8; 32],
        z_i: [u8; 32],
        acc_digest: [u8; 32],
        suffix_offset: usize,
        claims: &[neo_ccs::CcsClaim<Commitment, F>],
    ) -> Result<(), crate::paper::construction2::Error> {
        let suffix_end = suffix_offset + DelayedNebulaStepX::encoded_len(cfg.stacks);
        for claim in claims {
            if claim.x.len() < suffix_end {
                return Err(NebulaXError::Length {
                    label: "folded F' claim through delayed suffix",
                    want: suffix_end,
                    got: claim.x.len(),
                }
                .into());
            }
            let delayed = DelayedNebulaStepX::decode_suffix(&claim.x[suffix_offset..suffix_end], cfg.stacks)?;
            if let Some(d_pre) = delayed.d_pre {
                self.open_segment(cfg, vk_digest, z_i, acc_digest, d_pre)?;
            }
            self.advance(cfg, &delayed.step, claim.adv.as_ref())?;
        }
        Ok(())
    }

    /// The compact handle absorbed into `state_x_out` and the F′ step
    /// transcript in constant space.
    pub fn digest(&self) -> [F; 4] {
        digest::nebula_lane_digest(
            &self.program_binding_digest,
            self.seg_idx,
            self.idx,
            self.ts,
            self.gamma.as_ref(),
            &self.h,
            &self.sp,
            &self.d_pre,
            &self.d_seen,
            &self.d_mem,
        )
    }
}
