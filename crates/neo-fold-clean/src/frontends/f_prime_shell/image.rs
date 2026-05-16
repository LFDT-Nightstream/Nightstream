//! App-agnostic encoded-F' source-image skeleton.
//!
//! Composes the four Phase 1.1 primitives (`poseidon_trace`,
//! `poseidon2_transcript`, `ring_action_trace`, plus the existing
//! `source_image` boundary layer) into a single layout + image for one
//! recursive F' step. Concrete app frontends
//! (`crate::frontends::fibonacci_f_prime`,
//! `crate::frontends::r1cs_f_prime`) build on this shell.
//!
//! This is a skeleton: the layout describes every region (boundary–poseidon from
//! the Phase 1.1 spec §2) with non-overlapping offsets, and the image
//! starts zero-filled (only `z[0] = F::ONE`, the CCS constant slot).
//! Callers splice primitive sub-traces into the right offsets via the
//! `splice_*` methods, then decode back via the `decode_*` helpers to
//! verify composition parity.
//!
//! Out of scope for Phase 1.2:
//! - Per-region encoder logic (Phase 1.3).
//! - CCS structure for `enc(F')` (Phase 1.4).
//! - Lifecycle migration (Phase 1.5).
//! - Generic `AppStep` trait, Spartan, anything that turns an
//!   `ivc_invariants` test green.

use neo_math::ring::D;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use p3_field::PrimeField64;

use crate::engine::ccs_native::poseidon2::POSEIDON2_GOLDILOCKS_BITS;
use crate::engine::ccs_native::poseidon2_transcript::SpongeTraceImage;
use crate::paper::f_prime::poseidon_trace::{PoseidonTraceImage, PoseidonTraceLayout, BITS_PER_PERMUTATION};
use crate::paper::f_prime::ring_action_trace::{RingActionTraceImage, RingActionTraceLayout};

/// 4 lanes × 64 bits = 256 bits per digest.
const DIGEST_BITS: usize = 4 * POSEIDON2_GOLDILOCKS_BITS;

/// state_in state-in: 6 four-lane digests (vk_fs, structure, z_0, z_i_in,
/// acc_digest_in, public_trace_in).
const STATE_IN_BITS: usize = 6 * DIGEST_BITS;

/// state_out state-out: 2 u64 counters (new chunk_count, new step_count) + 3
/// four-lane digests (new z_i, new public_trace, new acc_digest).
const STATE_OUT_BITS: usize = 2 * POSEIDON2_GOLDILOCKS_BITS + 3 * DIGEST_BITS;

/// chunk_digest chunk digest: one four-lane digest.
const CHUNK_DIGEST_BITS: usize = DIGEST_BITS;
/// Single committed bit marking whether this step is the base case
/// (`is_base = 1`, no prior fold) or a recursive case (`is_base = 0`).
/// Used by the unified-accumulator selector constraint to pick which
/// accumulator Poseidon trace's digest enters `state_out.new_acc_digest`.
const IS_BASE_BITS: usize = 1;

/// kmul K-mul Karatsuba intermediate: 3 K-limbs × 2 lanes × 64 bits each.
const KMUL_SLOT_BITS: usize = 3 * 2 * POSEIDON2_GOLDILOCKS_BITS;

/// A contiguous bit range within the F' source image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegionRange {
    pub offset: usize,
    pub bits: usize,
}

impl RegionRange {
    pub fn end(&self) -> usize {
        self.offset + self.bits
    }
}

/// Knobs sizing the F' image for one recursive step.
#[derive(Clone, Debug)]
pub struct FPrimeImageConfig {
    /// Fibonacci `LIMBS`; private bits = `LIMBS - 1` carries.
    pub limbs: usize,
    /// boundary bit count (existing `source_image` BitRange size).
    pub boundary_bits: usize,
    /// nifs_payloads NIFS payload shapes, in fill order. The layout derives
    /// `nifs_payloads.bits` from `Σ shape.bits()` and exposes per-payload
    /// offsets via `nifs_payload_offsets`.
    pub nifs_payload_shapes: Vec<NifsPayloadShape>,
    /// kmul K-mul Karatsuba intermediate count.
    pub kmul_count: usize,
    /// Number of ring-action lane-pairs in one step (κ · k_total).
    pub ring_action_pair_count: usize,
    /// Shared per-pair ring-action layout (encoding widths per subregion).
    pub ring_action_pair_layout: RingActionTraceLayout,
    /// Preimage lengths for each one-shot Poseidon hash invoked in this step.
    pub poseidon_one_shot_preimage_lens: Vec<usize>,
    /// Number of permutations in the F' sponge transcript session.
    pub sponge_transcript_permutes: usize,
    /// Optional bindings: each entry asserts that one-shot Poseidon
    /// trace at `one_shot_index`'s digest output (4 lanes) equals the
    /// state_out digest at `state_out_target`. Default empty — no bindings emitted.
    /// Used by Phase 1.4c-c to close the "trace produces the committed
    /// digest" loop for the boundary_update / public_trace_update /
    /// accumulator state-advance hashes.
    pub one_shot_digest_to_state_out_bindings: Vec<OneShotDigestToStateOutBinding>,
    /// Optional bindings: each entry asserts that the one-shot Poseidon
    /// trace at `one_shot_index`'s digest output (4 lanes) equals the
    /// 4 canonical-u64 lanes at the listed bit offsets inside the
    /// image's boundary region. Default empty. Used by Phase 1.4c-d to close
    /// the IVC public-output loop: the `state_x_out` Poseidon hash
    /// produces the public `x_out` bits committed in boundary.
    pub one_shot_digest_to_public_x_out_bindings: Vec<OneShotDigestToPublicXOutBinding>,
    /// Optional Phase 1.4d-a-3 enforcements: for each entry, the F'
    /// structure mechanically ports the bit-backed Poseidon2 CCS rows
    /// for `poseidon2_hash(preimage)` into the spliced one-shot trace
    /// region. Default empty.
    pub poseidon_transition_enforcements: Vec<PoseidonTransitionEnforcement>,
    /// Unified accumulator selector — present when the F' structure
    /// hosts **both** the base-case accumulator preimage (`H(tag, 0)`)
    /// and the recursive-case accumulator preimage
    /// (`H(tag, child_count, c_data_entries, c_data ...)`), with
    /// `state_out.new_acc_digest` selected by the `is_base` lane via
    ///
    /// ```text
    /// (1 - is_base) * (recursive_lane - base_lane)
    ///   = (new_acc_digest_lane - base_lane)
    /// ```
    ///
    /// When `Some(..)` the structure builder skips the direct linear
    /// binding for `StateOutDigestTarget::NewAccDigest` and emits the
    /// four selector product rows instead. When `None`, the legacy
    /// single-accumulator path applies.
    pub unified_accumulator_selector: Option<UnifiedAccumulatorSelector>,
}

/// Indices into `poseidon_one_shot_preimage_lens` for the two
/// accumulator Poseidon traces the unified-mode F' structure hosts.
///
/// Both indices are required (the structure emits both bindings + the
/// selector that picks between them). The `is_base` lane in the image
/// drives the selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnifiedAccumulatorSelector {
    /// One-shot index of the **base** accumulator trace (preimage
    /// `H(tag, 0)`, producing the empty-accumulator digest).
    pub base_trace_index: usize,
    /// One-shot index of the **recursive** accumulator trace (preimage
    /// `H(tag, child_count, c_data_entries, c_data ...)`).
    pub recursive_trace_index: usize,
}

/// Which state_out digest a one-shot trace's output binds to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StateOutDigestTarget {
    /// state_out's `new_z_i` (4 lanes after the two u64 counters).
    NewZI,
    /// state_out's `new_public_trace` (4 lanes after `new_z_i`).
    NewPublicTrace,
    /// state_out's `new_acc_digest` (4 lanes after `new_public_trace`).
    NewAccDigest,
}

/// One Phase 1.4c-c binding: trace's digest output ↔ state_out digest lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OneShotDigestToStateOutBinding {
    /// Index into `poseidon_one_shot_preimage_lens` / `one_shot_poseidon_splices`.
    pub one_shot_index: usize,
    pub state_out_target: StateOutDigestTarget,
}

/// One Phase 1.4c-d binding: trace's digest output ↔ four canonical-u64
/// boundary lanes (e.g., the public `x_out` digest bits).
///
/// `public_x_out_lane_bit_starts` carries the absolute bit offset (in the image's
/// `values`) of each of the 4 digest lanes. Caller chooses the
/// positions; they must lie inside the boundary region.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OneShotDigestToPublicXOutBinding {
    pub one_shot_index: usize,
    pub public_x_out_lane_bit_starts: [usize; 4],
}

/// Where one Poseidon preimage lane reads from inside the F' image.
///
/// The Poseidon transition enforcement uses these sources to bind each
/// absorbed preimage value to a specific decoded F' image column (or to
/// a hard constant). The native bit-backed Poseidon2 builder bakes the
/// absorbed inputs as row constants; F' replaces those rows with
/// source-aware absorb rows so the structure stays valid across
/// different preimage values as long as they agree with the named
/// source. This is the closure step for Phase 1.4d-a-4.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PoseidonPreimageLaneSource {
    /// A literal field constant. Useful for length prefixes / header
    /// bytes that don't vary across IVC steps.
    Constant(F),
    /// The full canonical-u64 value of a state-in / state-out /
    /// chunk-digest lane — `lane_slots.state_lanes[i]`. Resolved by the
    /// structure as `lane_terms(slot)` (i.e. `Σ_{j<64} 2^j · z[bit_start + j]`);
    /// no decoded witness column is allocated.
    StateLane(usize),
    /// The low 32 bits of `lane_slots.state_lanes[i]` read directly
    /// from bit columns. Equals `Σ_{j<32} 2^j · z[bit_start + j]`.
    /// Used by hashes whose preimages encode counters as
    /// `[F::from_u64(value & 0xffff_ffff), F::from_u64(value >> 32)]`.
    StateLaneLowHalf(usize),
    /// The high 32 bits of `lane_slots.state_lanes[i]` read directly
    /// from bit columns. Equals `Σ_{j<32} 2^j · z[bit_start + 32 + j]`.
    StateLaneHighHalf(usize),
    /// Canonical-u64 value of `lane_slots.nifs_payload_lanes[payload_index][lane_index]`.
    /// Resolved through `lane_terms(slot)` directly from bit columns.
    NifsPayloadLane {
        payload_index: usize,
        lane_index: usize,
    },
    /// Canonical-u64 value of `lane_slots.ring_action_lanes[i]`,
    /// resolved through `lane_terms(slot)`.
    RingActionLane(usize),
    /// Another Poseidon trace's word — the chain-hash case. Resolved
    /// through `lane_terms(lane_slots.poseidon_trace_lanes[trace_index][lane_index])`.
    PoseidonTraceLane {
        trace_index: usize,
        lane_index: usize,
    },
    /// Canonical-u64 value of `lane_slots.sponge_transcript_lanes[i]`,
    /// resolved through `lane_terms(slot)`.
    SpongeTranscriptLane(usize),
    /// Canonical-u64 value of the boundary public-x_out lane bound by
    /// `lane_slots.public_x_out_binding_lanes[binding_index][lane]`,
    /// resolved through `lane_terms(slot)`.
    PublicXOutBindingLane { binding_index: usize, lane: usize },
    /// Canonical-u64 value of the `j`-th app-assignment variable —
    /// `lane_slots.app_assignment_lanes[var_index]`. Resolved through
    /// `lane_terms(slot)` directly from the 64 committed bits at
    /// `layout.app_private.offset + var_index * 64`.
    ///
    /// Intended for app frontends (R1CS today) that bind app-level
    /// public-input bits to an algebraically-enforced Poseidon hash so
    /// the verifier-visible `state_x_out` is bound to the proven public
    /// input `x`. The Fibonacci frontend's `app_private` region holds
    /// carries (not 64-bit lanes) and does not use this variant.
    AppAssignmentLane(usize),
}

/// One Phase 1.4d-a-4 enforcement: bind the trace at `one_shot_index`
/// to compute `poseidon2_hash(preimage_lanes)`, where each preimage
/// lane reads from a named F' image source instead of being baked into
/// the structure as a constant.
///
/// The shape (preimage length, absorb count) is fixed by
/// `preimage_lanes.len()`. The values come from sources at constraint
/// time, not from config-side constants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoseidonTransitionEnforcement {
    pub one_shot_index: usize,
    pub preimage_lanes: Vec<PoseidonPreimageLaneSource>,
}

/// One nifs_payloads NIFS-payload slot's shape. The image config carries a list
/// of these in fill order; the layout reserves contiguous bits per
/// payload and tests/decoders index by payload position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NifsPayloadShape {
    CcsClaim(NifsCcsClaimShape),
    CeClaim(NifsCeClaimShape),
}

impl NifsPayloadShape {
    pub fn bits(&self) -> usize {
        match self {
            Self::CcsClaim(s) => s.bits(),
            Self::CeClaim(s) => s.bits(),
        }
    }
}

/// Concrete layout for one F' image.
#[derive(Clone, Debug)]
pub struct FPrimeImageLayout {
    pub config: FPrimeImageConfig,
    pub boundary: RegionRange,
    pub state_in: RegionRange,
    pub state_out: RegionRange,
    pub chunk_digest: RegionRange,
    pub app_private: RegionRange,
    /// Single committed bit; `1` for base step, `0` for recursive step.
    /// Drives the unified-accumulator selector constraint emitted in
    /// `fibonacci_structure::build_f_prime_shell_structure`. Always
    /// reserved (one bit) even when the plan has no
    /// `AccumulatorPlanOptions` — the encoder writes `0` there in that
    /// case, and the structure's bit-validity row covers the binary
    /// constraint either way.
    pub is_base: RegionRange,
    pub nifs_payloads: RegionRange,
    /// Per-nifs_payloads-payload absolute offset (in image `values`), one entry per
    /// `config.nifs_payload_shapes`.
    pub nifs_payload_offsets: Vec<usize>,
    pub kmul: RegionRange,
    pub ring_action: RegionRange,
    /// Splice offset (in image `values`) for each ring-action pair's
    /// non-constant bits.
    pub ring_action_pair_splices: Vec<usize>,
    pub poseidon: RegionRange,
    /// Splice offset for each one-shot Poseidon trace.
    pub one_shot_poseidon_splices: Vec<usize>,
    /// Per-call layout descriptor for each one-shot Poseidon trace.
    pub one_shot_poseidon_layouts: Vec<PoseidonTraceLayout>,
    /// Splice offset for the sponge transcript session.
    pub sponge_transcript_splice: usize,
    /// Bit count for the sponge transcript session.
    pub sponge_transcript_bits: usize,
    /// One past the last bit used by the image.
    pub end: usize,
}

impl FPrimeImageLayout {
    pub fn new(config: FPrimeImageConfig) -> Self {
        // z[0] = constant slot; non-constant regions start at offset 1.
        let mut cursor = 1usize;

        let boundary = RegionRange {
            offset: cursor,
            bits: config.boundary_bits,
        };
        cursor = boundary.end();

        let state_in = RegionRange {
            offset: cursor,
            bits: STATE_IN_BITS,
        };
        cursor = state_in.end();

        let state_out = RegionRange {
            offset: cursor,
            bits: STATE_OUT_BITS,
        };
        cursor = state_out.end();

        let chunk_digest = RegionRange {
            offset: cursor,
            bits: CHUNK_DIGEST_BITS,
        };
        cursor = chunk_digest.end();

        let app_private = RegionRange {
            offset: cursor,
            bits: config.limbs.saturating_sub(1),
        };
        cursor = app_private.end();

        let is_base = RegionRange {
            offset: cursor,
            bits: IS_BASE_BITS,
        };
        cursor = is_base.end();

        let nifs_start = cursor;
        let mut nifs_payload_offsets = Vec::with_capacity(config.nifs_payload_shapes.len());
        for shape in &config.nifs_payload_shapes {
            nifs_payload_offsets.push(cursor);
            cursor += shape.bits();
        }
        let nifs_payloads = RegionRange {
            offset: nifs_start,
            bits: cursor - nifs_start,
        };

        let kmul = RegionRange {
            offset: cursor,
            bits: config.kmul_count * KMUL_SLOT_BITS,
        };
        cursor = kmul.end();

        // ring_action: each pair contributes (pair_layout.end - 1) bits (drop the
        // pair-local constant slot — the F' image's z[0] is shared).
        let pair_bits = config.ring_action_pair_layout.end - 1;
        let ring_action_start = cursor;
        let ring_action_pair_splices: Vec<usize> = (0..config.ring_action_pair_count)
            .map(|i| ring_action_start + i * pair_bits)
            .collect();
        cursor = ring_action_start + config.ring_action_pair_count * pair_bits;
        let ring_action = RegionRange {
            offset: ring_action_start,
            bits: cursor - ring_action_start,
        };

        // poseidon: one-shot Poseidon traces, then the sponge-transcript trace.
        let poseidon_start = cursor;
        let mut one_shot_poseidon_splices = Vec::with_capacity(config.poseidon_one_shot_preimage_lens.len());
        let mut one_shot_poseidon_layouts = Vec::with_capacity(config.poseidon_one_shot_preimage_lens.len());
        for &preimage_len in &config.poseidon_one_shot_preimage_lens {
            let layout = PoseidonTraceLayout::from_preimage_len(preimage_len);
            one_shot_poseidon_splices.push(cursor);
            one_shot_poseidon_layouts.push(layout);
            cursor += layout.trace_len;
        }
        let sponge_transcript_splice = cursor;
        let sponge_transcript_bits = config.sponge_transcript_permutes * BITS_PER_PERMUTATION;
        cursor += sponge_transcript_bits;
        let poseidon = RegionRange {
            offset: poseidon_start,
            bits: cursor - poseidon_start,
        };

        let end = cursor;

        Self {
            config,
            boundary,
            state_in,
            state_out,
            chunk_digest,
            app_private,
            is_base,
            nifs_payloads,
            nifs_payload_offsets,
            kmul,
            ring_action,
            ring_action_pair_splices,
            poseidon,
            one_shot_poseidon_splices,
            one_shot_poseidon_layouts,
            sponge_transcript_splice,
            sponge_transcript_bits,
            end,
        }
    }

    /// Top-level region ranges in spec order (boundary..poseidon).
    pub fn top_level_regions(&self) -> [RegionRange; 10] {
        [
            self.boundary,
            self.state_in,
            self.state_out,
            self.chunk_digest,
            self.app_private,
            self.is_base,
            self.nifs_payloads,
            self.kmul,
            self.ring_action,
            self.poseidon,
        ]
    }
}

/// One F' recursive step's bit-backed witness, plus the
/// layout that names its regions.
#[derive(Clone, Debug)]
pub struct FPrimeImage {
    pub layout: FPrimeImageLayout,
    /// `values[0] = F::ONE`; all later entries are `{0, 1}` once any
    /// splice has populated them. Unspliced regions remain zero.
    pub values: Vec<F>,
}

impl FPrimeImage {
    pub fn new(layout: FPrimeImageLayout) -> Self {
        let mut values = vec![F::ZERO; layout.end];
        values[0] = F::ONE;
        Self { layout, values }
    }

    /// Write the `is_base` bit (1 for base step, 0 for recursive). The
    /// unified-accumulator selector reads this lane to pick which
    /// accumulator Poseidon trace's digest enters
    /// `state_out.new_acc_digest`.
    pub fn fill_is_base(&mut self, is_base: bool) {
        self.values[self.layout.is_base.offset] = if is_base { F::ONE } else { F::ZERO };
    }

    /// Read the `is_base` bit. Returns `true` if the step is the base
    /// case; `false` for a recursive step.
    pub fn decode_is_base(&self) -> bool {
        match self.values[self.layout.is_base.offset] {
            v if v == F::ONE => true,
            v if v == F::ZERO => false,
            other => panic!("is_base region must be 0 or 1, got {other:?}"),
        }
    }

    /// Splice a one-shot Poseidon trace into its poseidon slot. Drops the
    /// primitive's local constant slot; copies its `values[1..]` into
    /// `self.values[splice_offset..]`.
    pub fn splice_one_shot_poseidon(&mut self, index: usize, trace: &PoseidonTraceImage) {
        let splice_offset = self.layout.one_shot_poseidon_splices[index];
        let expected = self.layout.one_shot_poseidon_layouts[index];
        assert_eq!(
            trace.layout, expected,
            "one-shot Poseidon layout mismatch at index {index}"
        );
        let bits = trace.layout.trace_len;
        self.values[splice_offset..splice_offset + bits].copy_from_slice(&trace.values[1..1 + bits]);
    }

    /// Splice a ring-action pair trace into its ring_action slot.
    pub fn splice_ring_action_pair(&mut self, index: usize, trace: &RingActionTraceImage) {
        let splice_offset = self.layout.ring_action_pair_splices[index];
        assert_eq!(
            trace.layout, self.layout.config.ring_action_pair_layout,
            "ring-action pair layout mismatch at index {index}"
        );
        let bits = trace.layout.end - 1;
        self.values[splice_offset..splice_offset + bits].copy_from_slice(&trace.values[1..1 + bits]);
    }

    /// Splice the F' sponge transcript session into its poseidon slot.
    pub fn splice_sponge_transcript(&mut self, trace: &SpongeTraceImage) {
        let splice_offset = self.layout.sponge_transcript_splice;
        let bits = trace.values.len() - 1;
        assert_eq!(
            bits, self.layout.sponge_transcript_bits,
            "sponge transcript bit count {bits} must match layout config {}",
            self.layout.sponge_transcript_bits
        );
        self.values[splice_offset..splice_offset + bits].copy_from_slice(&trace.values[1..]);
    }

    /// Decode the four-lane digest output of one spliced one-shot
    /// Poseidon trace. Translates the primitive's z-frame to the F'
    /// image's z-frame (primitive `z[k]` for `k ≥ 1` lives at
    /// `self.values[splice_offset + k - 1]`).
    pub fn decode_one_shot_poseidon_digest(&self, index: usize) -> [F; 4] {
        let splice_offset = self.layout.one_shot_poseidon_splices[index];
        let layout = self.layout.one_shot_poseidon_layouts[index];
        let mut out = [F::ZERO; 4];
        for lane in 0..4 {
            let lane_start_self = splice_offset + layout.digest_lane_start(lane) - 1;
            out[lane] = decode_u64_lane(&self.values, lane_start_self);
        }
        out
    }

    /// Decode the D output lanes of one spliced ring-action pair.
    pub fn decode_ring_action_pair_output(&self, index: usize) -> [F; D] {
        let splice_offset = self.layout.ring_action_pair_splices[index];
        let layout = self.layout.config.ring_action_pair_layout;
        let mut out = [F::ZERO; D];
        for m in 0..D {
            let lane_start_self = splice_offset + layout.out_lane_start(m) - 1;
            let mut acc = F::ZERO;
            for i in 0..layout.out_enc.limb_count() {
                let bit = self.values[lane_start_self + i];
                assert!(bit == F::ZERO || bit == F::ONE);
                if bit == F::ONE {
                    acc += layout.out_enc.limb_coef(i);
                }
            }
            out[m] = acc;
        }
        out
    }
}

/// Decode a canonical-`u64` 64-bit lane at the given offset by
/// `Σ 2^i · b_i`.
fn decode_u64_lane(values: &[F], offset: usize) -> F {
    let mut acc = F::ZERO;
    let mut pow = F::ONE;
    for bit in 0..POSEIDON2_GOLDILOCKS_BITS {
        let v = values[offset + bit];
        assert!(v == F::ZERO || v == F::ONE);
        if v == F::ONE {
            acc += pow;
        }
        pow *= F::from_u64(2);
    }
    acc
}

// ── Phase 1.3a — boundary–app_private fill + decode ─────────────────────────────────────

/// state_in state-in view: six four-lane Goldilocks digests carried into one
/// F' step. Each lane is bit-decomposed to 64 canonical bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateIn {
    pub vk_fs_digest: [F; 4],
    pub structure_digest: [F; 4],
    pub z_0: [F; 4],
    pub z_i_in: [F; 4],
    pub acc_digest_in: [F; 4],
    pub public_trace_in: [F; 4],
}

/// state_out state-out view: post-step counters and digests. `new_x_out`
/// shares boundary's `public_x_out_bits` slot and is not stored here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateOut {
    pub new_chunk_count: u64,
    pub new_step_count: u64,
    pub new_z_i: [F; 4],
    pub new_public_trace: [F; 4],
    pub new_acc_digest: [F; 4],
}

/// Write a canonical-Goldilocks 64-bit decomposition of `value` to
/// `values[offset..offset+64]`.
fn write_lane_bits(values: &mut [F], offset: usize, value: F) {
    let v = value.as_canonical_u64();
    for i in 0..POSEIDON2_GOLDILOCKS_BITS {
        values[offset + i] = F::from_u64((v >> i) & 1);
    }
}

/// Write a u64 as 64 little-endian bits.
fn write_u64_bits(values: &mut [F], offset: usize, value: u64) {
    for i in 0..POSEIDON2_GOLDILOCKS_BITS {
        values[offset + i] = F::from_u64((value >> i) & 1);
    }
}

/// Write a four-lane digest's 4·64 bits to `values[offset..offset+256]`.
fn write_digest_bits(values: &mut [F], offset: usize, digest: [F; 4]) {
    for (lane, &v) in digest.iter().enumerate() {
        write_lane_bits(values, offset + lane * POSEIDON2_GOLDILOCKS_BITS, v);
    }
}

/// Read a four-lane digest's bits back via `decode_u64_lane` per lane.
fn read_digest_bits(values: &[F], offset: usize) -> [F; 4] {
    std::array::from_fn(|lane| decode_u64_lane(values, offset + lane * POSEIDON2_GOLDILOCKS_BITS))
}

impl FPrimeImage {
    /// Copy `bits` verbatim into the boundary region. Length must
    /// match `config.boundary_bits`. Each entry must be `{0, 1}`.
    pub fn fill_boundary(&mut self, bits: &[F]) {
        assert_eq!(
            bits.len(),
            self.layout.boundary.bits,
            "boundary bit count must match layout"
        );
        for (i, &b) in bits.iter().enumerate() {
            assert!(b == F::ZERO || b == F::ONE, "boundary bit {i} must be in {{0,1}}");
        }
        let range = self.layout.boundary.offset..self.layout.boundary.end();
        self.values[range].copy_from_slice(bits);
    }

    /// Bit-decompose the 6 state-in digests into state_in.
    pub fn fill_state_in(&mut self, state: &StateIn) {
        let mut cursor = self.layout.state_in.offset;
        let digests = [
            state.vk_fs_digest,
            state.structure_digest,
            state.z_0,
            state.z_i_in,
            state.acc_digest_in,
            state.public_trace_in,
        ];
        for digest in digests {
            write_digest_bits(&mut self.values, cursor, digest);
            cursor += 4 * POSEIDON2_GOLDILOCKS_BITS;
        }
        debug_assert_eq!(cursor, self.layout.state_in.end());
    }

    /// Bit-decompose the post-step counters + digests into state_out.
    pub fn fill_state_out(&mut self, state: &StateOut) {
        let mut cursor = self.layout.state_out.offset;
        write_u64_bits(&mut self.values, cursor, state.new_chunk_count);
        cursor += POSEIDON2_GOLDILOCKS_BITS;
        write_u64_bits(&mut self.values, cursor, state.new_step_count);
        cursor += POSEIDON2_GOLDILOCKS_BITS;
        for digest in [state.new_z_i, state.new_public_trace, state.new_acc_digest] {
            write_digest_bits(&mut self.values, cursor, digest);
            cursor += 4 * POSEIDON2_GOLDILOCKS_BITS;
        }
        debug_assert_eq!(cursor, self.layout.state_out.end());
    }

    /// Bit-decompose the chunk_digest into chunk_digest.
    pub fn fill_chunk_digest(&mut self, chunk_digest: [F; 4]) {
        write_digest_bits(&mut self.values, self.layout.chunk_digest.offset, chunk_digest);
    }

    /// Copy Fibonacci app-private carry bits into app_private. Length must equal
    /// `LIMBS - 1`. Each entry must be `{0, 1}`.
    pub fn fill_app_private(&mut self, carries: &[F]) {
        assert_eq!(
            carries.len(),
            self.layout.app_private.bits,
            "app_private app-private bit count must match LIMBS - 1"
        );
        for (i, &b) in carries.iter().enumerate() {
            assert!(b == F::ZERO || b == F::ONE, "app_private carry {i} must be in {{0,1}}");
        }
        let range = self.layout.app_private.offset..self.layout.app_private.end();
        self.values[range].copy_from_slice(carries);
    }

    /// Copy boundary bits out for parity.
    pub fn decode_boundary(&self) -> Vec<F> {
        self.values[self.layout.boundary.offset..self.layout.boundary.end()].to_vec()
    }

    /// Decode state_in's 6 state-in digests by canonical 64-bit recomposition.
    pub fn decode_state_in(&self) -> StateIn {
        let mut cursor = self.layout.state_in.offset;
        let mut next = || {
            let d = read_digest_bits(&self.values, cursor);
            cursor += 4 * POSEIDON2_GOLDILOCKS_BITS;
            d
        };
        StateIn {
            vk_fs_digest: next(),
            structure_digest: next(),
            z_0: next(),
            z_i_in: next(),
            acc_digest_in: next(),
            public_trace_in: next(),
        }
    }

    /// Decode state_out's post-step counters + digests.
    pub fn decode_state_out(&self) -> StateOut {
        let mut cursor = self.layout.state_out.offset;
        let new_chunk_count = decode_u64_lane(&self.values, cursor).as_canonical_u64();
        cursor += POSEIDON2_GOLDILOCKS_BITS;
        let new_step_count = decode_u64_lane(&self.values, cursor).as_canonical_u64();
        cursor += POSEIDON2_GOLDILOCKS_BITS;
        let new_z_i = read_digest_bits(&self.values, cursor);
        cursor += 4 * POSEIDON2_GOLDILOCKS_BITS;
        let new_public_trace = read_digest_bits(&self.values, cursor);
        cursor += 4 * POSEIDON2_GOLDILOCKS_BITS;
        let new_acc_digest = read_digest_bits(&self.values, cursor);
        StateOut {
            new_chunk_count,
            new_step_count,
            new_z_i,
            new_public_trace,
            new_acc_digest,
        }
    }

    /// Decode chunk_digest's chunk digest.
    pub fn decode_chunk_digest(&self) -> [F; 4] {
        read_digest_bits(&self.values, self.layout.chunk_digest.offset)
    }

    /// Copy app_private carries out for parity.
    pub fn decode_app_private(&self) -> Vec<F> {
        self.values[self.layout.app_private.offset..self.layout.app_private.end()].to_vec()
    }
}

// ── Phase 1.3b — nifs_payloads NIFS payload fill + decode ───────────────────────────

/// Length-header bit width — every count/shape in nifs_payloads is one
/// canonical-u64 lane.
const NIFS_LEN_HEADER_BITS: usize = POSEIDON2_GOLDILOCKS_BITS;
/// One K-extension element = 2 base-field limbs = 2·64 bits.
const NIFS_K_LIMB_BITS: usize = 2 * POSEIDON2_GOLDILOCKS_BITS;
/// `fold_digest: [u8; 32]` encoded as `[F; 4]` × 64 bits = 256 bits.
const NIFS_FOLD_DIGEST_BITS: usize = 4 * POSEIDON2_GOLDILOCKS_BITS;

/// nifs_payloads view of one fresh `CcsClaim` payload. Mirrors
/// `paper::digest::ccs_claim_digest`'s preimage shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NifsCcsClaimView {
    /// `Commitment::d` (`usize` in production; encoded as `u64` here).
    pub d: u64,
    /// `Commitment::kappa` (same encoding contract).
    pub kappa: u64,
    pub c_data: Vec<F>,
    pub x: Vec<F>,
    pub m_in: u64,
}

/// Shape of one fresh CcsClaim payload — the sizes the encoder/decoder
/// agree on. Computed from a real `CcsClaim` or set explicitly by the
/// caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NifsCcsClaimShape {
    pub c_data_entries: usize,
    pub x_entries: usize,
}

impl NifsCcsClaimShape {
    pub fn bits(&self) -> usize {
        // d + kappa + c_data_len + c_data + x_len + x + m_in
        2 * NIFS_LEN_HEADER_BITS
            + NIFS_LEN_HEADER_BITS
            + self.c_data_entries * POSEIDON2_GOLDILOCKS_BITS
            + NIFS_LEN_HEADER_BITS
            + self.x_entries * POSEIDON2_GOLDILOCKS_BITS
            + NIFS_LEN_HEADER_BITS
    }
}

/// nifs_payloads view of one CE claim payload. Covers commitment, active `X`,
/// evaluation point `r`, `y_ring`, `y_zcol`, `s_col`, `m_in`, and
/// `fold_digest`. Encoding order mirrors `ce_claim_digest`'s preimage
/// for the FS-bound subset (`... y_ring, m_in, fold_digest`), then
/// appends `y_zcol` and `s_col` (which are part of the CeClaim but not
/// in the FS-bound digest).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NifsCeClaimView {
    pub d: u64,
    pub kappa: u64,
    pub c_data: Vec<F>,
    pub x_rows: u64,
    pub x_cols: u64,
    pub x_active_cols: u64,
    /// `x_rows × x_active_cols` F values, row-major.
    pub x_active_flat: Vec<F>,
    /// K-extension `r` as `[c0, c1]` F pairs.
    pub r: Vec<[F; 2]>,
    /// `y_ring[row][col]` as K-element pairs. Each inner Vec may have a
    /// different length (per `y_ring_inner_lens` in the shape).
    pub y_ring: Vec<Vec<[F; 2]>>,
    pub y_zcol: Vec<[F; 2]>,
    pub s_col: Vec<[F; 2]>,
    pub m_in: u64,
    /// `digest32_as_fields(fold_digest)` — four F values, each 64 bits.
    pub fold_digest_fields: [F; 4],
}

/// Shape of one CE claim payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NifsCeClaimShape {
    pub c_data_entries: usize,
    pub x_rows: usize,
    pub x_active_cols: usize,
    pub r_len: usize,
    pub y_ring_inner_lens: Vec<usize>,
    pub y_zcol_len: usize,
    pub s_col_len: usize,
}

impl NifsCeClaimShape {
    pub fn bits(&self) -> usize {
        let mut total = 0;
        total += 2 * NIFS_LEN_HEADER_BITS; // d, kappa
        total += NIFS_LEN_HEADER_BITS + self.c_data_entries * POSEIDON2_GOLDILOCKS_BITS; // c_data
        total += 3 * NIFS_LEN_HEADER_BITS; // x_rows, x_cols, x_active_cols
        total += self.x_rows * self.x_active_cols * POSEIDON2_GOLDILOCKS_BITS; // X entries
        total += NIFS_LEN_HEADER_BITS + self.r_len * NIFS_K_LIMB_BITS; // r
        total += NIFS_LEN_HEADER_BITS; // y_ring outer
        for &inner in &self.y_ring_inner_lens {
            total += NIFS_LEN_HEADER_BITS + inner * NIFS_K_LIMB_BITS;
        }
        total += NIFS_LEN_HEADER_BITS; // m_in
        total += NIFS_FOLD_DIGEST_BITS; // fold_digest
        total += NIFS_LEN_HEADER_BITS + self.y_zcol_len * NIFS_K_LIMB_BITS; // y_zcol
        total += NIFS_LEN_HEADER_BITS + self.s_col_len * NIFS_K_LIMB_BITS; // s_col
        total
    }
}

impl FPrimeImage {
    /// Encode a fresh `CcsClaim` payload starting at `nifs_offset` (a
    /// nifs_payloads-relative offset). Returns the next free nifs_payloads-relative offset.
    pub fn fill_nifs_ccs_claim_at(&mut self, nifs_offset: usize, view: &NifsCcsClaimView) -> usize {
        let shape = NifsCcsClaimShape {
            c_data_entries: view.c_data.len(),
            x_entries: view.x.len(),
        };
        let total = shape.bits();
        assert!(
            nifs_offset + total <= self.layout.nifs_payloads.bits,
            "nifs_payloads CcsClaim payload at offset {nifs_offset} ({total} bits) overflows region ({} bits)",
            self.layout.nifs_payloads.bits,
        );

        let mut cursor = self.layout.nifs_payloads.offset + nifs_offset;
        write_u64_bits(&mut self.values, cursor, view.d as u64);
        cursor += NIFS_LEN_HEADER_BITS;
        write_u64_bits(&mut self.values, cursor, view.kappa as u64);
        cursor += NIFS_LEN_HEADER_BITS;
        write_u64_bits(&mut self.values, cursor, view.c_data.len() as u64);
        cursor += NIFS_LEN_HEADER_BITS;
        for &v in &view.c_data {
            write_lane_bits(&mut self.values, cursor, v);
            cursor += POSEIDON2_GOLDILOCKS_BITS;
        }
        write_u64_bits(&mut self.values, cursor, view.x.len() as u64);
        cursor += NIFS_LEN_HEADER_BITS;
        for &v in &view.x {
            write_lane_bits(&mut self.values, cursor, v);
            cursor += POSEIDON2_GOLDILOCKS_BITS;
        }
        write_u64_bits(&mut self.values, cursor, view.m_in);
        cursor += NIFS_LEN_HEADER_BITS;

        debug_assert_eq!(cursor, self.layout.nifs_payloads.offset + nifs_offset + total);
        nifs_offset + total
    }

    /// Decode a fresh `CcsClaim` payload from `nifs_offset` using `shape`
    /// to size the variable-length fields.
    pub fn decode_nifs_ccs_claim_at(&self, nifs_offset: usize, shape: &NifsCcsClaimShape) -> NifsCcsClaimView {
        let mut cursor = self.layout.nifs_payloads.offset + nifs_offset;
        let d = decode_u64_lane(&self.values, cursor).as_canonical_u64();
        cursor += NIFS_LEN_HEADER_BITS;
        let kappa = decode_u64_lane(&self.values, cursor).as_canonical_u64();
        cursor += NIFS_LEN_HEADER_BITS;
        let c_data_len = decode_u64_lane(&self.values, cursor).as_canonical_u64() as usize;
        cursor += NIFS_LEN_HEADER_BITS;
        assert_eq!(
            c_data_len, shape.c_data_entries,
            "nifs_payloads CcsClaim c_data len mismatch"
        );
        let c_data: Vec<F> = (0..c_data_len)
            .map(|i| decode_u64_lane(&self.values, cursor + i * POSEIDON2_GOLDILOCKS_BITS))
            .collect();
        cursor += c_data_len * POSEIDON2_GOLDILOCKS_BITS;
        let x_len = decode_u64_lane(&self.values, cursor).as_canonical_u64() as usize;
        cursor += NIFS_LEN_HEADER_BITS;
        assert_eq!(x_len, shape.x_entries, "nifs_payloads CcsClaim x len mismatch");
        let x: Vec<F> = (0..x_len)
            .map(|i| decode_u64_lane(&self.values, cursor + i * POSEIDON2_GOLDILOCKS_BITS))
            .collect();
        cursor += x_len * POSEIDON2_GOLDILOCKS_BITS;
        let m_in = decode_u64_lane(&self.values, cursor).as_canonical_u64();
        NifsCcsClaimView {
            d,
            kappa,
            c_data,
            x,
            m_in,
        }
    }

    /// Encode one CE claim payload starting at `nifs_offset`. Returns the
    /// next free nifs_payloads-relative offset.
    pub fn fill_nifs_ce_claim_at(&mut self, nifs_offset: usize, view: &NifsCeClaimView) -> usize {
        let shape = NifsCeClaimShape {
            c_data_entries: view.c_data.len(),
            x_rows: view.x_rows as usize,
            x_active_cols: view.x_active_cols as usize,
            r_len: view.r.len(),
            y_ring_inner_lens: view.y_ring.iter().map(|row| row.len()).collect(),
            y_zcol_len: view.y_zcol.len(),
            s_col_len: view.s_col.len(),
        };
        let total = shape.bits();
        assert!(
            nifs_offset + total <= self.layout.nifs_payloads.bits,
            "nifs_payloads CeClaim payload at offset {nifs_offset} ({total} bits) overflows region ({} bits)",
            self.layout.nifs_payloads.bits,
        );

        let mut cursor = self.layout.nifs_payloads.offset + nifs_offset;
        write_u64_bits(&mut self.values, cursor, view.d as u64);
        cursor += NIFS_LEN_HEADER_BITS;
        write_u64_bits(&mut self.values, cursor, view.kappa as u64);
        cursor += NIFS_LEN_HEADER_BITS;
        write_u64_bits(&mut self.values, cursor, view.c_data.len() as u64);
        cursor += NIFS_LEN_HEADER_BITS;
        for &v in &view.c_data {
            write_lane_bits(&mut self.values, cursor, v);
            cursor += POSEIDON2_GOLDILOCKS_BITS;
        }
        write_u64_bits(&mut self.values, cursor, view.x_rows);
        cursor += NIFS_LEN_HEADER_BITS;
        write_u64_bits(&mut self.values, cursor, view.x_cols);
        cursor += NIFS_LEN_HEADER_BITS;
        write_u64_bits(&mut self.values, cursor, view.x_active_cols);
        cursor += NIFS_LEN_HEADER_BITS;
        assert_eq!(
            view.x_active_flat.len(),
            (view.x_rows * view.x_active_cols) as usize,
            "nifs_payloads CeClaim x_active_flat length must match x_rows × x_active_cols"
        );
        for &v in &view.x_active_flat {
            write_lane_bits(&mut self.values, cursor, v);
            cursor += POSEIDON2_GOLDILOCKS_BITS;
        }
        write_u64_bits(&mut self.values, cursor, view.r.len() as u64);
        cursor += NIFS_LEN_HEADER_BITS;
        for k in &view.r {
            write_lane_bits(&mut self.values, cursor, k[0]);
            cursor += POSEIDON2_GOLDILOCKS_BITS;
            write_lane_bits(&mut self.values, cursor, k[1]);
            cursor += POSEIDON2_GOLDILOCKS_BITS;
        }
        write_u64_bits(&mut self.values, cursor, view.y_ring.len() as u64);
        cursor += NIFS_LEN_HEADER_BITS;
        for row in &view.y_ring {
            write_u64_bits(&mut self.values, cursor, row.len() as u64);
            cursor += NIFS_LEN_HEADER_BITS;
            for k in row {
                write_lane_bits(&mut self.values, cursor, k[0]);
                cursor += POSEIDON2_GOLDILOCKS_BITS;
                write_lane_bits(&mut self.values, cursor, k[1]);
                cursor += POSEIDON2_GOLDILOCKS_BITS;
            }
        }
        // FS-bound prefix ends here: production `ce_claim_digest` puts
        // `m_in` and `fold_digest` immediately after `y_ring`. Keep that
        // exact order so the first N bits of nifs_payloads round-trip back to the
        // production preimage prefix.
        write_u64_bits(&mut self.values, cursor, view.m_in);
        cursor += NIFS_LEN_HEADER_BITS;
        write_digest_bits(&mut self.values, cursor, view.fold_digest_fields);
        cursor += NIFS_FOLD_DIGEST_BITS;
        // y_zcol and s_col are NOT in the FS-bound digest; appended as
        // the non-FS-bound tail.
        write_u64_bits(&mut self.values, cursor, view.y_zcol.len() as u64);
        cursor += NIFS_LEN_HEADER_BITS;
        for k in &view.y_zcol {
            write_lane_bits(&mut self.values, cursor, k[0]);
            cursor += POSEIDON2_GOLDILOCKS_BITS;
            write_lane_bits(&mut self.values, cursor, k[1]);
            cursor += POSEIDON2_GOLDILOCKS_BITS;
        }
        write_u64_bits(&mut self.values, cursor, view.s_col.len() as u64);
        cursor += NIFS_LEN_HEADER_BITS;
        for k in &view.s_col {
            write_lane_bits(&mut self.values, cursor, k[0]);
            cursor += POSEIDON2_GOLDILOCKS_BITS;
            write_lane_bits(&mut self.values, cursor, k[1]);
            cursor += POSEIDON2_GOLDILOCKS_BITS;
        }

        debug_assert_eq!(cursor, self.layout.nifs_payloads.offset + nifs_offset + total);
        nifs_offset + total
    }

    /// Decode one CE claim payload from `nifs_offset` using `shape` to
    /// size the variable-length fields.
    pub fn decode_nifs_ce_claim_at(&self, nifs_offset: usize, shape: &NifsCeClaimShape) -> NifsCeClaimView {
        let mut cursor = self.layout.nifs_payloads.offset + nifs_offset;
        let d = decode_u64_lane(&self.values, cursor).as_canonical_u64();
        cursor += NIFS_LEN_HEADER_BITS;
        let kappa = decode_u64_lane(&self.values, cursor).as_canonical_u64();
        cursor += NIFS_LEN_HEADER_BITS;
        let c_data_len = decode_u64_lane(&self.values, cursor).as_canonical_u64() as usize;
        cursor += NIFS_LEN_HEADER_BITS;
        assert_eq!(c_data_len, shape.c_data_entries, "nifs_payloads CeClaim c_data len");
        let c_data: Vec<F> = (0..c_data_len)
            .map(|i| decode_u64_lane(&self.values, cursor + i * POSEIDON2_GOLDILOCKS_BITS))
            .collect();
        cursor += c_data_len * POSEIDON2_GOLDILOCKS_BITS;
        let x_rows = decode_u64_lane(&self.values, cursor).as_canonical_u64();
        cursor += NIFS_LEN_HEADER_BITS;
        let x_cols = decode_u64_lane(&self.values, cursor).as_canonical_u64();
        cursor += NIFS_LEN_HEADER_BITS;
        let x_active_cols = decode_u64_lane(&self.values, cursor).as_canonical_u64();
        cursor += NIFS_LEN_HEADER_BITS;
        assert_eq!(x_rows as usize, shape.x_rows, "nifs_payloads CeClaim x_rows");
        assert_eq!(
            x_active_cols as usize, shape.x_active_cols,
            "nifs_payloads CeClaim x_active_cols"
        );
        let x_count = (x_rows * x_active_cols) as usize;
        let x_active_flat: Vec<F> = (0..x_count)
            .map(|i| decode_u64_lane(&self.values, cursor + i * POSEIDON2_GOLDILOCKS_BITS))
            .collect();
        cursor += x_count * POSEIDON2_GOLDILOCKS_BITS;
        let r_len = decode_u64_lane(&self.values, cursor).as_canonical_u64() as usize;
        cursor += NIFS_LEN_HEADER_BITS;
        assert_eq!(r_len, shape.r_len, "nifs_payloads CeClaim r_len");
        let r: Vec<[F; 2]> = (0..r_len)
            .map(|i| {
                let base = cursor + i * NIFS_K_LIMB_BITS;
                let c0 = decode_u64_lane(&self.values, base);
                let c1 = decode_u64_lane(&self.values, base + POSEIDON2_GOLDILOCKS_BITS);
                [c0, c1]
            })
            .collect();
        cursor += r_len * NIFS_K_LIMB_BITS;
        let y_ring_outer = decode_u64_lane(&self.values, cursor).as_canonical_u64() as usize;
        cursor += NIFS_LEN_HEADER_BITS;
        assert_eq!(
            y_ring_outer,
            shape.y_ring_inner_lens.len(),
            "nifs_payloads CeClaim y_ring outer"
        );
        let mut y_ring: Vec<Vec<[F; 2]>> = Vec::with_capacity(y_ring_outer);
        for &expected_inner in &shape.y_ring_inner_lens {
            let inner = decode_u64_lane(&self.values, cursor).as_canonical_u64() as usize;
            cursor += NIFS_LEN_HEADER_BITS;
            assert_eq!(inner, expected_inner, "nifs_payloads CeClaim y_ring inner");
            let row: Vec<[F; 2]> = (0..inner)
                .map(|i| {
                    let base = cursor + i * NIFS_K_LIMB_BITS;
                    [
                        decode_u64_lane(&self.values, base),
                        decode_u64_lane(&self.values, base + POSEIDON2_GOLDILOCKS_BITS),
                    ]
                })
                .collect();
            cursor += inner * NIFS_K_LIMB_BITS;
            y_ring.push(row);
        }
        // FS-bound prefix: read `m_in` + `fold_digest` immediately after
        // `y_ring`, mirroring `ce_claim_digest`'s preimage order.
        let m_in = decode_u64_lane(&self.values, cursor).as_canonical_u64();
        cursor += NIFS_LEN_HEADER_BITS;
        let fold_digest_fields = read_digest_bits(&self.values, cursor);
        cursor += NIFS_FOLD_DIGEST_BITS;
        // y_zcol and s_col are the non-FS-bound tail.
        let y_zcol_len = decode_u64_lane(&self.values, cursor).as_canonical_u64() as usize;
        cursor += NIFS_LEN_HEADER_BITS;
        assert_eq!(y_zcol_len, shape.y_zcol_len, "nifs_payloads CeClaim y_zcol_len");
        let y_zcol: Vec<[F; 2]> = (0..y_zcol_len)
            .map(|i| {
                let base = cursor + i * NIFS_K_LIMB_BITS;
                [
                    decode_u64_lane(&self.values, base),
                    decode_u64_lane(&self.values, base + POSEIDON2_GOLDILOCKS_BITS),
                ]
            })
            .collect();
        cursor += y_zcol_len * NIFS_K_LIMB_BITS;
        let s_col_len = decode_u64_lane(&self.values, cursor).as_canonical_u64() as usize;
        cursor += NIFS_LEN_HEADER_BITS;
        assert_eq!(s_col_len, shape.s_col_len, "nifs_payloads CeClaim s_col_len");
        let s_col: Vec<[F; 2]> = (0..s_col_len)
            .map(|i| {
                let base = cursor + i * NIFS_K_LIMB_BITS;
                [
                    decode_u64_lane(&self.values, base),
                    decode_u64_lane(&self.values, base + POSEIDON2_GOLDILOCKS_BITS),
                ]
            })
            .collect();
        NifsCeClaimView {
            d,
            kappa,
            c_data,
            x_rows,
            x_cols,
            x_active_cols,
            x_active_flat,
            r,
            y_ring,
            y_zcol,
            s_col,
            m_in,
            fold_digest_fields,
        }
    }
}

// ── Phase 1.3c — kmul K-mul fill + decode ──────────────────────────────────

/// kmul view of one Karatsuba K-mul intermediate set. Each intermediate
/// occupies a K-word slot (2 × 64 bits = 128 bits); a full K-mul
/// captures three intermediates `p`, `q`, `r` for a total of
/// [`KMUL_SLOT_BITS`] = 384 bits.
///
/// Karatsuba semantics: given K-element inputs `a = (a0, a1)` and
/// `b = (b0, b1)`, the intermediates are `p = a0 · b0`, `q = a1 · b1`,
/// `r = (a0 + a1) · (b0 + b1)`, and the K-product reduces to
/// `(p + W · q, r − p − q)`. In the F-valued K-mul case the high half
/// of each pair is unused (zero); the slot is sized two-wide to leave
/// room for future K-mul shapes (e.g., recursive K extensions). The
/// layout/decoder treat both halves uniformly as canonical 64-bit F
/// values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KMulView {
    pub p: [F; 2],
    pub q: [F; 2],
    pub r: [F; 2],
}

impl FPrimeImage {
    /// Encode one K-mul slot at `index` (must be `< kmul_count`).
    pub fn fill_kmul_at(&mut self, index: usize, view: &KMulView) {
        assert!(
            index < self.layout.config.kmul_count,
            "kmul K-mul index {index} out of range (kmul_count = {})",
            self.layout.config.kmul_count,
        );
        let base = self.layout.kmul.offset + index * KMUL_SLOT_BITS;
        let pair_stride = 2 * POSEIDON2_GOLDILOCKS_BITS;
        for (pair_idx, pair) in [view.p, view.q, view.r].iter().enumerate() {
            let pair_offset = base + pair_idx * pair_stride;
            write_lane_bits(&mut self.values, pair_offset, pair[0]);
            write_lane_bits(&mut self.values, pair_offset + POSEIDON2_GOLDILOCKS_BITS, pair[1]);
        }
    }

    /// Decode one K-mul slot at `index`.
    pub fn decode_kmul_at(&self, index: usize) -> KMulView {
        assert!(
            index < self.layout.config.kmul_count,
            "kmul K-mul index {index} out of range (kmul_count = {})",
            self.layout.config.kmul_count,
        );
        let base = self.layout.kmul.offset + index * KMUL_SLOT_BITS;
        let pair_stride = 2 * POSEIDON2_GOLDILOCKS_BITS;
        let read_pair = |pair_idx: usize| {
            let offset = base + pair_idx * pair_stride;
            [
                decode_u64_lane(&self.values, offset),
                decode_u64_lane(&self.values, offset + POSEIDON2_GOLDILOCKS_BITS),
            ]
        };
        KMulView {
            p: read_pair(0),
            q: read_pair(1),
            r: read_pair(2),
        }
    }

    /// Fill all kmul K-mul slots. `views.len()` must equal
    /// `config.kmul_count`.
    pub fn fill_all_kmul(&mut self, views: &[KMulView]) {
        assert_eq!(
            views.len(),
            self.layout.config.kmul_count,
            "kmul K-mul view count must equal kmul_count"
        );
        for (i, view) in views.iter().enumerate() {
            self.fill_kmul_at(i, view);
        }
    }

    /// Decode all kmul K-mul slots.
    pub fn decode_kmul_all(&self) -> Vec<KMulView> {
        (0..self.layout.config.kmul_count)
            .map(|i| self.decode_kmul_at(i))
            .collect()
    }
}
