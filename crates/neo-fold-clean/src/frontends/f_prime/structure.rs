//! App-agnostic CCS structure for one `enc(F')` step.
//!
//! The image layout's regions are owned by
//! [`crate::frontends::f_prime::image`]; for quick reference:
//!
//! | Region | Holds |
//! |---|---|
//! | boundary (`boundary`)         | public-IO bits — `enc_inst(x_out)`, `enc_inst(prior_x_out)`, counter words |
//! | state_in (`state_in`)         | six four-lane state-in digests (vk_fs, structure, z_0, z_i_in, acc_digest_in, public_trace_in) |
//! | state_out (`state_out`)        | two u64 counters + three four-lane post-step digests (new_z_i, new_public_trace, new_acc_digest) |
//! | chunk_digest (`chunk_digest`)     | one four-lane chunk digest |
//! | app_private (`app_private`)      | app-private carry bits (Fibonacci witness or R1CS bit-decomposed assignment) |
//! | nifs_payloads (`nifs_payloads`)    | NIFS CcsClaim / CeClaim payloads (parent_authority etc.) |
//! | kmul (`kmul`)             | K-mul Karatsuba intermediates (one slot per K-mul invocation) |
//! | ring_action (`ring_action`)      | ring-action pair traces (ρ, c, products, output per pair) |
//! | poseidon (`poseidon`)         | one-shot Poseidon traces + the F' sponge transcript |
//!
//! This file owns:
//! - **Bit validity**: `z[k] · (z[k] − 1) = 0` for every committed bit.
//! - **Ring-action products** in ring_action: `ρ[i] · c[j] = prod[i][j]`.
//! - **Ring-action outputs** in ring_action: `out[m] = Σ Φ_TABLE[i+j][m] · prod[i][j]`.
//! - **One-shot trace digest ↔ state-out digest** binding (poseidon → state_out).
//! - **One-shot trace digest ↔ public-x_out** binding (poseidon → boundary).
//! - The carrier `CcsStructure` (`m`, `n`, matrices, polynomial `f`).
//!
//! ## Layout
//!
//! Strict SuperNeo low-norm shape: every committed coordinate is in
//! `{0, 1}` except the constant slot `z[0] = 1`. Canonical-u64 lanes are
//! **not** materialised as fresh witness columns — every constraint that
//! reads a lane substitutes `Σ_{i<64} 2^i · z[bit_start + i]` (or its
//! 32-bit half variants) directly.
//!
//! - **Witness columns** (`m = layout.end`):
//!     - `z[0]              = F::ONE`              (CCS constant slot)
//!     - `z[1..layout.end]` = the committed image bits (norm 1).
//! - **Rows** (`n` = bit-validity + product + output + binding rows):
//!     - bit-validity rows                 (one per committed bit)
//!     - ring-action product rows          (D² per ring_action pair)
//!     - ring-action output rows           (D per ring_action pair)
//!     - state-out digest binding rows     (`POSEIDON2_DIGEST_LEN` per binding)
//!     - public-x_out binding rows         (`POSEIDON2_DIGEST_LEN` per binding)
//!
//! ## Out of scope
//!
//! - Functional app_private constraints.
//! - Remaining functional F' relations (e.g., `new_chunk_count == chunk_count_in + 1`).
//! - Lifecycle wiring: `preprocess_seeded` still folds the bit-carrier R1CS;
//!   nothing here is plumbed into the prover flow yet.
//!
//! ## Encoding contract
//!
//! Every lane this module references is a canonical-u64 lane (64 bits
//! per lane, coefficient `2^i` for bit `i`). For ring_action this requires the
//! [`RingActionTraceLayout`] to use [`LowNormEncoding::U64`] on all four
//! subregions; [`f_prime_lane_slots`] panics otherwise.
//!
//! ## Production count pinning
//!
//! [`PRODUCTION_KMUL_COUNT`] / [`PRODUCTION_RING_ACTION_PAIR_COUNT`]
//! are pinned to what the Phase 1.3d coverage gate observed the F' R1CS
//! emitter actually invoking per recursive step. A future emitter change
//! must update these constants in lock-step or
//! `phase_1_4a_production_config_pins_emitter_counts` regresses.

use neo_ccs::{CcsMatrix, CcsStructure, CscMat, SparsePoly, Term};
use neo_math::ring::D;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::ccs_native::poseidon2::{POSEIDON2_DIGEST_LEN, POSEIDON2_GOLDILOCKS_BITS, POSEIDON2_WIDTH};
use crate::engine::r1cs_circuit::ring_action::phi_reduction_coeff;
use crate::frontends::f_prime::image::{
    FPrimeImage, FPrimeImageConfig, FPrimeImageLayout, NifsPayloadShape, PoseidonPreimageLaneSource,
    StateOutDigestTarget,
};
use crate::paper::f_prime::ring_action_trace::{LowNormEncoding, RingActionTraceLayout};

/// Number of state_in four-lane digests (vk_fs, structure, z_0, z_i_in, acc, public_trace).
const STATE_IN_DIGEST_COUNT: usize = 6;
/// Number of state_out u64 counters (new chunk_count, new step_count).
const STATE_OUT_COUNTER_COUNT: usize = 2;
/// Number of state_out four-lane digests (new z_i, new public_trace, new acc).
const STATE_OUT_DIGEST_COUNT: usize = 3;
/// chunk_digest holds one chunk_digest (four lanes).
const CHUNK_DIGEST_LANE_COUNT: usize = 4;
/// kmul K-mul slot: 3 K-pairs `(p, q, r)`, each pair has low + high lane.
const KMUL_LANES_PER_SLOT: usize = 6;
/// kmul K-mul slot bit width: 6 lanes × 64 bits.
const KMUL_SLOT_BITS: usize = KMUL_LANES_PER_SLOT * POSEIDON2_GOLDILOCKS_BITS;
/// ring_action ring-action pair lanes: D ρ + D c + D² prod + D out.
const RING_ACTION_LANES_PER_PAIR: usize = 3 * D + D * D;
/// ring_action product matrix lanes per ring-action pair.
const RING_ACTION_PRODUCT_LANES_PER_PAIR: usize = D * D;
/// ring_action output lanes per ring-action pair.
const RING_ACTION_OUTPUT_LANES_PER_PAIR: usize = D;

/// kmul K-mul invocations per F' recursive step. Pinned by the Phase 1.3d
/// coverage gate's measurement of the actual `enforce_k_mul_with_intermediates`
/// call count.
pub const PRODUCTION_KMUL_COUNT: usize = 7100;

/// ring_action ring-action pair invocations per F' recursive step. Pinned by the
/// Phase 1.3d coverage gate's measurement.
pub const PRODUCTION_RING_ACTION_PAIR_COUNT: usize = 465;

/// The currently pinned production **kmul/ring_action shell** configuration.
///
/// This is not the final full F' image configuration: boundary/nifs_payloads/poseidon remain
/// zero-sized until later slices wire their functional regions into the
/// structure. What it does pin today is the production kmul/ring_action audit count
/// and the U64 ring-action encoding choice.
pub fn production_kmul_ring_action_shell_image_config() -> FPrimeImageConfig {
    FPrimeImageConfig {
        limbs: 3,
        boundary_bits: 0,
        nifs_payload_shapes: vec![],
        kmul_count: PRODUCTION_KMUL_COUNT,
        ring_action_pair_count: PRODUCTION_RING_ACTION_PAIR_COUNT,
        ring_action_pair_layout: RingActionTraceLayout::new(
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
            LowNormEncoding::U64,
        ),
        poseidon_one_shot_preimage_lens: vec![],
        sponge_transcript_permutes: 0,
        one_shot_digest_to_state_out_bindings: vec![],
        one_shot_digest_to_public_x_out_bindings: vec![],
        poseidon_transition_enforcements: vec![],
        unified_accumulator_selector: None,
    }
}

/// One canonical-u64 lane: a 64-bit window inside the image's `values`.
///
/// Constraints that need the lane's recomposed F-value substitute
/// `Σ_{i<64} 2^i · z[bit_start + i]` (see [`lane_terms`]); no fresh
/// witness column is allocated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneSlot {
    pub bit_start: usize,
}

/// Linear combination that recomposes a canonical-u64 lane from its 64
/// committed bits: `Σ_{i<64} 2^i · z[bit_start + i]`.
pub(crate) fn lane_terms(slot: LaneSlot) -> impl Iterator<Item = (usize, F)> {
    (0..POSEIDON2_GOLDILOCKS_BITS).map(move |i| (slot.bit_start + i, F::from_u64(1u64 << i)))
}

/// Linear combination that recomposes a lane's low 32 bits.
fn lane_low_half_terms(slot: LaneSlot) -> impl Iterator<Item = (usize, F)> {
    (0..32).map(move |i| (slot.bit_start + i, F::from_u64(1u64 << i)))
}

/// Linear combination that recomposes a lane's high 32 bits.
fn lane_high_half_terms(slot: LaneSlot) -> impl Iterator<Item = (usize, F)> {
    (0..32).map(move |i| (slot.bit_start + 32 + i, F::from_u64(1u64 << i)))
}

/// `coeff · lane_terms(slot)` — used when a lane appears with a non-unit
/// coefficient in a linear combination (e.g., the ring_action output sum).
fn scaled_lane_terms(slot: LaneSlot, coeff: F) -> impl Iterator<Item = (usize, F)> {
    lane_terms(slot).map(move |(col, c)| (col, c * coeff))
}

/// All lane-decode positions, grouped by region.
///
/// Decoded columns appear in the extended witness `z` in this order:
/// `state_lanes`, `nifs_payload_lanes` (per payload), `kmul_lanes`,
/// `ring_action_lanes`, `poseidon_trace_lanes` (per trace),
/// `sponge_transcript_lanes`, `public_x_out_binding_lanes` (per binding).
/// Tests can access each region's slots directly via its field.
#[derive(Clone, Debug)]
pub struct FPrimeLaneSlots {
    pub state_lanes: Vec<LaneSlot>,
    /// One inner `Vec<LaneSlot>` per spliced nifs_payloads NIFS payload. Each
    /// inner Vec enumerates every 64-bit lane in the payload's fill
    /// order (u64 counters, F values, K-element halves).
    pub nifs_payload_lanes: Vec<Vec<LaneSlot>>,
    pub kmul_lanes: Vec<LaneSlot>,
    pub ring_action_lanes: Vec<LaneSlot>,
    /// One inner `Vec<LaneSlot>` per spliced one-shot Poseidon trace.
    /// Each inner Vec enumerates the trace's `trace_len / 64` lanes in
    /// trace order.
    pub poseidon_trace_lanes: Vec<Vec<LaneSlot>>,
    /// One canonical-u64 lane per 64-bit word in the poseidon sponge transcript.
    pub sponge_transcript_lanes: Vec<LaneSlot>,
    /// boundary digest lanes pulled in by each [`OneShotDigestToPublicXOutBinding`]. Each
    /// outer entry is one binding's 4 digest lanes; allocation order
    /// matches `config.one_shot_digest_to_public_x_out_bindings`.
    pub public_x_out_binding_lanes: Vec<[LaneSlot; 4]>,
    /// One canonical-u64 lane per app-assignment variable when
    /// `app_private` is laid out as 64-bit lanes (R1CS frontends). For
    /// frontends whose `app_private` region is not a multiple of 64 bits
    /// (Fibonacci carries), this Vec is empty.
    pub app_assignment_lanes: Vec<LaneSlot>,
}

impl FPrimeLaneSlots {
    pub fn total(&self) -> usize {
        self.state_lanes.len()
            + self.nifs_payload_lanes.iter().map(Vec::len).sum::<usize>()
            + self.kmul_lanes.len()
            + self.ring_action_lanes.len()
            + self
                .poseidon_trace_lanes
                .iter()
                .map(Vec::len)
                .sum::<usize>()
            + self.sponge_transcript_lanes.len()
            + self.public_x_out_binding_lanes.len() * 4
            + self.app_assignment_lanes.len()
    }
}

/// Enumerate every canonical-u64 lane the structure references, in fixed
/// order. Lanes only describe 64-bit windows into `image.values`; the
/// structure substitutes `Σ 2^i · bit` directly where it needs the lane's
/// recomposed value, so no fresh witness columns are minted here.
pub fn f_prime_lane_slots(layout: &FPrimeImageLayout) -> FPrimeLaneSlots {
    FPrimeLaneSlots {
        state_lanes: collect_state_lane_slots(layout),
        nifs_payload_lanes: collect_nifs_payload_slots(layout),
        kmul_lanes: collect_kmul_slots(layout),
        ring_action_lanes: collect_ring_action_slots(layout),
        poseidon_trace_lanes: collect_poseidon_trace_slots(layout),
        sponge_transcript_lanes: collect_sponge_transcript_slots(layout),
        public_x_out_binding_lanes: collect_public_x_out_binding_slots(layout),
        app_assignment_lanes: collect_app_assignment_lane_slots(layout),
    }
}

/// Enumerate `app_private` as 64-bit canonical-u64 lanes (one lane per
/// app-assignment variable). Used by app frontends that bit-decompose
/// the R1CS assignment into `app_private`. Returns an empty Vec when
/// the region size isn't a multiple of 64 (e.g. Fibonacci's 2-bit
/// carries).
fn collect_app_assignment_lane_slots(layout: &FPrimeImageLayout) -> Vec<LaneSlot> {
    let bits = layout.app_private.bits;
    if bits == 0 || bits % POSEIDON2_GOLDILOCKS_BITS != 0 {
        return Vec::new();
    }
    let lane_count = bits / POSEIDON2_GOLDILOCKS_BITS;
    let base = layout.app_private.offset;
    (0..lane_count)
        .map(|i| LaneSlot {
            bit_start: base + i * POSEIDON2_GOLDILOCKS_BITS,
        })
        .collect()
}

fn collect_public_x_out_binding_slots(layout: &FPrimeImageLayout) -> Vec<[LaneSlot; 4]> {
    let boundary_start = layout.boundary.offset;
    let boundary_end = layout.boundary.end();
    let mut all = Vec::with_capacity(layout.config.one_shot_digest_to_public_x_out_bindings.len());
    for binding in &layout.config.one_shot_digest_to_public_x_out_bindings {
        let lanes: [LaneSlot; 4] = std::array::from_fn(|m| {
            let bit_start = binding.public_x_out_lane_bit_starts[m];
            assert!(
                bit_start >= boundary_start && bit_start + POSEIDON2_GOLDILOCKS_BITS <= boundary_end,
                "poseidon↔boundary binding lane {m} bit_start={bit_start} (64-bit lane) outside boundary [{boundary_start}, {boundary_end})"
            );
            LaneSlot { bit_start }
        });
        all.push(lanes);
    }
    all
}

fn collect_nifs_payload_slots(layout: &FPrimeImageLayout) -> Vec<Vec<LaneSlot>> {
    let mut payloads = Vec::with_capacity(layout.config.nifs_payload_shapes.len());
    for (shape, &payload_offset) in layout
        .config
        .nifs_payload_shapes
        .iter()
        .zip(layout.nifs_payload_offsets.iter())
    {
        let mut slots = Vec::new();
        let mut cursor = payload_offset;
        match shape {
            NifsPayloadShape::CcsClaim(s) => {
                // Fill order (per fill_nifs_ccs_claim_at):
                //   d, kappa, c_data_len, c_data[..], x_len, x[..], m_in.
                push_u64_lane(&mut slots, &mut cursor); // d
                push_u64_lane(&mut slots, &mut cursor); // kappa
                push_u64_lane(&mut slots, &mut cursor); // c_data_len
                for _ in 0..s.c_data_entries {
                    push_u64_lane(&mut slots, &mut cursor);
                }
                push_u64_lane(&mut slots, &mut cursor); // x_len
                for _ in 0..s.x_entries {
                    push_u64_lane(&mut slots, &mut cursor);
                }
                push_u64_lane(&mut slots, &mut cursor); // m_in
            }
            NifsPayloadShape::CeClaim(s) => {
                // Fill order (per fill_nifs_ce_claim_at):
                //   d, kappa, c_data_len, c_data[..],
                //   x_rows, x_cols, x_active_cols, x_active_flat[..],
                //   r_len, r-K-pairs[..],
                //   y_ring_outer_len, per-row(inner_len, K-pairs[..]),
                //   m_in, fold_digest_fields(4 lanes),
                //   y_zcol_len, y_zcol-K-pairs[..],
                //   s_col_len, s_col-K-pairs[..].
                push_u64_lane(&mut slots, &mut cursor); // d
                push_u64_lane(&mut slots, &mut cursor); // kappa
                push_u64_lane(&mut slots, &mut cursor); // c_data_len
                for _ in 0..s.c_data_entries {
                    push_u64_lane(&mut slots, &mut cursor);
                }
                push_u64_lane(&mut slots, &mut cursor); // x_rows
                push_u64_lane(&mut slots, &mut cursor); // x_cols
                push_u64_lane(&mut slots, &mut cursor); // x_active_cols
                for _ in 0..(s.x_rows * s.x_active_cols) {
                    push_u64_lane(&mut slots, &mut cursor);
                }
                push_u64_lane(&mut slots, &mut cursor); // r_len
                for _ in 0..s.r_len {
                    push_u64_lane(&mut slots, &mut cursor); // K c0
                    push_u64_lane(&mut slots, &mut cursor); // K c1
                }
                push_u64_lane(&mut slots, &mut cursor); // y_ring outer_len
                for &inner in &s.y_ring_inner_lens {
                    push_u64_lane(&mut slots, &mut cursor); // inner_len
                    for _ in 0..inner {
                        push_u64_lane(&mut slots, &mut cursor);
                        push_u64_lane(&mut slots, &mut cursor);
                    }
                }
                push_u64_lane(&mut slots, &mut cursor); // m_in
                for _ in 0..4 {
                    push_u64_lane(&mut slots, &mut cursor); // fold_digest lane
                }
                push_u64_lane(&mut slots, &mut cursor); // y_zcol_len
                for _ in 0..s.y_zcol_len {
                    push_u64_lane(&mut slots, &mut cursor);
                    push_u64_lane(&mut slots, &mut cursor);
                }
                push_u64_lane(&mut slots, &mut cursor); // s_col_len
                for _ in 0..s.s_col_len {
                    push_u64_lane(&mut slots, &mut cursor);
                    push_u64_lane(&mut slots, &mut cursor);
                }
            }
        }
        debug_assert_eq!(
            cursor - payload_offset,
            shape.bits(),
            "nifs_payloads payload lane enumeration must cover exactly shape.bits()"
        );
        payloads.push(slots);
    }
    payloads
}

fn push_u64_lane(slots: &mut Vec<LaneSlot>, cursor: &mut usize) {
    slots.push(LaneSlot { bit_start: *cursor });
    *cursor += POSEIDON2_GOLDILOCKS_BITS;
}

fn collect_sponge_transcript_slots(layout: &FPrimeImageLayout) -> Vec<LaneSlot> {
    assert_eq!(
        layout.sponge_transcript_bits % POSEIDON2_GOLDILOCKS_BITS,
        0,
        "poseidon sponge transcript bit count must be 64-bit aligned"
    );
    let lane_count = layout.sponge_transcript_bits / POSEIDON2_GOLDILOCKS_BITS;
    let mut slots = Vec::with_capacity(lane_count);
    for lane_idx in 0..lane_count {
        slots.push(LaneSlot {
            bit_start: layout.sponge_transcript_splice + lane_idx * POSEIDON2_GOLDILOCKS_BITS,
        });
    }
    slots
}

fn collect_poseidon_trace_slots(layout: &FPrimeImageLayout) -> Vec<Vec<LaneSlot>> {
    let mut all = Vec::with_capacity(layout.one_shot_poseidon_splices.len());
    for (&splice, trace_layout) in layout
        .one_shot_poseidon_splices
        .iter()
        .zip(layout.one_shot_poseidon_layouts.iter())
    {
        assert_eq!(
            trace_layout.trace_len % POSEIDON2_GOLDILOCKS_BITS,
            0,
            "one-shot Poseidon trace length must be 64-bit aligned"
        );
        let lane_count = trace_layout.trace_len / POSEIDON2_GOLDILOCKS_BITS;
        let mut slots = Vec::with_capacity(lane_count);
        for lane_idx in 0..lane_count {
            slots.push(LaneSlot {
                bit_start: splice + lane_idx * POSEIDON2_GOLDILOCKS_BITS,
            });
        }
        all.push(slots);
    }
    all
}

fn collect_state_lane_slots(layout: &FPrimeImageLayout) -> Vec<LaneSlot> {
    let lane_bits = POSEIDON2_GOLDILOCKS_BITS;
    let state_in_lanes = STATE_IN_DIGEST_COUNT * 4;
    let state_out_lanes = STATE_OUT_COUNTER_COUNT + STATE_OUT_DIGEST_COUNT * 4;
    let total = state_in_lanes + state_out_lanes + CHUNK_DIGEST_LANE_COUNT;
    let mut slots = Vec::with_capacity(total);

    let mut bit = layout.state_in.offset;
    for _ in 0..state_in_lanes {
        slots.push(LaneSlot { bit_start: bit });
        bit += lane_bits;
    }
    debug_assert_eq!(bit, layout.state_in.end());

    let mut bit = layout.state_out.offset;
    for _ in 0..state_out_lanes {
        slots.push(LaneSlot { bit_start: bit });
        bit += lane_bits;
    }
    debug_assert_eq!(bit, layout.state_out.end());

    let mut bit = layout.chunk_digest.offset;
    for _ in 0..CHUNK_DIGEST_LANE_COUNT {
        slots.push(LaneSlot { bit_start: bit });
        bit += lane_bits;
    }
    debug_assert_eq!(bit, layout.chunk_digest.end());

    slots
}

fn collect_kmul_slots(layout: &FPrimeImageLayout) -> Vec<LaneSlot> {
    let kmul_count = layout.config.kmul_count;
    let mut slots = Vec::with_capacity(KMUL_LANES_PER_SLOT * kmul_count);
    for i in 0..kmul_count {
        let base = layout.kmul.offset + i * KMUL_SLOT_BITS;
        // p (low, high), q (low, high), r (low, high).
        for lane_idx in 0..KMUL_LANES_PER_SLOT {
            slots.push(LaneSlot {
                bit_start: base + lane_idx * POSEIDON2_GOLDILOCKS_BITS,
            });
        }
    }
    slots
}

fn collect_ring_action_slots(layout: &FPrimeImageLayout) -> Vec<LaneSlot> {
    let pair_layout = layout.config.ring_action_pair_layout;
    assert_eq!(
        pair_layout.rho_enc,
        LowNormEncoding::U64,
        "ring_action lane slots require U64 ρ encoding (non-U64 ring_action not supported by 1.4b structure)"
    );
    assert_eq!(
        pair_layout.c_enc,
        LowNormEncoding::U64,
        "ring_action lane slots require U64 c encoding"
    );
    assert_eq!(
        pair_layout.prod_enc,
        LowNormEncoding::U64,
        "ring_action lane slots require U64 prod encoding"
    );
    assert_eq!(
        pair_layout.out_enc,
        LowNormEncoding::U64,
        "ring_action lane slots require U64 out encoding"
    );

    let pair_count = layout.config.ring_action_pair_count;
    let lanes_per_pair = 3 * D + D * D;
    let mut slots = Vec::with_capacity(pair_count * lanes_per_pair);

    for pair_idx in 0..pair_count {
        let splice = layout.ring_action_pair_splices[pair_idx];
        // The pair layout exposes pair-local primitive z indices (where
        // index 0 is the pair's constant slot). The image's z[0] is
        // shared, so pair-local index `k ≥ 1` maps to image-frame
        // offset `splice + k − 1`.
        let to_image = |pair_local_z: usize| splice + pair_local_z - 1;

        for j in 0..D {
            slots.push(LaneSlot {
                bit_start: to_image(pair_layout.rho_limb_start(j)),
            });
        }
        for j in 0..D {
            slots.push(LaneSlot {
                bit_start: to_image(pair_layout.c_limb_start(j)),
            });
        }
        for i in 0..D {
            for j in 0..D {
                slots.push(LaneSlot {
                    bit_start: to_image(pair_layout.prod_limb_start(i, j)),
                });
            }
        }
        for m in 0..D {
            slots.push(LaneSlot {
                bit_start: to_image(pair_layout.out_lane_start(m)),
            });
        }
    }
    slots
}

/// Phase 1.4 CCS structure for one `enc(F')` step. Carries the
/// [`FPrimeImageLayout`] it was built from plus the per-region
/// lane-slot lists so downstream consumers (1.4c, 1.4d) can extend the
/// witness without re-deriving offsets.
#[derive(Clone, Debug)]
pub struct FPrimeStructure {
    pub layout: FPrimeImageLayout,
    pub ccs: CcsStructure<F>,
    /// Lane-decode positions grouped by region.
    pub lane_slots: FPrimeLaneSlots,
}

/// Build the CCS structure: bit-validity (1.4a), state_in/state_out/chunk_digest (1.4b-a)
/// plus nifs_payloads/kmul/ring_action/poseidon lane decode binding, ring_action product binding (1.4c-a),
/// ring_action output binding (1.4c-b), and poseidon↔state_out digest binding (1.4c-c).
///
/// Panics if `layout.end < 2` (need at least the constant slot + one
/// bit column to form a valid CCS structure) or if ring_action uses a non-U64
/// encoding.
/// Matrix-index assignment for the mixed-gate F' CCS polynomial:
/// bitness + product + Poseidon S-box + linear equality.
pub(crate) mod gate {
    pub const BITNESS: usize = 0;
    pub const PRODUCT_LEFT: usize = 1;
    pub const PRODUCT_RIGHT: usize = 2;
    pub const PRODUCT_OUT: usize = 3;
    pub const SBOX_IN: usize = 4;
    pub const SBOX_OUT: usize = 5;
    pub const LINEAR_LHS: usize = 6;
    pub const LINEAR_RHS: usize = 7;
    pub const ARITY: usize = 8;
}

/// Mixed-gate row builder for the F' CCS structure.
///
/// Each public method appends exactly one constraint row, populating
/// only the matrices its gate uses; the others are zero at that row.
/// The polynomial is
///
/// ```text
/// f = (B² − B) + (Pl·Pr − Po) + (X⁷ − Y) + (Ll − Lr)
/// ```
///
/// so each row's contribution to `f` is the relevant gate's term in
/// isolation. `finish` builds the eight `CscMat` matrices and the
/// corresponding `SparsePoly`.
pub(crate) struct MixedGateBuilder {
    trips: [Vec<(usize, usize, F)>; gate::ARITY],
    rows: usize,
}

impl MixedGateBuilder {
    pub(crate) fn with_estimated_rows(estimated_rows: usize) -> Self {
        Self {
            trips: std::array::from_fn(|_| Vec::with_capacity(estimated_rows)),
            rows: 0,
        }
    }

    /// Current row count. Useful for sibling structure builders that
    /// need to know where their appended row block starts.
    #[allow(dead_code)]
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    /// `z[col] · (z[col] − 1) = 0`. Populates the BITNESS matrix only.
    pub(crate) fn bitness(&mut self, col: usize) -> usize {
        let row = self.rows;
        self.trips[gate::BITNESS].push((row, col, F::ONE));
        self.rows += 1;
        row
    }

    /// `(Σ left) · (Σ right) = (Σ out)`. Populates the three product matrices.
    /// Each operand is an arbitrary linear combination `(col, coeff)` so the
    /// builder can represent products of lane-recomposed values directly.
    pub(crate) fn product<L, R, O>(&mut self, left: L, right: R, out: O) -> usize
    where
        L: IntoIterator<Item = (usize, F)>,
        R: IntoIterator<Item = (usize, F)>,
        O: IntoIterator<Item = (usize, F)>,
    {
        let row = self.rows;
        for (col, coeff) in left {
            self.trips[gate::PRODUCT_LEFT].push((row, col, coeff));
        }
        for (col, coeff) in right {
            self.trips[gate::PRODUCT_RIGHT].push((row, col, coeff));
        }
        for (col, coeff) in out {
            self.trips[gate::PRODUCT_OUT].push((row, col, coeff));
        }
        self.rows += 1;
        row
    }

    /// `(Σ lhs) = (Σ rhs)`. Populates the LINEAR_LHS / LINEAR_RHS matrices.
    pub(crate) fn linear<L, R>(&mut self, lhs: L, rhs: R) -> usize
    where
        L: IntoIterator<Item = (usize, F)>,
        R: IntoIterator<Item = (usize, F)>,
    {
        let row = self.rows;
        for (col, coeff) in lhs {
            self.trips[gate::LINEAR_LHS].push((row, col, coeff));
        }
        for (col, coeff) in rhs {
            self.trips[gate::LINEAR_RHS].push((row, col, coeff));
        }
        self.rows += 1;
        row
    }

    /// `(Σ sbox_in)^7 = (Σ sbox_out)`. Populates the SBOX_IN / SBOX_OUT matrices.
    /// Both sides accept arbitrary linear combinations; this mirrors the
    /// native Poseidon2 row shape where the S-box input is an Expr
    /// (state-word + round-constant) and the output is a single Word
    /// expanded to its 64 bit-coefficients.
    fn sbox7_general<I, O>(&mut self, sbox_in: I, sbox_out: O) -> usize
    where
        I: IntoIterator<Item = (usize, F)>,
        O: IntoIterator<Item = (usize, F)>,
    {
        let row = self.rows;
        for (col, coeff) in sbox_in {
            self.trips[gate::SBOX_IN].push((row, col, coeff));
        }
        for (col, coeff) in sbox_out {
            self.trips[gate::SBOX_OUT].push((row, col, coeff));
        }
        self.rows += 1;
        row
    }

    /// Build the eight sparse matrices + the mixed-gate polynomial.
    pub(crate) fn finish(self, cols: usize) -> CcsStructure<F> {
        let n = self.rows;
        let matrices: Vec<CcsMatrix<F>> = self
            .trips
            .into_iter()
            .map(|trips| CcsMatrix::Csc(CscMat::from_triplets(trips, n, cols)))
            .collect();
        let zero_exps = || vec![0u32; gate::ARITY];
        let exps_at = |idx: usize, power: u32| {
            let mut e = zero_exps();
            e[idx] = power;
            e
        };
        let exps_product = || {
            let mut e = zero_exps();
            e[gate::PRODUCT_LEFT] = 1;
            e[gate::PRODUCT_RIGHT] = 1;
            e
        };
        let f = SparsePoly::new(
            gate::ARITY,
            vec![
                Term {
                    coeff: F::ONE,
                    exps: exps_at(gate::BITNESS, 2),
                },
                Term {
                    coeff: F::ZERO - F::ONE,
                    exps: exps_at(gate::BITNESS, 1),
                },
                Term {
                    coeff: F::ONE,
                    exps: exps_product(),
                },
                Term {
                    coeff: F::ZERO - F::ONE,
                    exps: exps_at(gate::PRODUCT_OUT, 1),
                },
                Term {
                    coeff: F::ONE,
                    exps: exps_at(gate::SBOX_IN, 7),
                },
                Term {
                    coeff: F::ZERO - F::ONE,
                    exps: exps_at(gate::SBOX_OUT, 1),
                },
                Term {
                    coeff: F::ONE,
                    exps: exps_at(gate::LINEAR_LHS, 1),
                },
                Term {
                    coeff: F::ZERO - F::ONE,
                    exps: exps_at(gate::LINEAR_RHS, 1),
                },
            ],
        );
        CcsStructure::new_sparse(matrices, f).expect("mixed-gate F' CCS structure must be well-formed")
    }
}

pub fn build_f_prime_structure(layout: FPrimeImageLayout) -> FPrimeStructure {
    let image_end = layout.end;
    assert!(
        image_end >= 2,
        "FPrimeImageLayout::end = {image_end} too small; need constant slot + ≥1 bit column"
    );

    let lane_slots = f_prime_lane_slots(&layout);
    let mut builder = MixedGateBuilder::with_estimated_rows(image_end);
    emit_shell_rows(&layout, &lane_slots, &mut builder);
    let ccs = builder.finish(image_end);

    FPrimeStructure {
        layout,
        ccs,
        lane_slots,
    }
}

/// Emit every shell row the F' structure owns into `builder`:
/// bit-validity, ring-action products/outputs, trace↔state-out digest
/// bindings, the unified-accumulator selector, trace↔public-x_out
/// bindings, and the Poseidon transition enforcements.
///
/// Sibling structure builders (e.g. R1CS F') call this on a fresh
/// `MixedGateBuilder`, then append their app-level rows, then finish.
/// The image layout (column count, bit positions, lane slots) is
/// identical to the Fibonacci build; only the appended rows differ.
pub(crate) fn emit_shell_rows(
    layout: &FPrimeImageLayout,
    lane_slots: &FPrimeLaneSlots,
    builder: &mut MixedGateBuilder,
) {
    let image_end = layout.end;
    let bit_count = image_end - 1;
    let ring_action_product_count = layout.config.ring_action_pair_count * RING_ACTION_PRODUCT_LANES_PER_PAIR;
    let ring_action_output_count = layout.config.ring_action_pair_count * RING_ACTION_OUTPUT_LANES_PER_PAIR;
    let state_out_binding_count = layout.config.one_shot_digest_to_state_out_bindings.len() * POSEIDON2_DIGEST_LEN;
    let public_x_out_binding_count =
        layout.config.one_shot_digest_to_public_x_out_bindings.len() * POSEIDON2_DIGEST_LEN;
    // Unified accumulator selector: 4 product rows (one per digest
    // lane). `is_base`'s binary constraint is already covered by the
    // bit-validity loop above.
    let unified_selector_count = if layout.config.unified_accumulator_selector.is_some() {
        POSEIDON2_DIGEST_LEN
    } else {
        0
    };
    let total_shell_rows = bit_count
        + ring_action_product_count
        + ring_action_output_count
        + state_out_binding_count
        + public_x_out_binding_count
        + unified_selector_count;
    let base_row = builder.rows();

    // ── Bit-validity rows: `z[col] · (z[col] − 1) = 0` for every committed bit.
    for col in 1..image_end {
        builder.bitness(col);
    }
    debug_assert_eq!(builder.rows() - base_row, bit_count);

    // ── Ring-action product rows: `(Σ 2^i · ρ_bits) · (Σ 2^i · c_bits) = (Σ 2^i · prod_bits)`.
    for pair_idx in 0..layout.config.ring_action_pair_count {
        let pair_base = pair_idx * RING_ACTION_LANES_PER_PAIR;
        for i in 0..D {
            for j in 0..D {
                let rho = lane_slots.ring_action_lanes[pair_base + i];
                let c = lane_slots.ring_action_lanes[pair_base + D + j];
                let prod = lane_slots.ring_action_lanes[pair_base + 2 * D + i * D + j];
                builder.product(lane_terms(rho), lane_terms(c), lane_terms(prod));
            }
        }
    }
    debug_assert_eq!(builder.rows() - base_row, bit_count + ring_action_product_count);

    // ── Ring-action output rows: `Σ 2^i · out_m_bits = Σ Φ[i+j][m] · (Σ 2^i · prod_ij_bits)`.
    for pair_idx in 0..layout.config.ring_action_pair_count {
        let pair_base = pair_idx * RING_ACTION_LANES_PER_PAIR;
        for m_idx in 0..D {
            let out = lane_slots.ring_action_lanes[pair_base + 2 * D + D * D + m_idx];
            let mut rhs: Vec<(usize, F)> = Vec::new();
            for i in 0..D {
                for j in 0..D {
                    let coeff = phi_reduction_coeff(i + j, m_idx);
                    if coeff != F::ZERO {
                        let prod = lane_slots.ring_action_lanes[pair_base + 2 * D + i * D + j];
                        rhs.extend(scaled_lane_terms(prod, coeff));
                    }
                }
            }
            builder.linear(lane_terms(out), rhs);
        }
    }
    debug_assert_eq!(
        builder.rows() - base_row,
        bit_count + ring_action_product_count + ring_action_output_count
    );

    // ── Trace digest ↔ state-out digest binding rows.
    for binding in &layout.config.one_shot_digest_to_state_out_bindings {
        let trace_slots = &lane_slots.poseidon_trace_lanes[binding.one_shot_index];
        // Per `PoseidonTraceLayout`: the post-final-permutation state's
        // 8 words live in the last `WIDTH` lanes; the digest is the
        // first `DIGEST_LEN` of those.
        let digest_lane_base = trace_slots.len() - POSEIDON2_WIDTH;
        let state_out_lane_base = state_out_digest_lane_base(binding.state_out_target);
        for lane in 0..POSEIDON2_DIGEST_LEN {
            let trace = trace_slots[digest_lane_base + lane];
            let state_out = lane_slots.state_lanes[state_out_lane_base + lane];
            builder.linear(lane_terms(trace), lane_terms(state_out));
        }
    }
    debug_assert_eq!(
        builder.rows() - base_row,
        bit_count + ring_action_product_count + ring_action_output_count + state_out_binding_count
    );

    // ── Unified-accumulator selector rows.
    //
    // For each `new_acc_digest` lane (4 lanes total):
    //     (1 - is_base) · (recursive_lane − base_lane)
    //   = (new_acc_digest_lane − base_lane)
    //
    // - `is_base = 1` ⇒ left side is 0, so RHS forces
    //   `new_acc_digest_lane = base_lane`.
    // - `is_base = 0` ⇒ left side is `(recursive − base)`, so RHS
    //   forces `new_acc_digest_lane = recursive_lane`.
    //
    // `is_base ∈ {0, 1}` is enforced by the bit-validity loop above
    // (the `is_base` lane sits in `layout.is_base`, a single bit column).
    //
    // `base_lane` and `recursive_lane` are the digest lanes of the two
    // accumulator Poseidon traces the plan emitted at
    // `selector.base_trace_index` / `selector.recursive_trace_index`.
    // Both traces are constraint-bound by their own Poseidon transition
    // enforcements, so the selected digest is backed by an authoritative
    // preimage — it is not pure digest authority.
    if let Some(selector) = layout.config.unified_accumulator_selector {
        let is_base_col = layout.is_base.offset;
        let base_trace_slots = &lane_slots.poseidon_trace_lanes[selector.base_trace_index];
        let rec_trace_slots = &lane_slots.poseidon_trace_lanes[selector.recursive_trace_index];
        let base_digest_lane_base = base_trace_slots.len() - POSEIDON2_WIDTH;
        let rec_digest_lane_base = rec_trace_slots.len() - POSEIDON2_WIDTH;
        let acc_digest_lane_base = state_out_digest_lane_base(StateOutDigestTarget::NewAccDigest);

        for lane in 0..POSEIDON2_DIGEST_LEN {
            let base_lane = base_trace_slots[base_digest_lane_base + lane];
            let rec_lane = rec_trace_slots[rec_digest_lane_base + lane];
            let acc_lane = lane_slots.state_lanes[acc_digest_lane_base + lane];

            // left = 1 - is_base = (constant column 0, +1) + (is_base column, -1)
            let left: Vec<(usize, F)> = vec![(0, F::ONE), (is_base_col, F::ZERO - F::ONE)];
            // right = recursive_lane - base_lane
            let right: Vec<(usize, F)> = lane_terms(rec_lane)
                .chain(scaled_lane_terms(base_lane, F::ZERO - F::ONE))
                .collect();
            // out = new_acc_digest_lane - base_lane
            let out: Vec<(usize, F)> = lane_terms(acc_lane)
                .chain(scaled_lane_terms(base_lane, F::ZERO - F::ONE))
                .collect();
            builder.product(left, right, out);
        }
    }
    debug_assert_eq!(
        builder.rows() - base_row,
        bit_count
            + ring_action_product_count
            + ring_action_output_count
            + state_out_binding_count
            + unified_selector_count
    );

    // ── Trace digest ↔ public-x_out binding rows.
    for (binding_idx, binding) in layout
        .config
        .one_shot_digest_to_public_x_out_bindings
        .iter()
        .enumerate()
    {
        let trace_slots = &lane_slots.poseidon_trace_lanes[binding.one_shot_index];
        let digest_lane_base = trace_slots.len() - POSEIDON2_WIDTH;
        let public_x_out_lanes = lane_slots.public_x_out_binding_lanes[binding_idx];
        for lane in 0..POSEIDON2_DIGEST_LEN {
            let trace = trace_slots[digest_lane_base + lane];
            let public_x_out = public_x_out_lanes[lane];
            builder.linear(lane_terms(trace), lane_terms(public_x_out));
        }
    }
    debug_assert_eq!(builder.rows() - base_row, total_shell_rows);

    // ── Optional Poseidon transition rows + variable preimage binding (1.4d-a-4) ──
    // For each enforcement:
    //   1. Build the native CCS using a *dummy zero preimage* of the
    //      correct length. The structure shape (round constraints,
    //      bitness, S-boxes, MDS, etc.) depends only on preimage length,
    //      not values.
    //   2. Lift every non-bitness, non-absorb row through the mixed-gate
    //      builder. Absorb rows are skipped because they bake the
    //      (dummy) preimage as row constants — they don't match the
    //      caller's source-based preimage.
    //   3. Re-emit absorb rows ourselves: each post-absorb state word
    //      equals previous state's last lane (or 0 for the first
    //      absorb) plus the source-lane value (or 0 if no addition for
    //      this lane in this absorb).
    for enforcement in &layout.config.poseidon_transition_enforcements {
        let dummy_preimage: Vec<F> = vec![F::ZERO; enforcement.preimage_lanes.len()];
        let native_bundle = crate::engine::ccs_native::poseidon2::build_bit_backed_poseidon2_hash(&dummy_preimage);
        let splice = layout.one_shot_poseidon_splices[enforcement.one_shot_index];
        let absorb_row_skip: std::collections::BTreeSet<usize> = native_bundle
            .absorb_rows
            .iter()
            .flatten()
            .copied()
            .collect();
        lift_native_poseidon_rows_skipping(&native_bundle.structure, splice, &absorb_row_skip, builder);
        emit_variable_poseidon_absorb_rows(enforcement, lane_slots, builder);
    }
}

/// Mechanically port the non-bitness, non-skipped rows of a bit-backed
/// Poseidon2 native CCS into the F' mixed-gate builder.
///
/// Native column remap: `0 → 0` (shared CCS constant slot); `k ≥ 1 →
/// splice + k − 1` (the bit at native z-index `k` lives at this image
/// position once `splice_one_shot_poseidon` has run). Bitness rows are
/// skipped because the F' builder already enforces every committed bit
/// is in `{0, 1}` via its own bitness block. `skip_rows` carries the
/// extra row indices the caller wants to handle externally (typically
/// absorb rows, which 1.4d-a-4 re-emits with source-bound preimage
/// lanes instead of baked constants).
fn lift_native_poseidon_rows_skipping(
    native: &CcsStructure<F>,
    splice: usize,
    skip_rows: &std::collections::BTreeSet<usize>,
    builder: &mut MixedGateBuilder,
) {
    assert_eq!(
        native.matrices.len(),
        5,
        "native bit-backed Poseidon2 CCS has 5 mixed-gate matrices"
    );
    let n = native.n;
    let remap = |native_col: usize| -> usize {
        if native_col == 0 {
            0
        } else {
            splice + native_col - 1
        }
    };
    let per_row = |matrix: &CcsMatrix<F>| -> Vec<Vec<(usize, F)>> {
        let mut entries: Vec<Vec<(usize, F)>> = vec![Vec::new(); n];
        if let Some(csc) = matrix.as_csc() {
            for col in 0..csc.ncols {
                let start = csc.col_ptr[col];
                let end = csc.col_ptr[col + 1];
                for k in start..end {
                    let r = csc.row_idx[k];
                    let v = csc.vals[k];
                    entries[r].push((remap(col), v));
                }
            }
        }
        entries
    };

    // Native matrix indices (per `engine::ccs_native::poseidon2`):
    //   0 = B (bitness), 1 = X (sbox in), 2 = Y (sbox out),
    //   3 = Lhs (linear lhs), 4 = Rhs (linear rhs).
    let bitness_rows = per_row(&native.matrices[0]);
    let sbox_in_rows = per_row(&native.matrices[1]);
    let sbox_out_rows = per_row(&native.matrices[2]);
    let lhs_rows = per_row(&native.matrices[3]);
    let rhs_rows = per_row(&native.matrices[4]);

    for r in 0..n {
        if skip_rows.contains(&r) {
            continue;
        }
        let bit_active = !bitness_rows[r].is_empty();
        let sbox_active = !sbox_in_rows[r].is_empty() || !sbox_out_rows[r].is_empty();
        let linear_active = !lhs_rows[r].is_empty() || !rhs_rows[r].is_empty();

        if bit_active {
            // F' already constrains every committed bit; skip.
            debug_assert!(
                !sbox_active && !linear_active,
                "native row {r} mixes bitness with another gate"
            );
            continue;
        }
        if sbox_active {
            debug_assert!(!linear_active, "native row {r} mixes sbox with linear");
            builder.sbox7_general(sbox_in_rows[r].clone(), sbox_out_rows[r].clone());
        } else if linear_active {
            builder.linear(lhs_rows[r].clone(), rhs_rows[r].clone());
        }
        // else: empty row — should not happen in a well-formed native CCS,
        //       but quietly ignored so we don't crash on degenerate inputs.
    }
}

/// Resolve one preimage source to its `(col, coeff)` terms in the F'
/// bit-frame. A `Constant` becomes `(0, value)` (col 0 = ONE). Every
/// image-region source expands to its 64-bit (or 32-bit half) lane sum
/// directly; no decoded witness column is referenced.
fn poseidon_preimage_source_terms(
    source: &PoseidonPreimageLaneSource,
    lane_slots: &FPrimeLaneSlots,
) -> Vec<(usize, F)> {
    match *source {
        PoseidonPreimageLaneSource::Constant(v) => {
            if v == F::ZERO {
                Vec::new()
            } else {
                vec![(0, v)]
            }
        }
        PoseidonPreimageLaneSource::StateLane(i) => lane_terms(lane_slots.state_lanes[i]).collect(),
        PoseidonPreimageLaneSource::StateLaneLowHalf(i) => lane_low_half_terms(lane_slots.state_lanes[i]).collect(),
        PoseidonPreimageLaneSource::StateLaneHighHalf(i) => lane_high_half_terms(lane_slots.state_lanes[i]).collect(),
        PoseidonPreimageLaneSource::NifsPayloadLane {
            payload_index,
            lane_index,
        } => lane_terms(lane_slots.nifs_payload_lanes[payload_index][lane_index]).collect(),
        PoseidonPreimageLaneSource::RingActionLane(i) => lane_terms(lane_slots.ring_action_lanes[i]).collect(),
        PoseidonPreimageLaneSource::PoseidonTraceLane {
            trace_index,
            lane_index,
        } => lane_terms(lane_slots.poseidon_trace_lanes[trace_index][lane_index]).collect(),
        PoseidonPreimageLaneSource::SpongeTranscriptLane(i) => {
            lane_terms(lane_slots.sponge_transcript_lanes[i]).collect()
        }
        PoseidonPreimageLaneSource::PublicXOutBindingLane { binding_index, lane } => {
            lane_terms(lane_slots.public_x_out_binding_lanes[binding_index][lane]).collect()
        }
        PoseidonPreimageLaneSource::AppAssignmentLane(var_index) => {
            lane_terms(lane_slots.app_assignment_lanes[var_index]).collect()
        }
    }
}

/// Emit absorb-binding rows for one Poseidon transition enforcement.
///
/// For each absorb (sponge chunk + final padding), the bit-backed
/// builder commits 8 post-absorb state words at the start of that
/// absorb's permutation. We constrain each of those 8 words:
///
/// ```text
/// post_state_lane = previous_final_state_lane  +  preimage_source_lane (if absorbed this chunk)
///                  + 1 (if padding chunk, lane 0)
/// ```
///
/// `previous_final_state_lane` is zero for the first absorb (sponge
/// starts at all-zero state). The 8 post-absorb words sit at the first
/// `WIDTH` words of each permutation's word block; the previous
/// permutation's final 8 words sit at the last `WIDTH` words of the
/// preceding permutation's block.
fn emit_variable_poseidon_absorb_rows(
    enforcement: &super::image::PoseidonTransitionEnforcement,
    lane_slots: &FPrimeLaneSlots,
    builder: &mut MixedGateBuilder,
) {
    use crate::engine::ccs_native::poseidon2::{BIT_BACKED_PERMUTATION_WORDS, POSEIDON2_RATE};

    let trace_idx = enforcement.one_shot_index;
    let preimage_len = enforcement.preimage_lanes.len();
    let absorbs = preimage_len.div_ceil(POSEIDON2_RATE) + 1;
    let trace_slots = &lane_slots.poseidon_trace_lanes[trace_idx];

    for absorb_idx in 0..absorbs {
        for lane in 0..POSEIDON2_WIDTH {
            // Post-absorb state word: first `WIDTH` words of this absorb's
            // permutation block.
            let post_word = absorb_idx * BIT_BACKED_PERMUTATION_WORDS + lane;
            let post_slot = trace_slots[post_word];

            // Build the RHS: previous final state lane (if any) + absorbed
            // value (if this lane absorbs in this absorb) + padding +1
            // (if last absorb, lane 0).
            let mut rhs: Vec<(usize, F)> = Vec::new();

            if absorb_idx > 0 {
                // Last `WIDTH` words of previous permutation block.
                let prev_final_word = absorb_idx * BIT_BACKED_PERMUTATION_WORDS - POSEIDON2_WIDTH + lane;
                rhs.extend(lane_terms(trace_slots[prev_final_word]));
            }

            let is_padding_absorb = absorb_idx == absorbs - 1;
            if is_padding_absorb {
                // Padding: state[0] += F::ONE before the final permutation;
                // all other lanes have no extra addition.
                if lane == 0 {
                    rhs.push((0, F::ONE));
                }
            } else if lane < POSEIDON2_RATE {
                let preimage_idx = absorb_idx * POSEIDON2_RATE + lane;
                if preimage_idx < preimage_len {
                    let source_terms =
                        poseidon_preimage_source_terms(&enforcement.preimage_lanes[preimage_idx], lane_slots);
                    rhs.extend(source_terms);
                }
            }

            builder.linear(lane_terms(post_slot), rhs);
        }
    }
}

impl FPrimeStructure {
    /// Return the strict low-norm witness for this image: just `image.values`.
    ///
    /// Strict-encoding invariant (Phase 1.5b-0): the CCS witness is
    /// exactly the committed bit-vector, all entries in `{0, 1}` except
    /// `z[0] = 1`. Lane-recomposed u64 values appear inside constraint
    /// rows as `Σ 2^i · z[bit_start + i]`, never as fresh witness columns.
    pub fn extend_witness_from_image(&self, image: &FPrimeImage) -> Vec<F> {
        assert_eq!(
            image.values.len(),
            self.layout.end,
            "image must have been built against this structure's layout"
        );
        debug_assert_eq!(self.ccs.m, self.layout.end);
        image.values.clone()
    }

    /// Evaluate `f(M_0·z, …, M_7·z)` row-by-row under the mixed-gate
    /// polynomial. Each entry is zero iff the corresponding constraint
    /// (bit validity, lane decode, ring-action product, ring-action
    /// output, or trace-digest binding) holds for `z`.
    pub fn evaluate_constraints(&self, z: &[F]) -> Vec<F> {
        assert_eq!(z.len(), self.ccs.m, "witness length must equal structure.m");
        assert_eq!(
            self.ccs.matrices.len(),
            gate::ARITY,
            "evaluate_constraints expects {} mixed-gate matrices, found {}",
            gate::ARITY,
            self.ccs.matrices.len(),
        );
        let n = self.ccs.n;
        let mut matrix_z: [Vec<F>; gate::ARITY] = std::array::from_fn(|_| vec![F::ZERO; n]);
        for (m_idx, matrix) in self.ccs.matrices.iter().enumerate() {
            matrix.add_mul_into(z, &mut matrix_z[m_idx], n);
        }
        (0..n)
            .map(|r| {
                let point: [F; gate::ARITY] = std::array::from_fn(|m_idx| matrix_z[m_idx][r]);
                self.ccs.f.eval(&point)
            })
            .collect()
    }

    /// `true` iff every row of `f(M_0·z, …, M_7·z) = 0`.
    pub fn is_satisfied(&self, z: &[F]) -> bool {
        self.first_unsatisfied_row(z).is_none()
    }

    /// Index of the first row that fails, or `None` if all hold.
    pub fn first_unsatisfied_row(&self, z: &[F]) -> Option<usize> {
        self.evaluate_constraints(z)
            .into_iter()
            .position(|v| v != F::ZERO)
    }

    /// Number of ring_action product rows (`D²` per ring-action pair).
    pub fn ring_action_product_row_count(&self) -> usize {
        self.layout.config.ring_action_pair_count * RING_ACTION_PRODUCT_LANES_PER_PAIR
    }

    /// Number of ring_action output rows (`D` per ring-action pair).
    pub fn ring_action_output_row_count(&self) -> usize {
        self.layout.config.ring_action_pair_count * RING_ACTION_OUTPUT_LANES_PER_PAIR
    }

    /// Row index for the ring_action constraint `ρ[i] · c[j] = prod[i][j]`.
    pub fn ring_action_product_row(&self, pair_idx: usize, i: usize, j: usize) -> usize {
        assert!(
            pair_idx < self.layout.config.ring_action_pair_count,
            "pair_idx {pair_idx} out of range for {} ring_action pairs",
            self.layout.config.ring_action_pair_count
        );
        assert!(i < D, "ring_action product row i={i} out of range D={D}");
        assert!(j < D, "ring_action product row j={j} out of range D={D}");
        self.ring_action_product_row_start() + pair_idx * RING_ACTION_PRODUCT_LANES_PER_PAIR + i * D + j
    }

    /// First row of the ring_action product-constraint block. Rows preceding it
    /// are the bit-validity rows (one per committed bit, i.e. `layout.end - 1`).
    pub fn ring_action_product_row_start(&self) -> usize {
        self.layout.end - 1
    }

    /// Row index for the ring_action constraint
    /// `out[m] = Σ Φ_TABLE[i+j][m] · prod[i][j]`.
    pub fn ring_action_output_row(&self, pair_idx: usize, m: usize) -> usize {
        assert!(
            pair_idx < self.layout.config.ring_action_pair_count,
            "pair_idx {pair_idx} out of range for {} ring_action pairs",
            self.layout.config.ring_action_pair_count
        );
        assert!(m < D, "ring_action output row m={m} out of range D={D}");
        self.ring_action_output_row_start() + pair_idx * RING_ACTION_OUTPUT_LANES_PER_PAIR + m
    }

    /// First row of the ring_action output-constraint block.
    pub fn ring_action_output_row_start(&self) -> usize {
        self.ring_action_product_row_start() + self.ring_action_product_row_count()
    }

    /// Number of poseidon↔state_out digest binding rows
    /// (`POSEIDON2_DIGEST_LEN` per binding).
    pub fn state_out_digest_binding_row_count(&self) -> usize {
        self.layout
            .config
            .one_shot_digest_to_state_out_bindings
            .len()
            * POSEIDON2_DIGEST_LEN
    }

    /// First row of the poseidon↔state_out binding block.
    pub fn state_out_digest_binding_row_start(&self) -> usize {
        self.ring_action_output_row_start() + self.ring_action_output_row_count()
    }

    /// Row index for the constraint
    /// `poseidon_trace_digest_lane[binding_idx][lane] = state_out_digest_lane[lane]`.
    pub fn state_out_digest_binding_row(&self, binding_idx: usize, lane: usize) -> usize {
        assert!(
            binding_idx
                < self
                    .layout
                    .config
                    .one_shot_digest_to_state_out_bindings
                    .len(),
            "binding_idx {binding_idx} out of range for {} state-out digest bindings",
            self.layout
                .config
                .one_shot_digest_to_state_out_bindings
                .len()
        );
        assert!(
            lane < POSEIDON2_DIGEST_LEN,
            "state-out digest binding lane {lane} out of range (digest has {POSEIDON2_DIGEST_LEN} lanes)"
        );
        self.state_out_digest_binding_row_start() + binding_idx * POSEIDON2_DIGEST_LEN + lane
    }

    /// Number of one-shot-trace ↔ public-x_out binding rows
    /// (`POSEIDON2_DIGEST_LEN` per binding).
    pub fn public_x_out_binding_row_count(&self) -> usize {
        self.layout
            .config
            .one_shot_digest_to_public_x_out_bindings
            .len()
            * POSEIDON2_DIGEST_LEN
    }

    /// First row of the public-x_out binding block.
    pub fn public_x_out_binding_row_start(&self) -> usize {
        self.state_out_digest_binding_row_start() + self.state_out_digest_binding_row_count()
    }

    /// Row index for the constraint
    /// `poseidon_trace_digest_lane[binding_idx][lane] = public_x_out_lane[binding_idx][lane]`.
    pub fn public_x_out_binding_row(&self, binding_idx: usize, lane: usize) -> usize {
        assert!(
            binding_idx
                < self
                    .layout
                    .config
                    .one_shot_digest_to_public_x_out_bindings
                    .len(),
            "binding_idx {binding_idx} out of range for {} public-x_out bindings",
            self.layout
                .config
                .one_shot_digest_to_public_x_out_bindings
                .len()
        );
        assert!(
            lane < POSEIDON2_DIGEST_LEN,
            "public-x_out binding lane {lane} out of range (digest has {POSEIDON2_DIGEST_LEN} lanes)"
        );
        self.public_x_out_binding_row_start() + binding_idx * POSEIDON2_DIGEST_LEN + lane
    }
}

/// First state_in/state_out/chunk_digest lane slot index for the four-lane state_out digest at
/// `target`. state_in occupies the first 24 lanes; state_out starts at index 24
/// with two u64 counters (chunk_count, step_count), then three
/// four-lane digests in fill order: new_z_i, new_public_trace,
/// new_acc_digest.
fn state_out_digest_lane_base(target: StateOutDigestTarget) -> usize {
    const STATE_IN_LANES: usize = 24;
    const STATE_OUT_COUNTER_LANES: usize = 2;
    let state_out_digests_start = STATE_IN_LANES + STATE_OUT_COUNTER_LANES;
    match target {
        StateOutDigestTarget::NewZI => state_out_digests_start,
        StateOutDigestTarget::NewPublicTrace => state_out_digests_start + 4,
        StateOutDigestTarget::NewAccDigest => state_out_digests_start + 8,
    }
}
