//! App-agnostic CCS structure for one low-norm `enc(F')` source image.
//!
//! The image layout lives in `frontends::f_prime::image`; this module
//! turns it into the mixed-gate [`CcsStructure`]. Canonical-u64 lanes
//! are not materialized as fresh columns: rows substitute
//! `Σ 2^i · z[bit_start+i]` directly.

use neo_ccs::{CcsMatrix, CcsStructure, CscMat, SparsePoly, Term};
use neo_math::ring::D;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::ccs_native::poseidon2::{POSEIDON2_DIGEST_LEN, POSEIDON2_GOLDILOCKS_BITS, POSEIDON2_WIDTH};
use crate::engine::r1cs_circuit::ring_action::phi_reduction_coeff;
use crate::frontends::f_prime::image::{
    FPrimeImage, FPrimeImageConfig, FPrimeImageLayout, NifsPayloadShape, PoseidonPreimageLaneSource,
    StateInDigestTarget, StateOutDigestTarget,
};
use crate::frontends::f_prime::projection_structure::{
    collect_projection_slots, emit_projection_semantic_rows, projection_semantic_row_count, ProjectionLaneSlots,
};
use crate::frontends::f_prime::recursive_plan::{STATE_LANE_NEW_ACC_DIGEST_BASE, STATE_LANE_NEW_SEMANTIC_STATE_BASE};
use crate::paper::f_prime::ring_action_trace::LowNormEncoding;

pub use crate::frontends::f_prime::projection_structure::{
    production_kmul_d2_ring_action_shell_image_config, production_kmul_ring_action_shell_image_config,
    production_projection_batches, PRODUCTION_KMUL_COUNT, PRODUCTION_PROJECTION_IDENTITY_COUNT,
    PRODUCTION_RING_ACTION_PAIR_COUNT,
};

const STATE_IN_DIGEST_COUNT: usize = 7;
const STATE_OUT_COUNTER_COUNT: usize = 2;
const STATE_OUT_DIGEST_COUNT: usize = 4;
const CHUNK_DIGEST_LANE_COUNT: usize = 4;
const KMUL_LANES_PER_SLOT: usize = 6;
const KMUL_SLOT_BITS: usize = KMUL_LANES_PER_SLOT * POSEIDON2_GOLDILOCKS_BITS;
const RING_ACTION_LANES_PER_PAIR: usize = 3 * D + D * D;
const RING_ACTION_PRODUCT_LANES_PER_PAIR: usize = D * D;
const RING_ACTION_OUTPUT_LANES_PER_PAIR: usize = D;
/// Rows deriving the base selector from the outgoing chunk counter.
const IS_BASE_COUNTER_LINK_ROWS: usize = 2;

/// One canonical-u64 lane: a 64-bit window inside the image's `values`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneSlot {
    pub bit_start: usize,
}

/// One app-private variable slot. R1CS-F' may use fewer than 64 committed
/// bits when the R1CS shape proves the variable is bounded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AppVariableSlot {
    pub bit_start: usize,
    pub bits: usize,
}

/// Recompose a canonical-u64 lane from its 64 committed bits.
pub(crate) fn lane_terms(slot: LaneSlot) -> impl Iterator<Item = (usize, F)> {
    (0..POSEIDON2_GOLDILOCKS_BITS).map(move |i| (slot.bit_start + i, F::from_u64(1u64 << i)))
}

/// Recompose an app variable from its committed slot.
pub(crate) fn app_variable_terms(slot: AppVariableSlot) -> Vec<(usize, F)> {
    assert!((1..=POSEIDON2_GOLDILOCKS_BITS).contains(&slot.bits));
    (0..slot.bits)
        .map(move |i| (slot.bit_start + i, F::from_u64(1u64 << i)))
        .collect()
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
#[derive(Clone, Debug)]
pub struct FPrimeLaneSlots {
    pub state_lanes: Vec<LaneSlot>,
    pub nifs_payload_lanes: Vec<Vec<LaneSlot>>,
    pub kmul_lanes: Vec<LaneSlot>,
    pub ring_action_lanes: Vec<LaneSlot>,
    pub projection_lanes: ProjectionLaneSlots,
    pub poseidon_trace_lanes: Vec<Vec<LaneSlot>>,
    pub sponge_transcript_lanes: Vec<LaneSlot>,
    pub public_x_out_binding_lanes: Vec<[LaneSlot; 4]>,
    pub app_assignment_lanes: Vec<AppVariableSlot>,
}

/// Enumerate every canonical-u64 lane the structure references.
pub fn f_prime_lane_slots(layout: &FPrimeImageLayout) -> FPrimeLaneSlots {
    FPrimeLaneSlots {
        state_lanes: collect_state_lane_slots(layout),
        nifs_payload_lanes: collect_nifs_payload_slots(layout),
        kmul_lanes: collect_kmul_slots(layout),
        ring_action_lanes: collect_ring_action_slots(layout),
        projection_lanes: collect_projection_slots(layout),
        poseidon_trace_lanes: collect_poseidon_trace_slots(layout),
        sponge_transcript_lanes: collect_sponge_transcript_slots(layout),
        public_x_out_binding_lanes: collect_public_x_out_binding_slots(layout),
        app_assignment_lanes: collect_app_assignment_lane_slots(layout),
    }
}

/// Enumerate `app_private` as app-variable slots when applicable.
fn collect_app_assignment_lane_slots(layout: &FPrimeImageLayout) -> Vec<AppVariableSlot> {
    if !layout.config.app_private_var_widths.is_empty() {
        let mut cursor = layout.app_private.offset;
        return layout
            .config
            .app_private_var_widths
            .iter()
            .map(|&bits| {
                let slot = AppVariableSlot {
                    bit_start: cursor,
                    bits,
                };
                cursor += bits;
                slot
            })
            .collect();
    }

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
        .map(|slot| AppVariableSlot {
            bit_start: slot.bit_start,
            bits: POSEIDON2_GOLDILOCKS_BITS,
        })
        .collect()
}

/// Columns that remain semantically Boolean in the F' image itself.
fn semantic_boolean_columns(layout: &FPrimeImageLayout) -> Vec<usize> {
    let mut cols = Vec::with_capacity(semantic_boolean_row_count(layout));
    cols.extend(layout.boundary.offset..layout.boundary.end());
    cols.extend(layout.is_base.offset..layout.is_base.end());
    if app_private_is_semantic_bits(layout) {
        cols.extend(layout.app_private.offset..layout.app_private.end());
    }
    cols
}

fn semantic_boolean_row_count(layout: &FPrimeImageLayout) -> usize {
    layout.boundary.bits
        + layout.is_base.bits
        + if app_private_is_semantic_bits(layout) {
            layout.app_private.bits
        } else {
            0
        }
}

fn app_private_is_semantic_bits(layout: &FPrimeImageLayout) -> bool {
    layout.config.app_private_var_widths.is_empty()
        && layout.app_private.bits > 0
        && layout.app_private.bits % POSEIDON2_GOLDILOCKS_BITS != 0
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
                //   m_in, fold_digest_fields(4 lanes).
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

/// Phase 1.4 CCS structure for one `enc(F')` step.
#[derive(Clone, Debug)]
pub struct FPrimeStructure {
    pub layout: FPrimeImageLayout,
    pub ccs: CcsStructure<F>,
    /// Lane-decode positions grouped by region.
    pub lane_slots: FPrimeLaneSlots,
}

/// Matrix-index assignment for the mixed-gate F' CCS polynomial.
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

/// Row builder for `f = (B²-B) + (Pl·Pr-Po) + (X⁷-Y) + (Ll-Lr)`.
pub(crate) struct MixedGateBuilder {
    trips: [Vec<(usize, usize, F)>; gate::ARITY],
    rows: usize,
}

impl MixedGateBuilder {
    pub(crate) fn with_estimated_rows(estimated_rows: usize) -> Self {
        // Bitness owns nearly all rows in large low-norm relations. Reserving
        // `estimated_rows` in every gate matrix multiplies peak memory by the
        // polynomial arity before a single coefficient is emitted.
        let mut trips = std::array::from_fn(|_| Vec::new());
        trips[gate::BITNESS] = Vec::with_capacity(estimated_rows);
        Self { trips, rows: 0 }
    }

    #[allow(dead_code)]
    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn bitness(&mut self, col: usize) -> usize {
        let row = self.rows;
        self.trips[gate::BITNESS].push((row, col, F::ONE));
        self.rows += 1;
        row
    }

    /// `(Σ left) · (Σ right) = (Σ out)`.
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

    /// `(Σ lhs) = (Σ rhs)`.
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

    /// `(Σ sbox_in)^7 = (Σ sbox_out)`.
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
        "FPrimeImageLayout::end = {image_end} too small; need constant slot + ≥1 image coordinate"
    );

    let lane_slots = f_prime_lane_slots(&layout);
    let mut builder = MixedGateBuilder::with_estimated_rows(estimated_shell_row_capacity(&layout));
    emit_shell_rows(&layout, &lane_slots, &mut builder);
    let ccs = builder.finish(image_end);

    FPrimeStructure {
        layout,
        ccs,
        lane_slots,
    }
}

fn estimated_shell_row_capacity(layout: &FPrimeImageLayout) -> usize {
    semantic_boolean_row_count(layout)
        + layout.carrier_padding.bits
        + IS_BASE_COUNTER_LINK_ROWS
        + layout.config.ring_action_pair_count
            * (RING_ACTION_PRODUCT_LANES_PER_PAIR + RING_ACTION_OUTPUT_LANES_PER_PAIR)
        + projection_semantic_row_count(&layout.config.projection_batches)
        + (layout.config.one_shot_digest_to_state_in_bindings.len()
            + layout.config.one_shot_digest_to_state_out_bindings.len()
            + layout.config.one_shot_digest_to_public_x_out_bindings.len())
            * POSEIDON2_DIGEST_LEN
        + if layout
            .config
            .one_shot_digest_to_public_x_out_bindings
            .is_empty()
        {
            0
        } else {
            POSEIDON2_DIGEST_LEN
        }
        + layout
            .config
            .unified_accumulator_selector
            .map_or(0, |_| POSEIDON2_DIGEST_LEN)
        + if has_stateless_state_x_out(&layout.config) {
            POSEIDON2_DIGEST_LEN
        } else {
            0
        }
        + layout
            .config
            .initial_semantic_state_digest_anchor
            .map_or(0, |_| POSEIDON2_DIGEST_LEN)
}

/// Emit every shell row the F' structure owns into `builder`.
pub(crate) fn emit_shell_rows(
    layout: &FPrimeImageLayout,
    lane_slots: &FPrimeLaneSlots,
    builder: &mut MixedGateBuilder,
) {
    let semantic_boolean_count = semantic_boolean_row_count(layout);
    let carrier_padding_count = layout.carrier_padding.bits;
    let control_count = semantic_boolean_count + carrier_padding_count + IS_BASE_COUNTER_LINK_ROWS;
    let ring_action_product_count = layout.config.ring_action_pair_count * RING_ACTION_PRODUCT_LANES_PER_PAIR;
    let ring_action_output_count = layout.config.ring_action_pair_count * RING_ACTION_OUTPUT_LANES_PER_PAIR;
    let projection_row_count = projection_semantic_row_count(&layout.config.projection_batches);
    let state_in_binding_count = layout.config.one_shot_digest_to_state_in_bindings.len() * POSEIDON2_DIGEST_LEN;
    let state_out_binding_count = layout.config.one_shot_digest_to_state_out_bindings.len() * POSEIDON2_DIGEST_LEN;
    // Canonical unified F' carries `new_z_i = chunk_digest` directly.
    let chunk_boundary_mirror_count = if layout
        .config
        .one_shot_digest_to_public_x_out_bindings
        .is_empty()
    {
        0
    } else {
        POSEIDON2_DIGEST_LEN
    };
    // Emit mirror rows when a `state_x_out` public binding is present.
    let public_trace_mirror_count = if layout
        .config
        .one_shot_digest_to_public_x_out_bindings
        .is_empty()
    {
        0
    } else {
        POSEIDON2_DIGEST_LEN
    };
    let public_x_out_binding_count =
        layout.config.one_shot_digest_to_public_x_out_bindings.len() * POSEIDON2_DIGEST_LEN;
    // Unified accumulator selector: 4 product rows, one per digest lane.
    let unified_selector_count = if layout.config.unified_accumulator_selector.is_some() {
        POSEIDON2_DIGEST_LEN
    } else {
        0
    };
    // Base-step semantic anchor:
    // `is_base * (state_in.semantic_state_digest_in[k] - anchor[k]) = 0`.
    let initial_semantic_anchor_count = if layout.config.initial_semantic_state_digest_anchor.is_some() {
        POSEIDON2_DIGEST_LEN
    } else {
        0
    };
    let stateless_semantic_acc_count = if has_stateless_state_x_out(&layout.config) {
        POSEIDON2_DIGEST_LEN
    } else {
        0
    };
    let total_shell_rows = control_count
        + ring_action_product_count
        + ring_action_output_count
        + projection_row_count
        + state_in_binding_count
        + state_out_binding_count
        + chunk_boundary_mirror_count
        + public_trace_mirror_count
        + public_x_out_binding_count
        + unified_selector_count
        + initial_semantic_anchor_count
        + stateless_semantic_acc_count;
    let base_row = builder.rows();

    // ── Semantic Boolean rows: `z[col] · (z[col] − 1) = 0`.
    for col in semantic_boolean_columns(layout) {
        builder.bitness(col);
    }
    debug_assert_eq!(builder.rows() - base_row, semantic_boolean_count);

    // ── Verifier-fixed SuperNeo public-carrier completion.
    for col in layout.carrier_padding.offset..layout.carrier_padding.end() {
        builder.linear([(col, F::ONE)], std::iter::empty::<(usize, F)>());
    }
    debug_assert_eq!(
        builder.rows() - base_row,
        semantic_boolean_count + carrier_padding_count
    );

    // ── Base selector ↔ post-step counter link and inverse witness.
    let is_base_col = layout.is_base.offset;
    let is_base_inv = LaneSlot {
        bit_start: layout.is_base.offset + 1,
    };
    let new_chunk_count = lane_terms(lane_slots.state_lanes[STATE_IN_DIGEST_COUNT * 4]);
    let count_minus_one: Vec<(usize, F)> = new_chunk_count
        .chain(std::iter::once((0, F::ZERO - F::ONE)))
        .collect();
    builder.product(count_minus_one.clone(), vec![(is_base_col, F::ONE)], Vec::new());
    builder.product(
        count_minus_one,
        lane_terms(is_base_inv),
        vec![(0, F::ONE), (is_base_col, F::ZERO - F::ONE)],
    );
    debug_assert_eq!(builder.rows() - base_row, control_count);

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
    debug_assert_eq!(builder.rows() - base_row, control_count + ring_action_product_count);

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
        control_count + ring_action_product_count + ring_action_output_count
    );

    emit_projection_semantic_rows(&layout.config.projection_batches, &lane_slots.projection_lanes, builder);
    debug_assert_eq!(
        builder.rows() - base_row,
        control_count + ring_action_product_count + ring_action_output_count + projection_row_count
    );

    // ── Trace digest ↔ state-in digest binding rows.
    for binding in &layout.config.one_shot_digest_to_state_in_bindings {
        let trace_slots = &lane_slots.poseidon_trace_lanes[binding.one_shot_index];
        let digest_lane_base = trace_slots.len() - POSEIDON2_WIDTH;
        let state_in_lane_base = state_in_digest_lane_base(binding.state_in_target);
        for lane in 0..POSEIDON2_DIGEST_LEN {
            let trace = trace_slots[digest_lane_base + lane];
            let state_in = lane_slots.state_lanes[state_in_lane_base + lane];
            builder.linear(lane_terms(trace), lane_terms(state_in));
        }
    }
    debug_assert_eq!(
        builder.rows() - base_row,
        control_count
            + ring_action_product_count
            + ring_action_output_count
            + projection_row_count
            + state_in_binding_count
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
        control_count
            + ring_action_product_count
            + ring_action_output_count
            + projection_row_count
            + state_in_binding_count
            + state_out_binding_count
    );

    // ── Canonical chunk-boundary mirror rows.
    if chunk_boundary_mirror_count != 0 {
        let new_z_i_lane_base = state_out_digest_lane_base(StateOutDigestTarget::NewZI);
        let chunk_digest_lane_base = STATE_IN_DIGEST_COUNT * 4 + STATE_OUT_COUNTER_COUNT + STATE_OUT_DIGEST_COUNT * 4;
        for lane in 0..POSEIDON2_DIGEST_LEN {
            let z_i = lane_slots.state_lanes[new_z_i_lane_base + lane];
            let chunk = lane_slots.state_lanes[chunk_digest_lane_base + lane];
            builder.linear(lane_terms(z_i), lane_terms(chunk));
        }
    }
    debug_assert_eq!(
        builder.rows() - base_row,
        control_count
            + ring_action_product_count
            + ring_action_output_count
            + projection_row_count
            + state_in_binding_count
            + state_out_binding_count
            + chunk_boundary_mirror_count
    );

    // ── Canonical public_trace mirror rows.
    //
    // `public_trace` is kept in the state/public image for now, but the
    // canonical F' transition sets `new_public_trace = new_z_i`. This
    // lets `state_x_out` avoid absorbing the same digest twice while
    // still constraining the retained state lane.
    if public_trace_mirror_count != 0 {
        let new_z_i_lane_base = state_out_digest_lane_base(StateOutDigestTarget::NewZI);
        let new_public_trace_lane_base = state_out_digest_lane_base(StateOutDigestTarget::NewPublicTrace);
        for lane in 0..POSEIDON2_DIGEST_LEN {
            let z_i = lane_slots.state_lanes[new_z_i_lane_base + lane];
            let public_trace = lane_slots.state_lanes[new_public_trace_lane_base + lane];
            builder.linear(lane_terms(public_trace), lane_terms(z_i));
        }
    }
    debug_assert_eq!(
        builder.rows() - base_row,
        control_count
            + ring_action_product_count
            + ring_action_output_count
            + projection_row_count
            + state_in_binding_count
            + state_out_binding_count
            + chunk_boundary_mirror_count
            + public_trace_mirror_count
    );

    // ── Unified-accumulator selector rows:
    // `(1 - is_base) · (recursive_lane - base) = new_acc_digest_lane - base`.
    if let Some(selector) = layout.config.unified_accumulator_selector {
        let is_base_col = layout.is_base.offset;
        let rec_trace_slots = &lane_slots.poseidon_trace_lanes[selector.recursive_trace_index];
        let rec_digest_lane_base = rec_trace_slots.len() - POSEIDON2_WIDTH;
        let acc_digest_lane_base = state_out_digest_lane_base(StateOutDigestTarget::NewAccDigest);

        for lane in 0..POSEIDON2_DIGEST_LEN {
            let base_const = selector.base_digest[lane];
            let rec_lane = rec_trace_slots[rec_digest_lane_base + lane];
            let acc_lane = lane_slots.state_lanes[acc_digest_lane_base + lane];

            let left: Vec<(usize, F)> = vec![(0, F::ONE), (is_base_col, F::ZERO - F::ONE)];
            let right: Vec<(usize, F)> = lane_terms(rec_lane)
                .chain(std::iter::once((0, F::ZERO - base_const)))
                .collect();
            let out: Vec<(usize, F)> = lane_terms(acc_lane)
                .chain(std::iter::once((0, F::ZERO - base_const)))
                .collect();
            builder.product(left, right, out);
        }
    }
    debug_assert_eq!(
        builder.rows() - base_row,
        control_count
            + ring_action_product_count
            + ring_action_output_count
            + projection_row_count
            + state_in_binding_count
            + state_out_binding_count
            + chunk_boundary_mirror_count
            + public_trace_mirror_count
            + unified_selector_count
    );

    // ── Stateless semantic/accumulator equality rows.
    //
    // When the state_x_out preimage omits `new_semantic_state_digest`, the
    // semantic coordinate is only sound because stateless mode requires it to
    // equal the outgoing accumulator handle. This row is the CCS-side
    // authority for that mode.
    if stateless_semantic_acc_count != 0 {
        let semantic_base = state_out_digest_lane_base(StateOutDigestTarget::NewSemanticStateDigest);
        let acc_base = state_out_digest_lane_base(StateOutDigestTarget::NewAccDigest);
        for lane in 0..POSEIDON2_DIGEST_LEN {
            let semantic = lane_slots.state_lanes[semantic_base + lane];
            let acc = lane_slots.state_lanes[acc_base + lane];
            builder.linear(lane_terms(semantic), lane_terms(acc));
        }
    }
    debug_assert_eq!(
        builder.rows() - base_row,
        control_count
            + ring_action_product_count
            + ring_action_output_count
            + projection_row_count
            + state_in_binding_count
            + state_out_binding_count
            + chunk_boundary_mirror_count
            + public_trace_mirror_count
            + unified_selector_count
            + stateless_semantic_acc_count
    );

    // ── Initial semantic-state anchor rows (base-step gated).
    if let Some(anchor_bytes) = layout.config.initial_semantic_state_digest_anchor {
        let is_base_col = layout.is_base.offset;
        let anchor_lanes = crate::paper::digest::digest32_as_fields(anchor_bytes);
        let semantic_in_lane_base = state_in_digest_lane_base(StateInDigestTarget::SemanticStateDigestIn);
        for lane in 0..POSEIDON2_DIGEST_LEN {
            let semantic_in_lane = lane_slots.state_lanes[semantic_in_lane_base + lane];
            let left: Vec<(usize, F)> = vec![(is_base_col, F::ONE)];
            let right: Vec<(usize, F)> = lane_terms(semantic_in_lane)
                .chain(std::iter::once((0, F::ZERO - anchor_lanes[lane])))
                .collect();
            let out: Vec<(usize, F)> = Vec::new();
            builder.product(left, right, out);
        }
    }
    debug_assert_eq!(
        builder.rows() - base_row,
        control_count
            + ring_action_product_count
            + ring_action_output_count
            + projection_row_count
            + state_in_binding_count
            + state_out_binding_count
            + chunk_boundary_mirror_count
            + public_trace_mirror_count
            + unified_selector_count
            + stateless_semantic_acc_count
            + initial_semantic_anchor_count
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

    // ── Optional Poseidon transition rows + variable preimage binding.
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

/// Lift non-bitness, non-skipped rows of a bit-backed Poseidon2 CCS.
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
                for k in csc.column_range(col) {
                    let r = csc.row_index(k);
                    let v = csc.vals[k];
                    entries[r].push((remap(col), v));
                }
            }
        }
        entries
    };

    // Native matrix indices: 0=B, 1=X, 2=Y, 3=Lhs, 4=Rhs.
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
    }
}

/// Resolve one preimage source to `(col, coeff)` terms in the F' bit-frame.
fn poseidon_preimage_source_terms(
    source: &PoseidonPreimageLaneSource,
    lane_slots: &FPrimeLaneSlots,
) -> Vec<(usize, F)> {
    match source {
        PoseidonPreimageLaneSource::Constant(v) => {
            if *v == F::ZERO {
                Vec::new()
            } else {
                vec![(0, *v)]
            }
        }
        PoseidonPreimageLaneSource::StateLane(i) => lane_terms(lane_slots.state_lanes[*i]).collect(),
        PoseidonPreimageLaneSource::StateLaneLowHalf(i) => lane_low_half_terms(lane_slots.state_lanes[*i]).collect(),
        PoseidonPreimageLaneSource::StateLaneHighHalf(i) => lane_high_half_terms(lane_slots.state_lanes[*i]).collect(),
        PoseidonPreimageLaneSource::NifsPayloadLane {
            payload_index,
            lane_index,
        } => lane_terms(lane_slots.nifs_payload_lanes[*payload_index][*lane_index]).collect(),
        PoseidonPreimageLaneSource::RingActionLane(i) => lane_terms(lane_slots.ring_action_lanes[*i]).collect(),
        PoseidonPreimageLaneSource::PoseidonTraceLane {
            trace_index,
            lane_index,
        } => lane_terms(lane_slots.poseidon_trace_lanes[*trace_index][*lane_index]).collect(),
        PoseidonPreimageLaneSource::SpongeTranscriptLane(i) => {
            lane_terms(lane_slots.sponge_transcript_lanes[*i]).collect()
        }
        PoseidonPreimageLaneSource::PublicXOutBindingLane { binding_index, lane } => {
            lane_terms(lane_slots.public_x_out_binding_lanes[*binding_index][*lane]).collect()
        }
        PoseidonPreimageLaneSource::AppAssignmentLane(var_index) => {
            app_variable_terms(lane_slots.app_assignment_lanes[*var_index])
        }
        PoseidonPreimageLaneSource::AppAssignmentBitPack(var_indices) => {
            let mut terms = Vec::new();
            let mut bit_weight = F::ONE;
            for &var_index in var_indices {
                terms.extend(
                    app_variable_terms(lane_slots.app_assignment_lanes[var_index])
                        .into_iter()
                        .map(|(col, coeff)| (col, coeff * bit_weight)),
                );
                bit_weight *= F::from_u64(2);
            }
            terms
        }
    }
}

fn has_stateless_state_x_out(config: &FPrimeImageConfig) -> bool {
    config
        .one_shot_digest_to_public_x_out_bindings
        .iter()
        .filter_map(|binding| {
            config
                .poseidon_transition_enforcements
                .iter()
                .find(|enforcement| enforcement.one_shot_index == binding.one_shot_index)
        })
        .any(|enforcement| {
            let mut absorbs_acc = false;
            let mut absorbs_semantic = false;
            for source in &enforcement.preimage_lanes {
                if let PoseidonPreimageLaneSource::StateLane(index) = source {
                    if (STATE_LANE_NEW_ACC_DIGEST_BASE..STATE_LANE_NEW_ACC_DIGEST_BASE + POSEIDON2_DIGEST_LEN)
                        .contains(index)
                    {
                        absorbs_acc = true;
                    }
                    if (STATE_LANE_NEW_SEMANTIC_STATE_BASE..STATE_LANE_NEW_SEMANTIC_STATE_BASE + POSEIDON2_DIGEST_LEN)
                        .contains(index)
                    {
                        absorbs_semantic = true;
                    }
                }
            }
            absorbs_acc && !absorbs_semantic
        })
}

/// Emit absorb-binding rows for one Poseidon transition enforcement.
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
            let post_word = absorb_idx * BIT_BACKED_PERMUTATION_WORDS + lane;
            let post_slot = trace_slots[post_word];

            let mut rhs: Vec<(usize, F)> = Vec::new();

            if absorb_idx > 0 {
                let prev_final_word = absorb_idx * BIT_BACKED_PERMUTATION_WORDS - POSEIDON2_WIDTH + lane;
                rhs.extend(lane_terms(trace_slots[prev_final_word]));
            }

            let is_padding_absorb = absorb_idx == absorbs - 1;
            if is_padding_absorb {
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
    /// exactly the committed image vector, with low-norm entries and
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

    /// Explicit semantic-bit rows, excluding internal low-norm digits.
    pub fn semantic_boolean_row_count(&self) -> usize {
        semantic_boolean_row_count(&self.layout)
    }

    /// Number of rows fixing public ring-completion coordinates to zero.
    pub fn carrier_padding_row_count(&self) -> usize {
        self.layout.carrier_padding.bits
    }

    /// Rows deriving `is_base` from `state_out.new_chunk_count`.
    pub fn is_base_counter_link_row_count(&self) -> usize {
        IS_BASE_COUNTER_LINK_ROWS
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

    /// First ring-action product row, after Boolean, padding, and control rows.
    pub fn ring_action_product_row_start(&self) -> usize {
        self.semantic_boolean_row_count() + self.carrier_padding_row_count() + IS_BASE_COUNTER_LINK_ROWS
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
        self.state_in_digest_binding_row_start() + self.state_in_digest_binding_row_count()
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

    /// Number of rows enforcing canonical `new_public_trace == new_z_i`.
    pub fn public_trace_mirror_row_count(&self) -> usize {
        if self
            .layout
            .config
            .one_shot_digest_to_public_x_out_bindings
            .is_empty()
        {
            0
        } else {
            POSEIDON2_DIGEST_LEN
        }
    }

    fn chunk_boundary_mirror_row_count(&self) -> usize {
        if self
            .layout
            .config
            .one_shot_digest_to_public_x_out_bindings
            .is_empty()
        {
            0
        } else {
            POSEIDON2_DIGEST_LEN
        }
    }

    fn chunk_boundary_mirror_row_start(&self) -> usize {
        self.state_out_digest_binding_row_start() + self.state_out_digest_binding_row_count()
    }

    /// First row of the canonical `new_public_trace == new_z_i` block.
    pub fn public_trace_mirror_row_start(&self) -> usize {
        self.chunk_boundary_mirror_row_start() + self.chunk_boundary_mirror_row_count()
    }

    /// Number of poseidon↔state_in digest binding rows
    /// (`POSEIDON2_DIGEST_LEN` per binding).
    pub fn state_in_digest_binding_row_count(&self) -> usize {
        self.layout
            .config
            .one_shot_digest_to_state_in_bindings
            .len()
            * POSEIDON2_DIGEST_LEN
    }

    /// First row of the poseidon↔state_in binding block.
    pub fn state_in_digest_binding_row_start(&self) -> usize {
        self.ring_action_output_row_start() + self.ring_action_output_row_count()
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
        self.public_trace_mirror_row_start()
            + self.public_trace_mirror_row_count()
            + self.unified_accumulator_selector_row_count()
            + self.stateless_semantic_acc_row_count()
            + self.initial_semantic_anchor_row_count()
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

    fn unified_accumulator_selector_row_count(&self) -> usize {
        if self.layout.config.unified_accumulator_selector.is_some() {
            POSEIDON2_DIGEST_LEN
        } else {
            0
        }
    }

    fn stateless_semantic_acc_row_count(&self) -> usize {
        if has_stateless_state_x_out(&self.layout.config) {
            POSEIDON2_DIGEST_LEN
        } else {
            0
        }
    }

    fn initial_semantic_anchor_row_count(&self) -> usize {
        if self
            .layout
            .config
            .initial_semantic_state_digest_anchor
            .is_some()
        {
            POSEIDON2_DIGEST_LEN
        } else {
            0
        }
    }
}

fn state_in_digest_lane_base(target: StateInDigestTarget) -> usize {
    match target {
        StateInDigestTarget::SemanticStateDigestIn => 16,
    }
}

fn state_out_digest_lane_base(target: StateOutDigestTarget) -> usize {
    const STATE_IN_LANES: usize = 28;
    const STATE_OUT_COUNTER_LANES: usize = 2;
    let state_out_digests_start = STATE_IN_LANES + STATE_OUT_COUNTER_LANES;
    match target {
        StateOutDigestTarget::NewZI => state_out_digests_start,
        StateOutDigestTarget::NewPublicTrace => state_out_digests_start + 4,
        StateOutDigestTarget::NewSemanticStateDigest => state_out_digests_start + 8,
        StateOutDigestTarget::NewAccDigest => state_out_digests_start + 12,
    }
}
