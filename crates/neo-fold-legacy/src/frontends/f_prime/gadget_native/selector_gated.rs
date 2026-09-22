//! Fixed base/recursive selector estimate for gadget-native lowering.
//!
//! Owns: cost accounting for combining two exact branch estimates with one
//! selector and the exact inactive binding for each private semantic value.
//!
//! Does not own: either branch's semantic reduction, selector-gated constraint
//! emission, a combined assignment materializer, or inactive-branch
//! satisfaction.
//!
//! Emits constraints: no. This module computes a formula-only estimate.
//!
//! Authority boundary: branch estimates are inputs to accounting only. Until a
//! combined materializer and selector-soundness proof exist, this estimate is
//! not an executable `enc(F')` relation and cannot authorize row removal.
//!
//! | Cost family | Exact owner |
//! |---|---|
//! | Shared public and selector coordinates | this module |
//! | Branch semantic rows | branch `GadgetNativeEstimate` |
//! | Ordinary-private coordinates | 41 centered coordinates per field, paired only with this family in the combined selector stage |
//! | Inactive ordinary-private values | one weighted decoded-field zero binding per 41-word; all 41 local centered checks remain |
//! | Inactive aggregate acceptance | fourteen output-zero bindings plus `accept = 1`; inverse is canonically derived and the executable bridge remains open |
//! | Inactive packed Mod-5 chunk | thirteen low-bit zero rows plus `(L+1)^2 - 7(R+1)^2 = 0` |

use crate::engine::r1cs_circuit::builder::BALANCED_TERNARY_DIGITS;
use crate::engine::r1cs_circuit::{R1csEncodingTrace, R1csSnapshot};

use super::mod5;
use super::slots::GOLDILOCKS_CANONICALITY_PAIR_ROWS;
use super::{
    estimate_r1cs_gadget_native, GadgetNativeBooleanPairingBreakdown, GadgetNativeError, GadgetNativeEstimate,
    GadgetNativePairTailCount, CANONICAL_SLOT_WIDTH, ORDINARY_PRIVATE_DIGITS,
};

mod layout;
pub use layout::SelectorGatedGadgetNativeCostLayout;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectorGatedGadgetNativeEstimate {
    pub public_input_len: usize,
    pub encoded_cols: usize,
    pub encoded_rows: usize,
    pub max_degree: u32,
    pub canonical_binary_field_slots: usize,
    pub ordinary_private_field_slots: usize,
    pub ordinary_private_coordinates: usize,
    /// One weighted source-value zero binding for every inactive 41-word.
    pub ordinary_private_inactive_binding_rows: usize,
    pub balanced_ternary_field_slots: usize,
    pub one_bit_slots: usize,
    /// Fifteen aggregate coordinates per chunk across both branches.
    pub acceptance_coordinates: usize,
    /// Formula-only bindings for fourteen zero tree outputs and `accept = 1`.
    pub acceptance_inactive_binding_rows: usize,
    /// Total packed chunks across both branch-private slot sets.
    pub packed_mod5_chunks: usize,
    /// Thirteen committed low quotient bits per packed chunk. These are part
    /// of `one_bit_slots`, but their active packed rows replace common bitness.
    pub packed_mod5_low_bit_slots: usize,
    /// Two centered residue coordinates per packed chunk. Their packed row
    /// replaces both common centered-unit gates.
    pub packed_mod5_residue_coordinates: usize,
    /// Formula-only inactive `bit = 0` bindings for the low quotient bits.
    pub packed_mod5_inactive_low_bit_rows: usize,
    /// Formula-only shifted nonresidue bindings forcing residue `(-1,-1)`.
    pub packed_mod5_inactive_residue_pair_rows: usize,
    /// Formula-owned single-stage pairing census for the combined slot set.
    pub boolean_pairing: GadgetNativeBooleanPairingBreakdown,
    /// Formula-owned single-stage centered residual pair/tail census.
    pub centered_pairing: GadgetNativePairTailCount,
    /// Disjoint ordinary-private child of `centered_pairing`.
    pub ordinary_private_centered_pairing: GadgetNativePairTailCount,
    /// Disjoint SIS-opening child of `centered_pairing`.
    pub sis_centered_pairing: GadgetNativePairTailCount,
    /// All formula-only inactive bindings, including acceptance and packed families.
    pub inactive_binding_rows: usize,
    pub base: GadgetNativeEstimate,
    pub recursive: GadgetNativeEstimate,
}

pub fn estimate_selector_gated_r1cs_gadget_native(
    base_source: &R1csSnapshot,
    base_trace: &R1csEncodingTrace,
    base_public_bit_columns: &[usize],
    recursive_source: &R1csSnapshot,
    recursive_trace: &R1csEncodingTrace,
    recursive_public_bit_columns: &[usize],
) -> Result<SelectorGatedGadgetNativeEstimate, GadgetNativeError> {
    if base_public_bit_columns.len() != recursive_public_bit_columns.len() {
        return Err(GadgetNativeError::BranchPublicInputLength {
            base: base_public_bit_columns.len(),
            recursive: recursive_public_bit_columns.len(),
        });
    }
    let base = estimate_r1cs_gadget_native(base_source, base_trace, base_public_bit_columns)?;
    let recursive = estimate_r1cs_gadget_native(recursive_source, recursive_trace, recursive_public_bit_columns)?;
    let public_bits = base_public_bit_columns.len();
    let public_input_len = super::canonical_superneo_public_input_len(public_bits)?;
    let public_padding = public_input_len - (1 + public_bits);
    let base_private_bits = base.one_bit_source_cols - public_bits;
    let recursive_private_bits = recursive.one_bit_source_cols - public_bits;
    let canonical_binary_field_slots = base
        .canonical_binary_field_source_cols
        .saturating_add(base.synthetic_ring_fields)
        .saturating_add(base.synthetic_product_sum_fields)
        .saturating_add(recursive.canonical_binary_field_source_cols)
        .saturating_add(recursive.synthetic_ring_fields)
        .saturating_add(recursive.synthetic_product_sum_fields);
    let ordinary_private_field_slots = base
        .ordinary_private_field_source_cols
        .saturating_add(recursive.ordinary_private_field_source_cols);
    let ordinary_private_coordinates = ordinary_private_field_slots.saturating_mul(ORDINARY_PRIVATE_DIGITS);
    let balanced_ternary_field_slots = base
        .balanced_ternary_field_source_cols
        .saturating_add(recursive.balanced_ternary_field_source_cols);
    let packed_mod5_chunks = base
        .packed_mod5_chunks
        .saturating_add(recursive.packed_mod5_chunks);
    let packed_mod5_low_bit_slots = packed_mod5_chunks.saturating_mul(mod5::LOW_BIT_COORDINATES_PER_CHUNK);
    let packed_mod5_residue_coordinates = packed_mod5_chunks.saturating_mul(mod5::RESIDUE_COORDINATES_PER_CHUNK);
    let acceptance_chunks = base
        .acceptance_chunks
        .saturating_add(recursive.acceptance_chunks);
    let acceptance_coordinates = base
        .acceptance_encoded_cols
        .saturating_add(recursive.acceptance_encoded_cols);
    let acceptance_tree_output_coordinates = base
        .acceptance_tree_output_cols
        .saturating_add(recursive.acceptance_tree_output_cols);
    let one_bit_slots = public_bits
        .saturating_add(1)
        .saturating_add(base_private_bits)
        .saturating_add(recursive_private_bits);
    let encoded_cols = 1usize
        .saturating_add(one_bit_slots)
        .saturating_add(public_padding)
        .saturating_add(acceptance_tree_output_coordinates)
        .saturating_add(ordinary_private_coordinates)
        .saturating_add(balanced_ternary_field_slots.saturating_mul(BALANCED_TERNARY_DIGITS))
        .saturating_add(packed_mod5_residue_coordinates)
        .saturating_add(canonical_binary_field_slots.saturating_mul(CANONICAL_SLOT_WIDTH));
    let base_canonical = base
        .canonical_binary_field_source_cols
        .saturating_add(base.synthetic_ring_fields)
        .saturating_add(base.synthetic_product_sum_fields);
    let recursive_canonical = recursive
        .canonical_binary_field_source_cols
        .saturating_add(recursive.synthetic_ring_fields)
        .saturating_add(recursive.synthetic_product_sum_fields);
    let base_coordinate_rows = base
        .boolean_pairing
        .total_rows()
        .saturating_add(base.centered_pairing.total_rows());
    let recursive_coordinate_rows = recursive
        .boolean_pairing
        .total_rows()
        .saturating_add(recursive.centered_pairing.total_rows());
    let base_semantic_rows = base
        .encoded_rows
        .saturating_sub(base_coordinate_rows)
        .saturating_sub(base_canonical.saturating_mul(GOLDILOCKS_CANONICALITY_PAIR_ROWS));
    let recursive_semantic_rows = recursive
        .encoded_rows
        .saturating_sub(recursive_coordinate_rows)
        .saturating_sub(recursive_canonical.saturating_mul(GOLDILOCKS_CANONICALITY_PAIR_ROWS));
    let packed_mod5_inactive_low_bit_rows = packed_mod5_low_bit_slots;
    let packed_mod5_inactive_residue_pair_rows = packed_mod5_chunks;
    let acceptance_inactive_binding_rows = acceptance_coordinates;
    let ordinary_private_inactive_binding_rows = ordinary_private_field_slots;
    let inactive_binding_rows = base_private_bits
        .saturating_add(recursive_private_bits)
        .saturating_sub(acceptance_chunks)
        .saturating_add(balanced_ternary_field_slots)
        .saturating_add(ordinary_private_inactive_binding_rows)
        .saturating_add(canonical_binary_field_slots)
        .saturating_add(acceptance_inactive_binding_rows)
        .saturating_add(packed_mod5_inactive_residue_pair_rows);
    let balanced_binary_coordinates = base
        .balanced_ternary_binary_source_cols
        .saturating_add(recursive.balanced_ternary_binary_source_cols);
    let specialized_common_coordinates = base
        .acceptance_encoded_cols
        .saturating_sub(base.acceptance_tree_output_cols)
        .saturating_add(
            recursive
                .acceptance_encoded_cols
                .saturating_sub(recursive.acceptance_tree_output_cols),
        )
        .saturating_add(
            base.packed_mod5_encoded_cols.saturating_sub(
                base.packed_mod5_chunks
                    .saturating_mul(mod5::RESIDUE_COORDINATES_PER_CHUNK),
            ),
        )
        .saturating_add(
            recursive.packed_mod5_encoded_cols.saturating_sub(
                recursive
                    .packed_mod5_chunks
                    .saturating_mul(mod5::RESIDUE_COORDINATES_PER_CHUNK),
            ),
        );
    let retained_common_coordinates = one_bit_slots
        .saturating_sub(balanced_binary_coordinates)
        .saturating_sub(specialized_common_coordinates);
    let boolean_pairing = GadgetNativeBooleanPairingBreakdown::one_stage(
        retained_common_coordinates,
        base.canonical_binary_field_source_cols
            .saturating_add(recursive.canonical_binary_field_source_cols),
        base.synthetic_ring_fields
            .saturating_add(recursive.synthetic_ring_fields),
        base.synthetic_product_sum_fields
            .saturating_add(recursive.synthetic_product_sum_fields),
    );
    let ordinary_private_centered_pairing = GadgetNativePairTailCount::from_coordinates(ordinary_private_coordinates);
    let sis_centered_pairing = GadgetNativePairTailCount::from_coordinates(
        balanced_ternary_field_slots.saturating_mul(BALANCED_TERNARY_DIGITS),
    );
    let mut centered_pairing = ordinary_private_centered_pairing;
    centered_pairing.add(sis_centered_pairing);
    let encoded_rows = boolean_pairing
        .total_rows()
        .saturating_add(centered_pairing.total_rows())
        .saturating_add(canonical_binary_field_slots.saturating_mul(GOLDILOCKS_CANONICALITY_PAIR_ROWS))
        .saturating_add(base_semantic_rows)
        .saturating_add(recursive_semantic_rows)
        .saturating_add(inactive_binding_rows);
    Ok(SelectorGatedGadgetNativeEstimate {
        public_input_len,
        encoded_cols,
        encoded_rows,
        max_degree: 8,
        canonical_binary_field_slots,
        ordinary_private_field_slots,
        ordinary_private_coordinates,
        ordinary_private_inactive_binding_rows,
        balanced_ternary_field_slots,
        one_bit_slots,
        acceptance_coordinates,
        acceptance_inactive_binding_rows,
        packed_mod5_chunks,
        packed_mod5_low_bit_slots,
        packed_mod5_residue_coordinates,
        packed_mod5_inactive_low_bit_rows,
        packed_mod5_inactive_residue_pair_rows,
        boolean_pairing,
        centered_pairing,
        ordinary_private_centered_pairing,
        sis_centered_pairing,
        inactive_binding_rows,
        base,
        recursive,
    })
}
