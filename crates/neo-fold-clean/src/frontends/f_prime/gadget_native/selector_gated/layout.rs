//! Cost-column layout for the formula-only fixed selector composition.
//!
//! Owns: one ordered, abutting range for every column family counted by the
//! fixed selector-gated estimate.
//!
//! Does not own: assignment materialization, constraint emission, branch
//! selection semantics, or proof authority.
//!
//! Emits constraints: no.
//!
//! Authority boundary: this layout only names estimated column positions. It
//! cannot authorize row removal or serve as `enc(F')` until an executable
//! materializer and selector-soundness proof instantiate the same ABI.
//!
//! | Column range | Mathematical value family | Width source |
//! |---|---|---|
//! | Constant | Multiplicative identity | one fixed column |
//! | One-bit | Public bits, selector, and branch-private Boolean values | combined estimate |
//! | Acceptance outputs | Aggregate acceptance tree outputs | branch estimates |
//! | Ordinary private | Shifted centered base-3 coordinates | combined estimate |
//! | SIS balanced | Balanced-ternary opening coordinates | combined estimate |
//! | Packed residues | Two centered Mod-5 residues per packed chunk | combined estimate |
//! | Canonical source | Raw bits then prefix auxiliaries for source fields | Boolean-family census |
//! | Canonical ring | Raw bits then prefix auxiliaries for ring fields | Boolean-family census |
//! | Canonical product-sum | Raw bits then prefix auxiliaries for product-sum fields | Boolean-family census |

use std::ops::Range;

use crate::engine::r1cs_circuit::builder::BALANCED_TERNARY_DIGITS;

use super::SelectorGatedGadgetNativeEstimate;

/// Ordered columns of the non-executable fixed selector estimate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectorGatedGadgetNativeCostLayout {
    pub constant_one: Range<usize>,
    pub one_bit: Range<usize>,
    pub acceptance_tree_outputs: Range<usize>,
    pub ordinary_private: Range<usize>,
    pub balanced_ternary: Range<usize>,
    pub packed_mod5_residues: Range<usize>,
    pub canonical_source_raw: Range<usize>,
    pub canonical_source_prefix: Range<usize>,
    pub canonical_ring_raw: Range<usize>,
    pub canonical_ring_prefix: Range<usize>,
    pub canonical_product_sum_raw: Range<usize>,
    pub canonical_product_sum_prefix: Range<usize>,
}

impl SelectorGatedGadgetNativeCostLayout {
    /// Derive the fixed column ABI from the exact component counts already
    /// owned by `estimate`; no independent total is accepted.
    pub fn from_estimate(estimate: &SelectorGatedGadgetNativeEstimate) -> Self {
        let mut cursor = 0;
        let constant_one = take(&mut cursor, 1);
        let one_bit = take(&mut cursor, estimate.one_bit_slots);
        let acceptance_tree_outputs = take(
            &mut cursor,
            estimate
                .base
                .acceptance_tree_output_cols
                .checked_add(estimate.recursive.acceptance_tree_output_cols)
                .expect("selector-gated acceptance-output width overflow"),
        );
        let ordinary_private = take(&mut cursor, estimate.ordinary_private_coordinates);
        let balanced_ternary = take(
            &mut cursor,
            estimate
                .balanced_ternary_field_slots
                .checked_mul(BALANCED_TERNARY_DIGITS)
                .expect("selector-gated balanced-ternary width overflow"),
        );
        let packed_mod5_residues = take(&mut cursor, estimate.packed_mod5_residue_coordinates);
        let canonical_source_raw = take(&mut cursor, estimate.boolean_pairing.source_raw64.coordinates);
        let canonical_source_prefix = take(&mut cursor, estimate.boolean_pairing.source_prefix31.coordinates);
        let canonical_ring_raw = take(&mut cursor, estimate.boolean_pairing.synthetic_ring_raw64.coordinates);
        let canonical_ring_prefix = take(
            &mut cursor,
            estimate.boolean_pairing.synthetic_ring_prefix31.coordinates,
        );
        let canonical_product_sum_raw = take(
            &mut cursor,
            estimate
                .boolean_pairing
                .synthetic_product_sum_raw64
                .coordinates,
        );
        let canonical_product_sum_prefix = take(
            &mut cursor,
            estimate
                .boolean_pairing
                .synthetic_product_sum_prefix31
                .coordinates,
        );

        assert_eq!(
            cursor, estimate.encoded_cols,
            "selector-gated column families must reconcile with the production estimate"
        );

        Self {
            constant_one,
            one_bit,
            acceptance_tree_outputs,
            ordinary_private,
            balanced_ternary,
            packed_mod5_residues,
            canonical_source_raw,
            canonical_source_prefix,
            canonical_ring_raw,
            canonical_ring_prefix,
            canonical_product_sum_raw,
            canonical_product_sum_prefix,
        }
    }

    /// All ABI ranges in physical column order.
    pub fn ordered_ranges(&self) -> [(&'static str, &Range<usize>); 12] {
        [
            ("constant_one", &self.constant_one),
            ("one_bit", &self.one_bit),
            ("acceptance_tree_outputs", &self.acceptance_tree_outputs),
            ("ordinary_private", &self.ordinary_private),
            ("balanced_ternary", &self.balanced_ternary),
            ("packed_mod5_residues", &self.packed_mod5_residues),
            ("canonical_source_raw", &self.canonical_source_raw),
            ("canonical_source_prefix", &self.canonical_source_prefix),
            ("canonical_ring_raw", &self.canonical_ring_raw),
            ("canonical_ring_prefix", &self.canonical_ring_prefix),
            ("canonical_product_sum_raw", &self.canonical_product_sum_raw),
            ("canonical_product_sum_prefix", &self.canonical_product_sum_prefix),
        ]
    }

    /// Complete fixed column span, derived from the first and final families.
    pub fn all_columns(&self) -> Range<usize> {
        self.constant_one.start..self.canonical_product_sum_prefix.end
    }
}

fn take(cursor: &mut usize, width: usize) -> Range<usize> {
    let start = *cursor;
    *cursor = cursor
        .checked_add(width)
        .expect("selector-gated column-layout cursor overflow");
    start..*cursor
}
