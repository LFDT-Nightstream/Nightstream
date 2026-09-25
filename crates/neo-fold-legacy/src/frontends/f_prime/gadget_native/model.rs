//! Internal source-decoder and synthetic-slot model.
//!
//! Owns: the typed inverse-map nodes used to reconstruct the source witness,
//! synthetic ring-field slots, and the validated trace marks shared by
//! estimation, profiling, and materialization.
//!
//! Does not own: classification precedence, slot allocation, row emission, or
//! public plan APIs.
//!
//! Emits constraints: no.
//!
//! Authority boundary: these values are internal products of exact trace and
//! schedule validation; no caller may construct them across the public API.
//!
//! | Type | Mathematical role | Constraint owner |
//! |---|---|---|
//! | `SourceColumn` | exact inverse map from encoded assignment to source witness | translated source rows |
//! | `RingSyntheticSlots` | materialized Toom-3 convolution coefficients | ring product-sum rows |
//! | `TraceMarks` | exclusive validated source row/column ownership | validation only |

use neo_math::F;

use crate::engine::r1cs_circuit::Lc;

use super::slots::ValueSlot;
use super::{acceptance, balanced_ternary, canonical_u64, mod5, product_sum, TOOM_COEFFICIENTS};

#[derive(Clone, Debug)]
pub(super) struct LinearDefinition {
    pub(super) terms: Vec<(usize, F)>,
    /// Removed generic source row from which these terms were solved. Gadget
    /// linears owned by another validated replacement have no such row.
    pub(super) source_row: Option<usize>,
}

#[derive(Clone, Debug)]
pub(super) struct ProductDefinition {
    pub(super) left: Lc,
    pub(super) right: Lc,
}

#[derive(Clone, Debug)]
pub(super) enum SourceColumn {
    One,
    Encoded(ValueSlot),
    EncodedLinear(Vec<(usize, F)>),
    Linear(LinearDefinition),
    GadgetLinear(LinearDefinition),
    Product(ProductDefinition),
    CanonicalNonzeroInverse(Lc),
}

#[derive(Clone, Debug)]
pub(super) struct RingSyntheticSlots {
    pub(super) coefficients: Vec<ValueSlot>,
}

impl RingSyntheticSlots {
    pub(super) fn coefficient(&self, evaluation: usize, coefficient: usize) -> ValueSlot {
        self.coefficients[evaluation * TOOM_COEFFICIENTS + coefficient]
    }
}

pub(super) struct TraceMarks {
    pub(super) covered_rows: Vec<bool>,
    pub(super) gadget_columns: Vec<bool>,
    pub(super) product_sums: product_sum::ValidatedProductSums,
    pub(super) balanced_ternary: balanced_ternary::ValidatedBalancedTernary,
    pub(super) canonical_u64: canonical_u64::ValidatedCanonicalU64,
    pub(super) acceptance: acceptance::ValidatedAcceptance,
    pub(super) mod5: mod5::ValidatedMod5,
}
