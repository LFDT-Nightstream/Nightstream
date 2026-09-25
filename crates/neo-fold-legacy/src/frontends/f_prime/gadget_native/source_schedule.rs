//! One exact source-column decision schedule for gadget-native lowering.
//!
//! Owns: validation, exclusive source ownership, production-precedence
//! decisions, role projection, and pointwise decoder-kind validation.
//!
//! Does not own: encoded-coordinate allocation, source-row emission, stage
//! compaction, estimator arithmetic, or permission to remove a constraint.
//!
//! Emits constraints: no.
//!
//! Authority boundary: the source R1CS and fully validated trace are
//! authoritative. Both production materialization and diagnostic manifests
//! consume this schedule; neither may reconstruct source roles independently.
//!
//! | Decision phase | Exact precedence | Materialized decoder | Public role |
//! |---|---|---|---|
//! | ABI | constant one, then validated public bits | one / Boolean slot | constant/public |
//! | Deferred projection | acceptance inverse, then packed Mod-5 roles | inverse / encoded-linear / product | temporary/linear/product |
//! | Structural opening | balanced field, then centered digit alias | balanced / centered slot | SIS opening/alias |
//! | General projection | product, batch-linear, then generic linear | product / gadget-linear / linear | product/gadget/linear |
//! | Retained scalar | exact private Boolean, direct canonical-u64, then explicit ordinary eligibility | Boolean / 95-coordinate canonical binary / 41-coordinate shifted centered | Boolean/u64/ordinary |

use crate::engine::r1cs_circuit::{R1csEncodingTrace, R1csSnapshot};

use super::acceptance;
use super::boolean_dedup::validate_public_columns;
use super::canonical_u64::CanonicalU64Audit;
use super::mod5;
use super::slots::ValueEncoding;
use super::{
    build_linear_definitions, build_product_definitions, reject_public_gadget_columns, validate_and_mark_trace,
    validate_source_one, GadgetNativeError, LinearDefinition, ProductDefinition, SourceColumn, TraceMarks,
};

/// Exhaustive source role shared with the Lean fixed-F-prime layout schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum GadgetNativeSourceRole {
    ConstantOne,
    OrdinaryPrivateField,
    PrivateBoolean,
    PublicBit,
    CanonicalU64,
    SisOpening,
    LinearlyDerived,
    StructuralBalancedAlias,
    GadgetDerived,
    ProductDerived,
    GadgetTemporary,
}

impl GadgetNativeSourceRole {
    pub const ALL: [Self; 11] = [
        Self::ConstantOne,
        Self::OrdinaryPrivateField,
        Self::PrivateBoolean,
        Self::PublicBit,
        Self::CanonicalU64,
        Self::SisOpening,
        Self::LinearlyDerived,
        Self::StructuralBalancedAlias,
        Self::GadgetDerived,
        Self::ProductDerived,
        Self::GadgetTemporary,
    ];

    pub(super) const fn index(self) -> usize {
        match self {
            Self::ConstantOne => 0,
            Self::OrdinaryPrivateField => 1,
            Self::PrivateBoolean => 2,
            Self::PublicBit => 3,
            Self::CanonicalU64 => 4,
            Self::SisOpening => 5,
            Self::LinearlyDerived => 6,
            Self::StructuralBalancedAlias => 7,
            Self::GadgetDerived => 8,
            Self::ProductDerived => 9,
            Self::GadgetTemporary => 10,
        }
    }

    #[doc(hidden)]
    pub const fn lean_role_count_field(self) -> &'static str {
        match self {
            Self::ConstantOne => "constantOne",
            Self::OrdinaryPrivateField => "ordinaryPrivateField",
            Self::PrivateBoolean => "privateBoolean",
            Self::PublicBit => "publicBit",
            Self::CanonicalU64 => "canonicalU64",
            Self::SisOpening => "sisOpening",
            Self::LinearlyDerived => "linearlyDerived",
            Self::StructuralBalancedAlias => "structuralBalancedAlias",
            Self::GadgetDerived => "gadgetDerived",
            Self::ProductDerived => "productDerived",
            Self::GadgetTemporary => "gadgetTemporary",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PrivateBooleanOrigin {
    Explicit,
    AcceptanceAccept,
    Mod5LowQuotient,
    BalancedNegative,
    BalancedBorrow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CanonicalFieldKind {
    OrdinaryPrivate,
    DirectCanonicalU64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Mod5LinearKind {
    Index,
    Quotient,
    HighBit,
}

#[derive(Clone, Debug)]
pub(super) enum ProjectedColumnDecision {
    Product(ProductDefinition),
    GadgetLinear(LinearDefinition),
    AcceptanceInverse,
    Mod5Linear(Mod5LinearKind),
    Mod5Product,
}

impl ProjectedColumnDecision {
    const fn materialization_kind(&self) -> SourceMaterializationKind {
        match self {
            Self::Product(_) | Self::Mod5Product => SourceMaterializationKind::Product,
            Self::GadgetLinear(_) => SourceMaterializationKind::GadgetLinear,
            Self::AcceptanceInverse => SourceMaterializationKind::NonzeroInverse,
            Self::Mod5Linear(kind) => match kind {
                Mod5LinearKind::Index | Mod5LinearKind::Quotient | Mod5LinearKind::HighBit => {
                    SourceMaterializationKind::EncodedLinear
                }
            },
        }
    }
}

#[derive(Clone, Debug)]
pub(super) enum SourceColumnDecision {
    ConstantOne,
    PublicBit,
    PrivateBoolean(PrivateBooleanOrigin),
    BalancedOpening { opening: usize },
    BalancedDigitAlias { field: usize, digit: usize },
    CanonicalField(CanonicalFieldKind),
    GenericLinear(LinearDefinition),
    Projected(ProjectedColumnDecision),
}

impl SourceColumnDecision {
    pub(super) const fn role(&self) -> GadgetNativeSourceRole {
        match self {
            Self::ConstantOne => GadgetNativeSourceRole::ConstantOne,
            Self::PublicBit => GadgetNativeSourceRole::PublicBit,
            Self::PrivateBoolean(_) => GadgetNativeSourceRole::PrivateBoolean,
            Self::BalancedOpening { .. } => GadgetNativeSourceRole::SisOpening,
            Self::BalancedDigitAlias { .. } => GadgetNativeSourceRole::StructuralBalancedAlias,
            Self::CanonicalField(CanonicalFieldKind::OrdinaryPrivate) => GadgetNativeSourceRole::OrdinaryPrivateField,
            Self::CanonicalField(CanonicalFieldKind::DirectCanonicalU64) => GadgetNativeSourceRole::CanonicalU64,
            Self::GenericLinear(_) | Self::Projected(ProjectedColumnDecision::Mod5Linear(_)) => {
                GadgetNativeSourceRole::LinearlyDerived
            }
            Self::Projected(ProjectedColumnDecision::GadgetLinear(_)) => GadgetNativeSourceRole::GadgetDerived,
            Self::Projected(ProjectedColumnDecision::Product(_) | ProjectedColumnDecision::Mod5Product) => {
                GadgetNativeSourceRole::ProductDerived
            }
            Self::Projected(ProjectedColumnDecision::AcceptanceInverse) => GadgetNativeSourceRole::GadgetTemporary,
        }
    }

    const fn materialization_kind(&self) -> SourceMaterializationKind {
        match self {
            Self::ConstantOne => SourceMaterializationKind::One,
            Self::PublicBit => SourceMaterializationKind::Boolean,
            Self::PrivateBoolean(origin) => match origin {
                PrivateBooleanOrigin::Explicit
                | PrivateBooleanOrigin::AcceptanceAccept
                | PrivateBooleanOrigin::Mod5LowQuotient
                | PrivateBooleanOrigin::BalancedNegative
                | PrivateBooleanOrigin::BalancedBorrow => SourceMaterializationKind::Boolean,
            },
            Self::BalancedOpening { .. } => SourceMaterializationKind::Balanced,
            Self::BalancedDigitAlias { .. } => SourceMaterializationKind::CenteredAlias,
            Self::CanonicalField(CanonicalFieldKind::OrdinaryPrivate) => {
                SourceMaterializationKind::OrdinaryCenteredTernary
            }
            Self::CanonicalField(CanonicalFieldKind::DirectCanonicalU64) => SourceMaterializationKind::CanonicalBinary,
            Self::GenericLinear(_) => SourceMaterializationKind::Linear,
            Self::Projected(projected) => projected.materialization_kind(),
        }
    }

    const fn is_projected(&self) -> bool {
        matches!(self, Self::Projected(_))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SourceMaterializationKind {
    One,
    Boolean,
    Balanced,
    CenteredAlias,
    OrdinaryCenteredTernary,
    CanonicalBinary,
    Linear,
    GadgetLinear,
    Product,
    NonzeroInverse,
    EncodedLinear,
}

/// Validated, exclusive schedule consumed by both production and audit paths.
pub(super) struct ValidatedSourceSchedule {
    pub(super) is_public: Vec<bool>,
    pub(super) explicit_bits: Vec<bool>,
    pub(super) marks: TraceMarks,
    decisions: Vec<SourceColumnDecision>,
    pub(super) removed_definition_rows: Vec<bool>,
    canonical_u64: CanonicalU64Audit,
    balanced_binary_columns: usize,
}

impl ValidatedSourceSchedule {
    pub(super) fn checked(
        source: &R1csSnapshot,
        trace: &R1csEncodingTrace,
        public_bit_columns: &[usize],
    ) -> Result<Self, GadgetNativeError> {
        validate_source_one(source)?;
        let (is_public, explicit_bits) = validate_public_columns(source, public_bit_columns)?;
        let marks = validate_and_mark_trace(source, trace)?;
        reject_public_gadget_columns(&marks.gadget_columns, &is_public)?;
        marks.balanced_ternary.reject_public_columns(&is_public)?;

        let (mut products, mut gadget_linears) = build_product_definitions(source, trace, &marks.product_sums)?;
        let (mut generic_linears, removed_definition_rows) = build_linear_definitions(source, &is_public, &marks);
        let generic_linear_mask = generic_linears
            .iter()
            .map(Option::is_some)
            .collect::<Vec<_>>();
        let canonical_u64 = marks
            .canonical_u64
            .report(source, trace, &generic_linear_mask)?;
        let direct_canonical_u64 = direct_canonical_u64_columns(
            source,
            &canonical_u64,
            &is_public,
            &explicit_bits,
            &generic_linear_mask,
            &marks,
        )?;
        let boolean_origins = private_boolean_origins(source, trace, &explicit_bits)?;
        let mut special_projected = special_projected_decisions(source, trace, &marks)?;

        for column in 1..source.cols() {
            let owners = usize::from(special_projected[column].is_some())
                + usize::from(products[column].is_some())
                + usize::from(gadget_linears[column].is_some());
            if owners > 1 || marks.gadget_columns[column] != (owners == 1) {
                return Err(GadgetNativeError::SourceDecisionConflict {
                    column,
                    detail: "projected gadget ownership",
                });
            }
            if marks.balanced_ternary.is_structural(column) && owners != 0 {
                return Err(GadgetNativeError::SourceDecisionConflict {
                    column,
                    detail: "balanced/projected ownership overlap",
                });
            }
        }

        let mut decisions = Vec::with_capacity(source.cols());
        decisions.push(SourceColumnDecision::ConstantOne);
        for column in 1..source.cols() {
            let decision = if is_public[column] {
                SourceColumnDecision::PublicBit
            } else if let Some(projected) = special_projected[column].take() {
                SourceColumnDecision::Projected(projected)
            } else if let Some(opening) = marks.balanced_ternary.opening_for_field(column) {
                SourceColumnDecision::BalancedOpening { opening }
            } else if let Some((field, digit)) = marks.balanced_ternary.digit_alias(column) {
                SourceColumnDecision::BalancedDigitAlias { field, digit }
            } else if let Some(definition) = products[column].take() {
                SourceColumnDecision::Projected(ProjectedColumnDecision::Product(definition))
            } else if let Some(definition) = gadget_linears[column].take() {
                SourceColumnDecision::Projected(ProjectedColumnDecision::GadgetLinear(definition))
            } else if let Some(definition) = generic_linears[column].take() {
                SourceColumnDecision::GenericLinear(definition)
            } else if let Some(origin) = boolean_origins[column] {
                SourceColumnDecision::PrivateBoolean(origin)
            } else if direct_canonical_u64[column] {
                SourceColumnDecision::CanonicalField(CanonicalFieldKind::DirectCanonicalU64)
            } else if ordinary_private_eligible(
                column,
                &is_public,
                &explicit_bits,
                &generic_linear_mask,
                &direct_canonical_u64,
                &marks,
            ) {
                SourceColumnDecision::CanonicalField(CanonicalFieldKind::OrdinaryPrivate)
            } else {
                return Err(GadgetNativeError::UnclassifiedSourceColumn { column });
            };
            if marks.gadget_columns[column] != decision.is_projected() {
                return Err(GadgetNativeError::UnclassifiedProjectedSourceColumn { column });
            }
            decisions.push(decision);
        }

        Ok(Self {
            is_public,
            explicit_bits,
            marks,
            decisions,
            removed_definition_rows,
            canonical_u64,
            balanced_binary_columns: trace
                .balanced_ternary_openings()
                .iter()
                .map(|opening| opening.negative_cols.len() + opening.borrow_cols.len())
                .sum(),
        })
    }

    pub(super) fn decisions(&self) -> &[SourceColumnDecision] {
        &self.decisions
    }

    pub(super) fn canonical_u64(&self) -> &CanonicalU64Audit {
        &self.canonical_u64
    }

    pub(super) fn balanced_binary_columns(&self) -> usize {
        self.balanced_binary_columns
    }

    pub(super) fn into_materialization(self) -> SourceMaterializationSchedule {
        let roles = self
            .decisions
            .iter()
            .map(SourceColumnDecision::role)
            .collect();
        let expected = self
            .decisions
            .iter()
            .map(SourceColumnDecision::materialization_kind)
            .collect();
        SourceMaterializationSchedule {
            is_public: self.is_public,
            explicit_bits: self.explicit_bits,
            marks: self.marks,
            decisions: self.decisions,
            roles,
            expected,
            removed_definition_rows: self.removed_definition_rows,
        }
    }
}

pub(super) struct SourceMaterializationSchedule {
    pub(super) is_public: Vec<bool>,
    pub(super) explicit_bits: Vec<bool>,
    pub(super) marks: TraceMarks,
    pub(super) decisions: Vec<SourceColumnDecision>,
    pub(super) roles: Vec<GadgetNativeSourceRole>,
    expected: Vec<SourceMaterializationKind>,
    pub(super) removed_definition_rows: Vec<bool>,
}

impl SourceMaterializationSchedule {
    pub(super) fn validate_materialized(&self, source_columns: &[SourceColumn]) -> Result<(), GadgetNativeError> {
        if source_columns.len() != self.expected.len() {
            return Err(GadgetNativeError::SourceDecisionWidth {
                expected: self.expected.len(),
                got: source_columns.len(),
            });
        }
        for (column, (&expected, source_column)) in self.expected.iter().zip(source_columns).enumerate() {
            if !materialization_matches(expected, source_column) {
                return Err(GadgetNativeError::SourceMaterializationMismatch { column });
            }
        }
        Ok(())
    }
}

/// The ordinary case is an explicit residual classification, never a catch-all.
/// It requires a private, nonprojected, non-linear, nonstructural, non-Boolean,
/// non-canonical-u64 source column. Any new marked/projected family therefore
/// fails before it can reach ordinary materialization.
fn ordinary_private_eligible(
    column: usize,
    is_public: &[bool],
    explicit_bits: &[bool],
    generic_linears: &[bool],
    direct_canonical_u64: &[bool],
    marks: &TraceMarks,
) -> bool {
    column != 0
        && !is_public[column]
        && !marks.gadget_columns[column]
        && !generic_linears[column]
        && !marks.balanced_ternary.is_structural(column)
        && !explicit_bits[column]
        && !direct_canonical_u64[column]
}

fn direct_canonical_u64_columns(
    source: &R1csSnapshot,
    audit: &CanonicalU64Audit,
    is_public: &[bool],
    explicit_bits: &[bool],
    generic_linears: &[bool],
    marks: &TraceMarks,
) -> Result<Vec<bool>, GadgetNativeError> {
    let mut direct = vec![false; source.cols()];
    for entry in &audit.entries {
        let column = entry.field_column;
        if entry.field_linearly_derived {
            if !generic_linears[column] {
                return Err(GadgetNativeError::SourceDecisionConflict {
                    column,
                    detail: "canonical-u64 linear overlay",
                });
            }
            continue;
        }
        if is_public[column]
            || explicit_bits[column]
            || marks.gadget_columns[column]
            || marks.balanced_ternary.is_structural(column)
            || generic_linears[column]
            || std::mem::replace(&mut direct[column], true)
        {
            return Err(GadgetNativeError::SourceDecisionConflict {
                column,
                detail: "direct canonical-u64 ownership",
            });
        }
    }
    Ok(direct)
}

fn private_boolean_origins(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
    explicit_bits: &[bool],
) -> Result<Vec<Option<PrivateBooleanOrigin>>, GadgetNativeError> {
    let mut origins = vec![None; source.cols()];
    for event in trace.acceptance_chunks() {
        install_boolean_origin(&mut origins, event.accept.col(), PrivateBooleanOrigin::AcceptanceAccept)?;
    }
    for event in trace.mod5_chunks() {
        for bit in &event.quotient_bits[..mod5::LOW_QUOTIENT_BITS] {
            install_boolean_origin(&mut origins, bit.col(), PrivateBooleanOrigin::Mod5LowQuotient)?;
        }
    }
    for opening in trace.balanced_ternary_openings() {
        for &column in &opening.negative_cols {
            install_boolean_origin(&mut origins, column, PrivateBooleanOrigin::BalancedNegative)?;
        }
        for &column in &opening.borrow_cols {
            install_boolean_origin(&mut origins, column, PrivateBooleanOrigin::BalancedBorrow)?;
        }
    }
    for column in 1..source.cols() {
        if origins[column].is_some_and(|origin| {
            !matches!(
                origin,
                PrivateBooleanOrigin::BalancedNegative | PrivateBooleanOrigin::BalancedBorrow
            )
        }) && !explicit_bits[column]
        {
            return Err(GadgetNativeError::SourceDecisionConflict {
                column,
                detail: "specialized Boolean lacks exact source row",
            });
        }
        if origins[column].is_none() && explicit_bits[column] {
            origins[column] = Some(PrivateBooleanOrigin::Explicit);
        }
    }
    Ok(origins)
}

fn install_boolean_origin(
    origins: &mut [Option<PrivateBooleanOrigin>],
    column: usize,
    origin: PrivateBooleanOrigin,
) -> Result<(), GadgetNativeError> {
    if column == 0 || column >= origins.len() || origins[column].replace(origin).is_some() {
        return Err(GadgetNativeError::SourceDecisionConflict {
            column,
            detail: "private Boolean ownership",
        });
    }
    Ok(())
}

fn special_projected_decisions(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
    marks: &TraceMarks,
) -> Result<Vec<Option<ProjectedColumnDecision>>, GadgetNativeError> {
    let mut projected = vec![None; source.cols()];
    for event in trace.acceptance_chunks() {
        install_projected(
            &mut projected,
            event.inverse.col(),
            ProjectedColumnDecision::AcceptanceInverse,
        )?;
    }
    for event in trace.mod5_chunks() {
        for (column, kind) in [
            (event.index.col(), Mod5LinearKind::Index),
            (event.quotient.col(), Mod5LinearKind::Quotient),
            (
                event.quotient_bits[mod5::LOW_QUOTIENT_BITS].col(),
                Mod5LinearKind::HighBit,
            ),
        ] {
            install_projected(&mut projected, column, ProjectedColumnDecision::Mod5Linear(kind))?;
        }
        for product in event.index_products {
            install_projected(&mut projected, product.col(), ProjectedColumnDecision::Mod5Product)?;
        }
    }
    for (column, decision) in projected.iter().enumerate().skip(1) {
        let validated = match (
            marks.acceptance.projected_role(column),
            marks.mod5.projected_role(column),
        ) {
            (None, None) => None,
            (Some(acceptance::ProjectedRole::CanonicalNonzeroInverse), None) => {
                Some(SourceMaterializationKind::NonzeroInverse)
            }
            (None, Some(mod5::ProjectedRole::EncodedLinear { .. })) => Some(SourceMaterializationKind::EncodedLinear),
            (None, Some(mod5::ProjectedRole::Product)) => Some(SourceMaterializationKind::Product),
            (Some(_), Some(_)) => {
                return Err(GadgetNativeError::SourceDecisionConflict {
                    column,
                    detail: "acceptance/Mod-5 projected overlap",
                });
            }
        };
        if validated
            != decision
                .as_ref()
                .map(ProjectedColumnDecision::materialization_kind)
        {
            return Err(GadgetNativeError::UnclassifiedProjectedSourceColumn { column });
        }
    }
    Ok(projected)
}

fn install_projected(
    projected: &mut [Option<ProjectedColumnDecision>],
    column: usize,
    decision: ProjectedColumnDecision,
) -> Result<(), GadgetNativeError> {
    if column == 0 || column >= projected.len() || projected[column].replace(decision).is_some() {
        return Err(GadgetNativeError::SourceDecisionConflict {
            column,
            detail: "special projected ownership",
        });
    }
    Ok(())
}

fn materialization_matches(expected: SourceMaterializationKind, source_column: &SourceColumn) -> bool {
    match (expected, source_column) {
        (SourceMaterializationKind::One, SourceColumn::One) => true,
        (SourceMaterializationKind::Boolean, SourceColumn::Encoded(slot)) => {
            slot.width == 1 && slot.encoding == ValueEncoding::Boolean
        }
        (SourceMaterializationKind::Balanced, SourceColumn::Encoded(slot)) => {
            slot.encoding == ValueEncoding::BalancedTernary
        }
        (SourceMaterializationKind::CenteredAlias, SourceColumn::Encoded(slot)) => {
            slot.width == 1 && slot.encoding == ValueEncoding::CenteredUnit
        }
        (SourceMaterializationKind::OrdinaryCenteredTernary, SourceColumn::Encoded(slot)) => {
            slot.width == super::ordinary_private_field::ORDINARY_PRIVATE_DIGITS
                && slot.encoding == ValueEncoding::OrdinaryCenteredTernary
        }
        (SourceMaterializationKind::CanonicalBinary, SourceColumn::Encoded(slot)) => {
            matches!(slot.encoding, ValueEncoding::CanonicalBinary { .. })
        }
        (SourceMaterializationKind::Linear, SourceColumn::Linear(_))
        | (SourceMaterializationKind::GadgetLinear, SourceColumn::GadgetLinear(_))
        | (SourceMaterializationKind::Product, SourceColumn::Product(_))
        | (SourceMaterializationKind::NonzeroInverse, SourceColumn::CanonicalNonzeroInverse(_))
        | (SourceMaterializationKind::EncodedLinear, SourceColumn::EncodedLinear(_)) => true,
        _ => false,
    }
}
