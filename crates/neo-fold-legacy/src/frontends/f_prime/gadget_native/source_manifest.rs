//! Exact source-column role census for one gadget-native branch.
//!
//! Owns: physical-stage ownership, adjacent `(stage, role)` runs, and exact
//! reconciliation of the shared production schedule with the estimator.
//!
//! Does not own: encoded-coordinate placement, CE placement, selector
//! composition, witness materialization, row removal, or semantic authority.
//!
//! Emits constraints: no.
//!
//! Authority boundary: source R1CS rows and validated production traces are
//! authoritative. A role census is diagnostic data and cannot authorize an
//! encoding change.
//!
//! | Source role | Shared schedule owner | Reconciliation |
//! |---|---|---|
//! | constant/public/private | source ABI and exact Boolean rows | source/public/one-bit totals |
//! | ordinary private field | exact 41-coordinate shifted centered word | ordinary-field and coordinate totals |
//! | canonical u64 | non-linear 95-coordinate canonical-binary source field | independent validated overlap census |
//! | SIS opening/structural alias | balanced-ternary trace | field/alias/binary totals |
//! | linear/gadget/product/temporary | exact source decoder variant | estimator linear/projected totals |

use std::ops::Range;

use thiserror::Error;

use crate::engine::r1cs_circuit::{R1csEncodingTrace, R1csSnapshot};

use super::coordinate_gates::PhysicalStageLayout;
use super::source_schedule::{GadgetNativeSourceRole, ValidatedSourceSchedule};
use super::{estimate_r1cs_gadget_native, GadgetNativeError};

const CONSTANT_ONE_STAGE: &str = "fprime.assignment.constant_one";

/// One maximal adjacent run with one physical source stage and one role.
#[derive(Clone, Debug, PartialEq, Eq)]
struct GadgetNativeSourceRun {
    stage: &'static str,
    role: GadgetNativeSourceRole,
    columns: Range<usize>,
}

/// Exact exclusive-role totals for one source relation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct GadgetNativeSourceRoleTotals {
    counts: [usize; 11],
}

impl GadgetNativeSourceRoleTotals {
    fn count(self, role: GadgetNativeSourceRole) -> usize {
        self.counts[role.index()]
    }

    fn total(self) -> usize {
        self.counts.into_iter().sum()
    }

    fn add(&mut self, role: GadgetNativeSourceRole, count: usize) {
        self.counts[role.index()] += count;
    }
}

/// Independent overlap census for validated canonical-u64 field columns.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct GadgetNativeCanonicalU64Overlap {
    traced_fields: usize,
    linearly_derived_fields: usize,
    direct_role_fields: usize,
}

/// Exact balanced-opening source roles kept separate from the exclusive total.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct GadgetNativeBalancedSourceCensus {
    opening_fields: usize,
    digit_aliases: usize,
    binary_columns: usize,
}

/// Source-only audit artifact for one concrete branch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GadgetNativeSourceManifest {
    source_columns: usize,
    runs: Vec<GadgetNativeSourceRun>,
    totals: GadgetNativeSourceRoleTotals,
    canonical_u64: GadgetNativeCanonicalU64Overlap,
    balanced: GadgetNativeBalancedSourceCensus,
}

impl GadgetNativeSourceManifest {
    pub fn source_columns(&self) -> usize {
        self.source_columns
    }

    pub fn run_count(&self) -> usize {
        self.runs.len()
    }

    pub fn run(&self, index: usize) -> Option<(&'static str, GadgetNativeSourceRole, Range<usize>)> {
        let run = self.runs.get(index)?;
        Some((run.stage, run.role, run.columns.clone()))
    }

    pub fn role_count(&self, role: GadgetNativeSourceRole) -> usize {
        self.totals.count(role)
    }

    pub fn role_for_column(&self, column: usize) -> Option<GadgetNativeSourceRole> {
        if column >= self.source_columns {
            return None;
        }
        self.runs
            .iter()
            .find(|run| run.columns.contains(&column))
            .map(|run| run.role)
    }

    /// `(all traced fields, generic-linear overlap, direct role fields)`.
    pub fn canonical_u64_overlap(&self) -> (usize, usize, usize) {
        (
            self.canonical_u64.traced_fields,
            self.canonical_u64.linearly_derived_fields,
            self.canonical_u64.direct_role_fields,
        )
    }

    /// `(opening fields, centered digit aliases, negative/borrow bits)`.
    pub fn balanced_source_census(&self) -> (usize, usize, usize) {
        (
            self.balanced.opening_fields,
            self.balanced.digit_aliases,
            self.balanced.binary_columns,
        )
    }

    /// Recheck the compact partition and every stored subtotal.
    pub fn validate(&self) -> Result<(), GadgetNativeSourceManifestError> {
        if self.source_columns == 0 || self.runs.is_empty() {
            return Err(partition("nonempty source census"));
        }
        let mut cursor = 0usize;
        let mut totals = GadgetNativeSourceRoleTotals::default();
        for (index, run) in self.runs.iter().enumerate() {
            if run.stage.is_empty() || run.columns.start != cursor || run.columns.is_empty() {
                return Err(partition("positive abutting source runs"));
            }
            if index > 0 {
                let prior = &self.runs[index - 1];
                if prior.stage == run.stage && prior.role == run.role {
                    return Err(partition("maximal adjacent stage-role runs"));
                }
            }
            totals.add(run.role, run.columns.len());
            cursor = run.columns.end;
        }
        if cursor != self.source_columns || totals != self.totals || totals.total() != self.source_columns {
            return Err(partition("source partition and role totals"));
        }
        let first = &self.runs[0];
        if first.stage != CONSTANT_ONE_STAGE
            || first.role != GadgetNativeSourceRole::ConstantOne
            || first.columns != (0..1)
            || totals.count(GadgetNativeSourceRole::ConstantOne) != 1
        {
            return Err(partition("constant-one source ABI"));
        }
        if self.canonical_u64.traced_fields
            != self.canonical_u64.linearly_derived_fields + self.canonical_u64.direct_role_fields
            || self.canonical_u64.direct_role_fields != totals.count(GadgetNativeSourceRole::CanonicalU64)
        {
            return Err(partition("canonical-u64 overlap census"));
        }
        if self.balanced.opening_fields != totals.count(GadgetNativeSourceRole::SisOpening)
            || self.balanced.digit_aliases != totals.count(GadgetNativeSourceRole::StructuralBalancedAlias)
            || self.balanced.binary_columns > totals.count(GadgetNativeSourceRole::PrivateBoolean)
        {
            return Err(partition("balanced-opening source census"));
        }
        Ok(())
    }
}

#[derive(Debug, Error)]
pub enum GadgetNativeSourceManifestError {
    #[error(transparent)]
    Lowering(#[from] GadgetNativeError),
    #[error("source-role manifest has invalid {detail}")]
    Partition { detail: &'static str },
    #[error("source-role manifest does not reconcile the production {family} census")]
    Census { family: &'static str },
}

/// Validate and classify every source column without allocating an encoded
/// witness or a combined fixed-relation coordinate arena.
pub fn audit_r1cs_gadget_native_source_manifest(
    source: &R1csSnapshot,
    trace: &R1csEncodingTrace,
    public_bit_columns: &[usize],
) -> Result<GadgetNativeSourceManifest, GadgetNativeSourceManifestError> {
    let schedule = ValidatedSourceSchedule::checked(source, trace, public_bit_columns)?;
    let canonical_census = schedule.canonical_u64().census;
    let balanced_binary_columns = schedule.balanced_binary_columns();
    let layout = PhysicalStageLayout::checked(source, trace)?;
    let mut roles = Vec::with_capacity(source.cols());
    for (column, decision) in schedule.decisions().iter().enumerate() {
        let stage = if column == 0 {
            CONSTANT_ONE_STAGE
        } else {
            layout.source_column_label(column)?
        };
        roles.push((stage, decision.role()));
    }

    let runs = compact_runs(&roles);
    let mut totals = GadgetNativeSourceRoleTotals::default();
    for &(_, role) in &roles {
        totals.add(role, 1);
    }
    let manifest = GadgetNativeSourceManifest {
        source_columns: source.cols(),
        runs,
        totals,
        canonical_u64: GadgetNativeCanonicalU64Overlap {
            traced_fields: canonical_census.total,
            linearly_derived_fields: canonical_census.field_linearly_derived,
            direct_role_fields: totals.count(GadgetNativeSourceRole::CanonicalU64),
        },
        balanced: GadgetNativeBalancedSourceCensus {
            opening_fields: trace.balanced_ternary_openings().len(),
            digit_aliases: totals.count(GadgetNativeSourceRole::StructuralBalancedAlias),
            binary_columns: balanced_binary_columns,
        },
    };
    manifest.validate()?;

    let estimate = estimate_r1cs_gadget_native(source, trace, public_bit_columns)?;
    require(
        totals.count(GadgetNativeSourceRole::PublicBit) == public_bit_columns.len(),
        "public-bit",
    )?;
    require(
        totals.count(GadgetNativeSourceRole::PublicBit) + totals.count(GadgetNativeSourceRole::PrivateBoolean)
            == estimate.one_bit_source_cols,
        "one-bit",
    )?;
    require(
        totals.count(GadgetNativeSourceRole::OrdinaryPrivateField) == estimate.ordinary_private_field_source_cols
            && estimate.ordinary_private_encoded_cols
                == estimate.ordinary_private_field_source_cols * super::ORDINARY_PRIVATE_DIGITS,
        "ordinary-private-field",
    )?;
    require(
        totals.count(GadgetNativeSourceRole::CanonicalU64) == estimate.canonical_binary_field_source_cols,
        "canonical-binary-field",
    )?;
    require(
        totals.count(GadgetNativeSourceRole::SisOpening) == estimate.balanced_ternary_field_source_cols
            && totals.count(GadgetNativeSourceRole::StructuralBalancedAlias)
                == estimate.balanced_ternary_alias_source_cols
            && manifest.balanced.binary_columns == estimate.balanced_ternary_binary_source_cols,
        "balanced-ternary",
    )?;
    require(
        totals.count(GadgetNativeSourceRole::LinearlyDerived) == estimate.linearly_derived_source_cols,
        "linearly-derived",
    )?;
    require(
        totals.count(GadgetNativeSourceRole::GadgetDerived)
            + totals.count(GadgetNativeSourceRole::ProductDerived)
            + totals.count(GadgetNativeSourceRole::GadgetTemporary)
            == estimate.gadget_derived_source_cols,
        "projected-gadget",
    )?;
    Ok(manifest)
}

fn compact_runs(roles: &[(&'static str, GadgetNativeSourceRole)]) -> Vec<GadgetNativeSourceRun> {
    let mut runs = Vec::<GadgetNativeSourceRun>::new();
    for (column, &(stage, role)) in roles.iter().enumerate() {
        if let Some(last) = runs.last_mut() {
            if last.stage == stage && last.role == role && last.columns.end == column {
                last.columns.end += 1;
                continue;
            }
        }
        runs.push(GadgetNativeSourceRun {
            stage,
            role,
            columns: column..column + 1,
        });
    }
    runs
}

fn require(condition: bool, family: &'static str) -> Result<(), GadgetNativeSourceManifestError> {
    condition
        .then_some(())
        .ok_or(GadgetNativeSourceManifestError::Census { family })
}

fn partition(detail: &'static str) -> GadgetNativeSourceManifestError {
    GadgetNativeSourceManifestError::Partition { detail }
}
