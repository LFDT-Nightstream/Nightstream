//! Exact selector-port coverage of a compiled selective CCS relation.
//!
//! Owns: the exhaustive emitted-family-to-gate map, direct validation of the
//! final general/evaluation selector matrices, selector homogeneity of the
//! actual sparse polynomial, and a run-compressed coverage certificate joined
//! to the compiler's exclusive row ledger.
//!
//! Does not own: branch semantics, source-row rewrite correctness, selector
//! total soundness, constant-one connectivity, constraint necessity, or row
//! removal.
//!
//! Emits constraints: no. This is a read-only refinement audit over an already
//! compiled relation. Runtime work is linear in the two selector-port column
//! pointer arrays, their nonzero entries, the owner runs, and 27 polynomial
//! terms; it never scans the other eleven matrices or materializes one item per
//! constraint row.
//!
//! Authority boundary: family labels determine only the expected support. The
//! certificate is returned only after the two final matrix ports match that
//! expectation coefficient-for-coefficient and every actual polynomial term
//! contains exactly one selector linearly. The exact small polynomial syntax
//! is retained for an independent Lean comparison. Compact selector-port
//! matrices, malformed arm labels, unexpected columns, gaps, and overlaps fail
//! closed.
//!
//! | Stage path | Expected final port | Expected column |
//! |---|---|---|
//! | `f_prime.selective_ccs.common.*` | general selector | constant-one column 0 |
//! | `f_prime.selective_ccs.arm.{domain,retained,poseidon2,centered,canonical}` | general selector | arm selector |
//! | `f_prime.selective_ccs.arm.{evaluation,product_sum}` | evaluation selector | arm selector |

use core::ops::Range;
use std::collections::BTreeMap;

use neo_ccs::{CcsMatrix, CscMat};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

use super::lowering::SelectiveLowNormSnapshot;
use super::selective_audit::{SelectiveEmittedRowFamily, SelectiveEmittedRowRunAudit};
use super::selective_census::SelectiveMatrixTag;

const SELECTIVE_PORT_COUNT: usize = 13;
const SELECTIVE_POLYNOMIAL_TERM_COUNT: usize = 27;
const GENERAL_SELECTOR_PORT: usize = 1;
const EVALUATION_SELECTOR_PORT: usize = 7;

/// Version of the run-compressed Rust-to-Lean selector-coverage wire format.
pub const SELECTIVE_SELECTOR_GATE_COVERAGE_SCHEMA_VERSION: usize = 1;

/// The only two polynomial ports allowed to activate an emitted row.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SelectiveGatePort {
    General,
    Evaluation,
}

impl SelectiveGatePort {
    fn matrix_index(self) -> usize {
        match self {
            Self::General => GENERAL_SELECTOR_PORT,
            Self::Evaluation => EVALUATION_SELECTOR_PORT,
        }
    }
}

/// One nonempty selector-support interval aligned to one nonempty owner run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveSelectorGateRun {
    emitted_rows: Range<usize>,
    port: SelectiveGatePort,
    column: usize,
    coefficient: F,
}

/// One literal term read from the final relation's sparse polynomial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveSelectorPolynomialTerm {
    coefficient: F,
    exponents: Vec<u32>,
}

impl SelectiveSelectorPolynomialTerm {
    pub fn coefficient(&self) -> F {
        self.coefficient
    }

    pub fn exponents(&self) -> &[u32] {
        &self.exponents
    }
}

impl SelectiveSelectorGateRun {
    pub fn emitted_rows(&self) -> Range<usize> {
        self.emitted_rows.clone()
    }

    pub fn port(&self) -> SelectiveGatePort {
        self.port
    }

    pub fn column(&self) -> usize {
        self.column
    }

    pub fn coefficient(&self) -> F {
        self.coefficient
    }
}

/// Run-compressed proof object returned after final-matrix reconciliation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveSelectorGateCoverage {
    rows: usize,
    columns: usize,
    selector_columns: Vec<usize>,
    owner_runs: Vec<SelectiveEmittedRowRunAudit>,
    gate_runs: Vec<SelectiveSelectorGateRun>,
    polynomial_arity: usize,
    polynomial_terms: Vec<SelectiveSelectorPolynomialTerm>,
}

impl SelectiveSelectorGateCoverage {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn selector_columns(&self) -> &[usize] {
        &self.selector_columns
    }

    /// Complete compiler ledger, including zero-length organizational runs.
    pub fn owner_runs(&self) -> &[SelectiveEmittedRowRunAudit] {
        &self.owner_runs
    }

    /// Nonempty final-matrix support split at every owner boundary.
    pub fn gate_runs(&self) -> &[SelectiveSelectorGateRun] {
        &self.gate_runs
    }

    /// Exact ordered sparse syntax read from the final relation.
    pub fn polynomial_terms(&self) -> &[SelectiveSelectorPolynomialTerm] {
        &self.polynomial_terms
    }

    pub fn polynomial_arity(&self) -> usize {
        self.polynomial_arity
    }
}

/// A selector-support claim that cannot be reconciled with the final relation.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum SelectiveSelectorGateCoverageError {
    #[error("selective relation has {matrices} matrices and polynomial arity {polynomial_arity}; expected 13")]
    PortCount {
        matrices: usize,
        polynomial_arity: usize,
    },
    #[error("selective relation dimensions must be nonzero, got {rows} rows and {columns} columns")]
    EmptyDimensions { rows: usize, columns: usize },
    #[error("selective polynomial has {terms} terms; expected 27")]
    PolynomialTermCount { terms: usize },
    #[error("selective polynomial term {term} has {exponents} exponents; expected 13")]
    PolynomialExponentCount { term: usize, exponents: usize },
    #[error("selective polynomial term {term} has zero coefficient")]
    PolynomialZeroCoefficient { term: usize },
    #[error("selective polynomial term {term} is not linearly selector-homogeneous: general exponent {general}, evaluation exponent {evaluation}")]
    PolynomialSelectorHomogeneity {
        term: usize,
        general: u32,
        evaluation: u32,
    },
    #[error("selector ledger has {ledger_rows} rows, final structure has {structure_rows}")]
    LedgerRowCount {
        ledger_rows: usize,
        structure_rows: usize,
    },
    #[error(
        "selector ledger has {ledger_arms} arms, snapshot has {snapshot_arms} arms and {selector_count} selectors"
    )]
    ArmCount {
        ledger_arms: usize,
        snapshot_arms: usize,
        selector_count: usize,
    },
    #[error("emitted family {family:?} requires no arm label, got arm {arm}")]
    UnexpectedArm {
        family: SelectiveEmittedRowFamily,
        arm: usize,
    },
    #[error("emitted family {family:?} requires an arm label")]
    MissingArm { family: SelectiveEmittedRowFamily },
    #[error("emitted family {family:?} references arm {arm}, outside 0..{arms}")]
    ArmOutOfRange {
        family: SelectiveEmittedRowFamily,
        arm: usize,
        arms: usize,
    },
    #[error("selector column {column} for arm {arm} is outside final width {columns}")]
    SelectorColumnOutOfRange {
        arm: usize,
        column: usize,
        columns: usize,
    },
    #[error("selector column for arm {arm} is constant-one column zero")]
    ConstantSelectorColumn { arm: usize },
    #[error("selector column {column} is shared by arms {first_arm} and {second_arm}")]
    DuplicateSelectorColumn {
        column: usize,
        first_arm: usize,
        second_arm: usize,
    },
    #[error("selector port {port:?} uses compact matrix tag {tag:?}; exact coverage requires plain CSC")]
    CompactSelectorPort {
        port: SelectiveGatePort,
        tag: SelectiveMatrixTag,
    },
    #[error("selector port {port:?} has shape {rows} x {columns}; expected {expected_rows} x {expected_columns}")]
    MatrixDimensions {
        port: SelectiveGatePort,
        rows: usize,
        columns: usize,
        expected_rows: usize,
        expected_columns: usize,
    },
    #[error("selector port {port:?} has {pointers} column pointers; expected {expected}")]
    ColumnPointerLength {
        port: SelectiveGatePort,
        pointers: usize,
        expected: usize,
    },
    #[error("selector port {port:?} has {rows} row indices and {values} values")]
    ParallelEntryLength {
        port: SelectiveGatePort,
        rows: usize,
        values: usize,
    },
    #[error("selector port {port:?} column-pointer endpoints are {head:?} and {tail:?}; expected 0 and {entries}")]
    ColumnPointerEndpoints {
        port: SelectiveGatePort,
        head: Option<u32>,
        tail: Option<u32>,
        entries: usize,
    },
    #[error(
        "selector port {port:?}, column {column} has noncanonical pointer range {start}..{end} for {entries} entries"
    )]
    ColumnPointerRange {
        port: SelectiveGatePort,
        column: usize,
        start: usize,
        end: usize,
        entries: usize,
    },
    #[error("selector port {port:?} stores {actual} entries; expected {expected}")]
    SelectorEntryCount {
        port: SelectiveGatePort,
        actual: usize,
        expected: usize,
    },
    #[error("selector port {port:?}, column {column} has invalid entry range {start}..{end} at cursor {cursor} of {entries}")]
    SelectorEntryRange {
        port: SelectiveGatePort,
        column: usize,
        start: usize,
        end: usize,
        cursor: usize,
        entries: usize,
    },
    #[error("selector port {port:?}, column {column}, row {row} has coefficient {coefficient:?}, expected one")]
    NonUnitCoefficient {
        port: SelectiveGatePort,
        column: usize,
        row: usize,
        coefficient: F,
    },
    #[error("selector support mismatch at port {port:?}, column {column}: expected row {expected:?}, got {actual:?}")]
    SupportMismatch {
        port: SelectiveGatePort,
        column: usize,
        expected: Option<usize>,
        actual: Option<usize>,
    },
    #[error("nonempty emitted runs do not partition rows at {cursor}: next run is {start}..{end}")]
    OwnerPartition {
        cursor: usize,
        start: usize,
        end: usize,
    },
    #[error("final selector support does not partition rows at {cursor}: next run is {start}..{end}")]
    GatePartition {
        cursor: usize,
        start: usize,
        end: usize,
    },
    #[error("owner run {owner_start}..{owner_end} is not contained in selector support {gate_start}..{gate_end}")]
    OwnerGateBoundary {
        owner_start: usize,
        owner_end: usize,
        gate_start: usize,
        gate_end: usize,
    },
}

type ExpectedSupport = BTreeMap<(SelectiveGatePort, usize), Vec<Range<usize>>>;

fn validate_structure_header(
    structure: &crate::paper::relations::Structure,
) -> Result<(), SelectiveSelectorGateCoverageError> {
    if structure.matrices.len() != SELECTIVE_PORT_COUNT || structure.f.arity() != SELECTIVE_PORT_COUNT {
        return Err(SelectiveSelectorGateCoverageError::PortCount {
            matrices: structure.matrices.len(),
            polynomial_arity: structure.f.arity(),
        });
    }
    if structure.n == 0 || structure.m == 0 {
        return Err(SelectiveSelectorGateCoverageError::EmptyDimensions {
            rows: structure.n,
            columns: structure.m,
        });
    }
    Ok(())
}

fn validate_polynomial(
    structure: &crate::paper::relations::Structure,
) -> Result<Vec<SelectiveSelectorPolynomialTerm>, SelectiveSelectorGateCoverageError> {
    if structure.f.terms().len() != SELECTIVE_POLYNOMIAL_TERM_COUNT {
        return Err(SelectiveSelectorGateCoverageError::PolynomialTermCount {
            terms: structure.f.terms().len(),
        });
    }
    structure
        .f
        .terms()
        .iter()
        .enumerate()
        .map(|(term_index, term)| {
            if term.exps.len() != SELECTIVE_PORT_COUNT {
                return Err(SelectiveSelectorGateCoverageError::PolynomialExponentCount {
                    term: term_index,
                    exponents: term.exps.len(),
                });
            }
            if term.coeff == F::ZERO {
                return Err(SelectiveSelectorGateCoverageError::PolynomialZeroCoefficient { term: term_index });
            }
            let general = term.exps[GENERAL_SELECTOR_PORT];
            let evaluation = term.exps[EVALUATION_SELECTOR_PORT];
            if !((general == 1 && evaluation == 0) || (general == 0 && evaluation == 1)) {
                return Err(SelectiveSelectorGateCoverageError::PolynomialSelectorHomogeneity {
                    term: term_index,
                    general,
                    evaluation,
                });
            }
            Ok(SelectiveSelectorPolynomialTerm {
                coefficient: term.coeff,
                exponents: term.exps.clone(),
            })
        })
        .collect()
}

impl SelectiveLowNormSnapshot<'_> {
    /// Validate the complete general/evaluation selector matrices against the
    /// exclusive emitted-run ledger and return their compact physical support.
    pub fn selector_gate_coverage(&self) -> Result<SelectiveSelectorGateCoverage, SelectiveSelectorGateCoverageError> {
        validate_structure_header(self.structure())?;
        let polynomial_terms = validate_polynomial(self.structure())?;
        let row_audit = self.compiler_audit().rows();
        if row_audit.total_rows() != self.structure().n {
            return Err(SelectiveSelectorGateCoverageError::LedgerRowCount {
                ledger_rows: row_audit.total_rows(),
                structure_rows: self.structure().n,
            });
        }
        if row_audit.arms().len() != self.arm_count() || self.selector_cols().len() != self.arm_count() {
            return Err(SelectiveSelectorGateCoverageError::ArmCount {
                ledger_arms: row_audit.arms().len(),
                snapshot_arms: self.arm_count(),
                selector_count: self.selector_cols().len(),
            });
        }
        for (arm, &column) in self.selector_cols().iter().enumerate() {
            if column >= self.structure().m {
                return Err(SelectiveSelectorGateCoverageError::SelectorColumnOutOfRange {
                    arm,
                    column,
                    columns: self.structure().m,
                });
            }
            if column == 0 {
                return Err(SelectiveSelectorGateCoverageError::ConstantSelectorColumn { arm });
            }
            if let Some(first_arm) = self.selector_cols()[..arm]
                .iter()
                .position(|&previous| previous == column)
            {
                return Err(SelectiveSelectorGateCoverageError::DuplicateSelectorColumn {
                    column,
                    first_arm,
                    second_arm: arm,
                });
            }
        }

        validate_owner_partition(row_audit.emitted_runs(), self.structure().n)?;
        let expected = expected_support(row_audit.emitted_runs(), self.selector_cols())?;
        let mut matrix_gate_runs = Vec::new();
        validate_selector_matrix(
            self.structure(),
            SelectiveGatePort::General,
            &expected,
            &mut matrix_gate_runs,
        )?;
        validate_selector_matrix(
            self.structure(),
            SelectiveGatePort::Evaluation,
            &expected,
            &mut matrix_gate_runs,
        )?;
        matrix_gate_runs.sort_unstable_by_key(|run| run.emitted_rows.start);
        validate_gate_partition(&matrix_gate_runs, self.structure().n)?;
        let gate_runs = split_at_owner_boundaries(row_audit.emitted_runs(), &matrix_gate_runs)?;
        validate_gate_partition(&gate_runs, self.structure().n)?;

        Ok(SelectiveSelectorGateCoverage {
            rows: self.structure().n,
            columns: self.structure().m,
            selector_columns: self.selector_cols().to_vec(),
            owner_runs: row_audit.emitted_runs().to_vec(),
            gate_runs,
            polynomial_arity: self.structure().f.arity(),
            polynomial_terms,
        })
    }
}

fn expected_support(
    runs: &[SelectiveEmittedRowRunAudit],
    selector_columns: &[usize],
) -> Result<ExpectedSupport, SelectiveSelectorGateCoverageError> {
    let mut expected = ExpectedSupport::new();
    for run in runs {
        let (port, column) = expected_gate(run.family(), run.arm(), selector_columns)?;
        if !run.emitted_rows().is_empty() {
            expected
                .entry((port, column))
                .or_default()
                .push(run.emitted_rows());
        }
    }
    Ok(expected)
}

fn expected_gate(
    family: SelectiveEmittedRowFamily,
    arm: Option<usize>,
    selector_columns: &[usize],
) -> Result<(SelectiveGatePort, usize), SelectiveSelectorGateCoverageError> {
    use SelectiveEmittedRowFamily as Family;
    let common = matches!(
        family,
        Family::SelectorDomain
            | Family::SharedDomain
            | Family::OneHot
            | Family::PublicPadding
            | Family::PrivatePadding
            | Family::RingPadding
    );
    if common {
        if let Some(arm) = arm {
            return Err(SelectiveSelectorGateCoverageError::UnexpectedArm { family, arm });
        }
        return Ok((SelectiveGatePort::General, 0));
    }
    let arm = arm.ok_or(SelectiveSelectorGateCoverageError::MissingArm { family })?;
    let column = *selector_columns
        .get(arm)
        .ok_or(SelectiveSelectorGateCoverageError::ArmOutOfRange {
            family,
            arm,
            arms: selector_columns.len(),
        })?;
    let port = match family {
        Family::ArmDomain
        | Family::Retained
        | Family::Poseidon2
        | Family::CenteredUnit
        | Family::ShiftedTernaryCanonical => SelectiveGatePort::General,
        Family::PolynomialEvaluation | Family::ProductSum => SelectiveGatePort::Evaluation,
        Family::SelectorDomain
        | Family::SharedDomain
        | Family::OneHot
        | Family::PublicPadding
        | Family::PrivatePadding
        | Family::RingPadding => unreachable!("common families returned above"),
    };
    Ok((port, column))
}

fn validate_owner_partition(
    runs: &[SelectiveEmittedRowRunAudit],
    rows: usize,
) -> Result<(), SelectiveSelectorGateCoverageError> {
    let mut cursor = 0usize;
    for run in runs {
        let range = run.emitted_rows();
        if range.start != cursor || range.start > range.end || range.end > rows {
            return Err(SelectiveSelectorGateCoverageError::OwnerPartition {
                cursor,
                start: range.start,
                end: range.end,
            });
        }
        cursor = range.end;
    }
    if cursor != rows {
        return Err(SelectiveSelectorGateCoverageError::OwnerPartition {
            cursor,
            start: rows,
            end: rows,
        });
    }
    Ok(())
}

fn plain_csc(matrix: &CcsMatrix<F>, port: SelectiveGatePort) -> Result<&CscMat<F>, SelectiveSelectorGateCoverageError> {
    match matrix {
        CcsMatrix::Csc(csc) => Ok(csc),
        matrix => Err(SelectiveSelectorGateCoverageError::CompactSelectorPort {
            port,
            tag: SelectiveMatrixTag::from_matrix(matrix),
        }),
    }
}

fn validate_selector_matrix(
    structure: &crate::paper::relations::Structure,
    port: SelectiveGatePort,
    expected: &ExpectedSupport,
    gate_runs: &mut Vec<SelectiveSelectorGateRun>,
) -> Result<(), SelectiveSelectorGateCoverageError> {
    let csc = plain_csc(&structure.matrices[port.matrix_index()], port)?;
    if csc.nrows != structure.n || csc.ncols != structure.m {
        return Err(SelectiveSelectorGateCoverageError::MatrixDimensions {
            port,
            rows: csc.nrows,
            columns: csc.ncols,
            expected_rows: structure.n,
            expected_columns: structure.m,
        });
    }
    if csc.col_ptr.len() != structure.m + 1 {
        return Err(SelectiveSelectorGateCoverageError::ColumnPointerLength {
            port,
            pointers: csc.col_ptr.len(),
            expected: structure.m + 1,
        });
    }
    if csc.row_idx.len() != csc.vals.len() {
        return Err(SelectiveSelectorGateCoverageError::ParallelEntryLength {
            port,
            rows: csc.row_idx.len(),
            values: csc.vals.len(),
        });
    }
    let entries = csc.vals.len();
    if csc.col_ptr.first().copied() != Some(0) || csc.col_ptr.last().copied().map(|tail| tail as usize) != Some(entries)
    {
        return Err(SelectiveSelectorGateCoverageError::ColumnPointerEndpoints {
            port,
            head: csc.col_ptr.first().copied(),
            tail: csc.col_ptr.last().copied(),
            entries,
        });
    }
    for (column, pointers) in csc.col_ptr.windows(2).enumerate() {
        let start = pointers[0] as usize;
        let end = pointers[1] as usize;
        if start > end || end > entries {
            return Err(SelectiveSelectorGateCoverageError::ColumnPointerRange {
                port,
                column,
                start,
                end,
                entries,
            });
        }
    }
    let expected_entries = expected
        .iter()
        .filter(|((expected_port, _), _)| *expected_port == port)
        .flat_map(|(_, ranges)| ranges)
        .map(Range::len)
        .sum::<usize>();
    if entries != expected_entries {
        return Err(SelectiveSelectorGateCoverageError::SelectorEntryCount {
            port,
            actual: entries,
            expected: expected_entries,
        });
    }

    let mut entry_cursor = 0usize;
    for ((_, column), expected_ranges) in expected
        .iter()
        .filter(|((expected_port, _), _)| *expected_port == port)
    {
        let entry_range = csc.column_range(*column);
        if entry_range.start != entry_cursor || entry_range.start > entry_range.end || entry_range.end > entries {
            return Err(SelectiveSelectorGateCoverageError::SelectorEntryRange {
                port,
                column: *column,
                start: entry_range.start,
                end: entry_range.end,
                cursor: entry_cursor,
                entries,
            });
        }
        let mut expected_rows = expected_ranges.iter().flat_map(|range| range.clone());
        let mut run_start = None;
        let mut previous_row = None;
        for entry in entry_range.clone() {
            let row = csc.row_index(entry);
            let coefficient = csc.vals[entry];
            if coefficient != F::ONE {
                return Err(SelectiveSelectorGateCoverageError::NonUnitCoefficient {
                    port,
                    column: *column,
                    row,
                    coefficient,
                });
            }
            let expected_row = expected_rows.next();
            if expected_row != Some(row) {
                return Err(SelectiveSelectorGateCoverageError::SupportMismatch {
                    port,
                    column: *column,
                    expected: expected_row,
                    actual: Some(row),
                });
            }
            match previous_row {
                Some(previous) if previous + 1 == row => {}
                Some(previous) => {
                    gate_runs.push(SelectiveSelectorGateRun {
                        emitted_rows: run_start.expect("previous row establishes run start")..previous + 1,
                        port,
                        column: *column,
                        coefficient,
                    });
                    run_start = Some(row);
                }
                None => run_start = Some(row),
            }
            previous_row = Some(row);
        }
        if let Some(expected_row) = expected_rows.next() {
            return Err(SelectiveSelectorGateCoverageError::SupportMismatch {
                port,
                column: *column,
                expected: Some(expected_row),
                actual: None,
            });
        }
        if let Some(previous) = previous_row {
            gate_runs.push(SelectiveSelectorGateRun {
                emitted_rows: run_start.expect("previous row establishes run start")..previous + 1,
                port,
                column: *column,
                coefficient: F::ONE,
            });
        }
        entry_cursor = entry_range.end;
    }
    if entry_cursor != entries {
        return Err(SelectiveSelectorGateCoverageError::SelectorEntryRange {
            port,
            column: structure.m,
            start: entry_cursor,
            end: entries,
            cursor: entry_cursor,
            entries,
        });
    }
    Ok(())
}

fn validate_gate_partition(
    runs: &[SelectiveSelectorGateRun],
    rows: usize,
) -> Result<(), SelectiveSelectorGateCoverageError> {
    let mut cursor = 0usize;
    for run in runs {
        if run.emitted_rows.start != cursor || run.emitted_rows.end <= cursor || run.emitted_rows.end > rows {
            return Err(SelectiveSelectorGateCoverageError::GatePartition {
                cursor,
                start: run.emitted_rows.start,
                end: run.emitted_rows.end,
            });
        }
        cursor = run.emitted_rows.end;
    }
    if cursor != rows {
        return Err(SelectiveSelectorGateCoverageError::GatePartition {
            cursor,
            start: rows,
            end: rows,
        });
    }
    Ok(())
}

fn split_at_owner_boundaries(
    owners: &[SelectiveEmittedRowRunAudit],
    matrix_runs: &[SelectiveSelectorGateRun],
) -> Result<Vec<SelectiveSelectorGateRun>, SelectiveSelectorGateCoverageError> {
    let mut gate_index = 0usize;
    let mut aligned = Vec::with_capacity(owners.len());
    for owner in owners {
        let owner_rows = owner.emitted_rows();
        if owner_rows.is_empty() {
            continue;
        }
        while gate_index < matrix_runs.len() && matrix_runs[gate_index].emitted_rows.end <= owner_rows.start {
            gate_index += 1;
        }
        let gate = matrix_runs
            .get(gate_index)
            .ok_or(SelectiveSelectorGateCoverageError::OwnerGateBoundary {
                owner_start: owner_rows.start,
                owner_end: owner_rows.end,
                gate_start: owner_rows.end,
                gate_end: owner_rows.end,
            })?;
        if gate.emitted_rows.start > owner_rows.start || gate.emitted_rows.end < owner_rows.end {
            return Err(SelectiveSelectorGateCoverageError::OwnerGateBoundary {
                owner_start: owner_rows.start,
                owner_end: owner_rows.end,
                gate_start: gate.emitted_rows.start,
                gate_end: gate.emitted_rows.end,
            });
        }
        aligned.push(SelectiveSelectorGateRun {
            emitted_rows: owner_rows,
            port: gate.port,
            column: gate.column,
            coefficient: gate.coefficient,
        });
    }
    Ok(aligned)
}
