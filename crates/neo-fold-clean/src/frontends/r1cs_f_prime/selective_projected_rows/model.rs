//! Data returned by bounded selective-row projection.

use neo_ccs::SeededPhi81LinearBlock;
use neo_math::F;

use crate::frontends::r1cs_f_prime::selective_audit::{SelectiveEmittedRowFamily, SelectiveRewriteKind};
use crate::frontends::r1cs_f_prime::selective_row_artifact::SELECTIVE_ROW_ARTIFACT_SCHEMA_VERSION;

use super::super::SELECTIVE_ARITY;

/// Exact storage census for one port's explicit emitter-order term stream
/// under constant-coefficient affine run encoding.
///
/// This is a format-design measurement. It does not replace the emitted
/// matrix or authorize a compact artifact claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectiveProjectedExplicitRunCensus {
    pub(super) term_count: usize,
    pub(super) affine_run_count: usize,
    pub(super) affine_run_terms: usize,
    pub(super) literal_count: usize,
}

impl SelectiveProjectedExplicitRunCensus {
    pub fn term_count(self) -> usize {
        self.term_count
    }

    pub fn affine_run_count(self) -> usize {
        self.affine_run_count
    }

    pub fn affine_run_terms(self) -> usize {
        self.affine_run_terms
    }

    pub fn literal_count(self) -> usize {
        self.literal_count
    }

    pub fn record_count(self) -> usize {
        self.affine_run_count + self.literal_count
    }
}

/// One exact source-field term retained in compiler provenance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectiveProjectedSourceTerm {
    pub(super) column: usize,
    pub(super) coefficient: F,
}

impl SelectiveProjectedSourceTerm {
    pub fn column(self) -> usize {
        self.column
    }

    pub fn coefficient(self) -> F {
        self.coefficient
    }
}

/// Final low-norm slot used to decode one retained source field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectiveProjectedSourceSlot {
    pub(super) column: usize,
    pub(super) start: usize,
    pub(super) width: usize,
}

impl SelectiveProjectedSourceSlot {
    pub fn column(self) -> usize {
        self.column
    }

    pub fn start(self) -> usize {
        self.start
    }

    pub fn width(self) -> usize {
        self.width
    }
}

/// One compiler-validated linear substitution reachable from the focused
/// source rows. Definitions remain in compiler order so recursive replay is
/// deterministic and fail-closed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveProjectedSourceDefinition {
    pub(super) target: usize,
    pub(super) constant: F,
    pub(super) terms: Vec<SelectiveProjectedSourceTerm>,
}

impl SelectiveProjectedSourceDefinition {
    pub fn target(&self) -> usize {
        self.target
    }

    pub fn constant(&self) -> F {
        self.constant
    }

    pub fn terms(&self) -> &[SelectiveProjectedSourceTerm] {
        &self.terms
    }
}

/// One scaled product in a compiler-introduced grouped product sum.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveProjectedProductFactor {
    pub(super) left_constant: F,
    pub(super) left_terms: Vec<SelectiveProjectedSourceTerm>,
    pub(super) right_constant: F,
    pub(super) right_terms: Vec<SelectiveProjectedSourceTerm>,
    pub(super) coefficient: F,
}

/// Source linear combination or compiler-derived slot finalized by one
/// grouped rewrite row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SelectiveProjectedRewriteOutput {
    Source {
        constant: F,
        terms: Vec<SelectiveProjectedSourceTerm>,
    },
    DerivedProductSum {
        compiler_index: usize,
    },
}

/// Executable grouped-product recurrence for one emitted evaluation row:
/// `output = base + previous + sum(coefficient * left * right)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveProjectedRewriteStep {
    pub(super) emitted_row: usize,
    pub(super) rewrite_id: usize,
    pub(super) kind: SelectiveRewriteKind,
    pub(super) source_rows: Vec<(usize, usize)>,
    pub(super) output: SelectiveProjectedRewriteOutput,
    pub(super) base_constant: F,
    pub(super) base_terms: Vec<SelectiveProjectedSourceTerm>,
    pub(super) previous: Option<usize>,
    pub(super) factors: Vec<SelectiveProjectedProductFactor>,
}

/// One exact source linear combination with its constant wire separated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveProjectedSourceLinearCombination {
    pub(super) constant: F,
    pub(super) terms: Vec<SelectiveProjectedSourceTerm>,
}

impl SelectiveProjectedSourceLinearCombination {
    pub fn constant(&self) -> F {
        self.constant
    }

    pub fn terms(&self) -> &[SelectiveProjectedSourceTerm] {
        &self.terms
    }
}

/// One compact Poseidon2 S-box row joined to its exact source trace.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveProjectedPoseidon2SboxStep {
    pub(super) emitted_row: usize,
    pub(super) rewrite_id: usize,
    pub(super) source_rows: Vec<(usize, usize)>,
    pub(super) input: SelectiveProjectedSourceLinearCombination,
    pub(super) output: SelectiveProjectedSourceLinearCombination,
}

impl SelectiveProjectedPoseidon2SboxStep {
    pub fn emitted_row(&self) -> usize {
        self.emitted_row
    }

    pub fn rewrite_id(&self) -> usize {
        self.rewrite_id
    }

    pub fn source_rows(&self) -> &[(usize, usize)] {
        &self.source_rows
    }

    pub fn input(&self) -> &SelectiveProjectedSourceLinearCombination {
        &self.input
    }

    pub fn output(&self) -> &SelectiveProjectedSourceLinearCombination {
        &self.output
    }
}

/// Exact source-row owner for one retained emitted check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveProjectedRetainedStep {
    pub(super) emitted_row: usize,
    pub(super) source_row: usize,
    pub(super) ports: [SelectiveProjectedSourceLinearCombination; 3],
}

impl SelectiveProjectedRetainedStep {
    pub fn emitted_row(&self) -> usize {
        self.emitted_row
    }

    pub fn source_row(&self) -> usize {
        self.source_row
    }

    pub fn a(&self) -> &SelectiveProjectedSourceLinearCombination {
        &self.ports[0]
    }

    pub fn b(&self) -> &SelectiveProjectedSourceLinearCombination {
        &self.ports[1]
    }

    pub fn c(&self) -> &SelectiveProjectedSourceLinearCombination {
        &self.ports[2]
    }
}

impl SelectiveProjectedRewriteStep {
    pub fn emitted_row(&self) -> usize {
        self.emitted_row
    }

    pub fn rewrite_id(&self) -> usize {
        self.rewrite_id
    }

    pub fn kind(&self) -> SelectiveRewriteKind {
        self.kind
    }

    pub fn source_rows(&self) -> &[(usize, usize)] {
        &self.source_rows
    }

    pub fn output(&self) -> &SelectiveProjectedRewriteOutput {
        &self.output
    }

    pub fn base_constant(&self) -> F {
        self.base_constant
    }

    pub fn base_terms(&self) -> &[SelectiveProjectedSourceTerm] {
        &self.base_terms
    }

    pub fn previous(&self) -> Option<usize> {
        self.previous
    }

    pub fn factors(&self) -> &[SelectiveProjectedProductFactor] {
        &self.factors
    }
}

impl SelectiveProjectedProductFactor {
    pub fn left_constant(&self) -> F {
        self.left_constant
    }

    pub fn left_terms(&self) -> &[SelectiveProjectedSourceTerm] {
        &self.left_terms
    }

    pub fn right_constant(&self) -> F {
        self.right_constant
    }

    pub fn right_terms(&self) -> &[SelectiveProjectedSourceTerm] {
        &self.right_terms
    }

    pub fn coefficient(&self) -> F {
        self.coefficient
    }
}

/// One derived grouped-product accumulator referenced by a selected row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveProjectedDerivedProductSum {
    pub(super) compiler_index: usize,
    pub(super) start: usize,
    pub(super) width: usize,
    pub(super) factors: Vec<SelectiveProjectedProductFactor>,
    pub(super) previous: Option<usize>,
}

impl SelectiveProjectedDerivedProductSum {
    pub fn compiler_index(&self) -> usize {
        self.compiler_index
    }

    pub fn start(&self) -> usize {
        self.start
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn factors(&self) -> &[SelectiveProjectedProductFactor] {
        &self.factors
    }

    pub fn previous(&self) -> Option<usize> {
        self.previous
    }
}

/// Exact source-column partition and compiler substitutions needed by one
/// bounded row projection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveProjectedSourceProvenance {
    pub(super) arm: usize,
    pub(super) source_columns: Vec<usize>,
    pub(super) retained_slots: Vec<SelectiveProjectedSourceSlot>,
    pub(super) linear_definitions: Vec<SelectiveProjectedSourceDefinition>,
    pub(super) trace_eliminated_columns: Vec<usize>,
    pub(super) poseidon2_sbox_steps: Vec<SelectiveProjectedPoseidon2SboxStep>,
    pub(super) derived_product_sums: Vec<SelectiveProjectedDerivedProductSum>,
    pub(super) rewrite_steps: Vec<SelectiveProjectedRewriteStep>,
    pub(super) retained_steps: Vec<SelectiveProjectedRetainedStep>,
}

impl SelectiveProjectedSourceProvenance {
    pub fn arm(&self) -> usize {
        self.arm
    }

    /// Ordered transitive closure of source columns referenced by the focused
    /// rows and any reachable compiler linear definitions.
    pub fn source_columns(&self) -> &[usize] {
        &self.source_columns
    }

    pub fn retained_slots(&self) -> &[SelectiveProjectedSourceSlot] {
        &self.retained_slots
    }

    pub fn linear_definitions(&self) -> &[SelectiveProjectedSourceDefinition] {
        &self.linear_definitions
    }

    /// Trace-local temporaries intentionally absent from the final assignment.
    /// Their values must be reconstructed from the corresponding trace
    /// semantics rather than treated as free decoder inputs.
    pub fn trace_eliminated_columns(&self) -> &[usize] {
        &self.trace_eliminated_columns
    }

    pub fn poseidon2_sbox_steps(&self) -> &[SelectiveProjectedPoseidon2SboxStep] {
        &self.poseidon2_sbox_steps
    }

    pub fn derived_product_sums(&self) -> &[SelectiveProjectedDerivedProductSum] {
        &self.derived_product_sums
    }

    pub fn rewrite_steps(&self) -> &[SelectiveProjectedRewriteStep] {
        &self.rewrite_steps
    }

    pub fn retained_steps(&self) -> &[SelectiveProjectedRetainedStep] {
        &self.retained_steps
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectiveProjectedTerm {
    pub(super) column: usize,
    pub(super) coefficient: F,
}

impl SelectiveProjectedTerm {
    pub fn column(self) -> usize {
        self.column
    }

    pub fn coefficient(self) -> F {
        self.coefficient
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectiveProjectedGeometricRun {
    pub(super) column_start: usize,
    pub(super) length: usize,
    pub(super) initial: F,
    pub(super) ratio: F,
}

impl SelectiveProjectedGeometricRun {
    pub fn column_start(self) -> usize {
        self.column_start
    }

    pub fn length(self) -> usize {
        self.length
    }

    pub fn initial(self) -> F {
        self.initial
    }

    pub fn ratio(self) -> F {
        self.ratio
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveProjectedPort {
    pub(super) explicit: Vec<SelectiveProjectedTerm>,
    pub(super) geometric_runs: Vec<SelectiveProjectedGeometricRun>,
    pub(super) seeded_blocks: Vec<SeededPhi81LinearBlock>,
}

impl SelectiveProjectedPort {
    pub fn explicit(&self) -> &[SelectiveProjectedTerm] {
        &self.explicit
    }

    pub fn geometric_runs(&self) -> &[SelectiveProjectedGeometricRun] {
        &self.geometric_runs
    }

    pub fn seeded_blocks(&self) -> &[SeededPhi81LinearBlock] {
        &self.seeded_blocks
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveProjectedRowArtifact {
    pub(super) rows: usize,
    pub(super) columns: usize,
    pub(super) emitted_row: usize,
    pub(super) run_index: usize,
    pub(super) family: SelectiveEmittedRowFamily,
    pub(super) arm: Option<usize>,
    pub(super) ports: [SelectiveProjectedPort; SELECTIVE_ARITY],
}

/// Verifier-owned source of one coordinate in the selectively compiled
/// public carrier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectiveProjectedPublicCoordinateSource {
    /// The encoder writes the conventional constant-one coordinate directly.
    ConstantOne,
    /// One source-R1CS public field is copied into this coordinate.
    SourceField(usize),
    /// The selective compiler inserts this public ring-completion zero.
    FixedZero,
}

/// Exact source owner for one coordinate in the final public assignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectiveProjectedPublicCoordinate {
    pub(super) column: usize,
    pub(super) source: SelectiveProjectedPublicCoordinateSource,
}

impl SelectiveProjectedPublicCoordinate {
    pub fn column(self) -> usize {
        self.column
    }

    pub fn source(self) -> SelectiveProjectedPublicCoordinateSource {
        self.source
    }
}

impl SelectiveProjectedRowArtifact {
    pub fn schema_version(&self) -> u64 {
        SELECTIVE_ROW_ARTIFACT_SCHEMA_VERSION
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn emitted_row(&self) -> usize {
        self.emitted_row
    }

    pub fn run_index(&self) -> usize {
        self.run_index
    }

    pub fn family(&self) -> SelectiveEmittedRowFamily {
        self.family
    }

    pub fn arm(&self) -> Option<usize> {
        self.arm
    }

    pub fn ports(&self) -> &[SelectiveProjectedPort; SELECTIVE_ARITY] {
        &self.ports
    }
}
