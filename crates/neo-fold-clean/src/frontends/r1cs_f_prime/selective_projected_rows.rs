//! Bounded row projection from the exact selective emitter term stream.
//!
//! This path serves assurance fixtures whose complete fixed-point column
//! domain is intentionally too large to materialize as thirteen full CSC
//! matrices. It invokes the same emitter as `build_structure`, then
//! canonicalizes only caller-selected rows before allocating column-sized
//! arrays.
//!
//! Owns: bounded projection of the shared emitter's exact thirteen-port term
//! stream and source/compiler provenance for caller-selected rows.
//!
//! Does not own: source semantics, selector truth, protocol authority,
//! security reductions, or permission to remove constraints.
//!
//! Emits constraints: no new rows; it observes rows emitted by the shared
//! selective structure path.
//!
//! | Child path | Mathematical obligation | Authority class |
//! |---|---|---|
//! | `selective.projected_rows.final` | selected A/B/C terms equal the shared emitter stream | direct dataflow |
//! | `selective.projected_rows.source` | retained slots and substitutions cover every referenced source column | computed |
//! | `selective.projected_rows.rewrite` | each compact rewrite records its exact factors and output | computed |

use std::collections::{BTreeMap, BTreeSet};

use neo_ccs::CcsMatrix;
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::builder::ProductFactorTrace;
use crate::engine::r1cs_circuit::Lc;

use super::super::selective_audit::{SelectiveCompilerAudit, SelectiveEmittedRowRunAudit};
use super::super::SparseR1cs;
use super::emit::{append_field, append_lc, append_lc_scaled, append_slot};
use super::projected_decoder::{
    decoder_provenance, decoder_run_provenance, SelectiveProjectedDecoderProvenance,
    SelectiveProjectedDecoderRunProvenance,
};
use super::terms::MatrixTerms;
use super::{
    prepare_selective_layout, structure, trace_error, LowNormR1csError, A, B, C, EVAL_GROUP_SIZE, EVAL_PAIRS,
    EVAL_SELECTOR, GENERAL_SELECTOR, SELECTIVE_ARITY,
};

/// One exact source-field term retained in compiler provenance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectiveProjectedSourceTerm {
    column: usize,
    coefficient: F,
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
    column: usize,
    start: usize,
    width: usize,
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
    target: usize,
    constant: F,
    terms: Vec<SelectiveProjectedSourceTerm>,
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
    left_constant: F,
    left_terms: Vec<SelectiveProjectedSourceTerm>,
    right_constant: F,
    right_terms: Vec<SelectiveProjectedSourceTerm>,
    coefficient: F,
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
    emitted_row: usize,
    rewrite_id: usize,
    kind: super::super::selective_audit::SelectiveRewriteKind,
    source_rows: Vec<(usize, usize)>,
    output: SelectiveProjectedRewriteOutput,
    base_constant: F,
    base_terms: Vec<SelectiveProjectedSourceTerm>,
    previous: Option<usize>,
    factors: Vec<SelectiveProjectedProductFactor>,
}

/// One exact source linear combination with its constant wire separated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveProjectedSourceLinearCombination {
    constant: F,
    terms: Vec<SelectiveProjectedSourceTerm>,
}

impl SelectiveProjectedSourceLinearCombination {
    pub fn constant(&self) -> F {
        self.constant
    }

    pub fn terms(&self) -> &[SelectiveProjectedSourceTerm] {
        &self.terms
    }
}

/// Exact source-row owner for one retained emitted check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveProjectedRetainedStep {
    emitted_row: usize,
    source_row: usize,
    ports: [SelectiveProjectedSourceLinearCombination; 3],
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

    pub fn kind(&self) -> super::super::selective_audit::SelectiveRewriteKind {
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
    compiler_index: usize,
    start: usize,
    width: usize,
    factors: Vec<SelectiveProjectedProductFactor>,
    previous: Option<usize>,
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
    arm: usize,
    source_columns: Vec<usize>,
    retained_slots: Vec<SelectiveProjectedSourceSlot>,
    linear_definitions: Vec<SelectiveProjectedSourceDefinition>,
    trace_eliminated_columns: Vec<usize>,
    derived_product_sums: Vec<SelectiveProjectedDerivedProductSum>,
    rewrite_steps: Vec<SelectiveProjectedRewriteStep>,
    retained_steps: Vec<SelectiveProjectedRetainedStep>,
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
    column: usize,
    coefficient: F,
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
    column_start: usize,
    length: usize,
    initial: F,
    ratio: F,
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
    explicit: Vec<SelectiveProjectedTerm>,
    geometric_runs: Vec<SelectiveProjectedGeometricRun>,
}

impl SelectiveProjectedPort {
    pub fn explicit(&self) -> &[SelectiveProjectedTerm] {
        &self.explicit
    }

    pub fn geometric_runs(&self) -> &[SelectiveProjectedGeometricRun] {
        &self.geometric_runs
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveProjectedRowArtifact {
    rows: usize,
    columns: usize,
    emitted_row: usize,
    run_index: usize,
    family: super::super::selective_audit::SelectiveEmittedRowFamily,
    arm: Option<usize>,
    ports: [SelectiveProjectedPort; SELECTIVE_ARITY],
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
    column: usize,
    source: SelectiveProjectedPublicCoordinateSource,
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
        super::super::selective_row_artifact::SELECTIVE_ROW_ARTIFACT_SCHEMA_VERSION
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

    pub fn family(&self) -> super::super::selective_audit::SelectiveEmittedRowFamily {
        self.family
    }

    pub fn arm(&self) -> Option<usize> {
        self.arm
    }

    pub fn ports(&self) -> &[SelectiveProjectedPort; SELECTIVE_ARITY] {
        &self.ports
    }
}

/// Exact selected rows emitted from one prepared selective compiler plan.
#[derive(Debug)]
pub struct SelectiveProjectedRowsAudit {
    rows: usize,
    columns: usize,
    selector_columns: Vec<usize>,
    compiler_audit: SelectiveCompilerAudit,
    public_coordinates: Vec<SelectiveProjectedPublicCoordinate>,
    public_padding_row_artifacts: Vec<SelectiveProjectedRowArtifact>,
    selector_domain_row_artifacts: Vec<SelectiveProjectedRowArtifact>,
    one_hot_row_artifact: SelectiveProjectedRowArtifact,
    private_padding_row_artifacts: Vec<SelectiveProjectedRowArtifact>,
    ring_padding_row_artifacts: Vec<SelectiveProjectedRowArtifact>,
    row_artifacts: Vec<SelectiveProjectedRowArtifact>,
    source_provenance: Option<SelectiveProjectedSourceProvenance>,
    decoder_provenance: Option<SelectiveProjectedDecoderProvenance>,
    decoder_run_provenance: Option<SelectiveProjectedDecoderRunProvenance>,
}

impl SelectiveProjectedRowsAudit {
    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn selector_columns(&self) -> &[usize] {
        &self.selector_columns
    }

    pub fn compiler_audit(&self) -> &SelectiveCompilerAudit {
        &self.compiler_audit
    }

    /// Complete public-coordinate decoder validated against every arm of the
    /// same prepared layout used by the projected emitter.
    pub fn public_coordinates(&self) -> &[SelectiveProjectedPublicCoordinate] {
        &self.public_coordinates
    }

    /// Exact public-padding zero rows projected independently of the
    /// caller-selected semantic slice.
    pub fn public_padding_row_artifacts(&self) -> &[SelectiveProjectedRowArtifact] {
        &self.public_padding_row_artifacts
    }

    /// Exact selector-domain rows projected independently of the
    /// caller-selected semantic slice.
    pub fn selector_domain_row_artifacts(&self) -> &[SelectiveProjectedRowArtifact] {
        &self.selector_domain_row_artifacts
    }

    /// Exact selector-total row projected independently of the
    /// caller-selected semantic slice.
    pub fn one_hot_row_artifact(&self) -> &SelectiveProjectedRowArtifact {
        &self.one_hot_row_artifact
    }

    /// Exact private-alignment zero rows projected independently of the
    /// caller-selected semantic slice.
    pub fn private_padding_row_artifacts(&self) -> &[SelectiveProjectedRowArtifact] {
        &self.private_padding_row_artifacts
    }

    /// Exact final ring-alignment zero rows projected independently of the
    /// caller-selected semantic slice.
    pub fn ring_padding_row_artifacts(&self) -> &[SelectiveProjectedRowArtifact] {
        &self.ring_padding_row_artifacts
    }

    pub fn row_artifacts(&self) -> &[SelectiveProjectedRowArtifact] {
        &self.row_artifacts
    }

    pub fn source_provenance(&self) -> Option<&SelectiveProjectedSourceProvenance> {
        self.source_provenance.as_ref()
    }

    /// Exact source-to-final-assignment decoder requested independently of
    /// the selected row certificate.
    pub fn decoder_provenance(&self) -> Option<&SelectiveProjectedDecoderProvenance> {
        self.decoder_provenance.as_ref()
    }

    /// Complete run-compressed decoder requested independently of selected
    /// row/source closure.  This remains layout data, not a value theorem.
    pub fn decoder_run_provenance(&self) -> Option<&SelectiveProjectedDecoderRunProvenance> {
        self.decoder_run_provenance.as_ref()
    }
}

fn public_coordinate_decoder(
    arms: &[SparseR1cs],
    layout: &super::SelectiveLayout,
) -> Result<Vec<SelectiveProjectedPublicCoordinate>, LowNormR1csError> {
    let audit = layout.compiler_audit.layout();
    let logical = audit.logical_public_input_len();
    let public = audit.public_input_len();
    if logical == 0 || public < logical {
        return Err(trace_error("selective public decoder has an invalid public range"));
    }
    if audit
        .public_padding_columns()
        .iter()
        .copied()
        .ne(logical..public)
    {
        return Err(trace_error(
            "selective public decoder padding differs from the emitted public range",
        ));
    }
    for (arm, source) in arms.iter().enumerate() {
        if source.m_in != logical {
            return Err(trace_error(
                "selective public decoder source width differs from the encoded logical prefix",
            ));
        }
        if layout.slots[arm][0].is_some()
            || layout.aliases[arm][0].is_some()
            || layout.equal_aliases[arm][0].is_some()
            || layout.plans[arm].centered[0]
        {
            return Err(trace_error(
                "selective public decoder constant coordinate has source-owned encoding metadata",
            ));
        }
        for field in 1..logical {
            if layout.slots[arm][field] != Some((field, 1))
                || layout.aliases[arm][field].is_some()
                || layout.equal_aliases[arm][field].is_some()
                || layout.plans[arm].centered[field]
            {
                return Err(trace_error(
                    "selective public decoder field is not the canonical direct coordinate",
                ));
            }
        }
    }

    let mut decoded = Vec::with_capacity(public);
    decoded.push(SelectiveProjectedPublicCoordinate {
        column: 0,
        source: SelectiveProjectedPublicCoordinateSource::ConstantOne,
    });
    decoded.extend((1..logical).map(|field| SelectiveProjectedPublicCoordinate {
        column: field,
        source: SelectiveProjectedPublicCoordinateSource::SourceField(field),
    }));
    decoded.extend((logical..public).map(|column| SelectiveProjectedPublicCoordinate {
        column,
        source: SelectiveProjectedPublicCoordinateSource::FixedZero,
    }));
    Ok(decoded)
}

fn project_port(terms: &MatrixTerms, row: usize, columns: usize) -> Result<SelectiveProjectedPort, LowNormR1csError> {
    let mut canonical = BTreeMap::<usize, F>::new();
    let mut add = |column: usize, coefficient: F| -> Result<(), LowNormR1csError> {
        if column >= columns {
            return Err(trace_error("projected selective term exceeds the final column domain"));
        }
        *canonical.entry(column).or_insert(F::ZERO) += coefficient;
        Ok(())
    };

    for &(term_row, column, coefficient) in &terms.explicit {
        if term_row == row {
            add(column, coefficient)?;
        }
    }
    let geometric_runs = terms
        .geometric_runs
        .iter()
        .filter(|run| run.row() == row)
        .map(|run| {
            if run.column_start() + run.len() > columns {
                return Err(trace_error("projected geometric run exceeds the final column domain"));
            }
            Ok(SelectiveProjectedGeometricRun {
                column_start: run.column_start(),
                length: run.len(),
                initial: *run.initial(),
                ratio: *run.ratio(),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    for block in &terms.seeded {
        if block.row_start() <= row && row < block.row_start() + D * block.kappa() {
            return Err(trace_error(
                "bounded selective projection intersects a compact seeded row",
            ));
        }
    }
    canonical.retain(|_, coefficient| *coefficient != F::ZERO);
    Ok(SelectiveProjectedPort {
        explicit: canonical
            .into_iter()
            .map(|(column, coefficient)| SelectiveProjectedTerm { column, coefficient })
            .collect(),
        geometric_runs,
    })
}

fn unique_owner(
    audit: &SelectiveCompilerAudit,
    row: usize,
) -> Result<(usize, &SelectiveEmittedRowRunAudit), LowNormR1csError> {
    let mut owners = audit
        .rows()
        .emitted_runs()
        .iter()
        .enumerate()
        .filter(|(_, run)| !run.emitted_rows().is_empty() && run.emitted_rows().contains(&row));
    let owner = owners
        .next()
        .ok_or_else(|| trace_error("projected selective row has no emitted-run owner"))?;
    if owners.next().is_some() {
        return Err(trace_error("projected selective row has multiple emitted-run owners"));
    }
    Ok(owner)
}

fn project_row_artifact(
    emitted: &structure::EmittedStructureTerms,
    audit: &SelectiveCompilerAudit,
    row: usize,
) -> Result<SelectiveProjectedRowArtifact, LowNormR1csError> {
    let ports = (0..SELECTIVE_ARITY)
        .map(|port| project_port(&emitted.matrix_terms[port], row, emitted.columns))
        .collect::<Result<Vec<_>, _>>()?;
    let ports: [SelectiveProjectedPort; SELECTIVE_ARITY] = ports
        .try_into()
        .expect("the selective port range has compile-time arity");
    let (run_index, owner) = unique_owner(audit, row)?;
    Ok(SelectiveProjectedRowArtifact {
        rows: emitted.rows,
        columns: emitted.columns,
        emitted_row: row,
        run_index,
        family: owner.family(),
        arm: owner.arm(),
        ports,
    })
}

fn source_terms(terms: &[(usize, F)]) -> Vec<SelectiveProjectedSourceTerm> {
    terms
        .iter()
        .map(|&(column, coefficient)| SelectiveProjectedSourceTerm { column, coefficient })
        .collect()
}

fn port_intersects_slot(port: &SelectiveProjectedPort, (start, width): (usize, usize)) -> bool {
    let end = start + width;
    port.explicit
        .iter()
        .any(|term| (start..end).contains(&term.column))
        || port.geometric_runs.iter().any(|run| {
            let run_end = run.column_start + run.length;
            start < run_end && run.column_start < end
        })
}

#[derive(Clone)]
enum PlannedRewriteOutput {
    Source(Lc),
    DerivedProductSum(usize),
}

#[derive(Clone)]
struct PlannedRewriteStep {
    emitted_row: usize,
    rewrite_id: usize,
    kind: super::super::selective_audit::SelectiveRewriteKind,
    source_rows: Vec<(usize, usize)>,
    output: PlannedRewriteOutput,
    base: Lc,
    previous: Option<usize>,
    factors: Vec<ProductFactorTrace>,
}

fn source_column_lc(column: usize) -> Lc {
    Lc {
        terms: vec![(column, F::ONE)],
        constant: F::ZERO,
    }
}

fn product_factor_traces_exact(left: &[ProductFactorTrace], right: &[ProductFactorTrace]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.coefficient == right.coefficient
                && left.left.constant == right.left.constant
                && left.left.terms == right.left.terms
                && left.right.constant == right.right.constant
                && left.right.terms == right.right.terms
        })
}

fn rewrite_geometry(
    layout: &super::SelectiveLayout,
    rewrite_id: super::super::selective_audit::SelectiveRewriteId,
    arm: usize,
    kind: super::super::selective_audit::SelectiveRewriteKind,
) -> Result<(Vec<(usize, usize)>, std::ops::Range<usize>), LowNormR1csError> {
    let rewrite = layout
        .compiler_audit
        .rows()
        .rewrites()
        .get(rewrite_id.index())
        .filter(|rewrite| rewrite.id() == rewrite_id)
        .ok_or_else(|| trace_error("projected rewrite identifier is absent from the compiler ledger"))?;
    if rewrite.arm() != arm || rewrite.kind() != kind {
        return Err(trace_error(
            "projected rewrite metadata differs from its compiler ledger owner",
        ));
    }
    Ok((
        rewrite
            .source_rows()
            .iter()
            .map(|rows| (rows.start, rows.end))
            .collect(),
        rewrite.emitted_rows(),
    ))
}

fn planned_rewrite_steps(
    source_arm: &SparseR1cs,
    layout: &super::SelectiveLayout,
    arm: usize,
) -> Result<Vec<PlannedRewriteStep>, LowNormR1csError> {
    use super::super::selective_audit::SelectiveRewriteKind;

    let prepared = layout.prepared_rows.arm(arm);
    let derived = &layout.derived_product_sums[arm];
    let mut derived_cursor = 0usize;
    let mut steps = Vec::new();

    for (trace_index, trace) in source_arm.polynomial_evaluation_traces().iter().enumerate() {
        let rewrite_id = prepared.polynomial_evaluation_rewrite(trace_index);
        let (source_rows, emitted_rows) =
            rewrite_geometry(layout, rewrite_id, arm, SelectiveRewriteKind::PolynomialEvaluation)?;
        let mut emitted_row = emitted_rows.start;
        for limb in 0..2 {
            let product_indices = (1..trace.coefficient_cols.len()).collect::<Vec<_>>();
            let groups = product_indices.chunks(EVAL_GROUP_SIZE).collect::<Vec<_>>();
            if groups.is_empty() {
                let base = if limb == 0 {
                    source_column_lc(trace.coefficient_cols[0])
                } else {
                    Lc::zero()
                };
                steps.push(PlannedRewriteStep {
                    emitted_row,
                    rewrite_id: rewrite_id.index(),
                    kind: SelectiveRewriteKind::PolynomialEvaluation,
                    source_rows: source_rows.clone(),
                    output: PlannedRewriteOutput::Source(source_column_lc(trace.output_cols[limb])),
                    base,
                    previous: None,
                    factors: Vec::new(),
                });
                emitted_row += 1;
                continue;
            }
            let mut previous = None;
            for (group_index, group) in groups.iter().enumerate() {
                let final_group = group_index + 1 == groups.len();
                let factors = group
                    .iter()
                    .map(|&term_index| ProductFactorTrace {
                        left: source_column_lc(trace.coefficient_cols[term_index]),
                        right: source_column_lc(trace.power_cols[term_index][limb]),
                        coefficient: F::ONE,
                    })
                    .collect::<Vec<_>>();
                let output = if final_group {
                    PlannedRewriteOutput::Source(source_column_lc(trace.output_cols[limb]))
                } else {
                    let Some(encoding) = derived.get(derived_cursor) else {
                        return Err(trace_error("projected polynomial rewrite exceeds derived-product plan"));
                    };
                    if encoding.previous != previous {
                        return Err(trace_error(
                            "projected polynomial predecessor differs from derived-product plan",
                        ));
                    }
                    if !product_factor_traces_exact(&factors, &encoding.factors) {
                        return Err(trace_error(
                            "projected polynomial factors differ from derived-product witness plan",
                        ));
                    }
                    let output = PlannedRewriteOutput::DerivedProductSum(derived_cursor);
                    derived_cursor += 1;
                    output
                };
                let base = if final_group && limb == 0 {
                    source_column_lc(trace.coefficient_cols[0])
                } else {
                    Lc::zero()
                };
                let next_previous = match &output {
                    PlannedRewriteOutput::DerivedProductSum(index) => Some(*index),
                    PlannedRewriteOutput::Source(_) => None,
                };
                steps.push(PlannedRewriteStep {
                    emitted_row,
                    rewrite_id: rewrite_id.index(),
                    kind: SelectiveRewriteKind::PolynomialEvaluation,
                    source_rows: source_rows.clone(),
                    output,
                    base,
                    previous,
                    factors,
                });
                previous = next_previous;
                emitted_row += 1;
            }
        }
        if emitted_row != emitted_rows.end {
            return Err(trace_error(
                "projected polynomial step count differs from compiler rewrite interval",
            ));
        }
    }

    for (batch_index, batch) in source_arm.product_sum_batch_traces().iter().enumerate() {
        let rewrite_id = prepared.product_sum_rewrite(batch_index);
        let (source_rows, emitted_rows) = rewrite_geometry(layout, rewrite_id, arm, SelectiveRewriteKind::ProductSum)?;
        let mut emitted_row = emitted_rows.start;
        for identity in &batch.identities {
            if identity.factors.len() <= EVAL_GROUP_SIZE {
                steps.push(PlannedRewriteStep {
                    emitted_row,
                    rewrite_id: rewrite_id.index(),
                    kind: SelectiveRewriteKind::ProductSum,
                    source_rows: source_rows.clone(),
                    output: PlannedRewriteOutput::Source(identity.result.clone()),
                    base: Lc::zero(),
                    previous: None,
                    factors: identity.factors.clone(),
                });
                emitted_row += 1;
                continue;
            }
            let groups = identity.factors.chunks(EVAL_GROUP_SIZE).collect::<Vec<_>>();
            let mut previous = None;
            for (group_index, group) in groups.iter().enumerate() {
                let final_group = group_index + 1 == groups.len();
                let factors = group.to_vec();
                let output = if final_group {
                    PlannedRewriteOutput::Source(identity.result.clone())
                } else {
                    let Some(encoding) = derived.get(derived_cursor) else {
                        return Err(trace_error(
                            "projected product-sum rewrite exceeds derived-product plan",
                        ));
                    };
                    if encoding.previous != previous {
                        return Err(trace_error(
                            "projected product-sum predecessor differs from derived-product plan",
                        ));
                    }
                    if !product_factor_traces_exact(&factors, &encoding.factors) {
                        return Err(trace_error(
                            "projected product-sum factors differ from derived-product witness plan",
                        ));
                    }
                    let output = PlannedRewriteOutput::DerivedProductSum(derived_cursor);
                    derived_cursor += 1;
                    output
                };
                let next_previous = match &output {
                    PlannedRewriteOutput::DerivedProductSum(index) => Some(*index),
                    PlannedRewriteOutput::Source(_) => None,
                };
                steps.push(PlannedRewriteStep {
                    emitted_row,
                    rewrite_id: rewrite_id.index(),
                    kind: SelectiveRewriteKind::ProductSum,
                    source_rows: source_rows.clone(),
                    output,
                    base: Lc::zero(),
                    previous,
                    factors,
                });
                previous = next_previous;
                emitted_row += 1;
            }
        }
        if emitted_row != emitted_rows.end {
            return Err(trace_error(
                "projected product-sum step count differs from compiler rewrite interval",
            ));
        }
    }
    if derived_cursor != derived.len() {
        return Err(trace_error(
            "projected rewrite plan did not consume every compiler-derived product sum",
        ));
    }
    Ok(steps)
}

fn verify_rewrite_step(
    step: &PlannedRewriteStep,
    artifact: &SelectiveProjectedRowArtifact,
    layout: &super::SelectiveLayout,
    arm: usize,
) -> Result<(), LowNormR1csError> {
    let emitted_run = layout
        .compiler_audit
        .rows()
        .emitted_runs()
        .get(artifact.run_index)
        .ok_or_else(|| trace_error("projected rewrite row has an invalid emitted-run owner"))?;
    if artifact.emitted_row != step.emitted_row
        || emitted_run.rewrite_id().map(|id| id.index()) != Some(step.rewrite_id)
        || emitted_run.family() != artifact.family
        || emitted_run.arm() != Some(arm)
    {
        return Err(trace_error(
            "projected executable rewrite step differs from its emitted-row owner",
        ));
    }

    let mut expected = (0..SELECTIVE_ARITY)
        .map(|_| MatrixTerms::new(false))
        .collect::<Vec<_>>();
    expected[EVAL_SELECTOR].push((0, layout.selector_cols[arm], F::ONE));
    match &step.output {
        PlannedRewriteOutput::Source(output) => {
            append_lc(
                &mut expected[C],
                0,
                output,
                &layout.slots[arm],
                &layout.plans[arm].definitions,
            )?;
        }
        PlannedRewriteOutput::DerivedProductSum(index) => {
            append_slot(
                &mut expected[C],
                0,
                layout.derived_product_sums[arm][*index].slot,
                F::ONE,
            );
        }
    }
    append_lc_scaled(
        &mut expected[C],
        0,
        &step.base,
        -F::ONE,
        &layout.slots[arm],
        &layout.plans[arm].definitions,
    )?;
    if let Some(previous) = step.previous {
        append_slot(
            &mut expected[C],
            0,
            layout.derived_product_sums[arm][previous].slot,
            -F::ONE,
        );
    }
    for (pair_index, factor) in step.factors.iter().enumerate() {
        let (left, right) = EVAL_PAIRS[pair_index];
        append_lc_scaled(
            &mut expected[left],
            0,
            &factor.left,
            factor.coefficient,
            &layout.slots[arm],
            &layout.plans[arm].definitions,
        )?;
        append_lc(
            &mut expected[right],
            0,
            &factor.right,
            &layout.slots[arm],
            &layout.plans[arm].definitions,
        )?;
    }
    for (port, terms) in expected.iter().enumerate() {
        if project_port(terms, 0, artifact.columns)? != artifact.ports[port] {
            return Err(trace_error(
                "projected executable rewrite step does not reproduce its exact emitted row",
            ));
        }
    }
    Ok(())
}

fn projected_factor(factor: &ProductFactorTrace) -> SelectiveProjectedProductFactor {
    SelectiveProjectedProductFactor {
        left_constant: factor.left.constant,
        left_terms: source_terms(&factor.left.terms),
        right_constant: factor.right.constant,
        right_terms: source_terms(&factor.right.terms),
        coefficient: factor.coefficient,
    }
}

fn source_row_lc(
    matrix: &CcsMatrix<F>,
    row: usize,
) -> Result<SelectiveProjectedSourceLinearCombination, LowNormR1csError> {
    let mut canonical = BTreeMap::<usize, F>::new();
    let mut add = |column: usize, coefficient: F| {
        if coefficient != F::ZERO {
            *canonical.entry(column).or_insert(F::ZERO) += coefficient;
        }
    };
    let mut append_csc = |csc: &neo_ccs::CscMat<F>| {
        for column in 0..csc.ncols {
            for entry in csc.column_range(column) {
                if csc.row_index(entry) == row {
                    add(column, csc.vals[entry]);
                }
            }
        }
    };
    match matrix {
        CcsMatrix::Identity { n } => {
            if row >= *n {
                return Err(trace_error("projected retained source row exceeds identity port"));
            }
            add(row, F::ONE);
        }
        CcsMatrix::Csc(csc) => append_csc(csc),
        CcsMatrix::CscWithSeededPhi81 {
            csc,
            blocks,
            geometric_runs,
        } => {
            if blocks
                .iter()
                .any(|block| (block.row_start()..block.row_end()).contains(&row))
            {
                return Err(trace_error(
                    "projected retained source row intersects a compact seeded row",
                ));
            }
            append_csc(csc);
            for run in geometric_runs.iter().filter(|run| run.row() == row) {
                let mut coefficient = *run.initial();
                for column in run.column_start()..run.column_start() + run.len() {
                    add(column, coefficient);
                    coefficient *= *run.ratio();
                }
            }
        }
    }
    canonical.retain(|_, coefficient| *coefficient != F::ZERO);
    let constant = canonical.remove(&0).unwrap_or(F::ZERO);
    Ok(SelectiveProjectedSourceLinearCombination {
        constant,
        terms: canonical
            .into_iter()
            .map(|(column, coefficient)| SelectiveProjectedSourceTerm { column, coefficient })
            .collect(),
    })
}

fn verify_retained_step(
    step: &SelectiveProjectedRetainedStep,
    artifact: &SelectiveProjectedRowArtifact,
    layout: &super::SelectiveLayout,
    arm: usize,
) -> Result<(), LowNormR1csError> {
    use super::super::selective_audit::SelectiveEmittedRowFamily;

    let emitted_run = layout
        .compiler_audit
        .rows()
        .emitted_runs()
        .get(artifact.run_index)
        .ok_or_else(|| trace_error("projected retained row has an invalid emitted-run owner"))?;
    if artifact.emitted_row != step.emitted_row
        || artifact.family != SelectiveEmittedRowFamily::Retained
        || emitted_run.family() != SelectiveEmittedRowFamily::Retained
        || emitted_run.rewrite_id().is_some()
        || emitted_run.arm() != Some(arm)
    {
        return Err(trace_error(
            "projected retained source step differs from its emitted-row owner",
        ));
    }

    let mut expected = (0..SELECTIVE_ARITY)
        .map(|_| MatrixTerms::new(false))
        .collect::<Vec<_>>();
    expected[GENERAL_SELECTOR].push((0, layout.selector_cols[arm], F::ONE));
    for (port, source) in [A, B, C].into_iter().zip(&step.ports) {
        append_field(
            &mut expected[port],
            0,
            0,
            source.constant,
            &layout.slots[arm],
            &layout.plans[arm].definitions,
        )?;
        for term in &source.terms {
            append_field(
                &mut expected[port],
                0,
                term.column,
                term.coefficient,
                &layout.slots[arm],
                &layout.plans[arm].definitions,
            )?;
        }
    }
    for (port, terms) in expected.iter().enumerate() {
        if project_port(terms, 0, artifact.columns)? != artifact.ports[port] {
            return Err(trace_error(
                "projected retained source step does not reproduce its exact emitted row",
            ));
        }
    }
    Ok(())
}

fn source_provenance(
    source_arm: &SparseR1cs,
    layout: &super::SelectiveLayout,
    arm: usize,
    requested_source_columns: &[usize],
    retained_row_pairs: &[(usize, usize)],
    row_artifacts: &[SelectiveProjectedRowArtifact],
) -> Result<SelectiveProjectedSourceProvenance, LowNormR1csError> {
    let Some(slots) = layout.slots.get(arm) else {
        return Err(trace_error("projected source-provenance arm is out of range"));
    };
    let plan = &layout.plans[arm];
    let mut closure = requested_source_columns
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if closure.iter().any(|&column| column >= slots.len()) {
        return Err(trace_error("projected source-provenance column exceeds its source arm"));
    }

    loop {
        let mut added = false;
        for column in closure.iter().copied().collect::<Vec<_>>() {
            let Some(rhs) = plan.definitions.get(column) else {
                continue;
            };
            for &(dependency, _) in &rhs.terms {
                if dependency >= slots.len() {
                    return Err(trace_error(
                        "projected compiler definition references an out-of-range source column",
                    ));
                }
                added |= closure.insert(dependency);
            }
        }
        if !added {
            break;
        }
    }

    let mut retained_slots = Vec::new();
    let mut trace_eliminated_columns = Vec::new();
    for &column in &closure {
        if column == 0 || plan.definitions.get(column).is_some() {
            continue;
        }
        if let Some((start, width)) = slots[column] {
            if width == 0 || start + width > layout.columns.next_multiple_of(D) {
                return Err(trace_error("projected retained source slot is out of range"));
            }
            retained_slots.push(SelectiveProjectedSourceSlot { column, start, width });
        } else {
            trace_eliminated_columns.push(column);
        }
    }

    let linear_definitions = plan
        .definitions
        .entries
        .iter()
        .filter(|definition| closure.contains(&definition.target))
        .map(|definition| SelectiveProjectedSourceDefinition {
            target: definition.target,
            constant: definition.rhs.constant,
            terms: source_terms(&definition.rhs.terms),
        })
        .collect::<Vec<_>>();
    let definition_targets = linear_definitions
        .iter()
        .map(SelectiveProjectedSourceDefinition::target)
        .collect::<BTreeSet<_>>();
    if definition_targets.len() != linear_definitions.len()
        || definition_targets
            != closure
                .iter()
                .copied()
                .filter(|&column| plan.definitions.get(column).is_some())
                .collect()
    {
        return Err(trace_error(
            "projected source-provenance definition closure is incomplete",
        ));
    }

    let derived = &layout.derived_product_sums[arm];
    let mut selected_derived = derived
        .iter()
        .enumerate()
        .filter(|(_, encoding)| {
            row_artifacts
                .iter()
                .flat_map(|row| row.ports.iter())
                .any(|port| port_intersects_slot(port, encoding.slot))
        })
        .map(|(index, _)| index)
        .collect::<BTreeSet<_>>();
    loop {
        let mut added = false;
        for index in selected_derived.iter().copied().collect::<Vec<_>>() {
            if let Some(previous) = derived[index].previous {
                if previous >= index {
                    return Err(trace_error(
                        "projected derived product predecessor is not earlier in compiler order",
                    ));
                }
                added |= selected_derived.insert(previous);
            }
        }
        if !added {
            break;
        }
    }
    let derived_product_sums = selected_derived
        .into_iter()
        .map(|compiler_index| {
            let encoding = &derived[compiler_index];
            SelectiveProjectedDerivedProductSum {
                compiler_index,
                start: encoding.slot.0,
                width: encoding.slot.1,
                factors: encoding
                    .factors
                    .iter()
                    .map(|factor| SelectiveProjectedProductFactor {
                        left_constant: factor.left.constant,
                        left_terms: source_terms(&factor.left.terms),
                        right_constant: factor.right.constant,
                        right_terms: source_terms(&factor.right.terms),
                        coefficient: factor.coefficient,
                    })
                    .collect(),
                previous: encoding.previous,
            }
        })
        .collect();

    let artifacts_by_row = row_artifacts
        .iter()
        .map(|artifact| (artifact.emitted_row, artifact))
        .collect::<BTreeMap<_, _>>();
    let retained_steps = retained_row_pairs
        .iter()
        .map(|&(source_row, emitted_row)| {
            if source_row >= source_arm.n {
                return Err(trace_error("projected retained source row is out of range"));
            }
            let artifact = artifacts_by_row
                .get(&emitted_row)
                .copied()
                .ok_or_else(|| trace_error("projected retained emitted row is absent"))?;
            let step = SelectiveProjectedRetainedStep {
                emitted_row,
                source_row,
                ports: [
                    source_row_lc(&source_arm.a, source_row)?,
                    source_row_lc(&source_arm.b, source_row)?,
                    source_row_lc(&source_arm.c, source_row)?,
                ],
            };
            verify_retained_step(&step, artifact, layout, arm)?;
            Ok(step)
        })
        .collect::<Result<Vec<_>, LowNormR1csError>>()?;
    let expected_retained_rows = row_artifacts
        .iter()
        .filter(|artifact| artifact.family == super::super::selective_audit::SelectiveEmittedRowFamily::Retained)
        .map(|artifact| artifact.emitted_row)
        .collect::<BTreeSet<_>>();
    let actual_retained_rows = retained_steps
        .iter()
        .map(|step| step.emitted_row)
        .collect::<BTreeSet<_>>();
    if retained_steps.len() != retained_row_pairs.len()
        || actual_retained_rows.len() != retained_steps.len()
        || actual_retained_rows != expected_retained_rows
    {
        return Err(trace_error(
            "projected retained source steps do not cover every selected retained row",
        ));
    }
    let selected_steps = planned_rewrite_steps(source_arm, layout, arm)?
        .into_iter()
        .filter(|step| artifacts_by_row.contains_key(&step.emitted_row))
        .collect::<Vec<_>>();
    let expected_step_count = row_artifacts
        .iter()
        .filter(|artifact| {
            matches!(
                artifact.family,
                super::super::selective_audit::SelectiveEmittedRowFamily::PolynomialEvaluation
                    | super::super::selective_audit::SelectiveEmittedRowFamily::ProductSum
            )
        })
        .count();
    if selected_steps.len() != expected_step_count {
        return Err(trace_error(
            "projected executable rewrite steps do not cover every selected rewrite row",
        ));
    }
    for step in &selected_steps {
        if matches!(step.output, PlannedRewriteOutput::DerivedProductSum(_))
            && (step.base.constant != F::ZERO || !step.base.terms.is_empty())
        {
            return Err(trace_error(
                "projected derived-product rewrite has a base term absent from the witness encoding",
            ));
        }
        verify_rewrite_step(step, artifacts_by_row[&step.emitted_row], layout, arm)?;
    }
    let rewrite_steps = selected_steps
        .into_iter()
        .map(|step| SelectiveProjectedRewriteStep {
            emitted_row: step.emitted_row,
            rewrite_id: step.rewrite_id,
            kind: step.kind,
            source_rows: step.source_rows,
            output: match step.output {
                PlannedRewriteOutput::Source(output) => SelectiveProjectedRewriteOutput::Source {
                    constant: output.constant,
                    terms: source_terms(&output.terms),
                },
                PlannedRewriteOutput::DerivedProductSum(compiler_index) => {
                    SelectiveProjectedRewriteOutput::DerivedProductSum { compiler_index }
                }
            },
            base_constant: step.base.constant,
            base_terms: source_terms(&step.base.terms),
            previous: step.previous,
            factors: step.factors.iter().map(projected_factor).collect(),
        })
        .collect::<Vec<_>>();

    let source_term_in_closure = |term: &SelectiveProjectedSourceTerm| closure.contains(&term.column);
    if rewrite_steps.iter().any(|step| {
        let output_outside = match &step.output {
            SelectiveProjectedRewriteOutput::Source { terms, .. } => {
                terms.iter().any(|term| !source_term_in_closure(term))
            }
            SelectiveProjectedRewriteOutput::DerivedProductSum { .. } => false,
        };
        output_outside
            || step
                .base_terms
                .iter()
                .any(|term| !source_term_in_closure(term))
            || step.factors.iter().any(|factor| {
                factor
                    .left_terms
                    .iter()
                    .chain(&factor.right_terms)
                    .any(|term| !source_term_in_closure(term))
            })
    }) {
        return Err(trace_error(
            "projected executable rewrite step references a source column outside its closure",
        ));
    }
    if retained_steps.iter().any(|step| {
        step.ports
            .iter()
            .flat_map(|port| &port.terms)
            .any(|term| !closure.contains(&term.column))
    }) {
        return Err(trace_error(
            "projected retained source step references a source column outside its closure",
        ));
    }

    Ok(SelectiveProjectedSourceProvenance {
        arm,
        source_columns: closure.into_iter().collect(),
        retained_slots,
        linear_definitions,
        trace_eliminated_columns,
        derived_product_sums,
        rewrite_steps,
        retained_steps,
    })
}

pub(crate) fn project_rows_with_alignment(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    shared_private_bit_fields: usize,
    modulus: usize,
    residue: usize,
    selected_rows: &[usize],
) -> Result<SelectiveProjectedRowsAudit, LowNormR1csError> {
    project_rows_inner(
        arms,
        shared_private_fields,
        shared_private_bit_fields,
        modulus,
        residue,
        selected_rows,
        None,
        None,
    )
}

pub(crate) fn project_rows_with_source_provenance_with_alignment(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    shared_private_bit_fields: usize,
    modulus: usize,
    residue: usize,
    selected_rows: &[usize],
    source_arm: usize,
    source_columns: &[usize],
    retained_row_pairs: &[(usize, usize)],
    decoder_source_columns: &[usize],
) -> Result<SelectiveProjectedRowsAudit, LowNormR1csError> {
    project_rows_inner(
        arms,
        shared_private_fields,
        shared_private_bit_fields,
        modulus,
        residue,
        selected_rows,
        Some((
            source_arm,
            source_columns,
            retained_row_pairs,
            Some(decoder_source_columns),
        )),
        None,
    )
}

/// Project a selected row slice and, in the same prepared-layout pass,
/// export one complete run-compressed decoder interval.  The extra decoder
/// request does not add selected rows or change the row projection.
#[allow(clippy::too_many_arguments)]
pub(crate) fn project_rows_with_source_provenance_and_decoder_runs_with_alignment(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    shared_private_bit_fields: usize,
    modulus: usize,
    residue: usize,
    selected_rows: &[usize],
    source_arm: usize,
    source_columns: &[usize],
    retained_row_pairs: &[(usize, usize)],
    decoder_source_columns: &[usize],
    decoder_run_source_range: std::ops::Range<usize>,
) -> Result<SelectiveProjectedRowsAudit, LowNormR1csError> {
    project_rows_inner(
        arms,
        shared_private_fields,
        shared_private_bit_fields,
        modulus,
        residue,
        selected_rows,
        Some((
            source_arm,
            source_columns,
            retained_row_pairs,
            Some(decoder_source_columns),
        )),
        Some((source_arm, decoder_run_source_range)),
    )
}

/// Project rows once and decode the exact transitive source-column closure
/// computed by that same provenance pass.
pub(crate) fn project_rows_with_complete_source_provenance_with_alignment(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    shared_private_bit_fields: usize,
    modulus: usize,
    residue: usize,
    selected_rows: &[usize],
    source_arm: usize,
    source_columns: &[usize],
    retained_row_pairs: &[(usize, usize)],
) -> Result<SelectiveProjectedRowsAudit, LowNormR1csError> {
    project_rows_inner(
        arms,
        shared_private_fields,
        shared_private_bit_fields,
        modulus,
        residue,
        selected_rows,
        Some((source_arm, source_columns, retained_row_pairs, None)),
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn project_rows_inner(
    arms: &[SparseR1cs],
    shared_private_fields: usize,
    shared_private_bit_fields: usize,
    modulus: usize,
    residue: usize,
    selected_rows: &[usize],
    source_request: Option<(usize, &[usize], &[(usize, usize)], Option<&[usize]>)>,
    decoder_run_request: Option<(usize, std::ops::Range<usize>)>,
) -> Result<SelectiveProjectedRowsAudit, LowNormR1csError> {
    let layout = prepare_selective_layout(arms, shared_private_fields, shared_private_bit_fields, modulus, residue)?;
    let public_coordinates = public_coordinate_decoder(arms, &layout)?;
    let emitted = structure::emit_structure_terms(
        arms,
        &layout.plans,
        &layout.slots,
        &layout.aliases,
        &layout.equal_aliases,
        shared_private_fields,
        &layout.derived_product_sums,
        &layout.selector_cols,
        &layout.public_padding_cols,
        &layout.private_padding_cols,
        layout.columns,
        &layout.prepared_rows,
    )?;
    if emitted.rows != layout.compiler_audit.rows().total_rows() {
        return Err(trace_error(
            "projected emitter row count differs from its compiler audit",
        ));
    }

    let mut unique = BTreeSet::new();
    for &row in selected_rows {
        if row >= emitted.rows {
            return Err(trace_error("requested selective projection row is out of range"));
        }
        if !unique.insert(row) {
            return Err(trace_error("requested selective projection row is duplicated"));
        }
    }

    let mut row_artifacts = Vec::with_capacity(selected_rows.len());
    for &row in selected_rows {
        row_artifacts.push(project_row_artifact(&emitted, &layout.compiler_audit, row)?);
    }

    let public_padding_runs = layout
        .compiler_audit
        .rows()
        .emitted_runs()
        .iter()
        .filter(|run| run.family() == super::super::selective_audit::SelectiveEmittedRowFamily::PublicPadding)
        .collect::<Vec<_>>();
    let [public_padding_run] = public_padding_runs.as_slice() else {
        return Err(trace_error(
            "projected emitter must have exactly one public-padding owner",
        ));
    };
    if public_padding_run.arm().is_some() || public_padding_run.emitted_rows().len() != layout.public_padding_cols.len()
    {
        return Err(trace_error(
            "projected public-padding owner differs from the prepared public range",
        ));
    }
    let public_padding_row_artifacts = public_padding_run
        .emitted_rows()
        .map(|row| project_row_artifact(&emitted, &layout.compiler_audit, row))
        .collect::<Result<Vec<_>, _>>()?;

    let selector_domain_runs = layout
        .compiler_audit
        .rows()
        .emitted_runs()
        .iter()
        .filter(|run| run.family() == super::super::selective_audit::SelectiveEmittedRowFamily::SelectorDomain)
        .collect::<Vec<_>>();
    let [selector_domain_run] = selector_domain_runs.as_slice() else {
        return Err(trace_error(
            "projected emitter must have exactly one selector-domain owner",
        ));
    };
    if selector_domain_run.arm().is_some() || selector_domain_run.emitted_rows().len() != layout.selector_cols.len() {
        return Err(trace_error(
            "projected selector-domain owner differs from the prepared selector range",
        ));
    }
    let selector_domain_row_artifacts = selector_domain_run
        .emitted_rows()
        .map(|row| project_row_artifact(&emitted, &layout.compiler_audit, row))
        .collect::<Result<Vec<_>, _>>()?;

    let one_hot_runs = layout
        .compiler_audit
        .rows()
        .emitted_runs()
        .iter()
        .filter(|run| run.family() == super::super::selective_audit::SelectiveEmittedRowFamily::OneHot)
        .collect::<Vec<_>>();
    let [one_hot_run] = one_hot_runs.as_slice() else {
        return Err(trace_error("projected emitter must have exactly one one-hot owner"));
    };
    if one_hot_run.arm().is_some() || one_hot_run.emitted_rows().len() != 1 {
        return Err(trace_error(
            "projected one-hot owner differs from the prepared selector-total row",
        ));
    }
    let one_hot_row_artifact =
        project_row_artifact(&emitted, &layout.compiler_audit, one_hot_run.emitted_rows().start)?;

    let private_padding_runs = layout
        .compiler_audit
        .rows()
        .emitted_runs()
        .iter()
        .filter(|run| run.family() == super::super::selective_audit::SelectiveEmittedRowFamily::PrivatePadding)
        .collect::<Vec<_>>();
    let [private_padding_run] = private_padding_runs.as_slice() else {
        return Err(trace_error(
            "projected emitter must have exactly one private-padding owner",
        ));
    };
    if private_padding_run.arm().is_some()
        || private_padding_run.emitted_rows().len() != layout.private_padding_cols.len()
    {
        return Err(trace_error(
            "projected private-padding owner differs from the prepared alignment range",
        ));
    }
    let private_padding_row_artifacts = private_padding_run
        .emitted_rows()
        .map(|row| project_row_artifact(&emitted, &layout.compiler_audit, row))
        .collect::<Result<Vec<_>, _>>()?;

    let ring_padding_runs = layout
        .compiler_audit
        .rows()
        .emitted_runs()
        .iter()
        .filter(|run| run.family() == super::super::selective_audit::SelectiveEmittedRowFamily::RingPadding)
        .collect::<Vec<_>>();
    let [ring_padding_run] = ring_padding_runs.as_slice() else {
        return Err(trace_error(
            "projected emitter must have exactly one ring-padding owner",
        ));
    };
    let expected_ring_padding_rows = layout.compiler_audit.rows().ring_padding_rows();
    if ring_padding_run.arm().is_some()
        || ring_padding_run.emitted_rows() != expected_ring_padding_rows
        || ring_padding_run.emitted_rows().len() != emitted.columns - layout.columns
    {
        return Err(trace_error(
            "projected ring-padding owner differs from the final alignment range",
        ));
    }
    let ring_padding_row_artifacts = ring_padding_run
        .emitted_rows()
        .map(|row| project_row_artifact(&emitted, &layout.compiler_audit, row))
        .collect::<Result<Vec<_>, _>>()?;

    let source_provenance = source_request
        .map(|(arm, source_columns, retained_row_pairs, _)| {
            let source_arm = arms
                .get(arm)
                .ok_or_else(|| trace_error("projected source-provenance arm is out of range"))?;
            source_provenance(
                source_arm,
                &layout,
                arm,
                source_columns,
                retained_row_pairs,
                &row_artifacts,
            )
        })
        .transpose()?;
    let decoder_provenance = match source_request {
        None => None,
        Some((arm, _, _, requested)) => {
            let decoder_source_columns = match requested {
                Some(columns) => columns,
                None => source_provenance
                    .as_ref()
                    .ok_or_else(|| trace_error("projected complete decoder omitted source provenance"))?
                    .source_columns(),
            };
            Some(decoder_provenance(&layout, arm, decoder_source_columns)?)
        }
    };
    let decoder_run_provenance = decoder_run_request
        .map(|(arm, source_range)| {
            let source_arm = arms
                .get(arm)
                .ok_or_else(|| trace_error("complete decoder arm is out of range"))?;
            decoder_run_provenance(&layout, arm, source_range, source_arm.column_family_ranges())
        })
        .transpose()?;

    Ok(SelectiveProjectedRowsAudit {
        rows: emitted.rows,
        columns: emitted.columns,
        selector_columns: layout.selector_cols,
        compiler_audit: layout.compiler_audit,
        public_coordinates,
        public_padding_row_artifacts,
        selector_domain_row_artifacts,
        one_hot_row_artifact,
        private_padding_row_artifacts,
        ring_padding_row_artifacts,
        row_artifacts,
        source_provenance,
        decoder_provenance,
        decoder_run_provenance,
    })
}
