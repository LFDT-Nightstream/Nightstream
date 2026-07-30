//! Read-only width and physical row attribution for selective lowering.
//!
//! Owns: inclusive row-family coverage, retained trace widths, branch widths,
//! alias counts, total selective-width aggregation, the exclusive rewrite
//! ledger, and the source arms' caller-supplied physical-stage intervals.
//!
//! Does not own: constraint semantics, row emission, source-trace validity,
//! stage-label semantics, expected-tree membership, or performance budgets.
//!
//! Emits constraints: no.
//!
//! Authority boundary: these records are diagnostic metadata derived from a
//! prepared layout and never justify relation acceptance or row removal.
//!
//! | Obligation | Local owner | Emits constraints? | Authority source |
//! |---|---|---|---|
//! | Coordinate layout | [`SelectiveLayoutAudit`] | no | Exact prepared ranges used by the emitter |
//! | Row-family width | [`SelectiveFamilyWidthAudit`] | no | Recorded family ranges |
//! | Trace width | [`SelectiveTraceWidthAudit`] | no | Retained trace descriptors |
//! | Branch/total width | arm and total audit types | no | Prepared selective layout |
//! | Source rewrite ledger | [`SelectiveRowMappingAudit`] | no | Prepared selective row plan |
//! | Source physical stages | [`SelectiveCompilerAudit`] | no | Validated source-row intervals; labels remain caller assertions |

use core::ops::Range;

use neo_ccs::{CcsMatrix, CscMat};

use crate::engine::r1cs_circuit::PhysicalStageRange;

/// Exact coordinate regions captured by the same prepared layout consumed by
/// the selective row emitter. Fields are private so callers cannot fabricate a
/// provenance-bearing record; accessors are read-only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveLayoutAudit {
    logical_public_input_len: usize,
    public_input_len: usize,
    public_padding_columns: Vec<usize>,
    selector_columns: Vec<usize>,
    private_alignment_padding_columns: Vec<usize>,
    shared_private_columns: Range<usize>,
    branch_columns: Range<usize>,
    ring_alignment_padding_columns: Range<usize>,
}

impl SelectiveLayoutAudit {
    pub(super) fn from_prepared_layout(
        logical_public_input_len: usize,
        public_input_len: usize,
        public_padding_columns: Vec<usize>,
        selector_columns: Vec<usize>,
        private_alignment_padding_columns: Vec<usize>,
        shared_private_columns: Range<usize>,
        branch_columns: Range<usize>,
        ring_alignment_padding_columns: Range<usize>,
    ) -> Self {
        Self {
            logical_public_input_len,
            public_input_len,
            public_padding_columns,
            selector_columns,
            private_alignment_padding_columns,
            shared_private_columns,
            branch_columns,
            ring_alignment_padding_columns,
        }
    }

    pub fn logical_public_input_len(&self) -> usize {
        self.logical_public_input_len
    }

    pub fn public_input_len(&self) -> usize {
        self.public_input_len
    }

    pub fn public_padding_columns(&self) -> &[usize] {
        &self.public_padding_columns
    }

    pub fn selector_columns(&self) -> &[usize] {
        &self.selector_columns
    }

    pub fn private_alignment_padding_columns(&self) -> &[usize] {
        &self.private_alignment_padding_columns
    }

    pub fn shared_private_columns(&self) -> Range<usize> {
        self.shared_private_columns.clone()
    }

    pub fn branch_columns(&self) -> Range<usize> {
        self.branch_columns.clone()
    }

    pub fn ring_alignment_padding_columns(&self) -> Range<usize> {
        self.ring_alignment_padding_columns.clone()
    }

    pub fn total_columns(&self) -> usize {
        self.ring_alignment_padding_columns.end
    }
}

/// Retained low-norm coordinates touched by one non-authoritative row-family
/// marker. Nested families overlap by design.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveFamilyWidthAudit {
    pub name: &'static str,
    pub inclusive_rows: usize,
    pub unit_columns: usize,
    pub balanced_columns: usize,
    pub binary_columns: usize,
    pub coordinates_before_aliases: usize,
    pub poseidon2_permutations: usize,
    pub poseidon2_coordinates: usize,
}

fn covered_rows(ranges: &[(usize, usize)]) -> usize {
    let mut ranges = ranges.to_vec();
    ranges.sort_unstable();
    let mut total = 0usize;
    let mut current = None::<(usize, usize)>;
    for (start, end) in ranges {
        current = match current {
            None => Some((start, end)),
            Some((current_start, current_end)) if start <= current_end => Some((current_start, current_end.max(end))),
            Some((current_start, current_end)) => {
                total += current_end - current_start;
                Some((start, end))
            }
        };
    }
    if let Some((start, end)) = current {
        total += end - start;
    }
    total
}

/// Retained source values owned by direct selective trace classes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveTraceWidthAudit {
    pub poseidon2_permutations: usize,
    pub poseidon2_columns: usize,
    pub poseidon2_coordinates: usize,
    pub polynomial_evaluation_columns: usize,
    pub polynomial_evaluation_coordinates: usize,
    pub product_sum_columns: usize,
    pub product_sum_coordinates: usize,
    pub product_sum_internal_columns: usize,
    pub product_sum_internal_coordinates: usize,
}

/// Exact committed-width census for one branch-private suffix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveArmWidthAudit {
    pub branch_source_columns: usize,
    pub eliminated_columns: usize,
    pub unit_columns: usize,
    pub balanced_columns: usize,
    pub binary_columns: usize,
    pub retained_coordinates_before_aliases: usize,
    pub decomposition_aliases: usize,
    pub equality_aliases: usize,
    pub branch_coordinates: usize,
    pub derived_product_sums: usize,
    pub derived_coordinates: usize,
    pub total_branch_coordinates: usize,
    pub traces: SelectiveTraceWidthAudit,
    pub row_families: Vec<SelectiveFamilyWidthAudit>,
}

/// Exact width contract produced before any CCS matrices are allocated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveLowNormWidthAudit {
    pub constant_coordinate: usize,
    /// Encoded F' public coordinates excluding the conventional constant.
    pub logical_public_coordinates: usize,
    /// Verifier-fixed zeros completing the public prefix to a ring boundary.
    pub public_carrier_padding: usize,
    /// Complete public prefix excluding the conventional constant.
    pub public_coordinates: usize,
    pub selector_coordinates: usize,
    /// Verifier-fixed zeros between selectors and shared private advice.
    pub alignment_padding: usize,
    pub shared_private_coordinates: usize,
    pub branch_start: usize,
    pub arms: Vec<SelectiveArmWidthAudit>,
    pub total_coordinates: usize,
}

/// Exact final-coordinate layout for one shifted-ternary opening.
///
/// The 41 digit coordinates are the Ajtai message word. The 20 borrow
/// coordinates are the retained endpoints between adjacent two-trit chunks.
/// Negative indicators and the other 20 source borrow columns are absent from
/// the final selective assignment. Coordinates are distinct within this
/// opening; separate openings may deliberately alias equal source values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveCanonicalOpeningAudit {
    source_field: usize,
    digit_coordinates: Vec<usize>,
    borrow_coordinates: Vec<usize>,
    emitted_rows: Range<usize>,
}

impl SelectiveCanonicalOpeningAudit {
    pub(super) fn new(
        source_field: usize,
        digit_coordinates: Vec<usize>,
        borrow_coordinates: Vec<usize>,
        emitted_rows: Range<usize>,
    ) -> Self {
        Self {
            source_field,
            digit_coordinates,
            borrow_coordinates,
            emitted_rows,
        }
    }

    pub fn source_field(&self) -> usize {
        self.source_field
    }

    pub fn digit_coordinates(&self) -> &[usize] {
        &self.digit_coordinates
    }

    pub fn borrow_coordinates(&self) -> &[usize] {
        &self.borrow_coordinates
    }

    pub fn emitted_rows(&self) -> Range<usize> {
        self.emitted_rows.clone()
    }

    pub fn coordinate_count(&self) -> usize {
        self.digit_coordinates.len() + self.borrow_coordinates.len()
    }
}

/// Stable identifier assigned by the prepared row plan to one physical rewrite.
///
/// The number is only a join key between source and emitted intervals. It does
/// not name a theorem or authorize the rewrite.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SelectiveRewriteId(u32);

impl SelectiveRewriteId {
    pub(super) fn from_index(index: usize) -> Option<Self> {
        u32::try_from(index).ok().map(Self)
    }

    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Mechanical source-row rewrite classes recognized by selective lowering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SelectiveRewriteKind {
    Poseidon2,
    CenteredUnit,
    ShiftedTernaryCanonical,
    PolynomialEvaluation,
    ProductSum,
    LinearDefinition,
}

/// Exclusive disposition of one physical source-row interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SelectiveSourceRowDisposition {
    Retained,
    Poseidon2(SelectiveRewriteId),
    CenteredUnit(SelectiveRewriteId),
    ShiftedTernaryCanonical(SelectiveRewriteId),
    PolynomialEvaluation(SelectiveRewriteId),
    ProductSum(SelectiveRewriteId),
    LinearDefinition(SelectiveRewriteId),
}

impl SelectiveSourceRowDisposition {
    pub fn rewrite_id(self) -> Option<SelectiveRewriteId> {
        match self {
            Self::Retained => None,
            Self::Poseidon2(id)
            | Self::CenteredUnit(id)
            | Self::ShiftedTernaryCanonical(id)
            | Self::PolynomialEvaluation(id)
            | Self::ProductSum(id)
            | Self::LinearDefinition(id) => Some(id),
        }
    }
}

/// Exclusive compiler owner of one emitted-row interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SelectiveEmittedRowFamily {
    SelectorDomain,
    SharedDomain,
    ArmDomain,
    OneHot,
    PublicPadding,
    PrivatePadding,
    Retained,
    Poseidon2,
    CenteredUnit,
    ShiftedTernaryCanonical,
    PolynomialEvaluation,
    ProductSum,
    RingPadding,
}

/// One maximal source-row run with uniform physical ownership.
///
/// Runs are split at physical-stage occurrence boundaries even when the same
/// path string is repeated. A retained run maps monotonically from
/// `source_rows.start` to `emitted_start`; rewritten runs use `None`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveSourceRowRunAudit {
    source_rows: Range<usize>,
    disposition: SelectiveSourceRowDisposition,
    stage_occurrence: Option<usize>,
    emitted_start: Option<usize>,
}

impl SelectiveSourceRowRunAudit {
    pub(super) fn new(
        source_rows: Range<usize>,
        disposition: SelectiveSourceRowDisposition,
        stage_occurrence: Option<usize>,
        emitted_start: Option<usize>,
    ) -> Self {
        Self {
            source_rows,
            disposition,
            stage_occurrence,
            emitted_start,
        }
    }

    pub fn source_rows(&self) -> Range<usize> {
        self.source_rows.clone()
    }

    pub fn disposition(&self) -> SelectiveSourceRowDisposition {
        self.disposition
    }

    /// Index in `SelectiveCompilerAudit::source_arm_physical_stages`.
    /// `None` means the source arm supplied no physical-stage schedule.
    pub fn stage_occurrence(&self) -> Option<usize> {
        self.stage_occurrence
    }

    pub fn emitted_start(&self) -> Option<usize> {
        self.emitted_start
    }
}

/// One exact emitted interval consumed in compiler order.
///
/// Empty intervals are retained in the plan so zero-row emitted families stay
/// visible. Source-to-empty rewrites live in [`SelectiveRewriteAudit`].
/// Nonempty intervals partition the final relation rows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveEmittedRowRunAudit {
    emitted_rows: Range<usize>,
    family: SelectiveEmittedRowFamily,
    arm: Option<usize>,
    rewrite_id: Option<SelectiveRewriteId>,
    source_stage_occurrence: Option<usize>,
}

impl SelectiveEmittedRowRunAudit {
    pub(super) fn new(
        emitted_rows: Range<usize>,
        family: SelectiveEmittedRowFamily,
        arm: Option<usize>,
        rewrite_id: Option<SelectiveRewriteId>,
        source_stage_occurrence: Option<usize>,
    ) -> Self {
        Self {
            emitted_rows,
            family,
            arm,
            rewrite_id,
            source_stage_occurrence,
        }
    }

    pub fn emitted_rows(&self) -> Range<usize> {
        self.emitted_rows.clone()
    }

    pub fn family(&self) -> SelectiveEmittedRowFamily {
        self.family
    }

    pub fn arm(&self) -> Option<usize> {
        self.arm
    }

    pub fn rewrite_id(&self) -> Option<SelectiveRewriteId> {
        self.rewrite_id
    }

    pub fn source_stage_occurrence(&self) -> Option<usize> {
        self.source_stage_occurrence
    }
}

/// One planned physical rewrite and its exact source/output geometry.
///
/// `emitted_rows` may be empty. In particular, a linear definition is recorded
/// as a source-to-empty rewrite rather than being hidden in a skipped-row bit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveRewriteAudit {
    id: SelectiveRewriteId,
    arm: usize,
    kind: SelectiveRewriteKind,
    source_rows: Vec<Range<usize>>,
    emitted_rows: Range<usize>,
    source_stage_occurrence: Option<usize>,
}

impl SelectiveRewriteAudit {
    pub(super) fn new(
        id: SelectiveRewriteId,
        arm: usize,
        kind: SelectiveRewriteKind,
        source_rows: Vec<Range<usize>>,
        source_stage_occurrence: Option<usize>,
    ) -> Self {
        Self {
            id,
            arm,
            kind,
            source_rows,
            emitted_rows: 0..0,
            source_stage_occurrence,
        }
    }

    pub(super) fn set_emitted_rows(&mut self, emitted_rows: Range<usize>) {
        self.emitted_rows = emitted_rows;
    }

    pub fn id(&self) -> SelectiveRewriteId {
        self.id
    }

    pub fn arm(&self) -> usize {
        self.arm
    }

    pub fn kind(&self) -> SelectiveRewriteKind {
        self.kind
    }

    pub fn source_rows(&self) -> &[Range<usize>] {
        &self.source_rows
    }

    pub fn emitted_rows(&self) -> Range<usize> {
        self.emitted_rows.clone()
    }

    pub fn source_stage_occurrence(&self) -> Option<usize> {
        self.source_stage_occurrence
    }
}

/// Exact compressed source-row partition for one source arm.
///
/// Every source run has one exclusive disposition and one physical-stage
/// occurrence when provenance exists. Rewritten runs join the global rewrite
/// and emitted-family records through a stable [`SelectiveRewriteId`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveArmRowMappingAudit {
    source_runs: Vec<SelectiveSourceRowRunAudit>,
    retained_emitted_rows: Range<usize>,
    emitted_rows: Range<usize>,
}

impl SelectiveArmRowMappingAudit {
    pub(super) fn new(
        source_runs: Vec<SelectiveSourceRowRunAudit>,
        retained_emitted_rows: Range<usize>,
        emitted_rows: Range<usize>,
    ) -> Self {
        Self {
            source_runs,
            retained_emitted_rows,
            emitted_rows,
        }
    }

    pub fn source_runs(&self) -> &[SelectiveSourceRowRunAudit] {
        &self.source_runs
    }

    pub fn retained_emitted_rows(&self) -> Range<usize> {
        self.retained_emitted_rows.clone()
    }

    pub fn emitted_rows(&self) -> Range<usize> {
        self.emitted_rows.clone()
    }
}

/// Exact source/output row ledger produced by one prepared compiler run.
///
/// Nonempty emitted runs partition every CCS row exactly once. Empty runs keep
/// zero-cost compiler families visible. Rewrite records join their source runs
/// to one exact emitted interval, including source-to-empty elimination.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveRowMappingAudit {
    prefix_rows: Range<usize>,
    arms: Vec<SelectiveArmRowMappingAudit>,
    ring_padding_rows: Range<usize>,
    emitted_runs: Vec<SelectiveEmittedRowRunAudit>,
    rewrites: Vec<SelectiveRewriteAudit>,
    total_rows: usize,
}

impl SelectiveRowMappingAudit {
    pub(super) fn new(
        prefix_rows: Range<usize>,
        arms: Vec<SelectiveArmRowMappingAudit>,
        ring_padding_rows: Range<usize>,
        emitted_runs: Vec<SelectiveEmittedRowRunAudit>,
        rewrites: Vec<SelectiveRewriteAudit>,
        total_rows: usize,
    ) -> Self {
        Self {
            prefix_rows,
            arms,
            ring_padding_rows,
            emitted_runs,
            rewrites,
            total_rows,
        }
    }

    pub fn prefix_rows(&self) -> Range<usize> {
        self.prefix_rows.clone()
    }

    pub fn arms(&self) -> &[SelectiveArmRowMappingAudit] {
        &self.arms
    }

    pub fn ring_padding_rows(&self) -> Range<usize> {
        self.ring_padding_rows.clone()
    }

    pub fn emitted_runs(&self) -> &[SelectiveEmittedRowRunAudit] {
        &self.emitted_runs
    }

    pub fn rewrites(&self) -> &[SelectiveRewriteAudit] {
        &self.rewrites
    }

    pub fn total_rows(&self) -> usize {
        self.total_rows
    }
}

/// Exact layout plus diagnostic width attribution from one prepared selective
/// compiler run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveCompilerAudit {
    layout: SelectiveLayoutAudit,
    width: SelectiveLowNormWidthAudit,
    rows: SelectiveRowMappingAudit,
    canonical_openings: Vec<Vec<SelectiveCanonicalOpeningAudit>>,
    source_arm_physical_stages: Vec<Vec<PhysicalStageRange>>,
}

impl SelectiveCompilerAudit {
    pub(super) fn new(
        layout: SelectiveLayoutAudit,
        width: SelectiveLowNormWidthAudit,
        rows: SelectiveRowMappingAudit,
        canonical_openings: Vec<Vec<SelectiveCanonicalOpeningAudit>>,
        source_arm_physical_stages: Vec<Vec<PhysicalStageRange>>,
    ) -> Self {
        Self {
            layout,
            width,
            rows,
            canonical_openings,
            source_arm_physical_stages,
        }
    }

    pub fn layout(&self) -> &SelectiveLayoutAudit {
        &self.layout
    }

    pub fn width(&self) -> &SelectiveLowNormWidthAudit {
        &self.width
    }

    pub fn rows(&self) -> &SelectiveRowMappingAudit {
        &self.rows
    }

    pub fn canonical_openings(&self) -> &[Vec<SelectiveCanonicalOpeningAudit>] {
        &self.canonical_openings
    }

    /// Sequential source-row intervals copied from each field-R1CS arm.
    ///
    /// Row coordinates were validated when the source relation was lowered.
    /// Stage labels remain caller assertions: a consumer claiming a complete
    /// protocol ledger must separately require its expected root and path set.
    /// An empty per-arm stage list means physical provenance was unavailable.
    pub fn source_arm_physical_stages(&self) -> &[Vec<PhysicalStageRange>] {
        &self.source_arm_physical_stages
    }

    pub(super) fn into_width(self) -> SelectiveLowNormWidthAudit {
        self.width
    }
}

pub(super) fn row_family_width_audits(
    arm: &super::SparseR1cs,
    widths: &[usize],
    branch_start: usize,
    balanced_width: usize,
    binary_width: usize,
) -> Vec<SelectiveFamilyWidthAudit> {
    let mut families = Vec::<(&'static str, Vec<(usize, usize)>)>::new();
    for family in arm.row_family_ranges() {
        if let Some((_, ranges)) = families.iter_mut().find(|(name, _)| *name == family.name) {
            ranges.push((family.row_start, family.row_end));
        } else {
            families.push((family.name, vec![(family.row_start, family.row_end)]));
        }
    }
    assert!(
        families.len() <= u64::BITS as usize,
        "too many row families for width audit"
    );
    let mut family_masks = vec![0u64; arm.m - branch_start];
    for matrix in [&arm.a, &arm.b, &arm.c] {
        for_each_explicit_term(matrix, |row, column| {
            if column < branch_start || widths[column] == 0 {
                return;
            }
            for (family_index, (_, ranges)) in families.iter().enumerate() {
                if ranges
                    .iter()
                    .any(|&(start, end)| (start..end).contains(&row))
                {
                    family_masks[column - branch_start] |= 1 << family_index;
                }
            }
        });
    }
    families
        .iter()
        .enumerate()
        .map(|(family_index, (name, ranges))| {
            let mut unit_columns = 0;
            let mut balanced_columns = 0;
            let mut binary_columns = 0;
            let mut coordinates_before_aliases = 0;
            for (offset, mask) in family_masks.iter().enumerate() {
                if mask & (1 << family_index) == 0 {
                    continue;
                }
                let width = widths[branch_start + offset];
                coordinates_before_aliases += width;
                unit_columns += usize::from(width == 1);
                balanced_columns += usize::from(width == balanced_width);
                binary_columns += usize::from(width == binary_width);
            }
            let mut poseidon2_permutations = 0;
            let mut poseidon2_coordinates = 0;
            for trace in arm.poseidon2_traces() {
                if !ranges
                    .iter()
                    .any(|&(start, end)| trace.row_start >= start && trace.row_end <= end)
                {
                    continue;
                }
                poseidon2_permutations += 1;
                let mut columns = trace
                    .sboxes
                    .iter()
                    .map(|sbox| sbox.output_col)
                    .collect::<Vec<_>>();
                columns.extend(trace.output_cols);
                columns.sort_unstable();
                columns.dedup();
                poseidon2_coordinates += columns
                    .into_iter()
                    .map(|column| widths[column])
                    .sum::<usize>();
            }
            SelectiveFamilyWidthAudit {
                name: *name,
                inclusive_rows: covered_rows(ranges),
                unit_columns,
                balanced_columns,
                binary_columns,
                coordinates_before_aliases,
                poseidon2_permutations,
                poseidon2_coordinates,
            }
        })
        .collect()
}

pub(super) fn retained_trace_widths(arm: &super::SparseR1cs, widths: &[usize]) -> SelectiveTraceWidthAudit {
    let mut poseidon2 = vec![false; arm.m];
    for trace in arm.poseidon2_traces() {
        for sbox in &trace.sboxes {
            poseidon2[sbox.output_col] = true;
        }
        for &column in &trace.output_cols {
            poseidon2[column] = true;
        }
    }
    let mut polynomial_evaluation = vec![false; arm.m];
    for trace in arm.polynomial_evaluation_traces() {
        for &column in &trace.output_cols {
            polynomial_evaluation[column] = true;
        }
    }
    let mut product_sum = vec![false; arm.m];
    for trace in arm.product_sum_batch_traces() {
        for &column in &trace.retained_columns {
            product_sum[column] = true;
        }
    }
    let census = |present: Vec<bool>| {
        present
            .into_iter()
            .enumerate()
            .filter(|(column, present)| *present && widths[*column] != 0)
            .fold((0usize, 0usize), |(columns, coordinates), (column, _)| {
                (columns + 1, coordinates + widths[column])
            })
    };
    let (poseidon2_columns, poseidon2_coordinates) = census(poseidon2);
    let (polynomial_evaluation_columns, polynomial_evaluation_coordinates) = census(polynomial_evaluation);
    let (product_sum_columns, product_sum_coordinates) = census(product_sum);
    let (product_sum_internal_columns, product_sum_internal_coordinates) = internal_product_sum_widths(arm, widths);
    SelectiveTraceWidthAudit {
        poseidon2_permutations: arm.poseidon2_traces().len(),
        poseidon2_columns,
        poseidon2_coordinates,
        polynomial_evaluation_columns,
        polynomial_evaluation_coordinates,
        product_sum_columns,
        product_sum_coordinates,
        product_sum_internal_columns,
        product_sum_internal_coordinates,
    }
}

fn internal_product_sum_widths(arm: &super::SparseR1cs, widths: &[usize]) -> (usize, usize) {
    let mut product_rows = vec![false; arm.n];
    let mut outputs = vec![false; arm.m];
    for trace in arm.product_sum_batch_traces() {
        product_rows[trace.row_start..trace.row_end].fill(true);
        for &column in &trace.retained_columns {
            outputs[column] = true;
        }
    }

    let mut external = vec![false; arm.m];
    for matrix in [&arm.a, &arm.b, &arm.c] {
        for_each_explicit_term(matrix, |row, column| {
            if !product_rows[row] {
                external[column] = true;
            }
        });
        if let CcsMatrix::CscWithSeededPhi81 { blocks, .. } = matrix {
            for block in blocks {
                for &start in block.word_starts() {
                    external[start..start + block.word_width()].fill(true);
                }
            }
        }
    }

    outputs
        .into_iter()
        .enumerate()
        .filter(|(column, output)| *output && !external[*column] && widths[*column] != 0)
        .fold((0, 0), |(columns, coordinates), (column, _)| {
            (columns + 1, coordinates + widths[column])
        })
}

fn for_each_explicit_term(matrix: &CcsMatrix<neo_math::F>, mut visit: impl FnMut(usize, usize)) {
    let mut visit_csc = |csc: &CscMat<neo_math::F>| {
        for column in 0..csc.ncols {
            for index in csc.column_range(column) {
                visit(csc.row_index(index), column);
            }
        }
    };
    match matrix {
        CcsMatrix::Identity { n } => {
            for row in 0..*n {
                visit(row, row);
            }
        }
        CcsMatrix::Csc(csc) => visit_csc(csc),
        CcsMatrix::CscWithSeededPhi81 { csc, .. } => visit_csc(csc),
    }
}
