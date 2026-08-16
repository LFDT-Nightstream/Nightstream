//! Compact exact row ledger for both normalized PiRLC parity bodies.
//!
//! Owns affine compression and pointwise replay against the compiler ledger.
//! It does not own row semantics, port images, matrix actions, or assignments.

use crate::frontends::r1cs_f_prime::{
    SelectiveCompilerAudit, SelectiveEmittedRowFamily, SelectiveRewriteKind, SelectiveSourceRowDisposition,
};

use super::{
    production_pi_rlc_family_body_compiler_audit, NebulaFPrimePiRlcFamilyRelationError, PI_RLC_FAMILY_BODY_EVEN_ROWS,
    PI_RLC_FAMILY_BODY_ODD_ROWS,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NebulaFPrimePiRlcBodyFixedFamily {
    SelectorDomain,
    SharedDomain,
    ArmDomain,
    OneHot,
    PublicPadding,
    PrivatePadding,
    RingPadding,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NebulaFPrimePiRlcBodyRewriteKind {
    Poseidon2,
    ShiftedTernaryCanonical,
    LinearDefinition,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimePiRlcBodyFixedEmittedRun {
    start: usize,
    length: usize,
    family: NebulaFPrimePiRlcBodyFixedFamily,
    arm: Option<usize>,
}

impl NebulaFPrimePiRlcBodyFixedEmittedRun {
    pub const fn start(self) -> usize {
        self.start
    }

    pub const fn length(self) -> usize {
        self.length
    }

    pub const fn family(self) -> NebulaFPrimePiRlcBodyFixedFamily {
        self.family
    }

    pub const fn arm(self) -> Option<usize> {
        self.arm
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimePiRlcBodyRetainedRun {
    arm: usize,
    source_start: usize,
    length: usize,
    emitted_start: usize,
}

impl NebulaFPrimePiRlcBodyRetainedRun {
    pub const fn arm(self) -> usize {
        self.arm
    }

    pub const fn source_start(self) -> usize {
        self.source_start
    }

    pub const fn length(self) -> usize {
        self.length
    }

    pub const fn emitted_start(self) -> usize {
        self.emitted_start
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NebulaFPrimePiRlcBodyRewriteBatch {
    rewrite_start: usize,
    count: usize,
    rewrite_stride: usize,
    arm: usize,
    kind: NebulaFPrimePiRlcBodyRewriteKind,
    source_start: usize,
    source_stride: usize,
    source_width: usize,
    emitted_start: usize,
    emitted_stride: usize,
    emitted_width: usize,
}

impl NebulaFPrimePiRlcBodyRewriteBatch {
    pub const fn rewrite_start(self) -> usize {
        self.rewrite_start
    }

    pub const fn count(self) -> usize {
        self.count
    }

    pub const fn rewrite_stride(self) -> usize {
        self.rewrite_stride
    }

    pub const fn arm(self) -> usize {
        self.arm
    }

    pub const fn kind(self) -> NebulaFPrimePiRlcBodyRewriteKind {
        self.kind
    }

    pub const fn source_start(self) -> usize {
        self.source_start
    }

    pub const fn source_stride(self) -> usize {
        self.source_stride
    }

    pub const fn source_width(self) -> usize {
        self.source_width
    }

    pub const fn emitted_start(self) -> usize {
        self.emitted_start
    }

    pub const fn emitted_stride(self) -> usize {
        self.emitted_stride
    }

    pub const fn emitted_width(self) -> usize {
        self.emitted_width
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NebulaFPrimePiRlcFamilyBodyRowLedger {
    rows: usize,
    columns: usize,
    source_rows: [usize; 2],
    rewrite_count: usize,
    fixed_runs: Vec<NebulaFPrimePiRlcBodyFixedEmittedRun>,
    retained_runs: Vec<NebulaFPrimePiRlcBodyRetainedRun>,
    rewrite_batches: Vec<NebulaFPrimePiRlcBodyRewriteBatch>,
}

impl NebulaFPrimePiRlcFamilyBodyRowLedger {
    pub const fn rows(&self) -> usize {
        self.rows
    }

    pub const fn columns(&self) -> usize {
        self.columns
    }

    pub const fn source_rows(&self) -> [usize; 2] {
        self.source_rows
    }

    pub const fn rewrite_count(&self) -> usize {
        self.rewrite_count
    }

    pub fn fixed_runs(&self) -> &[NebulaFPrimePiRlcBodyFixedEmittedRun] {
        &self.fixed_runs
    }

    pub fn retained_runs(&self) -> &[NebulaFPrimePiRlcBodyRetainedRun] {
        &self.retained_runs
    }

    pub fn rewrite_batches(&self) -> &[NebulaFPrimePiRlcBodyRewriteBatch] {
        &self.rewrite_batches
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RewritePoint {
    rewrite: usize,
    arm: usize,
    kind: NebulaFPrimePiRlcBodyRewriteKind,
    source_start: usize,
    source_width: usize,
    emitted_start: usize,
    emitted_width: usize,
}

fn ledger_error(reason: &'static str) -> NebulaFPrimePiRlcFamilyRelationError {
    NebulaFPrimePiRlcFamilyRelationError::RowLedger(reason)
}

fn rewrite_kind(
    kind: SelectiveRewriteKind,
) -> Result<NebulaFPrimePiRlcBodyRewriteKind, NebulaFPrimePiRlcFamilyRelationError> {
    match kind {
        SelectiveRewriteKind::Poseidon2 => Ok(NebulaFPrimePiRlcBodyRewriteKind::Poseidon2),
        SelectiveRewriteKind::ShiftedTernaryCanonical => Ok(NebulaFPrimePiRlcBodyRewriteKind::ShiftedTernaryCanonical),
        SelectiveRewriteKind::LinearDefinition => Ok(NebulaFPrimePiRlcBodyRewriteKind::LinearDefinition),
        _ => Err(ledger_error("unsupported rewrite family")),
    }
}

fn emitted_family(kind: NebulaFPrimePiRlcBodyRewriteKind) -> Option<SelectiveEmittedRowFamily> {
    match kind {
        NebulaFPrimePiRlcBodyRewriteKind::Poseidon2 => Some(SelectiveEmittedRowFamily::Poseidon2),
        NebulaFPrimePiRlcBodyRewriteKind::ShiftedTernaryCanonical => {
            Some(SelectiveEmittedRowFamily::ShiftedTernaryCanonical)
        }
        NebulaFPrimePiRlcBodyRewriteKind::LinearDefinition => None,
    }
}

fn fixed_family(
    family: SelectiveEmittedRowFamily,
) -> Result<Option<NebulaFPrimePiRlcBodyFixedFamily>, NebulaFPrimePiRlcFamilyRelationError> {
    Ok(match family {
        SelectiveEmittedRowFamily::SelectorDomain => Some(NebulaFPrimePiRlcBodyFixedFamily::SelectorDomain),
        SelectiveEmittedRowFamily::SharedDomain => Some(NebulaFPrimePiRlcBodyFixedFamily::SharedDomain),
        SelectiveEmittedRowFamily::ArmDomain => Some(NebulaFPrimePiRlcBodyFixedFamily::ArmDomain),
        SelectiveEmittedRowFamily::OneHot => Some(NebulaFPrimePiRlcBodyFixedFamily::OneHot),
        SelectiveEmittedRowFamily::PublicPadding => Some(NebulaFPrimePiRlcBodyFixedFamily::PublicPadding),
        SelectiveEmittedRowFamily::PrivatePadding => Some(NebulaFPrimePiRlcBodyFixedFamily::PrivatePadding),
        SelectiveEmittedRowFamily::RingPadding => Some(NebulaFPrimePiRlcBodyFixedFamily::RingPadding),
        SelectiveEmittedRowFamily::Retained
        | SelectiveEmittedRowFamily::Poseidon2
        | SelectiveEmittedRowFamily::ShiftedTernaryCanonical => None,
        _ => return Err(ledger_error("unsupported emitted row family")),
    })
}

fn rewrite_points(audit: &SelectiveCompilerAudit) -> Result<Vec<RewritePoint>, NebulaFPrimePiRlcFamilyRelationError> {
    audit
        .rows()
        .rewrites()
        .iter()
        .enumerate()
        .map(|(index, rewrite)| {
            if rewrite.id().index() != index || rewrite.source_rows().len() != 1 {
                return Err(ledger_error("rewrite geometry is not a single ordered source interval"));
            }
            let source = &rewrite.source_rows()[0];
            let emitted = rewrite.emitted_rows();
            Ok(RewritePoint {
                rewrite: index,
                arm: rewrite.arm(),
                kind: rewrite_kind(rewrite.kind())?,
                source_start: source.start,
                source_width: source.len(),
                emitted_start: emitted.start,
                emitted_width: emitted.len(),
            })
        })
        .collect()
}

fn compress_points(points: &[RewritePoint]) -> Vec<NebulaFPrimePiRlcBodyRewriteBatch> {
    let mut batches = Vec::new();
    let mut cursor = 0;
    while cursor < points.len() {
        let first = points[cursor];
        let compatible = |point: RewritePoint| {
            point.arm == first.arm
                && point.kind == first.kind
                && point.source_width == first.source_width
                && point.emitted_width == first.emitted_width
        };
        let (rewrite_stride, source_stride, emitted_stride, mut count) = if let Some(second) = points
            .get(cursor + 1)
            .copied()
            .filter(|&point| compatible(point))
            .and_then(|second| {
                Some((
                    second.rewrite.checked_sub(first.rewrite)?,
                    second.source_start.checked_sub(first.source_start)?,
                    second.emitted_start.checked_sub(first.emitted_start)?,
                    2,
                ))
            }) {
            second
        } else {
            (1, 0, 0, 1)
        };
        while cursor + count < points.len() {
            let next = points[cursor + count];
            if !compatible(next)
                || next.rewrite != first.rewrite + rewrite_stride * count
                || next.source_start != first.source_start + source_stride * count
                || next.emitted_start != first.emitted_start + emitted_stride * count
            {
                break;
            }
            count += 1;
        }
        batches.push(NebulaFPrimePiRlcBodyRewriteBatch {
            rewrite_start: first.rewrite,
            count,
            rewrite_stride,
            arm: first.arm,
            kind: first.kind,
            source_start: first.source_start,
            source_stride,
            source_width: first.source_width,
            emitted_start: first.emitted_start,
            emitted_stride,
            emitted_width: first.emitted_width,
        });
        cursor += count;
    }
    batches
}

fn expanded_points(batches: &[NebulaFPrimePiRlcBodyRewriteBatch]) -> Vec<RewritePoint> {
    batches
        .iter()
        .flat_map(|batch| {
            (0..batch.count).map(|index| RewritePoint {
                rewrite: batch.rewrite_start + batch.rewrite_stride * index,
                arm: batch.arm,
                kind: batch.kind,
                source_start: batch.source_start + batch.source_stride * index,
                source_width: batch.source_width,
                emitted_start: batch.emitted_start + batch.emitted_stride * index,
                emitted_width: batch.emitted_width,
            })
        })
        .collect()
}

fn mark_range(owners: &mut [bool], start: usize, length: usize) -> bool {
    let Some(slice) = owners.get_mut(start..start.saturating_add(length)) else {
        return false;
    };
    if slice.iter().any(|&owned| owned) {
        return false;
    }
    slice.fill(true);
    true
}

fn build_ledger(
    audit: &SelectiveCompilerAudit,
) -> Result<NebulaFPrimePiRlcFamilyBodyRowLedger, NebulaFPrimePiRlcFamilyRelationError> {
    let rows = audit.rows();
    if rows.arms().len() != 2 {
        return Err(ledger_error("compiler ledger does not have two parity arms"));
    }
    let source_rows = [PI_RLC_FAMILY_BODY_EVEN_ROWS, PI_RLC_FAMILY_BODY_ODD_ROWS];
    let points = rewrite_points(audit)?;
    let rewrite_batches = compress_points(&points);
    if expanded_points(&rewrite_batches) != points {
        return Err(ledger_error("affine rewrite batches do not replay the exact ledger"));
    }

    let mut retained_runs = Vec::new();
    for (arm_index, arm) in rows.arms().iter().enumerate() {
        for run in arm.source_runs() {
            match run.disposition() {
                SelectiveSourceRowDisposition::Retained => {
                    let emitted_start = run
                        .emitted_start()
                        .ok_or_else(|| ledger_error("retained source run has no emitted start"))?;
                    retained_runs.push(NebulaFPrimePiRlcBodyRetainedRun {
                        arm: arm_index,
                        source_start: run.source_rows().start,
                        length: run.source_rows().len(),
                        emitted_start,
                    });
                }
                SelectiveSourceRowDisposition::Poseidon2(_)
                | SelectiveSourceRowDisposition::ShiftedTernaryCanonical(_)
                | SelectiveSourceRowDisposition::LinearDefinition(_) => {}
                _ => return Err(ledger_error("unsupported source row disposition")),
            }
        }
    }

    let emitted_rewrites = rows
        .emitted_runs()
        .iter()
        .filter(|run| {
            matches!(
                run.family(),
                SelectiveEmittedRowFamily::Poseidon2 | SelectiveEmittedRowFamily::ShiftedTernaryCanonical
            )
        })
        .map(|run| {
            (
                run.emitted_rows().start,
                run.emitted_rows().len(),
                run.family(),
                run.arm(),
                run.rewrite_id().map(|id| id.index()),
            )
        })
        .collect::<Vec<_>>();
    let expected_emitted_rewrites = points
        .iter()
        .filter_map(|point| {
            Some((
                point.emitted_start,
                point.emitted_width,
                emitted_family(point.kind)?,
                Some(point.arm),
                Some(point.rewrite),
            ))
        })
        .collect::<Vec<_>>();
    if emitted_rewrites != expected_emitted_rewrites {
        return Err(ledger_error("rewrite batches differ from emitted rewrite owners"));
    }

    let emitted_retained = rows
        .emitted_runs()
        .iter()
        .filter(|run| run.family() == SelectiveEmittedRowFamily::Retained)
        .map(|run| (run.emitted_rows().start, run.emitted_rows().len(), run.arm()))
        .collect::<Vec<_>>();
    let expected_emitted_retained = retained_runs
        .iter()
        .map(|run| (run.emitted_start, run.length, Some(run.arm)))
        .collect::<Vec<_>>();
    if emitted_retained != expected_emitted_retained {
        return Err(ledger_error("retained source runs differ from emitted retained owners"));
    }

    let mut fixed_runs = Vec::new();
    for run in rows.emitted_runs() {
        if let Some(family) = fixed_family(run.family())? {
            fixed_runs.push(NebulaFPrimePiRlcBodyFixedEmittedRun {
                start: run.emitted_rows().start,
                length: run.emitted_rows().len(),
                family,
                arm: run.arm(),
            });
        }
    }

    let mut source_owners = source_rows.map(|length| vec![false; length]);
    for run in &retained_runs {
        if !mark_range(&mut source_owners[run.arm], run.source_start, run.length) {
            return Err(ledger_error("retained source run is duplicated or out of range"));
        }
    }
    let mut rewrite_owners = vec![false; points.len()];
    for point in &points {
        if point.rewrite >= rewrite_owners.len()
            || std::mem::replace(&mut rewrite_owners[point.rewrite], true)
            || !mark_range(
                source_owners
                    .get_mut(point.arm)
                    .ok_or_else(|| ledger_error("rewrite arm is out of range"))?,
                point.source_start,
                point.source_width,
            )
        {
            return Err(ledger_error(
                "rewrite source or identifier is duplicated or out of range",
            ));
        }
    }
    if source_owners.iter().flatten().any(|&owned| !owned) || rewrite_owners.iter().any(|&owned| !owned) {
        return Err(ledger_error("source rows or rewrite identifiers have a gap"));
    }

    let mut emitted_owners = vec![false; rows.total_rows()];
    for run in &fixed_runs {
        if !mark_range(&mut emitted_owners, run.start, run.length) {
            return Err(ledger_error("fixed emitted run is duplicated or out of range"));
        }
    }
    for run in &retained_runs {
        if !mark_range(&mut emitted_owners, run.emitted_start, run.length) {
            return Err(ledger_error("retained emitted run is duplicated or out of range"));
        }
    }
    for point in &points {
        if point.emitted_width > 0 && !mark_range(&mut emitted_owners, point.emitted_start, point.emitted_width) {
            return Err(ledger_error("rewrite emitted run is duplicated or out of range"));
        }
    }
    if emitted_owners.iter().any(|&owned| !owned) {
        return Err(ledger_error("emitted rows have a gap"));
    }

    Ok(NebulaFPrimePiRlcFamilyBodyRowLedger {
        rows: rows.total_rows(),
        columns: audit.layout().total_columns(),
        source_rows,
        rewrite_count: points.len(),
        fixed_runs,
        retained_runs,
        rewrite_batches,
    })
}

pub fn production_pi_rlc_family_body_row_ledger(
) -> Result<NebulaFPrimePiRlcFamilyBodyRowLedger, NebulaFPrimePiRlcFamilyRelationError> {
    let audit = production_pi_rlc_family_body_compiler_audit()?;
    build_ledger(&audit)
}
