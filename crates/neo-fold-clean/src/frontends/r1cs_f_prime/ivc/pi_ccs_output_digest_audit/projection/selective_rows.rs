//! Exact interval join from the selected projection source rows to selective lowering.
//!
//! Owns: the 14 stable source-stage leaves, their exclusive source-row
//! dispositions, rewrite IDs, and exact emitted-row intervals in the steady
//! recursive arm.
//!
//! Does not own: rewrite semantics, emitted matrix coefficients, column
//! ownership, selector truth, transcript authority, or row-removal authority.
//!
//! Emits constraints: no.
//!
//! | Branch | Source rows | Selective rows |
//! |---|---:|---:|
//! | `projection_shared` | 1,892 | 438 |
//! | `identities.y_zcol.limb0` | 1,916 | 408 |
//! | `identities.y_zcol.limb1` | 1,916 | 408 |
//! | selected cross-branch certificate | 5,724 | 1,254 |

use std::collections::{BTreeSet, HashSet};
use std::ops::Range;

use crate::engine::r1cs_circuit::PhysicalStageRange;
use crate::frontends::r1cs_f_prime::{
    SelectiveEmittedRowFamily, SelectiveRewriteAudit, SelectiveRewriteId, SelectiveRewriteKind,
    SelectiveRowMappingAudit, SelectiveSourceRowDisposition,
};
use crate::paper::reductions::pi_rlc_circuit::stage;

use super::super::invalid;
use super::{PiRlcYZcolProjectionIdentityAudit, R1csIvcError};

const SOURCE_ROW_COUNT: usize = 5_724;
const SELECTIVE_ROW_COUNT: usize = 1_254;
const STAGE_LEAF_COUNT: usize = 14;
const FRAGMENT_COUNT: usize = 139;

/// Exclusive compiler treatment of one selected source fragment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PiRlcYZcolProjectionLoweringDisposition {
    Retained,
    Rewrite {
        id: SelectiveRewriteId,
        kind: SelectiveRewriteKind,
    },
}

/// One retained interval or one complete selective rewrite.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiRlcYZcolProjectionLoweredFragmentAudit {
    source_rows: Vec<Range<usize>>,
    emitted_rows: Range<usize>,
    disposition: PiRlcYZcolProjectionLoweringDisposition,
}

impl PiRlcYZcolProjectionLoweredFragmentAudit {
    pub fn source_rows(&self) -> &[Range<usize>] {
        &self.source_rows
    }

    pub fn emitted_rows(&self) -> Range<usize> {
        self.emitted_rows.clone()
    }

    pub fn disposition(&self) -> PiRlcYZcolProjectionLoweringDisposition {
        self.disposition
    }

    pub fn source_row_count(&self) -> usize {
        self.source_rows.iter().map(Range::len).sum()
    }

    pub fn emitted_row_count(&self) -> usize {
        self.emitted_rows.len()
    }
}

/// One real Rust stage leaf and every selective fragment derived from it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiRlcYZcolProjectionLeafRowMappingAudit {
    stage_path: &'static str,
    source_rows: Vec<Range<usize>>,
    fragments: Vec<PiRlcYZcolProjectionLoweredFragmentAudit>,
}

impl PiRlcYZcolProjectionLeafRowMappingAudit {
    pub fn stage_path(&self) -> &'static str {
        self.stage_path
    }

    pub fn source_rows(&self) -> &[Range<usize>] {
        &self.source_rows
    }

    pub fn fragments(&self) -> &[PiRlcYZcolProjectionLoweredFragmentAudit] {
        &self.fragments
    }

    pub fn source_row_count(&self) -> usize {
        self.source_rows.iter().map(Range::len).sum()
    }

    pub fn emitted_row_count(&self) -> usize {
        self.fragments
            .iter()
            .map(PiRlcYZcolProjectionLoweredFragmentAudit::emitted_row_count)
            .sum()
    }
}

/// Compact source-to-selective ownership certificate for the selected bundle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PiRlcYZcolProjectionRowMappingAudit {
    source_arm_row_count: usize,
    final_relation_row_count: usize,
    steady_arm_rows: Range<usize>,
    leaves: Vec<PiRlcYZcolProjectionLeafRowMappingAudit>,
}

impl PiRlcYZcolProjectionRowMappingAudit {
    pub fn source_arm_row_count(&self) -> usize {
        self.source_arm_row_count
    }

    pub fn final_relation_row_count(&self) -> usize {
        self.final_relation_row_count
    }

    pub fn steady_arm_rows(&self) -> Range<usize> {
        self.steady_arm_rows.clone()
    }

    pub fn leaves(&self) -> &[PiRlcYZcolProjectionLeafRowMappingAudit] {
        &self.leaves
    }

    pub fn source_row_count(&self) -> usize {
        self.leaves
            .iter()
            .map(PiRlcYZcolProjectionLeafRowMappingAudit::source_row_count)
            .sum()
    }

    pub fn emitted_row_count(&self) -> usize {
        self.leaves
            .iter()
            .map(PiRlcYZcolProjectionLeafRowMappingAudit::emitted_row_count)
            .sum()
    }
}

struct LeafSpec {
    stage_path: &'static str,
    source_rows: Vec<Range<usize>>,
    expected_emitted_rows: usize,
}

fn leaf_specs(identity: &PiRlcYZcolProjectionIdentityAudit) -> Vec<LeafSpec> {
    let shared = identity.shared();
    let mut leaves = vec![
        LeafSpec {
            stage_path: stage::PROJECTION_SHARED_BETA_LADDER,
            source_rows: vec![shared.beta_ladder_rows()],
            expected_emitted_rows: 108,
        },
        LeafSpec {
            stage_path: stage::PROJECTION_SHARED_RHO_EVALUATIONS,
            source_rows: shared
                .rho_evaluations()
                .iter()
                .map(|owner| owner.rows())
                .collect(),
            expected_emitted_rows: 330,
        },
    ];
    for limb in 0..2 {
        let owner = identity.limb(limb);
        let paths = match limb {
            0 => [
                stage::IDENTITIES_Y_ZCOL_EVALUATIONS_INPUTS_LIMB0,
                stage::IDENTITIES_Y_ZCOL_K_PRODUCTS_RHO_TIMES_INPUT_LIMB0,
                stage::IDENTITIES_Y_ZCOL_EVALUATIONS_OUTPUT_LIMB0,
                stage::IDENTITIES_Y_ZCOL_EVALUATIONS_QUOTIENT_LIMB0,
                stage::IDENTITIES_Y_ZCOL_K_PRODUCTS_QUOTIENT_TIMES_PHI_LIMB0,
                stage::IDENTITIES_Y_ZCOL_FINAL_LIMB_CHECKS_LIMB0,
            ],
            1 => [
                stage::IDENTITIES_Y_ZCOL_EVALUATIONS_INPUTS_LIMB1,
                stage::IDENTITIES_Y_ZCOL_K_PRODUCTS_RHO_TIMES_INPUT_LIMB1,
                stage::IDENTITIES_Y_ZCOL_EVALUATIONS_OUTPUT_LIMB1,
                stage::IDENTITIES_Y_ZCOL_EVALUATIONS_QUOTIENT_LIMB1,
                stage::IDENTITIES_Y_ZCOL_K_PRODUCTS_QUOTIENT_TIMES_PHI_LIMB1,
                stage::IDENTITIES_Y_ZCOL_FINAL_LIMB_CHECKS_LIMB1,
            ],
            _ => unreachable!("two coefficient limbs"),
        };
        leaves.extend([
            LeafSpec {
                stage_path: paths[0],
                source_rows: owner
                    .input_evaluations()
                    .iter()
                    .map(|evaluation| evaluation.rows())
                    .collect(),
                expected_emitted_rows: 330,
            },
            LeafSpec {
                stage_path: paths[1],
                source_rows: owner
                    .rho_products()
                    .iter()
                    .map(|product| product.rows())
                    .collect(),
                expected_emitted_rows: 30,
            },
            LeafSpec {
                stage_path: paths[2],
                source_rows: vec![owner.output_evaluation_rows()],
                expected_emitted_rows: 22,
            },
            LeafSpec {
                stage_path: paths[3],
                source_rows: vec![owner.quotient_evaluation_rows()],
                expected_emitted_rows: 22,
            },
            LeafSpec {
                stage_path: paths[4],
                source_rows: vec![owner.quotient_phi_rows()],
                expected_emitted_rows: 2,
            },
            LeafSpec {
                stage_path: paths[5],
                source_rows: vec![owner.final_rows()],
                expected_emitted_rows: 2,
            },
        ]);
    }
    leaves
}

fn rewrite_disposition(
    disposition: SelectiveSourceRowDisposition,
) -> Result<Option<(SelectiveRewriteId, SelectiveRewriteKind)>, R1csIvcError> {
    let selected = match disposition {
        SelectiveSourceRowDisposition::Retained => return Ok(None),
        SelectiveSourceRowDisposition::PolynomialEvaluation(id) => (id, SelectiveRewriteKind::PolynomialEvaluation),
        SelectiveSourceRowDisposition::ProductSum(id) => (id, SelectiveRewriteKind::ProductSum),
        SelectiveSourceRowDisposition::LinearDefinition(id) => (id, SelectiveRewriteKind::LinearDefinition),
        other => {
            return Err(invalid(format!(
                "selected PiRLC projection source row uses unsupported selective disposition {other:?}"
            )))
        }
    };
    Ok(Some(selected))
}

fn expected_family(kind: SelectiveRewriteKind) -> Option<SelectiveEmittedRowFamily> {
    match kind {
        SelectiveRewriteKind::PolynomialEvaluation => Some(SelectiveEmittedRowFamily::PolynomialEvaluation),
        SelectiveRewriteKind::ProductSum => Some(SelectiveEmittedRowFamily::ProductSum),
        SelectiveRewriteKind::LinearDefinition => None,
        _ => None,
    }
}

fn normalize_ranges(mut ranges: Vec<Range<usize>>) -> Result<Vec<Range<usize>>, R1csIvcError> {
    ranges.sort_by_key(|range| (range.start, range.end));
    let mut normalized = Vec::<Range<usize>>::new();
    for range in ranges {
        if range.is_empty() {
            return Err(invalid("selected PiRLC projection contains an empty source interval"));
        }
        match normalized.last_mut() {
            Some(previous) if range.start < previous.end => {
                return Err(invalid("selected PiRLC projection source intervals overlap"))
            }
            Some(previous) if range.start == previous.end => previous.end = range.end,
            _ => normalized.push(range),
        }
    }
    Ok(normalized)
}

fn range_is_covered(range: &Range<usize>, owners: &[Range<usize>]) -> bool {
    owners
        .iter()
        .any(|owner| owner.start <= range.start && range.end <= owner.end)
}

fn validate_rewrite(
    rewrite: &SelectiveRewriteAudit,
    id: SelectiveRewriteId,
    kind: SelectiveRewriteKind,
    arm_index: usize,
    stage_occurrences: &HashSet<usize>,
    emitted_runs: &[crate::frontends::r1cs_f_prime::SelectiveEmittedRowRunAudit],
) -> Result<(), R1csIvcError> {
    if rewrite.id() != id
        || rewrite.kind() != kind
        || rewrite.arm() != arm_index
        || !rewrite
            .source_stage_occurrence()
            .is_some_and(|occurrence| stage_occurrences.contains(&occurrence))
    {
        return Err(invalid(format!(
            "selective rewrite {} does not match its PiRLC projection source owner",
            id.index()
        )));
    }
    let matches = emitted_runs
        .iter()
        .filter(|run| run.rewrite_id() == Some(id))
        .collect::<Vec<_>>();
    match expected_family(kind) {
        None => {
            if !rewrite.emitted_rows().is_empty() || !matches.is_empty() {
                return Err(invalid(
                    "linear projection definition unexpectedly emits selective rows",
                ));
            }
        }
        Some(family) => {
            let [run] = matches.as_slice() else {
                return Err(invalid(format!(
                    "selective rewrite {} has {} emitted owners",
                    id.index(),
                    matches.len()
                )));
            };
            if run.family() != family
                || run.arm() != Some(arm_index)
                || run.emitted_rows() != rewrite.emitted_rows()
                || run.source_stage_occurrence() != rewrite.source_stage_occurrence()
            {
                return Err(invalid(format!(
                    "selective rewrite {} emitted interval disagrees with its row owner",
                    id.index()
                )));
            }
        }
    }
    Ok(())
}

fn recover_leaf(
    spec: LeafSpec,
    rows: &SelectiveRowMappingAudit,
    stages: &[PhysicalStageRange],
    arm_index: usize,
) -> Result<PiRlcYZcolProjectionLeafRowMappingAudit, R1csIvcError> {
    let stage_occurrences = stages
        .iter()
        .enumerate()
        .filter(|(_, stage)| stage.path() == spec.stage_path && !stage.rows().is_empty())
        .map(|(index, stage)| (index, stage.rows()))
        .collect::<Vec<_>>();
    let physical_ranges = stage_occurrences
        .iter()
        .map(|(_, range)| range.clone())
        .collect::<Vec<_>>();
    if normalize_ranges(physical_ranges.clone())? != normalize_ranges(spec.source_rows.clone())? {
        return Err(invalid(format!(
            "stage `{}` source intervals {physical_ranges:?} differ from the exact projection trace {:?}",
            spec.stage_path, spec.source_rows
        )));
    }
    let occurrence_set = stage_occurrences
        .iter()
        .map(|(index, _)| *index)
        .collect::<HashSet<_>>();
    let arm = rows
        .arms()
        .get(arm_index)
        .ok_or_else(|| invalid(format!("selective row ledger omits recursive arm {arm_index}")))?;
    let selected_runs = arm
        .source_runs()
        .iter()
        .filter(|run| {
            run.stage_occurrence()
                .is_some_and(|index| occurrence_set.contains(&index))
        })
        .collect::<Vec<_>>();
    if selected_runs.is_empty() {
        return Err(invalid(format!(
            "stage `{}` has no selective source disposition",
            spec.stage_path
        )));
    }

    let mut fragments = Vec::new();
    let mut rewrite_ids = BTreeSet::new();
    for run in selected_runs {
        let source_rows = run.source_rows();
        if !range_is_covered(&source_rows, &spec.source_rows) {
            return Err(invalid(format!(
                "stage `{}` selective source run {source_rows:?} escapes its trace ranges",
                spec.stage_path
            )));
        }
        match rewrite_disposition(run.disposition())? {
            None => {
                let Some(emitted_start) = run.emitted_start() else {
                    return Err(invalid("retained projection source run omits its emitted start"));
                };
                let emitted_rows = emitted_start..emitted_start + source_rows.len();
                let matches = rows
                    .emitted_runs()
                    .iter()
                    .filter(|owner| {
                        owner.family() == SelectiveEmittedRowFamily::Retained
                            && owner.arm() == Some(arm_index)
                            && owner.emitted_rows() == emitted_rows
                            && owner.source_stage_occurrence() == run.stage_occurrence()
                    })
                    .count();
                if matches != 1 {
                    return Err(invalid("retained projection source run lacks one exact emitted owner"));
                }
                fragments.push(PiRlcYZcolProjectionLoweredFragmentAudit {
                    source_rows: vec![source_rows],
                    emitted_rows,
                    disposition: PiRlcYZcolProjectionLoweringDisposition::Retained,
                });
            }
            Some((id, _)) => {
                rewrite_ids.insert(id.index());
            }
        }
    }

    for index in rewrite_ids {
        let rewrite = rows
            .rewrites()
            .get(index)
            .ok_or_else(|| invalid(format!("selective row ledger omits projection rewrite {index}")))?;
        let id = rewrite.id();
        let kind = rewrite.kind();
        if !matches!(
            kind,
            SelectiveRewriteKind::PolynomialEvaluation
                | SelectiveRewriteKind::ProductSum
                | SelectiveRewriteKind::LinearDefinition
        ) {
            return Err(invalid(format!(
                "projection rewrite {index} has unsupported kind {kind:?}"
            )));
        }
        if rewrite
            .source_rows()
            .iter()
            .any(|range| !range_is_covered(range, &spec.source_rows))
        {
            return Err(invalid(format!(
                "projection rewrite {index} crosses stage `{}`",
                spec.stage_path
            )));
        }
        validate_rewrite(rewrite, id, kind, arm_index, &occurrence_set, rows.emitted_runs())?;
        fragments.push(PiRlcYZcolProjectionLoweredFragmentAudit {
            source_rows: rewrite.source_rows().to_vec(),
            emitted_rows: rewrite.emitted_rows(),
            disposition: PiRlcYZcolProjectionLoweringDisposition::Rewrite { id, kind },
        });
    }
    fragments.sort_by_key(|fragment| {
        fragment
            .source_rows
            .first()
            .map_or(usize::MAX, |range| range.start)
    });
    let fragment_ranges = fragments
        .iter()
        .flat_map(|fragment| fragment.source_rows.iter().cloned())
        .collect::<Vec<_>>();
    if normalize_ranges(fragment_ranges)? != normalize_ranges(spec.source_rows.clone())? {
        return Err(invalid(format!(
            "stage `{}` selective fragments do not cover its exact source rows",
            spec.stage_path
        )));
    }
    let emitted_count = fragments
        .iter()
        .map(PiRlcYZcolProjectionLoweredFragmentAudit::emitted_row_count)
        .sum::<usize>();
    if emitted_count != spec.expected_emitted_rows {
        return Err(invalid(format!(
            "stage `{}` lowers to {emitted_count} rows, expected {}",
            spec.stage_path, spec.expected_emitted_rows
        )));
    }
    Ok(PiRlcYZcolProjectionLeafRowMappingAudit {
        stage_path: spec.stage_path,
        source_rows: spec.source_rows,
        fragments,
    })
}

pub(super) fn recover(
    identity: &PiRlcYZcolProjectionIdentityAudit,
    rows: &SelectiveRowMappingAudit,
    stages: &[PhysicalStageRange],
    arm_index: usize,
) -> Result<PiRlcYZcolProjectionRowMappingAudit, R1csIvcError> {
    let specs = leaf_specs(identity);
    if specs.len() != STAGE_LEAF_COUNT
        || specs
            .iter()
            .map(|leaf| leaf.stage_path)
            .collect::<HashSet<_>>()
            .len()
            != STAGE_LEAF_COUNT
    {
        return Err(invalid(
            "PiRLC projection stage vocabulary is not exactly 14 unique leaves",
        ));
    }
    let leaves = specs
        .into_iter()
        .map(|spec| recover_leaf(spec, rows, stages, arm_index))
        .collect::<Result<Vec<_>, _>>()?;
    let source_row_count = leaves
        .iter()
        .map(PiRlcYZcolProjectionLeafRowMappingAudit::source_row_count)
        .sum::<usize>();
    let emitted_row_count = leaves
        .iter()
        .map(PiRlcYZcolProjectionLeafRowMappingAudit::emitted_row_count)
        .sum::<usize>();
    let fragment_count = leaves
        .iter()
        .map(|leaf| leaf.fragments.len())
        .sum::<usize>();
    if source_row_count != SOURCE_ROW_COUNT
        || emitted_row_count != SELECTIVE_ROW_COUNT
        || fragment_count != FRAGMENT_COUNT
    {
        return Err(invalid(format!(
            "PiRLC projection selective census is {source_row_count} source rows, {emitted_row_count} emitted rows, and {fragment_count} fragments; expected {SOURCE_ROW_COUNT}, {SELECTIVE_ROW_COUNT}, and {FRAGMENT_COUNT}"
        )));
    }
    let mut emitted = leaves
        .iter()
        .flat_map(|leaf| leaf.fragments.iter())
        .map(PiRlcYZcolProjectionLoweredFragmentAudit::emitted_rows)
        .filter(|range| !range.is_empty())
        .collect::<Vec<_>>();
    emitted.sort_by_key(|range| (range.start, range.end));
    if emitted.windows(2).any(|pair| pair[1].start < pair[0].end) {
        return Err(invalid("PiRLC projection selective emitted intervals overlap"));
    }
    let arm = rows
        .arms()
        .get(arm_index)
        .ok_or_else(|| invalid("selective row ledger omits the steady arm"))?;
    Ok(PiRlcYZcolProjectionRowMappingAudit {
        source_arm_row_count: identity
            .source_rows()
            .last()
            .map_or(0, |row| row.index() + 1)
            .max(stages.last().map_or(0, PhysicalStageRange::row_end)),
        final_relation_row_count: rows.total_rows(),
        steady_arm_rows: arm.emitted_rows(),
        leaves,
    })
}
