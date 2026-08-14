//! Exact compiler audit for the fixed first-accepted selection rewrite.
//!
//! Owns: recognition of the 11-candidate, three-identity source trace and its
//! join to exact source-stage, rewrite, and emitted-row intervals.
//!
//! Does not own: sampler one-hotness, final low-norm row semantics, or
//! permission to remove a relation row.

use core::ops::Range;

use neo_math::{D, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::engine::r1cs_circuit::builder::{ProductSumBatchTrace, ProductSumIdentityTrace};

use super::selective_audit::{SelectiveRewriteId, SelectiveRewriteKind, SelectiveRowMappingAudit};
use super::{LowNormR1csError, SparseR1cs};

const STAGE_PATH: &str = "nifs.pi_rlc.challenge.sampler.selection.products";
const CANDIDATES: usize = 11;
const SOURCE_ROWS: usize = 3 * CANDIDATES + 3;
const EMITTED_ROWS: usize = 9;

/// One exact fixed-width selection rewrite from the production compiler.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveFirstAcceptedSelectionAudit {
    arm: usize,
    rewrite_id: SelectiveRewriteId,
    stage_occurrence: usize,
    source_rows: Range<usize>,
    emitted_rows: Range<usize>,
    position: usize,
    selectors: Vec<usize>,
    accepts: Vec<usize>,
    prefixes: Vec<usize>,
    symbols: Vec<usize>,
    accepted_products: Vec<usize>,
    prefix_products: Vec<usize>,
    symbol_products: Vec<usize>,
    output: usize,
}

impl SelectiveFirstAcceptedSelectionAudit {
    pub fn arm(&self) -> usize {
        self.arm
    }

    pub fn rewrite_id(&self) -> SelectiveRewriteId {
        self.rewrite_id
    }

    pub fn stage_occurrence(&self) -> usize {
        self.stage_occurrence
    }

    pub fn source_rows(&self) -> Range<usize> {
        self.source_rows.clone()
    }

    pub fn emitted_rows(&self) -> Range<usize> {
        self.emitted_rows.clone()
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn selectors(&self) -> &[usize] {
        &self.selectors
    }

    pub fn accepts(&self) -> &[usize] {
        &self.accepts
    }

    pub fn prefixes(&self) -> &[usize] {
        &self.prefixes
    }

    pub fn symbols(&self) -> &[usize] {
        &self.symbols
    }

    pub fn accepted_products(&self) -> &[usize] {
        &self.accepted_products
    }

    pub fn prefix_products(&self) -> &[usize] {
        &self.prefix_products
    }

    pub fn symbol_products(&self) -> &[usize] {
        &self.symbol_products
    }

    pub fn output(&self) -> usize {
        self.output
    }
}

pub(super) fn audit_first_accepted_selections(
    arms: &[SparseR1cs],
    rows: &SelectiveRowMappingAudit,
) -> Result<Vec<SelectiveFirstAcceptedSelectionAudit>, LowNormR1csError> {
    let mut audits = Vec::new();
    for (arm_index, arm) in arms.iter().enumerate() {
        let rewrites = rows
            .rewrites()
            .iter()
            .filter(|rewrite| rewrite.arm() == arm_index && rewrite.kind() == SelectiveRewriteKind::ProductSum)
            .collect::<Vec<_>>();
        if rewrites.len() != arm.product_sum_batch_traces().len() {
            return Err(trace_error("product-sum traces do not match the rewrite ledger"));
        }

        let first = audits.len();
        for (trace, rewrite) in arm.product_sum_batch_traces().iter().zip(rewrites) {
            if rewrite.source_rows() != [trace.row_start..trace.row_end] {
                return Err(trace_error("product-sum source rows differ from the rewrite ledger"));
            }
            let Some(stage_occurrence) = rewrite.source_stage_occurrence() else {
                continue;
            };
            let stage = arm
                .physical_stage_ranges()
                .get(stage_occurrence)
                .ok_or_else(|| trace_error("product-sum rewrite has an invalid source stage"))?;
            if stage.path() != STAGE_PATH {
                continue;
            }
            if stage.rows() != (trace.row_start..trace.row_end) {
                return Err(trace_error("selection stage rows differ from its product-sum trace"));
            }
            audits.push(parse_selection(
                arm_index,
                rewrite.id(),
                stage_occurrence,
                rewrite.emitted_rows(),
                trace,
            )?);
        }

        let arm_audits = &audits[first..];
        let stage_count = arm
            .physical_stage_ranges()
            .iter()
            .filter(|stage| stage.path() == STAGE_PATH)
            .count();
        if arm_audits.len() != stage_count {
            return Err(trace_error(
                "selection stages do not match recognized product-sum traces",
            ));
        }
        if arm_audits
            .iter()
            .enumerate()
            .any(|(index, audit)| audit.position != index % D)
        {
            return Err(trace_error(
                "selection positions do not form complete ordered output blocks",
            ));
        }
        if arm_audits.len() % D != 0 {
            return Err(trace_error("selection rewrite count is not a whole output block"));
        }
    }
    Ok(audits)
}

fn parse_selection(
    arm: usize,
    rewrite_id: SelectiveRewriteId,
    stage_occurrence: usize,
    emitted_rows: Range<usize>,
    trace: &ProductSumBatchTrace,
) -> Result<SelectiveFirstAcceptedSelectionAudit, LowNormR1csError> {
    if trace.row_end - trace.row_start != SOURCE_ROWS
        || emitted_rows.len() != EMITTED_ROWS
        || trace.identities.len() != 3
        || trace
            .identities
            .iter()
            .any(|identity| identity.factors.len() != CANDIDATES)
        || trace.allocated_columns.len() != 3 * CANDIDATES + 1
    {
        return Err(trace_error("selection product-sum trace has the wrong fixed geometry"));
    }

    let accepted = &trace.identities[0];
    let prefix = &trace.identities[1];
    let symbol = &trace.identities[2];
    let selectors = factor_columns(accepted, true)?;
    if factor_columns(prefix, true)? != selectors || factor_columns(symbol, true)? != selectors {
        return Err(trace_error("selection identities use different selector columns"));
    }
    let accepts = factor_columns(accepted, false)?;
    let prefixes = factor_columns(prefix, false)?;
    let symbols = factor_columns(symbol, false)?;

    if accepted.result.constant != F::ONE || !accepted.result.terms.is_empty() || !prefix.result.terms.is_empty() {
        return Err(trace_error("selection accepted or prefix result has the wrong form"));
    }
    let position = usize::try_from(prefix.result.constant.as_canonical_u64())
        .map_err(|_| trace_error("selection position does not fit usize"))?;
    if position >= D {
        return Err(trace_error("selection position is outside the fixed output width"));
    }
    let output = single_unit_column(&symbol.result)?;
    if trace.retained_columns != [output] || trace.allocated_columns[3 * CANDIDATES] != output {
        return Err(trace_error("selection output does not match retained trace authority"));
    }

    let products = &trace.allocated_columns[..3 * CANDIDATES];
    if products.windows(2).any(|pair| pair[1] != pair[0] + 1) {
        return Err(trace_error(
            "selection product columns are not one contiguous allocation",
        ));
    }
    Ok(SelectiveFirstAcceptedSelectionAudit {
        arm,
        rewrite_id,
        stage_occurrence,
        source_rows: trace.row_start..trace.row_end,
        emitted_rows,
        position,
        selectors,
        accepts,
        prefixes,
        symbols,
        accepted_products: products.iter().skip(1).step_by(3).copied().collect(),
        prefix_products: products.iter().skip(2).step_by(3).copied().collect(),
        symbol_products: products.iter().step_by(3).copied().collect(),
        output,
    })
}

fn factor_columns(identity: &ProductSumIdentityTrace, left: bool) -> Result<Vec<usize>, LowNormR1csError> {
    identity
        .factors
        .iter()
        .map(|factor| {
            if factor.coefficient != F::ONE {
                return Err(trace_error("selection product factor has a non-unit coefficient"));
            }
            single_unit_column(if left { &factor.left } else { &factor.right })
        })
        .collect()
}

fn single_unit_column(lc: &crate::engine::r1cs_circuit::Lc) -> Result<usize, LowNormR1csError> {
    match (lc.constant, lc.terms.as_slice()) {
        (constant, [(column, coefficient)]) if constant == F::ZERO && *coefficient == F::ONE => Ok(*column),
        _ => Err(trace_error("selection trace linear form is not one source column")),
    }
}

fn trace_error(message: &str) -> LowNormR1csError {
    LowNormR1csError::SelectiveTrace(message.to_owned())
}
