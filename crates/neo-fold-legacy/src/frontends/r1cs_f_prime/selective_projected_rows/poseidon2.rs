//! Exact projected provenance for compact Poseidon2 S-box rows.

use std::collections::BTreeMap;

use neo_math::F;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::Lc;

use super::super::super::selective_audit::{SelectiveEmittedRowFamily, SelectiveRewriteKind};
use super::super::super::SparseR1cs;
use super::super::emit::{append_field, append_lc, append_lc_scaled};
use super::super::terms::MatrixTerms;
use super::super::{trace_error, LowNormR1csError, C, GENERAL_SELECTOR, SBOX_INPUT, SELECTIVE_ARITY};
use super::{
    project_port, rewrite_geometry, source_terms, SelectiveProjectedPoseidon2OutputStep,
    SelectiveProjectedPoseidon2SboxStep, SelectiveProjectedRowArtifact, SelectiveProjectedSourceLinearCombination,
};

fn source_lc(constant: F, terms: &[(usize, F)]) -> SelectiveProjectedSourceLinearCombination {
    SelectiveProjectedSourceLinearCombination {
        constant,
        terms: source_terms(terms),
    }
}

fn verify_sbox_step(
    step: &SelectiveProjectedPoseidon2SboxStep,
    artifact: &SelectiveProjectedRowArtifact,
    layout: &super::super::SelectiveLayout,
    arm: usize,
) -> Result<(), LowNormR1csError> {
    let emitted_run = layout
        .compiler_audit
        .rows()
        .emitted_runs()
        .get(artifact.run_index)
        .ok_or_else(|| trace_error("projected Poseidon2 row has an invalid emitted-run owner"))?;
    if artifact.emitted_row != step.emitted_row
        || emitted_run.rewrite_id().map(|id| id.index()) != Some(step.rewrite_id)
        || emitted_run.family() != SelectiveEmittedRowFamily::Poseidon2
        || emitted_run.arm() != Some(arm)
    {
        return Err(trace_error(
            "projected Poseidon2 S-box step differs from its emitted-row owner",
        ));
    }

    let mut expected = (0..SELECTIVE_ARITY)
        .map(|_| MatrixTerms::new(false))
        .collect::<Vec<_>>();
    expected[GENERAL_SELECTOR].push((0, layout.selector_cols[arm], F::ONE));
    let input = Lc {
        terms: step
            .input
            .terms
            .iter()
            .map(|term| (term.column, term.coefficient))
            .collect(),
        constant: step.input.constant,
    };
    append_lc(
        &mut expected[SBOX_INPUT],
        0,
        &input,
        &layout.slots[arm],
        &layout.plans[arm].definitions,
    )?;
    let output = step
        .output
        .terms
        .first()
        .filter(|_| step.output.constant == F::ZERO && step.output.terms.len() == 1)
        .ok_or_else(|| trace_error("projected Poseidon2 S-box output is not one source column"))?;
    append_field(
        &mut expected[C],
        0,
        output.column,
        output.coefficient,
        &layout.slots[arm],
        &layout.plans[arm].definitions,
    )?;
    for (port, terms) in expected.iter().enumerate() {
        if project_port(terms, 0, artifact.columns)? != artifact.ports[port] {
            return Err(trace_error(
                "projected Poseidon2 S-box step does not reproduce its exact emitted row",
            ));
        }
    }
    Ok(())
}

fn verify_output_step(
    step: &SelectiveProjectedPoseidon2OutputStep,
    artifact: &SelectiveProjectedRowArtifact,
    layout: &super::super::SelectiveLayout,
    arm: usize,
) -> Result<(), LowNormR1csError> {
    let emitted_run = layout
        .compiler_audit
        .rows()
        .emitted_runs()
        .get(artifact.run_index)
        .ok_or_else(|| trace_error("projected Poseidon2 output row has an invalid emitted-run owner"))?;
    if artifact.emitted_row != step.emitted_row
        || emitted_run.rewrite_id().map(|id| id.index()) != Some(step.rewrite_id)
        || emitted_run.family() != SelectiveEmittedRowFamily::Poseidon2
        || emitted_run.arm() != Some(arm)
    {
        return Err(trace_error(
            "projected Poseidon2 output step differs from its emitted-row owner",
        ));
    }

    let mut expected = (0..SELECTIVE_ARITY)
        .map(|_| MatrixTerms::new(false))
        .collect::<Vec<_>>();
    expected[GENERAL_SELECTOR].push((0, layout.selector_cols[arm], F::ONE));
    let output = step
        .output
        .terms
        .first()
        .filter(|_| step.output.constant == F::ZERO && step.output.terms.len() == 1)
        .ok_or_else(|| trace_error("projected Poseidon2 retained output is not one source column"))?;
    append_field(
        &mut expected[C],
        0,
        output.column,
        output.coefficient,
        &layout.slots[arm],
        &layout.plans[arm].definitions,
    )?;
    let linear_form = Lc {
        terms: step
            .linear_form
            .terms
            .iter()
            .map(|term| (term.column, term.coefficient))
            .collect(),
        constant: step.linear_form.constant,
    };
    append_lc_scaled(
        &mut expected[C],
        0,
        &linear_form,
        -F::ONE,
        &layout.slots[arm],
        &layout.plans[arm].definitions,
    )?;
    for (port, terms) in expected.iter().enumerate() {
        if project_port(terms, 0, artifact.columns)? != artifact.ports[port] {
            return Err(trace_error(
                "projected Poseidon2 output step does not reproduce its exact emitted row",
            ));
        }
    }
    Ok(())
}

pub(super) fn project_steps(
    source_arm: &SparseR1cs,
    layout: &super::super::SelectiveLayout,
    arm: usize,
    row_artifacts: &[SelectiveProjectedRowArtifact],
) -> Result<
    (
        Vec<SelectiveProjectedPoseidon2SboxStep>,
        Vec<SelectiveProjectedPoseidon2OutputStep>,
    ),
    LowNormR1csError,
> {
    let prepared = layout.prepared_rows.arm(arm);
    let artifacts = row_artifacts
        .iter()
        .map(|artifact| (artifact.emitted_row, artifact))
        .collect::<BTreeMap<_, _>>();
    let mut sbox_steps = Vec::new();
    let mut output_steps = Vec::new();
    for (trace_index, trace) in source_arm.poseidon2_traces().iter().enumerate() {
        let rewrite_id = prepared.poseidon2_rewrite(trace_index);
        let (source_rows, emitted_rows) = rewrite_geometry(layout, rewrite_id, arm, SelectiveRewriteKind::Poseidon2)?;
        if emitted_rows.len() < trace.sboxes.len() {
            return Err(trace_error("projected Poseidon2 rewrite omits S-box rows"));
        }
        for (offset, sbox) in trace.sboxes.iter().enumerate() {
            let emitted_row = emitted_rows.start + offset;
            let Some(artifact) = artifacts.get(&emitted_row).copied() else {
                continue;
            };
            let step = SelectiveProjectedPoseidon2SboxStep {
                emitted_row,
                rewrite_id: rewrite_id.index(),
                source_rows: source_rows.clone(),
                input: source_lc(sbox.input.constant, &sbox.input.terms),
                output: source_lc(F::ZERO, &[(sbox.output_col, F::ONE)]),
            };
            verify_sbox_step(&step, artifact, layout, arm)?;
            sbox_steps.push(step);
        }
        let mut emitted_row = emitted_rows.start + trace.sboxes.len();
        for (&output_col, linear_form) in trace.output_cols.iter().zip(&trace.output_linear_forms) {
            if layout.plans[arm].definitions.get(output_col).is_some() {
                continue;
            }
            if let Some(artifact) = artifacts.get(&emitted_row).copied() {
                let step = SelectiveProjectedPoseidon2OutputStep {
                    emitted_row,
                    rewrite_id: rewrite_id.index(),
                    source_rows: source_rows.clone(),
                    output: source_lc(F::ZERO, &[(output_col, F::ONE)]),
                    linear_form: source_lc(linear_form.constant, &linear_form.terms),
                };
                verify_output_step(&step, artifact, layout, arm)?;
                output_steps.push(step);
            }
            emitted_row += 1;
        }
        if emitted_row != emitted_rows.end {
            return Err(trace_error(
                "projected Poseidon2 output count differs from its compiler ledger",
            ));
        }
    }
    let selected_poseidon2_rows = row_artifacts
        .iter()
        .filter(|artifact| artifact.family == SelectiveEmittedRowFamily::Poseidon2)
        .count();
    if sbox_steps.len() + output_steps.len() != selected_poseidon2_rows {
        return Err(trace_error("projected Poseidon2 steps do not cover every selected row"));
    }
    Ok((sbox_steps, output_steps))
}
