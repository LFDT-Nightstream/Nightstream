//! Lean data emission for one exact bound fixed-point slice.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use neo_fold_clean::frontends::r1cs_f_prime::{
    SelectiveEmittedRowFamily, SelectiveProjectedPort, SelectiveRewriteKind,
};
use p3_field::PrimeField64;
use recursive_constraint_minimizer::{
    row_is_satisfied, validate_scalar_certificate, FieldModel, Problem, ScalarCertificate, Scope, Selection,
};

use super::{ExportError, FixedPointProblemExport, TerminalProblemExport};

/// Render one self-contained Lean data module with exact source and final rows.
///
/// Coherence must be proved in separate bounded leaf modules and assembled
/// with `BoundArtifact.StructuralCertificate.sound`. The data module contains
/// no solver result and makes no validity, redundancy, or necessity claim.
pub fn render_bound_artifact_data_lean(
    export: &FixedPointProblemExport,
    namespace: &str,
) -> Result<String, ExportError> {
    render_bound_artifact_for(export, namespace)
}

/// Render a complete fixed-point branch data artifact after Rust checks full
/// source-row and family coverage. A separate structural Lean certificate is
/// still required before any proof can use that coverage.
pub fn render_complete_bound_artifact_data_lean(
    export: &FixedPointProblemExport,
    namespace: &str,
) -> Result<String, ExportError> {
    validate_complete_problem(export.problem())?;
    render_bound_artifact_for(export, namespace)
}

fn render_bound_artifact_for(export: &FixedPointProblemExport, namespace: &str) -> Result<String, ExportError> {
    validate_namespace(namespace)?;
    export
        .problem()
        .validate()
        .map_err(|error| ExportError::new(format!("cannot emit invalid problem: {error}")))?;
    let binding = export.binding();
    if export
        .problem()
        .rows
        .iter()
        .map(|row| row.source_index)
        .ne(binding.requested_source_rows().iter().copied())
        || binding
            .projected_rows()
            .iter()
            .map(|row| row.emitted_row())
            .ne(binding.emitted_rows().iter().copied())
        || binding
            .projected_rows()
            .iter()
            .any(|row| row.ports().len() != 13)
    {
        return Err(ExportError::new(
            "bound artifact is not coherent enough for Lean emission",
        ));
    }

    let mut out = String::new();
    writeln!(out, "import Nightstream.Assurance.ConstraintMinimization\n").unwrap();
    writeln!(out, "namespace {namespace}\n").unwrap();
    writeln!(out, "open Nightstream.Assurance.ConstraintMinimization\n").unwrap();
    render_source_artifact(&mut out, export.problem())?;
    render_selective_binding(&mut out, export)?;
    writeln!(
        out,
        "def boundArtifact : BoundArtifact :=\n  {{ source := sourceArtifact, binding := selectiveBinding }}\n"
    )
    .unwrap();
    render_structural_certificate_notice(&mut out, "boundArtifact")?;
    writeln!(out, "end {namespace}").unwrap();
    Ok(out)
}

/// Render one self-contained Lean module with exact terminal source and
/// padded Spartan rows.
///
/// Coherence must be proved in separate bounded leaf modules and assembled
/// with `TerminalBoundArtifact.StructuralCertificate.sound`.
pub fn render_terminal_bound_artifact_data_lean(
    export: &TerminalProblemExport,
    namespace: &str,
) -> Result<String, ExportError> {
    render_terminal_bound_artifact_for(export, namespace)
}

/// Render a complete terminal polynomial data artifact after Rust checks full
/// source-row coverage. A separate structural Lean certificate is required.
pub fn render_complete_terminal_bound_artifact_data_lean(
    export: &TerminalProblemExport,
    namespace: &str,
) -> Result<String, ExportError> {
    validate_complete_problem(export.problem())?;
    render_terminal_bound_artifact_for(export, namespace)
}

fn render_terminal_bound_artifact_for(export: &TerminalProblemExport, namespace: &str) -> Result<String, ExportError> {
    validate_namespace(namespace)?;
    export
        .problem()
        .validate()
        .map_err(|error| ExportError::new(format!("cannot emit invalid problem: {error}")))?;
    validate_terminal_export(export)?;

    let mut out = String::new();
    writeln!(out, "import Nightstream.Assurance.ConstraintMinimization\n").unwrap();
    writeln!(out, "namespace {namespace}\n").unwrap();
    writeln!(out, "open Nightstream.Assurance.ConstraintMinimization\n").unwrap();
    render_source_artifact(&mut out, export.problem())?;
    render_terminal_binding(&mut out, export)?;
    writeln!(
        out,
        "def terminalBoundArtifact : TerminalBoundArtifact :=\n  {{ source := sourceArtifact, binding := terminalBinding }}\n"
    )
    .unwrap();
    render_structural_certificate_notice(&mut out, "terminalBoundArtifact")?;
    writeln!(out, "end {namespace}").unwrap();
    Ok(out)
}

/// Render exact scalar-certificate data against a separately generated bound
/// artifact. Separate bounded leaf modules must prove `FamilyCertificate.Valid`
/// before the candidate can authorize a removal.
#[allow(clippy::too_many_arguments)]
pub fn render_redundancy_candidate_lean(
    complete_problem: &Problem,
    query_problem: &Problem,
    certificate: &ScalarCertificate,
    artifact_module: &str,
    artifact_namespace: &str,
    namespace: &str,
    reviewed_plan: &[String],
) -> Result<String, ExportError> {
    render_redundancy_certificate_for(
        complete_problem,
        query_problem,
        certificate,
        artifact_module,
        artifact_namespace,
        namespace,
        reviewed_plan,
    )
}

/// Render one Lean module that checks a scalar terminal-family redundancy
/// certificate against a separately generated terminal artifact module.
#[allow(clippy::too_many_arguments)]
pub fn render_terminal_redundancy_candidate_lean(
    complete_problem: &Problem,
    query_problem: &Problem,
    certificate: &ScalarCertificate,
    artifact_module: &str,
    artifact_namespace: &str,
    namespace: &str,
    reviewed_plan: &[String],
) -> Result<String, ExportError> {
    render_redundancy_certificate_for(
        complete_problem,
        query_problem,
        certificate,
        artifact_module,
        artifact_namespace,
        namespace,
        reviewed_plan,
    )
}

/// Render complete Rust-replayed removal-counterexample data for one fixed-point
/// branch. It remains a candidate until bounded Lean row-replay leaves prove
/// `RemovalCounterexample.Valid`.
#[allow(clippy::too_many_arguments)]
pub fn render_removal_counterexample_candidate_lean(
    complete_problem: &Problem,
    model: &FieldModel,
    removed_family: &str,
    artifact_module: &str,
    artifact_namespace: &str,
    namespace: &str,
    reviewed_plan: &[String],
) -> Result<String, ExportError> {
    render_removal_counterexample_for(
        complete_problem,
        model,
        removed_family,
        artifact_module,
        artifact_namespace,
        namespace,
        reviewed_plan,
    )
}

/// Render complete Rust-replayed terminal removal-counterexample data. It does
/// not authorize a family removal without a separate structural Lean proof.
#[allow(clippy::too_many_arguments)]
pub fn render_terminal_removal_counterexample_candidate_lean(
    complete_problem: &Problem,
    model: &FieldModel,
    removed_family: &str,
    artifact_module: &str,
    artifact_namespace: &str,
    namespace: &str,
    reviewed_plan: &[String],
) -> Result<String, ExportError> {
    render_removal_counterexample_for(
        complete_problem,
        model,
        removed_family,
        artifact_module,
        artifact_namespace,
        namespace,
        reviewed_plan,
    )
}

#[allow(clippy::too_many_arguments)]
fn render_removal_counterexample_for(
    complete_problem: &Problem,
    model: &FieldModel,
    removed_family: &str,
    artifact_module: &str,
    artifact_namespace: &str,
    namespace: &str,
    reviewed_plan: &[String],
) -> Result<String, ExportError> {
    validate_namespace(artifact_module)?;
    validate_namespace(artifact_namespace)?;
    validate_namespace(namespace)?;
    validate_complete_problem(complete_problem)?;
    validate_reviewed_plan(complete_problem, removed_family, reviewed_plan)?;
    validate_removal_model(complete_problem, model, removed_family, reviewed_plan)?;

    let mut out = String::new();
    writeln!(out, "import {artifact_module}\n").unwrap();
    writeln!(out, "namespace {namespace}\n").unwrap();
    writeln!(out, "open Nightstream.Assurance.ConstraintMinimization").unwrap();
    writeln!(out, "open Nightstream.SuperNeo.CheckPlan").unwrap();
    writeln!(out, "open {artifact_namespace}\n").unwrap();
    write!(out, "def reviewedPlan : List String := [").unwrap();
    for (index, entry) in reviewed_plan.iter().enumerate() {
        write!(out, "{}{}", lean_string(entry), separator(index, reviewed_plan.len())).unwrap();
    }
    writeln!(out, "]\n").unwrap();
    writeln!(out, "def removalCounterexample : RemovalCounterexample where").unwrap();
    writeln!(out, "  removedFamily := {}", lean_string(removed_family)).unwrap();
    write!(out, "  values := [").unwrap();
    for (index, value) in model.values().iter().enumerate() {
        write!(out, "{value}{}", separator(index, model.values().len())).unwrap();
    }
    writeln!(out, "]\n").unwrap();
    render_structural_certificate_notice(&mut out, "removalCounterexample")?;
    writeln!(out, "end {namespace}").unwrap();
    Ok(out)
}

fn validate_complete_problem(problem: &Problem) -> Result<(), ExportError> {
    problem
        .validate()
        .map_err(|error| ExportError::new(format!("cannot use invalid complete problem: {error}")))?;
    let source_rows_are_complete = problem.rows.len() == problem.source.total_rows
        && problem
            .rows
            .iter()
            .enumerate()
            .all(|(source_index, row)| row.source_index == source_index);
    let row_families = problem
        .rows
        .iter()
        .map(|row| row.family.as_str())
        .collect::<BTreeSet<_>>();
    let complete_families = problem
        .complete_families
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if !source_rows_are_complete || row_families != complete_families {
        return Err(ExportError::new("Lean result requires every source row and family"));
    }
    Ok(())
}

fn validate_reviewed_plan(
    problem: &Problem,
    removed_family: &str,
    reviewed_plan: &[String],
) -> Result<(), ExportError> {
    let mut seen = BTreeSet::new();
    if removed_family.is_empty()
        || reviewed_plan.is_empty()
        || !reviewed_plan.iter().all(|family| {
            !family.is_empty()
                && seen.insert(family.as_str())
                && problem
                    .complete_families
                    .iter()
                    .any(|complete| complete == family)
        })
        || !reviewed_plan.iter().any(|family| family == removed_family)
    {
        return Err(ExportError::new(
            "reviewed plan must contain the removed family and only unique complete families",
        ));
    }
    Ok(())
}

fn validate_removal_model(
    problem: &Problem,
    model: &FieldModel,
    removed_family: &str,
    reviewed_plan: &[String],
) -> Result<(), ExportError> {
    if model.values().len() != problem.column_count {
        return Err(ExportError::new(
            "removal model width differs from the complete source relation",
        ));
    }
    if model.values()[problem.constant_one_column] != 1 {
        return Err(ExportError::new(
            "removal model does not set the constant-one column to one",
        ));
    }
    let retained = reviewed_plan
        .iter()
        .filter(|family| family.as_str() != removed_family)
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let mut removed_row_fails = false;
    for row in &problem.rows {
        let holds = row_is_satisfied(row, model)
            .map_err(|error| ExportError::new(format!("complete removal-model replay failed: {error}")))?;
        if retained.contains(row.family.as_str()) && !holds {
            return Err(ExportError::new(format!(
                "removal model violates retained family {:?} at source row {}",
                row.family, row.source_index
            )));
        }
        if row.family == removed_family && !holds {
            removed_row_fails = true;
        }
    }
    if !removed_row_fails {
        return Err(ExportError::new(
            "removal model does not violate a row in the removed family",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn render_redundancy_certificate_for(
    complete_problem: &Problem,
    query_problem: &Problem,
    certificate: &ScalarCertificate,
    artifact_module: &str,
    artifact_namespace: &str,
    namespace: &str,
    reviewed_plan: &[String],
) -> Result<String, ExportError> {
    validate_namespace(artifact_module)?;
    validate_namespace(artifact_namespace)?;
    validate_namespace(namespace)?;
    validate_complete_problem(complete_problem)?;
    validate_scalar_certificate(query_problem, certificate)
        .map_err(|error| ExportError::new(format!("invalid scalar certificate: {error}")))?;
    let Selection::Family(family) = &certificate.selection else {
        return Err(ExportError::new(
            "Lean family emission requires a complete family selection",
        ));
    };
    validate_reviewed_plan(complete_problem, family, reviewed_plan)?;
    validate_certificate_slice(complete_problem, query_problem, family)?;

    let mut out = String::new();
    writeln!(out, "import {artifact_module}\n").unwrap();
    writeln!(out, "namespace {namespace}\n").unwrap();
    writeln!(out, "open Nightstream.Assurance.ConstraintMinimization").unwrap();
    writeln!(out, "open Nightstream.SuperNeo.CheckPlan").unwrap();
    writeln!(out, "open {artifact_namespace}\n").unwrap();
    write!(out, "def reviewedPlan : List String := [").unwrap();
    for (index, entry) in reviewed_plan.iter().enumerate() {
        write!(out, "{}{}", lean_string(entry), separator(index, reviewed_plan.len())).unwrap();
    }
    writeln!(out, "]\n").unwrap();

    writeln!(out, "def familyCertificate : FamilyCertificate where").unwrap();
    writeln!(out, "  family := {}", lean_string(family)).unwrap();
    writeln!(out, "  certificates := [").unwrap();
    for (index, row_certificate) in certificate.rows.iter().enumerate() {
        let candidate = source_row(query_problem, row_certificate.candidate_source_index)?;
        write!(out, "    {{ candidate := ").unwrap();
        render_indexed_row(&mut out, candidate)?;
        writeln!(out, ", support := [").unwrap();
        for (support_index, support) in row_certificate.support.iter().enumerate() {
            let source = source_row(query_problem, support.source_index)?;
            let coefficient = support.coefficient.parse::<u64>().map_err(|_| {
                ExportError::new(format!(
                    "cannot emit noncanonical certificate coefficient {:?}",
                    support.coefficient
                ))
            })?;
            write!(out, "      {{ source := ").unwrap();
            render_indexed_row(&mut out, source)?;
            writeln!(
                out,
                ", coefficient := ({} : Field) }}{}",
                coefficient,
                separator(support_index, row_certificate.support.len())
            )
            .unwrap();
        }
        writeln!(out, "    ] }}{}", separator(index, certificate.rows.len())).unwrap();
    }
    writeln!(out, "  ]\n").unwrap();
    render_structural_certificate_notice(&mut out, "familyCertificate")?;
    writeln!(out, "end {namespace}").unwrap();
    Ok(out)
}

fn validate_certificate_slice(
    complete_problem: &Problem,
    query_problem: &Problem,
    candidate_family: &str,
) -> Result<(), ExportError> {
    query_problem
        .validate()
        .map_err(|error| ExportError::new(format!("cannot use invalid query problem: {error}")))?;
    if query_problem.schema != complete_problem.schema
        || query_problem.source != complete_problem.source
        || query_problem.field_modulus != complete_problem.field_modulus
        || query_problem.column_count != complete_problem.column_count
        || query_problem.constant_one_column != complete_problem.constant_one_column
        || query_problem.public_input_count != complete_problem.public_input_count
    {
        return Err(ExportError::new(
            "certificate query identity differs from the complete source relation",
        ));
    }
    for row in &query_problem.rows {
        if source_row(complete_problem, row.source_index)? != row {
            return Err(ExportError::new(format!(
                "certificate query row {} differs from the complete source relation",
                row.source_index
            )));
        }
    }
    let complete_candidates = complete_problem
        .rows
        .iter()
        .filter(|row| row.family == candidate_family)
        .collect::<Vec<_>>();
    let query_candidates = query_problem
        .rows
        .iter()
        .filter(|row| row.family == candidate_family)
        .collect::<Vec<_>>();
    if complete_candidates != query_candidates
        || !query_problem
            .complete_families
            .iter()
            .any(|family| family == candidate_family)
    {
        return Err(ExportError::new(
            "certificate query does not contain the complete candidate family",
        ));
    }
    Ok(())
}

fn render_source_artifact(out: &mut String, problem: &Problem) -> Result<(), ExportError> {
    writeln!(out, "def sourceArtifact : Artifact :=").unwrap();
    writeln!(out, "  {{ schema := {}", lean_string(&problem.schema)).unwrap();
    writeln!(out, "    profile := {}", lean_string(&problem.source.profile)).unwrap();
    writeln!(out, "    scope := {}", lean_string(scope_name(&problem.source.scope))).unwrap();
    writeln!(
        out,
        "    diagnosticDigest := {}",
        lean_string(&problem.source.artifact_digest)
    )
    .unwrap();
    writeln!(out, "    fieldModulus := {}", lean_string(&problem.field_modulus)).unwrap();
    writeln!(out, "    totalRows := {}", problem.source.total_rows).unwrap();
    writeln!(out, "    columnCount := {}", problem.column_count).unwrap();
    writeln!(out, "    constantOneColumn := {}", problem.constant_one_column).unwrap();
    writeln!(out, "    publicInputCount := {}", problem.public_input_count).unwrap();
    write!(out, "    completeFamilies := [").unwrap();
    for (index, family) in problem.complete_families.iter().enumerate() {
        write!(
            out,
            "{}{}",
            lean_string(family),
            separator(index, problem.complete_families.len())
        )
        .unwrap();
    }
    writeln!(out, "]").unwrap();
    writeln!(out, "    rows := [").unwrap();
    for (index, row) in problem.rows.iter().enumerate() {
        write!(
            out,
            "      {{ sourceIndex := {}, family := {}, row := {{ a := ",
            row.source_index,
            lean_string(&row.family)
        )
        .unwrap();
        render_terms(out, &row.a)?;
        write!(out, ", b := ").unwrap();
        render_terms(out, &row.b)?;
        write!(out, ", c := ").unwrap();
        render_terms(out, &row.c)?;
        writeln!(out, " }} }}{}", separator(index, problem.rows.len())).unwrap();
    }
    writeln!(out, "    ] }}\n").unwrap();
    Ok(())
}

fn render_terminal_binding(out: &mut String, export: &TerminalProblemExport) -> Result<(), ExportError> {
    let binding = export.binding();
    let layout = binding.column_layout();
    writeln!(out, "def terminalBinding : TerminalBinding :=").unwrap();
    write!(out, "  {{ requestedSourceRows := ").unwrap();
    render_nats(out, binding.requested_source_rows());
    writeln!(out).unwrap();
    write!(out, "    verifierNativeGuards := [").unwrap();
    for (index, guard) in binding.verifier_native_guards().iter().enumerate() {
        write!(
            out,
            "{}{}",
            lean_string(guard),
            separator(index, binding.verifier_native_guards().len())
        )
        .unwrap();
    }
    writeln!(out, "]").unwrap();
    writeln!(out, "    columnLayout :=").unwrap();
    writeln!(
        out,
        "      {{ sourcePublicColumns := {}",
        layout.source_public_columns()
    )
    .unwrap();
    writeln!(
        out,
        "        sourcePrivateColumns := {}",
        layout.source_private_columns()
    )
    .unwrap();
    writeln!(
        out,
        "        spartanPrivateColumns := {} }}",
        layout.spartan_private_columns()
    )
    .unwrap();
    writeln!(out, "    projectedRows := [").unwrap();
    for (index, row) in binding.projected_rows().iter().enumerate() {
        write!(
            out,
            "      {{ sourceRow := {}, spartanRow := {}, row := {{ a := ",
            row.source_row(),
            row.spartan_row()
        )
        .unwrap();
        render_terms(out, row.a())?;
        write!(out, ", b := ").unwrap();
        render_terms(out, row.b())?;
        write!(out, ", c := ").unwrap();
        render_terms(out, row.c())?;
        writeln!(out, " }} }}{}", separator(index, binding.projected_rows().len())).unwrap();
    }
    writeln!(out, "    ]").unwrap();
    writeln!(out, "    spartanRows := {}", binding.spartan_rows()).unwrap();
    writeln!(out, "    spartanColumns := {}", binding.spartan_columns()).unwrap();
    writeln!(
        out,
        "    spartanPaddingRows := {}",
        lean_range(binding.spartan_padding_rows())
    )
    .unwrap();
    writeln!(
        out,
        "    spartanPrivatePaddingColumns := {}",
        binding.spartan_private_padding_columns()
    )
    .unwrap();
    writeln!(
        out,
        "    diagnosticDigest := {} }}\n",
        lean_string(binding.diagnostic_digest())
    )
    .unwrap();
    Ok(())
}

fn validate_terminal_export(export: &TerminalProblemExport) -> Result<(), ExportError> {
    let problem = export.problem();
    let binding = export.binding();
    let layout = binding.column_layout();
    let source_rows_match = problem
        .rows
        .iter()
        .map(|row| row.source_index)
        .eq(binding.requested_source_rows().iter().copied());
    let projected_rows_match = binding
        .projected_rows()
        .iter()
        .map(|row| row.source_row())
        .eq(binding.requested_source_rows().iter().copied());
    let projected_rows_in_range = binding
        .projected_rows()
        .iter()
        .all(|row| row.source_row() == row.spartan_row() && row.spartan_row() < binding.spartan_rows());
    let source_columns = layout
        .source_public_columns()
        .checked_add(layout.source_private_columns());
    let spartan_columns = layout
        .spartan_private_columns()
        .checked_add(layout.source_public_columns());
    let private_padding = layout
        .spartan_private_columns()
        .checked_sub(layout.source_private_columns());
    let padding_rows = binding.spartan_padding_rows();
    if !source_rows_match
        || !projected_rows_match
        || !projected_rows_in_range
        || source_columns != Some(problem.column_count)
        || spartan_columns != Some(binding.spartan_columns())
        || private_padding != Some(binding.spartan_private_padding_columns())
        || padding_rows.start != problem.source.total_rows
        || padding_rows.end != binding.spartan_rows()
    {
        return Err(ExportError::new(
            "terminal bound artifact is not coherent enough for Lean emission",
        ));
    }
    Ok(())
}

fn render_selective_binding(out: &mut String, export: &FixedPointProblemExport) -> Result<(), ExportError> {
    let binding = export.binding();
    writeln!(out, "def selectiveBinding : SelectiveBinding :=").unwrap();
    writeln!(out, "  {{ branch := {}", lean_string(binding.branch())).unwrap();
    write!(out, "    requestedSourceRows := ").unwrap();
    render_nats(out, binding.requested_source_rows());
    writeln!(out).unwrap();
    write!(out, "    closureSourceRows := ").unwrap();
    render_nats(out, binding.closure_source_rows());
    writeln!(out).unwrap();
    write!(out, "    additionalSourceRows := ").unwrap();
    render_nats(out, binding.additional_source_rows());
    writeln!(out).unwrap();
    writeln!(out, "    retainedRows := [").unwrap();
    for (index, row) in binding.retained_rows().iter().enumerate() {
        writeln!(
            out,
            "      {{ sourceRow := {}, emittedRow := {}, stageOccurrence := {} }}{}",
            row.source_row(),
            row.emitted_row(),
            lean_option(row.stage_occurrence()),
            separator(index, binding.retained_rows().len())
        )
        .unwrap();
    }
    writeln!(out, "    ]").unwrap();
    writeln!(out, "    rewrites := [").unwrap();
    for (index, rewrite) in binding.rewrites().iter().enumerate() {
        write!(
            out,
            "      {{ rewriteId := {}, kind := {}, sourceRows := ",
            rewrite.rewrite_id(),
            lean_string(rewrite_kind_name(rewrite.kind()))
        )
        .unwrap();
        render_ranges(out, rewrite.source_rows());
        writeln!(
            out,
            ", emittedRows := {}, stageOccurrence := {} }}{}",
            lean_range(rewrite.emitted_rows()),
            lean_option(rewrite.stage_occurrence()),
            separator(index, binding.rewrites().len())
        )
        .unwrap();
    }
    writeln!(out, "    ]").unwrap();
    write!(out, "    emittedRows := ").unwrap();
    render_nats(out, binding.emitted_rows());
    writeln!(out).unwrap();
    writeln!(out, "    finalRows := {}", binding.final_rows()).unwrap();
    writeln!(out, "    finalColumns := {}", binding.final_columns()).unwrap();
    writeln!(
        out,
        "    finalPublicInputCount := {}",
        binding.final_public_input_count()
    )
    .unwrap();
    writeln!(
        out,
        "    finalPlanDigest := {}",
        lean_string(binding.final_plan_digest())
    )
    .unwrap();
    writeln!(
        out,
        "    projectedSliceDigest := {}",
        lean_string(binding.projected_slice_digest())
    )
    .unwrap();
    writeln!(out, "    projectedRows := [").unwrap();
    for (index, row) in binding.projected_rows().iter().enumerate() {
        write!(
            out,
            "      {{ emittedRow := {}, runIndex := {}, family := {}, arm := {}, ports := [",
            row.emitted_row(),
            row.run_index(),
            lean_string(emitted_family_name(row.family())),
            lean_option(row.arm())
        )
        .unwrap();
        for (port_index, port) in row.ports().iter().enumerate() {
            render_port(out, port)?;
            write!(out, "{}", separator(port_index, row.ports().len())).unwrap();
        }
        writeln!(out, "] }}{}", separator(index, binding.projected_rows().len())).unwrap();
    }
    writeln!(out, "    ] }}\n").unwrap();
    Ok(())
}

fn render_port(out: &mut String, port: &SelectiveProjectedPort) -> Result<(), ExportError> {
    write!(out, "{{ explicit := [").unwrap();
    for (index, term) in port.explicit().iter().enumerate() {
        write!(
            out,
            "({}, {}){}",
            term.column(),
            term.coefficient().as_canonical_u64(),
            separator(index, port.explicit().len())
        )
        .unwrap();
    }
    write!(out, "], geometricRuns := [").unwrap();
    for (index, run) in port.geometric_runs().iter().enumerate() {
        write!(
            out,
            "{{ columnStart := {}, length := {}, initial := {}, ratio := {} }}{}",
            run.column_start(),
            run.length(),
            run.initial().as_canonical_u64(),
            run.ratio().as_canonical_u64(),
            separator(index, port.geometric_runs().len())
        )
        .unwrap();
    }
    write!(out, "], seededBlocks := [").unwrap();
    for (index, block) in port.seeded_blocks().iter().enumerate() {
        render_seeded_block(out, block)?;
        write!(out, "{}", separator(index, port.seeded_blocks().len())).unwrap();
    }
    write!(out, "] }}").unwrap();
    Ok(())
}

fn render_seeded_block(out: &mut String, block: &neo_ccs::SeededPhi81LinearBlock) -> Result<(), ExportError> {
    write!(out, "{{ rowStart := {}, wordStarts := ", block.row_start()).unwrap();
    render_nats(out, block.word_starts());
    write!(
        out,
        ", wordWidth := {}, kappa := {}, messageCols := {}, chunkSize := {}, chunkSeedsByRow := [",
        block.word_width(),
        block.kappa(),
        block.message_cols(),
        block.chunk_size()
    )
    .unwrap();
    for (output_index, output_seeds) in block.chunk_seeds_by_row().iter().enumerate() {
        write!(out, "[").unwrap();
        for (seed_index, seed) in output_seeds.iter().enumerate() {
            write!(out, "[").unwrap();
            for (byte_index, byte) in seed.iter().enumerate() {
                write!(out, "{}{}", byte, separator(byte_index, seed.len())).unwrap();
            }
            write!(out, "]{}", separator(seed_index, output_seeds.len())).unwrap();
        }
        write!(out, "]{}", separator(output_index, block.chunk_seeds_by_row().len())).unwrap();
    }
    write!(
        out,
        "], superneoTransformedColumns := {} }}",
        block.has_superneo_transformed_columns()
    )
    .unwrap();
    Ok(())
}

fn render_terms(out: &mut String, terms: &[recursive_constraint_minimizer::Term]) -> Result<(), ExportError> {
    write!(out, "[").unwrap();
    for (index, term) in terms.iter().enumerate() {
        let coefficient = term
            .coefficient
            .parse::<u64>()
            .map_err(|_| ExportError::new(format!("cannot emit noncanonical coefficient {:?}", term.coefficient)))?;
        write!(
            out,
            "({}, {}){}",
            term.column,
            coefficient,
            separator(index, terms.len())
        )
        .unwrap();
    }
    write!(out, "]").unwrap();
    Ok(())
}

fn render_indexed_row(out: &mut String, row: &recursive_constraint_minimizer::Row) -> Result<(), ExportError> {
    write!(
        out,
        "{{ sourceIndex := {}, family := {}, row := {{ a := ",
        row.source_index,
        lean_string(&row.family)
    )
    .unwrap();
    render_terms(out, &row.a)?;
    write!(out, ", b := ").unwrap();
    render_terms(out, &row.b)?;
    write!(out, ", c := ").unwrap();
    render_terms(out, &row.c)?;
    write!(out, " }} }}").unwrap();
    Ok(())
}

fn source_row(problem: &Problem, source_index: usize) -> Result<&recursive_constraint_minimizer::Row, ExportError> {
    problem
        .rows
        .binary_search_by_key(&source_index, |row| row.source_index)
        .map(|index| &problem.rows[index])
        .map_err(|_| {
            ExportError::new(format!(
                "certificate references source row {source_index} outside the bound artifact"
            ))
        })
}

fn render_nats(out: &mut String, values: &[usize]) {
    write!(out, "[").unwrap();
    for (index, value) in values.iter().enumerate() {
        write!(out, "{value}{}", separator(index, values.len())).unwrap();
    }
    write!(out, "]").unwrap();
}

fn render_ranges(out: &mut String, ranges: &[std::ops::Range<usize>]) {
    write!(out, "[").unwrap();
    for (index, range) in ranges.iter().enumerate() {
        write!(out, "{}{}", lean_range(range.clone()), separator(index, ranges.len())).unwrap();
    }
    write!(out, "]").unwrap();
}

fn lean_range(range: std::ops::Range<usize>) -> String {
    format!("{{ start := {}, stop := {} }}", range.start, range.end)
}

fn lean_option(value: Option<usize>) -> String {
    value.map_or_else(|| "none".to_owned(), |value| format!("some {value}"))
}

fn lean_string(value: &str) -> String {
    format!("{value:?}")
}

fn render_structural_certificate_notice(out: &mut String, value: &str) -> Result<(), ExportError> {
    writeln!(
        out,
        "/- {value} is exact data only. A separate bounded structural Lean\ncertificate is required before this value can authorize a constraint removal. -/\n"
    )
    .map_err(|error| ExportError::new(format!("cannot render Lean certificate notice: {error}")))
}

fn scope_name(scope: &Scope) -> &'static str {
    match scope {
        Scope::Local => "local",
        Scope::Branch => "branch",
        Scope::Lifecycle => "lifecycle",
    }
}

fn separator(index: usize, length: usize) -> &'static str {
    if index + 1 == length {
        ""
    } else {
        ","
    }
}

fn rewrite_kind_name(kind: SelectiveRewriteKind) -> &'static str {
    match kind {
        SelectiveRewriteKind::Poseidon2 => "poseidon2",
        SelectiveRewriteKind::CenteredUnit => "centered_unit",
        SelectiveRewriteKind::ShiftedTernaryCanonical => "shifted_ternary_canonical",
        SelectiveRewriteKind::PolynomialEvaluation => "polynomial_evaluation",
        SelectiveRewriteKind::ProductSum => "product_sum",
        SelectiveRewriteKind::LinearDefinition => "linear_definition",
    }
}

fn emitted_family_name(family: SelectiveEmittedRowFamily) -> &'static str {
    match family {
        SelectiveEmittedRowFamily::SelectorDomain => "selector_domain",
        SelectiveEmittedRowFamily::SharedDomain => "shared_domain",
        SelectiveEmittedRowFamily::ArmDomain => "arm_domain",
        SelectiveEmittedRowFamily::OneHot => "one_hot",
        SelectiveEmittedRowFamily::PublicPadding => "public_padding",
        SelectiveEmittedRowFamily::PrivatePadding => "private_padding",
        SelectiveEmittedRowFamily::Retained => "retained",
        SelectiveEmittedRowFamily::Poseidon2 => "poseidon2",
        SelectiveEmittedRowFamily::CenteredUnit => "centered_unit",
        SelectiveEmittedRowFamily::ShiftedTernaryCanonical => "shifted_ternary_canonical",
        SelectiveEmittedRowFamily::PolynomialEvaluation => "polynomial_evaluation",
        SelectiveEmittedRowFamily::ProductSum => "product_sum",
        SelectiveEmittedRowFamily::RingPadding => "ring_padding",
    }
}

fn validate_namespace(namespace: &str) -> Result<(), ExportError> {
    let valid = !namespace.is_empty()
        && namespace.split('.').all(|part| {
            !part.is_empty()
                && part
                    .chars()
                    .next()
                    .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
                && part
                    .chars()
                    .all(|character| character.is_ascii_alphanumeric() || character == '_')
        });
    if !valid {
        return Err(ExportError::new(format!("invalid Lean namespace {namespace:?}")));
    }
    Ok(())
}
