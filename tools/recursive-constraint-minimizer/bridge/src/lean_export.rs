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

/// One generated Lean module: full dotted module name plus file content.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneratedLeanModule {
    pub module_name: String,
    pub content: String,
}

const GENERATED_MODULE_SPLIT_BYTES: usize = 16 * 1024 * 1024;

/// Split one rendered artifact module into data submodules plus the assembly
/// module. Data modules reuse the same namespace, so unqualified references
/// keep resolving; spillable blocks are pure data defs taken as a prefix in
/// dependency order.
fn split_generated_module(content: &str, module_name: &str) -> Result<Vec<GeneratedLeanModule>, ExportError> {
    if content.len() <= GENERATED_MODULE_SPLIT_BYTES {
        return Ok(vec![GeneratedLeanModule {
            module_name: module_name.to_owned(),
            content: content.to_owned(),
        }]);
    }
    let namespace_line = format!("namespace {module_name}\n");
    let end_line = format!("end {module_name}");
    let header_end = content
        .find(&namespace_line)
        .ok_or_else(|| ExportError::new("generated module is missing its namespace line"))?
        + namespace_line.len();
    let header = &content[..header_end];
    let body_end = content
        .rfind(&end_line)
        .ok_or_else(|| ExportError::new("generated module is missing its end line"))?;
    let body = &content[header_end..body_end];
    let footer = &content[body_end..];

    let blocks = body.split("\n\n").collect::<Vec<_>>();
    let spillable = |block: &str| {
        let block = block.trim_start();
        block.starts_with("def hoistedList")
            || (block.starts_with("def ") && {
                let name_end = block[4..]
                    .find(|character: char| character == ' ' || character == ':')
                    .map(|offset| 4 + offset)
                    .unwrap_or(block.len());
                let name = &block[4..name_end];
                name.contains("Chunk")
            })
    };
    let mut modules = Vec::new();
    let mut current = String::new();
    let mut assembly_blocks = Vec::new();
    let mut preamble_blocks = Vec::new();
    let mut seen_def = false;
    for block in blocks {
        if !seen_def {
            if block.trim_start().starts_with("def ") {
                seen_def = true;
            } else {
                preamble_blocks.push(block);
                continue;
            }
        }
        if block.trim() == LEAN_SPLIT_MARKER {
            if current.len() >= GENERATED_MODULE_SPLIT_BYTES {
                modules.push(current);
                current = String::new();
            }
        } else if spillable(block) {
            current.push_str(block);
            current.push_str("\n\n");
        } else {
            assembly_blocks.push(block);
        }
    }
    if !current.is_empty() {
        modules.push(current);
    }
    let preamble = preamble_blocks
        .iter()
        .filter(|block| !block.trim().is_empty())
        .map(|block| format!("{block}\n\n"))
        .collect::<String>();

    let mut generated = Vec::with_capacity(modules.len() + 1);
    let mut data_imports = String::new();
    for (index, module_body) in modules.into_iter().enumerate() {
        let data_name = format!("{module_name}Data{index}");
        let mut data_content = header.to_owned();
        data_content.push('\n');
        data_content.push_str(&preamble);
        data_content.push_str(&module_body);
        data_content.push_str(&format!("end {module_name}\n"));
        writeln!(data_imports, "import {data_name}").unwrap();
        generated.push(GeneratedLeanModule {
            module_name: data_name,
            content: data_content,
        });
    }
    let mut assembly = String::new();
    let import_line = "import Nightstream.Assurance.ConstraintMinimization\n";
    let header_with_imports = header.replacen(import_line, &format!("{import_line}{data_imports}"), 1);
    assembly.push_str(&header_with_imports);
    assembly.push('\n');
    assembly.push_str(&preamble);
    assembly.push_str(&assembly_blocks.join("\n\n"));
    assembly.push_str(footer);
    generated.push(GeneratedLeanModule {
        module_name: module_name.to_owned(),
        content: assembly,
    });
    Ok(generated)
}

/// Render a complete fixed-point artifact as data submodules plus assembly.
pub fn render_complete_bound_artifact_modules(
    export: &FixedPointProblemExport,
    namespace: &str,
) -> Result<Vec<GeneratedLeanModule>, ExportError> {
    let content = render_complete_bound_artifact_lean(export, namespace)?;
    split_generated_module(&content, namespace)
}

/// Render a complete terminal artifact as data submodules plus assembly.
pub fn render_complete_terminal_bound_artifact_modules(
    export: &TerminalProblemExport,
    namespace: &str,
) -> Result<Vec<GeneratedLeanModule>, ExportError> {
    let content = render_complete_terminal_bound_artifact_lean(export, namespace)?;
    split_generated_module(&content, namespace)
}

/// Render one self-contained Lean module with exact source and final rows.
///
/// The output contains data and a decidable coherence theorem. It contains no
/// solver result and makes no redundancy or necessity claim.
pub fn render_bound_artifact_lean(export: &FixedPointProblemExport, namespace: &str) -> Result<String, ExportError> {
    render_bound_artifact_for(export, namespace, false)
}

/// Render a complete fixed-point branch artifact. The additional coverage
/// theorem is required by generated redundancy and necessity proofs.
pub fn render_complete_bound_artifact_lean(
    export: &FixedPointProblemExport,
    namespace: &str,
) -> Result<String, ExportError> {
    validate_complete_problem(export.problem())?;
    render_bound_artifact_for(export, namespace, true)
}

fn render_bound_artifact_for(
    export: &FixedPointProblemExport,
    namespace: &str,
    complete: bool,
) -> Result<String, ExportError> {
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
    writeln!(out, "import Nightstream.Assurance.ConstraintMinimization").unwrap();
    writeln!(out, "set_option maxHeartbeats 2000000").unwrap();
    writeln!(out, "set_option maxRecDepth 65536\n").unwrap();
    writeln!(out, "namespace {namespace}\n").unwrap();
    writeln!(out, "open Nightstream.Assurance.ConstraintMinimization\n").unwrap();
    let mut hoist_counter = 0usize;
    render_source_artifact(&mut out, &mut hoist_counter, export.problem())?;
    render_selective_binding(&mut out, &mut hoist_counter, export)?;
    writeln!(
        out,
        "def boundArtifact : BoundArtifact :=\n  {{ source := sourceArtifact, binding := selectiveBinding }}\n"
    )
    .unwrap();
    writeln!(
        out,
        "theorem boundArtifact_coherent : boundArtifact.Coherent := by\n  native_decide\n"
    )
    .unwrap();
    if complete {
        writeln!(
            out,
            "theorem sourceArtifact_row_count :\n    sourceArtifact.rows.length = sourceArtifact.totalRows := by\n  native_decide\n"
        )
        .unwrap();
        writeln!(
            out,
            "theorem boundArtifact_coversFullRelation :\n    boundArtifact.CoversFullRelation := by\n  native_decide\n"
        )
        .unwrap();
    }
    writeln!(out, "end {namespace}").unwrap();
    Ok(out)
}

/// Render one self-contained Lean module with exact terminal source and
/// padded Spartan rows.
///
/// The output contains data and a decidable coherence theorem. It contains no
/// solver result and makes no redundancy or necessity claim.
pub fn render_terminal_bound_artifact_lean(
    export: &TerminalProblemExport,
    namespace: &str,
) -> Result<String, ExportError> {
    render_terminal_bound_artifact_for(export, namespace, false)
}

/// Render a complete terminal polynomial artifact with an exact source-row
/// coverage theorem.
pub fn render_complete_terminal_bound_artifact_lean(
    export: &TerminalProblemExport,
    namespace: &str,
) -> Result<String, ExportError> {
    validate_complete_problem(export.problem())?;
    render_terminal_bound_artifact_for(export, namespace, true)
}

fn render_terminal_bound_artifact_for(
    export: &TerminalProblemExport,
    namespace: &str,
    complete: bool,
) -> Result<String, ExportError> {
    validate_namespace(namespace)?;
    export
        .problem()
        .validate()
        .map_err(|error| ExportError::new(format!("cannot emit invalid problem: {error}")))?;
    validate_terminal_export(export)?;

    let mut out = String::new();
    writeln!(out, "import Nightstream.Assurance.ConstraintMinimization").unwrap();
    writeln!(out, "set_option maxHeartbeats 2000000").unwrap();
    writeln!(out, "set_option maxRecDepth 65536\n").unwrap();
    writeln!(out, "namespace {namespace}\n").unwrap();
    writeln!(out, "open Nightstream.Assurance.ConstraintMinimization\n").unwrap();
    let mut hoist_counter = 0usize;
    render_source_artifact(&mut out, &mut hoist_counter, export.problem())?;
    render_terminal_binding(&mut out, &mut hoist_counter, export)?;
    writeln!(
        out,
        "def terminalBoundArtifact : TerminalBoundArtifact :=\n  {{ source := sourceArtifact, binding := terminalBinding }}\n"
    )
    .unwrap();
    writeln!(
        out,
        "theorem terminalBoundArtifact_coherent : terminalBoundArtifact.Coherent := by\n  native_decide\n"
    )
    .unwrap();
    if complete {
        writeln!(
            out,
            "theorem sourceArtifact_row_count :\n    sourceArtifact.rows.length = sourceArtifact.totalRows := by\n  native_decide\n"
        )
        .unwrap();
        writeln!(
            out,
            "theorem terminalBoundArtifact_coversFullRelation :\n    terminalBoundArtifact.CoversFullRelation := by\n  native_decide\n"
        )
        .unwrap();
    }
    writeln!(out, "end {namespace}").unwrap();
    Ok(out)
}

/// Render one Lean module that checks a scalar redundancy certificate against
/// a separately generated bound-artifact module.
#[allow(clippy::too_many_arguments)]
pub fn render_redundancy_certificate_lean(
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
        "boundArtifact",
        "redundant_of_full_bound_valid",
        "boundArtifact_coversFullRelation",
    )
}

/// Render one Lean module that checks a scalar terminal-family redundancy
/// certificate against a separately generated terminal artifact module.
#[allow(clippy::too_many_arguments)]
pub fn render_terminal_redundancy_certificate_lean(
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
        "terminalBoundArtifact",
        "redundant_of_full_terminal_bound_valid",
        "terminalBoundArtifact_coversFullRelation",
    )
}

/// Render a complete Rust-replayed removal counterexample for one fixed-point
/// branch as a Lean proof.
#[allow(clippy::too_many_arguments)]
pub fn render_removal_counterexample_lean(
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
        "boundArtifact",
        "necessary_of_full_bound_valid",
        "necessary_normalized_of_full_bound_valid",
        "boundArtifact_coversFullRelation",
    )
}

/// Render a complete Rust-replayed terminal removal counterexample as a Lean
/// proof against the terminal source-to-Spartan artifact.
#[allow(clippy::too_many_arguments)]
pub fn render_terminal_removal_counterexample_lean(
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
        "terminalBoundArtifact",
        "necessary_of_full_terminal_bound_valid",
        "necessary_normalized_of_full_terminal_bound_valid",
        "terminalBoundArtifact_coversFullRelation",
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
    bound_artifact: &str,
    transport_theorem: &str,
    normalized_transport_theorem: &str,
    coverage_theorem: &str,
) -> Result<String, ExportError> {
    validate_namespace(artifact_module)?;
    validate_namespace(artifact_namespace)?;
    validate_namespace(namespace)?;
    validate_complete_problem(complete_problem)?;
    validate_reviewed_plan(complete_problem, removed_family, reviewed_plan)?;
    validate_removal_model(complete_problem, model, removed_family, reviewed_plan)?;

    let mut out = String::new();
    writeln!(out, "import {artifact_module}").unwrap();
    writeln!(out, "set_option maxHeartbeats 2000000").unwrap();
    writeln!(out, "set_option maxRecDepth 65536\n").unwrap();
    writeln!(out, "namespace {namespace}\n").unwrap();
    writeln!(out, "open Nightstream.Assurance.ConstraintMinimization").unwrap();
    writeln!(out, "open Nightstream.SuperNeo.CheckPlan").unwrap();
    writeln!(out, "open {artifact_namespace}\n").unwrap();
    write!(out, "def reviewedPlan : List String := [").unwrap();
    for (index, entry) in reviewed_plan.iter().enumerate() {
        write!(out, "{}{}", lean_string(entry), separator(index, reviewed_plan.len())).unwrap();
    }
    writeln!(out, "]\n").unwrap();
    let values = model
        .values()
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>();
    write_chunked_list_def(&mut out, "removalCounterexampleValues", "Field", &values);
    writeln!(out, "def removalCounterexample : RemovalCounterexample where").unwrap();
    writeln!(out, "  removedFamily := {}", lean_string(removed_family)).unwrap();
    writeln!(out, "  values := removalCounterexampleValues\n").unwrap();
    writeln!(
        out,
        "theorem removalCounterexample_valid :\n    removalCounterexample.Valid {bound_artifact}.source reviewedPlan := by\n  native_decide\n"
    )
    .unwrap();
    writeln!(
        out,
        "theorem necessary :\n    NecessaryForSoundness (FamilyHolds {bound_artifact}.source)\n      (Target {bound_artifact}.source) reviewedPlan {} :=\n  removalCounterexample.{transport_theorem}\n    {bound_artifact} {bound_artifact} reviewedPlan {coverage_theorem}\n    (by native_decide) removalCounterexample_valid\n",
        lean_string(removed_family),
    )
    .unwrap();
    writeln!(
        out,
        "theorem necessaryNormalized :\n    NecessaryForSoundness\n      (NormalizedFamilyHolds {bound_artifact}.source)\n      (NormalizedTarget {bound_artifact}.source) reviewedPlan {} :=\n  removalCounterexample.{normalized_transport_theorem}\n    {bound_artifact} {bound_artifact} reviewedPlan {coverage_theorem}\n    (by native_decide) removalCounterexample_valid\n",
        lean_string(removed_family),
    )
    .unwrap();
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
    bound_artifact: &str,
    transport_theorem: &str,
    coverage_theorem: &str,
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
    writeln!(out, "import {artifact_module}").unwrap();
    writeln!(out, "set_option maxHeartbeats 2000000").unwrap();
    writeln!(out, "set_option maxRecDepth 65536\n").unwrap();
    writeln!(out, "namespace {namespace}\n").unwrap();
    writeln!(out, "open Nightstream.Assurance.ConstraintMinimization").unwrap();
    writeln!(out, "open Nightstream.SuperNeo.CheckPlan").unwrap();
    writeln!(out, "open {artifact_namespace}\n").unwrap();
    write!(out, "def reviewedPlan : List String := [").unwrap();
    for (index, entry) in reviewed_plan.iter().enumerate() {
        write!(out, "{}{}", lean_string(entry), separator(index, reviewed_plan.len())).unwrap();
    }
    writeln!(out, "]\n").unwrap();

    let mut counter = 0usize;
    let certificates = certificate
        .rows
        .iter()
        .map(|row_certificate| {
            let candidate = source_row(query_problem, row_certificate.candidate_source_index)?;
            let mut hoisted = String::new();
            let mut item = String::new();
            write!(item, "{{ candidate := ").unwrap();
            render_indexed_row_hoisted(&mut hoisted, &mut counter, &mut item, candidate)?;
            write!(item, ", support := [").unwrap();
            for (support_index, support) in row_certificate.support.iter().enumerate() {
                let source = source_row(query_problem, support.source_index)?;
                let coefficient = support.coefficient.parse::<u64>().map_err(|_| {
                    ExportError::new(format!(
                        "cannot emit noncanonical certificate coefficient {:?}",
                        support.coefficient
                    ))
                })?;
                write!(item, "{{ source := ").unwrap();
                render_indexed_row_hoisted(&mut hoisted, &mut counter, &mut item, source)?;
                write!(
                    item,
                    ", coefficient := ({} : Field) }}{}",
                    coefficient,
                    separator(support_index, row_certificate.support.len())
                )
                .unwrap();
            }
            write!(item, "] }}").unwrap();
            Ok((hoisted, item))
        })
        .collect::<Result<Vec<_>, ExportError>>()?;
    write_chunked_list_def_grouped(
        &mut out,
        "familyCertificateCertificates",
        "ScalarCertificate",
        &certificates,
    );
    writeln!(out, "def familyCertificate : FamilyCertificate where").unwrap();
    writeln!(out, "  family := {}", lean_string(family)).unwrap();
    writeln!(out, "  certificates := familyCertificateCertificates\n").unwrap();
    writeln!(
        out,
        "theorem familyCertificate_valid :\n    familyCertificate.Valid sourceArtifact reviewedPlan := by"
    )
    .unwrap();
    writeln!(
        out,
        "  simp [FamilyCertificate.Valid, familyCertificate, sourceArtifact,\n    reviewedPlan, ScalarCertificate.Valid, scalarCombination, candidateRows,\n    Algebraic.residual, Algebraic.linearPolynomial] <;> ring\n"
    )
    .unwrap();
    writeln!(
        out,
        "theorem redundant :\n    Redundant (FamilyHolds {bound_artifact}.source) reviewedPlan {} :=\n  familyCertificate.{transport_theorem} {bound_artifact} {bound_artifact}\n    reviewedPlan {coverage_theorem} (by native_decide)\n    familyCertificate_valid\n",
        lean_string(family),
    )
    .unwrap();
    writeln!(
        out,
        "theorem normalizedRedundant :\n    Redundant (NormalizedFamilyHolds {bound_artifact}.source)\n      reviewedPlan {} :=\n  normalizedRedundant_of_redundant {bound_artifact}.source\n    reviewedPlan {} redundant\n",
        lean_string(family),
        lean_string(family),
    )
    .unwrap();
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

const LEAN_LIST_CHUNK: usize = 256;

/// Inline short lists; hoist long ones into chunked defs written to `defs`.
/// Returns the Lean expression to reference at the use site.
fn inline_or_hoist(defs: &mut String, counter: &mut usize, element_type: &str, items: &[String]) -> String {
    if items.len() <= LEAN_LIST_CHUNK {
        return format!("[{}]", items.join(", "));
    }
    let name = format!("hoistedList{counter}");
    *counter += 1;
    write_chunked_list_def(defs, &name, element_type, items);
    name
}

const LEAN_SPLIT_MARKER: &str = "-- lean-split-safe";

/// Chunked list writer for items that carry their own hoisted defs. Hoisted
/// defs interleave immediately before the chunk that consumes them, and every
/// cluster boundary is marked as a safe split point.
fn write_chunked_list_def_grouped(out: &mut String, name: &str, element_type: &str, items: &[(String, String)]) {
    if items.is_empty() {
        writeln!(out, "def {name} : List {element_type} := []\n").unwrap();
        return;
    }
    let chunk_count = items.chunks(LEAN_LIST_CHUNK).count();
    for (chunk_index, chunk) in items.chunks(LEAN_LIST_CHUNK).enumerate() {
        writeln!(out, "{LEAN_SPLIT_MARKER}\n").unwrap();
        for (hoisted, _) in chunk {
            if !hoisted.is_empty() {
                out.push_str(hoisted);
            }
        }
        if chunk_count == 1 {
            writeln!(out, "def {name} : List {element_type} := [").unwrap();
        } else {
            writeln!(out, "def {name}Chunk{chunk_index} : List {element_type} := [").unwrap();
        }
        for (index, (_, item)) in chunk.iter().enumerate() {
            writeln!(out, "  {item}{}", separator(index, chunk.len())).unwrap();
        }
        writeln!(out, "]\n").unwrap();
    }
    if chunk_count > 1 {
        writeln!(out, "{LEAN_SPLIT_MARKER}\n").unwrap();
        writeln!(out, "def {name} : List {element_type} :=\n  List.flatten [").unwrap();
        for chunk_index in 0..chunk_count {
            writeln!(
                out,
                "    {name}Chunk{chunk_index}{}",
                separator(chunk_index, chunk_count)
            )
            .unwrap();
        }
        writeln!(out, "  ]\n").unwrap();
    }
}

fn render_terms_hoisted(
    defs: &mut String,
    counter: &mut usize,
    out: &mut String,
    terms: &[recursive_constraint_minimizer::Term],
) -> Result<(), ExportError> {
    let items = terms
        .iter()
        .map(|term| {
            let coefficient = term.coefficient.parse::<u64>().map_err(|_| {
                ExportError::new(format!("cannot emit noncanonical coefficient {:?}", term.coefficient))
            })?;
            Ok(format!("({}, {})", term.column, coefficient))
        })
        .collect::<Result<Vec<_>, ExportError>>()?;
    write!(out, "{}", inline_or_hoist(defs, counter, "(Nat × Nat)", &items)).unwrap();
    Ok(())
}

fn render_indexed_row_hoisted(
    defs: &mut String,
    counter: &mut usize,
    out: &mut String,
    row: &recursive_constraint_minimizer::Row,
) -> Result<(), ExportError> {
    write!(
        out,
        "{{ sourceIndex := {}, family := {}, row := {{ a := ",
        row.source_index,
        lean_string(&row.family)
    )
    .unwrap();
    render_terms_hoisted(defs, counter, out, &row.a)?;
    write!(out, ", b := ").unwrap();
    render_terms_hoisted(defs, counter, out, &row.b)?;
    write!(out, ", c := ").unwrap();
    render_terms_hoisted(defs, counter, out, &row.c)?;
    write!(out, " }} }}").unwrap();
    Ok(())
}

/// Emit one list as chunked top-level defs so no literal exceeds the
/// elaborator's recursion depth, then one flattened definition.
fn write_chunked_list_def(out: &mut String, name: &str, element_type: &str, items: &[String]) {
    if items.is_empty() {
        writeln!(out, "def {name} : List {element_type} := []\n").unwrap();
        return;
    }
    if items.len() <= LEAN_LIST_CHUNK {
        writeln!(out, "def {name} : List {element_type} := [").unwrap();
        for (index, item) in items.iter().enumerate() {
            writeln!(out, "  {item}{}", separator(index, items.len())).unwrap();
        }
        writeln!(out, "]\n").unwrap();
        return;
    }
    let chunk_count = items.chunks(LEAN_LIST_CHUNK).count();
    for (chunk_index, chunk) in items.chunks(LEAN_LIST_CHUNK).enumerate() {
        writeln!(out, "def {name}Chunk{chunk_index} : List {element_type} := [").unwrap();
        for (index, item) in chunk.iter().enumerate() {
            writeln!(out, "  {item}{}", separator(index, chunk.len())).unwrap();
        }
        writeln!(out, "]\n").unwrap();
    }
    writeln!(out, "def {name} : List {element_type} :=\n  List.flatten [").unwrap();
    for chunk_index in 0..chunk_count {
        writeln!(
            out,
            "    {name}Chunk{chunk_index}{}",
            separator(chunk_index, chunk_count)
        )
        .unwrap();
    }
    writeln!(out, "  ]\n").unwrap();
}

fn render_source_artifact(out: &mut String, counter: &mut usize, problem: &Problem) -> Result<(), ExportError> {
    let rows = problem
        .rows
        .iter()
        .map(|row| {
            let mut hoisted = String::new();
            let mut item = String::new();
            render_indexed_row_hoisted(&mut hoisted, counter, &mut item, row)?;
            Ok((hoisted, item))
        })
        .collect::<Result<Vec<_>, ExportError>>()?;
    write_chunked_list_def_grouped(out, "sourceArtifactRows", "IndexedRow", &rows);
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
    writeln!(out, "    rows := sourceArtifactRows }}\n").unwrap();
    Ok(())
}

fn render_terminal_binding(
    out: &mut String,
    counter: &mut usize,
    export: &TerminalProblemExport,
) -> Result<(), ExportError> {
    let binding = export.binding();
    let layout = binding.column_layout();
    let nat_items = |values: &[usize]| {
        values
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
    };
    write_chunked_list_def(
        out,
        "terminalBindingRequestedSourceRows",
        "Nat",
        &nat_items(binding.requested_source_rows()),
    );
    let projected = binding
        .projected_rows()
        .iter()
        .map(|row| {
            let mut hoisted = String::new();
            let mut item = String::new();
            write!(
                item,
                "{{ sourceRow := {}, spartanRow := {}, row := {{ a := ",
                row.source_row(),
                row.spartan_row()
            )
            .unwrap();
            render_terms_hoisted(&mut hoisted, counter, &mut item, row.a())?;
            write!(item, ", b := ").unwrap();
            render_terms_hoisted(&mut hoisted, counter, &mut item, row.b())?;
            write!(item, ", c := ").unwrap();
            render_terms_hoisted(&mut hoisted, counter, &mut item, row.c())?;
            write!(item, " }} }}").unwrap();
            Ok((hoisted, item))
        })
        .collect::<Result<Vec<_>, ExportError>>()?;
    write_chunked_list_def_grouped(out, "terminalBindingProjectedRows", "TerminalProjectedRow", &projected);
    writeln!(out, "def terminalBinding : TerminalBinding :=").unwrap();
    writeln!(out, "  {{ requestedSourceRows := terminalBindingRequestedSourceRows").unwrap();
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
    writeln!(out, "    projectedRows := terminalBindingProjectedRows").unwrap();
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

fn render_selective_binding(
    out: &mut String,
    counter: &mut usize,
    export: &FixedPointProblemExport,
) -> Result<(), ExportError> {
    let binding = export.binding();
    let nat_items = |values: &[usize]| {
        values
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
    };
    write_chunked_list_def(
        out,
        "selectiveBindingRequestedSourceRows",
        "Nat",
        &nat_items(binding.requested_source_rows()),
    );
    write_chunked_list_def(
        out,
        "selectiveBindingClosureSourceRows",
        "Nat",
        &nat_items(binding.closure_source_rows()),
    );
    write_chunked_list_def(
        out,
        "selectiveBindingAdditionalSourceRows",
        "Nat",
        &nat_items(binding.additional_source_rows()),
    );
    write_chunked_list_def(
        out,
        "selectiveBindingEmittedRows",
        "Nat",
        &nat_items(binding.emitted_rows()),
    );
    let retained = binding
        .retained_rows()
        .iter()
        .map(|row| {
            format!(
                "{{ sourceRow := {}, emittedRow := {}, stageOccurrence := {} }}",
                row.source_row(),
                row.emitted_row(),
                lean_option(row.stage_occurrence())
            )
        })
        .collect::<Vec<_>>();
    write_chunked_list_def(out, "selectiveBindingRetainedRows", "RetainedRowBinding", &retained);
    let rewrites = binding
        .rewrites()
        .iter()
        .map(|rewrite| {
            let mut item = String::new();
            write!(
                item,
                "{{ rewriteId := {}, kind := {}, sourceRows := ",
                rewrite.rewrite_id(),
                lean_string(rewrite_kind_name(rewrite.kind()))
            )
            .unwrap();
            render_ranges(&mut item, &rewrite.source_rows());
            write!(
                item,
                ", emittedRows := {}, stageOccurrence := {} }}",
                lean_range(rewrite.emitted_rows()),
                lean_option(rewrite.stage_occurrence())
            )
            .unwrap();
            item
        })
        .collect::<Vec<_>>();
    write_chunked_list_def(out, "selectiveBindingRewrites", "RewriteBinding", &rewrites);
    let projected = binding
        .projected_rows()
        .iter()
        .map(|row| {
            let mut hoisted = String::new();
            let mut item = String::new();
            write!(
                item,
                "{{ emittedRow := {}, runIndex := {}, family := {}, arm := {}, ports := [",
                row.emitted_row(),
                row.run_index(),
                lean_string(emitted_family_name(row.family())),
                lean_option(row.arm())
            )
            .unwrap();
            for (port_index, port) in row.ports().iter().enumerate() {
                render_port(&mut hoisted, counter, &mut item, port)?;
                write!(item, "{}", separator(port_index, row.ports().len())).unwrap();
            }
            write!(item, "] }}").unwrap();
            Ok((hoisted, item))
        })
        .collect::<Result<Vec<_>, ExportError>>()?;
    write_chunked_list_def_grouped(out, "selectiveBindingProjectedRows", "FinalRow", &projected);
    writeln!(out, "def selectiveBinding : SelectiveBinding :=").unwrap();
    writeln!(out, "  {{ branch := {}", lean_string(binding.branch())).unwrap();
    writeln!(out, "    requestedSourceRows := selectiveBindingRequestedSourceRows").unwrap();
    writeln!(out, "    closureSourceRows := selectiveBindingClosureSourceRows").unwrap();
    writeln!(out, "    additionalSourceRows := selectiveBindingAdditionalSourceRows").unwrap();
    writeln!(out, "    retainedRows := selectiveBindingRetainedRows").unwrap();
    writeln!(out, "    rewrites := selectiveBindingRewrites").unwrap();
    writeln!(out, "    emittedRows := selectiveBindingEmittedRows").unwrap();
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
    writeln!(out, "    projectedRows := selectiveBindingProjectedRows }}\n").unwrap();
    Ok(())
}

fn render_port(
    defs: &mut String,
    counter: &mut usize,
    out: &mut String,
    port: &SelectiveProjectedPort,
) -> Result<(), ExportError> {
    let explicit = port
        .explicit()
        .iter()
        .map(|term| format!("({}, {})", term.column(), term.coefficient().as_canonical_u64()))
        .collect::<Vec<_>>();
    write!(
        out,
        "{{ explicit := {}",
        inline_or_hoist(defs, counter, "(Nat × Nat)", &explicit)
    )
    .unwrap();
    let runs = port
        .geometric_runs()
        .iter()
        .map(|run| {
            format!(
                "{{ columnStart := {}, length := {}, initial := {}, ratio := {} }}",
                run.column_start(),
                run.length(),
                run.initial().as_canonical_u64(),
                run.ratio().as_canonical_u64()
            )
        })
        .collect::<Vec<_>>();
    write!(
        out,
        ", geometricRuns := {}",
        inline_or_hoist(defs, counter, "FinalGeometricRun", &runs)
    )
    .unwrap();
    write!(out, ", seededBlocks := [").unwrap();
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
