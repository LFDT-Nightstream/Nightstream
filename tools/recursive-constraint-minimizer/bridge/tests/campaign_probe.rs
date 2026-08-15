//! Campaign measurement probes. All probes are `#[ignore]` printers; they
//! change nothing and authorize nothing.

use neo_fold_clean::frontends::nebula::f_prime::{
    NebulaFPrimeBranch, NebulaFPrimeChainBuilder, NebulaFPrimePreprocessing, NebulaFPrimeRelation,
};
use neo_fold_clean::frontends::nebula::layout::NebulaParams;
use neo_fold_clean::frontends::nebula::plan::NebulaPlan;
use neo_fold_clean::frontends::nebula::trace::Memory;
use neo_fold_clean::paper::params::Params;
use neo_math::F;
use neo_params::{goldilocks_paper_b2, NeoParams};
use std::collections::{BTreeMap, BTreeSet};

use nightstream_constraint_exporter::{
    analyze_nebula_branch, export_complete_nebula_problem, find_exclusive_column_witness, nebula_family_census,
    refine_nebula_with_cvc5, render_complete_bound_artifact_modules, render_removal_counterexample_lean, ExportRequest,
    FixedPointFamilySearch,
};
use recursive_constraint_minimizer::{Scope, Selection, SolverConfig, SolverMode};

fn minimal_params() -> Params {
    let inner = NeoParams::new(
        goldilocks_paper_b2::Q,
        goldilocks_paper_b2::ETA as u32,
        goldilocks_paper_b2::D as u32,
        1,
        goldilocks_paper_b2::M,
        goldilocks_paper_b2::B_BASE,
        2,
        1,
        goldilocks_paper_b2::EXTENSION_DEGREE,
        1,
    )
    .expect("minimal parameters satisfy the exact RLC guard");
    Params::test_only_from_neo_params(inner)
}

fn accepted_base() -> (
    neo_fold_clean::frontends::nebula::f_prime::NebulaFPrimeConstraintSourceAudit,
    Vec<F>,
) {
    let params = minimal_params();
    let memory_params = NebulaParams::new(0, 0, 1, 2, 1).expect("one-step memory profile");
    let rom = [7];
    let plan = NebulaPlan::new(memory_params, rom.to_vec(), [0xDA; 32], params.kappa() as usize).expect("Nebula plan");
    let audit = NebulaFPrimeRelation::audit_fixed_point_constraint_sources(&params, &plan)
        .expect("discover exact Nebula source arms");
    let prep = NebulaFPrimePreprocessing::new_seeded(params, plan, 0xDA00_0001).expect("fixed Nebula preprocessing");
    let mut memory = Memory::new(memory_params, &rom).expect("memory");
    let mut chain = NebulaFPrimeChainBuilder::new(&prep);
    let trace = {
        let mut segment = memory.begin_segment().expect("segment");
        segment.write(true, 0, 5).expect("RAM write");
        segment.finish().expect("accepted trace")
    };
    let witnesses = chain
        .append_segment_with_constraint_witness_audit(&trace)
        .expect("accepted Nebula step");
    assert_eq!(witnesses[0].branch(), NebulaFPrimeBranch::Base);
    let assignment = witnesses[0].source_assignment().to_vec();
    (audit, assignment)
}

#[test]
#[ignore = "campaign census probe; run with --ignored --nocapture"]
fn print_base_branch_search_census() {
    let (audit, assignment) = accepted_base();
    let report = analyze_nebula_branch(
        &audit,
        NebulaFPrimeBranch::Base,
        &assignment,
        "campaign-base-census-probe",
        &SolverConfig {
            mode: SolverMode::Gb,
            timeout_ms: 60_000,
            ..SolverConfig::default()
        },
        24,
    )
    .expect("build the base branch search ledger");

    let mut redundant = 0usize;
    let mut counterexample = 0usize;
    let mut inconclusive = 0usize;
    for record in report.families() {
        let (state, detail) = match record.search() {
            FixedPointFamilySearch::RedundancyCertificate { certificate, .. } => {
                redundant += 1;
                ("certificate", format!("{} rows", certificate.rows.len()))
            }
            FixedPointFamilySearch::RustCounterexampleCandidate { report } => {
                counterexample += 1;
                (
                    "counterexample",
                    format!("iterations {}", report.refinement().iterations),
                )
            }
            FixedPointFamilySearch::Inconclusive { reason, .. } => {
                inconclusive += 1;
                ("inconclusive", reason.clone())
            }
        };
        eprintln!(
            "census family={} rows={} state={state} detail={detail}",
            record.name(),
            record.source_rows().len(),
        );
    }
    eprintln!(
        "census totals: families={} certificate={redundant} counterexample={counterexample} inconclusive={inconclusive}",
        report.families().len(),
    );
}

#[test]
#[ignore = "solver-mode probe for the smallest base family; run with --ignored --nocapture"]
fn print_small_family_solver_mode_probe() {
    let (audit, assignment) = accepted_base();
    let branch = NebulaFPrimeBranch::Base;
    let arm = audit.arm(branch);
    let census = nebula_family_census(&audit, branch).expect("complete reviewed Nebula family ownership");
    let family = census
        .iter()
        .find(|family| family.name() == "fprime.base.step.initial")
        .expect("smallest base control family");
    for mode in [SolverMode::Gb, SolverMode::Split] {
        let started = std::time::Instant::now();
        let outcome = refine_nebula_with_cvc5(
            &audit,
            branch,
            &assignment,
            ExportRequest {
                profile: "campaign-solver-mode-probe".to_owned(),
                scope: Scope::Branch,
                public_input_count: arm.m_in,
                source_rows: family.source_rows().to_vec(),
                complete_families: vec![family.name().to_owned()],
            },
            &Selection::Family(family.name().to_owned()),
            &SolverConfig {
                mode,
                timeout_ms: 300_000,
                ..SolverConfig::default()
            },
            4,
        );
        match outcome {
            Ok(report) => eprintln!(
                "probe mode={mode:?} conclusion={:?} iterations={} slice_rows={} pending={:?} elapsed_ms={}",
                report.refinement().conclusion,
                report.refinement().iterations,
                report.refinement().problem.rows.len(),
                report.refinement().pending_retained_row,
                started.elapsed().as_millis(),
            ),
            Err(error) => eprintln!(
                "probe mode={mode:?} error={error} elapsed_ms={}",
                started.elapsed().as_millis(),
            ),
        }
    }
}

#[test]
#[ignore = "exact-duplicate family sweep; run with --ignored --nocapture"]
fn print_recursive_exact_duplicate_family_sweep() {
    let (audit, _) = accepted_base();
    let branch = NebulaFPrimeBranch::Recursive;
    let export = export_complete_nebula_problem(&audit, branch, "campaign-duplicate-sweep")
        .expect("export the complete recursive arm");
    let problem = export.problem();

    // Key rows by their exact canonical (a, b, c) term lists.
    let mut rows_by_shape = BTreeMap::<String, Vec<usize>>::new();
    for (index, row) in problem.rows.iter().enumerate() {
        rows_by_shape
            .entry(format!("{:?}|{:?}|{:?}", row.a, row.b, row.c))
            .or_default()
            .push(index);
    }

    let census = nebula_family_census(&audit, branch).expect("complete reviewed Nebula family ownership");
    for family in &census {
        let mut support_families = BTreeSet::new();
        let mut uncovered = 0usize;
        for &source_row in family.source_rows() {
            let row = &problem.rows[source_row];
            let shape = format!("{:?}|{:?}|{:?}", row.a, row.b, row.c);
            let duplicates = rows_by_shape
                .get(&shape)
                .expect("every row is keyed")
                .iter()
                .filter(|&&other| problem.rows[other].family != row.family)
                .map(|&other| problem.rows[other].family.clone())
                .collect::<BTreeSet<_>>();
            if duplicates.is_empty() {
                uncovered += 1;
            } else {
                support_families.extend(duplicates);
            }
        }
        if uncovered == 0 {
            eprintln!(
                "duplicate-family candidate={} rows={} supports={:?}",
                family.name(),
                family.source_rows().len(),
                support_families,
            );
        } else if uncovered < family.source_rows().len() {
            eprintln!(
                "partial-duplicate family={} rows={} uncovered={uncovered} supports={:?}",
                family.name(),
                family.source_rows().len(),
                support_families,
            );
        }
    }
    eprintln!("duplicate sweep complete: {} families scanned", census.len());
}

#[test]
#[ignore = "one-shot full-context necessity probe; run with --ignored --nocapture"]
fn print_full_context_necessity_probe() {
    let (audit, assignment) = accepted_base();
    let branch = NebulaFPrimeBranch::Base;
    let arm = audit.arm(branch);
    let started = std::time::Instant::now();
    let outcome = refine_nebula_with_cvc5(
        &audit,
        branch,
        &assignment,
        ExportRequest {
            profile: "campaign-full-context-necessity-probe".to_owned(),
            scope: Scope::Branch,
            public_input_count: arm.m_in,
            source_rows: (0..arm.n).collect(),
            complete_families: vec!["fprime.base.step.initial".to_owned()],
        },
        &Selection::Family("fprime.base.step.initial".to_owned()),
        &SolverConfig {
            mode: SolverMode::Gb,
            timeout_ms: 300_000,
            ..SolverConfig::default()
        },
        1,
    );
    match outcome {
        Ok(report) => eprintln!(
            "full-context probe conclusion={:?} iterations={} slice_rows={} elapsed_ms={}",
            report.refinement().conclusion,
            report.refinement().iterations,
            report.refinement().problem.rows.len(),
            started.elapsed().as_millis(),
        ),
        Err(error) => eprintln!(
            "full-context probe error={error} elapsed_ms={}",
            started.elapsed().as_millis(),
        ),
    }
}

#[test]
#[ignore = "exclusive-column witness census over the base arm; run with --ignored --nocapture"]
fn print_base_exclusive_column_witness_census() {
    let (audit, assignment) = accepted_base();
    let branch = NebulaFPrimeBranch::Base;
    let export = export_complete_nebula_problem(&audit, branch, "campaign-exclusive-column-census")
        .expect("export the complete base arm");
    let problem = export.problem();
    let background = assignment
        .iter()
        .map(|value| p3_field::PrimeField64::as_canonical_u64(value))
        .collect::<Vec<_>>();
    let census = nebula_family_census(&audit, branch).expect("complete reviewed Nebula family ownership");
    let mut found = 0usize;
    for family in &census {
        let started = std::time::Instant::now();
        match find_exclusive_column_witness(problem, &background, family.name()) {
            Ok(Some(witness)) => {
                found += 1;
                eprintln!(
                    "witness family={} column={} delta={} violated={} elapsed_ms={}",
                    family.name(),
                    witness.column(),
                    witness.delta(),
                    witness.violated_rows().len(),
                    started.elapsed().as_millis(),
                );
            }
            Ok(None) => eprintln!(
                "no-exclusive-witness family={} elapsed_ms={}",
                family.name(),
                started.elapsed().as_millis(),
            ),
            Err(error) => eprintln!("witness-error family={} error={error}", family.name()),
        }
    }
    eprintln!("exclusive-column census: {found}/{} families", census.len());
}

#[test]
#[ignore = "emit the base-arm Lean classification batch to NIGHTSTREAM_EMIT_DIR; run with --ignored --nocapture"]
fn emit_base_arm_lean_classification_batch() {
    let emit_dir = std::env::var("NIGHTSTREAM_EMIT_DIR").expect("set NIGHTSTREAM_EMIT_DIR");
    let (audit, assignment) = accepted_base();
    let branch = NebulaFPrimeBranch::Base;
    let export = export_complete_nebula_problem(&audit, branch, "campaign-base-classification-v1")
        .expect("export the complete base arm");
    let problem = export.problem().clone();
    let background = assignment
        .iter()
        .map(|value| p3_field::PrimeField64::as_canonical_u64(value))
        .collect::<Vec<_>>();
    let census = nebula_family_census(&audit, branch).expect("complete reviewed Nebula family ownership");
    let plan = census
        .iter()
        .map(|family| family.name().to_owned())
        .collect::<Vec<_>>();

    let artifact_namespace = "Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact";
    let artifact_modules = render_complete_bound_artifact_modules(&export, artifact_namespace)
        .expect("render the complete base bound artifact modules");
    for module in &artifact_modules {
        let file = module
            .module_name
            .rsplit('.')
            .next()
            .expect("module name has a final segment");
        let path = format!("{emit_dir}/{file}.lean");
        std::fs::write(&path, &module.content).expect("write artifact module");
        eprintln!("emitted {path} bytes={}", module.content.len());
    }

    for family in &census {
        let witness = find_exclusive_column_witness(&problem, &background, family.name())
            .expect("witness search must run")
            .expect("every base family has an exclusive-column witness");
        let module_stem = family
            .name()
            .split('.')
            .map(|part| {
                let mut chars = part.chars();
                match chars.next() {
                    Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                    None => String::new(),
                }
            })
            .collect::<Vec<_>>()
            .join("");
        let namespace =
            format!("Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.{module_stem}Necessity");
        let lean = render_removal_counterexample_lean(
            &problem,
            witness.model(),
            family.name(),
            artifact_namespace,
            artifact_namespace,
            &namespace,
            &plan,
        )
        .expect("render the removal counterexample");
        let path = format!("{emit_dir}/{module_stem}Necessity.lean");
        std::fs::write(&path, &lean).expect("write counterexample module");
        eprintln!("emitted {path} bytes={}", lean.len());
    }
}

#[test]
#[ignore = "export per-family solver problems to NIGHTSTREAM_EMIT_DIR; run with --ignored"]
fn export_solver_campaign_problems() {
    let emit_dir = std::env::var("NIGHTSTREAM_EMIT_DIR").expect("set NIGHTSTREAM_EMIT_DIR");
    let (audit, _) = accepted_base();
    for branch in [NebulaFPrimeBranch::Base, NebulaFPrimeBranch::Recursive] {
        let arm = audit.arm(branch);
        let arm_ref = arm;
        let census = nebula_family_census(&audit, branch).expect("complete reviewed Nebula family ownership");
        let tag = match branch {
            NebulaFPrimeBranch::Base => "base",
            _ => "recursive",
        };
        for family in &census {
            let export = nightstream_constraint_exporter::export_sparse_problem(
                arm_ref,
                ExportRequest {
                    profile: format!("campaign-solver-{tag}-{}", family.name()),
                    scope: Scope::Branch,
                    public_input_count: arm.m_in,
                    source_rows: family.source_rows().to_vec(),
                    complete_families: vec![family.name().to_owned()],
                },
            )
            .expect("export one family problem");
            let path = format!("{emit_dir}/{tag}--{}.json", family.name());
            std::fs::write(&path, serde_json::to_vec(&export).expect("serialize problem")).expect("write problem JSON");
        }
        eprintln!("exported {tag}: {} families", census.len());
    }
}
