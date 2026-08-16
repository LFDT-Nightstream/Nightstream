use std::collections::BTreeSet;

use neo_fold_clean::frontends::nebula::f_prime::{
    NebulaFPrimeBranch, NebulaFPrimeChainBuilder, NebulaFPrimeConstraintSourceAudit, NebulaFPrimePreprocessing,
    NebulaFPrimeRelation,
};
use neo_fold_clean::frontends::nebula::layout::NebulaParams;
use neo_fold_clean::frontends::nebula::plan::NebulaPlan;
use neo_fold_clean::frontends::nebula::trace::Memory;
use neo_fold_clean::frontends::r1cs_f_prime::SparseR1cs;
use neo_fold_clean::paper::params::Params;
use neo_math::F;
use nightstream_constraint_exporter::{
    analyze_nebula_branch, bind_nebula_source_assignment, export_nebula_problem, export_sparse_problem,
    load_nebula_source_assignment, nebula_family_census, refine_nebula_with_cvc5, render_bound_artifact_lean,
    sparse_family_census, validate_paper_obligation_ledger, ExportRequest, FixedPointFamilySearch,
    NebulaPhysicalSourceArm,
};
use p3_field::PrimeCharacteristicRing;
use recursive_constraint_minimizer::{
    derive_scalar_certificate, render_query, run_cvc5, validate_scalar_certificate, Conclusion, Problem, Scope,
    Selection, SolverConfig, SolverStatus, Term,
};

fn minimal_params() -> Params {
    nightstream_constraint_exporter::campaign_profile_params()
}

fn source_audit() -> neo_fold_clean::frontends::nebula::f_prime::NebulaFPrimeConstraintSourceAudit {
    let params = minimal_params();
    let memory = NebulaParams::new(0, 0, 1, 2, 1).expect("one-step memory profile");
    let plan = NebulaPlan::new(memory, vec![7], [0xD9; 32], params.kappa() as usize).expect("Nebula plan");
    NebulaFPrimeRelation::audit_fixed_point_constraint_sources(&params, &plan)
        .expect("discover exact Nebula source arms")
}

fn accepted_source_assignments(
    segment_count: usize,
) -> (NebulaFPrimeConstraintSourceAudit, Vec<(NebulaFPrimeBranch, Vec<F>)>) {
    assert!(segment_count > 0);
    let params = minimal_params();
    let memory_params = NebulaParams::new(0, 0, 1, 2, segment_count as u64).expect("one-step memory profile");
    let rom = [7];
    let plan = NebulaPlan::new(memory_params, rom.to_vec(), [0xDA; 32], params.kappa() as usize).expect("Nebula plan");
    let audit = NebulaFPrimeRelation::audit_fixed_point_constraint_sources(&params, &plan)
        .expect("discover exact Nebula source arms");
    let prep = NebulaFPrimePreprocessing::new_seeded(params, plan, 0xDA00_0001).expect("fixed Nebula preprocessing");
    let mut memory = Memory::new(memory_params, &rom).expect("memory");
    let mut chain = NebulaFPrimeChainBuilder::new(&prep);
    let mut accepted = Vec::with_capacity(segment_count);
    for index in 0..segment_count {
        let trace = {
            let mut segment = memory.begin_segment().expect("segment");
            segment.write(true, 0, 5 + index as u32).expect("RAM write");
            segment.finish().expect("accepted trace")
        };
        let witnesses = chain
            .append_segment_with_constraint_witness_audit(&trace)
            .expect("accepted Nebula step");
        assert_eq!(witnesses.len(), 1);
        accepted.push((witnesses[0].branch(), witnesses[0].source_assignment().to_vec()));
    }
    (audit, accepted)
}

fn accepted_base_source_assignment() -> (NebulaFPrimeConstraintSourceAudit, Vec<F>) {
    let (audit, mut accepted) = accepted_source_assignments(1);
    let (branch, assignment) = accepted.pop().expect("one accepted source assignment");
    assert_eq!(branch, NebulaFPrimeBranch::Base);
    (audit, assignment)
}

#[test]
fn accepted_base_assignment_replays_against_the_exact_source_arm() {
    let (audit, assignment) = accepted_base_source_assignment();
    audit
        .arm(NebulaFPrimeBranch::Base)
        .is_satisfied_by(&assignment)
        .expect("accepted assignment must satisfy the exact source arm");

    const PROFILE: &str = "nebula-saved-assignment-control";
    let checked = bind_nebula_source_assignment(&audit, NebulaFPrimeBranch::Base, PROFILE, &assignment)
        .expect("bind the exact accepted assignment");
    assert_eq!(checked.profile(), PROFILE);
    assert_eq!(checked.source_arm(), NebulaPhysicalSourceArm::Base);
    assert!(checked.source_artifact_digest().starts_with("sha256:"));
    assert_eq!(checked.values(), assignment);

    let json = checked.to_json_vec().expect("serialize checked assignment");
    let loaded = load_nebula_source_assignment(&audit, NebulaFPrimeBranch::Base, PROFILE, &json)
        .expect("load and replay the saved assignment");
    assert_eq!(loaded, checked);

    let mut changed: serde_json::Value = serde_json::from_slice(&json).expect("assignment JSON");
    changed["source_artifact_digest"] = "sha256:changed".into();
    let changed = serde_json::to_vec(&changed).expect("changed assignment JSON");
    assert!(load_nebula_source_assignment(&audit, NebulaFPrimeBranch::Base, PROFILE, &changed).is_err());

    let mut changed: serde_json::Value = serde_json::from_slice(&json).expect("assignment JSON");
    changed["values"][0] = "0".into();
    let changed = serde_json::to_vec(&changed).expect("changed assignment JSON");
    assert!(load_nebula_source_assignment(&audit, NebulaFPrimeBranch::Base, PROFILE, &changed).is_err());

    let mut changed: serde_json::Value = serde_json::from_slice(&json).expect("assignment JSON");
    changed["values"][1] = "00".into();
    let changed = serde_json::to_vec(&changed).expect("changed assignment JSON");
    assert!(load_nebula_source_assignment(&audit, NebulaFPrimeBranch::Base, PROFILE, &changed).is_err());

    assert!(load_nebula_source_assignment(&audit, NebulaFPrimeBranch::Base, "different-profile", &json,).is_err());
}

#[test]
#[ignore = "two accepted CPU lifecycle steps are a manual recursive-arm assignment capture"]
fn accepted_assignments_cover_and_satisfy_both_physical_source_arms() {
    let (audit, accepted) = accepted_source_assignments(2);
    assert_eq!(
        accepted
            .iter()
            .map(|(branch, _)| *branch)
            .collect::<Vec<_>>(),
        [NebulaFPrimeBranch::Base, NebulaFPrimeBranch::BootstrapRecursive]
    );
    for (branch, assignment) in &accepted {
        audit
            .arm(*branch)
            .is_satisfied_by(assignment)
            .unwrap_or_else(|error| panic!("accepted {branch:?} assignment failed exact source replay: {error}"));
    }
    audit
        .arm(NebulaFPrimeBranch::Recursive)
        .is_satisfied_by(&accepted[1].1)
        .expect("the accepted bootstrap assignment must satisfy the shared recursive source arm");

    const PROFILE: &str = "nebula-saved-recursive-assignment";
    let checked =
        bind_nebula_source_assignment(&audit, NebulaFPrimeBranch::BootstrapRecursive, PROFILE, &accepted[1].1)
            .expect("bind the accepted bootstrap assignment");
    assert_eq!(checked.source_arm(), NebulaPhysicalSourceArm::Recursive);
    let json = checked
        .to_json_vec()
        .expect("serialize recursive assignment");
    load_nebula_source_assignment(&audit, NebulaFPrimeBranch::Recursive, PROFILE, &json)
        .expect("the recursive branch must accept the shared physical-arm assignment");
}

#[test]
#[ignore = "two accepted CPU lifecycle steps; writes the checked recursive assignment for bar 4"]
fn capture_and_save_the_bootstrap_recursive_assignment() {
    const PROFILE: &str = "nebula-saved-recursive-assignment";
    const OUTPUT: &str = "evidence/nebula-recursive-assignment.json";

    let (audit, accepted) = accepted_source_assignments(2);
    assert_eq!(
        accepted
            .iter()
            .map(|(branch, _)| *branch)
            .collect::<Vec<_>>(),
        [NebulaFPrimeBranch::Base, NebulaFPrimeBranch::BootstrapRecursive]
    );
    let checked =
        bind_nebula_source_assignment(&audit, NebulaFPrimeBranch::BootstrapRecursive, PROFILE, &accepted[1].1)
            .expect("bind the accepted bootstrap assignment");
    assert_eq!(checked.source_arm(), NebulaPhysicalSourceArm::Recursive);
    let json = checked
        .to_json_vec()
        .expect("serialize recursive assignment");
    load_nebula_source_assignment(&audit, NebulaFPrimeBranch::Recursive, PROFILE, &json)
        .expect("the recursive branch must accept the shared physical-arm assignment");
    let path = format!("{}/../{OUTPUT}", env!("CARGO_MANIFEST_DIR"));
    std::fs::create_dir_all(
        std::path::Path::new(&path)
            .parent()
            .expect("evidence directory"),
    )
    .expect("create evidence directory");
    std::fs::write(&path, &json).expect("write the checked recursive assignment");
    eprintln!("saved checked recursive assignment: {path} ({} bytes)", json.len());
}

#[test]
fn installed_cvc5_runs_one_exact_nebula_refinement_iteration() {
    let (audit, assignment) = accepted_base_source_assignment();
    let branch = NebulaFPrimeBranch::Base;
    let arm = audit.arm(branch);
    let census = nebula_family_census(&audit, branch).expect("complete reviewed Nebula family ownership");
    let family = census
        .iter()
        .find(|family| family.name() == "fprime.base.step.initial")
        .expect("small base control family");
    let selection = Selection::Family(family.name().to_owned());
    let report = refine_nebula_with_cvc5(
        &audit,
        branch,
        &assignment,
        ExportRequest {
            profile: "nebula-reduced-cvc5-control".to_owned(),
            scope: Scope::Branch,
            public_input_count: arm.m_in,
            source_rows: family.source_rows().to_vec(),
            complete_families: vec![family.name().to_owned()],
        },
        &selection,
        &SolverConfig {
            timeout_ms: 60_000,
            ..SolverConfig::default()
        },
        1,
    )
    .expect("run one exact Nebula refinement iteration");

    eprintln!(
        "exact Nebula cvc5 control: status={:?} conclusion={:?} pending={:?}",
        report.refinement().solver_run.status,
        report.refinement().conclusion,
        report.refinement().pending_retained_row,
    );
    assert_eq!(report.refinement().iterations, 1);
    assert_eq!(report.refinement().solver_run.status, SolverStatus::Sat);
    assert_eq!(report.refinement().conclusion, Conclusion::Inconclusive);
    assert!(report.refinement().pending_retained_row.is_some());
    assert_eq!(report.source_export().binding().branch(), "nebula_base");
    assert_eq!(report.source_export().binding().final_rows(), audit.verifier_rows());
}

#[test]
fn nebula_arms_have_complete_exportable_family_censuses() {
    let audit = source_audit();
    assert!(!audit.application_bound());
    assert!(std::ptr::eq(
        audit.arm(NebulaFPrimeBranch::BootstrapRecursive),
        audit.arm(NebulaFPrimeBranch::Recursive),
    ));

    for branch in [
        NebulaFPrimeBranch::Base,
        NebulaFPrimeBranch::BootstrapRecursive,
        NebulaFPrimeBranch::Recursive,
    ] {
        let arm = audit.arm(branch);
        let census = nebula_family_census(&audit, branch).expect("complete reviewed Nebula family ownership");
        assert!(!census.is_empty());
        assert_eq!(
            census
                .iter()
                .map(|family| family.source_rows().len())
                .sum::<usize>(),
            arm.n,
        );
        assert!(census
            .windows(2)
            .all(|pair| pair[0].name() < pair[1].name()));
    }
}

#[test]
fn paper_obligation_rows_exist_in_the_exact_nebula_census() {
    let audit = source_audit();
    let base = nebula_family_census(&audit, NebulaFPrimeBranch::Base).expect("complete reviewed base family ownership");
    let recursive = nebula_family_census(&audit, NebulaFPrimeBranch::Recursive)
        .expect("complete reviewed recursive family ownership");
    let base_names = base.iter().map(|family| family.name()).collect::<Vec<_>>();
    let recursive_names = recursive
        .iter()
        .map(|family| family.name())
        .collect::<Vec<_>>();

    validate_paper_obligation_ledger(&base_names, &recursive_names)
        .expect("every mapped paper obligation must cite an exact reviewed owner");
}

#[test]
fn nebula_family_export_binds_exact_final_rows() {
    let audit = source_audit();
    let branch = NebulaFPrimeBranch::Recursive;
    let arm = audit.arm(branch);
    let mapping = &audit.compiler_audit().rows().arms()[branch.relation_arm_index()];
    let rewritten_rows = mapping
        .source_runs()
        .iter()
        .filter(|run| run.disposition().rewrite_id().is_some())
        .flat_map(|run| run.source_rows())
        .collect::<BTreeSet<_>>();
    let census = nebula_family_census(&audit, branch).expect("complete reviewed Nebula family ownership");
    let family = census
        .iter()
        .filter(|family| {
            family
                .source_rows()
                .iter()
                .any(|row| rewritten_rows.contains(row))
        })
        .min_by_key(|family| family.source_rows().len())
        .expect("one rewrite-owned Nebula family");
    let export = export_nebula_problem(
        &audit,
        branch,
        ExportRequest {
            profile: "nebula-minimal-test".to_owned(),
            scope: Scope::Branch,
            public_input_count: arm.m_in,
            source_rows: family.source_rows().to_vec(),
            complete_families: vec![family.name().to_owned()],
        },
    )
    .expect("export one complete Nebula family");

    assert_eq!(export.problem().rows.len(), family.source_rows().len());
    assert!(export
        .problem()
        .rows
        .iter()
        .all(|row| row.family == family.name()));
    let binding = export.binding();
    assert_eq!(binding.branch(), "nebula_recursive");
    assert_eq!(binding.requested_source_rows(), family.source_rows());
    assert!(!binding.rewrites().is_empty());
    assert_eq!(binding.final_rows(), audit.verifier_rows());
    assert_eq!(binding.final_columns(), audit.verifier_columns());
    assert_eq!(
        binding.final_public_input_count(),
        audit.compiler_audit().layout().public_input_len(),
    );
    assert_eq!(
        binding
            .projected_rows()
            .iter()
            .map(|row| row.emitted_row())
            .collect::<Vec<_>>(),
        binding.emitted_rows(),
    );
    assert!(binding.final_plan_digest().starts_with("sha256:"));
    assert!(binding.projected_slice_digest().starts_with("sha256:"));
}

#[test]
fn compact_seeded_row_slice_binds_and_renders_without_dense_expansion() {
    let audit = source_audit();
    let branch = NebulaFPrimeBranch::Recursive;
    let arm = audit.arm(branch);
    let mapping = &audit.compiler_audit().rows().arms()[branch.relation_arm_index()];
    let source_block = [&arm.a, &arm.b, &arm.c]
        .into_iter()
        .flat_map(|matrix| matrix.seeded_phi81_blocks())
        .next()
        .expect("recursive source arm has one compact seeded block")
        .clone();
    let source_row = source_block.row_start();
    let owner = mapping
        .source_runs()
        .iter()
        .find(|run| run.source_rows().contains(&source_row))
        .expect("seeded source row has one compiler owner");
    assert_eq!(
        owner.disposition(),
        neo_fold_clean::frontends::r1cs_f_prime::SelectiveSourceRowDisposition::Retained,
    );
    let emitted_row = owner
        .emitted_start()
        .expect("retained row has an emitted start")
        + source_row
        - owner.source_rows().start;
    let export = export_nebula_problem(
        &audit,
        branch,
        ExportRequest {
            profile: "nebula-compact-seeded-row-control".to_owned(),
            scope: Scope::Branch,
            public_input_count: arm.m_in,
            source_rows: vec![source_row],
            complete_families: Vec::new(),
        },
    )
    .expect("bind one compact seeded row without expanding final coefficients");

    assert_eq!(export.binding().emitted_rows(), [emitted_row]);
    let [projected] = export.binding().projected_rows() else {
        panic!("one projected seeded row");
    };
    let blocks = projected
        .ports()
        .iter()
        .flat_map(|port| port.seeded_blocks())
        .collect::<Vec<_>>();
    assert!(!blocks.is_empty());
    assert!(blocks.iter().all(|block| {
        (block.row_start()..block.row_end()).contains(&emitted_row)
            && block
                .validate_matrix_shape(export.binding().final_rows(), export.binding().final_columns())
                .is_ok()
    }));
    let rebound = blocks
        .iter()
        .find(|block| block.chunk_seeds_by_row() == source_block.chunk_seeds_by_row())
        .expect("the projected port retains the exact source seed schedule");
    assert_eq!(rebound.row_start(), emitted_row);
    assert_eq!(rebound.word_width(), source_block.word_width());
    assert_eq!(rebound.kappa(), source_block.kappa());
    assert_eq!(rebound.message_cols(), source_block.message_cols());
    assert_eq!(rebound.chunk_size(), source_block.chunk_size());
    assert_eq!(
        rebound.has_superneo_transformed_columns(),
        source_block.has_superneo_transformed_columns(),
    );

    let lean = render_bound_artifact_lean(&export, "Generated.CompactSeededRow")
        .expect("render exact compact seeded metadata for Lean");
    assert!(lean.contains("seededBlocks := ["));
    assert!(lean.contains(&format!("rowStart := {emitted_row}")));
    assert!(lean.contains("chunkSeedsByRow := [["));
    assert!(lean.contains("superneoTransformedColumns :="));
}

#[test]
fn emitter_order_constant_affine_run_census_is_exact() {
    let audit = source_audit();
    for run in audit.compiler_audit().rows().emitted_runs() {
        eprintln!(
            "selective family={:?} arm={:?} rows={}",
            run.family(),
            run.arm(),
            run.emitted_rows().len(),
        );
    }
    let projected = audit
        .audit_selective_rows(&[])
        .expect("measure the exact final emitter stream");
    assert_eq!(projected.explicit_run_census().len(), 13);
    let terms = projected
        .explicit_run_census()
        .iter()
        .map(|port| port.term_count())
        .sum::<usize>();
    let run_terms = projected
        .explicit_run_census()
        .iter()
        .map(|port| port.affine_run_terms())
        .sum::<usize>();
    let runs = projected
        .explicit_run_census()
        .iter()
        .map(|port| port.affine_run_count())
        .sum::<usize>();
    let literals = projected
        .explicit_run_census()
        .iter()
        .map(|port| port.literal_count())
        .sum::<usize>();
    let records = projected
        .explicit_run_census()
        .iter()
        .map(|port| port.record_count())
        .sum::<usize>();
    assert_eq!(run_terms + literals, terms);
    assert!(records <= terms);
    for (port, census) in projected.explicit_run_census().iter().enumerate() {
        eprintln!(
            "selective emitter port={port} terms={} run_terms={} runs={} literals={} records={}",
            census.term_count(),
            census.affine_run_terms(),
            census.affine_run_count(),
            census.literal_count(),
            census.record_count(),
        );
    }
    eprintln!(
        "selective emitter affine census: terms={terms} run_terms={run_terms} runs={runs} literals={literals} records={records}"
    );
}

#[test]
fn centered_unit_source_to_empty_rewrite_binds_to_the_final_plan() {
    let audit = source_audit();
    let branch = NebulaFPrimeBranch::Recursive;
    let rewrite = audit
        .compiler_audit()
        .rows()
        .rewrites()
        .iter()
        .find(|rewrite| {
            rewrite.arm() == branch.relation_arm_index()
                && rewrite.kind() == neo_fold_clean::frontends::r1cs_f_prime::SelectiveRewriteKind::CenteredUnit
                && rewrite.emitted_rows().is_empty()
        })
        .expect("one centered-unit source-to-empty rewrite");
    let source_rows = rewrite
        .source_rows()
        .iter()
        .flat_map(|range| range.clone())
        .collect::<Vec<_>>();
    let export = export_nebula_problem(
        &audit,
        branch,
        ExportRequest {
            profile: "nebula-centered-unit-source-to-empty-test".to_owned(),
            scope: Scope::Branch,
            public_input_count: audit.arm(branch).m_in,
            source_rows: source_rows.clone(),
            complete_families: Vec::new(),
        },
    )
    .expect("bind one centered-unit source-to-empty rewrite");

    assert_eq!(export.binding().requested_source_rows(), source_rows);
    let [bound_rewrite] = export.binding().rewrites() else {
        panic!("one exact bound rewrite");
    };
    assert_eq!(bound_rewrite.rewrite_id(), rewrite.id().index());
    assert_eq!(bound_rewrite.kind(), rewrite.kind());
    assert!(bound_rewrite.emitted_rows().is_empty());
    assert!(export.binding().emitted_rows().is_empty());
    assert!(export.binding().projected_rows().is_empty());
    assert_eq!(export.binding().final_rows(), audit.verifier_rows());
    assert_eq!(export.binding().final_columns(), audit.verifier_columns());
}

#[test]
fn sparse_export_matches_each_nebula_source_matrix() {
    let audit = source_audit();
    for (arm_index, arm) in audit.physical_arms().iter().enumerate() {
        let census = sparse_family_census(arm).expect("complete physical-stage census");
        let full = export_sparse_problem(
            arm,
            ExportRequest {
                profile: format!("nebula-source-conformance-full-{arm_index}"),
                scope: Scope::Branch,
                public_input_count: arm.m_in,
                source_rows: (0..arm.n).collect(),
                complete_families: census
                    .iter()
                    .map(|family| family.name().to_owned())
                    .collect(),
            },
        )
        .expect("export every source row");
        assert_export_matches_matrices(arm, &full);

        let mut sampled_rows = (0..arm.n).step_by(97).collect::<Vec<_>>();
        if sampled_rows.last().copied() != Some(arm.n - 1) {
            sampled_rows.push(arm.n - 1);
        }
        let sampled = export_sparse_problem(
            arm,
            ExportRequest {
                profile: format!("nebula-source-conformance-sampled-{arm_index}"),
                scope: Scope::Branch,
                public_input_count: arm.m_in,
                source_rows: sampled_rows,
                complete_families: Vec::new(),
            },
        )
        .expect("export a strict source-row sample");
        assert_eq!(sampled.source.artifact_digest, full.source.artifact_digest);
        assert_export_matches_matrices(arm, &sampled);
    }
}

#[test]
fn recursive_pi_rlc_padding_has_an_exact_scalar_certificate() {
    const CANDIDATE: &str = "nifs.pi_rlc.verify.padding.y_ring";
    const PI_CCS_SUPPORT: &str = "nifs.pi_ccs.padded_row.canonicality";
    const PI_DEC_SUPPORT: &str = "nifs.pi_dec.verify";

    let audit = source_audit();
    let branch = NebulaFPrimeBranch::Recursive;
    let arm = audit.arm(branch);
    let census = nebula_family_census(&audit, branch).expect("complete reviewed Nebula family ownership");
    let family = |name: &str| {
        census
            .iter()
            .find(|family| family.name() == name)
            .unwrap_or_else(|| panic!("missing exact family {name}"))
    };
    let source_rows = [CANDIDATE, PI_CCS_SUPPORT, PI_DEC_SUPPORT]
        .into_iter()
        .flat_map(|name| family(name).source_rows().iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let candidate_export = export_nebula_problem(
        &audit,
        branch,
        ExportRequest {
            profile: "nebula-recursive-exact-duplicate-control".to_owned(),
            scope: Scope::Branch,
            public_input_count: arm.m_in,
            source_rows: family(CANDIDATE).source_rows().to_vec(),
            complete_families: vec![CANDIDATE.to_owned()],
        },
    )
    .expect("bind the exact duplicate candidate rows");
    assert_eq!(candidate_export.binding().retained_rows().len(), 3_640);
    assert_eq!(candidate_export.binding().rewrites().len(), 280);
    assert_eq!(candidate_export.binding().closure_source_rows().len(), 3_920);
    assert!(candidate_export
        .binding()
        .additional_source_rows()
        .is_empty());
    assert_eq!(candidate_export.binding().emitted_rows().len(), 3_640);
    assert!(candidate_export.binding().rewrites().iter().all(|rewrite| {
        rewrite.kind() == neo_fold_clean::frontends::r1cs_f_prime::SelectiveRewriteKind::LinearDefinition
            && rewrite.emitted_rows().is_empty()
    }));
    let export = export_nebula_problem(
        &audit,
        branch,
        ExportRequest {
            profile: "nebula-recursive-exact-duplicate-control".to_owned(),
            scope: Scope::Branch,
            public_input_count: arm.m_in,
            source_rows,
            complete_families: vec![CANDIDATE.to_owned()],
        },
    )
    .expect("export the exact duplicate candidate and support rows");
    let selection = Selection::Family(CANDIDATE.to_owned());
    let certificate = derive_scalar_certificate(export.problem(), &selection)
        .expect("derive the scalar certificate")
        .expect("every candidate row is in the retained scalar span");
    validate_scalar_certificate(export.problem(), &certificate).expect("replay the scalar certificate");

    assert_eq!(certificate.rows.len(), family(CANDIDATE).source_rows().len());
    let mut pi_ccs_support = 0;
    let mut pi_dec_support = 0;
    for row in &certificate.rows {
        let [support] = row.support.as_slice() else {
            panic!("each exact duplicate must use one support row");
        };
        assert_eq!(support.coefficient, "1");
        let support_row = export
            .problem()
            .rows
            .binary_search_by_key(&support.source_index, |row| row.source_index)
            .map(|index| &export.problem().rows[index])
            .expect("certificate support belongs to the exported source slice");
        match support_row.family.as_str() {
            PI_CCS_SUPPORT => pi_ccs_support += 1,
            PI_DEC_SUPPORT => pi_dec_support += 1,
            other => panic!("unexpected exact-duplicate support family {other}"),
        }
    }
    assert_eq!((pi_ccs_support, pi_dec_support), (3_640, 280));
}

#[test]
fn installed_cvc5_finds_the_recursive_pi_rlc_padding_candidate_unsat() {
    const CANDIDATE: &str = "nifs.pi_rlc.verify.padding.y_ring";
    const PI_CCS_SUPPORT: &str = "nifs.pi_ccs.padded_row.canonicality";
    const PI_DEC_SUPPORT: &str = "nifs.pi_dec.verify";

    let audit = source_audit();
    let branch = NebulaFPrimeBranch::Recursive;
    let arm = audit.arm(branch);
    let census = nebula_family_census(&audit, branch).expect("complete reviewed Nebula family ownership");
    let family = |name: &str| {
        census
            .iter()
            .find(|family| family.name() == name)
            .unwrap_or_else(|| panic!("missing exact family {name}"))
    };
    let source_rows = [CANDIDATE, PI_CCS_SUPPORT, PI_DEC_SUPPORT]
        .into_iter()
        .flat_map(|name| family(name).source_rows().iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let export = export_nebula_problem(
        &audit,
        branch,
        ExportRequest {
            profile: "nebula-recursive-cvc5-duplicate-control".to_owned(),
            scope: Scope::Branch,
            public_input_count: arm.m_in,
            source_rows,
            complete_families: vec![CANDIDATE.to_owned()],
        },
    )
    .expect("export the exact duplicate candidate and support rows");
    let query = render_query(export.problem(), &Selection::Family(CANDIDATE.to_owned()))
        .expect("render exact recursive duplicate query");
    let run = run_cvc5(&query, &SolverConfig::default()).expect("run installed cvc5");

    eprintln!(
        "exact recursive duplicate cvc5 control: status={:?} conclusion={:?} elapsed_ms={}",
        run.status, run.conclusion, run.elapsed_ms,
    );
    assert_eq!(run.status, SolverStatus::Unsat);
    assert_eq!(run.conclusion, Conclusion::RedundancyCandidate);
}

#[test]
fn nebula_branch_driver_records_every_family_on_failure() {
    let audit = source_audit();
    let branch = NebulaFPrimeBranch::Base;
    let census = nebula_family_census(&audit, branch).expect("complete reviewed Nebula family ownership");
    let report = analyze_nebula_branch(
        &audit,
        branch,
        &[],
        "nebula-exhaustive-inconclusive-test",
        &SolverConfig::default(),
        1,
    )
    .expect("build exact exhaustive Nebula branch ledger");

    assert_eq!(report.profile(), "nebula-exhaustive-inconclusive-test");
    assert_eq!(report.branch(), branch);
    assert!(report.source_artifact_digest().starts_with("sha256:"));
    assert!(report.final_plan_digest().starts_with("sha256:"));
    assert_eq!(report.source_rows(), audit.arm(branch).n);
    assert_eq!(report.source_columns(), audit.arm(branch).m);
    assert_eq!(report.source_public_columns(), audit.arm(branch).m_in);
    assert_eq!(report.final_rows(), audit.verifier_rows());
    assert_eq!(report.final_columns(), audit.verifier_columns());
    assert_eq!(
        report.final_public_columns(),
        audit.compiler_audit().layout().public_input_len(),
    );
    assert_eq!(report.families().len(), census.len());
    for (record, expected) in report.families().iter().zip(&census) {
        assert_eq!(record.name(), expected.name());
        assert_eq!(record.source_rows(), expected.source_rows());
        assert_eq!(record.search().family(), expected.name());
        assert!(matches!(
            record.search(),
            FixedPointFamilySearch::Inconclusive { reason, .. }
                if reason.contains("background assignment")
        ));
    }
}

fn assert_export_matches_matrices(arm: &SparseR1cs, problem: &Problem) {
    let assignment = (0..arm.m)
        .map(|column| {
            if column == 0 {
                F::ONE
            } else {
                F::from_u64(
                    (column as u64)
                        .wrapping_mul(0x9E37_79B9)
                        .wrapping_add(0xD1B5_4A32),
                )
            }
        })
        .collect::<Vec<_>>();
    let mut az = vec![F::ZERO; arm.n];
    let mut bz = vec![F::ZERO; arm.n];
    let mut cz = vec![F::ZERO; arm.n];
    arm.a.add_mul_into(&assignment, &mut az, arm.n);
    arm.b.add_mul_into(&assignment, &mut bz, arm.n);
    arm.c.add_mul_into(&assignment, &mut cz, arm.n);

    for row in &problem.rows {
        let source = row.source_index;
        assert_eq!(evaluate(&row.a, &assignment), az[source], "A row {source}");
        assert_eq!(evaluate(&row.b, &assignment), bz[source], "B row {source}");
        assert_eq!(evaluate(&row.c, &assignment), cz[source], "C row {source}");
    }
}

fn evaluate(terms: &[Term], assignment: &[F]) -> F {
    terms.iter().fold(F::ZERO, |sum, term| {
        let coefficient = term
            .coefficient
            .parse::<u64>()
            .expect("canonical field residue");
        sum + F::from_u64(coefficient) * assignment[term.column]
    })
}
