//! Drift gate: committed compact-source Lean modules equal fresh emission.
//!
//! The base arm is the pilot: its string-payload artifact must expand to the
//! exact committed literal `BaseBoundArtifact.sourceArtifact`, which the
//! generated assembly pins with a `native_decide` equality theorem.

use neo_fold_clean::frontends::nebula::f_prime::{NebulaFPrimeBranch, NebulaFPrimeRelation};
use nightstream_constraint_exporter::render_compact_source_artifact_modules;

const GENERATED_DIR: &str =
    "../../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/MinimizerCampaign/Generated";

fn campaign_audit() -> neo_fold_clean::frontends::nebula::f_prime::NebulaFPrimeConstraintSourceAudit {
    nightstream_constraint_exporter::campaign_profile_audit().expect("discover campaign source arms")
}

fn assert_modules_match_committed(modules: &[nightstream_constraint_exporter::GeneratedLeanModule]) {
    let mut drifted = Vec::new();
    for module in modules {
        let file = module
            .module_name
            .rsplit('.')
            .next()
            .expect("module name has a final segment");
        let path = format!("{GENERATED_DIR}/{file}.lean");
        let committed = std::fs::read_to_string(&path).unwrap_or_default();
        if committed != module.content {
            std::fs::write(format!("{path}.expected"), &module.content).expect("write expected compact module");
            drifted.push(path);
        }
    }
    assert!(
        drifted.is_empty(),
        "compact source modules drifted; inspect and promote: {drifted:?}"
    );
}

#[test]
fn committed_base_compact_source_modules_match_the_emitter() {
    let audit = campaign_audit();
    let equality = nightstream_constraint_exporter::CommittedEquality {
        namespace: "Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact".to_owned(),
        chunk_prefix: "sourceArtifactRowsChunk".to_owned(),
    };
    let emission = render_compact_source_artifact_modules(
        audit.arm(NebulaFPrimeBranch::Base),
        "campaign-base-classification-v1",
        "Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifact",
        256,
        Some(&equality),
    )
    .expect("render the compact base source artifact");
    assert_eq!(emission.replayed_rows, audit.arm(NebulaFPrimeBranch::Base).n);
    assert_modules_match_committed(&emission.modules);
}

#[test]
fn committed_base_compact_necessity_pilot_matches_the_emitter() {
    use neo_fold_clean::frontends::nebula::f_prime::{NebulaFPrimeChainBuilder, NebulaFPrimePreprocessing};
    use neo_fold_clean::frontends::nebula::trace::Memory;
    use nightstream_constraint_exporter::{
        export_sparse_problem, find_exclusive_column_witness, nebula_family_census, render_assignment_payload_modules,
        render_compact_removal_counterexample_lean, ExportRequest,
    };
    use recursive_constraint_minimizer::Scope;

    const FAMILY: &str = "fprime.base.step.initial";
    const GENERATED_NS: &str = "Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated";

    let (params, memory_params, plan) =
        nightstream_constraint_exporter::campaign_profile_plan(1).expect("campaign plan");
    let rom = [7];
    let audit = NebulaFPrimeRelation::audit_fixed_point_constraint_sources(&params, &plan)
        .expect("discover campaign source arms");
    let prep = NebulaFPrimePreprocessing::new_seeded(
        params,
        plan,
        nightstream_constraint_exporter::CAMPAIGN_PREPROCESSING_SEED,
    )
    .expect("fixed Nebula preprocessing");
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
    let background = witnesses[0]
        .source_assignment()
        .iter()
        .map(|value| p3_field::PrimeField64::as_canonical_u64(value))
        .collect::<Vec<_>>();

    let arm = audit.arm(NebulaFPrimeBranch::Base);
    let census = nebula_family_census(&audit, NebulaFPrimeBranch::Base).expect("complete reviewed family ownership");
    let problem = export_sparse_problem(
        arm,
        ExportRequest {
            profile: "campaign-base-classification-v1".to_owned(),
            scope: Scope::Branch,
            public_input_count: arm.m_in,
            source_rows: (0..arm.n).collect(),
            complete_families: census
                .iter()
                .map(|family| family.name().to_owned())
                .collect(),
        },
    )
    .expect("export the complete binding-free base problem");
    let witness = find_exclusive_column_witness(&problem, &background, FAMILY)
        .expect("witness search must run")
        .expect("the base control family has an exclusive-column witness");

    let mut modules = render_assignment_payload_modules(&background, &format!("{GENERATED_NS}.BaseCampaignAssignment"))
        .expect("render the shared base assignment payload");
    let mutated = {
        let modulus = recursive_constraint_minimizer::GOLDILOCKS_MODULUS
            .parse::<u128>()
            .expect("fixed Goldilocks modulus");
        ((u128::from(background[witness.column()]) + u128::from(witness.delta())) % modulus) as u64
    };
    let override_entry = nightstream_constraint_exporter::ClassificationOverride {
        family: FAMILY.to_owned(),
        column: witness.column(),
        value: mutated,
    };
    let chunk_count = arm.n.div_ceil(256);
    modules.extend(
        nightstream_constraint_exporter::render_classification_leaves_modules(
            &format!("{GENERATED_NS}.BaseCompactSourceArtifact"),
            &format!("{GENERATED_NS}.BaseCampaignAssignment"),
            &format!("{GENERATED_NS}.BaseClassificationLeaves"),
            chunk_count,
            std::slice::from_ref(&override_entry),
        )
        .expect("render the base classification leaves"),
    );
    let violated_source = witness.violated_rows()[0];
    let violated_row = problem
        .rows
        .iter()
        .find(|row| row.source_index == violated_source)
        .expect("violated row in the complete problem")
        .clone();
    let necessity = render_compact_removal_counterexample_lean(
        &format!("{GENERATED_NS}.BaseCompactSourceArtifact"),
        &format!("{GENERATED_NS}.BaseCampaignAssignment"),
        &format!("{GENERATED_NS}.BaseClassificationLeaves"),
        &format!("{GENERATED_NS}.BaseCompactStepInitialNecessity"),
        &override_entry,
        &violated_row,
        violated_source / 256,
        chunk_count,
    )
    .expect("render the compact necessity pilot module");
    modules.push(nightstream_constraint_exporter::GeneratedLeanModule {
        module_name: format!("{GENERATED_NS}.BaseCompactStepInitialNecessity"),
        content: necessity,
    });
    assert_modules_match_committed(&modules);
}

#[test]
#[ignore = "builds the complete 4.5M-row problem (~22 GB peak); run explicitly during campaign iterations"]
fn committed_y_ring_compact_redundancy_module_matches_the_emitter() {
    use nightstream_constraint_exporter::{
        export_sparse_problem, nebula_family_census, render_compact_redundancy_certificate_lean, ExportRequest,
    };
    use recursive_constraint_minimizer::{derive_scalar_certificate, Scope, Selection};
    use std::collections::BTreeSet;

    const CANDIDATE: &str = "nifs.pi_rlc.verify.padding.y_ring";
    const PI_CCS_SUPPORT: &str = "nifs.pi_ccs.padded_row.canonicality";
    const PI_DEC_SUPPORT: &str = "nifs.pi_dec.verify";
    const GENERATED_NS: &str = "Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated";

    let audit = campaign_audit();
    let arm = audit.arm(NebulaFPrimeBranch::Recursive);
    let census = nebula_family_census(&audit, NebulaFPrimeBranch::Recursive).expect("complete reviewed ownership");
    let plan_names = census
        .iter()
        .map(|family| family.name().to_owned())
        .collect::<Vec<_>>();
    let family = |name: &str| {
        census
            .iter()
            .find(|family| family.name() == name)
            .unwrap_or_else(|| panic!("missing exact family {name}"))
    };
    let complete = export_sparse_problem(
        arm,
        ExportRequest {
            profile: "campaign-recursive-classification-v1".to_owned(),
            scope: Scope::Branch,
            public_input_count: arm.m_in,
            source_rows: (0..arm.n).collect(),
            complete_families: plan_names.clone(),
        },
    )
    .expect("export the complete binding-free recursive problem");
    let slice_rows = [CANDIDATE, PI_CCS_SUPPORT, PI_DEC_SUPPORT]
        .into_iter()
        .flat_map(|name| family(name).source_rows().iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let slice = export_sparse_problem(
        arm,
        ExportRequest {
            profile: "campaign-recursive-classification-v1".to_owned(),
            scope: Scope::Branch,
            public_input_count: arm.m_in,
            source_rows: slice_rows,
            complete_families: vec![CANDIDATE.to_owned()],
        },
    )
    .expect("export the y_ring candidate and support slice");
    let certificate = derive_scalar_certificate(&slice, &Selection::Family(CANDIDATE.to_owned()))
        .expect("derive the scalar certificate")
        .expect("every y_ring row has a scalar certificate");
    let content = render_compact_redundancy_certificate_lean(
        &complete,
        &slice,
        &certificate,
        &format!("{GENERATED_NS}.RecursiveCompactSourceArtifact"),
        &format!("{GENERATED_NS}.RecursiveCompactSourceArtifact"),
        &format!("{GENERATED_NS}.RecursiveNifsPiRlcVerifyPaddingYRingRedundancy"),
        &plan_names,
    )
    .expect("render the y_ring compact redundancy module");
    assert_modules_match_committed(&[nightstream_constraint_exporter::GeneratedLeanModule {
        module_name: format!("{GENERATED_NS}.RecursiveNifsPiRlcVerifyPaddingYRingRedundancy"),
        content,
    }]);
}

#[test]
fn committed_recursive_compact_source_modules_match_the_emitter() {
    let audit = campaign_audit();
    let emission = render_compact_source_artifact_modules(
        audit.arm(NebulaFPrimeBranch::Recursive),
        "campaign-recursive-classification-v1",
        "Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifact",
        65_536,
        None,
    )
    .expect("render the compact recursive source artifact");
    assert_eq!(emission.replayed_rows, audit.arm(NebulaFPrimeBranch::Recursive).n);
    assert_modules_match_committed(&emission.modules);
}
