//! Drift gate: committed compact-source Lean modules equal fresh emission.
//!
//! The base arm is the pilot: its string-payload artifact must expand to the
//! exact committed literal `BaseBoundArtifact.sourceArtifact`, which the
//! generated assembly pins with a `native_decide` equality theorem.

use neo_fold_clean::frontends::nebula::f_prime::{NebulaFPrimeBranch, NebulaFPrimeRelation};
use neo_fold_clean::frontends::nebula::layout::NebulaParams;
use neo_fold_clean::frontends::nebula::plan::NebulaPlan;
use neo_fold_clean::paper::params::Params;
use nightstream_constraint_exporter::render_compact_source_artifact_modules;

const GENERATED_DIR: &str =
    "../../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/MinimizerCampaign/Generated";

fn campaign_audit() -> neo_fold_clean::frontends::nebula::f_prime::NebulaFPrimeConstraintSourceAudit {
    let inner = neo_params::NeoParams::new(
        neo_params::goldilocks_paper_b2::Q,
        neo_params::goldilocks_paper_b2::ETA as u32,
        neo_params::goldilocks_paper_b2::D as u32,
        1,
        neo_params::goldilocks_paper_b2::M,
        neo_params::goldilocks_paper_b2::B_BASE,
        2,
        1,
        neo_params::goldilocks_paper_b2::EXTENSION_DEGREE,
        1,
    )
    .expect("campaign profile parameters");
    let params = Params::test_only_from_neo_params(inner);
    let memory = NebulaParams::new(0, 0, 1, 2, 1).expect("campaign memory profile");
    let plan = NebulaPlan::new(memory, vec![7], [0xDA; 32], params.kappa() as usize).expect("campaign Nebula plan");
    NebulaFPrimeRelation::audit_fixed_point_constraint_sources(&params, &plan).expect("discover campaign source arms")
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
    let emission = render_compact_source_artifact_modules(
        audit.arm(NebulaFPrimeBranch::Base),
        "campaign-base-classification-v1",
        "Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseCompactSourceArtifact",
        Some("Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact"),
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

    let inner = neo_params::NeoParams::new(
        neo_params::goldilocks_paper_b2::Q,
        neo_params::goldilocks_paper_b2::ETA as u32,
        neo_params::goldilocks_paper_b2::D as u32,
        1,
        neo_params::goldilocks_paper_b2::M,
        neo_params::goldilocks_paper_b2::B_BASE,
        2,
        1,
        neo_params::goldilocks_paper_b2::EXTENSION_DEGREE,
        1,
    )
    .expect("campaign profile parameters");
    let params = Params::test_only_from_neo_params(inner);
    let memory_params = NebulaParams::new(0, 0, 1, 2, 1).expect("one-step memory profile");
    let rom = [7];
    let plan = NebulaPlan::new(memory_params, rom.to_vec(), [0xDA; 32], params.kappa() as usize).expect("Nebula plan");
    let audit = NebulaFPrimeRelation::audit_fixed_point_constraint_sources(&params, &plan)
        .expect("discover campaign source arms");
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
    let plan_names = census
        .iter()
        .map(|family| family.name().to_owned())
        .collect::<Vec<_>>();
    let witness = find_exclusive_column_witness(&problem, &background, FAMILY)
        .expect("witness search must run")
        .expect("the base control family has an exclusive-column witness");

    let mut modules = render_assignment_payload_modules(&background, &format!("{GENERATED_NS}.BaseCampaignAssignment"))
        .expect("render the shared base assignment payload");
    let necessity = render_compact_removal_counterexample_lean(
        &problem,
        &background,
        &witness,
        &format!("{GENERATED_NS}.BaseCompactSourceArtifact"),
        &format!("{GENERATED_NS}.BaseCampaignAssignment"),
        &format!("{GENERATED_NS}.BaseCompactStepInitialNecessity"),
        &plan_names,
    )
    .expect("render the compact necessity pilot module");
    modules.push(nightstream_constraint_exporter::GeneratedLeanModule {
        module_name: format!("{GENERATED_NS}.BaseCompactStepInitialNecessity"),
        content: necessity,
    });
    assert_modules_match_committed(&modules);
}

#[test]
fn committed_recursive_compact_source_modules_match_the_emitter() {
    let audit = campaign_audit();
    let emission = render_compact_source_artifact_modules(
        audit.arm(NebulaFPrimeBranch::Recursive),
        "campaign-recursive-classification-v1",
        "Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.RecursiveCompactSourceArtifact",
        None,
    )
    .expect("render the compact recursive source artifact");
    assert_eq!(emission.replayed_rows, audit.arm(NebulaFPrimeBranch::Recursive).n);
    assert_modules_match_committed(&emission.modules);
}
