//! Drift gate: committed generated Lean mirrors equal fresh emitter output.

use neo_fold_clean::frontends::nebula::f_prime::{
    NebulaFPrimeBranch, NebulaFPrimeChainBuilder, NebulaFPrimePreprocessing, NebulaFPrimeRelation,
};
use neo_fold_clean::frontends::nebula::trace::Memory;
use nightstream_constraint_exporter::{
    export_complete_nebula_problem, find_exclusive_column_witness, nebula_family_census,
    render_complete_bound_artifact_modules, render_removal_counterexample_lean,
};

const GENERATED_DIR: &str =
    "../../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/MinimizerCampaign/Generated";

fn module_stem(family: &str) -> String {
    family
        .split('.')
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join("")
}

#[test]
fn committed_base_batch_mirrors_match_the_emitter() {
    let (params, memory_params, plan) =
        nightstream_constraint_exporter::campaign_profile_plan(1).expect("campaign plan");
    let rom = [7];
    let audit = NebulaFPrimeRelation::audit_fixed_point_constraint_sources(&params, &plan)
        .expect("discover exact Nebula source arms");
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
    let assignment = witnesses[0].source_assignment().to_vec();

    let branch = NebulaFPrimeBranch::Base;
    let export = export_complete_nebula_problem(&audit, branch, "campaign-base-classification-v1")
        .expect("export the complete base arm");
    let problem = export.problem().clone();
    let background = assignment
        .iter()
        .map(|value| p3_field::PrimeField64::as_canonical_u64(value))
        .collect::<Vec<_>>();
    let census = nebula_family_census(&audit, branch).expect("complete reviewed Nebula family ownership");
    let plan_names = census
        .iter()
        .map(|family| family.name().to_owned())
        .collect::<Vec<_>>();

    let artifact_namespace = "Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.BaseBoundArtifact";
    let artifact_modules = render_complete_bound_artifact_modules(&export, artifact_namespace)
        .expect("render the complete base bound artifact modules");
    let mut drifted = Vec::new();
    for module in &artifact_modules {
        let file = module
            .module_name
            .rsplit('.')
            .next()
            .expect("module name has a final segment");
        let path = format!("{GENERATED_DIR}/{file}.lean");
        let committed = std::fs::read_to_string(&path).unwrap_or_default();
        if module.content != committed {
            std::fs::write(format!("{path}.expected"), &module.content).expect("write expected artifact mirror");
            drifted.push(path);
        }
    }

    for family in &census {
        let witness = find_exclusive_column_witness(&problem, &background, family.name())
            .expect("witness search must run")
            .expect("every base family has an exclusive-column witness");
        let stem = module_stem(family.name());
        let namespace =
            format!("Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.{stem}Necessity");
        let lean = render_removal_counterexample_lean(
            &problem,
            witness.model(),
            family.name(),
            artifact_namespace,
            artifact_namespace,
            &namespace,
            &plan_names,
        )
        .expect("render the removal counterexample");
        let path = format!("{GENERATED_DIR}/{stem}Necessity.lean");
        let committed = std::fs::read_to_string(&path).unwrap_or_default();
        if lean != committed {
            std::fs::write(format!("{path}.expected"), &lean).expect("write expected necessity mirror");
            drifted.push(path);
        }
    }
    assert!(
        drifted.is_empty(),
        "base batch mirrors drifted; inspect and promote {} files",
        drifted.len()
    );
}
