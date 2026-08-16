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
