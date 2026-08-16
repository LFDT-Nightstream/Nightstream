//! Recursive-arm witness census and compact necessity emission (bar 5).
//!
//! Requires the committed bar-4 assignment
//! (`evidence/nebula-recursive-assignment.json`). Every family gets an
//! exclusive-column witness search over the complete binding-free problem;
//! misses are recorded fail-closed (the family stays Inconclusive for this
//! constructor). Successful witnesses render compact necessity modules
//! against the string-payload recursive source artifact.

use neo_fold_clean::frontends::nebula::f_prime::{NebulaFPrimeBranch, NebulaFPrimeRelation};
use neo_fold_clean::frontends::nebula::layout::NebulaParams;
use neo_fold_clean::frontends::nebula::plan::NebulaPlan;
use neo_fold_clean::paper::params::Params;
use nightstream_constraint_exporter::{
    export_sparse_problem, find_exclusive_column_witness, load_nebula_source_assignment, nebula_family_census,
    render_assignment_payload_modules, render_compact_removal_counterexample_lean, ExportRequest, GeneratedLeanModule,
};
use recursive_constraint_minimizer::Scope;

const GENERATED_DIR: &str =
    "../../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/MinimizerCampaign/Generated";
const GENERATED_NS: &str = "Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated";
const ASSIGNMENT_PROFILE: &str = "nebula-saved-recursive-assignment";

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

fn module_stem(family: &str) -> String {
    family
        .split(['.', '_'])
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
#[ignore = "bar-5 recursive witness census; needs the committed bar-4 assignment; run with --ignored --nocapture"]
fn recursive_witness_census_emits_compact_necessity_modules() {
    let assignment_path = format!(
        "{}/../evidence/nebula-recursive-assignment.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let json = std::fs::read(&assignment_path).expect("committed bar-4 recursive assignment");
    let audit = campaign_audit();
    let checked = load_nebula_source_assignment(&audit, NebulaFPrimeBranch::Recursive, ASSIGNMENT_PROFILE, &json)
        .expect("replay the committed recursive assignment");
    let background = checked
        .values()
        .iter()
        .map(|value| p3_field::PrimeField64::as_canonical_u64(value))
        .collect::<Vec<_>>();

    let arm = audit.arm(NebulaFPrimeBranch::Recursive);
    let census = nebula_family_census(&audit, NebulaFPrimeBranch::Recursive).expect("complete reviewed ownership");
    let plan_names = census
        .iter()
        .map(|family| family.name().to_owned())
        .collect::<Vec<_>>();
    eprintln!("exporting the complete binding-free recursive problem ({} rows)", arm.n);
    let problem = export_sparse_problem(
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

    let mut modules: Vec<GeneratedLeanModule> =
        render_assignment_payload_modules(&background, &format!("{GENERATED_NS}.RecursiveCampaignAssignment"))
            .expect("render the shared recursive assignment payload");
    let mut summary = Vec::new();
    for family in &census {
        let start = std::time::Instant::now();
        let outcome =
            find_exclusive_column_witness(&problem, &background, family.name()).expect("witness search must run");
        match outcome {
            Some(witness) => {
                let stem = format!("Recursive{}Necessity", module_stem(family.name()));
                let content = render_compact_removal_counterexample_lean(
                    &problem,
                    &background,
                    &witness,
                    &format!("{GENERATED_NS}.RecursiveCompactSourceArtifact"),
                    &format!("{GENERATED_NS}.RecursiveCampaignAssignment"),
                    &format!("{GENERATED_NS}.{stem}"),
                    &plan_names,
                )
                .expect("render the compact necessity module");
                modules.push(GeneratedLeanModule {
                    module_name: format!("{GENERATED_NS}.{stem}"),
                    content,
                });
                summary.push(format!(
                    "{} witness column={} delta={} violated={} ms={}",
                    family.name(),
                    witness.column(),
                    witness.delta(),
                    witness.violated_rows().len(),
                    start.elapsed().as_millis(),
                ));
            }
            None => {
                summary.push(format!(
                    "{} NO-EXCLUSIVE-COLUMN ms={}",
                    family.name(),
                    start.elapsed().as_millis(),
                ));
            }
        }
        eprintln!("{}", summary.last().expect("recorded"));
    }

    let evidence_path = format!(
        "{}/../evidence/recursive-witness-census.txt",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::write(&evidence_path, summary.join("\n") + "\n").expect("write census evidence");

    let mut drifted = Vec::new();
    for module in &modules {
        let file = module
            .module_name
            .rsplit('.')
            .next()
            .expect("module name has a final segment");
        let path = format!("{GENERATED_DIR}/{file}.lean");
        let committed = std::fs::read_to_string(&path).unwrap_or_default();
        if committed != module.content {
            std::fs::write(format!("{path}.expected"), &module.content).expect("write expected module");
            drifted.push(path);
        }
    }
    assert!(
        drifted.is_empty(),
        "recursive compact modules drifted; inspect and promote {} files",
        drifted.len()
    );
}
