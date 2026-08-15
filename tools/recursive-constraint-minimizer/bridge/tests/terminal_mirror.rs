//! Drift gate: committed terminal Lean mirrors equal fresh emitter output.

use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_fold_clean::frontends::direct_ccs::ajtai;
use neo_fold_clean::frontends::r1cs_f_prime::terminal_r1cs::{compile_combined_terminal_r1cs, TerminalR1csInput};
use neo_fold_clean::paper::params::Params;
use neo_fold_clean::paper::relations::{
    superneo_has_canonical_x_shape, superneo_public_x_cols, CcsInstance, CeClaim, WitnessMat,
};
use neo_math::{D, F, K};
use nightstream_constraint_exporter::{
    export_complete_terminal_problem, find_exclusive_column_witness, render_complete_terminal_bound_artifact_modules,
    render_terminal_removal_counterexample_lean, terminal_family_census,
};
use p3_field::PrimeCharacteristicRing;

#[path = "../../../../crates/neo-fold-clean/tests/support/lean_manifest_fixture.rs"]
mod lean_manifest_fixture;
use lean_manifest_fixture::{combined_manifest, parse_combined, TEST_AJTAI_SEED};

const GENERATED_DIR: &str =
    "../../../formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/MinimizerCampaign/Generated";

fn zero_superneo_public_x(m_in: usize) -> Mat<F> {
    let x = Mat::zero(D, superneo_public_x_cols(m_in), F::ZERO);
    assert!(superneo_has_canonical_x_shape(&x, m_in));
    x
}

fn compile_fixture() -> neo_fold_clean::frontends::r1cs_f_prime::terminal_r1cs::CompiledTerminalR1cs {
    let manifest = parse_combined(&combined_manifest()).expect("valid combined manifest");
    let mut public = vec![F::ZERO; manifest.public_carrier_width()];
    public[0] = F::ONE;
    let emission = manifest
        .emit(&public, |_| Some(F::ZERO), &[F::ZERO])
        .expect("honest combined emission");
    assert!(emission.is_satisfied());

    let params = Params::goldilocks_paper_b2();
    let log = ajtai::setup_seeded(&params, emission.structure(), TEST_AJTAI_SEED);
    let fresh = CcsInstance::from_low_norm_assignment(
        &params,
        &log,
        emission.structure(),
        emission.assignment(),
        manifest.public_carrier_width(),
    )
    .expect("honest combined fresh instance");
    let zero_witness = Mat::zero(D, emission.structure().m / D, F::ZERO);
    let joint_row_variables = emission
        .structure()
        .n
        .max(emission.structure().m)
        .next_power_of_two()
        .max(2)
        .trailing_zeros() as usize;
    let zero_claim = CeClaim {
        c: Commitment::zeros(D, manifest.terminal_r1cs().verifier_rows()),
        X: zero_superneo_public_x(manifest.public_carrier_width()),
        r: vec![K::ZERO; joint_row_variables],
        y_ring: vec![vec![K::ZERO; D.next_power_of_two()]; emission.structure().t() + 1],
        ct: vec![K::ZERO; emission.structure().t() + 1],
        m_in: manifest.public_carrier_width(),
        fold_digest: [0; 32],
        adv: None,
    };
    let running_claims = vec![zero_claim; 14];
    let running_witnesses: Vec<WitnessMat> = vec![zero_witness; 14];

    compile_combined_terminal_r1cs(
        &manifest,
        &log,
        TerminalR1csInput {
            running_claims: &running_claims,
            running_witnesses: &running_witnesses,
            fresh: &fresh,
        },
    )
    .expect("honest combined terminal R1CS")
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
fn committed_terminal_batch_mirrors_match_the_emitter() {
    let relation = compile_fixture();
    let audit = relation.constraint_audit();
    let export = export_complete_terminal_problem(&audit, "campaign-terminal-classification-v1")
        .expect("export the complete terminal relation");
    let problem = export.problem().clone();
    let background = audit
        .source()
        .witness()
        .iter()
        .map(|value| p3_field::PrimeField64::as_canonical_u64(value))
        .collect::<Vec<_>>();
    let census = terminal_family_census(&audit).expect("complete reviewed terminal family ownership");
    let plan = census
        .iter()
        .map(|family| family.name().to_owned())
        .collect::<Vec<_>>();

    let artifact_namespace =
        "Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.TerminalBoundArtifact";
    let artifact_modules = render_complete_terminal_bound_artifact_modules(&export, artifact_namespace)
        .expect("render the complete terminal bound artifact modules");
    for module in &artifact_modules {
        let file = module
            .module_name
            .rsplit('.')
            .next()
            .expect("module name has a final segment");
        let committed = std::fs::read_to_string(format!("{GENERATED_DIR}/{file}.lean"))
            .expect("committed terminal artifact mirror exists");
        assert_eq!(module.content, committed, "{file}.lean drifted");
    }

    for family in &census {
        let witness = find_exclusive_column_witness(&problem, &background, family.name())
            .expect("witness search must run")
            .expect("every terminal family has an exclusive-column witness");
        let stem = module_stem(family.name());
        let namespace =
            format!("Nightstream.Implementation.R1CS.Artifacts.MinimizerCampaign.Generated.{stem}Necessity");
        let lean = render_terminal_removal_counterexample_lean(
            &problem,
            witness.model(),
            family.name(),
            artifact_namespace,
            artifact_namespace,
            &namespace,
            &plan,
        )
        .expect("render the terminal removal counterexample");
        let committed = std::fs::read_to_string(format!("{GENERATED_DIR}/{stem}Necessity.lean"))
            .expect("committed terminal necessity mirror exists");
        assert_eq!(lean, committed, "{stem}Necessity.lean drifted");
    }
}
