use std::collections::BTreeSet;

use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_fold_clean::frontends::direct_ccs::ajtai;
use neo_fold_clean::frontends::r1cs_f_prime::terminal_r1cs::{
    compile_combined_terminal_r1cs, TerminalR1csInput, TERMINAL_R1CS_FAMILY_NAMES,
};
use neo_fold_clean::paper::params::Params;
use neo_fold_clean::paper::relations::{
    superneo_has_canonical_x_shape, superneo_public_x_cols, CcsInstance, CeClaim, WitnessMat,
};
use neo_math::{D, F, K};
use nightstream_constraint_exporter::{
    export_complete_terminal_problem, export_terminal_problem, find_exclusive_column_witness,
    refine_terminal_with_cvc5, render_complete_terminal_bound_artifact_modules,
    render_terminal_removal_counterexample_lean, terminal_family_census, ExportRequest,
};
use p3_field::PrimeCharacteristicRing;
use recursive_constraint_minimizer::{Conclusion, Scope, Selection, SolverConfig, SolverStatus};

#[path = "../../../../crates/neo-fold-clean/tests/support/lean_manifest_fixture.rs"]
mod lean_manifest_fixture;
use lean_manifest_fixture::{combined_manifest, parse_combined, TEST_AJTAI_SEED};

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

#[test]
fn exact_combined_terminal_fixture_has_the_reviewed_exportable_census() {
    let relation = compile_fixture();
    let audit = relation.constraint_audit();
    assert!(audit.source().is_satisfied(audit.source().witness()));
    assert_eq!(audit.source().rows(), 58_593);
    assert_eq!(audit.source_public_columns(), 48_871);
    assert_eq!(audit.source_private_columns(), 9_721);

    let census = terminal_family_census(audit).expect("complete reviewed terminal family ownership");
    assert_eq!(
        census
            .iter()
            .map(|family| family.name())
            .collect::<BTreeSet<_>>(),
        TERMINAL_R1CS_FAMILY_NAMES.into_iter().collect()
    );
    assert_eq!(
        census
            .iter()
            .map(|family| family.source_rows().len())
            .sum::<usize>(),
        audit.source().rows()
    );

    let family = census
        .iter()
        .min_by_key(|family| family.source_rows().len())
        .expect("one terminal family");
    let export = export_terminal_problem(
        audit,
        ExportRequest {
            profile: "combined-nebula-terminal-fixture".to_owned(),
            scope: Scope::Branch,
            public_input_count: audit.source_public_columns(),
            source_rows: family.source_rows().to_vec(),
            complete_families: vec![family.name().to_owned()],
        },
    )
    .expect("export the smallest exact terminal family");
    assert_eq!(export.problem().rows.len(), family.source_rows().len());
    assert_eq!(export.binding().spartan_rows(), audit.spartan_rows());
    assert_eq!(export.binding().spartan_columns(), audit.spartan_columns());
}

#[test]
fn installed_cvc5_runs_one_exact_combined_terminal_iteration() {
    let relation = compile_fixture();
    let audit = relation.constraint_audit();
    let census = terminal_family_census(audit).expect("complete reviewed terminal family ownership");
    let family = census
        .iter()
        .min_by_key(|family| family.source_rows().len())
        .expect("one terminal family");
    let selection = Selection::Family(family.name().to_owned());
    let report = refine_terminal_with_cvc5(
        audit,
        ExportRequest {
            profile: "combined-nebula-terminal-cvc5-control".to_owned(),
            scope: Scope::Branch,
            public_input_count: audit.source_public_columns(),
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
    .expect("run one exact combined terminal refinement iteration");

    eprintln!(
        "exact combined terminal cvc5 control: family={} status={:?} conclusion={:?} pending={:?}",
        family.name(),
        report.refinement().solver_run.status,
        report.refinement().conclusion,
        report.refinement().pending_retained_row,
    );
    assert_eq!(report.refinement().iterations, 1);
    assert_eq!(report.refinement().solver_run.status, SolverStatus::Sat);
    assert_eq!(report.refinement().conclusion, Conclusion::Inconclusive);
    assert!(report.refinement().pending_retained_row.is_some());
}

#[test]
#[ignore = "terminal exclusive-column witness census; run with --ignored --nocapture"]
fn print_terminal_exclusive_column_witness_census() {
    let relation = compile_fixture();
    let audit = relation.constraint_audit();
    let export = export_complete_terminal_problem(&audit, "campaign-terminal-witness-census")
        .expect("export the complete terminal relation");
    let problem = export.problem();
    let background = audit
        .source()
        .witness()
        .iter()
        .map(|value| p3_field::PrimeField64::as_canonical_u64(value))
        .collect::<Vec<_>>();
    let census = terminal_family_census(&audit).expect("complete reviewed terminal family ownership");
    let mut found = 0usize;
    for family in &census {
        match find_exclusive_column_witness(problem, &background, family.name()) {
            Ok(Some(witness)) => {
                found += 1;
                eprintln!(
                    "terminal witness family={} column={} delta={} violated={}",
                    family.name(),
                    witness.column(),
                    witness.delta(),
                    witness.violated_rows().len(),
                );
            }
            Ok(None) => eprintln!("terminal no-exclusive-witness family={}", family.name()),
            Err(error) => eprintln!("terminal witness-error family={} error={error}", family.name()),
        }
    }
    eprintln!("terminal exclusive-column census: {found}/{} families", census.len());
}

#[test]
#[ignore = "emit the terminal Lean classification batch to NIGHTSTREAM_EMIT_DIR; run with --ignored --nocapture"]
fn emit_terminal_lean_classification_batch() {
    let emit_dir = std::env::var("NIGHTSTREAM_EMIT_DIR").expect("set NIGHTSTREAM_EMIT_DIR");
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
        let path = format!("{emit_dir}/{file}.lean");
        std::fs::write(&path, &module.content).expect("write terminal artifact module");
        eprintln!("emitted {path} bytes={}", module.content.len());
    }

    for family in &census {
        let witness = find_exclusive_column_witness(&problem, &background, family.name())
            .expect("witness search must run")
            .expect("every terminal family has an exclusive-column witness");
        let module_stem = family
            .name()
            .split(['.', '_'])
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
        let path = format!("{emit_dir}/{module_stem}Necessity.lean");
        std::fs::write(&path, &lean).expect("write terminal counterexample module");
        eprintln!("emitted {path} bytes={}", lean.len());
    }
}
