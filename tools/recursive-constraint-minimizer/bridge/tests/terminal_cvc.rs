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
    export_terminal_problem, refine_terminal_with_cvc5, terminal_family_census, ExportRequest,
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
    assert_eq!(audit.reviewed_family_names(), TERMINAL_R1CS_FAMILY_NAMES);

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
#[ignore = "requires the installed GPL cvc5 finite-field solver"]
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
