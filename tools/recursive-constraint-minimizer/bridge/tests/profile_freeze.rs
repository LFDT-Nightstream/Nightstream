//! Digest measurement for the campaign profile freeze (bar 2).
//!
//! The `#[ignore]` printer measures candidate profiles. The freeze document
//! and its drift test pin the chosen digests.

use std::time::Instant;

use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_fold_clean::frontends::direct_ccs::ajtai;
use neo_fold_clean::frontends::nebula::f_prime::{NebulaFPrimeBranch, NebulaFPrimeRelation};
use neo_fold_clean::frontends::nebula::layout::NebulaParams;
use neo_fold_clean::frontends::nebula::plan::NebulaPlan;
use neo_fold_clean::frontends::r1cs_f_prime::terminal_r1cs::{compile_combined_terminal_r1cs, TerminalR1csInput};
use neo_fold_clean::paper::params::Params;
use neo_fold_clean::paper::relations::{
    superneo_has_canonical_x_shape, superneo_public_x_cols, CcsInstance, CeClaim, WitnessMat,
};
use neo_math::{D, F, K};
use nightstream_constraint_exporter::{
    export_complete_nebula_problem, export_complete_terminal_problem, export_nebula_problem, ExportRequest,
};
use p3_field::PrimeCharacteristicRing;
use recursive_constraint_minimizer::Scope;

#[path = "../../../../crates/neo-fold-clean/tests/support/lean_manifest_fixture.rs"]
mod lean_manifest_fixture;
use lean_manifest_fixture::{combined_manifest, parse_combined, TEST_AJTAI_SEED};

const FREEZE_PLAN_SEED: [u8; 32] = [0xF5; 32];
const FREEZE_ROM: [u32; 1] = [7];

fn freeze_candidate_audit(
    params: &Params,
) -> neo_fold_clean::frontends::nebula::f_prime::NebulaFPrimeConstraintSourceAudit {
    let memory = NebulaParams::new(0, 0, 1, 2, 1).expect("freeze memory profile");
    let plan = NebulaPlan::new(memory, FREEZE_ROM.to_vec(), FREEZE_PLAN_SEED, params.kappa() as usize)
        .expect("freeze Nebula plan");
    NebulaFPrimeRelation::audit_fixed_point_constraint_sources(params, &plan)
        .expect("discover freeze-candidate source arms")
}

#[test]
#[ignore = "measurement printer for the profile freeze; run with --ignored --nocapture"]
fn print_paper_b2_freeze_candidate_digests() {
    print_freeze_candidate("paper-b2", Params::goldilocks_paper_b2());
}

#[test]
#[ignore = "measurement printer for the profile freeze; run with --ignored --nocapture"]
fn print_paper_b2_lambda114_freeze_candidate_digests() {
    // Diagnostic only: paper B.2 with the security target lowered to the 114
    // bits this shape's census provides. The regime decision is not made here.
    let inner = neo_params::NeoParams::new(
        neo_params::goldilocks_paper_b2::Q,
        neo_params::goldilocks_paper_b2::ETA as u32,
        neo_params::goldilocks_paper_b2::D as u32,
        neo_params::goldilocks_paper_b2::KAPPA,
        neo_params::goldilocks_paper_b2::M,
        neo_params::goldilocks_paper_b2::B_BASE,
        neo_params::goldilocks_paper_b2::K_RHO,
        neo_params::goldilocks_paper_b2::T,
        neo_params::goldilocks_paper_b2::EXTENSION_DEGREE,
        114,
    )
    .expect("paper B.2 shape with a 114-bit target");
    print_freeze_candidate("paper-b2-lambda114", Params::test_only_from_neo_params(inner));
}

fn campaign_minimal_params() -> Params {
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
    .expect("minimal parameters satisfy the exact RLC guard");
    Params::test_only_from_neo_params(inner)
}

fn campaign_audit(
    plan_seed: [u8; 32],
) -> neo_fold_clean::frontends::nebula::f_prime::NebulaFPrimeConstraintSourceAudit {
    let params = campaign_minimal_params();
    let memory = NebulaParams::new(0, 0, 1, 2, 1).expect("campaign memory profile");
    let plan = NebulaPlan::new(memory, vec![7], plan_seed, params.kappa() as usize).expect("campaign Nebula plan");
    NebulaFPrimeRelation::audit_fixed_point_constraint_sources(&params, &plan).expect("discover campaign source arms")
}

fn compile_terminal_fixture() -> neo_fold_clean::frontends::r1cs_f_prime::terminal_r1cs::CompiledTerminalR1cs {
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
        X: {
            let x = Mat::zero(D, superneo_public_x_cols(manifest.public_carrier_width()), F::ZERO);
            assert!(superneo_has_canonical_x_shape(&x, manifest.public_carrier_width()));
            x
        },
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
fn campaign_profile_v1_digests_are_frozen() {
    // PROFILE.md is the freeze document. These pins must match its table.
    const BASE_SOURCE_DIGEST: &str = "sha256:54bec6fa7de4ec475e2fd43a1c015bfede809d2d1370b67677ea66dbda6839e7";
    const RECURSIVE_SOURCE_DIGEST: &str = "sha256:4c0a51647877cd072970c160d49d1dc78b7d34b39dd3e7613c716cef2869934e";
    const FINAL_PLAN_DIGEST: &str = "sha256:3024cf0eea6ac9093157e5dc1674187abc9fa3f17f8598d72ab41e45504e50fc";
    const TERMINAL_SOURCE_DIGEST: &str = "sha256:85b400cebcfaa8fac702072aff342d67c6acca87e4470199d86a935c98264461";
    const TERMINAL_DIAGNOSTIC_DIGEST: &str = "sha256:63664e95c3f91dcf35db99ad3e0dd235643d274e5ccfd9be6a18252eb8a12f98";

    let audit = campaign_audit([0xDA; 32]);
    for (branch, source_digest, rows, columns) in [
        (NebulaFPrimeBranch::Base, BASE_SOURCE_DIGEST, 39_949, 38_626),
        (
            NebulaFPrimeBranch::Recursive,
            RECURSIVE_SOURCE_DIGEST,
            4_530_315,
            4_480_464,
        ),
    ] {
        let arm = audit.arm(branch);
        let export = export_nebula_problem(
            &audit,
            branch,
            ExportRequest {
                profile: "campaign-profile-v1-freeze-gate".to_owned(),
                scope: Scope::Branch,
                public_input_count: arm.m_in,
                source_rows: vec![0],
                complete_families: Vec::new(),
            },
        )
        .expect("export one frozen-profile source row");
        let problem = export.problem();
        let binding = export.binding();
        assert_eq!(
            problem.source.artifact_digest, source_digest,
            "{branch:?} source digest drifted"
        );
        assert_eq!(problem.source.total_rows, rows, "{branch:?} source rows drifted");
        assert_eq!(problem.column_count, columns, "{branch:?} source columns drifted");
        assert_eq!(problem.public_input_count, 2_426, "{branch:?} public prefix drifted");
        assert_eq!(
            binding.final_plan_digest(),
            FINAL_PLAN_DIGEST,
            "{branch:?} final plan digest drifted"
        );
        assert_eq!(binding.final_rows(), 1_415_271, "{branch:?} final rows drifted");
        assert_eq!(binding.final_columns(), 6_559_326, "{branch:?} final columns drifted");
        assert_eq!(
            binding.final_public_input_count(),
            2_430,
            "{branch:?} final public columns drifted"
        );
    }

    let relation = compile_terminal_fixture();
    let terminal_audit = relation.constraint_audit();
    let export = export_complete_terminal_problem(&terminal_audit, "campaign-terminal-classification-v1")
        .expect("export the complete frozen terminal relation");
    let problem = export.problem();
    let binding = export.binding();
    assert_eq!(
        problem.source.artifact_digest, TERMINAL_SOURCE_DIGEST,
        "terminal source digest drifted"
    );
    assert_eq!(problem.source.total_rows, 58_593, "terminal source rows drifted");
    assert_eq!(problem.column_count, 58_592, "terminal source columns drifted");
    assert_eq!(problem.public_input_count, 48_871, "terminal public prefix drifted");
    assert_eq!(problem.complete_families.len(), 8, "terminal family count drifted");
    assert_eq!(
        binding.diagnostic_digest(),
        TERMINAL_DIAGNOSTIC_DIGEST,
        "terminal diagnostic digest drifted"
    );
    assert_eq!(binding.spartan_rows(), 65_536, "terminal Spartan rows drifted");
    assert_eq!(binding.spartan_columns(), 114_407, "terminal Spartan columns drifted");
}

#[test]
#[ignore = "measurement printer for the profile freeze; run with --ignored --nocapture"]
fn print_campaign_profile_v1_digests() {
    for (label, seed) in [
        ("campaign-v1-mirror-shape", [0xDA; 32]),
        ("campaign-v1-census-shape", [0xD9; 32]),
    ] {
        let start = Instant::now();
        let audit = campaign_audit(seed);
        eprintln!("candidate={label} audit build: {} ms", start.elapsed().as_millis());
        for branch in [NebulaFPrimeBranch::Base, NebulaFPrimeBranch::Recursive] {
            // A one-row export carries the selection-independent source
            // artifact digest and final plan digest without the complete
            // recursive projection.
            let start = Instant::now();
            let arm = audit.arm(branch);
            let export = export_nebula_problem(
                &audit,
                branch,
                ExportRequest {
                    profile: label.to_owned(),
                    scope: Scope::Branch,
                    public_input_count: arm.m_in,
                    source_rows: vec![0],
                    complete_families: Vec::new(),
                },
            )
            .expect("export one campaign-candidate source row");
            let problem = export.problem();
            let binding = export.binding();
            eprintln!(
                "candidate={label} branch={branch:?} n={} m={} m_in={} digest={} final_rows={} final_cols={} final_public={} final_plan_digest={} export_ms={}",
                problem.source.total_rows,
                problem.column_count,
                problem.public_input_count,
                problem.source.artifact_digest,
                binding.final_rows(),
                binding.final_columns(),
                binding.final_public_input_count(),
                binding.final_plan_digest(),
                start.elapsed().as_millis(),
            );
        }
    }

    let start = Instant::now();
    let relation = compile_terminal_fixture();
    let audit = relation.constraint_audit();
    eprintln!(
        "candidate=campaign-v1-terminal fixture build: {} ms",
        start.elapsed().as_millis()
    );
    let start = Instant::now();
    let export = export_complete_terminal_problem(&audit, "campaign-terminal-classification-v1")
        .expect("export the complete campaign terminal relation");
    let problem = export.problem();
    let binding = export.binding();
    eprintln!(
        "candidate=campaign-v1-terminal n={} m={} m_in={} families={} digest={} spartan_rows={} spartan_cols={} diagnostic_digest={} export_ms={}",
        problem.source.total_rows,
        problem.column_count,
        problem.public_input_count,
        problem.complete_families.len(),
        problem.source.artifact_digest,
        binding.spartan_rows(),
        binding.spartan_columns(),
        binding.diagnostic_digest(),
        start.elapsed().as_millis(),
    );
}

fn print_freeze_candidate(label: &str, params: Params) {
    let start = Instant::now();
    let audit = freeze_candidate_audit(&params);
    eprintln!("candidate={label} audit build: {} ms", start.elapsed().as_millis());

    for branch in [NebulaFPrimeBranch::Base, NebulaFPrimeBranch::Recursive] {
        let start = Instant::now();
        let export =
            export_complete_nebula_problem(&audit, branch, label).expect("export the complete freeze-candidate branch");
        let problem = export.problem();
        let binding = export.binding();
        eprintln!(
            "candidate={label} branch={branch:?} n={} m={} m_in={} families={} digest={} final_rows={} final_cols={} final_plan_digest={} export_ms={}",
            problem.source.total_rows,
            problem.column_count,
            problem.public_input_count,
            problem.complete_families.len(),
            problem.source.artifact_digest,
            binding.final_rows(),
            binding.final_columns(),
            binding.final_plan_digest(),
            start.elapsed().as_millis(),
        );
    }
}
