//! Digest measurement for the campaign profile freeze (bar 2).
//!
//! The `#[ignore]` printer measures candidate profiles. The freeze document
//! and its drift test pin the chosen digests.

use std::time::Instant;

use neo_fold_clean::frontends::nebula::f_prime::{NebulaFPrimeBranch, NebulaFPrimeRelation};
use neo_fold_clean::frontends::nebula::layout::NebulaParams;
use neo_fold_clean::frontends::nebula::plan::NebulaPlan;
use neo_fold_clean::paper::params::Params;
use nightstream_constraint_exporter::export_complete_nebula_problem;

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
