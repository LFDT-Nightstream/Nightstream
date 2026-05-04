use std::env;

use neo_fold_next::rv32im::audit::{
    audit_rv32im_public_proof_against_input_with_perf, audit_rv32im_public_proof_with_perf,
};
use neo_fold_next::rv32im::{
    build_mixed_opcode_perf_source_case, prove_rv32im_public_proof_with_perf, Rv32imProofInput,
    Rv32imPublicProofVerifyPerf, RV32IM_MIXED_OPCODE_PERF_DEFAULT_N,
};

fn perf_opcode_count_from_env() -> usize {
    match env::var("NS_DEBUG_N") {
        Ok(raw) => raw.parse().expect("NS_DEBUG_N must parse as usize"),
        Err(_) => RV32IM_MIXED_OPCODE_PERF_DEFAULT_N,
    }
}

fn per_unit(ms: f64, units: usize) -> f64 {
    if units == 0 {
        0.0
    } else {
        ms / units as f64
    }
}

fn print_root_main_lane_audit_breakdown(label: &str, perf: &Rv32imPublicProofVerifyPerf, opcode_count: usize) {
    let root_main_lane = &perf.root_main_lane;
    let session = &root_main_lane.session;
    println!();
    println!("{label}");
    println!("{}", "=".repeat(label.len()));
    println!("  {:28} {}", "chunk_count", session.chunk_count());
    println!("  {:28} {}", "fresh_steps", session.fresh_steps());
    println!(
        "  {:28} {:>12.3}",
        "root_main_lane_total_ms", perf.root_main_lane_proof_ms
    );
    println!(
        "  {:28} {:>12.3}",
        "prepare_public_steps_ms", root_main_lane.prepare_public_steps_ms
    );
    println!(
        "  {:28} {:>12.3}",
        "public_chunk_match_ms", root_main_lane.public_chunk_match_ms
    );
    println!(
        "  {:28} {:>12.3}",
        "packaged_statement_digest_ms", root_main_lane.packaged_statement_digest_ms
    );
    println!(
        "  {:28} {:>12.3}",
        "packaged_chunk_digests_ms", root_main_lane.packaged_chunk_digests_ms
    );
    println!(
        "  {:28} {:>12.3}",
        "packaged_final_claim_digests_ms", root_main_lane.packaged_final_main_claim_digests_ms
    );
    println!(
        "  {:28} {:>12.3}",
        "packaged_statement_hash_ms", root_main_lane.packaged_statement_hash_ms
    );
    println!(
        "  {:28} {:>12.3}",
        "packaged_schedule_checks_ms", root_main_lane.packaged_schedule_checks_ms
    );
    println!(
        "  {:28} {:>12.3}",
        "packaged_proof_digest_ms", root_main_lane.packaged_proof_digest_ms
    );
    println!(
        "  {:28} {:>12.3}",
        "packaged_final_claim_match_ms", root_main_lane.packaged_final_claim_match_ms
    );
    println!(
        "  {:28} {:>12.3}",
        "packaged_total_ms", root_main_lane.packaged_total_ms
    );
    println!("  {:28} {:>12.3}", "session_total_ms", session.total_ms);
    println!();
    println!("  {:28} {:>12} {:>14}", "phase", "wall ms", "ms/op");
    for (phase, ms) in [
        ("prepare_inputs", session.prepare_inputs_ms()),
        ("ccs_total", session.ccs_ms()),
        ("ccs_bind", session.ccs_bind_ms()),
        ("ccs_bind_header", session.ccs_bind_header_instances_ms()),
        ("ccs_bind_prefix", session.ccs_bind_header_prefix_ms()),
        ("ccs_bind_poly", session.ccs_bind_header_poly_ms()),
        (
            "ccs_bind_public_instances",
            session.ccs_bind_header_public_instances_ms(),
        ),
        ("ccs_bind_me_inputs", session.ccs_bind_me_inputs_ms()),
        ("ccs_bind_challenges", session.ccs_bind_sample_challenges_ms()),
        ("ccs_fe_sumcheck", session.ccs_fe_sumcheck_ms()),
        ("ccs_nc_sumcheck", session.ccs_nc_sumcheck_ms()),
        ("ccs_output_checks", session.ccs_output_checks_ms()),
        ("ccs_terminal", session.ccs_terminal_ms()),
        ("digest_checks", session.digest_checks_ms()),
        ("dims", session.dims_ms()),
        ("rlc_challenge", session.rlc_challenge_ms()),
        ("rlc_rho_mats", session.rlc_rho_mats_ms()),
        ("rlc_rho_k_lift", session.rlc_rho_k_lift_ms()),
        ("rlc_x", session.rlc_x_ms()),
        ("rlc_y", session.rlc_y_ms()),
        ("rlc_y_zcol", session.rlc_y_zcol_ms()),
        ("rlc_aux", session.rlc_aux_ms()),
        ("rlc_commitment_collect", session.rlc_commitment_collect_ms()),
        ("rlc_commitment_mix", session.rlc_commitment_mix_ms()),
        ("rlc_commitment", session.rlc_commitment_ms()),
        ("rlc_public", session.rlc_ms()),
        ("dec_public", session.dec_ms()),
    ] {
        println!("  {:28} {:>12.3} {:>14.4}", phase, ms, per_unit(ms, opcode_count));
    }
}

#[test]
#[ignore = "performance/debugging snapshot; run with --release -- --ignored --nocapture"]
fn rv32im_root_main_lane_verify_perf_snapshot() {
    let opcode_count = perf_opcode_count_from_env();
    let source = build_mixed_opcode_perf_source_case(opcode_count);
    let total_opcodes = source.program_words.len();
    let input = Rv32imProofInput {
        source,
        max_steps: total_opcodes,
    };
    let (proof, _) = prove_rv32im_public_proof_with_perf(&input).expect("prove rv32im public proof");
    let accepted_perf = audit_rv32im_public_proof_with_perf(&proof).expect("audit rv32im public proof");
    let replay_perf = audit_rv32im_public_proof_against_input_with_perf(&input, &proof)
        .expect("audit rv32im public proof against input");

    print_root_main_lane_audit_breakdown("Accepted Root Main Lane Audit", &accepted_perf, opcode_count);
    print_root_main_lane_audit_breakdown("Input-Consistency Root Main Lane Audit", &replay_perf, opcode_count);

    assert_eq!(
        accepted_perf.root_main_lane.session.chunk_count(),
        replay_perf.root_main_lane.session.chunk_count(),
        "accepted and input-consistency root-main-lane audits must cover the same chunk count",
    );
    assert!(
        accepted_perf.root_main_lane.session.chunk_count() > 0,
        "mixed-opcode snapshot must exercise at least one root-main-lane chunk",
    );
}
