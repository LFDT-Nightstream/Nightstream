use neo_fold_next::proof::FoldSchedule;
use std::time::Instant;

use neo_fold_next::rv64im::audit::debug_check_rv64im_terminal_decider_circuit;
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::ivc_snark::{setup_rv64im_ivc_snark_from_final, setup_rv64im_ivc_snark_from_final_cached};
use neo_fold_next::rv64im::main_proof::Rv64imCompressedMainProof;
use neo_fold_next::rv64im::{
    build_mixed_opcode_perf_source_case, build_rv64im_accepted_proof_artifact, parity_source_cases,
    prove_rv64im_public_proof_with_options, Rv64imProofInput, Rv64imPublicProofOptions,
};

fn should_stop_after_debug_check() -> bool {
    std::env::args().any(|arg| arg == "--stop-after-debug-check")
}

fn millis_since(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

fn proof_input_from_parity_case(name: &str) -> Rv64imProofInput {
    let source = parity_source_cases()
        .into_iter()
        .find(|case| case.manifest.name == name)
        .unwrap_or_else(|| panic!("missing parity source case {name}"));
    let max_steps = source.program_words.len();
    Rv64imProofInput { source, max_steps }
}

fn proof_input_from_mixed_opcode(opcode_count: usize) -> Rv64imProofInput {
    let source = build_mixed_opcode_perf_source_case(opcode_count);
    let max_steps = source.program_words.len();
    Rv64imProofInput { source, max_steps }
}

fn run_case(label: &str, input: Rv64imProofInput) {
    println!("case={label}");

    let prove_started = Instant::now();
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let public_proof = prove_rv64im_public_proof_with_options(&input, options).expect("prove public proof");
    println!("  public_proof_ms={:.3}", millis_since(prove_started));

    let accepted_started = Instant::now();
    let accepted = build_rv64im_accepted_proof_artifact(&public_proof).expect("build accepted proof artifact");
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted).expect("prove final statement");
    println!("  final_seam_ms={:.3}", millis_since(accepted_started));

    let debug_started = Instant::now();
    match debug_check_rv64im_terminal_decider_circuit(&final_statement, &final_proof) {
        Ok(()) => println!("  debug_check_ms={:.3}", millis_since(debug_started)),
        Err(err) => println!("  debug_check_err={} ({:.3} ms)", err, millis_since(debug_started)),
    }
    if should_stop_after_debug_check() {
        return;
    }

    let compress_started = Instant::now();
    let compressed = Rv64imCompressedMainProof::from_verified_final_seam(
        &final_statement,
        &final_proof,
        public_proof.statement.final_pc,
    )
    .expect("build compressed main proof");
    println!("  compressed_main_proof_ms={:.3}", millis_since(compress_started));

    let cached_verify_started = Instant::now();
    let cached_keys = setup_rv64im_ivc_snark_from_final_cached(&final_statement, &final_proof)
        .expect("setup cached terminal decider");
    compressed
        .verify(&cached_keys.as_ref().1)
        .expect("verify compressed main proof with cached setup");
    println!(
        "  cached_compressed_verify_ms={:.3}",
        millis_since(cached_verify_started)
    );

    let verify_started = Instant::now();
    let (_pk, vk) = setup_rv64im_ivc_snark_from_final(&final_statement, &final_proof).expect("setup terminal decider");
    compressed
        .verify(&vk)
        .expect("verify compressed main proof with fresh setup");
    println!("  fresh_compressed_verify_ms={:.3}", millis_since(verify_started));
}

fn main() {
    run_case(
        "parity:control_flow_jal_skip_ecall",
        proof_input_from_parity_case("control_flow_jal_skip_ecall"),
    );
    run_case("mixed_opcode:0", proof_input_from_mixed_opcode(0));
}
