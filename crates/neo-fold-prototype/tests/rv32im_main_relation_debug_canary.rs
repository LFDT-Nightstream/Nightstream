use std::env;

use neo_fold_prototype::rv32im::audit::{
    debug_check_rv32im_ivc_recursion_snark_circuit, prove_rv32im_public_proof_and_published_seam_with_perf,
};
use neo_fold_prototype::rv32im::final_relation::prove_rv32im_final_statement_from_accepted;
use neo_fold_prototype::rv32im::{
    build_mixed_opcode_perf_source_case, build_rv32im_accepted_proof_artifact, Rv32imProofInput,
    RV32IM_MIXED_OPCODE_PERF_DEFAULT_N,
};

fn perf_opcode_count_from_env() -> usize {
    match env::var("NS_DEBUG_N") {
        Ok(raw) => raw.parse().expect("NS_DEBUG_N must parse as usize"),
        Err(_) => RV32IM_MIXED_OPCODE_PERF_DEFAULT_N,
    }
}

#[test]
#[ignore = "manual debug canary for arbitrary-NS_DEBUG_N main decider satisfiability"]
fn rv32im_main_relation_debug_satisfiable() {
    let source = build_mixed_opcode_perf_source_case(perf_opcode_count_from_env());
    let input = Rv32imProofInput {
        max_steps: source.program_words.len(),
        source,
    };
    let ((proof, seam), _perf) =
        prove_rv32im_public_proof_and_published_seam_with_perf(&input).expect("build published seam");
    let artifact = build_rv32im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (_statement, final_proof) =
        prove_rv32im_final_statement_from_accepted(&artifact).expect("prove final statement");
    let final_statement = seam
        .rebuild_final_statement()
        .expect("rebuild final statement from the carried published seam");
    debug_check_rv32im_ivc_recursion_snark_circuit(&final_statement, &final_proof)
        .expect("debug check main spartan decider");
}
