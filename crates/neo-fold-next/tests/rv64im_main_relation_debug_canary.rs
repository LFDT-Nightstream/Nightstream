use std::env;

use neo_fold_next::rv64im::audit::{
    debug_check_rv64im_spartan2_decider_circuit, prove_rv64im_public_proof_and_published_seam_with_perf,
};
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::{
    build_mixed_opcode_perf_source_case, build_rv64im_accepted_proof_artifact, Rv64imProofInput,
    RV64IM_MIXED_OPCODE_PERF_DEFAULT_N,
};

fn perf_opcode_count_from_env() -> usize {
    match env::var("NS_DEBUG_N") {
        Ok(raw) => raw.parse().expect("NS_DEBUG_N must parse as usize"),
        Err(_) => RV64IM_MIXED_OPCODE_PERF_DEFAULT_N,
    }
}

#[test]
#[ignore = "manual debug canary for arbitrary-NS_DEBUG_N main decider satisfiability"]
fn rv64im_main_relation_debug_satisfiable() {
    let source = build_mixed_opcode_perf_source_case(perf_opcode_count_from_env());
    let input = Rv64imProofInput {
        max_steps: source.program_words.len(),
        source,
    };
    let ((proof, seam), _perf) =
        prove_rv64im_public_proof_and_published_seam_with_perf(&input).expect("build published seam");
    let artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&artifact).expect("prove final statement");
    debug_check_rv64im_spartan2_decider_circuit(
        seam.main_proof
            .final_statement_cache()
            .expect("locally built published seam should retain the final-statement cache"),
        &final_proof,
    )
    .expect("debug check main spartan decider");
}
