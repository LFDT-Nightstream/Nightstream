use std::collections::BTreeSet;

use neo_fold_prototype::rv64im::{
    build_all_parity_cases, build_rv64im_audit_witness_bundle, prove_rv64im_public_proof,
    verify_rv64im_public_proof, Rv64imProofInput,
};

const EXPECTED_PUBLIC_PROOF_CASE_NAMES: &[&str] = &[
    "aligned_negative_offset_roundtrip",
    "control_flow_beq_taken_skip_ecall",
    "control_flow_bge_taken_skip_ecall",
    "control_flow_bgeu_taken_skip_ecall",
    "control_flow_blt_taken_skip_ecall",
    "control_flow_bltu_taken_skip_ecall",
    "control_flow_bne_taken_skip_ecall",
    "control_flow_ecall_only",
    "control_flow_jal_skip_ecall",
    "control_flow_jalr_skip_ecall",
    "multiply_high_mulh_mulhu_mulhsu_ecall",
    "multiply_low_mul_mulw_ecall",
    "narrow_memory_load_extract_extend_ecall",
    "narrow_memory_store_blend_ecall",
    "native_add_chain_x0_ecall",
    "native_logic_compare_chain_ecall",
    "native_shift_chain_ecall",
    "native_sub_lui_auipc_fence_ecall",
    "native_word_arith_chain_ecall",
    "native_word_shift_chain_ecall",
    "signed_divrem_chain_ecall",
    "unsigned_divrem_chain_ecall",
    "vertical_add_sd_ld_ecall",
];

fn expected_case_names() -> BTreeSet<&'static str> {
    EXPECTED_PUBLIC_PROOF_CASE_NAMES.iter().copied().collect()
}

#[test]
fn rv64im_public_proof_vectors_cover_the_full_parity_corpus() {
    let cases = build_all_parity_cases().expect("build RV64IM parity cases");
    let actual_names: BTreeSet<_> = cases
        .iter()
        .map(|(source, _)| source.manifest.name.as_str())
        .collect();

    assert_eq!(
        actual_names,
        expected_case_names(),
        "RV64IM public-proof vector corpus names changed",
    );
    assert_eq!(
        cases.len(),
        EXPECTED_PUBLIC_PROOF_CASE_NAMES.len(),
        "RV64IM public-proof vector corpus size changed",
    );

    for (source, derived) in cases {
        let name = source.manifest.name.as_str();
        let input = Rv64imProofInput {
            source: source.clone(),
            max_steps: source.program_words.len(),
        };
        let witness = build_rv64im_audit_witness_bundle(&input)
            .unwrap_or_else(|err| panic!("build RV64IM audit witness vector {name}: {err}"));
        let proof = prove_rv64im_public_proof(&input)
            .unwrap_or_else(|err| panic!("prove RV64IM public proof vector {name}: {err}"));
        verify_rv64im_public_proof(&proof)
            .unwrap_or_else(|err| panic!("verify RV64IM public proof vector {name}: {err}"));
        let verified = build_rv64im_audit_witness_bundle(&input)
            .unwrap_or_else(|err| panic!("rebuild RV64IM audit witness vector {name}: {err}"));

        assert_eq!(
            verified.digest, witness.digest,
            "proof witness digest roundtrip for {name}",
        );
        assert_eq!(
            proof.statement.execution_digest, derived.kernel.execution_digest,
            "statement execution digest matches derived kernel for {name}",
        );
        assert_eq!(
            proof.statement.final_state_digest, derived.kernel.final_state_digest,
            "statement final-state digest matches derived kernel for {name}",
        );
        assert_eq!(
            proof.statement.transcript_final_digest, derived.kernel.transcript_final_digest,
            "statement transcript digest matches derived kernel for {name}",
        );
    }
}
