//! Focused tests for the witness-owned RV32IM kernel export seam.

use neo_fold_prototype::rv32im::{
    build_rv32im_kernel_export_relation, build_rv32im_kernel_export_witness, parity_source_cases, prepared_step_digest,
    prove_rv32im_public_proof, verify_rv32im_kernel_export_witness, Rv32imProofInput,
};

fn source_case(name: &str) -> neo_fold_prototype::rv32im::Rv32imParitySourceCase {
    parity_source_cases()
        .into_iter()
        .find(|case| case.manifest.name == name)
        .unwrap_or_else(|| panic!("missing parity source case {name}"))
}

fn proof_input(name: &str) -> Rv32imProofInput {
    let source = source_case(name);
    let max_steps = source.program_words.len();
    Rv32imProofInput { source, max_steps }
}

#[test]
fn rv32im_kernel_export_witness_round_trip() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv32im_public_proof(&input).expect("prove rv32im public proof");
    let relation = build_rv32im_kernel_export_relation(&proof).expect("build kernel export relation");
    let witness = build_rv32im_kernel_export_witness(&proof).expect("build kernel export witness");

    assert_eq!(witness.chunk_handoffs.len(), relation.chunk_count as usize);
    assert_ne!(witness.digest, [0; 32]);
    assert!(witness
        .chunk_handoffs
        .iter()
        .all(|chunk| chunk.digest != [0; 32]));
    for chunk in &witness.chunk_handoffs {
        assert_eq!(chunk.bridge_handoff.step_bindings.len(), chunk.chunk_input.steps.len());
        for (chunk_local_index, (binding, step)) in chunk
            .bridge_handoff
            .step_bindings
            .iter()
            .zip(chunk.chunk_input.steps.iter())
            .enumerate()
        {
            assert_eq!(
                binding.logical_index,
                (chunk.chunk_input.start_index + chunk_local_index) as u64
            );
            assert_eq!(binding.prepared_step_digest, prepared_step_digest(step));
            assert_ne!(binding.row_binding_digest, [0; 32]);
            assert_ne!(binding.digest, [0; 32]);
        }
    }

    verify_rv32im_kernel_export_witness(&relation, &witness).expect("verify kernel export witness");
}

#[test]
fn rv32im_kernel_export_witness_rejects_tampered_chunk_input() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv32im_public_proof(&input).expect("prove rv32im public proof");
    let relation = build_rv32im_kernel_export_relation(&proof).expect("build kernel export relation");
    let mut witness = build_rv32im_kernel_export_witness(&proof).expect("build kernel export witness");
    witness.chunk_handoffs[0].chunk_input.steps[0]
        .label
        .push_str("_tampered");

    let err = verify_rv32im_kernel_export_witness(&relation, &witness).expect_err("tampered chunk input must fail");
    assert!(format!("{err}").contains("kernel export witness") || format!("{err}").contains("digest"));
}

#[test]
fn rv32im_kernel_export_witness_rejects_tampered_bridge_handoff() {
    let input = proof_input("control_flow_jal_skip_ecall");
    let proof = prove_rv32im_public_proof(&input).expect("prove rv32im public proof");
    let relation = build_rv32im_kernel_export_relation(&proof).expect("build kernel export relation");
    let mut witness = build_rv32im_kernel_export_witness(&proof).expect("build kernel export witness");
    witness.chunk_handoffs[0].bridge_handoff.step_bindings[0].prepared_step_digest[0] ^= 1;

    let err = verify_rv32im_kernel_export_witness(&relation, &witness).expect_err("tampered bridge handoff must fail");
    assert!(format!("{err}").contains("bridge") || format!("{err}").contains("digest"));
}
