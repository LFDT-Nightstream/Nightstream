use neo_fold_next::core::finalize::FixedShapeChunkSummary;
use neo_fold_next::core::proof::FoldSchedule;
use neo_fold_next::decider::spartan2::{
    prove_spartan2_public_target_shell_with_perf, setup_spartan2_public_target_shell,
    verify_spartan2_public_target_shell, Spartan2ChunkTransitionBinding, Spartan2DeciderStatement,
    Spartan2DeciderTarget, Spartan2DeciderWitness, Spartan2PublicTargetShellError,
};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

fn synthetic_target() -> Spartan2DeciderTarget {
    let mut target = Spartan2DeciderTarget {
        statement: Spartan2DeciderStatement {
            public_statement_digest: [1u8; 32],
            relation_digest: [2u8; 32],
            final_proof_digest: [3u8; 32],
            initial_handle_digest: [F::from_u64(8); 4],
            terminal_handle_digest: [F::ZERO; 4],
            fold_schedule: FoldSchedule::RowsPerChunk(1),
            semantic_step_count: 1,
            chunk_summaries: vec![FixedShapeChunkSummary {
                start_index: 0,
                public_step_count: 1,
                public_chunk_digest: [6u8; 32],
                chunk_relation_digest: [7u8; 32],
            }],
        },
        witness: Spartan2DeciderWitness {
            base_component_digests: vec![[4u8; 32]],
            chunk_transition_bindings: vec![Spartan2ChunkTransitionBinding {
                claimed_chunk_relation_digest: [7u8; 32],
                transition_witness_digest: [5u8; 32],
            }],
        },
    };
    target.statement.terminal_handle_digest = target.statement.expected_terminal_handle_digest();
    target
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn spartan2_public_target_shell_round_trip() {
    let target = synthetic_target();
    let (pk, vk) = setup_spartan2_public_target_shell(&target.shape()).expect("setup public-target shell");
    let (proof, _) = prove_spartan2_public_target_shell_with_perf(&pk, &target).expect("prove public-target shell");

    verify_spartan2_public_target_shell(&vk, &target, &proof).expect("verify public-target shell");
    assert!(proof.snark_bytes_len() > 0);
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn spartan2_public_target_shell_rejects_tampered_target() {
    let target = synthetic_target();
    let (pk, vk) = setup_spartan2_public_target_shell(&target.shape()).expect("setup public-target shell");
    let (proof, _) = prove_spartan2_public_target_shell_with_perf(&pk, &target).expect("prove public-target shell");

    let mut tampered = target.clone();
    tampered.statement.final_proof_digest[0] ^= 1;

    let err = verify_spartan2_public_target_shell(&vk, &tampered, &proof).expect_err("tampered target must fail");
    assert!(matches!(err, Spartan2PublicTargetShellError::PublicIoMismatch));
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn spartan2_public_target_shell_setup_reuses_same_shape() {
    let target = synthetic_target();
    let mut other = target.clone();
    other.statement.public_statement_digest[0] ^= 1;
    other.statement.relation_digest[0] ^= 1;
    other.statement.final_proof_digest[0] ^= 1;
    other.witness.base_component_digests[0][0] ^= 1;
    other.witness.chunk_transition_bindings[0].transition_witness_digest[0] ^= 1;

    let (pk, vk) = setup_spartan2_public_target_shell(&target.shape()).expect("setup public-target shell");
    let (proof_a, _) =
        prove_spartan2_public_target_shell_with_perf(&pk, &target).expect("prove first public-target shell");
    let (proof_b, _) =
        prove_spartan2_public_target_shell_with_perf(&pk, &other).expect("prove second public-target shell");

    verify_spartan2_public_target_shell(&vk, &target, &proof_a).expect("verify first public-target shell");
    verify_spartan2_public_target_shell(&vk, &other, &proof_b).expect("verify second public-target shell");
}
