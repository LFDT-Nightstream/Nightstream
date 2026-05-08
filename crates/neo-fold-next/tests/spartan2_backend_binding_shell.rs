use neo_fold_next::decider::spartan2::{
    prove_spartan2_backend_binding_shell_with_perf, prove_spartan2_decider_with_perf,
    setup_spartan2_backend_binding_shell, setup_spartan2_decider, verify_spartan2_backend_binding_shell,
    verify_spartan2_decider, Spartan2BackendBindingShellError, Spartan2ChunkTransitionBinding, Spartan2DeciderError,
    Spartan2DeciderStatement, Spartan2DeciderTarget, Spartan2DeciderWitness,
};
use neo_fold_next::finalize::FixedShapeChunkSummary;
use neo_fold_next::proof::FoldSchedule;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

fn synthetic_target() -> Spartan2DeciderTarget {
    synthetic_target_with_layout(FoldSchedule::RowsPerChunk(1), &[1])
}

fn synthetic_target_with_layout(fold_schedule: FoldSchedule, public_step_counts: &[u64]) -> Spartan2DeciderTarget {
    let mut start_index = 0u64;
    let chunk_summaries = public_step_counts
        .iter()
        .enumerate()
        .map(|(index, &public_step_count)| {
            let summary = FixedShapeChunkSummary {
                start_index,
                public_step_count,
                public_chunk_digest: [6u8.wrapping_add(index as u8); 32],
                chunk_relation_digest: [7u8.wrapping_add(index as u8); 32],
            };
            start_index += public_step_count;
            summary
        })
        .collect();
    let mut target = Spartan2DeciderTarget {
        statement: Spartan2DeciderStatement {
            public_statement_digest: [1u8; 32],
            relation_digest: [2u8; 32],
            final_proof_digest: [0u8; 32],
            initial_handle_digest: [F::from_u64(8); 4],
            terminal_handle_digest: [F::ZERO; 4],
            fold_schedule,
            semantic_step_count: start_index,
            chunk_summaries,
        },
        witness: Spartan2DeciderWitness {
            base_component_digests: vec![[4u8; 32]],
            chunk_transition_bindings: public_step_counts
                .iter()
                .enumerate()
                .map(|(index, _)| Spartan2ChunkTransitionBinding {
                    claimed_chunk_relation_digest: [7u8.wrapping_add(index as u8); 32],
                    transition_witness_digest: [5u8.wrapping_add(index as u8); 32],
                })
                .collect(),
        },
    };
    target.statement.terminal_handle_digest = target.statement.expected_terminal_handle_digest();
    target.statement.final_proof_digest = target.expected_final_proof_digest();
    target
}

fn padded_zero_summary(semantic_step_count: u64) -> FixedShapeChunkSummary {
    FixedShapeChunkSummary {
        start_index: semantic_step_count,
        public_step_count: 0,
        public_chunk_digest: [0u8; 32],
        chunk_relation_digest: [0u8; 32],
    }
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn spartan2_decider_backend_round_trip() {
    let target = synthetic_target();
    let (pk, vk) = setup_spartan2_decider(&target.shape()).expect("setup decider backend");
    let (proof, _) = prove_spartan2_decider_with_perf(&pk, &target).expect("prove decider backend");

    verify_spartan2_decider(&vk, &target, &proof).expect("verify decider backend");
    assert!(proof.snark_bytes_len() > 0);
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn spartan2_decider_backend_rejects_tampered_target() {
    let target = synthetic_target();
    let (pk, vk) = setup_spartan2_decider(&target.shape()).expect("setup decider backend");
    let (proof, _) = prove_spartan2_decider_with_perf(&pk, &target).expect("prove decider backend");

    let mut tampered = target.clone();
    tampered.statement.final_proof_digest[0] ^= 1;

    let err = verify_spartan2_decider(&vk, &tampered, &proof).expect_err("tampered target must fail");
    assert!(matches!(err, Spartan2DeciderError::FinalProofDigestMismatch));
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn spartan2_backend_binding_shell_rejects_tampered_public_final_proof_digest() {
    let target = synthetic_target();
    let relation = target.backend_relation();
    let (pk, vk) = setup_spartan2_backend_binding_shell(&relation.shape()).expect("setup backend-binding shell");

    let (proof, _) =
        prove_spartan2_backend_binding_shell_with_perf(&pk, &relation).expect("prove backend-binding shell");

    let mut tampered = relation.clone();
    tampered.statement.final_proof_digest[0] ^= 1;

    let err = verify_spartan2_backend_binding_shell(&vk, &tampered, &proof)
        .expect_err("tampered public final proof digest must fail");
    assert!(matches!(err, Spartan2BackendBindingShellError::RelationSurface(_)));
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn spartan2_decider_backend_rejects_tampered_public_chunk_count() {
    let target = synthetic_target();
    let (pk, vk) = setup_spartan2_decider(&target.shape()).expect("setup decider backend");
    let (proof, _) = prove_spartan2_decider_with_perf(&pk, &target).expect("prove decider backend");

    let mut tampered = target.clone();
    tampered
        .statement
        .chunk_summaries
        .push(padded_zero_summary(tampered.statement.semantic_step_count));

    let err = verify_spartan2_decider(&vk, &tampered, &proof).expect_err("tampered chunk count must fail");
    assert!(matches!(err, Spartan2DeciderError::RelationSurface(_)));
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn spartan2_decider_backend_rejects_tampered_public_semantic_step_count() {
    let target = synthetic_target();
    let (pk, vk) = setup_spartan2_decider(&target.shape()).expect("setup decider backend");
    let (proof, _) = prove_spartan2_decider_with_perf(&pk, &target).expect("prove decider backend");

    let mut tampered = target.clone();
    tampered.statement.semantic_step_count += 1;

    let err = verify_spartan2_decider(&vk, &tampered, &proof).expect_err("tampered semantic step count must fail");
    assert!(matches!(err, Spartan2DeciderError::RelationSurface(_)));
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn spartan2_decider_backend_rejects_tampered_public_chunk_summary() {
    let target = synthetic_target();
    let (pk, vk) = setup_spartan2_decider(&target.shape()).expect("setup decider backend");
    let (proof, _) = prove_spartan2_decider_with_perf(&pk, &target).expect("prove decider backend");

    let mut tampered = target.clone();
    tampered.statement.chunk_summaries[0].public_step_count += 1;

    let err = verify_spartan2_decider(&vk, &tampered, &proof).expect_err("tampered chunk summary must fail");
    assert!(matches!(
        err,
        Spartan2DeciderError::RelationSurface(_) | Spartan2DeciderError::FinalProofDigestMismatch
    ));
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn spartan2_decider_backend_rejects_zero_length_public_chunk_summary() {
    let target = synthetic_target();
    let (pk, _vk) = setup_spartan2_decider(&target.shape()).expect("setup decider backend");

    let mut tampered = target.clone();
    tampered.statement.chunk_summaries[0].public_step_count = 0;
    tampered.statement.semantic_step_count = 0;
    tampered.statement.terminal_handle_digest = tampered.statement.expected_terminal_handle_digest();
    tampered.statement.final_proof_digest = tampered.expected_final_proof_digest();

    let err = prove_spartan2_decider_with_perf(&pk, &tampered)
        .expect_err("zero-length chunk summary must fail at prove time");
    assert!(matches!(err, Spartan2DeciderError::RelationSurface(_)));
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn spartan2_decider_backend_rejects_invalid_public_fold_schedule() {
    let target = synthetic_target();
    let (pk, _vk) = setup_spartan2_decider(&target.shape()).expect("setup decider backend");

    let mut tampered = target.clone();
    tampered.statement.fold_schedule = FoldSchedule::RowsPerChunk(0);

    let err =
        prove_spartan2_decider_with_perf(&pk, &tampered).expect_err("invalid fold schedule must fail at prove time");
    assert!(matches!(err, Spartan2DeciderError::RelationSurface(_)));
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn spartan2_decider_backend_rejects_mismatched_private_chunk_relation_binding() {
    let target = synthetic_target();
    let (pk, _vk) = setup_spartan2_decider(&target.shape()).expect("setup decider backend");

    let mut tampered = target.clone();
    tampered.witness.chunk_transition_bindings[0].claimed_chunk_relation_digest[0] ^= 1;

    let err = prove_spartan2_decider_with_perf(&pk, &tampered)
        .expect_err("private chunk-relation binding mismatch must fail at prove time");
    assert!(matches!(err, Spartan2DeciderError::RelationSurface(_)));
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn spartan2_decider_backend_rejects_whole_trace_with_multiple_chunk_summaries() {
    let target = synthetic_target_with_layout(FoldSchedule::WholeTrace, &[1, 1]);
    let (pk, _vk) = setup_spartan2_decider(&target.shape()).expect("setup decider backend");

    let err =
        prove_spartan2_decider_with_perf(&pk, &target).expect_err("WholeTrace with multiple chunk summaries must fail");
    assert!(matches!(err, Spartan2DeciderError::RelationSurface(_)));
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn spartan2_decider_backend_rejects_rows_per_chunk_with_short_non_final_chunk() {
    let target = synthetic_target_with_layout(FoldSchedule::RowsPerChunk(2), &[1, 2]);
    let (pk, _vk) = setup_spartan2_decider(&target.shape()).expect("setup decider backend");

    let err = prove_spartan2_decider_with_perf(&pk, &target)
        .expect_err("RowsPerChunk schedule with a short non-final chunk must fail");
    assert!(matches!(err, Spartan2DeciderError::RelationSurface(_)));
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn spartan2_decider_backend_rejects_tampered_public_terminal_handle() {
    let target = synthetic_target();
    let (pk, vk) = setup_spartan2_decider(&target.shape()).expect("setup decider backend");
    let (proof, _) = prove_spartan2_decider_with_perf(&pk, &target).expect("prove decider backend");

    let mut tampered = target.clone();
    tampered.statement.terminal_handle_digest[0] += F::ONE;

    let err = verify_spartan2_decider(&vk, &tampered, &proof).expect_err("tampered terminal handle must fail");
    assert!(matches!(err, Spartan2DeciderError::RelationSurface(_)));
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn spartan2_decider_backend_rejects_shape_mismatch() {
    let target = synthetic_target();
    let mismatched = Spartan2DeciderTarget {
        statement: target.statement.clone(),
        witness: Spartan2DeciderWitness {
            base_component_digests: vec![[9u8; 32]],
            chunk_transition_bindings: vec![],
        },
    };
    let (pk, _vk) = setup_spartan2_decider(&target.shape()).expect("setup decider backend");

    let err = prove_spartan2_decider_with_perf(&pk, &mismatched).expect_err("shape mismatch must fail");
    assert!(matches!(err, Spartan2DeciderError::ShapeMismatch));
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn spartan2_backend_binding_shell_rejects_tampered_private_base_count() {
    let target = synthetic_target();
    let relation = target.backend_relation();
    let (pk, vk) = setup_spartan2_backend_binding_shell(&relation.shape()).expect("setup backend-binding shell");

    let mut tampered = relation.clone();
    tampered.witness.base_component_count += 1;

    match prove_spartan2_backend_binding_shell_with_perf(&pk, &tampered) {
        Err(Spartan2BackendBindingShellError::RelationSurface(_)) => {}
        Err(other) => panic!("unexpected backend-binding prove error: {other}"),
        Ok((proof, _)) => {
            let err = verify_spartan2_backend_binding_shell(&vk, &tampered, &proof)
                .expect_err("tampered private base count must fail at verify time");
            assert!(matches!(err, Spartan2BackendBindingShellError::RelationSurface(_)));
        }
    }
}
