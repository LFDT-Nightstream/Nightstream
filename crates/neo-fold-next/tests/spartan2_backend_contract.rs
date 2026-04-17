#[path = "support/chip8.rs"]
mod chip8_support;

use neo_fold_next::chip8::decider::build_chip8_spartan2_decider_target;
use neo_fold_next::chip8::proof::prove_recursive;
use neo_fold_next::decider::spartan2::{
    Spartan2ChunkTransitionBinding, Spartan2DeciderShape, Spartan2DeciderStatement, Spartan2DeciderTarget,
    Spartan2DeciderWitness,
};
use neo_fold_next::finalize::FixedShapeChunkSummary;
use neo_fold_next::proof::FoldSchedule;
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

fn synthetic_target() -> Spartan2DeciderTarget {
    let mut target = Spartan2DeciderTarget {
        statement: Spartan2DeciderStatement {
            public_statement_digest: [1u8; 32],
            relation_digest: [2u8; 32],
            final_proof_digest: [0u8; 32],
            initial_handle_digest: [F::from_u64(8); 4],
            terminal_handle_digest: [F::ZERO; 4],
            fold_schedule: FoldSchedule::RowsPerChunk(1),
            chunk_count: 1,
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
    target.statement.final_proof_digest = target.expected_final_proof_digest();
    target
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn spartan2_backend_public_io_has_statement_prefix() {
    let target = synthetic_target();
    let statement_prefix = target.statement.public_io();
    let backend_public_io = target.backend_public_io();
    let backend_semantic_digest = target.backend_semantic_digest_fields();
    let backend_binding_digest = target.backend_binding_digest_fields();
    let semantic_start = statement_prefix.len();
    let binding_start = semantic_start + backend_semantic_digest.len();

    assert_eq!(
        &backend_public_io[..statement_prefix.len()],
        statement_prefix.as_slice()
    );
    assert!(backend_public_io.len() > statement_prefix.len());
    assert_eq!(
        &backend_public_io[semantic_start..binding_start],
        backend_semantic_digest.as_slice()
    );
    assert_eq!(&backend_public_io[binding_start..], backend_binding_digest.as_slice());
    assert_eq!(
        target.statement.final_proof_digest,
        target.expected_final_proof_digest()
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn spartan2_backend_public_io_changes_when_component_digest_changes() {
    let target = synthetic_target();
    let backend_public_io = target.backend_public_io();

    let mut tampered = target.clone();
    tampered.witness.base_component_digests[0][0] ^= 1;

    assert_ne!(tampered.witness.digest(), target.witness.digest());
    assert_ne!(tampered.backend_public_io(), backend_public_io);
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn spartan2_backend_public_io_changes_when_chunk_relation_binding_changes() {
    let target = synthetic_target();
    let backend_public_io = target.backend_public_io();

    let mut tampered = target.clone();
    tampered.witness.chunk_transition_bindings[0].claimed_chunk_relation_digest[0] ^= 1;

    assert_ne!(tampered.witness.digest(), target.witness.digest());
    assert_ne!(tampered.backend_public_io(), backend_public_io);
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn spartan2_backend_witness_is_order_sensitive() {
    let target = synthetic_target();
    let backend_witness = target.backend_witness();

    let mut swapped = target.clone();
    let binding = &mut swapped.witness.chunk_transition_bindings[0];
    std::mem::swap(
        &mut binding.claimed_chunk_relation_digest,
        &mut binding.transition_witness_digest,
    );

    assert_ne!(swapped.witness.digest(), target.witness.digest());
    assert_ne!(swapped.backend_public_io(), target.backend_public_io());
    assert_ne!(
        swapped.backend_witness().packed_fields(),
        backend_witness.packed_fields()
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn spartan2_backend_public_io_changes_when_statement_digest_changes() {
    let target = synthetic_target();
    let backend_witness = target.backend_witness().packed_fields();

    let mut tampered = target.clone();
    tampered.statement.relation_digest[0] ^= 1;

    assert_eq!(tampered.backend_witness().packed_fields(), backend_witness);
    assert_ne!(tampered.backend_public_io(), target.backend_public_io());
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn spartan2_backend_public_io_changes_when_public_chunk_count_changes() {
    let target = synthetic_target();
    let backend_witness = target.backend_witness().packed_fields();

    let mut tampered = target.clone();
    tampered.statement.chunk_count += 1;

    assert_eq!(tampered.backend_witness().packed_fields(), backend_witness);
    assert_ne!(tampered.backend_public_io(), target.backend_public_io());
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn spartan2_backend_public_io_changes_when_public_semantic_step_count_changes() {
    let target = synthetic_target();
    let backend_witness = target.backend_witness().packed_fields();

    let mut tampered = target.clone();
    tampered.statement.semantic_step_count += 1;

    assert_eq!(tampered.backend_witness().packed_fields(), backend_witness);
    assert_ne!(tampered.backend_public_io(), target.backend_public_io());
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn spartan2_backend_public_io_changes_when_public_fold_schedule_changes() {
    let target = synthetic_target();
    let backend_witness = target.backend_witness().packed_fields();

    let mut tampered = target.clone();
    tampered.statement.fold_schedule = FoldSchedule::WholeTrace;

    assert_eq!(tampered.backend_witness().packed_fields(), backend_witness);
    assert_ne!(tampered.backend_public_io(), target.backend_public_io());
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn spartan2_backend_public_io_changes_when_public_chunk_summary_changes() {
    let target = synthetic_target();
    let backend_witness = target.backend_witness().packed_fields();

    let mut tampered = target.clone();
    tampered.statement.chunk_summaries[0].public_step_count += 1;

    assert_eq!(tampered.backend_witness().packed_fields(), backend_witness);
    assert_ne!(tampered.backend_public_io(), target.backend_public_io());
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn chip8_spartan2_backend_contract_projects_generic_layout() {
    let input = chip8_support::build_jump_kernel_input(4);
    let (statement, proof) = prove_recursive(&input).expect("prove recursive");
    let target = build_chip8_spartan2_decider_target(&statement, &proof).expect("build chip8 decider target");

    assert_eq!(
        &target.backend_public_io()[..target.statement.public_io().len()],
        target.statement.public_io().as_slice()
    );
    assert_eq!(target.statement.chunk_count as usize, proof.steps.len());
    assert_eq!(target.backend_witness().base_component_count as usize, 1);
    assert_eq!(
        target.backend_witness().chunk_transition_count as usize,
        proof.steps.len()
    );
    assert_eq!(target.backend_witness().packed_fields(), target.witness.public_io());
    assert_eq!(
        &target.backend_public_io()[target.statement.public_io().len()
            ..target.statement.public_io().len() + target.backend_semantic_digest_fields().len()],
        target.backend_semantic_digest_fields().as_slice()
    );
    assert_eq!(
        &target.backend_public_io()
            [target.statement.public_io().len() + target.backend_semantic_digest_fields().len()..],
        target.backend_binding_digest_fields().as_slice()
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn spartan2_shape_tracks_public_and_backend_layout_lengths() {
    let target = synthetic_target();
    let shape = target.shape();

    assert_eq!(shape.public_io_len(), target.public_io().len());
    assert_eq!(shape.backend_public_io_len(), target.backend_public_io().len());
    assert_eq!(
        shape.backend_witness_field_len(),
        target.backend_witness().packed_fields().len()
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn spartan2_shape_depends_on_component_count_not_digest_values() {
    let target = synthetic_target();
    let mut other = target.clone();
    other.statement.public_statement_digest[0] ^= 1;
    other.statement.relation_digest[0] ^= 1;
    other.statement.final_proof_digest[0] ^= 1;
    other.statement.chunk_count += 7;
    other.witness.base_component_digests[0][0] ^= 1;
    other.witness.chunk_transition_bindings[0].transition_witness_digest[0] ^= 1;

    assert_eq!(target.shape(), other.shape());
    assert_eq!(target.shape().digest(), other.shape().digest());

    let different_shape = Spartan2DeciderShape {
        base_component_count: target.witness.base_component_digests.len(),
        chunk_transition_count: target.witness.chunk_transition_bindings.len() + 1,
    };
    assert_ne!(different_shape.digest(), target.shape().digest());
}
