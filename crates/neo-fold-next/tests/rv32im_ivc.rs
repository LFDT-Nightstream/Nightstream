use std::sync::LazyLock;

use neo_fold_next::finalize::FixedShapeChunkSummary;
use neo_fold_next::proof::FoldSchedule;
use neo_fold_next::rv32im::audit::{
    build_rv32im_chunk_step_ivc_relations, build_rv32im_published_proof_seam_with_perf,
};
use neo_fold_next::rv32im::construction2::Rv32imMainRecursionConstruction2PublicBoundary;
use neo_fold_next::rv32im::final_relation::prove_rv32im_final_statement_from_accepted;
use neo_fold_next::rv32im::ivc::{derive_rv32im_ivc_step_cap, Rv32imIvcPublicImage, Rv32imIvcState};
use neo_fold_next::rv32im::{
    build_mixed_opcode_perf_source_case, build_rv32im_accepted_proof_artifact,
    prove_rv32im_accepted_proof_with_options, prove_rv32im_public_proof_with_options, Rv32imChunkStepIvcRelation,
    Rv32imChunkStepIvcStatement, Rv32imChunkStepPublic, Rv32imEncodedPublicInput, Rv32imProofInput,
    Rv32imPublicProofOptions,
};
use neo_math::{D, F};
use neo_reductions::common::project_x_from_witness_mat;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

fn verified_step_statement_digest(statement: &Rv32imChunkStepIvcStatement) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/main_recursion_construction2_verified_step_statement");
    tr.append_message(
        b"neo.fold.next/rv32im/main_recursion_construction2_verified_step_statement/version",
        b"v2",
    );
    tr.append_u64s(
        b"neo.fold.next/rv32im/main_recursion_construction2_verified_step_statement/meta",
        &[
            statement.step_public.chunk_index,
            statement.step_public.step_lo,
            statement.step_public.step_hi,
            u64::from(statement.step_public.halted_out),
        ],
    );
    tr.append_fields(
        b"neo.fold.next/rv32im/main_recursion_construction2_verified_step_statement/state_in",
        &digest32_as_fields(statement.step_public.state_in),
    );
    tr.append_fields(
        b"neo.fold.next/rv32im/main_recursion_construction2_verified_step_statement/state_out",
        &digest32_as_fields(statement.step_public.state_out),
    );
    tr.append_fields(
        b"neo.fold.next/rv32im/main_recursion_construction2_verified_step_statement/public_chunk_digest",
        &digest32_as_fields(statement.chunk_summary.public_chunk_digest),
    );
    tr.append_fields(
        b"neo.fold.next/rv32im/main_recursion_construction2_verified_step_statement/chunk_relation_digest",
        &digest32_as_fields(statement.chunk_summary.chunk_relation_digest),
    );
    tr.digest32()
}

fn digest32_as_fields(digest: [u8; 32]) -> [F; 4] {
    [
        F::from_u64(u64::from_le_bytes(digest[0..8].try_into().expect("digest limb 0"))),
        F::from_u64(u64::from_le_bytes(digest[8..16].try_into().expect("digest limb 1"))),
        F::from_u64(u64::from_le_bytes(digest[16..24].try_into().expect("digest limb 2"))),
        F::from_u64(u64::from_le_bytes(digest[24..32].try_into().expect("digest limb 3"))),
    ]
}

fn build_relations(opcode_count: usize, schedule: FoldSchedule, label: &str) -> Vec<Rv32imChunkStepIvcRelation> {
    let source = build_mixed_opcode_perf_source_case(opcode_count);
    let max_steps = source.program_words.len();
    let input = Rv32imProofInput { source, max_steps };
    let options = Rv32imPublicProofOptions {
        root_fold_schedule: schedule,
    };
    let (accepted, _) = prove_rv32im_accepted_proof_with_options(&input, options)
        .unwrap_or_else(|err| panic!("prove {label} accepted artifact: {err}"));
    let (final_statement, final_proof) = prove_rv32im_final_statement_from_accepted(&accepted)
        .unwrap_or_else(|err| panic!("prove {label} final statement: {err}"));
    build_rv32im_chunk_step_ivc_relations(&final_statement, &final_proof)
        .unwrap_or_else(|err| panic!("build {label} chunk-step relations: {err}"))
}

static TWO_STEP_RELATIONS: LazyLock<Vec<Rv32imChunkStepIvcRelation>> =
    LazyLock::new(|| build_relations(2, FoldSchedule::RowsPerChunk(1), "two-step"));
static FIVE_STEP_CAP_RELATIONS: LazyLock<Vec<Rv32imChunkStepIvcRelation>> =
    LazyLock::new(|| build_relations(7, FoldSchedule::RowsPerChunk(5), "five-step-cap"));
static WHOLE_TRACE_RELATIONS: LazyLock<Vec<Rv32imChunkStepIvcRelation>> =
    LazyLock::new(|| build_relations(5, FoldSchedule::WholeTrace, "whole-trace"));

fn two_step_relations() -> &'static [Rv32imChunkStepIvcRelation] {
    &TWO_STEP_RELATIONS
}

fn five_step_cap_relations() -> &'static [Rv32imChunkStepIvcRelation] {
    &FIVE_STEP_CAP_RELATIONS
}

fn whole_trace_relations() -> &'static [Rv32imChunkStepIvcRelation] {
    &WHOLE_TRACE_RELATIONS
}

#[test]
fn rv32im_ivc_base_state_round_trips_through_serde() {
    let state = Rv32imIvcState::init_with_step_cap(1).expect("build canonical base IVC state");

    let encoded = bincode::serialize(&state).expect("serialize canonical base IVC state");
    let decoded: Rv32imIvcState = bincode::deserialize(&encoded).expect("deserialize canonical base IVC state");

    assert_eq!(
        decoded.public_image(),
        state.public_image(),
        "serializing the canonical base IVC state must preserve the public image"
    );
    assert!(
        decoded.latest_terminal_statement().is_none(),
        "the canonical base IVC state must not invent a terminal statement during serialization"
    );
}

#[test]
fn rv32im_ivc_deserialized_state_accepts_further_folds() {
    let relations = two_step_relations();
    assert!(
        relations.len() >= 2,
        "two-step canonical fixture must expose at least two chunk-step relations"
    );

    let one_shot = relations
        .iter()
        .try_fold(
            Rv32imIvcState::init_with_step_cap(1).expect("build one-shot IVC base state"),
            |state, relation| state.append(relation),
        )
        .expect("append the two-step canonical fixture in one shot");

    let first_step = Rv32imIvcState::init_with_step_cap(1)
        .expect("build resumed IVC base state")
        .append(&relations[0])
        .expect("append first fold before serialization");

    let encoded = bincode::serialize(&first_step).expect("serialize partially folded IVC state");
    let decoded: Rv32imIvcState = bincode::deserialize(&encoded).expect("deserialize partially folded IVC state");

    let resumed = relations
        .iter()
        .skip(1)
        .try_fold(decoded, |state, relation| state.append(relation))
        .expect("append the remaining folds after deserializing the IVC state");

    assert_eq!(
        resumed.public_image(),
        one_shot.public_image(),
        "resumed native IVC append must land on the same public image as one-shot append"
    );
}

#[test]
fn rv32im_ivc_append_rejects_stale_projection_digest_cargo() {
    let relation = two_step_relations()
        .first()
        .expect("two-step canonical fixture must expose at least one relation");
    let mut tampered = relation.clone();
    tampered.witness.state_out.carry.main_projection_digests[0][0] += F::ONE;

    Rv32imIvcState::init_with_step_cap(1)
        .expect("build canonical base IVC state")
        .append(&tampered)
        .expect_err("append must reject stale CE projection digest cargo");
}

#[test]
fn rv32im_ivc_final_carry_ce_x_projects_from_final_witnesses() {
    let state = two_step_relations()
        .iter()
        .try_fold(
            Rv32imIvcState::init_with_step_cap(1).expect("build canonical IVC base state"),
            |state, relation| state.append(relation),
        )
        .expect("append two-step fixture");
    let carry = &state.running_state().carry.main;
    assert_eq!(
        carry.claims.len(),
        carry.witnesses.len(),
        "final carry must pair every CE claim with a witness"
    );
    for (idx, (claim, witness)) in carry.claims.iter().zip(carry.witnesses.iter()).enumerate() {
        let expected_m = witness
            .cols()
            .checked_mul(D)
            .expect("packed witness width must not overflow");
        let projected = project_x_from_witness_mat(witness, expected_m, claim.m_in)
            .unwrap_or_else(|err| panic!("project final CE claim {idx} X from witness: {err}"));
        assert_eq!(
            claim.X, projected,
            "final carried CE claim {idx} must satisfy X = L_in(Z)"
        );
    }
}

#[test]
fn rv32im_ivc_multi_step_family_survives_serde_and_resume() {
    let relations = five_step_cap_relations();
    assert!(
        relations.len() >= 2,
        "five-step-cap fixture should expose at least two native relations"
    );
    assert_eq!(
        relations[0].statement.chunk_summary.public_step_count, 5,
        "first five-step-cap relation must be a full-width non-terminal chunk"
    );
    assert!(
        !relations[0].witness.terminal_step,
        "first five-step-cap relation must remain non-terminal"
    );
    assert!(
        relations
            .last()
            .expect("five-step-cap fixture must have a final relation")
            .witness
            .terminal_step,
        "final five-step-cap relation must be terminal"
    );

    let one_shot = relations
        .iter()
        .try_fold(
            Rv32imIvcState::init_with_step_cap(5).expect("build one-shot five-step-cap IVC base state"),
            |state, relation| state.append(relation),
        )
        .expect("append the five-step-cap fixture in one shot");
    assert_eq!(one_shot.step_cap(), 5);

    let first_chunk = Rv32imIvcState::init_with_step_cap(5)
        .expect("build resumed five-step-cap IVC base state")
        .append(&relations[0])
        .expect("append the full-width five-step-cap chunk before serialization");

    let encoded = bincode::serialize(&first_chunk).expect("serialize five-step-cap partially folded IVC state");
    let decoded: Rv32imIvcState =
        bincode::deserialize(&encoded).expect("deserialize five-step-cap partially folded IVC state");
    assert_eq!(decoded.step_cap(), 5);

    let resumed = relations
        .iter()
        .skip(1)
        .try_fold(decoded, |state, relation| state.append(relation))
        .expect("append the remaining five-step-cap chunks after deserialization");

    assert_eq!(
        resumed.public_image(),
        one_shot.public_image(),
        "resumed five-step-cap native IVC append must land on the same public image as one-shot append"
    );
}

#[test]
fn rv32im_ivc_whole_trace_family_round_trips_and_preserves_append_state() {
    let relations = whole_trace_relations();
    assert_eq!(
        relations.len(),
        1,
        "whole-trace fixture must collapse into a single native relation"
    );
    let semantic_step_count = relations[0].statement.step_public.step_hi as usize;
    let step_cap =
        derive_rv32im_ivc_step_cap(FoldSchedule::WholeTrace, semantic_step_count).expect("derive whole-trace step_cap");
    assert_eq!(
        step_cap, relations[0].statement.chunk_summary.public_step_count as usize,
        "whole-trace family must freeze its step_cap to the authoritative public step count"
    );

    let state = Rv32imIvcState::init_with_step_cap(step_cap)
        .expect("build whole-trace IVC base state")
        .append(&relations[0])
        .expect("append whole-trace native relation");
    assert_eq!(state.step_cap(), step_cap as u64);

    let encoded = bincode::serialize(&state).expect("serialize whole-trace IVC state");
    let decoded: Rv32imIvcState = bincode::deserialize(&encoded).expect("deserialize whole-trace IVC state");
    assert_eq!(decoded.step_cap(), step_cap as u64);
    assert_eq!(
        decoded.public_image(),
        state.public_image(),
        "whole-trace native IVC serialization must preserve the public image"
    );
}

#[test]
fn rv32im_ivc_public_image_rejects_terminal_metadata_tamper() {
    let construction2_u_i = Rv32imMainRecursionConstruction2PublicBoundary {
        fresh_instance_digest: [0; 32],
        commitment_digest: [0; 32],
        commitment_d: D as u64,
        commitment_kappa: 1,
        commitment_data: vec![F::from_u64(11); D],
        x_i: Rv32imEncodedPublicInput::from_digest_bytes([4; 32]),
    };
    let construction2_u_i = Rv32imMainRecursionConstruction2PublicBoundary {
        commitment_digest: construction2_u_i.expected_commitment_digest(),
        fresh_instance_digest: construction2_u_i.expected_fresh_instance_digest(),
        ..construction2_u_i
    };
    let terminal_statement = Rv32imChunkStepIvcStatement {
        step_public: Rv32imChunkStepPublic {
            program_digest: [7; 32],
            chunk_index: 0,
            step_lo: 0,
            step_hi: 2,
            state_in: [2; 32],
            state_out: [3; 32],
            halted_out: true,
        },
        chunk_summary: FixedShapeChunkSummary {
            start_index: 0,
            public_step_count: 2,
            public_chunk_digest: [8; 32],
            chunk_relation_digest: [9; 32],
        },
    };
    let public_image = Rv32imIvcPublicImage {
        vk_fs_digest: [1; 32],
        chunk_count: 1,
        step_count: 2,
        z_0: [2; 32],
        z_i: [3; 32],
        pc: 1,
        x_i: Rv32imEncodedPublicInput::from_digest_bytes([4; 32]),
        construction2_u_i,
        folded_accumulator_digest: [5; 32],
        terminal_bridge_handoff_digest: [6; 32],
        terminal_verified_step_statement_digest: verified_step_statement_digest(&terminal_statement),
        terminal_statement: Some(terminal_statement),
    };
    public_image
        .validate_final_construction2_public_boundary()
        .expect("canonical public image must satisfy the compressed verifier boundary");

    let mut missing_terminal = public_image.clone();
    missing_terminal.terminal_statement = None;
    missing_terminal
        .validate_final_construction2_public_boundary()
        .expect_err("compressed verifier boundary must require terminal metadata");

    let mut unhalted = public_image.clone();
    unhalted
        .terminal_statement
        .as_mut()
        .expect("terminal statement")
        .step_public
        .halted_out = false;
    unhalted
        .validate_final_construction2_public_boundary()
        .expect_err("compressed verifier boundary must reject an unhalted terminal chunk");

    let mut wrong_terminal_state = public_image.clone();
    wrong_terminal_state
        .terminal_statement
        .as_mut()
        .expect("terminal statement")
        .step_public
        .state_out[0] ^= 1;
    wrong_terminal_state
        .validate_final_construction2_public_boundary()
        .expect_err("compressed verifier boundary must bind terminal state_out to z_i");

    let mut wrong_construction2_x = public_image.clone();
    wrong_construction2_x.construction2_u_i.x_i = Rv32imEncodedPublicInput::from_digest_bytes([12; 32]);
    wrong_construction2_x
        .validate_final_construction2_public_boundary()
        .expect_err("compressed verifier boundary must bind final Construction-2 u_i.x_i to x_i");

    let mut noncanonical_x = public_image.clone();
    let mut noncanonical_x_bytes = [0u8; 32];
    noncanonical_x_bytes[..8].copy_from_slice(&0xffff_ffff_0000_0001u64.to_le_bytes());
    let noncanonical_x_i = Rv32imEncodedPublicInput::from_digest_bytes(noncanonical_x_bytes);
    noncanonical_x.x_i = noncanonical_x_i.clone();
    noncanonical_x.construction2_u_i.x_i = noncanonical_x_i;
    noncanonical_x.construction2_u_i.fresh_instance_digest = noncanonical_x
        .construction2_u_i
        .expected_fresh_instance_digest();
    noncanonical_x
        .validate_final_construction2_public_boundary()
        .expect_err("compressed verifier boundary must reject non-canonical x_i field-limb bytes");

    let mut wrong_construction2_digest = public_image.clone();
    wrong_construction2_digest
        .construction2_u_i
        .fresh_instance_digest[0] ^= 1;
    wrong_construction2_digest
        .validate_final_construction2_public_boundary()
        .expect_err("compressed verifier boundary must bind final Construction-2 u_i digest to its public parts");

    let mut wrong_construction2_commitment_digest = public_image.clone();
    wrong_construction2_commitment_digest
        .construction2_u_i
        .commitment_digest[0] ^= 1;
    wrong_construction2_commitment_digest
        .validate_final_construction2_public_boundary()
        .expect_err("compressed verifier boundary must bind final Construction-2 commitment digest to public C");

    let mut wrong_construction2_commitment_data = public_image.clone();
    wrong_construction2_commitment_data
        .construction2_u_i
        .commitment_data[0] += F::from_u64(1);
    wrong_construction2_commitment_data
        .validate_final_construction2_public_boundary()
        .expect_err("compressed verifier boundary must reject mutated final Construction-2 commitment data");

    let mut wrong_construction2_commitment_shape = public_image.clone();
    wrong_construction2_commitment_shape
        .construction2_u_i
        .commitment_data
        .pop();
    wrong_construction2_commitment_shape
        .validate_final_construction2_public_boundary()
        .expect_err("compressed verifier boundary must reject non-canonical final Construction-2 commitment shape");
}

#[test]
#[ignore = "expensive: published proof seam construction exceeds the local test budget"]
fn rv32im_published_seam_ivc_public_image_matches_direct_native_ivc_state() {
    let source = build_mixed_opcode_perf_source_case(2);
    let max_steps = source.program_words.len();
    let input = Rv32imProofInput { source, max_steps };
    let options = Rv32imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };

    let public_proof = prove_rv32im_public_proof_with_options(&input, options).expect("prove two-step public proof");
    let (published_seam, _) =
        build_rv32im_published_proof_seam_with_perf(&public_proof).expect("build published proof seam");
    let accepted_artifact = build_rv32im_accepted_proof_artifact(&public_proof).expect("build accepted proof artifact");
    let (final_statement, final_proof) =
        prove_rv32im_final_statement_from_accepted(&accepted_artifact).expect("prove final statement");
    let relations =
        build_rv32im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step relations");
    let direct_state = relations
        .iter()
        .try_fold(
            Rv32imIvcState::init_with_step_cap(1).expect("build direct native IVC base state"),
            |state, relation| state.append(relation),
        )
        .expect("append direct native IVC relations");

    assert_eq!(
        published_seam.main_proof.ivc_snark().public_image(),
        &direct_state.public_image(),
        "published seam must preserve the same native IVC public image as direct compression"
    );
    assert_eq!(
        published_seam.main_proof.published_statement().pc_final(),
        public_proof.statement.final_pc,
        "published statement must still carry the authoritative architectural final pc"
    );
}
