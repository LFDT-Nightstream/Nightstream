use std::sync::LazyLock;

use neo_fold_next::proof::FoldSchedule;
use neo_fold_next::rv64im::audit::build_rv64im_chunk_step_ivc_relations;
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::ivc::Rv64imIvcState;
use neo_fold_next::rv64im::{
    build_mixed_opcode_perf_source_case, prove_rv64im_accepted_proof_with_options, Rv64imChunkStepIvcRelation,
    Rv64imProofInput, Rv64imPublicProofOptions,
};

static TWO_STEP_RELATIONS: LazyLock<Vec<Rv64imChunkStepIvcRelation>> = LazyLock::new(|| {
    let source = build_mixed_opcode_perf_source_case(2);
    let max_steps = source.program_words.len();
    let input = Rv64imProofInput { source, max_steps };
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let (accepted, _) =
        prove_rv64im_accepted_proof_with_options(&input, options).expect("prove two-step accepted artifact");
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted).expect("prove two-step final statement");
    build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build two-step chunk-step relations")
});

#[test]
fn rv64im_ivc_base_state_round_trips_through_serde() {
    let state = Rv64imIvcState::init().expect("build canonical base IVC state");
    state.verify().expect("verify canonical base IVC state");

    let encoded = bincode::serialize(&state).expect("serialize canonical base IVC state");
    let decoded: Rv64imIvcState = bincode::deserialize(&encoded).expect("deserialize canonical base IVC state");

    decoded
        .verify()
        .expect("verify deserialized canonical base IVC state");
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
fn rv64im_ivc_deserialized_state_accepts_further_folds() {
    let relations = &*TWO_STEP_RELATIONS;
    assert!(
        relations.len() >= 2,
        "two-step canonical fixture must expose at least two chunk-step relations"
    );

    let one_shot = relations
        .iter()
        .try_fold(
            Rv64imIvcState::init().expect("build one-shot IVC base state"),
            |state, relation| state.append(relation),
        )
        .expect("append the two-step canonical fixture in one shot");
    one_shot
        .verify()
        .expect("verify one-shot two-step IVC state");

    let first_step = Rv64imIvcState::init()
        .expect("build resumed IVC base state")
        .append(&relations[0])
        .expect("append first fold before serialization");
    first_step
        .verify()
        .expect("verify first appended IVC state before serialization");

    let encoded = bincode::serialize(&first_step).expect("serialize partially folded IVC state");
    let decoded: Rv64imIvcState = bincode::deserialize(&encoded).expect("deserialize partially folded IVC state");
    decoded
        .verify()
        .expect("verify partially folded IVC state after deserialization");

    let resumed = relations
        .iter()
        .skip(1)
        .try_fold(decoded, |state, relation| state.append(relation))
        .expect("append the remaining folds after deserializing the IVC state");
    resumed.verify().expect("verify resumed two-step IVC state");

    assert_eq!(
        resumed.public_image(),
        one_shot.public_image(),
        "resumed native IVC append must land on the same public image as one-shot append"
    );
}
