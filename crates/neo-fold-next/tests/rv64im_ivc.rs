use std::sync::LazyLock;

use neo_fold_next::proof::FoldSchedule;
use neo_fold_next::rv64im::audit::build_rv64im_chunk_step_ivc_relations;
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::ivc::{derive_rv64im_ivc_step_cap, Rv64imIvcState};
use neo_fold_next::rv64im::{
    build_mixed_opcode_perf_source_case, prove_rv64im_accepted_proof_with_options, Rv64imChunkStepIvcRelation,
    Rv64imProofInput, Rv64imPublicProofOptions,
};

fn build_relations(opcode_count: usize, schedule: FoldSchedule, label: &str) -> Vec<Rv64imChunkStepIvcRelation> {
    let source = build_mixed_opcode_perf_source_case(opcode_count);
    let max_steps = source.program_words.len();
    let input = Rv64imProofInput { source, max_steps };
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: schedule,
    };
    let (accepted, _) = prove_rv64im_accepted_proof_with_options(&input, options)
        .unwrap_or_else(|err| panic!("prove {label} accepted artifact: {err}"));
    let (final_statement, final_proof) = prove_rv64im_final_statement_from_accepted(&accepted)
        .unwrap_or_else(|err| panic!("prove {label} final statement: {err}"));
    build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof)
        .unwrap_or_else(|err| panic!("build {label} chunk-step relations: {err}"))
}

static TWO_STEP_RELATIONS: LazyLock<Vec<Rv64imChunkStepIvcRelation>> =
    LazyLock::new(|| build_relations(2, FoldSchedule::RowsPerChunk(1), "two-step"));

static FIVE_STEP_CAP_RELATIONS: LazyLock<Vec<Rv64imChunkStepIvcRelation>> =
    LazyLock::new(|| build_relations(7, FoldSchedule::RowsPerChunk(5), "five-step-cap"));

static WHOLE_TRACE_RELATIONS: LazyLock<Vec<Rv64imChunkStepIvcRelation>> =
    LazyLock::new(|| build_relations(5, FoldSchedule::WholeTrace, "whole-trace"));

#[test]
fn rv64im_ivc_base_state_round_trips_through_serde() {
    let state = Rv64imIvcState::init_with_step_cap(1).expect("build canonical base IVC state");
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
            Rv64imIvcState::init_with_step_cap(1).expect("build one-shot IVC base state"),
            |state, relation| state.append(relation),
        )
        .expect("append the two-step canonical fixture in one shot");
    one_shot
        .verify()
        .expect("verify one-shot two-step IVC state");

    let first_step = Rv64imIvcState::init_with_step_cap(1)
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

#[test]
fn rv64im_ivc_multi_step_family_survives_serde_and_resume() {
    let relations = &*FIVE_STEP_CAP_RELATIONS;
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
            Rv64imIvcState::init_with_step_cap(5).expect("build one-shot five-step-cap IVC base state"),
            |state, relation| state.append(relation),
        )
        .expect("append the five-step-cap fixture in one shot");
    assert_eq!(one_shot.step_cap(), 5);
    one_shot
        .verify()
        .expect("verify one-shot five-step-cap IVC state");

    let first_chunk = Rv64imIvcState::init_with_step_cap(5)
        .expect("build resumed five-step-cap IVC base state")
        .append(&relations[0])
        .expect("append the full-width five-step-cap chunk before serialization");
    first_chunk
        .verify()
        .expect("verify full-width five-step-cap state before serialization");

    let encoded = bincode::serialize(&first_chunk).expect("serialize five-step-cap partially folded IVC state");
    let decoded: Rv64imIvcState =
        bincode::deserialize(&encoded).expect("deserialize five-step-cap partially folded IVC state");
    assert_eq!(decoded.step_cap(), 5);
    decoded
        .verify()
        .expect("verify deserialized five-step-cap partially folded IVC state");

    let resumed = relations
        .iter()
        .skip(1)
        .try_fold(decoded, |state, relation| state.append(relation))
        .expect("append the remaining five-step-cap chunks after deserialization");
    resumed
        .verify()
        .expect("verify resumed five-step-cap IVC state");

    assert_eq!(
        resumed.public_image(),
        one_shot.public_image(),
        "resumed five-step-cap native IVC append must land on the same public image as one-shot append"
    );
}

#[test]
fn rv64im_ivc_whole_trace_family_round_trips_and_verifies() {
    let relations = &*WHOLE_TRACE_RELATIONS;
    assert_eq!(
        relations.len(),
        1,
        "whole-trace fixture must collapse into a single native relation"
    );
    let semantic_step_count = relations[0].statement.step_public.step_hi as usize;
    let step_cap =
        derive_rv64im_ivc_step_cap(FoldSchedule::WholeTrace, semantic_step_count).expect("derive whole-trace step_cap");
    assert_eq!(
        step_cap, relations[0].statement.chunk_summary.public_step_count as usize,
        "whole-trace family must freeze its step_cap to the authoritative public step count"
    );

    let state = Rv64imIvcState::init_with_step_cap(step_cap)
        .expect("build whole-trace IVC base state")
        .append(&relations[0])
        .expect("append whole-trace native relation");
    assert_eq!(state.step_cap(), step_cap as u64);
    state.verify().expect("verify whole-trace native IVC state");

    let encoded = bincode::serialize(&state).expect("serialize whole-trace IVC state");
    let decoded: Rv64imIvcState = bincode::deserialize(&encoded).expect("deserialize whole-trace IVC state");
    assert_eq!(decoded.step_cap(), step_cap as u64);
    decoded
        .verify()
        .expect("verify deserialized whole-trace IVC state");
    assert_eq!(
        decoded.public_image(),
        state.public_image(),
        "whole-trace native IVC serialization must preserve the public image"
    );
}
