use std::env;
use std::time::Instant;

use neo_fold_next::proof::FoldSchedule;
use neo_fold_next::rv64im::audit::build_rv64im_chunk_step_ivc_relations;
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::ivc::Rv64imIvcState;
use neo_fold_next::rv64im::ivc_snark::{
    prove_rv64im_ivc_snark_from_final, setup_rv64im_ivc_snark_cached, setup_rv64im_ivc_snark_from_final,
    setup_rv64im_ivc_snark_from_final_cached, verify_rv64im_ivc_snark_against_final,
};
use neo_fold_next::rv64im::{
    build_mixed_opcode_perf_source_case, prove_rv64im_accepted_proof_with_options, Rv64imChunkStepIvcRelation,
    Rv64imProofInput, Rv64imPublicProofOptions, RV64IM_MIXED_OPCODE_PERF_DEFAULT_N,
};

fn millis_since(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

fn perf_opcode_count_from_env() -> usize {
    match env::var("NS_DEBUG_N") {
        Ok(raw) => raw.parse().expect("NS_DEBUG_N must parse as usize"),
        Err(_) => RV64IM_MIXED_OPCODE_PERF_DEFAULT_N,
    }
}

struct ProbeFixture {
    final_statement: neo_fold_next::rv64im::final_relation::Rv64imFinalStatement,
    final_proof: neo_fold_next::rv64im::final_relation::Rv64imFinalBuildProof,
    relations: Vec<Rv64imChunkStepIvcRelation>,
}

fn build_fixture(opcode_count: usize) -> ProbeFixture {
    let source = build_mixed_opcode_perf_source_case(opcode_count);
    let max_steps = source.program_words.len();
    let input = Rv64imProofInput { source, max_steps };
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let (accepted, _) =
        prove_rv64im_accepted_proof_with_options(&input, options).expect("prove accepted artifact for closure probe");
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted).expect("prove final statement for closure probe");
    let relations = build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof)
        .expect("build chunk-step IVC relations for closure probe");
    ProbeFixture {
        final_statement,
        final_proof,
        relations,
    }
}

fn check_round_trip_resume() {
    let base_state = Rv64imIvcState::init().expect("build canonical base IVC state");
    base_state
        .verify()
        .expect("verify canonical base IVC state");

    let encoded = bincode::serialize(&base_state).expect("serialize canonical base IVC state");
    let decoded: Rv64imIvcState = bincode::deserialize(&encoded).expect("deserialize canonical base IVC state");
    decoded
        .verify()
        .expect("verify deserialized canonical base IVC state");
    assert_eq!(
        decoded.public_image(),
        base_state.public_image(),
        "serializing the canonical base IVC state must preserve the public image"
    );
    assert!(
        decoded.latest_terminal_statement().is_none(),
        "the canonical base IVC state must not invent a terminal statement during serialization"
    );

    let relations = build_fixture(2).relations;
    assert!(
        relations.len() >= 2,
        "two-step canonical fixture must expose at least two chunk-step relations"
    );

    let one_shot = relations
        .iter()
        .try_fold(
            Rv64imIvcState::init().expect("build one-shot IVC state"),
            |state, relation| state.append(relation),
        )
        .expect("append canonical two-step fixture in one shot");
    one_shot
        .verify()
        .expect("verify one-shot canonical two-step state");

    let first_step = Rv64imIvcState::init()
        .expect("build resumed IVC base state")
        .append(&relations[0])
        .expect("append first canonical relation");
    first_step
        .verify()
        .expect("verify first appended canonical state");

    let encoded = bincode::serialize(&first_step).expect("serialize partially folded canonical state");
    let decoded: Rv64imIvcState = bincode::deserialize(&encoded).expect("deserialize partially folded canonical state");
    decoded
        .verify()
        .expect("verify deserialized partially folded canonical state");

    let resumed = relations
        .iter()
        .skip(1)
        .try_fold(decoded, |state, relation| state.append(relation))
        .expect("append remaining canonical relations after resume");
    resumed
        .verify()
        .expect("verify resumed canonical two-step state");

    assert_eq!(
        resumed.public_image(),
        one_shot.public_image(),
        "resumed append must land on the same public image as one-shot append"
    );
}

fn main() {
    check_round_trip_resume();

    let opcode_count = perf_opcode_count_from_env();
    let fixture = build_fixture(opcode_count);
    assert!(
        !fixture.relations.is_empty(),
        "closure perf probe requires at least one chunk-step relation"
    );

    let native_append_started = Instant::now();
    let mut state = Rv64imIvcState::init().expect("build initial IVC state");
    for relation in &fixture.relations {
        state = state.append(relation).expect("append native IVC relation");
    }
    let native_append_ms = millis_since(native_append_started);

    let native_verify_started = Instant::now();
    state.verify().expect("verify native IVC state");
    let native_verify_ms = millis_since(native_verify_started);

    let compress_started = Instant::now();
    let snark = state.compress().expect("compress native IVC state");
    let compress_ms = millis_since(compress_started);

    let (fresh_pk, fresh_vk) = setup_rv64im_ivc_snark_from_final(&fixture.final_statement, &fixture.final_proof)
        .expect("setup fresh final-seam IVC SNARK keys");
    let fresh_snark = prove_rv64im_ivc_snark_from_final(
        &fresh_pk,
        &fixture.final_statement,
        &fixture.final_proof,
        state.public_image(),
    )
    .expect("prove fresh final-seam IVC SNARK");
    verify_rv64im_ivc_snark_against_final(&fresh_vk, &fixture.final_statement, &fixture.final_proof, &fresh_snark)
        .expect("verify fresh final-seam compressed IVC proof against final seam");

    let final_keys = setup_rv64im_ivc_snark_from_final_cached(&fixture.final_statement, &fixture.final_proof)
        .expect("setup final-seam IVC SNARK verifier key");
    verify_rv64im_ivc_snark_against_final(
        &final_keys.as_ref().1,
        &fixture.final_statement,
        &fixture.final_proof,
        &snark,
    )
    .expect("verify compressed IVC proof against final seam");

    let public_image = state.public_image();
    let keys = setup_rv64im_ivc_snark_cached(&state).expect("setup IVC SNARK verifier key");
    let compressed_verify_started = Instant::now();
    snark
        .verify(&keys.as_ref().1, &public_image)
        .expect("verify compressed IVC proof");
    let compressed_verify_ms = millis_since(compressed_verify_started);

    println!("native_append_ms={native_append_ms:.3}");
    println!("native_verify_ms={native_verify_ms:.3}");
    println!("compress_ms={compress_ms:.3}");
    println!("compressed_verify_ms={compressed_verify_ms:.3}");
}
