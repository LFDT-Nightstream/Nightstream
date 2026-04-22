use neo_fold_next::rv64im::prove_rv64im_accepted_proof_with_options;
use std::fs;
use std::path::PathBuf;

fn closure_perf_opcode_count() -> usize {
    match env::var("NS_DEBUG_N") {
        Ok(raw) => raw.parse().expect("NS_DEBUG_N must parse as usize"),
        Err(_) => 1,
    }
}

struct IvcProductSurfaceFixture {
    relations: Vec<neo_fold_next::rv64im::Rv64imChunkStepIvcRelation>,
}

fn build_ivc_product_surface_fixture(opcode_count: usize) -> IvcProductSurfaceFixture {
    let source = build_mixed_opcode_perf_source_case(opcode_count);
    let max_steps = source.program_words.len();
    let input = Rv64imProofInput { source, max_steps };
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let (accepted, _) =
        prove_rv64im_accepted_proof_with_options(&input, options).expect("prove accepted artifact for IVC product surface");
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted).expect("prove final statement for IVC product surface");
    let relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build IVC product-surface relations");
    IvcProductSurfaceFixture { relations }
}

#[test]
#[ignore = "closure perf snapshot; run exact with --release -- --ignored --nocapture"]
fn rv64im_ivc_product_surface_native_append_snapshot() {
    let fixture = build_ivc_product_surface_fixture(closure_perf_opcode_count());
    assert!(
        !fixture.relations.is_empty(),
        "IVC product-surface fixture must expose at least one relation"
    );

    let native_append_started = Instant::now();
    let mut state = Rv64imIvcState::init_with_step_cap(1).expect("build initial IVC state");
    for relation in &fixture.relations {
        state = state.append(relation).expect("append native IVC relation");
    }
    let native_append_ms = millis_since(native_append_started);

    println!("native_append_ms={native_append_ms:.3}");
}

fn build_ivc_product_surface_state() -> Rv64imIvcState {
    let fixture = build_ivc_product_surface_fixture(closure_perf_opcode_count());
    assert!(
        !fixture.relations.is_empty(),
        "IVC product-surface fixture must expose at least one relation"
    );
    fixture
        .relations
        .iter()
        .try_fold(
            Rv64imIvcState::init_with_step_cap(1).expect("build initial IVC state"),
            |state, relation| state.append(relation),
        )
        .expect("append native IVC relations")
}

fn ivc_product_surface_state_fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rv64im_ivc_product_surface_state.bin")
}

fn load_ivc_product_surface_state_fixture() -> Rv64imIvcState {
    let bytes = fs::read(ivc_product_surface_state_fixture_path())
        .expect("read product-surface IVC state fixture; run rv64im_ivc_product_surface_regen_state_fixture first");
    bincode::deserialize(&bytes).expect("deserialize product-surface IVC state fixture")
}

#[test]
#[ignore = "closure perf snapshot; run exact with --release -- --ignored --nocapture"]
fn rv64im_ivc_product_surface_native_verify_snapshot() {
    let state = build_ivc_product_surface_state();
    let native_verify_started = Instant::now();
    state.verify().expect("verify native IVC state");
    let native_verify_ms = millis_since(native_verify_started);
    println!("native_verify_ms={native_verify_ms:.3}");
}

#[test]
#[ignore = "manual fixture generator for closure perf snapshots"]
fn rv64im_ivc_product_surface_regen_state_fixture() {
    let state = build_ivc_product_surface_state();
    let encoded = bincode::serialize(&state).expect("serialize product-surface IVC state fixture");
    fs::write(ivc_product_surface_state_fixture_path(), encoded).expect("write product-surface IVC state fixture");
}

#[test]
#[ignore = "closure perf snapshot; run exact with --release -- --ignored --nocapture"]
fn rv64im_ivc_product_surface_compress_and_verify_snapshot() {
    let state = load_ivc_product_surface_state_fixture();
    let keys = setup_rv64im_ivc_snark_cached(&state).expect("warm IVC SNARK key cache");
    let compress_started = Instant::now();
    let snark = state.compress().expect("compress native IVC state");
    let compress_ms = millis_since(compress_started);
    let public_image = state.public_image();
    let compressed_verify_started = Instant::now();
    snark
        .verify(&keys.as_ref().1, &public_image)
        .expect("verify compressed IVC proof");
    let compressed_verify_ms = millis_since(compressed_verify_started);
    println!("compress_ms={compress_ms:.3}");
    println!("compressed_verify_ms={compressed_verify_ms:.3}");
}
