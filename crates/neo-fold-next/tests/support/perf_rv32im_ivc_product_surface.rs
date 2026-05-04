use neo_fold_next::rv32im::prove_rv32im_accepted_proof_with_options;

fn ivc_product_surface_rows_per_chunk_from_args() -> usize {
    let mut rows_per_chunk = 1usize;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--chunk-size" | "--rows-per-chunk" => {
                let raw = args
                    .next()
                    .expect("expected integer after --chunk-size/--rows-per-chunk");
                rows_per_chunk = raw.parse().expect("chunk size must parse as usize");
            }
            _ => {
                if let Some(raw) = arg.strip_prefix("--chunk-size=") {
                    rows_per_chunk = raw.parse().expect("chunk size must parse as usize");
                } else if let Some(raw) = arg.strip_prefix("--rows-per-chunk=") {
                    rows_per_chunk = raw.parse().expect("chunk size must parse as usize");
                }
            }
        }
    }
    assert!(rows_per_chunk != 0, "chunk size must be at least one");
    rows_per_chunk
}

fn ivc_product_surface_root_fold_schedule() -> FoldSchedule {
    FoldSchedule::RowsPerChunk(ivc_product_surface_rows_per_chunk_from_args())
}

fn print_ivc_compressed_artifact_sizes(snark: &neo_fold_next::rv32im::Rv32imIvcSnark) {
    type IvcRecursionSnark = spartan2::spartan::R1CSSNARK<spartan2::provider::GoldilocksP3MerkleMleEngine>;

    let total_bytes = bincode::serialize(snark).expect("serialize compressed IVC artifact").len();
    let proof_wrapper_bytes = bincode::serialize(snark.proof())
        .expect("serialize compressed IVC proof wrapper")
        .len();
    let proof_spartan_bytes = snark.proof().snark_bytes_len();
    let public_image_bytes = bincode::serialize(snark.public_image())
        .expect("serialize compressed IVC public image")
        .len();
    let ivc_recursion_snark: IvcRecursionSnark = bincode::deserialize(
        &snark
            .proof()
            .terminal_f_prime_committed_step_proof
            .snark_data,
    )
    .expect("decode terminal F' committed-step Spartan proof");
    let spartan = ivc_recursion_snark
        .serialized_size_breakdown()
        .expect("measure IVC recursion Spartan proof size");

    let x_i_bytes = bincode::serialize(&snark.public_image().x_i)
        .expect("serialize compressed IVC x_i image")
        .len();
    let terminal_statement_bytes = snark
        .public_image()
        .terminal_statement
        .as_ref()
        .map(|statement| bincode::serialize(statement).expect("serialize compressed IVC terminal statement").len())
        .unwrap_or(0);
    let terminal_step_public_bytes = snark
        .public_image()
        .terminal_statement
        .as_ref()
        .map(|statement| {
            bincode::serialize(&statement.step_public)
                .expect("serialize compressed IVC terminal step_public")
                .len()
        })
        .unwrap_or(0);
    let terminal_chunk_summary_bytes = snark
        .public_image()
        .terminal_statement
        .as_ref()
        .map(|statement| {
            bincode::serialize(&statement.chunk_summary)
                .expect("serialize compressed IVC terminal chunk_summary")
                .len()
        })
        .unwrap_or(0);
    let mut public_image_without_terminal = snark.public_image().clone();
    public_image_without_terminal.terminal_statement = None;
    let public_image_without_terminal_bytes = bincode::serialize(&public_image_without_terminal)
        .expect("serialize compressed IVC public image without terminal statement")
        .len();

    println!("compressed_artifact_total_bytes={total_bytes}");
    println!("compressed_artifact_total_kib={:.3}", total_bytes as f64 / 1024.0);
    println!("compressed_artifact_proof_wrapper_bytes={proof_wrapper_bytes}");
    println!("compressed_artifact_spartan_bytes={proof_spartan_bytes}");
    println!("compressed_artifact_public_image_bytes={public_image_bytes}");
    println!("compressed_artifact_public_image_without_terminal_statement_bytes={public_image_without_terminal_bytes}");
    println!("compressed_artifact_public_image_x_i_bytes={x_i_bytes}");
    println!("compressed_artifact_public_image_terminal_statement_bytes={terminal_statement_bytes}");
    println!("compressed_artifact_public_image_terminal_step_public_bytes={terminal_step_public_bytes}");
    println!("compressed_artifact_public_image_terminal_chunk_summary_bytes={terminal_chunk_summary_bytes}");
    println!("compressed_artifact_spartan_instance_bytes={}", spartan.instance);
    println!("compressed_artifact_spartan_outer_sumcheck_bytes={}", spartan.outer_sumcheck);
    println!("compressed_artifact_spartan_outer_claims_bytes={}", spartan.outer_claims);
    println!("compressed_artifact_spartan_inner_sumcheck_bytes={}", spartan.inner_sumcheck);
    println!("compressed_artifact_spartan_eval_w_bytes={}", spartan.eval_w);
    println!("compressed_artifact_spartan_eval_arg_bytes={}", spartan.eval_arg);
    println!("compressed_artifact_spartan_inner_sum_claim_bytes={}", spartan.inner_sum_claim);
}

fn closure_perf_opcode_count() -> usize {
    match env::var("NS_DEBUG_N") {
        Ok(raw) => raw.parse().expect("NS_DEBUG_N must parse as usize"),
        Err(_) => 1,
    }
}

fn flush_ivc_product_surface_stdout() {
    let _ = io::stdout().flush();
}

fn print_ivc_product_surface_trace(line: &str) {
    println!("{line}");
    flush_ivc_product_surface_stdout();
}

struct IvcProductSurfaceFixture {
    relations: Vec<neo_fold_next::rv32im::Rv32imChunkStepIvcRelation>,
}

fn build_ivc_product_surface_fixture(opcode_count: usize) -> IvcProductSurfaceFixture {
    let source = build_mixed_opcode_perf_source_case(opcode_count);
    let max_steps = source.program_words.len();
    let input = Rv32imProofInput { source, max_steps };
    let options = Rv32imPublicProofOptions {
        root_fold_schedule: ivc_product_surface_root_fold_schedule(),
    };
    let (accepted, _) =
        prove_rv32im_accepted_proof_with_options(&input, options).expect("prove accepted artifact for IVC product surface");
    let (final_statement, final_proof) =
        prove_rv32im_final_statement_from_accepted(&accepted).expect("prove final statement for IVC product surface");
    let relations =
        build_rv32im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build IVC product-surface relations");
    IvcProductSurfaceFixture { relations }
}

#[test]
#[ignore = "closure perf snapshot; run exact with --release -- --ignored --nocapture"]
fn rv32im_ivc_product_surface_no_spartan_append_snapshot() {
    let rows_per_chunk = ivc_product_surface_rows_per_chunk_from_args();
    let fixture = build_ivc_product_surface_fixture(closure_perf_opcode_count());
    assert!(
        !fixture.relations.is_empty(),
        "IVC product-surface fixture must expose at least one relation"
    );

    let native_append_started = Instant::now();
    let mut state = Rv32imIvcState::init_with_step_cap(rows_per_chunk).expect("build initial IVC state");
    for relation in &fixture.relations {
        state = state.append(relation).expect("append no-Spartan IVC relation");
    }
    let native_append_ms = millis_since(native_append_started);

    println!("rows_per_chunk={rows_per_chunk}");
    println!("no_spartan_append_ms={native_append_ms:.3}");
}

fn build_ivc_product_surface_state_from_relations(
    rows_per_chunk: usize,
    relations: &[neo_fold_next::rv32im::Rv32imChunkStepIvcRelation],
) -> Rv32imIvcState {
    assert!(
        !relations.is_empty(),
        "IVC product-surface fixture must expose at least one relation"
    );
    relations
        .iter()
        .try_fold(
            Rv32imIvcState::init_with_step_cap(rows_per_chunk).expect("build initial IVC state"),
            |state, relation| state.append(relation),
        )
        .expect("append native IVC relations")
}

#[test]
#[ignore = "closure perf snapshot; run exact with --release -- --ignored --nocapture"]
fn rv32im_ivc_product_surface_with_spartan_compress_and_verify_snapshot() {
    let rows_per_chunk = ivc_product_surface_rows_per_chunk_from_args();
    let fixture_started = Instant::now();
    let fixture = build_ivc_product_surface_fixture(closure_perf_opcode_count());
    let fixture_ms = millis_since(fixture_started);
    println!("fixture_ms={fixture_ms:.3}");
    flush_ivc_product_surface_stdout();
    let state_started = Instant::now();
    let state = build_ivc_product_surface_state_from_relations(rows_per_chunk, &fixture.relations);
    let state_ms = millis_since(state_started);
    println!("state_build_ms={state_ms:.3}");
    flush_ivc_product_surface_stdout();
    let setup_started = Instant::now();
    let mut setup_trace = |line: &str| print_ivc_product_surface_trace(line);
    let keys = setup_rv32im_ivc_snark_cached_with_trace(&state, &mut setup_trace).expect("warm IVC SNARK key cache");
    let setup_ms = millis_since(setup_started);
    println!("setup_ms={setup_ms:.3}");
    flush_ivc_product_surface_stdout();
    let compress_started = Instant::now();
    let mut compress_trace = |line: &str| print_ivc_product_surface_trace(line);
    let snark = state
        .compress_with_trace(&mut compress_trace)
        .expect("compress IVC state into Spartan proof");
    let compress_ms = millis_since(compress_started);
    let public_image = snark.public_image().clone();
    let compressed_verify_started = Instant::now();
    snark
        .verify(&keys.as_ref().1, &public_image)
        .expect("verify compressed IVC SNARK");
    let compressed_verify_ms = millis_since(compressed_verify_started);
    println!("rows_per_chunk={rows_per_chunk}");
    println!("compress_ms={compress_ms:.3}");
    println!("compressed_verify_ms={compressed_verify_ms:.3}");
    println!("compressed_verify_mode=superneo_terminal_f_prime_r2");
    print_ivc_compressed_artifact_sizes(&snark);
    flush_ivc_product_surface_stdout();
}
