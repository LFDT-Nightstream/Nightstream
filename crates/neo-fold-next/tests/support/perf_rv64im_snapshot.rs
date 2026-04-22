#[test]
#[ignore = "performance/debugging snapshot; run with --release -- --ignored --nocapture"]
fn rv64im_mixed_opcode_perf_snapshot() {
    let end_to_end_started = Instant::now();
    let opcode_count = perf_opcode_count_from_env();
    let source = build_mixed_opcode_perf_source_case(opcode_count);
    let x1_increment_count = mixed_opcode_perf_expected_x1(opcode_count);
    let total_opcodes = source.program_words.len();
    let input = Rv64imProofInput {
        source: source.clone(),
        max_steps: total_opcodes,
    };

    let program = Rv64Program::new(source.start_pc, source.program_words.clone());
    let initial_state = Rv64State::new(source.start_pc, source.initial_registers, &source.initial_memory);

    let build_program_started = Instant::now();
    let build = build_program(&program, &initial_state, total_opcodes).expect("build program");
    let build_program_ms = millis_since(build_program_started);

    let stage1_started = Instant::now();
    let stage1 = build_stage1_summary(&build.rows);
    let stage1_ms = millis_since(stage1_started);

    let stage2_started = Instant::now();
    let stage2 = build_stage2_summary(&build.rows);
    let stage2_ms = millis_since(stage2_started);

    let stage3_started = Instant::now();
    let stage3 = build_stage3_summary(&build.rows);
    let stage3_ms = millis_since(stage3_started);

    let parity_started = Instant::now();
    let (_, derived) = build_parity_case_from_source(source.clone(), total_opcodes).expect("build derived parity case");
    let parity_ms = millis_since(parity_started);

    let build_started = Instant::now();
    let (output, build_perf) = build_simple_kernel_witness_with_perf(&input).expect("build simple kernel witness");
    let build_ms = millis_since(build_started);

    let public_proof_options = Rv64imPublicProofOptions {
        // Native Construction-2 append consumes one verified public step per recursive relation.
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let (proof, prove_perf) = prove_rv64im_public_proof_with_options_and_perf(&input, public_proof_options)
        .expect("prove rv64im public proof");
    let prove_ms = prove_perf.total_ms;

    let verify_started = Instant::now();
    let verify_perf = verify_rv64im_public_proof_with_perf(&proof).expect("verify rv64im public proof");
    let verify_ms = millis_since(verify_started);

    let accepted_artifact = build_rv64im_accepted_proof_artifact(&proof).expect("build accepted artifact");
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted_artifact).expect("prove rv64im final statement");
    let relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step ivc relations");

    let native_append_started = Instant::now();
    let mut ivc_state = Rv64imIvcState::init_with_step_cap(1).expect("build initial rv64im ivc state");
    for relation in &relations {
        ivc_state = ivc_state.append(relation).expect("append rv64im ivc relation");
    }
    let native_append_ms = millis_since(native_append_started);

    let native_verify_started = Instant::now();
    ivc_state.verify().expect("verify native rv64im ivc state");
    let native_verify_ms = millis_since(native_verify_started);

    let compress_started = Instant::now();
    let ivc_snark = ivc_state.compress().expect("compress rv64im ivc state");
    let compress_ms = millis_since(compress_started);
    let ivc_public_image = ivc_state.public_image();
    let ivc_snark_keys = setup_rv64im_ivc_snark_cached(&ivc_state).expect("setup rv64im ivc snark verifier key");

    let compressed_verify_started = Instant::now();
    ivc_snark
        .verify(&ivc_snark_keys.as_ref().1, &ivc_public_image)
        .expect("verify rv64im ivc snark");
    let compressed_verify_ms = millis_since(compressed_verify_started);

    let (published_seam, published_seam_perf) =
        build_rv64im_published_proof_seam_with_perf(&proof).expect("build rv64im published seam");
    assert_eq!(
        published_seam.accepted_artifact.digest,
        accepted_artifact.digest,
        "published seam must carry the same accepted artifact digest as the direct proof path"
    );
    let accepted_artifact = &published_seam.accepted_artifact;
    let compressed_main_proof = &published_seam.main_proof;
    let kernel_export_source = published_seam.kernel_export_source();
    assert_eq!(
        compressed_main_proof.ivc_snark().public_image(),
        &ivc_public_image,
        "published compact main proof must carry the same native IVC public image as direct compression"
    );

    let decider_setup_started = Instant::now();
    let decider_keys = setup_rv64im_ivc_snark_from_final_cached(&final_statement, &final_proof)
        .expect("setup rv64im spartan2 decider");
    let decider_setup_ms = millis_since(decider_setup_started);
    let decider_shape_sizes = decider_keys.as_ref().0.sizes();
    let decider_shape_debug_stats = decider_keys.as_ref().0.shape_debug_stats();
    let decider_proof_bytes = serialized_size_bytes(compressed_main_proof.terminal_decider_proof());

    let ((nightstream_statement, nightstream_proof), nightstream_build_perf) =
        build_rv64im_nightstream_from_published_proof_seam_with_perf(&published_seam, &published_seam_perf)
            .expect("build rv64im nightstream proof");
    let public_statement = proof.statement.clone();
    let nightstream_build_ms = nightstream_build_perf.total_ms;
    let side_statement = nightstream_proof
        .side_proof()
        .binding_statement(&nightstream_statement, &public_statement)
        .expect("build rv64im side binding statement");
    let side_keys = neo_fold_next::nightstream::rv64im::audit::setup_rv64im_side_binding_cached(
        &side_statement,
        nightstream_proof.side_proof().opening_public(),
    )
    .expect("setup rv64im side binding");
    let (opening_statement, opening_witness) =
        neo_fold_next::nightstream::rv64im::audit::build_rv64im_side_opening_relation_from_accepted_artifact(
            accepted_artifact,
        )
        .expect("build rv64im side opening relation");
    let side_opening_keys = neo_fold_next::nightstream::rv64im::audit::setup_rv64im_side_opening_spartan_cached(
        &opening_statement,
        &opening_witness,
    )
    .expect("setup rv64im side opening");

    let nightstream_verify_perf = verify_rv64im_nightstream_with_perf(
        &nightstream_statement,
        &nightstream_proof,
        proof.statement.root_params_id,
        &ivc_snark_keys.as_ref().1,
        &side_opening_keys.as_ref().1,
        &side_keys.as_ref().1,
        &public_statement,
    )
    .expect("verify rv64im nightstream proof");
    let nightstream_verify_ms = nightstream_verify_perf.total_ms;

    let execution_row_count = output.trace.execution_rows.len();
    let real_row_count = output
        .trace
        .execution_rows
        .iter()
        .filter(|row| row.is_real)
        .count();
    let effect_row_count = output
        .trace
        .execution_rows
        .iter()
        .filter(|row| row.is_effect_row)
        .count();
    let commit_row_count = output
        .trace
        .execution_rows
        .iter()
        .filter(|row| row.is_commit_row)
        .count();

    let root_ccs = rv64im_root_main_lane_ccs().expect("build RV64IM root CCS");
    let root_params = rv64im_simple_root_params();
    let root_ccs_n_p2 = root_ccs.n.next_power_of_two();
    let root_ccs_m_p2 = root_ccs.m.next_power_of_two();
    let ccs_total_nnz: usize = root_ccs
        .matrices
        .iter()
        .map(|matrix| {
            matrix
                .as_csc()
                .map(|csc| csc.vals.len())
                .unwrap_or(matrix.rows())
        })
        .sum();
    let ccs_identity_matrices = root_ccs
        .matrices
        .iter()
        .filter(|matrix| matrix.as_csc().is_none())
        .count();
    let approx_trace_constraints = root_ccs.n.saturating_mul(output.prepared_steps.len());
    let approx_trace_nnz = ccs_total_nnz.saturating_mul(output.prepared_steps.len());
    let family_rows = aggregate_family_rows(&output);
    let (lookup_summary, twist_family_counts) = aggregate_lookups(&output);
    let active_twist_family_count = twist_family_counts
        .iter()
        .filter(|count| **count > 0)
        .count();
    let stage1_exact_openings = ExactOpeningClaimStats::default();
    let stage2_exact_openings = ExactOpeningClaimStats::default();
    let stage3_exact_openings = ExactOpeningClaimStats::default();
    let stage1_packaged = packaged_proof_stats(&output.stage_packages.stage1.packaged);
    let stage2_packaged = packaged_proof_stats(&output.stage_packages.stage2.packaged);
    let stage3_packaged = packaged_proof_stats(&output.stage_packages.stage3.packaged);
    let kernel_binding_packaged = packaged_proof_stats(&output.kernel_opening.bindings.packaged);
    let kernel_prepared_packaged = packaged_proof_stats(&output.kernel_opening.prepared_steps.packaged);
    let mut selected_opening_labels = output.stage_packages.stage1.claim.labels();
    selected_opening_labels.extend(output.stage_packages.stage2.claim.labels());
    selected_opening_labels.extend(output.stage_packages.stage3.claim.labels());
    selected_opening_labels.extend(output.kernel_opening.claim.labels());
    let opening_totals = opening_surface_totals(
        &build_perf,
        &[stage1_exact_openings, stage2_exact_openings, stage3_exact_openings],
        &[
            stage1_packaged,
            stage2_packaged,
            stage3_packaged,
            kernel_binding_packaged,
            kernel_prepared_packaged,
        ],
        selected_opening_labels.len(),
    );
    let exact_stage_rows = exact_stage_perf_rows(&output, &build_perf);
    let serialized_sizes = [
        SerializedSizeRow {
            label: "proof.total",
            bytes: serialized_size_bytes(&(&proof.claim, &proof.statement, &proof.kernel, &proof.witness)),
        },
        SerializedSizeRow {
            label: "proof.statement",
            bytes: serialized_size_bytes(&proof.statement),
        },
        SerializedSizeRow {
            label: "proof.claim",
            bytes: serialized_size_bytes(&proof.claim),
        },
        SerializedSizeRow {
            label: "claim.accepted.terminal",
            bytes: serialized_size_bytes(&proof.claim.accepted.terminal),
        },
        SerializedSizeRow {
            label: "claim.opening.terminal",
            bytes: serialized_size_bytes(&proof.claim.opening.terminal),
        },
        SerializedSizeRow {
            label: "claim.root0.terminal",
            bytes: serialized_size_bytes(&proof.claim.root0.terminal),
        },
        SerializedSizeRow {
            label: "proof.kernel",
            bytes: serialized_size_bytes(&proof.kernel),
        },
        SerializedSizeRow {
            label: "proof.witness",
            bytes: serialized_size_bytes(&proof.witness),
        },
        SerializedSizeRow {
            label: "kernel.trace",
            bytes: serialized_size_bytes(&proof.kernel.trace),
        },
        SerializedSizeRow {
            label: "kernel.stages",
            bytes: serialized_size_bytes(&proof.kernel.stages),
        },
        SerializedSizeRow {
            label: "kernel.stage_claims",
            bytes: serialized_size_bytes(&proof.kernel.stage_claims),
        },
        SerializedSizeRow {
            label: "kernel.stage_claims.summary",
            bytes: serialized_size_bytes(&proof.kernel.stage_claims.summary),
        },
        SerializedSizeRow {
            label: "kernel.stage_claims.packaged",
            bytes: serialized_size_bytes(&proof.kernel.stage_claims.packaged),
        },
        SerializedSizeRow {
            label: "kernel.stage_packages",
            bytes: serialized_size_bytes(&proof.kernel.stage_packages),
        },
        SerializedSizeRow {
            label: "kernel.stage_packages.summary",
            bytes: serialized_size_bytes(&proof.kernel.stage_packages.summary),
        },
        SerializedSizeRow {
            label: "kernel.stage_packages.stage1.packaged",
            bytes: serialized_size_bytes(&proof.kernel.stage_packages.packages.stage1.packaged),
        },
        SerializedSizeRow {
            label: "kernel.stage_packages.stage2.packaged",
            bytes: serialized_size_bytes(&proof.kernel.stage_packages.packages.stage2.packaged),
        },
        SerializedSizeRow {
            label: "kernel.stage_packages.stage3.packaged",
            bytes: serialized_size_bytes(&proof.kernel.stage_packages.packages.stage3.packaged),
        },
        SerializedSizeRow {
            label: "kernel.kernel_opening",
            bytes: serialized_size_bytes(&proof.kernel.kernel_opening),
        },
        SerializedSizeRow {
            label: "kernel.kernel_opening.bindings",
            bytes: serialized_size_bytes(&proof.kernel.kernel_opening.bindings),
        },
        SerializedSizeRow {
            label: "kernel.kernel_opening.bindings.packaged",
            bytes: serialized_size_bytes(&proof.kernel.kernel_opening.opening.bindings.packaged),
        },
        SerializedSizeRow {
            label: "kernel.kernel_opening.prepared_steps.packaged",
            bytes: serialized_size_bytes(&proof.kernel.kernel_opening.opening.prepared_steps.packaged),
        },
        SerializedSizeRow {
            label: "kernel.kernel_claims",
            bytes: serialized_size_bytes(&proof.kernel.kernel_claims),
        },
        SerializedSizeRow {
            label: "kernel.kernel_claims.summary",
            bytes: serialized_size_bytes(&proof.kernel.kernel_claims.summary),
        },
        SerializedSizeRow {
            label: "kernel.kernel_claims.summary.terminal",
            bytes: serialized_size_bytes(&proof.kernel.kernel_claims.summary.terminal),
        },
        SerializedSizeRow {
            label: "kernel.kernel_claims.packaged",
            bytes: serialized_size_bytes(&proof.kernel.kernel_claims.packaged),
        },
        SerializedSizeRow {
            label: "kernel.main_lane",
            bytes: serialized_size_bytes(&proof.kernel.main_lane),
        },
        SerializedSizeRow {
            label: "kernel.root_lane_columns",
            bytes: serialized_size_bytes(&proof.kernel.root_lane_columns),
        },
        SerializedSizeRow {
            label: "kernel.root_lane_commitment",
            bytes: serialized_size_bytes(&proof.kernel.root_lane_commitment),
        },
        SerializedSizeRow {
            label: "kernel_export.source",
            bytes: serialized_size_bytes(&kernel_export_source),
        },
        SerializedSizeRow {
            label: "witness.trace",
            bytes: serialized_size_bytes(&proof.witness.trace),
        },
        SerializedSizeRow {
            label: "witness.stages",
            bytes: serialized_size_bytes(&proof.witness.stages),
        },
        SerializedSizeRow {
            label: "witness.stage_claims",
            bytes: serialized_size_bytes(&proof.witness.stage_claims),
        },
        SerializedSizeRow {
            label: "witness.stage_packages",
            bytes: serialized_size_bytes(&proof.witness.stage_packages),
        },
        SerializedSizeRow {
            label: "witness.kernel_opening",
            bytes: serialized_size_bytes(&proof.witness.kernel_opening),
        },
        SerializedSizeRow {
            label: "witness.kernel_claims",
            bytes: serialized_size_bytes(&proof.witness.kernel_claims),
        },
    ];
    let accepted_artifact_total_bytes = [
        serialized_size_bytes(&accepted_artifact.claim),
        serialized_size_bytes(&accepted_artifact.statement),
        serialized_size_bytes(&accepted_artifact.stage_claims),
        serialized_size_bytes(&accepted_artifact.stage_packages),
        serialized_size_bytes(&accepted_artifact.kernel_opening),
        serialized_size_bytes(&accepted_artifact.kernel_claims),
        serialized_size_bytes(&accepted_artifact.root_lane_columns),
        serialized_size_bytes(&accepted_artifact.root_lane_commitment),
        serialized_size_bytes(&accepted_artifact.main_lane),
        serialized_size_bytes(&accepted_artifact.transcript),
        serialized_size_bytes(&accepted_artifact.stage1),
        serialized_size_bytes(&accepted_artifact.stage2),
        serialized_size_bytes(&accepted_artifact.stage3),
        serialized_size_bytes(&accepted_artifact.root_execution),
        serialized_size_bytes(&accepted_artifact.step_composition),
        serialized_size_bytes(&accepted_artifact.soundness_accounting),
        serialized_size_bytes(&accepted_artifact.digest),
    ]
    .into_iter()
    .sum();
    let final_statement_bytes = serialized_size_bytes(
        &published_seam
            .rebuild_final_statement()
            .expect("rebuild final statement from the carried published seam"),
    );
    let nightstream_serialized_sizes = [
        SerializedSizeRow {
            label: "nightstream.total",
            bytes: serialized_size_bytes(&(nightstream_statement.clone(), nightstream_proof.clone())),
        },
        SerializedSizeRow {
            label: "nightstream.statement",
            bytes: serialized_size_bytes(&nightstream_statement),
        },
        SerializedSizeRow {
            label: "nightstream.proof",
            bytes: serialized_size_bytes(&nightstream_proof),
        },
        SerializedSizeRow {
            label: "nightstream.main_proof",
            bytes: serialized_size_bytes(nightstream_proof.main_proof()),
        },
        SerializedSizeRow {
            label: "nightstream.main_proof.published_statement",
            bytes: serialized_size_bytes(nightstream_proof.main_proof().published_statement()),
        },
        SerializedSizeRow {
            label: "nightstream.main_proof.ivc_snark",
            bytes: serialized_size_bytes(nightstream_proof.main_proof().ivc_snark()),
        },
        SerializedSizeRow {
            label: "nightstream.main_proof.terminal_decider_proof",
            bytes: serialized_size_bytes(nightstream_proof.main_proof().terminal_decider_proof()),
        },
        SerializedSizeRow {
            label: "nightstream.side_proof",
            bytes: serialized_size_bytes(nightstream_proof.side_proof()),
        },
    ];
    let proof_total_bytes = serialized_sizes[0].bytes;
    let proof_total_kib = bytes_to_kib(proof_total_bytes);
    let nightstream_total_bytes = nightstream_serialized_sizes[0].bytes;
    let nightstream_total_kib = bytes_to_kib(nightstream_total_bytes);

    assert_eq!(build.rows, output.trace.execution_rows);
    assert_eq!(build.final_state.pc, output.kernel_claims.kernel.final_pc);
    assert_eq!(stage1, output.stages.stage1);
    assert_eq!(stage2, output.stages.stage2);
    assert_eq!(stage3, output.stages.stage3);
    assert_eq!(derived.execution_rows, output.trace.execution_rows);
    assert_eq!(derived.stage1, output.stages.stage1);
    assert_eq!(derived.stage2, output.stages.stage2);
    assert_eq!(derived.stage3, output.stages.stage3);
    assert_eq!(derived.transcript, output.stages.transcript);
    assert_eq!(derived.kernel, output.kernel_claims.kernel);

    assert_eq!(
        proof.witness.root_lane_columns.time_len as usize,
        output.prepared_steps.len()
    );
    assert_eq!(proof.statement.public_step_count as usize, output.prepared_steps.len());
    assert_eq!(
        proof.kernel.root_lane_columns.time_len as usize,
        output.prepared_steps.len()
    );
    assert_eq!(execution_row_count, output.prepared_steps.len());
    assert_eq!(execution_row_count, output.root_lane_columns.time_len as usize);
    assert_eq!(
        proof.witness.trace.shape.execution_row_count as usize,
        execution_row_count
    );
    assert_eq!(proof.witness.trace.shape.real_row_count as usize, real_row_count);
    assert_eq!(proof.witness.trace.shape.effect_row_count as usize, effect_row_count);
    assert_eq!(proof.witness.trace.shape.commit_row_count as usize, commit_row_count);
    assert_eq!(
        proof.kernel.stages.summary.stage1_row_count as usize,
        output.stages.stage1.rows.len()
    );
    assert_eq!(
        proof.kernel.stages.summary.stage2_register_read_count as usize,
        output.stages.stage2.register_reads.len()
    );
    assert_eq!(
        proof.kernel.stages.summary.stage2_register_write_count as usize,
        output.stages.stage2.register_writes.len()
    );
    assert_eq!(
        proof.kernel.stages.summary.stage2_ram_event_count as usize,
        output.stages.stage2.ram_events.len()
    );
    assert_eq!(
        proof.kernel.stages.summary.stage2_twist_link_count as usize,
        output.stages.stage2.twist_links.len()
    );
    assert_eq!(
        proof.kernel.stages.summary.stage3_continuity_count as usize,
        output.stages.stage3.continuity.len()
    );
    assert_eq!(
        proof.kernel.stages.summary.transcript_event_count as usize,
        output.stages.transcript.events.len()
    );
    assert_eq!(proof.statement.final_pc, source.start_pc + (total_opcodes as u64) * 4);
    assert!(proof.statement.halted);
    assert_eq!(
        output.kernel_claims.kernel.final_registers[1],
        x1_increment_count as u64
    );
    assert_eq!(output.kernel_claims.kernel.final_pc, proof.statement.final_pc);
    assert!(output.kernel_claims.kernel.halted);
    assert_eq!(output.kernel_claims.kernel.final_memory.len(), 1);
    assert_eq!(
        output.kernel_claims.kernel.final_memory[0].addr,
        source.initial_memory[0].addr
    );
    assert_eq!(
        nightstream_statement.public_io_digest,
        published_seam
            .main_proof
            .published_statement()
            .expected_digest()
    );

    // ── Precompute published pipeline totals for executive summary ───────
    let total_executed_opcodes = build.executed_steps.len();
    let unique_opcode_labels = collect_unique_opcode_labels(&build);
    let published_prove_before_spartan_ms = prove_ms + published_seam_perf.total_ms + nightstream_build_ms;
    let spartan_setup_ms = decider_setup_ms;
    let published_verify_before_main_proof_ms = nightstream_verify_perf.before_main_proof_ms();
    let main_proof_verify_ms = nightstream_verify_perf.main_proof_ms;
    let published_pipeline_total_ms = spartan_setup_ms
        + published_prove_before_spartan_ms
        + published_verify_before_main_proof_ms
        + main_proof_verify_ms;
    let full_benchmark_wall_ms = millis_since(end_to_end_started);
    let benchmark_extras_ms = (full_benchmark_wall_ms - published_pipeline_total_ms).max(0.0);

    let recursive_relation_core_ms = nightstream_build_perf.final_statement_recursive_prepare_inputs_ms
        + nightstream_build_perf.final_statement_recursive_ccs_ms
        + nightstream_build_perf.final_statement_recursive_dims_ms
        + nightstream_build_perf.final_statement_recursive_rlc_prepare_ms
        + nightstream_build_perf.final_statement_recursive_rlc_ms
        + nightstream_build_perf.final_statement_recursive_dec_split_ms
        + nightstream_build_perf.final_statement_recursive_dec_commit_ms
        + nightstream_build_perf.final_statement_recursive_dec_ms;
    let recursive_wrapper_ms =
        (nightstream_build_perf.final_statement_recursive_proof_ms - recursive_relation_core_ms).max(0.0);

    // ── Input Shape ────────────────────────────────────────────────────────
    print_section("RV64IM Mixed Opcode Perf Snapshot");
    print_kv("ns_debug_n (non-halt ops)", opcode_count);
    print_kv("program_opcodes_total", total_opcodes);
    print_kv("mixed_block_len", RV64IM_MIXED_OPCODE_PERF_BLOCK_LEN);
    print_kv("family_tags", source.manifest.family_tags.len());
    print_kv("final_pc", proof.statement.final_pc);
    print_kv("final_x1", output.kernel_claims.kernel.final_registers[1]);
    print_kv("final_x7", output.kernel_claims.kernel.final_registers[7]);
    print_kv("final_mem_0x100", output.kernel_claims.kernel.final_memory[0].value);
    print_kv(
        "row_expansion",
        format!(
            "{execution_row_count}/{opcode_count} = {:.4} rows/op",
            per_unit(execution_row_count as f64, opcode_count)
        ),
    );
    print_kv(
        "prepared_step_expansion",
        format!(
            "{}/{} = {:.4} steps/op",
            output.prepared_steps.len(),
            opcode_count,
            per_unit(output.prepared_steps.len() as f64, opcode_count)
        ),
    );

    print_timing_table(
        "Raw Proving Timing",
        &[
            ("build_program", build_program_ms),
            ("stage1_summary", stage1_ms),
            ("stage2_summary", stage2_ms),
            ("stage3_summary", stage3_ms),
            ("build_parity_case", parity_ms),
            ("root_lane_witness", build_perf.root_lane_witness_ms),
            ("root_lane_columns", build_perf.root_lane_columns_ms),
            ("root_lane_commitment", build_perf.root_lane_commitment_ms),
            ("build_simple_kernel", build_ms),
            ("public.shared_trace", prove_perf.shared_trace_ms),
            ("public.kernel_projection", prove_perf.simple_kernel.total_ms),
            ("public.parallel_overlap", -prove_perf.parallel_overlap_ms),
            ("prove_rv64im_public_proof", prove_ms),
            (
                "build_rv64im_published_seam.accepted_artifact",
                published_seam_perf.accepted_artifact_ms,
            ),
            (
                "build_rv64im_published_seam.kernel_export_source",
                published_seam_perf.kernel_export_source_ms,
            ),
            (
                "build_rv64im_published_seam.final_statement",
                published_seam_perf.final_statement_ms,
            ),
            (
                "build_rv64im_published_seam.main_proof",
                published_seam_perf.main_proof_ms,
            ),
            ("setup_rv64im_ivc_snark_from_final.direct", decider_setup_ms),
            ("build_rv64im_nightstream", nightstream_build_ms),
        ],
        opcode_count,
        execution_row_count,
    );

    print_timing_table(
        "Raw Verify Timing",
        &[
            ("verify_rv64im_public_proof", verify_ms),
            ("verify_rv64im_nightstream", nightstream_verify_ms),
        ],
        opcode_count,
        execution_row_count,
    );

    print_section("Benchmark Extras");
    print_kv(
        "diagnostics and extra benchmark work",
        format_ms_per_opcode(benchmark_extras_ms, total_executed_opcodes),
    );
    print_kv(
        "includes",
        "report-only work outside the published prove+verify path".to_string(),
    );
    print_kv(
        "full benchmark wall time",
        format_ms_per_opcode(full_benchmark_wall_ms, total_executed_opcodes),
    );

    let prove_total_ms = published_prove_before_spartan_ms + spartan_setup_ms;
    let amortized_prove_ms =
        prove_total_ms - spartan_setup_ms - nightstream_build_perf.verified_seams.side_binding_setup_ms;

    print_section("Nightstream Opening Diagnostics");
    println!("  total: {:.3} ms", nightstream_build_perf.total_ms);
    println!();
    println!("  note: recursive/final-statement timers are summarized in the proving tree.");
    println!("  note: phase0 values below are nested accumulators and overlap by design.");
    println!();
    println!("  phase0 opening (nested accumulators — do not sum as flat partition):");
    {
        let vs = &nightstream_build_perf.verified_seams;
        println!(
            "    {:28} {:>9.3}  {:28} {:>9.3}",
            "claim_witnesses",
            vs.opening_phase0_claim_witnesses_ms,
            "relation_artifact",
            vs.opening_phase0_relation_artifact_ms
        );
        println!(
            "    {:28} {:>9.3}  {:28} {:>9.3}",
            "pack_columns",
            vs.opening_phase0_packed_columns_ms,
            "commit_vector",
            vs.opening_phase0_commitment_vector_ms
        );
        println!(
            "    {:28} {:>9.3}  {:28} {:>9.3}",
            "commit_many",
            vs.opening_phase0_commitment_commit_many_ms,
            "commit_root",
            vs.opening_phase0_commitment_root_ms
        );
        println!(
            "    {:28} {:>9.3}  {:28} {:>9.3}",
            "object_total",
            vs.opening_phase0_opened_object_total_ms,
            "object_id",
            vs.opening_phase0_opened_object_id_ms
        );
        println!(
            "    {:28} {:>9.3}  {:28} {:>9.3}",
            "bind_digest", vs.opening_phase0_binding_digest_ms, "point", vs.opening_phase0_point_derivation_ms
        );
        println!(
            "    {:28} {:>9.3}  {:28} {:>9.3}",
            "payload_eval", vs.opening_phase0_payload_eval_ms, "claim_build", vs.opening_phase0_claim_build_ms
        );
        println!(
            "    {:28} {:>9.3}",
            "slot_total", vs.opening_phase0_slot_claims_total_ms
        );
    }
    println!();
    println!("  opening convergence:");
    {
        let vs = &nightstream_build_perf.verified_seams;
        println!(
            "    {:18} {:>7.3}  {:18} {:>7.3}  {:18} {:>7.3}",
            "phase1",
            vs.opening_convergence_phase1_ms,
            "phase2",
            vs.opening_convergence_phase2_ms,
            "final_targets",
            vs.opening_convergence_final_openings_ms
        );
        println!(
            "    {:18} {:>7.3}  {:18} {:>7.3}  {:18} {:>7.3}",
            "targets.map",
            vs.opening_convergence_final_openings_witness_map_ms,
            "targets.rep",
            vs.opening_convergence_final_openings_representative_ms,
            "targets.commit",
            vs.opening_convergence_final_openings_commitment_validate_ms
        );
        println!(
            "    {:18} {:>7.3}  {:18} {:>7.3}  {:18} {:>7.3}",
            "targets.obj_digest",
            vs.opening_convergence_final_openings_opened_commitment_digest_ms,
            "targets.proof_dig",
            vs.opening_convergence_final_openings_opening_proof_digest_ms,
            "targets.target",
            vs.opening_convergence_final_openings_target_build_ms
        );
        println!(
            "    {:18} {:>7.3}  {:18} {:>7.3}",
            "digest", vs.opening_convergence_digest_ms, "support_wrap", vs.opening_support_wrap_ms
        );
    }
    println!();
    println!("  verified seams (other components):");
    {
        let vs = &nightstream_build_perf.verified_seams;
        println!(
            "    {:24} {:>9.3}  {:24} {:>9.3}",
            "final_surface_guard", vs.final_surface_guard_ms, "decider_relation", vs.decider_relation_ms
        );
        println!("    {:24} {:>9.3}", "main_proof", vs.main_proof_ms);
        println!(
            "    {:24} {:>9.3}  {:24} {:>9.3}",
            "statement", vs.statement_ms, "bind_side_core", vs.bind_side_statement_core_ms
        );
        println!("    {:24} {:>9.3}", "proof_binding_root", vs.proof_binding_root_ms);
    }

    print_section("CCS / Constraint Shape");
    print_kv("root_row_width", RV64IM_ROOT_ROW_WIDTH);
    print_kv("root_public_inputs", RV64IM_ROOT_PUBLIC_INPUTS);
    print_kv("constraints_per_step (n)", root_ccs.n);
    print_kv("columns_per_step (m)", root_ccs.m);
    print_kv("constraints_per_step_p2", root_ccs_n_p2);
    print_kv("columns_per_step_p2", root_ccs_m_p2);
    print_kv("matrix_count (t)", root_ccs.t());
    print_kv("max_degree", root_ccs.max_degree());
    print_kv("identity_matrices", ccs_identity_matrices);
    print_kv("total_nnz_per_step (non-zero matrix entries)", ccs_total_nnz);
    print_kv(
        "avg_nnz_per_constraint (non-zero matrix entries)",
        format!("{:.4}", per_unit(ccs_total_nnz as f64, root_ccs.n)),
    );
    print_kv("approx_constraints_for_trace", approx_trace_constraints);
    print_kv(
        "approx_constraints_per_non-halt_opcode",
        format!("{:.4}", per_unit(approx_trace_constraints as f64, opcode_count)),
    );
    print_kv("approx_nnz_for_trace (non-zero matrix entries)", approx_trace_nnz);
    print_kv(
        "approx_nnz_per_non-halt_opcode (non-zero matrix entries)",
        format!("{:.4}", per_unit(approx_trace_nnz as f64, opcode_count)),
    );
    print_kv(
        "root_params",
        format!(
            "d={} kappa={} m={} b={} k_rho={} B={} T={} s={} lambda={}",
            root_params.d,
            root_params.kappa,
            root_params.m,
            root_params.b,
            root_params.k_rho,
            root_params.B,
            root_params.T,
            root_params.s,
            root_params.lambda
        ),
    );

    print_section("Row / Step Shape");
    print_kv("execution_rows", execution_row_count);
    print_kv("real_rows", real_row_count);
    print_kv("effect_rows", effect_row_count);
    print_kv("commit_rows", commit_row_count);
    print_kv("prepared_steps", output.prepared_steps.len());
    print_kv("public_steps", output.root_lane_columns.time_len);
    print_kv("stage1_rows", output.stages.stage1.rows.len());
    print_kv("stage3_continuity", output.stages.stage3.continuity.len());
    print_kv("transcript_events", output.stages.transcript.events.len());
    print_root_main_lane_family(&output, &proof);

    print_section("Spartan Decider Shape");
    print_kv("num_cons_unpadded", decider_shape_sizes[0]);
    print_kv("num_shared_unpadded", decider_shape_sizes[1]);
    print_kv("num_precommitted_unpadded", decider_shape_sizes[2]);
    print_kv("num_rest_unpadded", decider_shape_sizes[3]);
    print_kv("num_cons_padded", decider_shape_sizes[4]);
    print_kv("num_shared_padded", decider_shape_sizes[5]);
    print_kv("num_precommitted_padded", decider_shape_sizes[6]);
    print_kv("num_rest_padded", decider_shape_sizes[7]);
    print_kv("num_public", decider_shape_sizes[8]);
    print_kv("num_challenges", decider_shape_sizes[9]);
    print_kv("a_nnz", decider_shape_debug_stats.a_nnz);
    print_kv("b_nnz", decider_shape_debug_stats.b_nnz);
    print_kv("c_nnz", decider_shape_debug_stats.c_nnz);
    print_kv("abc_total_nnz", decider_shape_debug_stats.total_nnz);
    print_kv("a_max_row_nnz", decider_shape_debug_stats.max_row_nnz_a);
    print_kv("b_max_row_nnz", decider_shape_debug_stats.max_row_nnz_b);
    print_kv("c_max_row_nnz", decider_shape_debug_stats.max_row_nnz_c);
    print_kv("abc_max_row_nnz", decider_shape_debug_stats.max_row_nnz_total);
    print_kv(
        "abc_avg_row_nnz",
        format!(
            "{:.2}",
            decider_shape_debug_stats.total_nnz as f64 / decider_shape_sizes[4].max(1) as f64
        ),
    );

    print_family_rows("Row Expansion by Family", &family_rows, opcode_count);
    print_lookup_summary(lookup_summary, opcode_count, &twist_family_counts);
    print_lookup_group_density(
        lookup_summary,
        opcode_count,
        &twist_family_counts,
        active_twist_family_count,
    );
    print_exact_stage_witness_shape(&exact_stage_rows);
    print_exact_opening_table(
        &[
            ("stage1", stage1_exact_openings),
            ("stage2", stage2_exact_openings),
            ("stage3", stage3_exact_openings),
        ],
        opcode_count,
        execution_row_count,
    );
    print_selected_vs_exact_amplification(&exact_stage_rows);
    print_exact_stage_build_breakdown(&exact_stage_rows);
    print_packaged_proof_table(&[
        ("stage1", stage1_packaged),
        ("stage2", stage2_packaged),
        ("stage3", stage3_packaged),
        ("kernel_bindings", kernel_binding_packaged),
        ("kernel_prepared", kernel_prepared_packaged),
    ]);
    print_compact_opening_build_breakdown(&build_perf);
    print_opening_surface_totals(opening_totals, opcode_count, execution_row_count);
    print_opening_reuse_proxy(&output);
    print_opening_label_summary(&selected_opening_labels);
    print_serialized_size_table("Serialized Sizes (Public Proof)", &serialized_sizes, proof_total_bytes);
    print_section("Nightstream Published Boundary");
    print_kv(
        "accepted_artifact_size",
        format!(
            "{accepted_artifact_total_bytes} bytes ({:.3} KiB)",
            bytes_to_kib(accepted_artifact_total_bytes)
        ),
    );
    print_kv(
        "final_statement_size",
        format!(
            "{final_statement_bytes} bytes ({:.3} KiB)",
            bytes_to_kib(final_statement_bytes)
        ),
    );
    print_kv(
        "kernel_export_source_size",
        format!(
            "{} bytes ({:.3} KiB)",
            serialized_size_bytes(&kernel_export_source),
            bytes_to_kib(serialized_size_bytes(&kernel_export_source))
        ),
    );
    print_kv(
        "spartan_decider_proof_size",
        format!(
            "{decider_proof_bytes} bytes ({:.3} KiB)",
            bytes_to_kib(decider_proof_bytes)
        ),
    );
    print_kv(
        "nightstream_main_ivc_snark_size",
        format!(
            "{} bytes ({:.3} KiB)",
            serialized_size_bytes(nightstream_proof.main_proof().ivc_snark()),
            bytes_to_kib(serialized_size_bytes(nightstream_proof.main_proof().ivc_snark()))
        ),
    );
    print_serialized_size_table(
        "Serialized Sizes (Nightstream)",
        &nightstream_serialized_sizes,
        nightstream_total_bytes,
    );
    print_section("Side Decider");
    let artifact_bytes = serialized_size_bytes(nightstream_proof.side_proof());
    println!("  {:32} {:>10} {:>10}", "component", "bytes", "KiB");
    println!(
        "  {:32} {:>10} {:>10.3}",
        "artifact (published)",
        artifact_bytes,
        bytes_to_kib(artifact_bytes)
    );
    print_verify_breakdown(
        "Theorem Verify Breakdown",
        &verify_perf,
        opcode_count,
        execution_row_count,
    );

    print_hotspot_table(
        "Critical Hotspots",
        published_pipeline_total_ms,
        total_executed_opcodes,
        &[
            ("spartan.setup", spartan_setup_ms),
            (
                "published_seam.final.kernel_export",
                nightstream_build_perf.final_statement_kernel_export_ms,
            ),
            (
                "published_seam.recursive.rlc",
                nightstream_build_perf.final_statement_recursive_rlc_ms,
            ),
            ("public.root_main_lane.rlc", prove_perf.root_main_lane.session.rlc_ms()),
            ("public.root_main_lane.package", {
                let root_prove = &prove_perf.root_main_lane;
                (root_prove.total_ms - root_prove.prepare_steps_ms - root_prove.session.total_ms).max(0.0)
            }),
            (
                "public.root_main_lane.prepare_inputs",
                prove_perf.root_main_lane.session.prepare_inputs_ms(),
            ),
            (
                "published_seam.accepted_artifact",
                published_seam_perf.accepted_artifact_ms,
            ),
            (
                "nightstream.side_proof",
                nightstream_build_perf.verified_seams.side_binding_ms,
            ),
            ("public.kernel_projection", prove_perf.simple_kernel.total_ms),
        ],
        8,
    );

    {
        let total = prove_total_ms;
        let max_bar = total;
        tree_header("PROVING BREAKDOWN", total, per_unit(total, total_executed_opcodes));
        tree_row("├─ ", "public proof", prove_ms, max_bar, total, true);
        tree_row(
            "│  ├─ ",
            "shared trace",
            prove_perf.shared_trace_ms,
            max_bar,
            total,
            false,
        );
        tree_row(
            "│  ├─ ",
            "kernel projection",
            prove_perf.simple_kernel.total_ms,
            max_bar,
            total,
            false,
        );

        let root_prove = &prove_perf.root_main_lane;
        let package_overhead_ms =
            (root_prove.total_ms - root_prove.prepare_steps_ms - root_prove.session.total_ms).max(0.0);
        tree_row("│  ├─ ", "root main lane", root_prove.total_ms, max_bar, total, false);
        tree_row("│     ├─ ", "package", package_overhead_ms, max_bar, total, false);
        tree_row("│     ├─ ", "Π_RLC", root_prove.session.rlc_ms(), max_bar, total, false);
        tree_row(
            "│     ├─ ",
            "prepare_inputs",
            root_prove.session.prepare_inputs_ms(),
            max_bar,
            total,
            false,
        );
        tree_row_annotated(
            "│     ├─ ",
            "Π_CCS",
            root_prove.session.ccs_ms(),
            &format!(
                "(FE {:.1}, NC {:.1})",
                root_prove.session.ccs_fe_sumcheck_ms(),
                root_prove.session.ccs_nc_sumcheck_ms()
            ),
        );
        tree_row("│     ├─ ", "Π_DEC", root_prove.session.dec_ms(), max_bar, total, false);
        tree_row(
            "│     └─ ",
            "prepare_steps",
            root_prove.prepare_steps_ms,
            max_bar,
            total,
            false,
        );
        if prove_perf.parallel_overlap_ms > 0.0 {
            tree_row_annotated(
                "│  ├─ ",
                "parallel overlap",
                -prove_perf.parallel_overlap_ms,
                "(kernel projection overlapped with root main lane)",
            );
        }
        tree_row_annotated(
            "│  └─ ",
            "other",
            (prove_ms - prove_perf.shared_trace_ms - prove_perf.simple_kernel.total_ms - root_prove.total_ms
                + prove_perf.parallel_overlap_ms)
                .max(0.0),
            "(main-lane binding + proof export)",
        );
        println!("  │");

        tree_row(
            "├─ ",
            "published seam",
            published_seam_perf.total_ms,
            max_bar,
            total,
            true,
        );
        tree_row(
            "│  ├─ ",
            "accepted_artifact",
            published_seam_perf.accepted_artifact_ms,
            max_bar,
            total,
            false,
        );
        tree_row(
            "│  ├─ ",
            "kernel_export_source",
            published_seam_perf.kernel_export_source_ms,
            max_bar,
            total,
            false,
        );
        tree_row(
            "│  ├─ ",
            "final_statement",
            published_seam_perf.final_statement_ms,
            max_bar,
            total,
            false,
        );
        tree_row(
            "│  │  ├─ ",
            "recursive_proof",
            nightstream_build_perf.final_statement_recursive_proof_ms,
            max_bar,
            total,
            false,
        );
        tree_row(
            "│  │  │  ├─ ",
            "Π_RLC",
            nightstream_build_perf.final_statement_recursive_rlc_ms,
            max_bar,
            total,
            false,
        );
        tree_row("│  │  │  ├─ ", "wrapper", recursive_wrapper_ms, max_bar, total, false);
        tree_row_annotated(
            "│  │  │  ├─ ",
            "Π_CCS",
            nightstream_build_perf.final_statement_recursive_ccs_ms,
            &format!(
                "(FE {:.1}, NC {:.1})",
                nightstream_build_perf.final_statement_recursive_ccs_fe_sumcheck_ms,
                nightstream_build_perf.final_statement_recursive_ccs_nc_sumcheck_ms
            ),
        );
        tree_row(
            "│  │  │  └─ ",
            "Π_DEC",
            nightstream_build_perf.final_statement_recursive_dec_ms,
            max_bar,
            total,
            false,
        );
        tree_row(
            "│  │  ├─ ",
            "kernel_export",
            nightstream_build_perf.final_statement_kernel_export_ms,
            max_bar,
            total,
            false,
        );
        tree_row(
            "│  │  ├─ ",
            "folded_digest",
            nightstream_build_perf.final_statement_folded_digest_ms,
            max_bar,
            total,
            false,
        );
        let final_other = (published_seam_perf.final_statement_ms
            - nightstream_build_perf.final_statement_recursive_proof_ms
            - nightstream_build_perf.final_statement_kernel_export_ms
            - nightstream_build_perf.final_statement_folded_digest_ms)
            .max(0.0);
        tree_row("│  │  └─ ", "other", final_other, max_bar, total, false);
        println!("  │");

        tree_row(
            "├─ ",
            "nightstream residual build",
            nightstream_build_ms,
            max_bar,
            total,
            true,
        );

        let vs = &nightstream_build_perf.verified_seams;
        tree_row("│  ├─ ", "verified_seams", vs.total_ms, max_bar, total, false);
        tree_row_annotated("│  │  ├─ ", "side_binding *", vs.side_binding_ms, "★ biggest item");
        tree_row_annotated("│  │  │  ├─ ", "setup", vs.side_binding_setup_ms, "← amortizable");
        tree_row("│  │  │  ├─ ", "prove", vs.side_binding_prove_ms, max_bar, total, false);
        tree_row(
            "│  │  │  └─ ",
            "prepare",
            vs.side_binding_prepare_ms,
            max_bar,
            total,
            false,
        );
        tree_row(
            "│  │  ├─ ",
            "phase0 opening",
            vs.opening_phase0_artifact_ms,
            max_bar,
            total,
            false,
        );
        tree_row_annotated(
            "│  │  ├─ ",
            "opening_support_bundle",
            vs.opening_support_bundle_ms,
            &format!(
                "(p1 {:.1}, p2 {:.1}, tgt {:.1})",
                vs.opening_convergence_phase1_ms,
                vs.opening_convergence_phase2_ms,
                vs.opening_convergence_final_openings_ms
            ),
        );
        let seam_other =
            (vs.total_ms - vs.side_binding_ms - vs.opening_phase0_artifact_ms - vs.opening_support_bundle_ms).max(0.0);
        tree_row("│  │  └─ ", "other seams", seam_other, max_bar, total, false);

        tree_row(
            "│  └─ ",
            "side_support_bundle",
            nightstream_build_perf.side_support_bundle_ms,
            max_bar,
            total,
            false,
        );
        println!("  │");
        tree_row("├─ ", "Spartan setup/keygen", spartan_setup_ms, max_bar, total, true);
        println!("  ─────────────────────────────────────────────────────────────────────");
        println!(
            "  prove total {:>7.1} ms  ({:.2} ms/op)    amortized: {:.1} ms ({:.2} ms/op)",
            total,
            per_unit(total, total_executed_opcodes),
            amortized_prove_ms,
            per_unit(amortized_prove_ms, total_executed_opcodes),
        );
    }

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!(
        "║  RV64IM Perf Snapshot   N={:<6}  rows={:<6}  {:.2} rows/op              ║",
        opcode_count,
        execution_row_count,
        per_unit(execution_row_count as f64, opcode_count)
    );
    println!("╠══════════════════════ IVC PRODUCT SURFACE ═══════════════════════════════╣");
    println!(
        "║  {:36} {:>8.1} ms  {:>6.2} ms/op             ║",
        "native append",
        native_append_ms,
        per_unit(native_append_ms, total_executed_opcodes)
    );
    println!(
        "║  {:36} {:>8.1} ms  {:>6.2} ms/op             ║",
        "native verify",
        native_verify_ms,
        per_unit(native_verify_ms, total_executed_opcodes)
    );
    println!(
        "║  {:36} {:>8.1} ms  {:>6.2} ms/op             ║",
        "compress",
        compress_ms,
        per_unit(compress_ms, total_executed_opcodes)
    );
    println!(
        "║  {:36} {:>8.1} ms  {:>6.2} ms/op             ║",
        "compressed verify",
        compressed_verify_ms,
        per_unit(compressed_verify_ms, total_executed_opcodes)
    );
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!(
        "  proof: {proof_total_bytes} bytes ({proof_total_kib:.1} KiB)  |  nightstream: {nightstream_total_bytes} bytes ({nightstream_total_kib:.1} KiB)"
    );
    println!("  opcodes: {total_executed_opcodes} ({unique_opcode_labels})");
}
