use std::time::Instant;

use neo_fold_next::rv32im::audit::Rv32imCeClaimDigestShape;
use neo_fold_next::rv32im::audit::{
    build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices,
    debug_measure_rv32im_main_recursion_step_chunk_replay_aux_counts,
    debug_measure_rv32im_main_recursion_step_chunk_replay_tail_aux_counts,
    debug_measure_rv32im_main_recursion_step_chunk_replay_tail_digest_aux_breakdown,
    debug_measure_rv32im_main_recursion_step_pi_ccs_aux_counts,
    debug_measure_rv32im_main_recursion_step_pi_ccs_bind_me_inputs_aux_breakdown,
    debug_measure_rv32im_main_recursion_step_pi_ccs_constraint_counts,
    debug_measure_rv32im_main_recursion_step_pi_ccs_fingerprint,
    debug_measure_rv32im_main_recursion_step_pi_ccs_sumcheck_constraint_breakdown,
    debug_measure_rv32im_main_recursion_step_pi_rlc_public_constraint_breakdown,
    debug_measure_rv32im_main_recursion_step_pi_rlc_public_stage_breakdown,
    debug_measure_rv32im_main_recursion_step_shape_only_circuit_shape,
    debug_measure_rv32im_main_recursion_step_spartan_circuit_shape,
    debug_measure_rv32im_main_recursion_step_spartan_shape_synthesis,
    debug_measure_rv32im_main_recursion_step_stage_aux_counts,
    debug_measure_rv32im_terminal_f_prime_committed_step_shape,
    debug_trace_rv32im_main_recursion_step_shape_only_fingerprint_synthesize,
};
use neo_fold_next::rv32im::final_relation::prove_rv32im_final_statement_from_accepted;
use neo_fold_next::rv32im::{
    build_mixed_opcode_perf_source_case, build_rv32im_chunk_step_ivc_relations,
    build_rv32im_main_recursion_f_prime_advices, debug_measure_rv32im_main_recursion_step_chunk_replay_fingerprint,
    prove_rv32im_accepted_proof_with_options_and_perf, Rv32imProofInput, Rv32imPublicProofOptions,
};
use neo_math::D;

#[path = "support/rv32im_main_recursion_shape_probe_support.rs"]
mod rv32im_main_recursion_shape_probe_support;

use rv32im_main_recursion_shape_probe_support::*;

fn main() {
    let probe_mode = probe_mode_from_args();
    let root_fold_schedule = root_fold_schedule_from_args();
    let opcode_count = perf_opcode_count_from_env();
    let source = build_mixed_opcode_perf_source_case(opcode_count);
    let input = Rv32imProofInput {
        max_steps: source.program_words.len(),
        source,
    };

    let fixture_started = Instant::now();
    let accepted_started = Instant::now();
    let ((accepted, _), accepted_perf) = unwrap_accepted_artifact_with_schedule_context(
        prove_rv32im_accepted_proof_with_options_and_perf(&input, Rv32imPublicProofOptions { root_fold_schedule }),
        root_fold_schedule,
        "prove accepted artifact",
    );
    let accepted_ms = millis_since(accepted_started);

    let final_statement_started = Instant::now();
    let (final_statement, final_proof) =
        prove_rv32im_final_statement_from_accepted(&accepted).expect("prove final statement");
    let final_statement_ms = millis_since(final_statement_started);

    let relations_started = Instant::now();
    let relations =
        build_rv32im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step relations");
    let relations_ms = millis_since(relations_started);

    let advices_started = Instant::now();
    let advices = build_rv32im_main_recursion_f_prime_advices(&relations).expect("build f-prime advices");
    let advices_ms = millis_since(advices_started);

    let backend_relations_started = Instant::now();
    let (spartan_shape, backend_relations) =
        build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices(&relations, &advices)
            .expect("build recursion backend relations");
    let backend_relations_ms = millis_since(backend_relations_started);
    let fixture_ms = millis_since(fixture_started);

    let selected_relation_index = selected_relation_index_from_args(backend_relations.len());
    let first_relation = backend_relations
        .get(selected_relation_index)
        .expect("shape probe requires the selected backend relation");
    let terminal_relation_index = backend_relations.len().saturating_sub(1);
    let terminal_relation = backend_relations
        .get(terminal_relation_index)
        .expect("shape probe requires a terminal backend relation");

    let shape_only_started = Instant::now();
    let shape_only = debug_measure_rv32im_main_recursion_step_shape_only_circuit_shape(&spartan_shape);
    let shape_only_ms = millis_since(shape_only_started);
    let terminal_committed_shape = if probe_mode == ProbeMode::Full {
        let terminal_committed_started = Instant::now();
        let shape = debug_measure_rv32im_terminal_f_prime_committed_step_shape(&spartan_shape, terminal_relation)
            .expect("measure terminal F' committed-step shape");
        Some((shape, millis_since(terminal_committed_started)))
    } else {
        None
    };

    let step_shape = &first_relation.payload.step_shape;
    let cover_shape = &first_relation.payload.cover_shape;
    let state_in_claim_shape = first_relation
        .payload
        .state_in_claims
        .first()
        .map(Rv32imCeClaimDigestShape::from_claim)
        .expect("state-in claim shape");
    let state_out_claim_shape = first_relation
        .payload
        .state_out_claims
        .first()
        .map(Rv32imCeClaimDigestShape::from_claim)
        .expect("state-out claim shape");
    let pi_rlc_parent_shape = Rv32imCeClaimDigestShape::from_claim(&first_relation.payload.pi_rlc.parent);
    let state_in_projection_fields_total: usize = first_relation
        .payload
        .state_in_claims
        .iter()
        .map(|claim| {
            projection_digest_field_count(
                claim.c.data.len(),
                claim.m_in,
                claim.r.len(),
                &claim.y_ring.iter().map(|row| row.len()).collect::<Vec<_>>(),
            )
        })
        .sum();
    let state_out_projection_fields_total: usize = first_relation
        .payload
        .state_out_claims
        .iter()
        .map(|claim| {
            projection_digest_field_count(
                claim.c.data.len(),
                claim.m_in,
                claim.r.len(),
                &claim.y_ring.iter().map(|row| row.len()).collect::<Vec<_>>(),
            )
        })
        .sum();
    let first_state_in = first_relation
        .payload
        .state_in_claims
        .first()
        .expect("state-in claim");
    let first_state_out = first_relation
        .payload
        .state_out_claims
        .first()
        .expect("state-out claim");
    let first_state_in_y_ring_row_lens = first_state_in
        .y_ring
        .iter()
        .map(|row| row.len())
        .collect::<Vec<_>>();
    let first_state_out_y_ring_row_lens = first_state_out
        .y_ring
        .iter()
        .map(|row| row.len())
        .collect::<Vec<_>>();
    let pi_rlc_parent_y_ring_row_lens = first_relation
        .payload
        .pi_rlc
        .parent
        .y_ring
        .iter()
        .map(|row| row.len())
        .collect::<Vec<_>>();
    let padded_child_count = usize::try_from(cover_shape.ccs_output_count).expect("padded child count");
    let actual_child_count = first_relation.payload.pi_ccs.ccs_outputs.len();
    let state_out_accumulator_phi_fields =
        accumulator_phi_dec_parent_hash_field_count(&first_relation.payload.state_out_claims);
    let work_units = ProbeWorkUnits {
        non_halt_opcode_count: opcode_count,
        semantic_step_count: usize::try_from(final_statement.folded.semantic_step_count)
            .expect("semantic step count fits usize"),
        chunk_count: usize::try_from(final_statement.folded.chunk_count).expect("chunk count fits usize"),
        chunk_fold_step_count: final_proof.steps.len(),
        relation_count: relations.len(),
        backend_relation_count: backend_relations.len(),
        fold_schedule: root_fold_schedule,
    };

    print_section("RV32IM Main Recursion Shape Probe");
    print_kv(
        "mode",
        match probe_mode {
            ProbeMode::Full => "full",
            ProbeMode::FastSummary => "fast-summary",
            ProbeMode::StageAux => "stage-aux",
            ProbeMode::ConstraintBreakdown => "constraint-breakdown",
            ProbeMode::TraceShape => "trace-shape",
        },
    );
    print_kv("mixed_opcode_non_halt_ops", opcode_count);
    print_kv("relation_count", relations.len());
    print_kv("backend_relation_count", backend_relations.len());
    print_kv("selected_relation_index", selected_relation_index);
    print_kv("terminal_relation_index", terminal_relation_index);
    print_kv("fixture_prep", format!("{fixture_ms:.3} ms"));
    print_probe_work_units("Execution Units", work_units);
    print_section("Fixture Breakdown");
    print_kv("accepted_proof.wall", format!("{accepted_ms:.3} ms"));
    print_kv("accepted_proof.total", format!("{:.3} ms", accepted_perf.total_ms));
    print_kv(
        "accepted_root_session",
        format!("{:.3} ms", accepted_perf.root_main_lane.session.total_ms),
    );
    print_kv(
        "accepted_root_rlc_dec",
        format!("{:.3} ms", accepted_root_rlc_dec_ms(&accepted_perf)),
    );
    print_kv(
        "accepted_root_ccs",
        format!("{:.3} ms", accepted_perf.root_main_lane.session.ccs_ms()),
    );
    print_kv("final_statement", format!("{final_statement_ms:.3} ms"));
    print_kv("build_relations", format!("{relations_ms:.3} ms"));
    print_kv("build_advices", format!("{advices_ms:.3} ms"));
    print_kv("build_backend_relations", format!("{backend_relations_ms:.3} ms"));

    print_shape_result(&shape_only, shape_only_ms, backend_relations.len());

    if let Some((terminal_committed_shape, terminal_committed_ms)) = &terminal_committed_shape {
        print_terminal_committed_shape(terminal_committed_shape, *terminal_committed_ms);
    }

    if probe_mode == ProbeMode::TraceShape {
        let traced =
            debug_trace_rv32im_main_recursion_step_shape_only_fingerprint_synthesize(&spartan_shape, "shape_trace")
                .expect("trace shape-only circuit");
        print_section("Shape Trace");
        print_kv("num_inputs", traced.num_inputs);
        print_kv("num_aux", traced.num_aux);
        print_kv("num_constraints", traced.num_constraints);
        return;
    }

    if probe_mode == ProbeMode::FastSummary {
        print_section("Payload Dimensions");
        print_kv("step_shape.state_in_claim_count", step_shape.state_in_claim_count);
        print_kv("step_shape.state_out_claim_count", step_shape.state_out_claim_count);
        print_kv("step_shape.fresh_claim_count", step_shape.fresh_claim_count);
        print_kv("step_shape.ccs_output_count", step_shape.ccs_output_count);
        print_kv("step_shape.child_count", step_shape.child_count);
        print_kv("cover_shape.ccs_output_count", cover_shape.ccs_output_count);
        print_kv("cover_shape.child_count", cover_shape.child_count);

        print_section("State In Claim Surface");
        print_kv("claim_count", first_relation.payload.state_in_claims.len());
        print_kv("claim.c_data_len", state_in_claim_shape.c_data_len);
        print_kv("claim.x_compact_len", first_state_in.m_in);
        print_kv("claim.r_len", state_in_claim_shape.r_len);
        print_kv("claim.y_ring_rows", state_in_claim_shape.y_ring_row_count);
        print_kv("claim.y_ring_row_lens", format!("{:?}", first_state_in_y_ring_row_lens));
        print_kv(
            "projection_hash_terms_per_claim",
            projection_digest_field_count(
                first_state_in.c.data.len(),
                first_state_in.m_in,
                first_state_in.r.len(),
                &first_state_in_y_ring_row_lens,
            ),
        );
        print_kv("projection_hash_terms_total", state_in_projection_fields_total);

        print_section("State Out Claim Surface");
        print_kv("claim_count", first_relation.payload.state_out_claims.len());
        print_kv("claim.c_data_len", state_out_claim_shape.c_data_len);
        print_kv("claim.x_compact_len", first_state_out.m_in);
        print_kv("claim.r_len", state_out_claim_shape.r_len);
        print_kv("claim.y_ring_rows", state_out_claim_shape.y_ring_row_count);
        print_kv(
            "claim.y_ring_row_lens",
            format!("{:?}", first_state_out_y_ring_row_lens),
        );
        print_kv(
            "projection_hash_terms_per_claim",
            projection_digest_field_count(
                first_state_out.c.data.len(),
                first_state_out.m_in,
                first_state_out.r.len(),
                &first_state_out_y_ring_row_lens,
            ),
        );
        print_kv("projection_hash_terms_total", state_out_projection_fields_total);
        print_kv("accumulator_phi_hash_terms", state_out_accumulator_phi_fields);

        print_section("Pi RLC Public Surface");
        print_kv("actual_child_count", actual_child_count);
        print_kv("padded_child_count", padded_child_count);
        print_kv("parent.c_data_len", pi_rlc_parent_shape.c_data_len);
        print_kv("parent.commitment_rows", D);
        print_kv(
            "parent.commitment_cols",
            usize::try_from(pi_rlc_parent_shape.c_data_len).expect("commitment len") / D,
        );
        print_kv("parent.x_compact_len", first_relation.payload.pi_rlc.parent.m_in);
        print_kv("parent.r_len", pi_rlc_parent_shape.r_len);
        print_kv("parent.y_ring_rows", pi_rlc_parent_shape.y_ring_row_count);
        print_kv("parent.y_ring_row_lens", format!("{:?}", pi_rlc_parent_y_ring_row_lens));
        print_kv("parent.y_zcol_len", pi_rlc_parent_shape.y_zcol_len);
        print_kv(
            "dense_c_scalars_across_children",
            padded_child_count * usize::try_from(pi_rlc_parent_shape.c_data_len).expect("parent c_data len"),
        );
        print_kv(
            "dense_y_ring_k_scalars_per_claim",
            first_relation
                .payload
                .pi_rlc
                .parent
                .y_ring
                .iter()
                .map(|row| row.len())
                .sum::<usize>(),
        );
        let fresh_child_count = usize::try_from(step_shape.fresh_claim_count).expect("fresh child count");
        print_pi_rlc_public_child_families(
            &first_relation,
            fresh_child_count,
            actual_child_count,
            &pi_rlc_parent_shape,
        );
        print_backend_relation_commitment_sparsity(&backend_relations);
        let fast_summary = FastSummaryPerf {
            fixture_ms,
            accepted_wall_ms: accepted_ms,
            accepted_perf: accepted_perf.clone(),
            final_statement_ms,
            relations_ms,
            advices_ms,
            backend_relations_ms,
            shape_only_ms,
        };
        print_probe_work_units("Key Per-Opcode Units", work_units);
        print_key_per_fold_summary("Key Per-Fold Summary", &fast_summary, work_units.chunk_fold_step_count);
        print_key_per_opcode_summary("Key Per-Opcode Summary", &fast_summary, opcode_count);
        print_per_opcode_components("Per-Opcode Components", &fast_summary, opcode_count);
        return;
    }

    if probe_mode == ProbeMode::ConstraintBreakdown {
        let chunk_replay_aux_started = Instant::now();
        let chunk_replay_aux = debug_measure_rv32im_main_recursion_step_chunk_replay_aux_counts(first_relation)
            .expect("measure first-step chunk replay aux counts");
        let chunk_replay_aux_ms = millis_since(chunk_replay_aux_started);

        let chunk_replay_tail_digest_started = Instant::now();
        let chunk_replay_tail_digest =
            debug_measure_rv32im_main_recursion_step_chunk_replay_tail_digest_aux_breakdown(first_relation)
                .expect("measure first-step chunk replay tail digest aux breakdown");
        let chunk_replay_tail_digest_ms = millis_since(chunk_replay_tail_digest_started);

        let pi_ccs_bind_me_inputs_started = Instant::now();
        let pi_ccs_bind_me_inputs =
            debug_measure_rv32im_main_recursion_step_pi_ccs_bind_me_inputs_aux_breakdown(first_relation)
                .expect("measure first-step pi_ccs bind_me_inputs aux breakdown");
        let pi_ccs_bind_me_inputs_ms = millis_since(pi_ccs_bind_me_inputs_started);

        let pi_ccs_constraints_started = Instant::now();
        let pi_ccs_constraints = debug_measure_rv32im_main_recursion_step_pi_ccs_constraint_counts(first_relation)
            .expect("measure first-step pi_ccs constraint counts");
        let pi_ccs_constraints_ms = millis_since(pi_ccs_constraints_started);

        let pi_ccs_sumcheck_started = Instant::now();
        let pi_ccs_sumcheck =
            debug_measure_rv32im_main_recursion_step_pi_ccs_sumcheck_constraint_breakdown(first_relation)
                .expect("measure first-step pi_ccs sumcheck constraint breakdown");
        let pi_ccs_sumcheck_ms = millis_since(pi_ccs_sumcheck_started);

        let pi_rlc_public_started = Instant::now();
        let pi_rlc_public = debug_measure_rv32im_main_recursion_step_pi_rlc_public_constraint_breakdown(first_relation)
            .expect("measure first-step pi_rlc public breakdown");
        let pi_rlc_public_ms = millis_since(pi_rlc_public_started);

        let pi_rlc_public_stage_started = Instant::now();
        let pi_rlc_public_stage =
            debug_measure_rv32im_main_recursion_step_pi_rlc_public_stage_breakdown(first_relation)
                .expect("measure first-step pi_rlc public stage breakdown");
        let pi_rlc_public_stage_ms = millis_since(pi_rlc_public_stage_started);

        print_section("Chunk NIFS Verifier Aux Hotspots");
        print_kv("measure.wall", format!("{chunk_replay_aux_ms:.3} ms"));
        print_cumulative_and_delta("after_state_cover", 0, chunk_replay_aux.after_state_cover);
        print_cumulative_and_delta(
            "after_public_chunk_meta",
            chunk_replay_aux.after_state_cover,
            chunk_replay_aux.after_chunk_meta,
        );
        print_cumulative_and_delta(
            "after_pi_ccs",
            chunk_replay_aux.after_chunk_meta,
            chunk_replay_aux.after_pi_ccs,
        );
        print_cumulative_and_delta(
            "after_synthetic_relation_io",
            chunk_replay_aux.after_pi_ccs,
            chunk_replay_aux.after_synthetic_relation_io,
        );
        print_cumulative_and_delta(
            "after_pi_rlc_parent_claim",
            chunk_replay_aux.after_synthetic_relation_io,
            chunk_replay_aux.after_pi_rlc_parent_claim,
        );
        print_cumulative_and_delta(
            "after_pi_rlc_rhos",
            chunk_replay_aux.after_pi_rlc_parent_claim,
            chunk_replay_aux.after_pi_rlc_rhos,
        );
        print_cumulative_and_delta(
            "after_pi_rlc_rho_mats",
            chunk_replay_aux.after_pi_rlc_rhos,
            chunk_replay_aux.after_pi_rlc_rho_mats,
        );
        print_cumulative_and_delta(
            "after_pi_rlc_public",
            chunk_replay_aux.after_pi_rlc_rho_mats,
            chunk_replay_aux.after_pi_rlc_public,
        );
        print_cumulative_and_delta(
            "after_pi_rlc",
            chunk_replay_aux.after_pi_rlc_public,
            chunk_replay_aux.after_pi_rlc,
        );
        print_cumulative_and_delta(
            "after_chunk_nifs_body",
            chunk_replay_aux.after_pi_rlc,
            chunk_replay_aux.after_chunk_body,
        );
        print_cumulative_and_delta(
            "after_chunk_nifs_verifier",
            chunk_replay_aux.after_chunk_body,
            chunk_replay_aux.after_chunk_replay,
        );

        let mut tail_claim_digest_deltas = Vec::with_capacity(chunk_replay_tail_digest.claim_after_digests.len());
        let mut tail_digest_prev = chunk_replay_tail_digest.after_header;
        let mut tail_claim_digest_total = 0usize;
        for idx in 0..chunk_replay_tail_digest.claim_after_digests.len() {
            let claim_delta = chunk_replay_tail_digest.claim_after_digests[idx].saturating_sub(tail_digest_prev);
            tail_claim_digest_total += claim_delta;
            tail_claim_digest_deltas.push((idx, claim_delta));
            tail_digest_prev = chunk_replay_tail_digest.claim_after_digests[idx];
        }
        let tail_outer_hash_delta = chunk_replay_tail_digest
            .after_outer_hash
            .saturating_sub(tail_digest_prev);
        print_section("Chunk NIFS Verifier Tail Digest Aux");
        print_kv("measure.wall", format!("{chunk_replay_tail_digest_ms:.3} ms"));
        print_kv(
            "header",
            chunk_replay_tail_digest
                .after_header
                .saturating_sub(chunk_replay_aux.after_chunk_body),
        );
        print_kv("claim_digest_total", tail_claim_digest_total);
        print_kv("outer_hash", tail_outer_hash_delta);
        for (idx, claim_delta) in &tail_claim_digest_deltas {
            print_kv(&format!("claim_{idx}.digest"), *claim_delta);
        }

        let mut pi_ccs_bind_me_input_deltas = Vec::with_capacity(1 + pi_ccs_bind_me_inputs.after_claim_digests.len());
        let mut bind_prev = pi_ccs_bind_me_inputs.after_bind_header;
        for (idx, end) in pi_ccs_bind_me_inputs.after_claim_digests.iter().enumerate() {
            pi_ccs_bind_me_input_deltas.push((format!("claim_digest_{idx}"), end.saturating_sub(bind_prev)));
            bind_prev = *end;
        }
        pi_ccs_bind_me_input_deltas.push((
            "bind_digests".to_string(),
            pi_ccs_bind_me_inputs
                .after_bind_digests
                .saturating_sub(bind_prev),
        ));
        print_section("Pi CCS Bind ME Inputs Aux");
        print_kv("measure.wall", format!("{pi_ccs_bind_me_inputs_ms:.3} ms"));
        for (name, delta) in &pi_ccs_bind_me_input_deltas {
            print_kv(name, *delta);
        }

        print_section("Pi CCS Constraints");
        print_kv("measure.wall", format!("{pi_ccs_constraints_ms:.3} ms"));
        print_cumulative_and_delta("after_bind_header", 0, pi_ccs_constraints.after_bind_header);
        print_cumulative_and_delta(
            "after_bind_me_inputs",
            pi_ccs_constraints.after_bind_header,
            pi_ccs_constraints.after_bind_me_inputs,
        );
        print_cumulative_and_delta(
            "after_sample_challenges",
            pi_ccs_constraints.after_bind_me_inputs,
            pi_ccs_constraints.after_sample_challenges,
        );
        print_cumulative_and_delta(
            "after_alloc_fresh_claims",
            pi_ccs_constraints.after_sample_challenges,
            pi_ccs_constraints.after_alloc_fresh_claims,
        );
        print_cumulative_and_delta(
            "after_fe_sumcheck",
            pi_ccs_constraints.after_alloc_fresh_claims,
            pi_ccs_constraints.after_fe_sumcheck,
        );
        print_cumulative_and_delta(
            "after_nc_sumcheck",
            pi_ccs_constraints.after_fe_sumcheck,
            pi_ccs_constraints.after_nc_sumcheck,
        );
        print_cumulative_and_delta(
            "after_fold_digest",
            pi_ccs_constraints.after_nc_sumcheck,
            pi_ccs_constraints.after_fold_digest,
        );
        print_cumulative_and_delta(
            "after_alloc_outputs",
            pi_ccs_constraints.after_fold_digest,
            pi_ccs_constraints.after_alloc_outputs,
        );
        print_cumulative_and_delta(
            "after_output_binding",
            pi_ccs_constraints.after_alloc_outputs,
            pi_ccs_constraints.after_output_binding,
        );
        print_cumulative_and_delta(
            "after_terminal_fe",
            pi_ccs_constraints.after_output_binding,
            pi_ccs_constraints.after_terminal_fe,
        );
        print_cumulative_and_delta(
            "after_terminal_nc",
            pi_ccs_constraints.after_terminal_fe,
            pi_ccs_constraints.after_terminal_nc,
        );

        print_named_constraint_breakdown(
            "Pi CCS FE Sumcheck Constraints",
            pi_ccs_sumcheck_ms,
            &pi_ccs_sumcheck.fe_cover_round_lengths,
            &pi_ccs_sumcheck.fe_effective_round_lengths,
            &pi_ccs_sumcheck.fe_stages,
        );
        print_named_constraint_breakdown(
            "Pi CCS NC Sumcheck Constraints",
            pi_ccs_sumcheck_ms,
            &pi_ccs_sumcheck.nc_cover_round_lengths,
            &pi_ccs_sumcheck.nc_effective_round_lengths,
            &pi_ccs_sumcheck.nc_stages,
        );

        print_section("Pi RLC Public");
        print_kv("measure.wall", format!("{pi_rlc_public_ms:.3} ms"));
        print_kv("shared_point", pi_rlc_public.shared_point_constraints);
        print_kv("x", pi_rlc_public.x_constraints);
        print_kv("c", pi_rlc_public.c_constraints);
        print_kv("y_ring", pi_rlc_public.y_ring_constraints);
        print_kv("y_zcol", pi_rlc_public.y_zcol_constraints);
        print_kv("aux", pi_rlc_public.aux_constraints);
        print_kv("total", pi_rlc_public.total_constraints);

        print_section("Pi RLC Public Stages");
        print_kv("measure.wall", format!("{pi_rlc_public_stage_ms:.3} ms"));
        for stage in &pi_rlc_public_stage.stages {
            print_kv(&stage.name, stage.delta);
        }
        return;
    }

    let top_level_aux_started = Instant::now();
    let top_level_aux = debug_measure_rv32im_main_recursion_step_stage_aux_counts(&spartan_shape, first_relation)
        .expect("measure first-step stage aux counts");
    let top_level_aux_ms = millis_since(top_level_aux_started);

    if probe_mode == ProbeMode::StageAux {
        let chunk_replay_aux_started = Instant::now();
        let chunk_replay_aux = debug_measure_rv32im_main_recursion_step_chunk_replay_aux_counts(first_relation)
            .expect("measure first-step chunk replay aux counts");
        let chunk_replay_aux_ms = millis_since(chunk_replay_aux_started);
        let chunk_replay_tail_aux_started = Instant::now();
        let chunk_replay_tail_aux =
            debug_measure_rv32im_main_recursion_step_chunk_replay_tail_aux_counts(first_relation)
                .expect("measure first-step chunk replay tail aux counts");
        let chunk_replay_tail_aux_ms = millis_since(chunk_replay_tail_aux_started);
        let chunk_replay_tail_digest_started = Instant::now();
        let chunk_replay_tail_digest =
            debug_measure_rv32im_main_recursion_step_chunk_replay_tail_digest_aux_breakdown(first_relation)
                .expect("measure first-step chunk replay tail digest aux breakdown");
        let chunk_replay_tail_digest_ms = millis_since(chunk_replay_tail_digest_started);

        print_section("Top-Level Aux");
        print_kv("measure.wall", format!("{top_level_aux_ms:.3} ms"));
        print_cumulative_and_delta(
            "after_private_witness_inputs",
            0,
            top_level_aux.after_private_witness_inputs,
        );
        print_cumulative_and_delta(
            "after_alloc_cover_states",
            top_level_aux.after_private_witness_inputs,
            top_level_aux.after_alloc_cover_states,
        );
        print_cumulative_and_delta(
            "after_bind_state_and_pc",
            top_level_aux.after_alloc_cover_states,
            top_level_aux.after_bind_state_and_pc,
        );
        print_cumulative_and_delta(
            "after_chunk_nifs_verifier",
            top_level_aux.after_bind_state_and_pc,
            top_level_aux.after_chunk_replay,
        );
        print_cumulative_and_delta(
            "after_inactive_side_lane_x_out",
            top_level_aux.after_chunk_replay,
            top_level_aux.after_inactive_side_lane_and_x_out,
        );
        print_cumulative_and_delta(
            "after_public_output_eq",
            top_level_aux.after_inactive_side_lane_and_x_out,
            top_level_aux.after_public_output_eq,
        );
        print_section("Chunk NIFS Verifier Aux");
        print_kv("measure.wall", format!("{chunk_replay_aux_ms:.3} ms"));
        print_cumulative_and_delta("after_state_cover", 0, chunk_replay_aux.after_state_cover);
        print_cumulative_and_delta(
            "after_public_chunk_meta",
            chunk_replay_aux.after_state_cover,
            chunk_replay_aux.after_chunk_meta,
        );
        print_cumulative_and_delta(
            "after_pi_ccs",
            chunk_replay_aux.after_chunk_meta,
            chunk_replay_aux.after_pi_ccs,
        );
        print_cumulative_and_delta(
            "after_synthetic_relation_io",
            chunk_replay_aux.after_pi_ccs,
            chunk_replay_aux.after_synthetic_relation_io,
        );
        print_cumulative_and_delta(
            "after_pi_rlc_parent_claim",
            chunk_replay_aux.after_synthetic_relation_io,
            chunk_replay_aux.after_pi_rlc_parent_claim,
        );
        print_cumulative_and_delta(
            "after_pi_rlc_rhos",
            chunk_replay_aux.after_pi_rlc_parent_claim,
            chunk_replay_aux.after_pi_rlc_rhos,
        );
        print_cumulative_and_delta(
            "after_pi_rlc_rho_mats",
            chunk_replay_aux.after_pi_rlc_rhos,
            chunk_replay_aux.after_pi_rlc_rho_mats,
        );
        print_cumulative_and_delta(
            "after_pi_rlc_public",
            chunk_replay_aux.after_pi_rlc_rho_mats,
            chunk_replay_aux.after_pi_rlc_public,
        );
        print_cumulative_and_delta(
            "after_pi_rlc",
            chunk_replay_aux.after_pi_rlc_public,
            chunk_replay_aux.after_pi_rlc,
        );
        print_cumulative_and_delta(
            "after_chunk_nifs_body",
            chunk_replay_aux.after_pi_rlc,
            chunk_replay_aux.after_chunk_body,
        );
        print_cumulative_and_delta(
            "after_chunk_nifs_verifier",
            chunk_replay_aux.after_chunk_body,
            chunk_replay_aux.after_chunk_replay,
        );
        print_section("Chunk NIFS Verifier Tail Aux");
        print_kv("measure.wall", format!("{chunk_replay_tail_aux_ms:.3} ms"));
        print_cumulative_and_delta(
            "after_state_out_projection_eq",
            chunk_replay_aux.after_chunk_body,
            chunk_replay_tail_aux.after_state_out_projection_eq,
        );
        print_cumulative_and_delta(
            "after_expected_digest",
            chunk_replay_tail_aux.after_state_out_projection_eq,
            chunk_replay_tail_aux.after_expected_digest,
        );
        print_cumulative_and_delta(
            "after_chunk_done_tag",
            chunk_replay_tail_aux.after_expected_digest,
            chunk_replay_tail_aux.after_chunk_done,
        );
        print_cumulative_and_delta(
            "after_transcript_state_eq",
            chunk_replay_tail_aux.after_chunk_done,
            chunk_replay_tail_aux.after_transcript_state_eq,
        );
        print_cumulative_and_delta(
            "after_transcript_absorbed_eq",
            chunk_replay_tail_aux.after_transcript_state_eq,
            chunk_replay_tail_aux.after_transcript_absorbed_eq,
        );
        let tail_header_delta = chunk_replay_tail_digest
            .after_header
            .saturating_sub(chunk_replay_aux.after_chunk_body);
        let mut tail_total_claim_digest = 0usize;
        let mut prev = chunk_replay_tail_digest.after_header;
        for after_digest in &chunk_replay_tail_digest.claim_after_digests {
            tail_total_claim_digest += after_digest.saturating_sub(prev);
            prev = *after_digest;
        }
        let tail_outer_hash_delta = chunk_replay_tail_digest
            .after_outer_hash
            .saturating_sub(prev);
        print_section("Chunk NIFS Verifier Tail Digest Aux");
        print_kv("measure.wall", format!("{chunk_replay_tail_digest_ms:.3} ms"));
        print_kv("header", tail_header_delta);
        print_kv("claim_digest_total", tail_total_claim_digest);
        print_kv("outer_hash", tail_outer_hash_delta);
        return;
    }

    let chunk_replay_aux_started = Instant::now();
    let chunk_replay_aux = debug_measure_rv32im_main_recursion_step_chunk_replay_aux_counts(first_relation)
        .expect("measure first-step chunk replay aux counts");
    let chunk_replay_aux_ms = millis_since(chunk_replay_aux_started);

    let chunk_replay_tail_aux_started = Instant::now();
    let chunk_replay_tail_aux = debug_measure_rv32im_main_recursion_step_chunk_replay_tail_aux_counts(first_relation)
        .expect("measure first-step chunk replay tail aux counts");
    let chunk_replay_tail_aux_ms = millis_since(chunk_replay_tail_aux_started);

    let pi_ccs_aux_started = Instant::now();
    let pi_ccs_aux = debug_measure_rv32im_main_recursion_step_pi_ccs_aux_counts(first_relation)
        .expect("measure first-step pi_ccs aux counts");
    let pi_ccs_aux_ms = millis_since(pi_ccs_aux_started);

    let pi_ccs_constraints_started = Instant::now();
    let pi_ccs_constraints = debug_measure_rv32im_main_recursion_step_pi_ccs_constraint_counts(first_relation)
        .expect("measure first-step pi_ccs constraint counts");
    let pi_ccs_constraints_ms = millis_since(pi_ccs_constraints_started);

    let pi_ccs_bind_me_inputs_started = Instant::now();
    let pi_ccs_bind_me_inputs =
        debug_measure_rv32im_main_recursion_step_pi_ccs_bind_me_inputs_aux_breakdown(first_relation)
            .expect("measure first-step pi_ccs bind_me_inputs aux breakdown");
    let pi_ccs_bind_me_inputs_ms = millis_since(pi_ccs_bind_me_inputs_started);

    let pi_ccs_sumcheck_started = Instant::now();
    let pi_ccs_sumcheck = debug_measure_rv32im_main_recursion_step_pi_ccs_sumcheck_constraint_breakdown(first_relation)
        .expect("measure first-step pi_ccs sumcheck constraint breakdown");
    let pi_ccs_sumcheck_ms = millis_since(pi_ccs_sumcheck_started);

    let pi_rlc_public_started = Instant::now();
    let pi_rlc_public = debug_measure_rv32im_main_recursion_step_pi_rlc_public_constraint_breakdown(first_relation)
        .expect("measure first-step pi_rlc public breakdown");
    let pi_rlc_public_ms = millis_since(pi_rlc_public_started);

    let pi_rlc_public_stage_started = Instant::now();
    let pi_rlc_public_stage = debug_measure_rv32im_main_recursion_step_pi_rlc_public_stage_breakdown(first_relation)
        .expect("measure first-step pi_rlc public stage breakdown");
    let pi_rlc_public_stage_ms = millis_since(pi_rlc_public_stage_started);

    let chunk_replay_tail_digest_started = Instant::now();
    let chunk_replay_tail_digest =
        debug_measure_rv32im_main_recursion_step_chunk_replay_tail_digest_aux_breakdown(first_relation)
            .expect("measure first-step chunk replay tail digest aux breakdown");
    let chunk_replay_tail_digest_ms = millis_since(chunk_replay_tail_digest_started);

    let synth_started = Instant::now();
    let shape_synth = debug_measure_rv32im_main_recursion_step_spartan_shape_synthesis(&spartan_shape, first_relation)
        .expect("measure first-step shape synthesis");
    let synth_ms = millis_since(synth_started);

    let live_shape_started = Instant::now();
    let live_shape = debug_measure_rv32im_main_recursion_step_spartan_circuit_shape(&spartan_shape, first_relation)
        .expect("measure first-step circuit shape");
    let live_shape_ms = millis_since(live_shape_started);
    let pi_ccs_fingerprint_started = Instant::now();
    let pi_ccs_fingerprint = debug_measure_rv32im_main_recursion_step_pi_ccs_fingerprint(first_relation)
        .expect("measure first-step pi_ccs fingerprint");
    let pi_ccs_fingerprint_ms = millis_since(pi_ccs_fingerprint_started);
    let chunk_replay_fingerprint_started = Instant::now();
    let chunk_replay_fingerprint = debug_measure_rv32im_main_recursion_step_chunk_replay_fingerprint(first_relation)
        .expect("measure first-step chunk replay fingerprint");
    let chunk_replay_fingerprint_ms = millis_since(chunk_replay_fingerprint_started);

    let fixed_shape_sanity_started = Instant::now();
    let mut perturbed_relation = first_relation.clone();
    perturb_backend_relation_values(&mut perturbed_relation);
    let perturbed_shape =
        debug_measure_rv32im_main_recursion_step_spartan_circuit_shape(&spartan_shape, &perturbed_relation);
    let perturbed_pi_ccs_fingerprint = debug_measure_rv32im_main_recursion_step_pi_ccs_fingerprint(&perturbed_relation);
    let perturbed_chunk_replay_fingerprint =
        debug_measure_rv32im_main_recursion_step_chunk_replay_fingerprint(&perturbed_relation);
    let fixed_shape_sanity_ms = millis_since(fixed_shape_sanity_started);

    let fixed_shape_family_started = Instant::now();
    let mut state_in_r_relation = first_relation.clone();
    perturb_state_in_r_values(&mut state_in_r_relation);
    let state_in_r_status =
        fixed_shape_family_status(&spartan_shape, &live_shape.constraint_fingerprint, &state_in_r_relation);
    let mut state_in_y_ring_relation = first_relation.clone();
    perturb_state_in_y_ring_values(&mut state_in_y_ring_relation);
    let state_in_y_ring_status = fixed_shape_family_status(
        &spartan_shape,
        &live_shape.constraint_fingerprint,
        &state_in_y_ring_relation,
    );
    let mut state_in_projection_relation = first_relation.clone();
    perturb_state_in_projection_values(&mut state_in_projection_relation);
    let state_in_projection_status = fixed_shape_family_status(
        &spartan_shape,
        &live_shape.constraint_fingerprint,
        &state_in_projection_relation,
    );
    let mut pi_ccs_alpha_relation = first_relation.clone();
    perturb_pi_ccs_alpha_values(&mut pi_ccs_alpha_relation);
    let pi_ccs_alpha_status = fixed_shape_family_status(
        &spartan_shape,
        &live_shape.constraint_fingerprint,
        &pi_ccs_alpha_relation,
    );
    let mut pi_ccs_gamma_relation = first_relation.clone();
    perturb_pi_ccs_gamma_value(&mut pi_ccs_gamma_relation);
    let pi_ccs_gamma_status = fixed_shape_family_status(
        &spartan_shape,
        &live_shape.constraint_fingerprint,
        &pi_ccs_gamma_relation,
    );
    let mut state_out_projection_relation = first_relation.clone();
    perturb_state_out_projection_values(&mut state_out_projection_relation);
    let state_out_projection_status = fixed_shape_family_status(
        &spartan_shape,
        &live_shape.constraint_fingerprint,
        &state_out_projection_relation,
    );
    let mut pi_ccs_output_y_ring_relation = first_relation.clone();
    perturb_pi_ccs_output_y_ring_values(&mut pi_ccs_output_y_ring_relation);
    let pi_ccs_output_y_ring_status = fixed_shape_family_status(
        &spartan_shape,
        &live_shape.constraint_fingerprint,
        &pi_ccs_output_y_ring_relation,
    );
    let mut pi_ccs_output_y_zcol_relation = first_relation.clone();
    perturb_pi_ccs_output_y_zcol_values(&mut pi_ccs_output_y_zcol_relation);
    let pi_ccs_output_y_zcol_status = fixed_shape_family_status(
        &spartan_shape,
        &live_shape.constraint_fingerprint,
        &pi_ccs_output_y_zcol_relation,
    );
    let mut pi_dec_child_y_ring_relation = first_relation.clone();
    perturb_pi_dec_child_y_ring_values(&mut pi_dec_child_y_ring_relation);
    let pi_dec_child_y_ring_status = fixed_shape_family_status(
        &spartan_shape,
        &live_shape.constraint_fingerprint,
        &pi_dec_child_y_ring_relation,
    );
    let mut pi_rlc_parent_relation = first_relation.clone();
    perturb_pi_rlc_parent_values(&mut pi_rlc_parent_relation);
    let pi_rlc_parent_status = fixed_shape_family_status(
        &spartan_shape,
        &live_shape.constraint_fingerprint,
        &pi_rlc_parent_relation,
    );
    let mut fresh_claim_relation = first_relation.clone();
    perturb_fresh_claim_values(&mut fresh_claim_relation);
    let fresh_claim_status = fixed_shape_family_status(
        &spartan_shape,
        &live_shape.constraint_fingerprint,
        &fresh_claim_relation,
    );
    let mut fresh_witness_relation = first_relation.clone();
    perturb_fresh_witness_values(&mut fresh_witness_relation);
    let fresh_witness_status = fixed_shape_family_status(
        &spartan_shape,
        &live_shape.constraint_fingerprint,
        &fresh_witness_relation,
    );
    let state_in_projection_pi_ccs_fingerprint =
        debug_measure_rv32im_main_recursion_step_pi_ccs_fingerprint(&state_in_projection_relation);
    let fresh_claim_pi_ccs_fingerprint =
        debug_measure_rv32im_main_recursion_step_pi_ccs_fingerprint(&fresh_claim_relation);
    let fixed_shape_family_ms = millis_since(fixed_shape_family_started);

    print_kv("live_shape.wall", format!("{live_shape_ms:.3} ms"));
    print_kv("live_shape.num_inputs", live_shape.num_inputs);
    print_kv("live_shape.num_aux", live_shape.num_aux);
    print_kv("live_shape.num_constraints", live_shape.num_constraints);
    print_kv(
        "live_shape.total_constraints_across_all_relations",
        live_shape.num_constraints * backend_relations.len(),
    );

    print_section("Shape Synthesis");
    print_kv("shape_synth.wall", format!("{synth_ms:.3} ms"));
    print_kv("shape_synth.shared", format!("{:.3} ms", shape_synth.shared_ms));
    print_kv(
        "shape_synth.precommitted",
        format!("{:.3} ms", shape_synth.precommitted_ms),
    );
    print_kv("shape_synth.synthesize", format!("{:.3} ms", shape_synth.synthesize_ms));
    print_kv("shape_synth.num_inputs", shape_synth.num_inputs);
    print_kv("shape_synth.num_aux", shape_synth.num_aux);
    print_kv("shape_synth.num_constraints", shape_synth.num_constraints);
    print_kv(
        "shape_synth.total_constraints_across_all_relations",
        shape_synth.num_constraints * backend_relations.len(),
    );

    print_section("Fixed-Shape Sanity");
    print_kv("measure.wall", format!("{fixed_shape_sanity_ms:.3} ms"));
    print_kv("baseline.constraint_fingerprint", &live_shape.constraint_fingerprint);
    match &perturbed_shape {
        Ok(perturbed_shape) => {
            print_kv(
                "perturbed.constraint_fingerprint",
                &perturbed_shape.constraint_fingerprint,
            );
            print_kv(
                "fingerprint_equal",
                if live_shape.constraint_fingerprint == perturbed_shape.constraint_fingerprint {
                    "yes"
                } else {
                    "no"
                },
            );
            print_kv(
                "num_constraints_equal",
                if live_shape.num_constraints == perturbed_shape.num_constraints {
                    "yes"
                } else {
                    "no"
                },
            );
            print_kv(
                "num_aux_equal",
                if live_shape.num_aux == perturbed_shape.num_aux {
                    "yes"
                } else {
                    "no"
                },
            );
        }
        Err(err) => {
            print_kv("perturbed.constraint_fingerprint", "unsat");
            print_kv("perturbed.error", err);
            print_kv("fingerprint_equal", "n/a");
            print_kv("num_constraints_equal", "n/a");
            print_kv("num_aux_equal", "n/a");
        }
    }

    print_section("Fixed-Shape Families");
    print_kv("measure.wall", format!("{fixed_shape_family_ms:.3} ms"));
    print_kv("state_in_r_only", state_in_r_status);
    print_kv("state_in_y_ring_only", state_in_y_ring_status);
    print_kv("state_in_projection_only", state_in_projection_status);
    print_kv("pi_ccs_alpha_only", pi_ccs_alpha_status);
    print_kv("pi_ccs_gamma_only", pi_ccs_gamma_status);
    print_kv("state_out_projection_only", state_out_projection_status);
    print_kv("pi_ccs_output_y_ring_only", pi_ccs_output_y_ring_status);
    print_kv("pi_ccs_output_y_zcol_only", pi_ccs_output_y_zcol_status);
    print_kv("pi_dec_child_y_ring_only", pi_dec_child_y_ring_status);
    print_kv("pi_rlc_parent_only", pi_rlc_parent_status);
    print_kv("fresh_claim_only", fresh_claim_status);
    print_kv("fresh_witness_only", fresh_witness_status);

    print_section("Fixed-Shape Drift Localizer");
    print_kv("measure.wall", format!("{pi_ccs_fingerprint_ms:.3} ms"));
    match &state_in_projection_pi_ccs_fingerprint {
        Ok(fingerprint) => {
            if let Some((stage, baseline, perturbed)) = first_pi_ccs_stage_diff(&pi_ccs_fingerprint, fingerprint) {
                print_kv("state_in_projection.first_diff_stage", stage);
                print_kv("state_in_projection.baseline", baseline);
                print_kv("state_in_projection.perturbed", perturbed);
            } else {
                print_kv("state_in_projection.first_diff_stage", "none");
            }
        }
        Err(_) => {
            print_kv("state_in_projection.first_diff_stage", "unsat");
        }
    }
    match &fresh_claim_pi_ccs_fingerprint {
        Ok(fingerprint) => {
            if let Some((stage, baseline, perturbed)) = first_pi_ccs_stage_diff(&pi_ccs_fingerprint, fingerprint) {
                print_kv("fresh_claim.first_diff_stage", stage);
                print_kv("fresh_claim.baseline", baseline);
                print_kv("fresh_claim.perturbed", perturbed);
            } else {
                print_kv("fresh_claim.first_diff_stage", "none");
            }
        }
        Err(_) => {
            print_kv("fresh_claim.first_diff_stage", "unsat");
        }
    }
    match &perturbed_pi_ccs_fingerprint {
        Ok(perturbed_pi_ccs_fingerprint) => {
            if let Some((stage, baseline, perturbed)) =
                first_pi_ccs_stage_diff(&pi_ccs_fingerprint, perturbed_pi_ccs_fingerprint)
            {
                print_kv("full_perturb.pi_ccs_first_diff_stage", stage);
                print_kv("full_perturb.pi_ccs_baseline", baseline);
                print_kv("full_perturb.pi_ccs_perturbed", perturbed);
            } else {
                print_kv("full_perturb.pi_ccs_first_diff_stage", "none");
            }
        }
        Err(_) => {
            print_kv("full_perturb.pi_ccs_first_diff_stage", "unsat");
        }
    }
    print_kv(
        "chunk_nifs_verifier.measure.wall",
        format!("{chunk_replay_fingerprint_ms:.3} ms"),
    );
    match &perturbed_chunk_replay_fingerprint {
        Ok(perturbed_chunk_replay_fingerprint) => {
            let chunk_replay_stages = [
                (
                    "after_state_cover",
                    &chunk_replay_fingerprint.after_state_cover,
                    &perturbed_chunk_replay_fingerprint.after_state_cover,
                ),
                (
                    "after_public_chunk_meta",
                    &chunk_replay_fingerprint.after_chunk_meta,
                    &perturbed_chunk_replay_fingerprint.after_chunk_meta,
                ),
                (
                    "after_pi_ccs",
                    &chunk_replay_fingerprint.after_pi_ccs,
                    &perturbed_chunk_replay_fingerprint.after_pi_ccs,
                ),
                (
                    "after_synthetic_relation_io",
                    &chunk_replay_fingerprint.after_synthetic_relation_io,
                    &perturbed_chunk_replay_fingerprint.after_synthetic_relation_io,
                ),
                (
                    "after_pi_rlc_parent_claim",
                    &chunk_replay_fingerprint.after_pi_rlc_parent_claim,
                    &perturbed_chunk_replay_fingerprint.after_pi_rlc_parent_claim,
                ),
                (
                    "after_pi_rlc_rhos",
                    &chunk_replay_fingerprint.after_pi_rlc_rhos,
                    &perturbed_chunk_replay_fingerprint.after_pi_rlc_rhos,
                ),
                (
                    "after_pi_rlc_rho_mats",
                    &chunk_replay_fingerprint.after_pi_rlc_rho_mats,
                    &perturbed_chunk_replay_fingerprint.after_pi_rlc_rho_mats,
                ),
                (
                    "after_pi_rlc_public",
                    &chunk_replay_fingerprint.after_pi_rlc_public,
                    &perturbed_chunk_replay_fingerprint.after_pi_rlc_public,
                ),
                (
                    "after_pi_rlc",
                    &chunk_replay_fingerprint.after_pi_rlc,
                    &perturbed_chunk_replay_fingerprint.after_pi_rlc,
                ),
                (
                    "after_chunk_nifs_body",
                    &chunk_replay_fingerprint.after_chunk_body,
                    &perturbed_chunk_replay_fingerprint.after_chunk_body,
                ),
                (
                    "after_chunk_nifs_verifier",
                    &chunk_replay_fingerprint.after_chunk_replay,
                    &perturbed_chunk_replay_fingerprint.after_chunk_replay,
                ),
            ];
            if let Some((stage, baseline, perturbed)) = chunk_replay_stages
                .into_iter()
                .find(|(_, baseline, perturbed)| baseline != perturbed)
            {
                print_kv("chunk_nifs_verifier.first_diff_stage", stage);
                print_kv("chunk_nifs_verifier.baseline", baseline);
                print_kv("chunk_nifs_verifier.perturbed", perturbed);
            } else {
                print_kv("chunk_nifs_verifier.first_diff_stage", "none");
            }
        }
        Err(_) => {
            print_kv("chunk_nifs_verifier.first_diff_stage", "unsat");
        }
    }

    print_section("Top-Level Aux");
    print_kv("measure.wall", format!("{top_level_aux_ms:.3} ms"));
    print_cumulative_and_delta(
        "after_private_witness_inputs",
        0,
        top_level_aux.after_private_witness_inputs,
    );
    print_cumulative_and_delta(
        "after_alloc_cover_states",
        top_level_aux.after_private_witness_inputs,
        top_level_aux.after_alloc_cover_states,
    );
    print_cumulative_and_delta(
        "after_bind_state_and_pc",
        top_level_aux.after_alloc_cover_states,
        top_level_aux.after_bind_state_and_pc,
    );
    print_cumulative_and_delta(
        "after_chunk_nifs_verifier",
        top_level_aux.after_bind_state_and_pc,
        top_level_aux.after_chunk_replay,
    );
    print_cumulative_and_delta(
        "after_inactive_side_lane_x_out",
        top_level_aux.after_chunk_replay,
        top_level_aux.after_inactive_side_lane_and_x_out,
    );
    print_cumulative_and_delta(
        "after_public_output_eq",
        top_level_aux.after_inactive_side_lane_and_x_out,
        top_level_aux.after_public_output_eq,
    );

    print_section("Chunk NIFS Verifier Aux");
    print_kv("measure.wall", format!("{chunk_replay_aux_ms:.3} ms"));
    print_cumulative_and_delta("after_state_cover", 0, chunk_replay_aux.after_state_cover);
    print_cumulative_and_delta(
        "after_public_chunk_meta",
        chunk_replay_aux.after_state_cover,
        chunk_replay_aux.after_chunk_meta,
    );
    print_cumulative_and_delta(
        "after_pi_ccs",
        chunk_replay_aux.after_chunk_meta,
        chunk_replay_aux.after_pi_ccs,
    );
    print_cumulative_and_delta(
        "after_synthetic_relation_io",
        chunk_replay_aux.after_pi_ccs,
        chunk_replay_aux.after_synthetic_relation_io,
    );
    print_cumulative_and_delta(
        "after_pi_rlc_parent_claim",
        chunk_replay_aux.after_synthetic_relation_io,
        chunk_replay_aux.after_pi_rlc_parent_claim,
    );
    print_cumulative_and_delta(
        "after_pi_rlc_rhos",
        chunk_replay_aux.after_pi_rlc_parent_claim,
        chunk_replay_aux.after_pi_rlc_rhos,
    );
    print_cumulative_and_delta(
        "after_pi_rlc_rho_mats",
        chunk_replay_aux.after_pi_rlc_rhos,
        chunk_replay_aux.after_pi_rlc_rho_mats,
    );
    print_cumulative_and_delta(
        "after_pi_rlc_public",
        chunk_replay_aux.after_pi_rlc_rho_mats,
        chunk_replay_aux.after_pi_rlc_public,
    );
    print_cumulative_and_delta(
        "after_pi_rlc",
        chunk_replay_aux.after_pi_rlc_public,
        chunk_replay_aux.after_pi_rlc,
    );
    print_cumulative_and_delta(
        "after_chunk_nifs_body",
        chunk_replay_aux.after_pi_rlc,
        chunk_replay_aux.after_chunk_body,
    );
    print_cumulative_and_delta(
        "after_chunk_nifs_verifier",
        chunk_replay_aux.after_chunk_body,
        chunk_replay_aux.after_chunk_replay,
    );

    print_section("Chunk NIFS Verifier Tail Aux");
    print_kv("measure.wall", format!("{chunk_replay_tail_aux_ms:.3} ms"));
    print_cumulative_and_delta(
        "after_state_out_projection_eq",
        chunk_replay_aux.after_chunk_body,
        chunk_replay_tail_aux.after_state_out_projection_eq,
    );
    print_cumulative_and_delta(
        "after_expected_digest",
        chunk_replay_tail_aux.after_state_out_projection_eq,
        chunk_replay_tail_aux.after_expected_digest,
    );
    print_cumulative_and_delta(
        "after_chunk_done",
        chunk_replay_tail_aux.after_expected_digest,
        chunk_replay_tail_aux.after_chunk_done,
    );
    print_cumulative_and_delta(
        "after_transcript_state_eq",
        chunk_replay_tail_aux.after_chunk_done,
        chunk_replay_tail_aux.after_transcript_state_eq,
    );
    print_cumulative_and_delta(
        "after_transcript_absorbed_eq",
        chunk_replay_tail_aux.after_transcript_state_eq,
        chunk_replay_tail_aux.after_transcript_absorbed_eq,
    );

    let tail_header_delta = chunk_replay_tail_digest
        .after_header
        .saturating_sub(chunk_replay_aux.after_chunk_body);
    let mut tail_claim_digest_deltas = Vec::with_capacity(chunk_replay_tail_digest.claim_after_digests.len());
    let mut prev = chunk_replay_tail_digest.after_header;
    let mut tail_total_claim_digest = 0usize;
    for idx in 0..chunk_replay_tail_digest.claim_after_digests.len() {
        let claim_digest_delta = chunk_replay_tail_digest.claim_after_digests[idx].saturating_sub(prev);
        tail_total_claim_digest += claim_digest_delta;
        tail_claim_digest_deltas.push((idx, claim_digest_delta));
        prev = chunk_replay_tail_digest.claim_after_digests[idx];
    }
    let tail_outer_hash_delta = chunk_replay_tail_digest
        .after_outer_hash
        .saturating_sub(prev);
    print_section("Chunk NIFS Verifier Tail Digest Aux");
    print_kv("measure.wall", format!("{chunk_replay_tail_digest_ms:.3} ms"));
    print_kv("header", tail_header_delta);
    print_kv("claim_digest_total", tail_total_claim_digest);
    print_kv("outer_hash", tail_outer_hash_delta);
    for (idx, claim_digest_delta) in &tail_claim_digest_deltas {
        print_kv(&format!("claim_{idx}.digest"), *claim_digest_delta);
    }

    print_section("Pi CCS Aux");
    print_kv("measure.wall", format!("{pi_ccs_aux_ms:.3} ms"));
    print_cumulative_and_delta("after_bind_header", 0, pi_ccs_aux.after_bind_header);
    print_cumulative_and_delta(
        "after_bind_me_inputs",
        pi_ccs_aux.after_bind_header,
        pi_ccs_aux.after_bind_me_inputs,
    );
    print_cumulative_and_delta(
        "after_sample_challenges",
        pi_ccs_aux.after_bind_me_inputs,
        pi_ccs_aux.after_sample_challenges,
    );
    print_cumulative_and_delta(
        "after_alloc_fresh_claims",
        pi_ccs_aux.after_sample_challenges,
        pi_ccs_aux.after_alloc_fresh_claims,
    );
    print_cumulative_and_delta(
        "after_fe_sumcheck",
        pi_ccs_aux.after_alloc_fresh_claims,
        pi_ccs_aux.after_fe_sumcheck,
    );
    print_cumulative_and_delta(
        "after_nc_sumcheck",
        pi_ccs_aux.after_fe_sumcheck,
        pi_ccs_aux.after_nc_sumcheck,
    );
    print_cumulative_and_delta(
        "after_fold_digest",
        pi_ccs_aux.after_nc_sumcheck,
        pi_ccs_aux.after_fold_digest,
    );
    print_cumulative_and_delta(
        "after_alloc_outputs",
        pi_ccs_aux.after_fold_digest,
        pi_ccs_aux.after_alloc_outputs,
    );
    print_cumulative_and_delta(
        "after_output_binding",
        pi_ccs_aux.after_alloc_outputs,
        pi_ccs_aux.after_output_binding,
    );
    print_cumulative_and_delta(
        "after_terminal_fe",
        pi_ccs_aux.after_output_binding,
        pi_ccs_aux.after_terminal_fe,
    );
    print_cumulative_and_delta(
        "after_terminal_nc",
        pi_ccs_aux.after_terminal_fe,
        pi_ccs_aux.after_terminal_nc,
    );

    let mut pi_ccs_bind_me_input_deltas = Vec::with_capacity(1 + pi_ccs_bind_me_inputs.after_claim_digests.len());
    let mut prev = pi_ccs_bind_me_inputs.after_bind_header;
    for (idx, end) in pi_ccs_bind_me_inputs.after_claim_digests.iter().enumerate() {
        pi_ccs_bind_me_input_deltas.push((format!("claim_digest_{idx}"), end.saturating_sub(prev)));
        prev = *end;
    }
    pi_ccs_bind_me_input_deltas.push((
        "bind_digests".to_string(),
        pi_ccs_bind_me_inputs
            .after_bind_digests
            .saturating_sub(prev),
    ));
    print_section("Pi CCS Bind ME Inputs Aux");
    print_kv("measure.wall", format!("{pi_ccs_bind_me_inputs_ms:.3} ms"));
    for (name, delta) in &pi_ccs_bind_me_input_deltas {
        print_kv(name, *delta);
    }

    print_section("Pi CCS Constraints");
    print_kv("measure.wall", format!("{pi_ccs_constraints_ms:.3} ms"));
    print_cumulative_and_delta("after_bind_header", 0, pi_ccs_constraints.after_bind_header);
    print_cumulative_and_delta(
        "after_bind_me_inputs",
        pi_ccs_constraints.after_bind_header,
        pi_ccs_constraints.after_bind_me_inputs,
    );
    print_cumulative_and_delta(
        "after_sample_challenges",
        pi_ccs_constraints.after_bind_me_inputs,
        pi_ccs_constraints.after_sample_challenges,
    );
    print_cumulative_and_delta(
        "after_alloc_fresh_claims",
        pi_ccs_constraints.after_sample_challenges,
        pi_ccs_constraints.after_alloc_fresh_claims,
    );
    print_cumulative_and_delta(
        "after_fe_sumcheck",
        pi_ccs_constraints.after_alloc_fresh_claims,
        pi_ccs_constraints.after_fe_sumcheck,
    );
    print_cumulative_and_delta(
        "after_nc_sumcheck",
        pi_ccs_constraints.after_fe_sumcheck,
        pi_ccs_constraints.after_nc_sumcheck,
    );
    print_cumulative_and_delta(
        "after_fold_digest",
        pi_ccs_constraints.after_nc_sumcheck,
        pi_ccs_constraints.after_fold_digest,
    );
    print_cumulative_and_delta(
        "after_alloc_outputs",
        pi_ccs_constraints.after_fold_digest,
        pi_ccs_constraints.after_alloc_outputs,
    );
    print_cumulative_and_delta(
        "after_output_binding",
        pi_ccs_constraints.after_alloc_outputs,
        pi_ccs_constraints.after_output_binding,
    );
    print_cumulative_and_delta(
        "after_terminal_fe",
        pi_ccs_constraints.after_output_binding,
        pi_ccs_constraints.after_terminal_fe,
    );
    print_cumulative_and_delta(
        "after_terminal_nc",
        pi_ccs_constraints.after_terminal_fe,
        pi_ccs_constraints.after_terminal_nc,
    );

    print_named_constraint_breakdown(
        "Pi CCS FE Sumcheck Constraints",
        pi_ccs_sumcheck_ms,
        &pi_ccs_sumcheck.fe_cover_round_lengths,
        &pi_ccs_sumcheck.fe_effective_round_lengths,
        &pi_ccs_sumcheck.fe_stages,
    );
    print_named_constraint_breakdown(
        "Pi CCS NC Sumcheck Constraints",
        pi_ccs_sumcheck_ms,
        &pi_ccs_sumcheck.nc_cover_round_lengths,
        &pi_ccs_sumcheck.nc_effective_round_lengths,
        &pi_ccs_sumcheck.nc_stages,
    );

    print_section("Pi RLC Public");
    print_kv("measure.wall", format!("{pi_rlc_public_ms:.3} ms"));
    print_kv("shared_point", pi_rlc_public.shared_point_constraints);
    print_kv("x", pi_rlc_public.x_constraints);
    print_kv("c", pi_rlc_public.c_constraints);
    print_kv("y_ring", pi_rlc_public.y_ring_constraints);
    print_kv("y_zcol", pi_rlc_public.y_zcol_constraints);
    print_kv("aux", pi_rlc_public.aux_constraints);
    print_kv("total", pi_rlc_public.total_constraints);
    print_section("Pi RLC Public Stages");
    print_kv("measure.wall", format!("{pi_rlc_public_stage_ms:.3} ms"));
    for stage in &pi_rlc_public_stage.stages {
        print_kv(&stage.name, stage.delta);
    }

    print_section("Payload Dimensions");
    print_kv("step_shape.state_in_claim_count", step_shape.state_in_claim_count);
    print_kv("step_shape.state_out_claim_count", step_shape.state_out_claim_count);
    print_kv("step_shape.fresh_claim_count", step_shape.fresh_claim_count);
    print_kv("step_shape.ccs_output_count", step_shape.ccs_output_count);
    print_kv("step_shape.child_count", step_shape.child_count);
    print_kv("cover_shape.ccs_output_count", cover_shape.ccs_output_count);
    print_kv("cover_shape.child_count", cover_shape.child_count);

    print_section("State In Claim Surface");
    print_kv("claim_count", first_relation.payload.state_in_claims.len());
    print_kv("claim.c_data_len", state_in_claim_shape.c_data_len);
    print_kv("claim.x_compact_len", first_state_in.m_in);
    print_kv("claim.r_len", state_in_claim_shape.r_len);
    print_kv("claim.y_ring_rows", state_in_claim_shape.y_ring_row_count);
    print_kv("claim.y_ring_row_lens", format!("{:?}", first_state_in_y_ring_row_lens));
    print_kv(
        "projection_hash_terms_per_claim",
        projection_digest_field_count(
            first_state_in.c.data.len(),
            first_state_in.m_in,
            first_state_in.r.len(),
            &first_state_in_y_ring_row_lens,
        ),
    );
    print_kv("projection_hash_terms_total", state_in_projection_fields_total);

    print_section("State Out Claim Surface");
    print_kv("claim_count", first_relation.payload.state_out_claims.len());
    print_kv("claim.c_data_len", state_out_claim_shape.c_data_len);
    print_kv("claim.x_compact_len", first_state_out.m_in);
    print_kv("claim.r_len", state_out_claim_shape.r_len);
    print_kv("claim.y_ring_rows", state_out_claim_shape.y_ring_row_count);
    print_kv(
        "claim.y_ring_row_lens",
        format!("{:?}", first_state_out_y_ring_row_lens),
    );
    print_kv(
        "projection_hash_terms_per_claim",
        projection_digest_field_count(
            first_state_out.c.data.len(),
            first_state_out.m_in,
            first_state_out.r.len(),
            &first_state_out_y_ring_row_lens,
        ),
    );
    print_kv("projection_hash_terms_total", state_out_projection_fields_total);
    print_kv("accumulator_phi_hash_terms", state_out_accumulator_phi_fields);

    print_section("Pi RLC Public Surface");
    print_kv("actual_child_count", actual_child_count);
    print_kv("padded_child_count", padded_child_count);
    print_kv("parent.c_data_len", pi_rlc_parent_shape.c_data_len);
    print_kv("parent.commitment_rows", D);
    print_kv(
        "parent.commitment_cols",
        usize::try_from(pi_rlc_parent_shape.c_data_len).expect("commitment len") / D,
    );
    print_kv("parent.x_compact_len", first_relation.payload.pi_rlc.parent.m_in);
    print_kv("parent.r_len", pi_rlc_parent_shape.r_len);
    print_kv("parent.y_ring_rows", pi_rlc_parent_shape.y_ring_row_count);
    print_kv("parent.y_ring_row_lens", format!("{:?}", pi_rlc_parent_y_ring_row_lens));
    print_kv("parent.y_zcol_len", pi_rlc_parent_shape.y_zcol_len);
    print_kv(
        "dense_c_scalars_across_children",
        padded_child_count * usize::try_from(pi_rlc_parent_shape.c_data_len).expect("parent c_data len"),
    );
    print_kv(
        "dense_y_ring_k_scalars_per_claim",
        first_relation
            .payload
            .pi_rlc
            .parent
            .y_ring
            .iter()
            .map(|row| row.len())
            .sum::<usize>(),
    );
    let fresh_child_count = usize::try_from(step_shape.fresh_claim_count).expect("fresh child count");
    print_pi_rlc_public_child_families(
        &first_relation,
        fresh_child_count,
        actual_child_count,
        &pi_rlc_parent_shape,
    );
    print_backend_relation_commitment_sparsity(&backend_relations);

    let rerun_summary = measure_fast_summary_perf(&input);
    print_probe_work_units("Fast Key Per-Opcode Units (Rerun)", work_units);
    print_key_per_fold_summary(
        "Fast Key Per-Fold Summary (Rerun)",
        &rerun_summary,
        work_units.chunk_fold_step_count,
    );
    print_key_per_opcode_summary("Fast Key Per-Opcode Summary (Rerun)", &rerun_summary, opcode_count);
    print_per_opcode_components("Fast Per-Opcode Components (Rerun)", &rerun_summary, opcode_count);
    print_section("Full-Only Extra Per-Opcode");
    print_kv("live_shape", format_ms_per_opcode(live_shape_ms, opcode_count));
    print_kv("shape_synth", format_ms_per_opcode(synth_ms, opcode_count));
}
