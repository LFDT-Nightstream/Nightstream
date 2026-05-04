use super::*;

pub(super) fn print_spartan(
    params: &NeoParams,
    ccs: &CcsStructure<F>,
    packaged: &PackagedProof,
    _final_carry: &Carry,
    _steps: &[StepInput],
    superneo_ivc: &SuperNeoIvcBuild,
    log: &AjtaiSModule,
) -> AppResult<Option<DirectCcsRecursiveIvcSnarkPerf>> {
    println!("== Spartan2 direct CCS F' terminal compression ==");
    println!("generic direct CCS/R1CS path; no RV32IM VM is used");
    println!(
        "target: direct-CCS F' terminal compression; compact prior F' source is diagnostic until authority rows are enabled"
    );
    println!(
        "public binding: final Construction-2 x_out image plus committed F' boundary; predecessor state stays private advice; terminal semantic CE projections stay private over {} claim(s)",
        packaged.statement.final_main_claims.len()
    );
    println!(
        "native direct carrier: canonical-zero-seeded chunks={}",
        superneo_ivc.relations.len()
    );
    println!(
        "authority status: single-step compression is enabled; multi-step recursive compression is refused until the compact low-norm enc(F') relation proves Poseidon2 linkage and NIFS.V"
    );
    println!(
        "Construction-2 guardrail: terminal committed/source-image exports are not accepted as recursive F' authority"
    );
    println!(
        "encoded F' authority boundary: no public caller-supplied encoded-F' authority is accepted until the crate owns a real low-norm enc(F') builder"
    );
    println!(
        "generic R1CS low-norm lowering: available for frontend field lanes; F' NIFS.V authority still requires verifier-body constraints"
    );
    println!(
        "recursive carried u_i: compact verifier-shaped F' image; folded F' CE consistency is terminal-only when a real enc(F') chain is present"
    );
    println!(
        "recursive F' guardrail: Construction-2 fold excludes embedded final-CE consistency; terminal CE checks are final-boundary only"
    );
    println!(
        "legacy cold-start packaged proof digest (diagnostic only): {:02x?}",
        packaged.proof.proof_digest
    );
    let program = DirectCcsProgram::new_with_public_input_len(params, ccs, FIB_STEP_TRACE_LEN)?;
    let mut recursive_state = DirectCcsRecursiveIvcState::new_with_canonical_zero_carry(program)?;
    for relation in &superneo_ivc.relations {
        recursive_state = recursive_state.append_relation(relation, log, ajtai_mixers())?;
    }
    let direct_state = recursive_state.direct_state();
    if direct_state.final_state().carry.claims != superneo_ivc.final_state.carry.claims
        || direct_state.final_state().carry.witnesses != superneo_ivc.final_state.carry.witnesses
    {
        return Err(invalid_input(
            "direct CCS IVC final state does not match canonical-zero SuperNeo final carry",
        ));
    }
    let latest = direct_state.latest_relation_and_advice()?;
    let recursive_summary = recursive_state.summary_with_verifier_body_measurement();
    println!(
        "latest F' summary: chunk_index={}, fresh_CCS={}, incoming_CE={}, Pi_CCS_outputs={}, final_CE={}, x_in={:02x?}, x_out={:02x?}",
        latest.chunk_index,
        latest.fresh_claims,
        latest.incoming_ce_claims,
        latest.output_ce_claims,
        latest.final_ce_claims,
        &latest.construction2_x_in.bytes()[..4],
        &latest.construction2_x_out.bytes()[..4]
    );
    println!(
        "recursive authority summary: semantic_chunks={}, folded_prior_f_prime_r2_steps={}, carried_semantic_CE={}, carried_f_prime_CE={}, standalone_proof_authority_ready={}",
        recursive_summary.semantic_chunks,
        recursive_summary.folded_f_prime_r2_steps,
        recursive_summary.carried_semantic_ce_claims,
        recursive_summary.carried_f_prime_ce_claims,
        recursive_summary.standalone_proof_authority_ready
    );
    if recursive_summary.f_prime_verifier_body_measured {
        println!(
            "verifier-shaped F' body: public_inputs={}, constraints={}, exact_low_norm_fallback_cap={}, latest_nifs_total={}, construction2_fold={}, public_link={}, chunk_done={}, terminal_final_ce={}",
            recursive_summary.f_prime_verifier_body_public_inputs,
            recursive_summary.f_prime_verifier_body_constraints,
            recursive_summary.f_prime_exact_encoder_row_cap,
            recursive_summary.f_prime_verifier_body_nifs_constraints,
            recursive_summary.f_prime_verifier_body_construction2_fold_constraints,
            recursive_summary.f_prime_verifier_body_public_link_constraints,
            recursive_summary.f_prime_verifier_body_chunk_done_constraints,
            recursive_summary.f_prime_verifier_body_final_ce_relation_constraints
        );
    } else if recursive_summary.f_prime_verifier_body_measure_skipped {
        println!(
            "verifier-shaped F' body: not measured by default for this large CCS shape (rows={}, cols={}, matrices={}); recursive enc(F') still excludes terminal final CE by construction",
            ccs.n,
            ccs.m,
            ccs.t()
        );
    } else {
        println!("verifier-shaped F' body: unavailable; latest native F' advice was not buildable");
    }
    println!(
        "verifier-shaped F' guardrail: terminal_final_ce=0 means recursive enc(F') excludes final semantic CE consistency"
    );
    let required_prior_f_prime_r2_steps = recursive_summary.semantic_chunks.saturating_sub(1);
    let compact_f_prime_digest = recursive_summary
        .compact_f_prime_image_digest
        .map(|digest| format!("{:02x?}", &digest[..4]))
        .unwrap_or_else(|| "none".to_owned());
    let low_norm_source_digest = recursive_summary
        .low_norm_f_prime_source_digest
        .map(|digest| format!("{:02x?}", &digest[..4]))
        .unwrap_or_else(|| "none".to_owned());
    println!(
        "encoded F' preflight: native_evaluator_available={}, required_for_prior_steps={}, low_norm_source_available={}, low_norm_source_len={}, low_norm_source_digest={}, source_r1cs_constraints={}, source_r1cs_variables={}, source_r1cs_nnz={}, low_norm_relation_available={}, compact_image_digest={}, blocker={}",
        recursive_summary.native_f_prime_evaluator_available,
        recursive_summary.f_prime_encoder_required,
        recursive_summary.low_norm_f_prime_source_available,
        recursive_summary.low_norm_f_prime_source_len,
        low_norm_source_digest,
        recursive_summary.low_norm_f_prime_source_r1cs_constraints,
        recursive_summary.low_norm_f_prime_source_r1cs_variables,
        recursive_summary.low_norm_f_prime_source_r1cs_nnz,
        recursive_summary.f_prime_encoder_available,
        compact_f_prime_digest,
        recursive_summary.f_prime_encoder_blocker.unwrap_or("none")
    );
    println!(
        "encoded F' source R1CS columns: public_inputs={}, private_source_bits={}, counter_carry_bits={}, total_variables={}",
        recursive_summary.low_norm_f_prime_source_public_inputs,
        recursive_summary.low_norm_f_prime_source_private_bits,
        recursive_summary.low_norm_f_prime_source_counter_carry_bits,
        recursive_summary.low_norm_f_prime_source_r1cs_variables
    );
    println!(
        "encoded F' source packed inputs: digests={}, u64_words={}, encoded_public_inputs={}, canonical_field_lanes={}, construction2_commitment_data_fields_in_source={}",
        recursive_summary.low_norm_f_prime_source_digest_count,
        recursive_summary.low_norm_f_prime_source_u64_count,
        recursive_summary.low_norm_f_prime_source_encoded_public_input_count,
        recursive_summary.low_norm_f_prime_source_field_lane_count,
        recursive_summary.low_norm_f_prime_source_construction2_commitment_fields
    );
    println!(
        "encoded F' source SuperNeo handles: latest_public_chunk_digest_fields=4, latest_fold_digest=1, latest_chunk_relation_digest=1"
    );
    if let Some(shape) = recursive_summary.low_norm_f_prime_nifs_payload_shape {
        println!(
            "encoded F' NIFS.V payload metadata: chunk_index={}, fresh_CCS={}, incoming_CE={}, Pi_CCS_outputs={}, final_CE={}, FE_rounds={} (messages={}), NC_rounds={} (messages={}), transcript_absorbed={} -> {}; encoded_authority_nifs_rows={}",
            shape.chunk_index,
            shape.fresh_claims,
            shape.incoming_ce_claims,
            shape.pi_ccs_outputs,
            shape.final_ce_claims,
            shape.fe_sumcheck_rounds,
            shape.fe_sumcheck_messages,
            shape.nc_sumcheck_rounds,
            shape.nc_sumcheck_messages,
            shape.transcript_absorbed_in,
            shape.transcript_absorbed_out,
            recursive_summary.low_norm_f_prime_source_nifs_v_verifier_constraints
        );
    }
    println!(
        "encoded F' source R1CS breakdown: shell_constraints={} [bitness={}, public_x_out_link={}, construction2_boundary_link={}, construction2_instance_digest_link={}, commitment_shape={}, structural_counters={} (fixed_arity={}, carry_bit_columns={}), canonical_field_lanes={} (aux_bits={})], digest_binding_constraints={} [poseidon_digest_recompute={}], proof_authority_constraints={} [nifs_v_verifier={}]",
        recursive_summary.low_norm_f_prime_source_shell_constraints,
        recursive_summary.low_norm_f_prime_source_bit_constraints,
        recursive_summary.low_norm_f_prime_source_x_out_link_constraints,
        recursive_summary.low_norm_f_prime_source_construction2_boundary_link_constraints,
        recursive_summary.low_norm_f_prime_source_construction2_instance_digest_link_constraints,
        recursive_summary.low_norm_f_prime_source_construction2_commitment_shape_constraints,
        recursive_summary.low_norm_f_prime_source_structural_counter_constraints,
        recursive_summary.low_norm_f_prime_source_structural_fixed_arity_constraints,
        recursive_summary.low_norm_f_prime_source_structural_counter_carry_bit_constraints,
        recursive_summary.low_norm_f_prime_source_canonical_field_lane_constraints,
        recursive_summary.low_norm_f_prime_source_canonical_field_lane_aux_bits,
        recursive_summary.low_norm_f_prime_source_poseidon_digest_recomputation_constraints,
        recursive_summary.low_norm_f_prime_source_poseidon_digest_recomputation_constraints,
        recursive_summary.low_norm_f_prime_source_authority_constraints,
        recursive_summary.low_norm_f_prime_source_nifs_v_verifier_constraints
    );
    println!("== direct F' authority layers ==");
    println!("layer                          | status              | size/claims | proof role");
    println!(
        "recursive_f_prime_relation     | {} | required={}, folded={} prior R2 step(s) | folds prior committed F' authority",
        if recursive_summary.f_prime_encoder_required && !recursive_summary.f_prime_encoder_available {
            "missing low-norm enc(F')"
        } else if recursive_summary.folded_f_prime_r2_steps == 0 {
            "base/not required"
        } else {
            "present"
        },
        required_prior_f_prime_r2_steps,
        recursive_summary.folded_f_prime_r2_steps
    );
    println!(
        "terminal_current_f_prime       | {} | 1 latest chunk | final committed F' check",
        if recursive_summary.standalone_proof_authority_ready {
            "enabled"
        } else {
            "blocked"
        }
    );
    println!(
        "folded_f_prime_accumulator_ce  | {} | {} CE claim(s) | Construction-2 induction authority",
        if recursive_summary.carried_f_prime_ce_claims == 0 {
            "default/missing"
        } else {
            "carried"
        },
        recursive_summary.carried_f_prime_ce_claims
    );
    println!(
        "terminal_private_semantic_ce   | private post-DEC   | {} CE claim(s) | final semantic accumulator consistency",
        recursive_summary.carried_semantic_ce_claims
    );
    let mut trace = |message: &str| println!("  {message}");
    let (recursive_snark, vk, mut recursive_perf) =
        match recursive_state.compress_recursive_snark_with_trace(&mut trace) {
            Ok(proved) => proved,
            Err(err) => {
                println!("direct CCS recursive compression: refused");
                println!("reason: {err}");
                println!("spartan proof: not produced for this multi-step direct CCS run");
                println!();
                return Ok(None);
            }
        };
    let verify_started = Instant::now();
    recursive_snark.verify(&vk, recursive_snark.public_image())?;
    recursive_perf.total_verify_ms = verify_started.elapsed().as_secs_f64() * 1_000.0;
    let snark = recursive_snark.terminal_snark();
    let perf = &recursive_perf.terminal;
    let image = snark.public_image();
    let recursive_image = recursive_snark.public_image();
    let statement = image.statement();
    println!(
        "direct CCS statement: digest={:02x?}, relation={:?}, vk_fs={:02x?}, pc={}, final_chunks={}, final_steps={}, initial_boundary={:02x?}, current_boundary={:02x?}, semantic_accumulator={:02x?}, construction2_accumulator={:02x?}, trace_boundary={:02x?}",
        statement.expected_digest(),
        statement.mat_digest,
        &statement.vk_fs_digest[..4],
        statement.pc,
        statement.chunk_count_out,
        statement.step_count_out,
        &statement.initial_boundary_digest[..4],
        &statement.current_boundary_digest[..4],
        &statement.accumulator_out_digest[..4],
        &statement.construction2_accumulator_digest[..4],
        &statement.public_trace_out_digest[..4]
    );
    println!(
        "Construction-2 public image: digest={:02x?}, snark_vk_digest={:02x?}, f_prime_x_out={:02x?}",
        image.expected_digest(),
        vk.expected_digest()?,
        &image.x_out.bytes()[..4]
    );
    println!(
        "recursive public image: digest={:02x?}, chunks={}, steps={}, semantic_accumulator_digest={:02x?}, f_prime_accumulator_digest={:02x?}, folded_prior_f_prime_r2_steps={}",
        recursive_image.expected_digest()?,
        recursive_image.proven_chunk_count,
        recursive_image.proven_step_count,
        &recursive_image.proven_accumulator_digest[..4],
        &recursive_image.proven_f_prime_accumulator_digest[..4],
        recursive_image.folded_f_prime_r2_steps
    );
    println!(
        "setup/keygen ms (not counted in final summary): terminal_f_prime={:.3}, standalone_f_prime_chain={:.3}, f_prime_final_ce={:.3}, total={:.3}",
        perf.setup_ms,
        recursive_perf.f_prime_chain_setup_ms,
        recursive_perf.f_prime_final_ce_setup_ms,
        perf.setup_ms + recursive_perf.f_prime_chain_setup_ms + recursive_perf.f_prime_final_ce_setup_ms
    );
    println!(
        "prove ms: terminal_prep={:.3}, terminal_snark={:.3}, terminal_encode={:.3}, standalone_f_prime_chain={:.3}, f_prime_final_ce={:.3}, total={:.3}",
        perf.prep_ms,
        perf.prove_ms,
        perf.encode_ms,
        recursive_perf.f_prime_chain_prove_ms,
        recursive_perf.f_prime_final_ce_prove_ms,
        recursive_perf.total_prove_ms
    );
    println!(
        "verify ms: final_wrapper={:.3} (component self-checks skipped during prove)",
        recursive_perf.total_verify_ms
    );
    println!(
        "direct CCS F' R1CS sizes [cons, shared, precommitted, rest, padded_cons, padded_shared, padded_precommitted, padded_rest, public, challenges]: {:?}",
        perf.r1cs_sizes
    );
    println!(
        "constraint attribution: public_inputs={}, terminal_chunks_synthesized={}, nifs_chunk_constraints_first4={:?}, construction2_fold={}, construction2_fold_embedded_final_ce={}, public_link={}, chunk_done={}, terminal_f_prime={}, final_ce_relation={}, inline_final_ce={}",
        perf.public_inputs,
        perf.chunk_count,
        perf.chunk_constraints_first4,
        perf.construction2_fold_constraints,
        perf.construction2_fold_final_ce_consistency_constraints,
        perf.public_link_constraints,
        perf.chunk_done_constraints,
        perf.terminal_f_prime_constraints,
        perf.final_ce_relation_constraints,
        perf.final_ce_relation_constraints
    );
    println!(
        "terminal Construction-2 boundary: committed_width={}, source_values={}, commitment_words={}",
        perf.terminal_committed_width, perf.terminal_source_values, perf.terminal_commitment_words
    );
    println!(
        "terminal source encodings: bit={}, u32={}, u64={}, unclassified={}",
        perf.terminal_source_bit_values,
        perf.terminal_source_u32_values,
        perf.terminal_source_u64_values,
        perf.terminal_unclassified_private_values
    );
    let tb = perf.terminal_committed_breakdown;
    let shape_row = |stage: &str, rows: usize, public_cols: usize, aux_cols: usize, primitive: &str| {
        println!(
            "{:<45} | {:>11} | {:>11} | {:>11} | {}",
            stage, rows, public_cols, aux_cols, primitive
        );
    };
    let stage_sum = tb.public_input_alloc
        + tb.boundary_input_alloc
        + tb.packed_witness_alloc
        + tb.public_boundary.total
        + tb.public_commitment_shape
        + tb.committed_image.total
        + tb.terminal_body_with_sources
        + tb.terminal_ajtai_commitment;
    println!("== terminal committed-step constraint breakdown ==");
    println!("R1CS vocabulary: rows=constraints; public_cols=public input columns; aux_cols=private/intermediate witness columns");
    println!(
        "{:<45} | {:>11} | {:>11} | {:>11} | behind it",
        "stage", "rows", "public_cols", "aux_cols"
    );
    shape_row(
        "public input allocation",
        tb.public_input_alloc_shape.rows,
        tb.public_input_alloc_shape.public_cols,
        tb.public_input_alloc_shape.aux_cols,
        "terminal public statement fields",
    );
    shape_row(
        "boundary input allocation",
        tb.boundary_input_alloc_shape.rows,
        tb.boundary_input_alloc_shape.public_cols,
        tb.boundary_input_alloc_shape.aux_cols,
        "public Construction-2 boundary u_i=(C_i,x_i)",
    );
    shape_row(
        "packed witness allocation",
        tb.packed_witness_alloc_shape.rows,
        tb.packed_witness_alloc_shape.public_cols,
        tb.packed_witness_alloc_shape.aux_cols,
        "private packed low-norm R2 source image",
    );
    shape_row(
        "public boundary checks",
        tb.public_boundary.total_shape.rows,
        tb.public_boundary.total_shape.public_cols,
        tb.public_boundary.total_shape.aux_cols,
        "Poseidon2 boundary hashing plus x_i bit/limb checks",
    );
    shape_row(
        "  boundary digest checks",
        tb.public_boundary.digest_checks_shape.rows,
        tb.public_boundary.digest_checks_shape.public_cols,
        tb.public_boundary.digest_checks_shape.aux_cols,
        "Poseidon2(C_i) and Poseidon2(C_i,x_i)",
    );
    shape_row(
        "  x_i bit checks",
        tb.public_boundary.x_i_bit_checks_shape.rows,
        tb.public_boundary.x_i_bit_checks_shape.public_cols,
        tb.public_boundary.x_i_bit_checks_shape.aux_cols,
        "boolean constraints for 256 public x_i bits",
    );
    shape_row(
        "  x_i limb links",
        tb.public_boundary.x_i_limb_links_shape.rows,
        tb.public_boundary.x_i_limb_links_shape.public_cols,
        tb.public_boundary.x_i_limb_links_shape.aux_cols,
        "pack 256 x_i bits into four field limbs",
    );
    shape_row(
        "public commitment shape",
        tb.public_commitment_shape_shape.rows,
        tb.public_commitment_shape_shape.public_cols,
        tb.public_commitment_shape_shape.aux_cols,
        "check public Ajtai commitment dimensions",
    );
    shape_row(
        "committed image checks",
        tb.committed_image.total_shape.rows,
        tb.committed_image.total_shape.public_cols,
        tb.committed_image.total_shape.aux_cols,
        "link public bits, force constant one, bit-check source image",
    );
    shape_row(
        "  public z links",
        tb.committed_image.public_z_links_shape.rows,
        tb.committed_image.public_z_links_shape.public_cols,
        tb.committed_image.public_z_links_shape.aux_cols,
        "copy public x_i bits into committed source columns",
    );
    shape_row(
        "  constant-one link",
        tb.committed_image.constant_one_link_shape.rows,
        tb.committed_image.constant_one_link_shape.public_cols,
        tb.committed_image.constant_one_link_shape.aux_cols,
        "force the committed constant column to one",
    );
    shape_row(
        "  low-norm bit checks",
        tb.committed_image.low_norm_bit_checks_shape.rows,
        tb.committed_image.low_norm_bit_checks_shape.public_cols,
        tb.committed_image.low_norm_bit_checks_shape.aux_cols,
        "one boolean row per committed source column",
    );
    shape_row(
        "  padding zero checks",
        tb.committed_image.padding_zero_checks_shape.rows,
        tb.committed_image.padding_zero_checks_shape.public_cols,
        tb.committed_image.padding_zero_checks_shape.aux_cols,
        "force unused packed source columns to zero",
    );
    shape_row(
        "terminal F' body with source links",
        tb.terminal_body_shape.rows,
        tb.terminal_body_shape.public_cols,
        tb.terminal_body_shape.aux_cols,
        "latest NIFS.V plus Construction-2 folded F' accumulator",
    );
    println!(
        "{:<45} | {:>11} | {:>11} | {:>11} | equality rows inside terminal body",
        "  committed source links", tb.terminal_body_source_links, "-", "-"
    );
    println!(
        "{:<45} | {:>11} | {:>11} | {:>11} | terminal body after subtracting source-link rows",
        "  terminal F' body excluding source links", tb.terminal_body_without_source_links, "-", "-"
    );
    shape_row(
        "terminal Ajtai commitment",
        tb.terminal_ajtai_commitment_shape.rows,
        tb.terminal_ajtai_commitment_shape.public_cols,
        tb.terminal_ajtai_commitment_shape.aux_cols,
        "linear fixed-A Ajtai opening check",
    );
    shape_row(
        "measured total",
        tb.total_shape.rows,
        tb.total_shape.public_cols,
        tb.total_shape.aux_cols,
        "full terminal committed-step R1CS shape",
    );
    println!("stage sum                                 | {:>11}", stage_sum);
    println!("measured total                            | {:>11}", tb.total);
    println!(
        "unattributed                              | {:>11}",
        tb.total.saturating_sub(stage_sum)
    );
    println!(
        "terminal body inner attribution: latest_nifs={:?}, construction2_fold={}, construction2_fold_embedded_final_ce={}, public_link={}, chunk_done={}, final_ce_relation={}",
        perf.chunk_constraints_by_chunk,
        perf.construction2_fold_constraints,
        perf.construction2_fold_final_ce_consistency_constraints,
        perf.public_link_constraints,
        perf.chunk_done_constraints,
        perf.final_ce_relation_constraints
    );
    println!();
    println!("terminal_nifs_chunk_constraints={:?}", perf.chunk_constraints_by_chunk);
    println!(
        "final_ce_private_checks=[projection_digest=0, digest_match=0, relation={}]",
        perf.final_ce_relation_constraints
    );
    let ce = perf.final_ce_relation_breakdown;
    println!(
        "final_ce_relation_by_component=[A*z={}, x_projection={}, y_eval={}, norm={}, total={}]",
        ce.commitment,
        ce.x_projection,
        ce.y_eval,
        ce.norm,
        ce.total()
    );
    let f_prime_final_ce_mode = if recursive_image.folded_f_prime_r2_steps == 0 {
        "base_default_accumulator_digest"
    } else {
        "inline_terminal_f_prime_construction2_fold"
    };
    println!(
        "folded F' accumulator authority: mode={}, claims={}, public_inputs={}, constraints={}, digest={}, digest_match={}, relation={}, proof_bytes={}",
        f_prime_final_ce_mode,
        recursive_perf.f_prime_final_ce_claims,
        recursive_perf.f_prime_final_ce_public_inputs,
        recursive_perf.f_prime_final_ce_constraints,
        recursive_perf.f_prime_final_ce_digest_constraints,
        recursive_perf.f_prime_final_ce_digest_match_constraints,
        recursive_perf.f_prime_final_ce_relation_constraints,
        recursive_perf.f_prime_final_ce_proof_bytes
    );
    println!(
        "standalone folded F' chain proof: constraints={}, setup_ms={:.3}, prove_ms={:.3}, verify_ms={:.3}, proof_bytes={} (expected zero on the current guarded path)",
        recursive_perf.f_prime_chain_constraints,
        recursive_perf.f_prime_chain_setup_ms,
        recursive_perf.f_prime_chain_prove_ms,
        recursive_perf.f_prime_chain_verify_ms,
        recursive_perf.f_prime_chain_proof_bytes
    );
    println!(
        "R1CS nonzero matrix entries (not constraints): direct={} avg_nnz_per_constraint={:.1}",
        perf.r1cs_nnz,
        perf.r1cs_nnz as f64 / perf.r1cs_sizes[0].max(1) as f64
    );
    println!(
        "proof bytes: final_serialized={}, snark_data={}, wrapper_overhead={}",
        recursive_perf.total_proof_bytes,
        perf.snark_bytes + recursive_perf.f_prime_chain_proof_bytes + recursive_perf.f_prime_final_ce_proof_bytes,
        recursive_perf.total_proof_bytes.saturating_sub(
            perf.snark_bytes + recursive_perf.f_prime_chain_proof_bytes + recursive_perf.f_prime_final_ce_proof_bytes
        )
    );
    println!("verify: ok");
    println!();
    Ok(Some(recursive_perf))
}

pub(super) fn print_superneo_ivc_carrier(build: &SuperNeoIvcBuild, initial_carry_len: usize) -> AppResult<()> {
    if build
        .relations
        .first()
        .map(|relation| relation.state_in.carry.claims.len())
        != Some(initial_carry_len)
    {
        return Err(invalid_input(
            "generic direct SuperNeo carrier did not start from the canonical zero carry",
        ));
    }
    println!("== generic SuperNeo IVC/NIFS.V carrier ==");
    println!("status: built and verified natively");
    println!("relations: {}", build.relations.len());
    println!(
        "initial state: carried_CE_claims={} (canonical zero CE(b)^k)",
        initial_carry_len
    );
    println!(
        "final state: chunks={}, steps={}, carried_CE_claims={}, transcript_absorbed={}",
        build.final_state.chunk_count,
        build.final_state.step_count,
        build.final_state.carry.claims.len(),
        build.final_state.transcript.absorbed
    );
    println!(
        "timing (not counted in final summary): cache_build={:.3} ms, total={:.3} ms",
        build.cache_build_ms, build.total_ms
    );
    println!(
        "hash boundary: no final CE digest is used here; Construction-2 public image is carried natively, and multi-step proof authority still requires low-norm enc(F')"
    );
    println!();
    Ok(())
}

pub(super) fn print_final_summary(prove_perf: &RunProvePerf, spartan_perf: Option<&DirectCcsRecursiveIvcSnarkPerf>) {
    let chunk_folds = prove_perf.chunk_count();
    let fresh_steps = prove_perf.fresh_steps();
    let ms_per_chunk_fold = prove_perf.total_ms / chunk_folds.max(1) as f64;
    let ms_per_fresh_step = prove_perf.total_ms / fresh_steps.max(1) as f64;

    println!("== final summary ==");
    println!("proving (before spartan): {:.3} ms", prove_perf.total_ms);
    println!(
        "  number of folds: {} SuperNeo chunk fold(s) over {} fresh CCS step(s)",
        chunk_folds, fresh_steps
    );
    println!(
        "  time per fold: {:.3} ms/chunk fold ({:.3} ms/fresh CCS step)",
        ms_per_chunk_fold, ms_per_fresh_step
    );
    match spartan_perf {
        Some(spartan_perf) => {
            let terminal = &spartan_perf.terminal;
            println!("proving (spartan): {:.3} ms", spartan_perf.total_prove_ms);
            println!(
                "setup/keygen (not counted): {:.3} ms",
                terminal.setup_ms + spartan_perf.f_prime_chain_setup_ms + spartan_perf.f_prime_final_ce_setup_ms
            );
            println!(
                "proving (total): {:.3} ms",
                prove_perf.total_ms + spartan_perf.total_prove_ms
            );
            println!("verifying (final proof): {:.3} ms", spartan_perf.total_verify_ms);
            let total_constraints = terminal.r1cs_sizes[0]
                + spartan_perf.f_prime_chain_constraints
                + spartan_perf.f_prime_final_ce_constraints;
            let total_padded = terminal.r1cs_sizes[4]
                + padded_constraints(spartan_perf.f_prime_chain_constraints)
                + padded_constraints(spartan_perf.f_prime_final_ce_constraints);
            println!(
                "constraints passed to Spartan2: {} backend R1CS constraints across single-step terminal direct CCS F' plus any enabled folded-F' authority (padded approximately to {})",
                total_constraints,
                total_padded
            );
            println!(
                "size final proof: {} bytes (snark_data={}, wrapper_overhead={})",
                spartan_perf.total_proof_bytes,
                terminal.snark_bytes
                    + spartan_perf.f_prime_chain_proof_bytes
                    + spartan_perf.f_prime_final_ce_proof_bytes,
                spartan_perf.total_proof_bytes.saturating_sub(
                    terminal.snark_bytes
                        + spartan_perf.f_prime_chain_proof_bytes
                        + spartan_perf.f_prime_final_ce_proof_bytes
                )
            );
        }
        None => {
            println!("proving (spartan): not run");
            println!("setup/keygen (not counted): not run");
            println!("proving (total): {:.3} ms", prove_perf.total_ms);
            println!("verifying (final proof): not run");
            println!("constraints passed to Spartan2: not available; Spartan terminal compression did not run");
            println!("size final proof: not available");
        }
    }
    println!();
}
