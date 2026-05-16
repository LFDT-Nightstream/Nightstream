use super::*;

#[test]
fn direct_ccs_append_api_matches_native_superneo_carry() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let steps = vec![step(&log, "direct_0", 1, 2, 3), step(&log, "direct_1", 2, 3, 5)];

    let native = build_superneo_ivc_relations_with_initial_carry_accumulator_handle_perf(
        FoldSchedule::RowsPerChunk(1),
        &params,
        &ccs,
        steps.clone(),
        Carry::default(),
        &log,
        ajtai_mixers(),
    )
    .expect("native SuperNeo IVC build");

    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let mut direct = DirectCcsIvcState::new(program).expect("direct CCS state");
    let base_boundary = direct.construction2_public_boundary();
    assert_eq!(base_boundary.commitment_kappa, params.kappa as u64);
    assert_eq!(base_boundary.commitment_data.len(), D * params.kappa as usize);
    assert!(
        base_boundary
            .commitment_data
            .iter()
            .all(|value| *value == F::ZERO),
        "direct CCS base state must carry full-shape canonical Construction-2 u_perp"
    );
    for step in steps {
        direct = direct
            .append_step(DirectCcsStep::new(step), &log, ajtai_mixers())
            .expect("append direct CCS step");
    }

    assert_eq!(direct.final_state().chunk_count, 2);
    assert_eq!(direct.final_state().step_count, 2);
    assert_eq!(direct.final_state().carry.claims, native.final_state.carry.claims);
    assert_eq!(direct.final_state().carry.witnesses, native.final_state.carry.witnesses);
    let construction2_boundary = direct.construction2_public_boundary();
    let latest = direct
        .latest_relation_and_advice()
        .expect("latest direct Construction-2 summary");
    assert_eq!(construction2_boundary.x_i, latest.construction2_x_out);
    assert!(construction2_boundary.commitment_kappa > 1);
    assert!(construction2_boundary.has_canonical_commitment_shape());
    assert_eq!(
        construction2_boundary.commitment_digest,
        construction2_boundary.expected_commitment_digest()
    );
    assert_eq!(
        construction2_boundary.fresh_instance_digest,
        construction2_boundary.expected_fresh_instance_digest()
    );
}

#[test]
fn direct_ccs_lifecycle_prove_extend_and_verify_use_same_append_flow() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let mixers = ajtai_mixers();
    let preprocessing = preprocess_direct_ccs(
        program,
        log.clone(),
        DirectCcsCommitmentOps::new(mixers.mix_rhos_commits, mixers.combine_b_pows),
    );
    let first = DirectCcsStep::new(step(&log, "lifecycle_0", 1, 2, 3));

    let batched = prove_direct_ccs(&preprocessing, vec![first.clone()]).expect("prove direct CCS");
    verify_direct_ccs(&preprocessing, &batched).expect("verify replayable direct CCS proof");

    let incremental = prove_direct_ccs(&preprocessing, Vec::<DirectCcsStep>::new()).expect("start empty proof");
    let incremental = extend_direct_ccs(&preprocessing, incremental, first).expect("extend first step");

    assert_eq!(batched.summary(), incremental.summary());
}

#[test]
fn direct_ccs_program_builds_canonical_zero_carry_shape() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let _log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");

    let carry = program
        .canonical_zero_carry()
        .expect("canonical direct zero carry");

    assert_eq!(carry.claims.len(), params.k_rho as usize);
    assert_eq!(carry.witnesses.len(), params.k_rho as usize);
    let first = &carry.claims[0];
    assert_eq!(first.c.d, D);
    assert_eq!(first.c.kappa, params.kappa as usize);
    assert_eq!(first.X.rows(), D);
    assert_eq!(first.X.cols(), 3);
    assert_eq!(first.m_in, 3);
    assert_eq!(first.r.len(), 1);
    assert_eq!(first.s_col.len(), 6);
    assert_eq!(first.y_ring.len(), 1);
    assert_eq!(first.y_ring[0].len(), 64);
    assert_eq!(first.ct.len(), 1);
    assert!(first.aux_openings.is_empty());
    assert!(first.c_step_coords.is_empty());
    assert_eq!(carry.witnesses[0].rows(), D);
    assert_eq!(carry.witnesses[0].cols(), ccs.m.div_ceil(D));
}

#[test]
fn direct_ccs_canonical_zero_seed_first_append_has_steady_accumulator_arity() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let direct = DirectCcsIvcState::start(program).expect("canonical zero-seeded state");
    let base_boundary = direct.construction2_public_boundary();
    assert_eq!(base_boundary.commitment_kappa, params.kappa as u64);
    assert_eq!(base_boundary.commitment_data.len(), D * params.kappa as usize);
    assert!(
        base_boundary
            .commitment_data
            .iter()
            .all(|value| *value == F::ZERO),
        "canonical zero-seeded direct state must use full-shape zero Construction-2 u_perp"
    );

    let direct = direct
        .append_step(
            DirectCcsStep::new(step(&log, "steady_direct_0", 1, 2, 3)),
            &log,
            ajtai_mixers(),
        )
        .expect("append first steady direct CCS step");
    let latest = direct
        .latest_relation_and_advice()
        .expect("latest steady direct Construction-2 summary");

    assert_eq!(latest.incoming_ce_claims, params.k_rho as usize);
    assert_eq!(latest.output_ce_claims, params.k_rho as usize + 1);
    assert_eq!(latest.final_ce_claims, params.k_rho as usize);
}

#[test]
fn direct_ccs_single_step_compression_uses_latest_chunk_and_no_final_ce_digest() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let mut direct = DirectCcsIvcState::new(program).expect("direct CCS state");
    direct = direct
        .append_step(
            DirectCcsStep::new(step(&log, "direct_0", 1, 2, 3)),
            &log,
            ajtai_mixers(),
        )
        .expect("append first step");
    let mut trace = Vec::new();
    let (snark, vk, perf) = direct
        .compress_snark_with_trace(&mut |message| trace.push(message.to_string()))
        .unwrap_or_else(|err| panic!("compress direct CCS IVC: {err}; trace={trace:?}"));
    snark
        .verify(&vk, snark.public_image())
        .expect("verify compressed direct CCS IVC through public image");
    let statement = snark.statement();
    verify_direct_ccs_statement(&vk, &statement, snark.proof())
        .expect("verify compressed direct CCS IVC through compact statement");
    let mut tampered_statement = statement.clone();
    tampered_statement.step_count_out += 1;
    assert!(
        verify_direct_ccs_statement(&vk, &tampered_statement, snark.proof()).is_err(),
        "statement verifier must reject a tampered step counter"
    );
    let mut tampered_public_image = snark.public_image().clone();
    tampered_public_image.step_count_out += 1;
    assert!(
        verify_direct_ccs_statement(&vk, &tampered_public_image.statement(), snark.proof()).is_err(),
        "public verifier must reject a tampered Construction-2 output counter"
    );

    let mut tampered_snark = snark.clone();
    tampered_snark.public_image_mut().pc += 1;
    assert!(
        tampered_snark
            .verify(&vk, tampered_snark.public_image())
            .is_err(),
        "public verifier must reject a tampered direct relation program counter"
    );

    assert_eq!(
        perf.chunks.count, 1,
        "terminal compression must synthesize only the latest F' chunk"
    );
    assert_eq!(perf.final_ce.bundle_digest_constraints, 0);
    assert_eq!(perf.final_ce.bundle_digest_match_constraints, 0);
    assert_eq!(perf.final_ce.bundle_constraints, 0);
    assert_eq!(
        perf.constraints.construction2_fold, 0,
        "plain direct compression has no prior F' accumulator to fold"
    );
    assert_eq!(
        perf.constraints.construction2_fold_final_ce_consistency, 0,
        "folded prior F' authority must not contain terminal final-CE consistency"
    );
    assert_eq!(perf.committed.source.unclassified_private_values, 0);
    assert!(perf.committed.source.u32_values > 0);
    assert!(perf.committed.source.u64_values > 0);
    assert_eq!(
        perf.committed.source.values,
        perf.committed.source.bit_values + perf.committed.source.u32_values + perf.committed.source.u64_values,
        "terminal committed source labels must be explicitly classified"
    );
}

#[test]
fn direct_ccs_public_verifier_rejects_authoritative_boundary_tampering() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let direct = DirectCcsIvcState::start(program)
        .expect("canonical zero-seeded direct state")
        .append_step(
            DirectCcsStep::new(step(&log, "direct_authority_0", 1, 2, 3)),
            &log,
            ajtai_mixers(),
        )
        .expect("append direct CCS step");
    let (snark, vk, perf) = direct
        .compress_snark()
        .expect("compress direct CCS IVC boundary");
    snark
        .verify(&vk, snark.public_image())
        .expect("baseline direct CCS proof verifies");
    assert_eq!(perf.final_ce.bundle_digest_constraints, 0);
    DirectCcsRecursiveIvcPublicImage::from_terminal_and_f_prime_accumulator(
        snark.public_image().clone(),
        snark.public_image().construction2_accumulator_digest,
        params.b,
        0,
        params.k_rho as u64,
    )
    .expect("recursive public image accepts matching terminal/F' accumulator digests");
    let mut mismatched_f_prime_accumulator = snark.public_image().construction2_accumulator_digest;
    mismatched_f_prime_accumulator[0] ^= 1;
    assert!(
        DirectCcsRecursiveIvcPublicImage::from_terminal_and_f_prime_accumulator(
            snark.public_image().clone(),
            mismatched_f_prime_accumulator,
            params.b,
            0,
            params.k_rho as u64,
        )
        .is_err(),
        "recursive public image must reject a terminal x_out digest that is not the proven F' accumulator digest"
    );

    let mut image = snark.public_image().clone();
    image.mat_digest[0] += F::ONE;
    assert_public_verify_rejects(&vk, &snark, image, "relation digest");

    let mut image = snark.public_image().clone();
    image.vk_fs_digest[0] ^= 1;
    assert_public_verify_rejects(&vk, &snark, image, "vk_fs digest");

    let mut image = snark.public_image().clone();
    image.initial_boundary_digest[0] ^= 1;
    assert_public_verify_rejects(&vk, &snark, image, "initial boundary digest");

    let mut image = snark.public_image().clone();
    image.current_boundary_digest[0] ^= 1;
    assert_public_verify_rejects(&vk, &snark, image, "current boundary digest");

    let mut image = snark.public_image().clone();
    image.accumulator_out_digest[0] ^= 1;
    assert_public_verify_rejects(&vk, &snark, image, "final accumulator digest");

    let mut image = snark.public_image().clone();
    image.public_trace_out_digest[0] ^= 1;
    assert_public_verify_rejects(&vk, &snark, image, "public trace digest");

    let mut image = snark.public_image().clone();
    image.construction2_accumulator_digest[0] ^= 1;
    assert_public_verify_rejects(&vk, &snark, image, "Construction-2 accumulator digest");

    let mut image = snark.public_image().clone();
    let mut bad_x = image.x_out.bytes();
    bad_x[0] ^= 1;
    image.construction2_u_i.x_i =
        neo_fold_prototype::core::construction2::Construction2EncodedPublicInput::from_digest_bytes(bad_x);
    image.construction2_u_i.fresh_instance_digest = image.construction2_u_i.expected_fresh_instance_digest();
    assert_public_verify_rejects(&vk, &snark, image, "Construction-2 x_i");

    let mut image = snark.public_image().clone();
    image.construction2_u_i.commitment_data[0] += F::ONE;
    assert_public_verify_rejects(&vk, &snark, image, "stale Construction-2 commitment digest");

    let mut image = snark.public_image().clone();
    image.construction2_u_i.commitment_data[0] += F::ONE;
    image.construction2_u_i.commitment_digest = image.construction2_u_i.expected_commitment_digest();
    image.construction2_u_i.fresh_instance_digest = image.construction2_u_i.expected_fresh_instance_digest();
    assert_public_verify_rejects(
        &vk,
        &snark,
        image,
        "coherently re-digested Construction-2 commitment data",
    );

    let mut image = snark.public_image().clone();
    image.construction2_u_i.commitment_kappa = 0;
    image.construction2_u_i.commitment_data.clear();
    image.construction2_u_i.commitment_digest = image.construction2_u_i.expected_commitment_digest();
    image.construction2_u_i.fresh_instance_digest = image.construction2_u_i.expected_fresh_instance_digest();
    assert_public_verify_rejects(&vk, &snark, image, "noncanonical Construction-2 commitment shape");

    let mut statement = snark.statement();
    statement.mat_digest[0] += F::ONE;
    assert_statement_verify_rejects(&vk, &snark, statement, "statement relation digest");

    let mut statement = snark.statement();
    statement.vk_fs_digest[0] ^= 1;
    assert_statement_verify_rejects(&vk, &snark, statement, "statement vk_fs digest");

    let mut statement = snark.statement();
    statement.initial_boundary_digest[0] ^= 1;
    assert_statement_verify_rejects(&vk, &snark, statement, "statement initial boundary digest");

    let mut statement = snark.statement();
    statement.current_boundary_digest[0] ^= 1;
    assert_statement_verify_rejects(&vk, &snark, statement, "statement current boundary digest");

    let mut statement = snark.statement();
    statement.pc += 1;
    assert_statement_verify_rejects(&vk, &snark, statement, "statement pc");

    let mut statement = snark.statement();
    statement.chunk_count_out += 1;
    assert_statement_verify_rejects(&vk, &snark, statement, "statement chunk counter");

    let mut statement = snark.statement();
    statement.accumulator_out_digest[0] ^= 1;
    assert_statement_verify_rejects(&vk, &snark, statement, "statement accumulator digest");

    let mut statement = snark.statement();
    statement.public_trace_out_digest[0] ^= 1;
    assert_statement_verify_rejects(&vk, &snark, statement, "statement trace digest");

    let mut statement = snark.statement();
    statement.construction2_u_i.commitment_data[0] += F::ONE;
    statement.construction2_u_i.commitment_digest = statement.construction2_u_i.expected_commitment_digest();
    statement.construction2_u_i.fresh_instance_digest = statement.construction2_u_i.expected_fresh_instance_digest();
    assert_statement_verify_rejects(
        &vk,
        &snark,
        statement,
        "statement re-digested Construction-2 commitment",
    );
}

#[test]
fn direct_ccs_multi_step_plain_terminal_compression_is_refused() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let mut direct = DirectCcsIvcState::new(program).expect("direct CCS state");
    direct = direct
        .append_step(
            DirectCcsStep::new(step(&log, "direct_0", 1, 2, 3)),
            &log,
            ajtai_mixers(),
        )
        .expect("append first step");
    direct = direct
        .append_step(
            DirectCcsStep::new(step(&log, "direct_1", 2, 3, 5)),
            &log,
            ajtai_mixers(),
        )
        .expect("append second step");

    let err = match direct.compress_snark() {
        Ok(_) => panic!("plain direct terminal compression must refuse multi-step latest-only proofs"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("latest-only") && err.to_string().contains("disabled for multi-step"),
        "unexpected direct multi-step terminal compression error: {err}"
    );
}

#[test]
fn direct_ccs_append_rejects_shape_drift() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let direct = DirectCcsIvcState::new(program).expect("direct CCS state");

    let mut bad = step(&log, "bad_shape", 1, 2, 3);
    bad.witness.Z = Mat::zero(D, 2, F::ZERO);

    assert!(
        direct
            .append_step(DirectCcsStep::new(bad), &log, ajtai_mixers())
            .is_err(),
        "direct CCS append must reject witness/shape drift before it enters terminal compression"
    );
}

#[test]
fn direct_ccs_append_rejects_public_input_layout_drift() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let direct = DirectCcsIvcState::new(program).expect("direct CCS state");

    let mut bad = step(&log, "bad_public_layout", 1, 2, 3);
    bad.mcs.m_in = 2;
    bad.mcs.x.truncate(2);
    bad.witness.w = vec![F::ZERO; ccs.m - bad.mcs.m_in];

    let err = match direct.append_step(DirectCcsStep::new(bad), &log, ajtai_mixers()) {
        Ok(_) => panic!("direct CCS append must reject public input layout drift"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("fixed program public input len"),
        "unexpected public input drift error: {err}"
    );
}
