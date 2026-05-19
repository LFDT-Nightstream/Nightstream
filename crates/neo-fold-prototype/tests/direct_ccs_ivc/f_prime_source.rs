use super::*;

#[test]
fn direct_native_f_prime_advice_evaluates_compact_image_and_construction2_boundaries() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let direct = DirectCcsIvcState::start(program)
        .expect("direct state")
        .append_step(
            DirectCcsStep::new(step(&log, "direct_f_prime_source_0", 1, 2, 3)),
            &log,
            ajtai_mixers(),
        )
        .expect("append first direct step")
        .append_step(
            DirectCcsStep::new(step(&log, "direct_f_prime_source_1", 2, 3, 5)),
            &log,
            ajtai_mixers(),
        )
        .expect("append second direct step")
        .append_step(
            DirectCcsStep::new(step(&log, "direct_f_prime_source_2", 3, 5, 8)),
            &log,
            ajtai_mixers(),
        )
        .expect("append third direct step");

    let advice = DirectCcsNativeFPrimeAdvice::from_latest_state(&direct).expect("native direct F' advice");
    let step_image = advice.evaluate().expect("native direct F' evaluation");
    let compact_digest = step_image
        .compact_image()
        .expected_digest()
        .expect("compact F' image digest");
    let source = advice
        .low_norm_source_image()
        .expect("native direct F' low-norm source image");

    assert_ne!(compact_digest, [0u8; 32]);
    assert_ne!(source.expected_digest(), [0u8; 32]);
    assert!(
        source
            .values()
            .iter()
            .all(|value| *value == F::ZERO || *value == F::ONE),
        "direct F' source image must be binary low-norm material before it can be encoded as SuperNeo CCS"
    );
    assert_eq!(source.digest_count(), 18);
    assert_eq!(source.encoded_public_input_count(), 4);
    assert_eq!(source.construction2_commitment_fields(), 0);
    assert_eq!(
        source.u64_count(),
        32,
        "direct F' source image should contain compact counters and handles, not terminal commitment data"
    );
    assert_eq!(
        source.len(),
        source.digest_count() * 256 + source.encoded_public_input_count() * 256 + source.u64_count() * 64,
        "direct F' source image length must be mechanically explained by its primitive encodings"
    );
    let bits = advice.compact_image().x_out.field_image();
    let digest_only_r1cs = DirectCcsFPrimeLowNormSourceR1cs::from_source_image(
        &source,
        &bits,
        params.kappa as u64,
        1,
        params.k_rho as u64,
    )
    .expect("digest-only F' low-norm source R1CS");
    assert!(
        digest_only_r1cs.shape.digest_binding_constraints() > 0,
        "source shell must bind Poseidon2 digest recomputation"
    );
    assert_eq!(
        digest_only_r1cs.shape.constraints.nifs_v_verifier, 0,
        "caller-supplied source images must not gain F' authority rows"
    );
    assert!(
        !digest_only_r1cs.shape.has_proof_authority(),
        "digest recomputation alone must not be treated as F' authority"
    );
    assert_eq!(
        digest_only_r1cs.shape.constraint_count,
        digest_only_r1cs.shape.shell_constraints() + digest_only_r1cs.shape.digest_binding_constraints()
    );

    let source_r1cs =
        DirectCcsFPrimeLowNormSourceR1cs::from_native_advice(&advice, params.kappa as u64, 1, params.k_rho as u64)
            .expect("native F' low-norm source R1CS");
    assert_eq!(source_r1cs.shape.source.private_bits, source.len());
    assert_eq!(source_r1cs.shape.public_input_len, 257);
    assert_eq!(
        source_r1cs.shape.constraint_count,
        source_r1cs.shape.shell_constraints() + source_r1cs.shape.authority_constraints()
    );
    assert_eq!(source_r1cs.shape.constraints.x_out_link, 256);
    assert_eq!(source_r1cs.shape.constraints.construction2_boundary_link, 512);
    assert_eq!(source_r1cs.shape.constraints.construction2_commitment_shape, 256);
    assert_eq!(source_r1cs.shape.constraints.structural_counter, 768);
    assert_eq!(
        source_r1cs
            .shape
            .constraints
            .structural_counter_carry_bitness,
        189
    );
    assert_eq!(source_r1cs.shape.source.canonical_field_lanes, 96);
    assert_eq!(source_r1cs.shape.constraints.canonical_field_lane, 6048);
    assert!(source_r1cs.shape.shell_constraints() < source_r1cs.shape.constraint_count);
    assert!(
        source_r1cs.shape.constraints.poseidon_digest_recomputation > 0,
        "source R1CS must recompute at least one Construction-2 Poseidon2 digest"
    );
    assert!(
        source_r1cs.shape.constraints.nifs_v_verifier > 0,
        "native F' source R1CS must carry compact NIFS.V authority rows"
    );
    assert!(
        source_r1cs.shape.has_proof_authority(),
        "crate-owned native advice must bind digest recomputation and compact NIFS.V authority"
    );
    let expected_vars = source.len()
        + source_r1cs.shape.public_input_len
        + source_r1cs.shape.variables.counter_carry_bits
        + source_r1cs.shape.variables.canonical_field_lane_aux_bits
        + source_r1cs
            .shape
            .variables
            .poseidon_digest_recomputation_aux_bits;
    assert_eq!(source_r1cs.shape.variable_count, expected_vars);
    assert_eq!(source_r1cs.witness.len(), source_r1cs.shape.variable_count);
    assert!(
        source_r1cs.is_satisfied(),
        "native F' low-norm source R1CS witness must satisfy the source-link shell; first_bad={:?}",
        source_r1cs.first_unsatisfied_row()
    );
    assert!(DirectCcsFPrimeLowNormSourceR1cs::from_native_advice(
        &advice,
        params.kappa as u64 + 1,
        1,
        params.k_rho as u64
    )
    .is_err());
    let wrong_kappa_r1cs = DirectCcsFPrimeLowNormSourceR1cs::from_source_image(
        &source,
        &bits,
        params.kappa as u64 + 1,
        1,
        params.k_rho as u64,
    )
    .unwrap();
    assert!(!wrong_kappa_r1cs.is_satisfied());
    let mut tampered = source_r1cs.clone();
    let source_start = tampered.shape.public_input_len;
    tampered.witness[source_start + source.compact_x_out_bit_offset()] += F::ONE;
    assert!(
        !tampered.is_satisfied(),
        "source R1CS must reject tampering with compact x_out bits"
    );
    let mut tampered = source_r1cs.clone();
    tampered.witness[1 + 7] += F::ONE;
    assert!(
        !tampered.is_satisfied(),
        "source R1CS must reject tampering with public x_out bits through the source equality link"
    );
    let mut tampered = source_r1cs.clone();
    tampered.witness[source_start + source.construction2_u_out_x_i_bit_offset()] += F::ONE;
    assert!(
        !tampered.is_satisfied(),
        "source R1CS must reject tampering with Construction-2 output x_i bits"
    );
    let mut tampered = source_r1cs.clone();
    tampered.witness[source_start + source.current_boundary_out_digest_bit_offset()] += F::ONE;
    assert!(
        !tampered.is_satisfied(),
        "source R1CS must reject tampering with the recomputed current boundary output digest"
    );
    let mut tampered = source_r1cs.clone();
    let bit = source_start + source.construction2_u_in_commitment_digest_bit_offset();
    tampered.witness[bit] = F::ONE - tampered.witness[bit];
    assert!(
        !tampered.is_satisfied(),
        "source R1CS must reject binary tampering with the recomputed Construction-2 input fresh digest preimage"
    );
    let mut tampered = source_r1cs.clone();
    tampered.witness[source_start + source.chunk_count_out_bit_offset()] += F::ONE;
    assert!(
        !tampered.is_satisfied(),
        "source R1CS must reject tampering with chunk_count_out = chunk_count_in + 1"
    );
    let mut tampered = source_r1cs.clone();
    tampered.witness[source_start + source.output_ce_claims_bit_offset()] += F::ONE;
    assert!(
        !tampered.is_satisfied(),
        "source R1CS must reject tampering with output_CE = incoming_CE + fresh_CCS"
    );
    let mut tampered = source_r1cs.clone();
    tampered.witness[source_start + source.fresh_claims_bit_offset()] += F::ONE;
    tampered.witness[source_start + source.output_ce_claims_bit_offset()] += F::ONE;
    assert!(
        !tampered.is_satisfied(),
        "source R1CS must reject non-fixed fresh claim arity"
    );
    for offset in [
        source.final_ce_claims_bit_offset(),
        source.nifs_chunk_index_bit_offset(),
        source.nifs_fresh_claims_bit_offset(),
        source.nifs_incoming_ce_claims_bit_offset(),
        source.nifs_pi_ccs_outputs_bit_offset(),
        source.nifs_final_ce_claims_bit_offset(),
        source.nifs_fe_sumcheck_rounds_bit_offset(),
        source.nifs_fe_sumcheck_messages_bit_offset(),
        source.nifs_nc_sumcheck_rounds_bit_offset(),
        source.nifs_nc_sumcheck_messages_bit_offset(),
        source.nifs_transcript_absorbed_in_bit_offset(),
        source.nifs_transcript_absorbed_out_bit_offset(),
    ] {
        let mut tampered = source_r1cs.clone();
        tampered.witness[source_start + offset] += F::ONE;
        assert!(!tampered.is_satisfied(), "tampered F' NIFS payload accepted");
    }
    let mut tampered = source_r1cs.clone();
    let lane = source_start + source.construction2_u_in_commitment_digest_bit_offset();
    for bit in 0..64 {
        tampered.witness[lane + bit] = if bit == 0 || bit >= 32 { F::ONE } else { F::ZERO };
    }
    assert!(
        !tampered.is_satisfied(),
        "source R1CS must reject non-canonical field lanes"
    );
    let program = source_r1cs
        .to_direct_ccs_program(&params)
        .expect("source R1CS converts to direct CCS program");
    let source_log = make_ajtai_module_for_cols(&params, source_r1cs.shape.variable_count.div_ceil(D));
    let step = source_r1cs
        .to_direct_ccs_step(&program, &source_log, "direct_f_prime_source")
        .expect("source R1CS witness is low-norm packable");
    assert_eq!(step.into_step_input().mcs.m_in, source_r1cs.shape.public_input_len);
    assert_eq!(
        &advice.construction2_u_in().x_i,
        &step_image.compact_image().x_in,
        "native direct F' advice must bind the input Construction-2 instance to x_i"
    );
    assert_eq!(
        &step_image.construction2_u_out().x_i,
        &step_image.compact_image().x_out,
        "native direct F' evaluation must bind the output Construction-2 instance to x_out"
    );
    step_image
        .terminal_public_image()
        .validate_final_construction2_public_boundary()
        .expect("native direct F' step image exports a valid terminal public image");
}

#[test]
fn direct_f_prime_source_authority_requires_digest_binding_and_nifs_v_rows() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let direct = DirectCcsIvcState::start(program)
        .expect("direct state")
        .append_step(
            DirectCcsStep::new(step(&log, "authority_gate_0", 1, 2, 3)),
            &log,
            ajtai_mixers(),
        )
        .expect("append first direct step")
        .append_step(
            DirectCcsStep::new(step(&log, "authority_gate_1", 2, 3, 5)),
            &log,
            ajtai_mixers(),
        )
        .expect("append second direct step");
    let advice = DirectCcsNativeFPrimeAdvice::from_latest_state(&direct).expect("native direct F' advice");
    let source = advice
        .low_norm_source_image()
        .expect("native direct F' low-norm source image");
    let bits = advice.compact_image().x_out.field_image();
    let digest_only_r1cs = DirectCcsFPrimeLowNormSourceR1cs::from_source_image(
        &source,
        &bits,
        params.kappa as u64,
        1,
        params.k_rho as u64,
    )
    .expect("digest-only F' low-norm source R1CS");
    let source_r1cs =
        DirectCcsFPrimeLowNormSourceR1cs::from_native_advice(&advice, params.kappa as u64, 1, params.k_rho as u64)
            .expect("native F' low-norm source R1CS");

    assert!(
        digest_only_r1cs.shape.digest_binding_constraints() > 0,
        "source shell must bind Poseidon2 digest recomputation before it can ever become authority"
    );
    assert_eq!(
        digest_only_r1cs.shape.constraints.nifs_v_verifier, 0,
        "caller-supplied source images intentionally have no compact NIFS.V authority rows"
    );
    assert!(
        !digest_only_r1cs.shape.has_proof_authority(),
        "digest recomputation without NIFS.V verifier rows must not be treated as F' authority"
    );
    assert_eq!(digest_only_r1cs.shape.authority_constraints(), 0);

    assert!(
        source_r1cs.shape.has_proof_authority(),
        "crate-owned native advice opens the authority gate by adding compact NIFS.V rows"
    );
    assert_eq!(
        source_r1cs.shape.authority_constraints(),
        source_r1cs.shape.digest_binding_constraints() + source_r1cs.shape.constraints.nifs_v_verifier
    );
}
