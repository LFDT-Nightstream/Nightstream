use super::*;

#[test]
#[ignore = "Spartan recursive terminal compression is intentionally expensive; run explicitly when measuring the direct recursive proof surface."]
fn direct_recursive_ivc_compresses_terminal_boundary_and_binds_accumulator_digest() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let recursive = DirectCcsRecursiveIvcState::start(program)
        .expect("direct recursive state")
        .append_step(
            DirectCcsStep::new(step(&log, "recursive_direct_0", 1, 2, 3)),
            &log,
            ajtai_mixers(),
        )
        .expect("append first recursive direct step");
    let summary = recursive.summary();
    assert_eq!(summary.semantic.chunks, 1);
    assert_eq!(summary.semantic.steps, 1);
    assert_eq!(
        summary.semantic.terminal_chunks_synthesized, 1,
        "recursive terminal compression must synthesize one latest F' chunk"
    );
    assert_eq!(summary.semantic.carried_ce_claims, params.k_rho as usize);
    assert_eq!(summary.f_prime.folded_r2_steps, 0);
    assert_eq!(
        summary.f_prime.carried_ce_claims, 0,
        "single-step terminal compression has no folded prior F' accumulator"
    );
    assert!(
        summary.proof.standalone_authority_ready,
        "a single-step direct proof is the Construction-2 base case and needs no folded prior F' chain"
    );
    assert!(summary.f_prime.native_evaluator_available);
    assert!(
        !summary.f_prime.encoder_required,
        "base case has no prior F' step to encode"
    );
    assert!(
        summary.f_prime.encoder_available,
        "the compact low-norm direct F' relation is available even though the base case does not need prior F' authority"
    );
    assert_eq!(summary.proof.encoder_blocker, None);
    assert!(
        summary.f_prime.compact_image_digest.is_some(),
        "base case still has a compact latest F' image"
    );
    assert!(summary.f_prime.low_norm_source.available);
    assert!(summary.f_prime.low_norm_source.len > 0);
    assert!(summary.f_prime.low_norm_source.digest.is_some());
    assert!(
        summary.f_prime.low_norm_source.r1cs.authority_constraints > 0,
        "crate-owned native advice should expose compact source authority rows"
    );
    assert!(
        summary
            .f_prime
            .low_norm_source
            .r1cs
            .poseidon_digest_recomputation_constraints
            > 0
    );
    assert!(
        summary
            .f_prime
            .low_norm_source
            .r1cs
            .nifs_v_verifier_constraints
            > 0
    );
    assert_eq!(recursive.direct_state().final_state().chunk_count, 1);

    let (snark, vk, perf) = recursive
        .compress_recursive_snark()
        .expect("recursive direct compression");
    snark
        .verify(&vk, snark.public_image())
        .expect("recursive direct proof verifies");
    snark
        .verify(&vk, snark.public_image())
        .expect("recursive direct public verifier accepts the baseline proof");
    snark
        .public_image()
        .expected_digest()
        .expect("recursive public image digest is well formed");
    assert_eq!(
        snark.public_image().proven_accumulator_digest,
        snark
            .public_image()
            .terminal_public_image
            .accumulator_out_digest,
        "recursive public image must bind the same accumulator digest proven by the terminal F' proof"
    );
    assert_eq!(
        snark.public_image().proven_chunk_count,
        snark.public_image().terminal_public_image.chunk_count_out
    );
    assert_eq!(
        snark.public_image().proven_step_count,
        snark.public_image().terminal_public_image.step_count_out
    );
    assert!(
        snark.f_prime_final_claims().is_empty(),
        "base F' authority comes from the verifier-key-bound default accumulator digest, not a carried CE proof"
    );
    assert_eq!(perf.f_prime_final_ce_constraints, 0);
    assert_eq!(perf.f_prime_final_ce_proof_bytes, 0);
    assert_eq!(
        perf.terminal.constraints.construction2_fold, 0,
        "first recursive step starts from the verifier-key-bound default F' accumulator"
    );
    assert!(snark.f_prime_chain_snark().is_none());
    assert_eq!(perf.f_prime_chain_constraints, 0);
    assert_eq!(perf.f_prime_chain_proof_bytes, 0);
    assert_eq!(perf.terminal_proof_bytes, perf.total_proof_bytes);
}

#[test]
#[ignore = "Compact F' authority append is intentionally heavy; the non-ignored direct_ccs_r1cs_low_norm positive authority test covers this summary path."]
fn direct_recursive_ivc_multi_step_summary_uses_compact_f_prime_authority() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let recursive = DirectCcsRecursiveIvcState::start(program)
        .expect("direct recursive state")
        .append_step(
            DirectCcsStep::new(step(&log, "recursive_direct_0", 1, 2, 3)),
            &log,
            ajtai_mixers(),
        )
        .expect("append first recursive direct step")
        .append_step(
            DirectCcsStep::new(step(&log, "recursive_direct_1", 2, 3, 5)),
            &log,
            ajtai_mixers(),
        )
        .expect("append second recursive direct step");

    assert!(
        recursive.summary().proof.standalone_authority_ready,
        "multi-step direct recursive state should report standalone authority after folding compact prior F'"
    );
    assert!(!recursive.summary().f_prime.encoder_required);
    assert!(recursive.summary().f_prime.native_evaluator_available);
    assert!(recursive.summary().f_prime.low_norm_source.available);
    assert!(recursive.summary().f_prime.low_norm_source.len > 0);
    assert!(recursive.summary().f_prime.encoder_available);
    assert!(
        recursive
            .summary()
            .f_prime
            .low_norm_source
            .r1cs
            .authority_constraints
            > 0
    );
    assert!(
        recursive
            .summary()
            .f_prime
            .low_norm_source
            .r1cs
            .poseidon_digest_recomputation_constraints
            > 0
    );
    assert!(
        recursive
            .summary()
            .f_prime
            .low_norm_source
            .r1cs
            .nifs_v_verifier_constraints
            > 0
    );
    assert_eq!(recursive.summary().proof.encoder_blocker, None);
}

#[test]
#[ignore = "Spartan recursive terminal compression is intentionally expensive; run explicitly when measuring the direct recursive proof surface."]
fn direct_recursive_ivc_public_image_rejects_unbound_accumulator_digest() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let recursive = DirectCcsRecursiveIvcState::start(program)
        .expect("direct recursive state")
        .append_step(
            DirectCcsStep::new(step(&log, "recursive_direct_0", 1, 2, 3)),
            &log,
            ajtai_mixers(),
        )
        .expect("append first recursive direct step");

    let (snark, vk, _) = recursive
        .compress_recursive_snark()
        .expect("recursive direct compression");
    let mut image = snark.public_image().clone();
    image.proven_accumulator_digest[0] ^= 1;
    assert!(
        image.validate_recursive_boundary().is_err(),
        "recursive boundary must reject an accumulator digest not bound to terminal x_out"
    );
    assert!(
        snark.verify(&vk, &image).is_err(),
        "recursive verifier must reject an accumulator digest not bound to terminal x_out"
    );

    let mut image = snark.public_image().clone();
    image.proven_chunk_count += 1;
    assert!(
        image.validate_recursive_boundary().is_err(),
        "recursive boundary must reject a chunk counter not bound to terminal x_out"
    );
    assert!(
        snark.verify(&vk, &image).is_err(),
        "recursive public verifier must reject a chunk counter not bound to terminal x_out"
    );

    let mut image = snark.public_image().clone();
    image.proven_step_count += 1;
    assert!(
        image.validate_recursive_boundary().is_err(),
        "recursive boundary must reject a step counter not bound to terminal x_out"
    );
    assert!(
        snark.verify(&vk, &image).is_err(),
        "recursive public verifier must reject a step counter not bound to terminal x_out"
    );

    let mut image = snark.public_image().clone();
    image.f_prime_final_ce_claims += 1;
    assert!(
        snark.verify(&vk, &image).is_err(),
        "recursive verifier must reject a final F' CE claim-count mismatch"
    );

    let mut image = snark.public_image().clone();
    image.f_prime_accumulator_base += 1;
    assert!(
        snark.verify(&vk, &image).is_err(),
        "recursive verifier must reject a final F' accumulator base mismatch"
    );

    let mut terminal = snark.public_image().terminal_public_image.clone();
    terminal.accumulator_out_digest[0] ^= 1;
    assert!(
        DirectCcsRecursiveIvcPublicImage::from_terminal_and_f_prime_accumulator(
            terminal,
            [0u8; 32],
            2,
            1,
            params.k_rho as u64
        )
        .is_err(),
        "recursive public image constructor must reject terminal images that fail direct boundary validation"
    );
}
