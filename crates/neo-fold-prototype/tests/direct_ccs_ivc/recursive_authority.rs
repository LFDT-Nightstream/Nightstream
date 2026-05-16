use super::*;

#[test]
fn direct_recursive_f_prime_authority_is_not_public_or_terminal_source_image_based() {
    let direct_mod = include_str!("../../src/frontends/direct_ccs/mod.rs");
    let recursive_src = include_str!("../../src/frontends/direct_ccs/recursive/mod.rs");
    let f_prime_chain_src = include_str!("../../src/frontends/direct_ccs/f_prime/chain/mod.rs");
    let construction2_fold_src = [
        include_str!("../../src/frontends/direct_ccs/terminal/construction2_fold/mod.rs"),
        include_str!("../../src/frontends/direct_ccs/terminal/construction2_fold/synthesis.rs"),
        include_str!("../../src/frontends/direct_ccs/terminal/construction2_fold/measurement.rs"),
        include_str!("../../src/frontends/direct_ccs/terminal/construction2_fold/types.rs"),
    ]
    .join("\n");
    let step_src = include_str!("../../src/frontends/direct_ccs/step.rs");
    let crate_root = include_str!("../../src/lib.rs");

    assert!(
        !direct_mod.contains("pub use f_prime_chain"),
        "the low-norm F' chain helper must stay internal so callers cannot supply arbitrary F' authority"
    );
    assert!(
        !crate_root.contains("DirectCcsFPrimeChain"),
        "public direct CCS API must not expose caller-supplied F' authority helpers"
    );
    assert!(
        !recursive_src.contains("append_latest_terminal_export"),
        "recursive direct CCS append must not fold terminal committed/source-image exports"
    );
    assert!(
        !recursive_src.contains("direct_terminal_shape_export")
            && !recursive_src.contains("DirectCcsTerminalCommittedRelation")
            && !f_prime_chain_src.contains("direct_terminal_shape_export")
            && !f_prime_chain_src.contains("DirectCcsTerminalCommittedRelation"),
        "direct F' authority must not be sourced from terminal committed Spartan exports"
    );
    assert!(
        step_src.contains("not SuperNeo low-norm packable")
            && step_src.contains("validate_direct_ccs_step_witness")
            && step_src.contains("embed_direct_ccs_witness")
            && step_src.contains("commit_embedded_witness"),
        "direct step construction must keep the low-norm witness-to-claim boundary explicit"
    );
    assert!(
        !construction2_fold_src.contains("enforce_direct_terminal_final_ce_consistency")
            && !construction2_fold_src.contains("direct_terminal_construction2_fold_final_ce")
            && !construction2_fold_src.contains("final_ce_consistency"),
        "Construction-2 folded F' authority must not include terminal final-CE consistency; that belongs at final compression"
    );
}

#[test]
fn direct_recursive_ivc_state_starts_without_terminal_step() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let _log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let recursive = DirectCcsRecursiveIvcState::start(program).expect("direct recursive state");
    let summary = recursive.summary();

    assert_eq!(summary.semantic.chunks, 0);
    assert_eq!(summary.semantic.steps, 0);
    assert_eq!(summary.semantic.terminal_chunks_synthesized, 0);
    assert_eq!(summary.semantic.carried_ce_claims, params.k_rho as usize);
    assert_eq!(summary.f_prime.folded_r2_steps, 0);
    assert_eq!(summary.f_prime.carried_ce_claims, 0);
    assert!(!summary.f_prime.native_evaluator_available);
    assert!(!summary.f_prime.encoder_required);
    assert!(!summary.f_prime.encoder_available);
    assert_eq!(summary.f_prime.compact_image_digest, None);
    assert!(!summary.f_prime.low_norm_source.available);
    assert_eq!(summary.f_prime.low_norm_source.len, 0);
    assert_eq!(summary.f_prime.low_norm_source.digest, None);
    assert_eq!(summary.f_prime.low_norm_source.r1cs.constraints, 0);
    assert_eq!(summary.f_prime.low_norm_source.r1cs.variables, 0);
    assert_eq!(summary.f_prime.low_norm_source.r1cs.nnz, 0);
    assert_eq!(summary.f_prime.low_norm_source.r1cs.shell_constraints, 0);
    assert_eq!(summary.f_prime.low_norm_source.r1cs.authority_constraints, 0);
    assert_eq!(
        summary
            .f_prime
            .low_norm_source
            .r1cs
            .poseidon_digest_recomputation_constraints,
        0
    );
    assert_eq!(
        summary
            .f_prime
            .low_norm_source
            .r1cs
            .nifs_v_verifier_constraints,
        0
    );
    assert_eq!(summary.proof.encoder_blocker, None);
    assert!(!summary.proof.standalone_authority_ready);
}

#[test]
fn direct_recursive_ivc_compression_requires_appended_step() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let _log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let recursive = DirectCcsRecursiveIvcState::start(program).expect("direct recursive state");
    let err = match recursive.compress_recursive_snark() {
        Ok(_) => panic!("empty recursive direct IVC compression must be rejected"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("at least one appended F' step"),
        "unexpected empty recursive compression error: {err}"
    );
}

#[test]
#[ignore = "Compact F' authority append is intentionally heavy; the non-ignored direct_ccs_r1cs_low_norm positive authority test covers this path."]
fn direct_recursive_ivc_append_does_not_fold_terminal_source_image_exports() {
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
    let summary = recursive.summary();

    assert_eq!(summary.semantic.chunks, 2);
    assert_eq!(summary.semantic.steps, 2);
    assert_eq!(
        summary.f_prime.folded_r2_steps, 1,
        "direct recursive append should fold one compact prior F' source relation"
    );
    assert_eq!(summary.semantic.carried_ce_claims, params.k_rho as usize);
    assert!(
        summary.f_prime.carried_ce_claims > 0,
        "the folded compact F' source relation must expose carried CE claims"
    );
    assert!(
        !summary.f_prime.encoder_required,
        "the prior F' relation was already folded during the second append"
    );
    assert!(
        summary.f_prime.native_evaluator_available,
        "the latest compact direct F' native evaluator should remain available for diagnostics"
    );
    assert!(
        summary.f_prime.encoder_available,
        "the compact low-norm F' authority relation should be available"
    );
    assert!(
        summary.f_prime.compact_image_digest.is_some(),
        "the compact F' image digest can exist, but it is not proof authority by itself"
    );
    assert!(
        summary.f_prime.low_norm_source.available,
        "the compact native F' advice should now export a low-norm source image"
    );
    assert!(
        summary.f_prime.low_norm_source.len > 0,
        "the low-norm F' source image should contain the bits needed by the compact authority relation"
    );
    assert!(
        summary.f_prime.low_norm_source.digest.is_some(),
        "the low-norm F' source image digest should be available as a diagnostic handle"
    );
    assert!(summary.f_prime.low_norm_source.r1cs.constraints > 0);
    assert!(summary.f_prime.low_norm_source.r1cs.variables > 0);
    assert!(summary.f_prime.low_norm_source.r1cs.nnz > 0);
    assert!(
        summary.f_prime.low_norm_source.r1cs.authority_constraints > 0,
        "authority requires Poseidon2 digest binding plus compact NIFS.V rows"
    );
    assert!(
        summary
            .f_prime
            .low_norm_source
            .r1cs
            .poseidon_digest_recomputation_constraints
            > 0,
        "Construction-2 digest fields must not remain self-consistent diagnostic data"
    );
    assert!(
        summary
            .f_prime
            .low_norm_source
            .r1cs
            .nifs_v_verifier_constraints
            > 0,
        "compact native advice should install NIFS.V authority rows"
    );
    assert_eq!(summary.proof.encoder_blocker, None);
    assert!(summary.proof.standalone_authority_ready);
}

#[test]
fn direct_recursive_latest_step_is_not_historical_replay() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let direct = DirectCcsIvcState::start(program)
        .expect("direct state")
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

    let latest = direct
        .latest_relation_and_advice()
        .expect("latest direct F' relation");
    assert_eq!(latest.chunk_index, 1);
    assert_eq!(latest.fresh_claims, 1);
    assert_eq!(latest.incoming_ce_claims, params.k_rho as usize);
    assert_eq!(latest.output_ce_claims, params.k_rho as usize + 1);
    assert_eq!(latest.final_ce_claims, params.k_rho as usize);
    assert_eq!(
        direct.final_state().chunk_count,
        2,
        "latest relation must come from the newest direct step, not historical replay"
    );
}

#[test]
fn direct_compact_f_prime_image_binds_latest_step_without_terminal_material() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(1).expect("valid Fibonacci params");
    let ccs = fibonacci_ccs();
    let log = make_ajtai_module(&params);
    let program = DirectCcsProgram::new_with_public_input_len(&params, &ccs, 3).expect("fixed direct public input");
    let direct = DirectCcsIvcState::start(program)
        .expect("direct state")
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

    let image = DirectCcsCompactFPrimeImage::from_latest_state(&direct).expect("compact direct F' image");
    let image_digest = image.expected_digest().expect("valid compact F' digest");

    assert_eq!(image.chunk_count_in, 1);
    assert_eq!(image.chunk_count_out, 2);
    assert_eq!(image.step_count_in, 1);
    assert_eq!(image.step_count_out, 2);
    assert_eq!(image.fresh_claims, 1);
    assert_eq!(image.incoming_ce_claims, params.k_rho as u64);
    assert_eq!(image.output_ce_claims, params.k_rho as u64 + 1);
    assert_eq!(image.final_ce_claims, params.k_rho as u64);
    assert_ne!(image_digest, [0u8; 32]);

    let mut bad = image.clone();
    bad.chunk_count_out += 1;
    assert!(
        bad.validate().is_err(),
        "compact direct F' image must bind the output chunk counter"
    );

    let mut bad = image.clone();
    bad.current_boundary_out_digest[0] ^= 1;
    assert!(
        bad.validate().is_err(),
        "compact direct F' image must bind the latest chunk into the current boundary"
    );

    let mut bad = image.clone();
    bad.x_out = image.x_in.clone();
    assert!(
        bad.validate().is_err(),
        "compact direct F' image must bind x_out to output counters and accumulator handles"
    );
}
