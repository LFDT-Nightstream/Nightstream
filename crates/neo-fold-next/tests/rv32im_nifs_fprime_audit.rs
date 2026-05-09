use std::env;

use neo_fold_next::core::proof::FoldSchedule;
use neo_fold_next::rv32im::audit::{
    audit_rv32im_nifs_round_trip_from_chunk_step_relation, build_rv32im_chunk_step_ivc_relations,
    build_rv32im_main_recursion_f_prime_advices,
    build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape,
    build_rv32im_main_recursion_f_prime_public_output,
    debug_check_rv32im_main_recursion_step_spartan_chunk_replay_surface,
    debug_check_rv32im_main_recursion_step_spartan_pi_ccs_replay_lengths,
    evaluate_rv32im_main_recursion_f_prime_advice,
    rv32im_main_recursion_advice_tamper_authoritative_ccs_replay_first_round_coeff,
    rv32im_main_recursion_advice_tamper_chunk_index,
    rv32im_main_recursion_advice_tamper_dec_child_commitment_first_word,
    rv32im_main_recursion_advice_tamper_vk_fs_main_lane_shape_digest_first_byte,
    rv32im_main_recursion_advice_tamper_x_hash_first_byte, verify_rv32im_main_recursion_f_prime_public_output,
    Rv32imChunkStepIvcRelation, Rv32imMainRecursionFPrimeAdvice, Rv32imMainRecursionFPrimeBackendRelation,
};
use neo_fold_next::rv32im::final_relation::prove_rv32im_final_statement_from_accepted;
use neo_fold_next::rv32im::{
    build_mixed_opcode_perf_source_case, build_rv32im_accepted_proof_artifact, prove_rv32im_public_proof_with_options,
    Rv32imProofInput, Rv32imPublicProofOptions, SimpleKernelError,
};

struct AuditFixture {
    chunk_step_relations: Vec<Rv32imChunkStepIvcRelation>,
    f_prime_advices: Vec<Rv32imMainRecursionFPrimeAdvice>,
    backend_relations: Vec<Rv32imMainRecursionFPrimeBackendRelation>,
}

fn perf_opcode_count_from_env() -> usize {
    match env::var("NS_DEBUG_N") {
        Ok(raw) => raw.parse().expect("NS_DEBUG_N must parse as usize"),
        Err(_) => 1,
    }
}

fn short_digest(digest: [u8; 32]) -> String {
    let mut out = String::with_capacity(16);
    for byte in digest.iter().take(8) {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn build_audit_fixture() -> Result<AuditFixture, SimpleKernelError> {
    let source = build_mixed_opcode_perf_source_case(perf_opcode_count_from_env());
    let input = Rv32imProofInput {
        max_steps: source.program_words.len(),
        source,
    };
    let options = Rv32imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let public_proof = prove_rv32im_public_proof_with_options(&input, options)?;
    let accepted_artifact = build_rv32im_accepted_proof_artifact(&public_proof)?;
    let (final_statement, final_proof) = prove_rv32im_final_statement_from_accepted(&accepted_artifact)?;
    let chunk_step_relations = build_rv32im_chunk_step_ivc_relations(&final_statement, &final_proof)?;
    let f_prime_advices = build_rv32im_main_recursion_f_prime_advices(&chunk_step_relations)?;
    let (_spartan_shape, backend_relations) = build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape(
        &chunk_step_relations,
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("build recursive-step backend relations failed: {err}")))?;

    if chunk_step_relations.len() != f_prime_advices.len() || chunk_step_relations.len() != backend_relations.len() {
        return Err(SimpleKernelError::Bridge(
            "audit fixture relation/advice/backend-relation lengths diverged".into(),
        ));
    }

    Ok(AuditFixture {
        chunk_step_relations,
        f_prime_advices,
        backend_relations,
    })
}

fn print_step_audit(
    step_index: usize,
    relation: &Rv32imChunkStepIvcRelation,
    advice: &Rv32imMainRecursionFPrimeAdvice,
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<(), SimpleKernelError> {
    let native_output = evaluate_rv32im_main_recursion_f_prime_advice(advice)?;
    let statement_bytes = bincode::serialize(&backend_relation.spartan_statement)
        .map_err(|err| SimpleKernelError::Bridge(format!("encode recursive-step statement failed: {err}")))?;
    let x_out_bytes = bincode::serialize(&backend_relation.spartan_statement.x_out)
        .map_err(|err| SimpleKernelError::Bridge(format!("encode recursive-step x_out failed: {err}")))?;

    println!("step #{step_index}");
    println!(
        "  chunk_index={} step_span=[{}, {}) halted_out={}",
        relation.statement.step_public.chunk_index,
        relation.statement.step_public.step_lo,
        relation.statement.step_public.step_hi,
        relation.witness.terminal_step
    );
    println!(
        "  x_i={} x_out={}",
        short_digest(advice.x_i().bytes()),
        short_digest(native_output.x_out().bytes())
    );
    println!(
        "  folded_in={} folded_out={}",
        short_digest(advice.folded_accumulator_in_digest()),
        short_digest(native_output.folded_accumulator_digest())
    );
    println!(
        "  statement_x_out={} pi_ccs_outputs={} fresh_claims={} dec_children={} fe_rounds={} nc_rounds={}",
        short_digest(backend_relation.spartan_statement.x_out.bytes()),
        backend_relation.payload.pi_ccs.ccs_outputs.len(),
        backend_relation.payload.fresh_claims.len(),
        backend_relation.payload.pi_dec.children.len(),
        backend_relation.payload.pi_ccs.replay.sumcheck_rounds.len(),
        backend_relation
            .payload
            .pi_ccs
            .replay
            .sumcheck_rounds_nc
            .len()
    );
    println!(
        "  public_statement_bytes={} x_out_only_bytes={}",
        statement_bytes.len(),
        x_out_bytes.len(),
    );

    Ok(())
}

fn expect_tamper_failure<F>(
    label: &str,
    baseline_output: &neo_fold_next::rv32im::Rv32imMainRecursionFPrimePublicOutput,
    baseline_advice: &Rv32imMainRecursionFPrimeAdvice,
    mutate: F,
    expected_snippet: &str,
) where
    F: FnOnce(&mut Rv32imMainRecursionFPrimeAdvice),
{
    let mut tampered = baseline_advice.clone();
    mutate(&mut tampered);
    let err = verify_rv32im_main_recursion_f_prime_public_output(baseline_output, &tampered).expect_err(label);
    assert!(
        err.to_string().contains(expected_snippet),
        "{label}: expected error containing `{expected_snippet}`, got `{err}`"
    );
}

fn assert_statement_matches_native_f_prime(
    step_index: usize,
    backend_relation: &Rv32imMainRecursionFPrimeBackendRelation,
    native_output: &neo_fold_next::rv32im::Rv32imMainRecursionFPrimeStepImage,
) {
    assert_eq!(
        backend_relation.spartan_statement.x_out,
        *native_output.x_out(),
        "step {step_index}: recursive-step statement x_out diverged from native F' x_out (statement={}, native={})",
        short_digest(backend_relation.spartan_statement.x_out.bytes()),
        short_digest(native_output.x_out().bytes()),
    );
}

#[test]
#[ignore = "manual audit canary for native NIFS/F' parity and recursive-step statement drift"]
fn rv32im_nifs_fprime_authoritative_parity_audit() -> Result<(), SimpleKernelError> {
    let fixture = build_audit_fixture()?;
    assert!(
        fixture.chunk_step_relations.len() > 1,
        "expected a multi-step fixture; rerun with RowsPerChunk(1) and NS_DEBUG_N>=1"
    );

    println!("rv32im nifs/f' audit");
    println!("  NS_DEBUG_N={}", perf_opcode_count_from_env());
    println!("  recursive_steps={}", fixture.chunk_step_relations.len());

    for (step_index, ((relation, advice), backend_relation)) in fixture
        .chunk_step_relations
        .iter()
        .zip(fixture.f_prime_advices.iter())
        .zip(fixture.backend_relations.iter())
        .enumerate()
    {
        print_step_audit(step_index, relation, advice, backend_relation)?;

        audit_rv32im_nifs_round_trip_from_chunk_step_relation(relation)?;

        let native_output = evaluate_rv32im_main_recursion_f_prime_advice(advice)?;
        let public_output = build_rv32im_main_recursion_f_prime_public_output(advice)?;
        let verified_output = verify_rv32im_main_recursion_f_prime_public_output(&public_output, advice)?;

        assert_eq!(
            public_output.x_out(),
            native_output.x_out(),
            "step {step_index}: public x_out drifted from native F'"
        );
        assert_eq!(
            verified_output.x_out(),
            native_output.x_out(),
            "step {step_index}: verified x_out drifted from native F'"
        );
        assert_statement_matches_native_f_prime(step_index, backend_relation, &native_output);

        let statement_bytes = bincode::serialize(&backend_relation.spartan_statement)
            .map_err(|err| SimpleKernelError::Bridge(format!("encode recursive-step statement failed: {err}")))?;
        let x_out_bytes = bincode::serialize(&backend_relation.spartan_statement.x_out)
            .map_err(|err| SimpleKernelError::Bridge(format!("encode recursive-step x_out failed: {err}")))?;
        assert_eq!(
            statement_bytes, x_out_bytes,
            "step {step_index}: recursive-step public statement carries more than x_out"
        );

        debug_check_rv32im_main_recursion_step_spartan_chunk_replay_surface(backend_relation).map_err(|err| {
            SimpleKernelError::Bridge(format!("step {step_index} replay-surface check failed: {err}"))
        })?;
        debug_check_rv32im_main_recursion_step_spartan_pi_ccs_replay_lengths(backend_relation).map_err(|err| {
            SimpleKernelError::Bridge(format!("step {step_index} Pi_CCS replay lengths failed: {err}"))
        })?;
    }

    let baseline_advice = fixture
        .f_prime_advices
        .first()
        .expect("non-empty recursive-step advice chain");
    let baseline_output = build_rv32im_main_recursion_f_prime_public_output(baseline_advice)?;

    expect_tamper_failure(
        "tampered chunk_index must fail",
        &baseline_output,
        baseline_advice,
        rv32im_main_recursion_advice_tamper_chunk_index,
        "chunk_index",
    );
    expect_tamper_failure(
        "tampered x_i must fail",
        &baseline_output,
        baseline_advice,
        rv32im_main_recursion_advice_tamper_x_hash_first_byte,
        "x_i",
    );
    expect_tamper_failure(
        "tampered vk_fs must fail",
        &baseline_output,
        baseline_advice,
        rv32im_main_recursion_advice_tamper_vk_fs_main_lane_shape_digest_first_byte,
        "vk_fs",
    );
    expect_tamper_failure(
        "tampered Pi_CCS replay must fail",
        &baseline_output,
        baseline_advice,
        rv32im_main_recursion_advice_tamper_authoritative_ccs_replay_first_round_coeff,
        "Pi_CCS replay",
    );
    expect_tamper_failure(
        "tampered Pi_DEC child must fail",
        &baseline_output,
        baseline_advice,
        |advice| rv32im_main_recursion_advice_tamper_dec_child_commitment_first_word(advice, 0),
        "DEC child",
    );

    let mut tampered_public_output = baseline_output.clone();
    tampered_public_output.x_out_mut().bytes_mut()[0] ^= 1;
    let err = verify_rv32im_main_recursion_f_prime_public_output(&tampered_public_output, baseline_advice)
        .expect_err("tampered public x_out must fail");
    assert!(
        err.to_string().contains("x_out"),
        "tampered public x_out should fail with an x_out-specific error, got `{err}`"
    );

    Ok(())
}
