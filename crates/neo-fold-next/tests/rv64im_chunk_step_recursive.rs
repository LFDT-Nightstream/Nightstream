use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CcsWitness, CeClaim};
use neo_fold_next::proof::FoldSchedule;
use neo_fold_next::rv64im::audit::{
    build_rv64im_chunk_step_ivc_recursive_step_cover_shape, build_rv64im_chunk_step_ivc_recursive_step_padding,
    build_rv64im_chunk_step_ivc_relations, build_rv64im_main_recursion_f_prime_advices,
    build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape,
    build_rv64im_main_recursion_f_prime_claim_cover, build_rv64im_main_recursion_f_prime_payload,
    build_rv64im_main_recursion_f_prime_payloads, build_rv64im_main_recursion_f_prime_payloads_with_spartan_shape,
    build_rv64im_main_recursion_step_spartan_shape,
    debug_check_rv64im_chunk_step_recursive_effective_chunk_trace_matches_native,
    evaluate_rv64im_main_recursion_f_prime_advice, Rv64imChunkStepIvcRelation,
};
use neo_fold_next::rv64im::final_relation::prove_rv64im_final_statement_from_accepted;
use neo_fold_next::rv64im::{
    build_rv64im_accepted_proof_artifact, parity_source_cases, prove_rv64im_public_proof_with_options,
    Rv64imProofInput, Rv64imPublicProofOptions,
};
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

fn multi_chunk_relations() -> Vec<Rv64imChunkStepIvcRelation> {
    let source = parity_source_cases()
        .into_iter()
        .find(|case| case.manifest.name == "control_flow_jal_skip_ecall")
        .expect("control-flow parity source case");
    let input = Rv64imProofInput {
        max_steps: source.program_words.len(),
        source,
    };
    let options = Rv64imPublicProofOptions {
        root_fold_schedule: FoldSchedule::RowsPerChunk(1),
    };
    let public_proof = prove_rv64im_public_proof_with_options(&input, options).expect("prove chunked public proof");
    let accepted_artifact = build_rv64im_accepted_proof_artifact(&public_proof).expect("build accepted artifact");
    let (final_statement, final_proof) =
        prove_rv64im_final_statement_from_accepted(&accepted_artifact).expect("build final statement");
    let relations =
        build_rv64im_chunk_step_ivc_relations(&final_statement, &final_proof).expect("build chunk-step IVC relations");
    assert!(relations.len() > 1, "expected multiple chunk-step relations");
    relations
}

fn commitment_is_zero(commitment: &Commitment) -> bool {
    commitment.data.iter().all(|value| *value == F::ZERO)
}

fn ce_claim_is_zero(claim: &CeClaim<Commitment, F, K>) -> bool {
    commitment_is_zero(&claim.c)
        && claim.X.as_slice().iter().all(|value| *value == F::ZERO)
        && claim.r.iter().all(|value| *value == K::ZERO)
        && claim.s_col.iter().all(|value| *value == K::ZERO)
        && claim
            .y_ring
            .iter()
            .all(|row| row.iter().all(|value| *value == K::ZERO))
        && claim.ct.iter().all(|value| *value == K::ZERO)
        && claim.aux_openings.iter().all(|value| *value == K::ZERO)
        && claim.y_zcol.iter().all(|value| *value == K::ZERO)
        && claim.fold_digest == [0; 32]
        && claim.c_step_coords.iter().all(|value| *value == F::ZERO)
}

fn ce_claim_has_zero_semantics_with_shared_point(
    claim: &CeClaim<Commitment, F, K>,
    shared_r: &[K],
    shared_s_col: &[K],
) -> bool {
    commitment_is_zero(&claim.c)
        && claim.X.as_slice().iter().all(|value| *value == F::ZERO)
        && claim.r == shared_r
        && claim.s_col == shared_s_col
        && claim
            .y_ring
            .iter()
            .all(|row| row.iter().all(|value| *value == K::ZERO))
        && claim.ct.iter().all(|value| *value == K::ZERO)
        && claim.aux_openings.iter().all(|value| *value == K::ZERO)
        && claim.y_zcol.iter().all(|value| *value == K::ZERO)
        && claim.fold_digest == [0; 32]
        && claim.c_step_coords.iter().all(|value| *value == F::ZERO)
}

fn ccs_claim_is_zero(claim: &CcsClaim<Commitment, F>) -> bool {
    commitment_is_zero(&claim.c) && claim.x.iter().all(|value| *value == F::ZERO)
}

fn ccs_witness_is_zero(witness: &CcsWitness<F>) -> bool {
    witness.w.iter().all(|value| *value == F::ZERO) && witness.Z.as_slice().iter().all(|value| *value == F::ZERO)
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_chunk_step_recursive_payload_matches_cover_shape_for_multi_step_chain() {
    let relations = multi_chunk_relations();
    let advices = build_rv64im_main_recursion_f_prime_advices(&relations).expect("build main recursion F' advices");
    let cover_shape =
        build_rv64im_chunk_step_ivc_recursive_step_cover_shape(&relations).expect("build recursive-step cover shape");
    let claim_cover =
        build_rv64im_main_recursion_f_prime_claim_cover(&advices).expect("build recursive-step claim cover");
    let spartan_shape =
        build_rv64im_main_recursion_step_spartan_shape(&relations).expect("build recursive-step spartan shape");
    for (relation, advice) in relations.iter().zip(advices.iter()) {
        let payload = build_rv64im_main_recursion_f_prime_payload(advice, &cover_shape, &claim_cover)
            .expect("build recursive-step payload");
        let expected_padding =
            build_rv64im_chunk_step_ivc_recursive_step_padding(&relation.statement, &relation.witness, &cover_shape)
                .expect("build recursive-step padding");

        assert_eq!(payload.cover_shape, cover_shape);
        assert_eq!(payload.padding, expected_padding);
        assert!(
            payload.matches_cover_shape(),
            "recursive-step payload must materialize the canonical cover shape"
        );
        assert!(spartan_shape.matches_payload(&payload));
        assert_eq!(
            payload.padded_fresh_claim_count(),
            cover_shape.fresh_claim_count as usize
        );
        assert_eq!(
            payload.effective_fresh_claim_count(),
            payload.step_shape.fresh_claim_count as usize
        );
    }
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_chunk_step_recursive_payload_zero_pads_tail_slots() {
    let relations = multi_chunk_relations();
    let advices = build_rv64im_main_recursion_f_prime_advices(&relations).expect("build main recursion F' advices");
    let cover_shape =
        build_rv64im_chunk_step_ivc_recursive_step_cover_shape(&relations).expect("build recursive-step cover shape");
    let claim_cover =
        build_rv64im_main_recursion_f_prime_claim_cover(&advices).expect("build recursive-step claim cover");
    let spartan_shape =
        build_rv64im_main_recursion_step_spartan_shape(&relations).expect("build recursive-step spartan shape");
    for advice in &advices {
        let payload = build_rv64im_main_recursion_f_prime_payload(advice, &cover_shape, &claim_cover)
            .expect("build recursive-step payload");
        assert!(spartan_shape.matches_payload(&payload));
        let step_shape = &payload.step_shape;
        for claim in payload
            .fresh_claims
            .iter()
            .skip(step_shape.fresh_claim_count as usize)
        {
            assert!(ccs_claim_is_zero(claim), "fresh-claim tail padding must be zeroed");
        }
        assert_eq!(
            payload.effective_fresh_claim_count(),
            step_shape.fresh_claim_count as usize
        );
        assert_eq!(
            payload.padded_fresh_claim_count(),
            cover_shape.fresh_claim_count as usize
        );
        for witness in payload
            .fresh_witnesses
            .iter()
            .skip(step_shape.fresh_witness_count as usize)
        {
            assert!(
                ccs_witness_is_zero(witness),
                "fresh-witness tail padding must be zeroed"
            );
        }
        for claim in payload
            .pi_ccs
            .ccs_outputs
            .iter()
            .skip(step_shape.ccs_output_count as usize)
        {
            assert!(
                ce_claim_has_zero_semantics_with_shared_point(
                    claim,
                    &payload.pi_ccs.ccs_outputs[0].r,
                    &payload.pi_ccs.ccs_outputs[0].s_col,
                ),
                "CCS-output tail padding must keep the shared point but zero the semantic payload"
            );
        }
        for claim in payload
            .pi_dec
            .children
            .iter()
            .skip(step_shape.child_count as usize)
        {
            assert!(ce_claim_is_zero(claim), "child tail padding must be zeroed");
        }
        for (idx, round) in payload.pi_ccs.replay.sumcheck_rounds.iter().enumerate() {
            let live_coeffs = step_shape.fe_round_lengths.get(idx).copied().unwrap_or(0) as usize;
            assert!(
                round
                    .iter()
                    .skip(live_coeffs)
                    .all(|value| *value == K::ZERO),
                "FE round tail padding must be zeroed"
            );
        }
        for (idx, round) in payload.pi_ccs.replay.sumcheck_rounds_nc.iter().enumerate() {
            let live_coeffs = step_shape.nc_round_lengths.get(idx).copied().unwrap_or(0) as usize;
            assert!(
                round
                    .iter()
                    .skip(live_coeffs)
                    .all(|value| *value == K::ZERO),
                "NC round tail padding must be zeroed"
            );
        }
    }
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_chunk_step_recursive_payload_batch_builder_matches_single_step_builder() {
    let relations = multi_chunk_relations();
    let advices = build_rv64im_main_recursion_f_prime_advices(&relations).expect("build main recursion F' advices");
    let cover_shape =
        build_rv64im_chunk_step_ivc_recursive_step_cover_shape(&relations).expect("build recursive-step cover shape");
    let claim_cover =
        build_rv64im_main_recursion_f_prime_claim_cover(&advices).expect("build recursive-step claim cover");
    let spartan_shape =
        build_rv64im_main_recursion_step_spartan_shape(&relations).expect("build recursive-step spartan shape");
    let batch_payloads =
        build_rv64im_main_recursion_f_prime_payloads(&advices, &spartan_shape).expect("build batched payloads");
    let (derived_spartan_shape, derived_payloads) =
        build_rv64im_main_recursion_f_prime_payloads_with_spartan_shape(&relations)
            .expect("build batched payloads with derived spartan shape");

    assert_eq!(derived_spartan_shape, spartan_shape);
    assert_eq!(batch_payloads.len(), relations.len());
    assert_eq!(derived_payloads.len(), relations.len());

    for ((advice, batched), derived) in advices
        .iter()
        .zip(batch_payloads.iter())
        .zip(derived_payloads.iter())
    {
        let single = build_rv64im_main_recursion_f_prime_payload(advice, &cover_shape, &claim_cover)
            .expect("build single recursive-step payload");

        assert_eq!(batched.cover_shape, cover_shape);
        assert_eq!(batched.step_shape, single.step_shape);
        assert_eq!(batched.padding, single.padding);
        assert_eq!(batched.state_in_claims.len(), single.state_in_claims.len());
        assert_eq!(batched.state_out_claims.len(), single.state_out_claims.len());
        assert_eq!(batched.fresh_claims.len(), single.fresh_claims.len());
        assert_eq!(batched.fresh_witnesses.len(), single.fresh_witnesses.len());
        assert_eq!(batched.pi_ccs.ccs_outputs.len(), single.pi_ccs.ccs_outputs.len());
        assert_eq!(batched.pi_rlc, single.pi_rlc);
        assert_eq!(batched.pi_dec.children.len(), single.pi_dec.children.len());
        assert_eq!(batched.pi_ccs.replay, single.pi_ccs.replay);

        assert_eq!(derived.step_shape, single.step_shape);
        assert_eq!(derived.padding, single.padding);
        assert_eq!(derived.pi_ccs, single.pi_ccs);
        assert_eq!(derived.pi_rlc, single.pi_rlc);
        assert_eq!(derived.pi_dec, single.pi_dec);
    }
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_chunk_step_recursive_claim_cover_dominates_multi_step_chain() {
    let relations = multi_chunk_relations();
    let advices = build_rv64im_main_recursion_f_prime_advices(&relations).expect("build main recursion F' advices");
    let claim_cover =
        build_rv64im_main_recursion_f_prime_claim_cover(&advices).expect("build recursive-step claim cover");

    for relation in &relations {
        assert!(
            claim_cover.covers_relation(relation),
            "recursive-step claim cover must dominate every carried-claim surface in the chain"
        );
    }
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_chunk_step_recursive_backend_relations_match_native_step_and_payload_builders() {
    let relations = multi_chunk_relations();
    let advices = build_rv64im_main_recursion_f_prime_advices(&relations).expect("build main recursion F' advices");
    let (spartan_shape, backend_relations) =
        build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape(&relations)
            .expect("build recursive-step backend relations");
    let batch_payloads =
        build_rv64im_main_recursion_f_prime_payloads(&advices, &spartan_shape).expect("build batched payloads");

    assert_eq!(backend_relations.len(), relations.len());
    assert_eq!(batch_payloads.len(), relations.len());

    for ((relation, backend_relation), payload) in relations
        .iter()
        .zip(backend_relations.iter())
        .zip(batch_payloads.iter())
    {
        let _step_output = evaluate_rv64im_main_recursion_f_prime_advice(&backend_relation.f_prime_advice)
            .expect("evaluate native recursive step");

        assert_ne!(backend_relation.spartan_statement.x_out.bytes(), [0; 32]);
        assert_ne!(
            backend_relation.f_prime_advice.step_statement_digest(),
            backend_relation.f_prime_advice.fresh_instance_digest(),
            "fresh-instance digest must stay distinct from the legacy transition step-statement digest"
        );
        assert_eq!(
            backend_relation.f_prime_advice.bridge_handoff_digest(),
            relation.witness.handoff.bridge_handoff.digest
        );

        assert_eq!(backend_relation.payload.cover_shape, spartan_shape.cover_shape);
        assert_eq!(backend_relation.payload.step_shape, payload.step_shape);
        assert_eq!(backend_relation.payload.padding, payload.padding);
        assert_eq!(backend_relation.payload.pi_ccs, payload.pi_ccs);
        assert_eq!(backend_relation.payload.pi_rlc, payload.pi_rlc);
        assert_eq!(backend_relation.payload.pi_dec, payload.pi_dec);
        assert!(
            spartan_shape.matches_payload(&backend_relation.payload),
            "backend relation payload must already satisfy the canonical recursive cover shape"
        );
    }
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_chunk_step_recursive_effective_chunk_trace_matches_native_trace() {
    let relations = multi_chunk_relations();
    let (_, backend_relations) = build_rv64im_main_recursion_f_prime_backend_relations_with_spartan_shape(&relations)
        .expect("build recursive-step backend relations");

    for backend_relation in &backend_relations {
        debug_check_rv64im_chunk_step_recursive_effective_chunk_trace_matches_native(backend_relation)
            .expect("effective chunk trace recovered from padded payload must match native trace");
    }
}
