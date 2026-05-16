#[path = "support/rv32im_main_recursion_step_spartan_exact.rs"]
mod rv32im_main_recursion_step_spartan_exact_support;
#[path = "support/rv32im_n2.rs"]
mod rv32im_n2_support;

use std::sync::OnceLock;

use neo_fold_prototype::core::proof::FoldSchedule;
use neo_fold_prototype::rv32im::audit::{
    audit_rv32im_main_recursion_step_spartan_published_target_matches_construction2_state_images,
    build_rv32im_chunk_step_ivc_relations, build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape,
    build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices,
    build_rv32im_main_recursion_f_prime_payload, build_rv32im_main_recursion_step_authoritative_chunk_surface,
    build_rv32im_main_recursion_step_spartan_published_target,
    debug_check_rv32im_main_recursion_f_prime_backend_relation_semantics,
    debug_check_rv32im_main_recursion_step_authoritative_chunk_surface_matches_native,
    debug_check_rv32im_main_recursion_step_spartan_chunk_replay_surface,
    debug_check_rv32im_main_recursion_step_spartan_circuit,
    debug_check_rv32im_main_recursion_step_spartan_embedded_body,
    debug_check_rv32im_main_recursion_step_spartan_fresh_output_accumulator_digest_parity,
    debug_check_rv32im_main_recursion_step_spartan_inactive_side_lane_constraints,
    debug_check_rv32im_main_recursion_step_spartan_live_claim_me_digest_parity,
    debug_check_rv32im_main_recursion_step_spartan_pi_ccs_replay_lengths,
    debug_check_rv32im_main_recursion_x_out_gadget_parity,
    debug_measure_rv32im_main_recursion_step_spartan_circuit_shape,
    rv32im_main_recursion_advice_tamper_side_witness_nonzero,
    rv32im_main_recursion_backend_relation_tamper_payload_chunk_digest_shell, Rv32imMainRecursionFPrimeBackendRelation,
    Rv32imMainRecursionStepSpartanError, Rv32imMainRecursionStepSpartanShape,
};
use neo_fold_prototype::rv32im::final_relation::prove_rv32im_final_statement_from_accepted;
use neo_fold_prototype::rv32im::{
    build_mixed_opcode_perf_source_case, build_rv32im_accepted_proof_artifact,
    build_rv32im_main_recursion_f_prime_advices, build_rv32im_main_recursion_f_prime_advices_with_side_opening_public,
    evaluate_rv32im_main_recursion_f_prime_advice, f_prime::RV32IM_MAIN_RECURSION_TRIVIAL_PC,
    prove_rv32im_public_proof_with_options, Rv32imProofInput, Rv32imPublicProofOptions,
};
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;
use rv32im_main_recursion_step_spartan_exact_support::{
    assert_backend_relation_exact_surface_contract, single_relation_backend_fixture,
};

fn chunk_step_relations_fixture() -> Vec<neo_fold_prototype::rv32im::audit::Rv32imChunkStepIvcRelation> {
    static FIXTURE: OnceLock<Vec<neo_fold_prototype::rv32im::audit::Rv32imChunkStepIvcRelation>> = OnceLock::new();
    FIXTURE
        .get_or_init(|| {
            let source = build_mixed_opcode_perf_source_case(1);
            let input = Rv32imProofInput {
                max_steps: source.program_words.len(),
                source,
            };
            let options = Rv32imPublicProofOptions {
                root_fold_schedule: FoldSchedule::RowsPerChunk(1),
            };
            let public_proof =
                prove_rv32im_public_proof_with_options(&input, options).expect("prove chunked public proof");
            let accepted_artifact =
                build_rv32im_accepted_proof_artifact(&public_proof).expect("build accepted artifact");
            let (final_statement, final_proof) =
                prove_rv32im_final_statement_from_accepted(&accepted_artifact).expect("build final statement");
            let relations = build_rv32im_chunk_step_ivc_relations(&final_statement, &final_proof)
                .expect("build chunk-step IVC relations");
            assert!(relations.len() > 1, "expected multiple chunk-step relations");
            relations
        })
        .clone()
}

fn backend_relations_fixture() -> (
    Rv32imMainRecursionStepSpartanShape,
    Vec<Rv32imMainRecursionFPrimeBackendRelation>,
) {
    static FIXTURE: OnceLock<(
        Rv32imMainRecursionStepSpartanShape,
        Vec<Rv32imMainRecursionFPrimeBackendRelation>,
    )> = OnceLock::new();
    FIXTURE
        .get_or_init(|| {
            let relations = chunk_step_relations_fixture();
            build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape(&relations)
                .expect("build recursive-step backend relations")
        })
        .clone()
}

fn backend_relations_prefix_fixture(
    prefix_len: usize,
) -> (
    Rv32imMainRecursionStepSpartanShape,
    Vec<Rv32imMainRecursionFPrimeBackendRelation>,
) {
    let relations = chunk_step_relations_fixture();
    let prefix_len = prefix_len.min(relations.len());
    build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape(&relations[..prefix_len])
        .expect("build recursive-step backend relations prefix")
}

fn side_aware_backend_relations_prefix_fixture(
    prefix_len: usize,
) -> (
    Rv32imMainRecursionStepSpartanShape,
    Vec<Rv32imMainRecursionFPrimeBackendRelation>,
) {
    let relations = chunk_step_relations_fixture();
    let prefix_len = prefix_len.min(relations.len());
    let fixture = rv32im_n2_support::build_rv32im_n2_fixture().expect("build rv32im n=2 fixture");
    let advices = build_rv32im_main_recursion_f_prime_advices_with_side_opening_public(
        &relations[..prefix_len],
        fixture.side_proof.opening_public(),
    )
    .expect("build side-aware recursion advices");
    build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices(
        &relations[..prefix_len],
        &advices,
    )
    .expect("build side-aware recursive-step backend relations prefix")
}

fn tamper_state_out_claim_projection_shell(claim: &mut neo_ccs::CeClaim<neo_ajtai::Commitment, F, K>) {
    if claim.X.rows() > 1 && claim.X.cols() > 0 {
        claim.X[(1, 0)] += F::ONE;
    }
    if let Some(first) = claim.s_col.first_mut() {
        *first += K::ONE;
    }
    if let Some(first) = claim.ct.first_mut() {
        *first += K::ONE;
    }
    if let Some(first) = claim.aux_openings.first_mut() {
        *first += K::ONE;
    }
    if let Some(first) = claim.y_zcol.first_mut() {
        *first += K::ONE;
    }
    if let Some(first) = claim.c_step_coords.first_mut() {
        *first += F::ONE;
    }
    claim.fold_digest[0] ^= 1;
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv32im_main_recursion_step_spartan_multi_step_backend_relations_build() {
    let (_, backend_relations) = backend_relations_fixture();
    assert!(
        backend_relations.len() > 1,
        "expected multi-step recursive backend relations"
    );
}

#[test]
fn rv32im_main_recursion_step_spartan_embedded_body_rejects_payload_chunk_relation_digest_shell() {
    let (spartan_shape, backend_relations) = single_relation_backend_fixture();
    let mut tampered_relation = backend_relations
        .first()
        .expect("first backend relation")
        .clone();
    rv32im_main_recursion_backend_relation_tamper_payload_chunk_digest_shell(&mut tampered_relation);

    let err = debug_check_rv32im_main_recursion_step_spartan_embedded_body(&spartan_shape, &tampered_relation)
        .expect_err("recursive-step embedded body must reject a tampered chunk-relation digest");
    assert!(format!("{err}").contains("chunk_relation_digest"));
}

#[test]
fn rv32im_main_recursion_step_spartan_embedded_body_ignores_payload_state_out_claim_projection_shell() {
    let (spartan_shape, backend_relations) = single_relation_backend_fixture();
    let mut tampered_relation = backend_relations
        .first()
        .expect("first backend relation")
        .clone();
    let first_claim = tampered_relation
        .payload
        .state_out_claims
        .first_mut()
        .expect("state_out claim");
    tamper_state_out_claim_projection_shell(first_claim);

    debug_check_rv32im_main_recursion_step_spartan_embedded_body(&spartan_shape, &tampered_relation).expect(
        "recursive-step embedded body must ignore non-authoritative state_out claim baggage outside the CE projection",
    );
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv32im_main_recursion_step_spartan_live_ccs_outputs_match_authoritative_f_surfaces() {
    let (_, backend_relations) = backend_relations_fixture();
    let first = backend_relations.first().expect("first backend relation");
    let fresh_count = first.payload.step_shape.fresh_claim_count as usize;
    let live_output_count = first.payload.step_shape.ccs_output_count as usize;

    for (idx, output) in first
        .payload
        .pi_ccs
        .ccs_outputs
        .iter()
        .take(live_output_count)
        .enumerate()
    {
        if idx < fresh_count {
            let fresh = &first.payload.fresh_claims[idx];
            assert_eq!(
                output.c.data.len(),
                fresh.c.data.len(),
                "fresh output commitment width drift at slot {idx}"
            );
            assert_eq!(output.X.rows(), D, "fresh output row count drift at slot {idx}");
            assert_eq!(
                output.X.cols(),
                fresh.m_in,
                "fresh output column count drift at slot {idx}"
            );
            assert_eq!(
                output.X.as_slice().len(),
                D * fresh.m_in,
                "fresh output flattened X width drift at slot {idx}"
            );
        } else {
            let carried_idx = idx - fresh_count;
            let carried = &first.payload.state_in_claims[carried_idx];
            assert_eq!(
                output.c.data.len(),
                carried.c.data.len(),
                "ME-input output commitment width drift at slot {idx}"
            );
            assert_eq!(
                output.X.rows(),
                carried.X.rows(),
                "ME-input output row count drift at slot {idx}"
            );
            assert_eq!(
                output.X.cols(),
                carried.X.cols(),
                "ME-input output column count drift at slot {idx}"
            );
            assert_eq!(
                output.X.as_slice().len(),
                carried.X.as_slice().len(),
                "ME-input output flattened X width drift at slot {idx}"
            );
        }
    }
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv32im_main_recursion_step_spartan_live_ccs_outputs_match_rlc_parent_surface() {
    let (_, backend_relations) = backend_relations_fixture();
    let first = backend_relations.first().expect("first backend relation");
    let live_output_count = first.payload.step_shape.ccs_output_count as usize;
    let parent = &first.payload.pi_rlc.parent;

    for (idx, output) in first
        .payload
        .pi_ccs
        .ccs_outputs
        .iter()
        .take(live_output_count)
        .enumerate()
    {
        assert_eq!(output.m_in, parent.m_in, "RLC m_in drift at slot {idx}");
        assert_eq!(output.X.rows(), parent.X.rows(), "RLC X row drift at slot {idx}");
        assert_eq!(output.X.cols(), parent.X.cols(), "RLC X col drift at slot {idx}");
        assert_eq!(
            output.c.data.len(),
            parent.c.data.len(),
            "RLC commitment width drift at slot {idx}"
        );
        assert_eq!(output.r, parent.r, "RLC row-challenge surface drift at slot {idx}");
        assert_eq!(
            output.s_col, parent.s_col,
            "RLC col-challenge surface drift at slot {idx}"
        );
        assert_eq!(
            output.y_ring.len(),
            parent.y_ring.len(),
            "RLC y_ring row-count drift at slot {idx}"
        );
        assert_eq!(output.ct.len(), parent.ct.len(), "RLC ct width drift at slot {idx}");
        assert_eq!(
            output.aux_openings.len(),
            parent.aux_openings.len(),
            "RLC aux-openings width drift at slot {idx}"
        );
        assert_eq!(
            output.y_zcol.len(),
            parent.y_zcol.len(),
            "RLC y_zcol width drift at slot {idx}"
        );
    }
}

#[test]
fn rv32im_main_recursion_step_spartan_prefix_x_out_gadget_parity_holds() {
    let (_, backend_relations) = backend_relations_prefix_fixture(1);
    let first = backend_relations.first().expect("first backend relation");
    debug_check_rv32im_main_recursion_x_out_gadget_parity(first)
        .expect("prefix first-step x_out gadget should match the canonical native F' image");
}

#[test]
fn rv32im_main_recursion_step_spartan_prefix_fresh_output_accumulator_digest_parity_holds() {
    let (_, backend_relations) = backend_relations_prefix_fixture(1);
    let first = backend_relations.first().expect("first backend relation");
    debug_check_rv32im_main_recursion_step_spartan_fresh_output_accumulator_digest_parity(first).expect(
        "prefix first-step fresh output accumulator digest should match the native recursive accumulator digest",
    );
}

#[test]
#[ignore = "diagnostic baseline during canonical fixed-transcript localization; re-enable after the canonical padded payload derives fixed_transcript_out cleanly from the shared chunk body"]
fn rv32im_main_recursion_step_spartan_canonical_live_claim_me_digest_parity_holds() {
    let (_, backend_relations) = backend_relations_fixture();
    let first = backend_relations.first().expect("first backend relation");
    debug_check_rv32im_main_recursion_step_spartan_live_claim_me_digest_parity(first)
        .expect("canonical first-step live carried claims should hash to the authoritative native ME digests");
}

#[test]
#[ignore = "diagnostic baseline during canonical fixed-transcript localization; re-enable after the canonical padded payload derives fixed_transcript_out cleanly from the shared chunk body"]
fn rv32im_main_recursion_step_spartan_canonical_payload_localizes_first_failing_step() {
    let raw_relations = chunk_step_relations_fixture();
    let spartan_shape =
        neo_fold_prototype::rv32im::audit::build_rv32im_main_recursion_step_spartan_shape(&raw_relations)
            .expect("build canonical recursive-step shape");
    let advices =
        build_rv32im_main_recursion_f_prime_advices(&raw_relations).expect("build raw recursive-step advices");

    for (idx, advice) in advices.iter().enumerate() {
        if let Err(err) = build_rv32im_main_recursion_f_prime_payload(advice, &spartan_shape) {
            panic!("first canonical payload failure at step {idx}: {err}");
        }
    }
}

#[test]
fn rv32im_main_recursion_step_spartan_first_advice_matches_raw_relations_prefix() {
    let raw_relations = chunk_step_relations_fixture();
    let raw_advices =
        build_rv32im_main_recursion_f_prime_advices(&raw_relations).expect("build raw recursive-step advices");
    let (_, backend_relations) = single_relation_backend_fixture();
    let exact_first = backend_relations.first().expect("first backend relation");
    let raw_first = raw_advices.first().expect("first raw advice");

    assert_eq!(
        raw_first.running_state().carry.main.claims,
        exact_first.f_prime_advice.running_state().carry.main.claims,
        "first carried-state claims drifted between the raw multi-step advice path and the exact one-step backend fixture"
    );
    assert_eq!(
        raw_first.running_state().transcript,
        exact_first.f_prime_advice.running_state().transcript,
        "first transcript input drifted between the raw multi-step advice path and the exact one-step backend fixture"
    );
}

#[test]
fn rv32im_main_recursion_step_spartan_payload_owns_explicit_z_pc_semantics() {
    let (_, backend_relations) = single_relation_backend_fixture();
    let first = backend_relations.first().expect("first backend relation");

    assert_eq!(
        first.payload.z_0(),
        first.f_prime_advice.z_0(),
        "payload z_0 drifted from the native F' advice"
    );
    assert_eq!(
        first.payload.z_i(),
        first.f_prime_advice.z_i(),
        "payload z_i drifted from the native F' advice"
    );
    assert_eq!(
        first.payload.z_next(),
        &first
            .f_prime_advice
            .fresh_state_out()
            .carry
            .terminal_handle
            .0,
        "payload z_next drifted from the authoritative native state_out terminal handle"
    );
    assert_eq!(
        first.payload.pc_i(),
        first.f_prime_advice.pc_i(),
        "payload pc_i drifted from the native F' advice"
    );
    assert_eq!(
        first.payload.pc_next(),
        RV32IM_MAIN_RECURSION_TRIVIAL_PC,
        "payload pc_next drifted from the trivial RV32IM control lane"
    );
    assert!(
        first.f_prime_advice.side_witness().is_zero(),
        "exact backend relation must currently carry a zero side lane"
    );
}

#[test]
fn rv32im_main_recursion_step_spartan_authoritative_chunk_surface_matches_native_first_step() {
    let (_, backend_relations) = single_relation_backend_fixture();
    let first = backend_relations.first().expect("first backend relation");

    debug_check_rv32im_main_recursion_step_authoritative_chunk_surface_matches_native(first)
        .expect("authoritative chunk surface should match the native chunk replay theorem surface");
    let authoritative_surface = build_rv32im_main_recursion_step_authoritative_chunk_surface(first)
        .expect("build carried authoritative chunk surface");

    assert_eq!(
        authoritative_surface.chunk_index,
        first.f_prime_advice.chunk_index(),
        "authoritative chunk surface chunk_index drifted from the native F' advice"
    );
    assert_eq!(
        authoritative_surface.state_out_claim_count, first.payload.step_shape.state_out_claim_count,
        "authoritative chunk surface state_out claim count drifted from the live step shape"
    );
    assert_eq!(
        authoritative_surface.child_claim_count, first.payload.step_shape.child_count,
        "authoritative chunk surface child-claim count drifted from the live replay surface"
    );
}

#[test]
fn rv32im_main_recursion_step_spartan_authoritative_chunk_surface_rejects_tampered_state_out_claim() {
    let (_, backend_relations) = single_relation_backend_fixture();
    let mut backend_relation = backend_relations
        .first()
        .expect("first backend relation")
        .clone();
    let first_data = backend_relation
        .payload
        .state_out_claims
        .first_mut()
        .and_then(|claim| claim.c.data.first_mut())
        .expect("first state_out claim should expose commitment data");
    *first_data += F::ONE;

    let err = debug_check_rv32im_main_recursion_step_authoritative_chunk_surface_matches_native(&backend_relation)
        .expect_err("tampered carried state_out claim must fail authoritative chunk surface parity");
    assert!(
        err.to_string().contains("authoritative chunk surface"),
        "unexpected authoritative chunk surface parity error: {err}",
    );
}

#[test]
fn rv32im_main_recursion_step_spartan_inactive_side_lane_constraints_hold() {
    let (_, backend_relations) = single_relation_backend_fixture();
    let first = backend_relations.first().expect("first backend relation");
    debug_check_rv32im_main_recursion_step_spartan_inactive_side_lane_constraints(first)
        .expect("exact first-step inactive side-lane constraints should hold");
}

#[test]
fn rv32im_main_recursion_step_spartan_rejects_nonzero_side_witness() {
    let (_, backend_relations) = single_relation_backend_fixture();
    let mut backend_relation = backend_relations
        .first()
        .expect("first backend relation")
        .clone();
    rv32im_main_recursion_advice_tamper_side_witness_nonzero(&mut backend_relation.f_prime_advice);

    let err = debug_check_rv32im_main_recursion_step_spartan_inactive_side_lane_constraints(&backend_relation)
        .expect_err("non-zero side_witness must fail inactive side-lane constraints");
    assert!(matches!(err, Rv32imMainRecursionStepSpartanError::Prepare(_)));
}

#[test]
fn rv32im_main_recursion_step_spartan_payload_carries_authoritative_phi_side_words() {
    let relations = chunk_step_relations_fixture();
    let fixture = rv32im_n2_support::build_rv32im_n2_fixture().expect("build rv32im n=2 fixture");
    let side_advices = build_rv32im_main_recursion_f_prime_advices_with_side_opening_public(
        &relations,
        fixture.side_proof.opening_public(),
    )
    .expect("build side-aware recursion advices");

    let (_, backend_relations) = build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices(
        &relations,
        &side_advices,
    )
    .expect("build side-aware recursive-step backend relations");

    assert_eq!(backend_relations.len(), side_advices.len());
    for (step_index, (backend_relation, advice)) in backend_relations
        .iter()
        .zip(side_advices.iter())
        .enumerate()
    {
        assert!(
            !advice.phi_side().is_zero(),
            "step {step_index}: side-aware advice should expose non-zero authoritative phi_side"
        );
        assert_eq!(
            backend_relation.payload.phi_side_commitment_words(),
            advice.phi_side().commitment_words(),
            "step {step_index}: recursive-step payload must carry authoritative phi_side commitment words"
        );
        assert!(
            backend_relation.payload.matches_explicit_semantics(advice),
            "step {step_index}: payload explicit semantics must include authoritative phi_side words"
        );
    }
}

#[test]
fn rv32im_main_recursion_step_spartan_side_aware_statements_match_zero_side_for_two_steps() {
    let (_, zero_backend_relations) = backend_relations_prefix_fixture(2);
    let (_, side_backend_relations) = side_aware_backend_relations_prefix_fixture(2);

    assert_eq!(
        side_backend_relations.len(),
        zero_backend_relations.len(),
        "expected side-aware and zero-side fixtures to cover the same chain length"
    );
    for (step_index, (zero_relation, side_relation)) in zero_backend_relations
        .iter()
        .zip(side_backend_relations.iter())
        .enumerate()
    {
        assert!(
            !side_relation.f_prime_advice.phi_side().is_zero(),
            "step {step_index}: side-aware backend should carry a non-zero authoritative phi_side"
        );
        assert_eq!(
            side_relation.spartan_statement, zero_relation.spartan_statement,
            "step {step_index}: authoritative phi_side must not change the canonical recursive-step Spartan statement"
        );
    }
}

#[test]
fn rv32im_main_recursion_step_spartan_rejects_tampered_per_step_statement_binding() {
    let (cover_shape, backend_relations) = single_relation_backend_fixture();
    let mut backend_relation = backend_relations
        .first()
        .expect("first backend relation")
        .clone();
    backend_relation.spartan_statement.folded_accumulator_digest[0] ^= 1;

    let err = debug_check_rv32im_main_recursion_step_spartan_circuit(&cover_shape, &backend_relation)
        .expect_err("tampered per-step Spartan statement must fail recursive-step builder binding");
    assert!(
        err.to_string()
            .contains("canonical per-step Spartan statement derived from native F'"),
        "unexpected per-step statement binding error: {err}",
    );
}

#[test]
fn rv32im_main_recursion_step_spartan_backend_relation_semantics_reject_tampered_statement() {
    let (_, backend_relations) = single_relation_backend_fixture();
    let mut backend_relation = backend_relations
        .first()
        .expect("first backend relation")
        .clone();
    backend_relation.spartan_statement.folded_accumulator_digest[0] ^= 1;

    let err = debug_check_rv32im_main_recursion_f_prime_backend_relation_semantics(&backend_relation)
        .expect_err("tampered backend relation statement must fail semantic preflight");
    assert!(
        err.to_string()
            .contains("canonical per-step Spartan statement derived from native F'"),
        "unexpected backend relation semantic error: {err}",
    );
}

#[test]
fn rv32im_main_recursion_step_spartan_state_out_claims_match_replayed_children_surface() {
    let (_, backend_relations) = single_relation_backend_fixture();
    let first = backend_relations.first().expect("first backend relation");
    let state_out_count = first.payload.step_shape.state_out_claim_count as usize;
    let child_count = first.payload.step_shape.child_count as usize;

    assert_eq!(
        state_out_count, child_count,
        "single-step exact fixture should carry exactly the replayed child claims"
    );

    for (idx, (state_out, child)) in first
        .payload
        .state_out_claims
        .iter()
        .take(state_out_count)
        .zip(first.payload.pi_dec.children.iter().take(child_count))
        .enumerate()
    {
        assert_eq!(
            state_out, child,
            "state_out claim drifted from replayed child surface at slot {idx}"
        );
    }
}

#[test]
fn rv32im_main_recursion_step_spartan_first_state_out_claim_preserves_ct_alias() {
    let (_, backend_relations) = single_relation_backend_fixture();
    let first = backend_relations.first().expect("first backend relation");
    let claim = first
        .payload
        .state_out_claims
        .first()
        .expect("first state_out claim");

    assert!(
        claim.y_ring.len() >= claim.ct.len(),
        "state_out first claim must have at least one y_ring row per ct entry"
    );
    for (idx, ct_value) in claim.ct.iter().enumerate() {
        assert_eq!(
            claim.y_ring[idx].first().copied(),
            Some(*ct_value),
            "state_out first claim lost ct/y_ring alias at row {idx}"
        );
    }
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv32im_main_recursion_step_spartan_canonical_me_output_shapes_match_state_in_shapes() {
    let (cover_shape, backend_relations) = backend_relations_fixture();
    let first = backend_relations.first().expect("first backend relation");
    let fresh_count = first.payload.step_shape.fresh_claim_count as usize;
    let live_output_count = first.payload.step_shape.ccs_output_count as usize;
    let me_output_count = live_output_count.saturating_sub(fresh_count);

    assert!(
        me_output_count <= cover_shape.claim_cover.state_in_claim_shapes.len(),
        "canonical cover must have enough carried-input claim slots for carried outputs"
    );

    for me_idx in 0..me_output_count {
        let output_idx = fresh_count + me_idx;
        assert_eq!(
            cover_shape.claim_cover.ccs_output_shapes[output_idx],
            cover_shape.claim_cover.state_in_claim_shapes[me_idx],
            "canonical carried-output shape drift at carried slot {me_idx}"
        );
    }
}

#[test]
fn rv32im_main_recursion_step_spartan_zero_claims_preserve_shaped_m_in() {
    let (exact_shape, _) = single_relation_backend_fixture();
    let claim_cover = &exact_shape.claim_cover;

    let mut claims = Vec::new();
    claims.push(claim_cover.parent_claim_shape.zero_claim());
    claims.extend(
        claim_cover
            .state_in_claim_shapes
            .iter()
            .map(|shape| shape.zero_claim()),
    );
    claims.extend(
        claim_cover
            .state_out_claim_shapes
            .iter()
            .map(|shape| shape.zero_claim()),
    );
    claims.extend(
        claim_cover
            .ccs_output_shapes
            .iter()
            .map(|shape| shape.zero_claim()),
    );
    claims.extend(
        claim_cover
            .child_claim_shapes
            .iter()
            .map(|shape| shape.zero_claim()),
    );

    for (idx, claim) in claims.into_iter().enumerate() {
        assert_eq!(
            claim.m_in,
            claim.X.cols(),
            "shaped CE zero-claim lost m_in/X.cols agreement at slot {idx}"
        );
    }
}

#[test]
#[ignore = "temporary during main-recursion F'/NIFS.V refactor; re-enable after the owner/runtime seam stabilizes and transition_steps cleanup lands"]
fn rv32im_main_recursion_step_spartan_rejects_tampered_replayed_children_payload() {
    let (cover_shape, backend_relations) = single_relation_backend_fixture();
    let mut backend_relation = backend_relations
        .first()
        .expect("first backend relation")
        .clone();
    backend_relation.payload.pi_dec.children[0].fold_digest[0] ^= 1;

    let err = debug_check_rv32im_main_recursion_step_spartan_circuit(&cover_shape, &backend_relation)
        .expect_err("tampered replayed child payload must fail recursive-step circuit");
    assert!(matches!(err, Rv32imMainRecursionStepSpartanError::Prepare(_)));
}

#[test]
#[ignore = "temporary during main-recursion F'/NIFS.V refactor; re-enable after the owner/runtime seam stabilizes and transition_steps cleanup lands"]
fn rv32im_main_recursion_step_spartan_rejects_tampered_pi_rlc_parent_payload() {
    let (cover_shape, backend_relations) = single_relation_backend_fixture();
    let mut backend_relation = backend_relations
        .first()
        .expect("first backend relation")
        .clone();
    backend_relation.payload.pi_rlc.parent.fold_digest[0] ^= 1;

    let err = debug_check_rv32im_main_recursion_step_spartan_circuit(&cover_shape, &backend_relation)
        .expect_err("tampered Pi_RLC parent payload must fail recursive-step circuit");
    assert!(matches!(err, Rv32imMainRecursionStepSpartanError::Prepare(_)));
}

#[test]
#[ignore = "temporary during main-recursion F'/NIFS.V refactor; re-enable after the owner/runtime seam stabilizes and transition_steps cleanup lands"]
fn rv32im_main_recursion_step_spartan_rejects_tampered_pi_ccs_output_payload() {
    let (cover_shape, backend_relations) = single_relation_backend_fixture();
    let mut backend_relation = backend_relations
        .first()
        .expect("first backend relation")
        .clone();
    backend_relation.payload.pi_ccs.ccs_outputs[0].c.data[0] += F::ONE;

    let err = debug_check_rv32im_main_recursion_step_spartan_circuit(&cover_shape, &backend_relation)
        .expect_err("tampered Pi_CCS output payload must fail recursive-step circuit");
    assert!(matches!(err, Rv32imMainRecursionStepSpartanError::Prepare(_)));
}

#[test]
#[ignore = "temporary during main-recursion F'/NIFS.V refactor; re-enable after the owner/runtime seam stabilizes and transition_steps cleanup lands"]
fn rv32im_main_recursion_step_spartan_rejects_tampered_pi_ccs_row_challenge_payload() {
    let (cover_shape, backend_relations) = single_relation_backend_fixture();
    let mut backend_relation = backend_relations
        .first()
        .expect("first backend relation")
        .clone();
    backend_relation.payload.pi_ccs.replay.sumcheck_rounds[0][0] += K::ONE;

    let err = debug_check_rv32im_main_recursion_step_spartan_circuit(&cover_shape, &backend_relation)
        .expect_err("tampered Pi_CCS replay round payload must fail recursive-step circuit");
    assert!(matches!(err, Rv32imMainRecursionStepSpartanError::Prepare(_)));
}

#[test]
fn rv32im_main_recursion_step_spartan_side_aware_two_step_surface_contract_holds_for_every_step() {
    let (_, backend_relations) = side_aware_backend_relations_prefix_fixture(2);
    assert_eq!(
        backend_relations.len(),
        2,
        "expected a two-step side-aware backend fixture"
    );
    for (step_index, relation) in backend_relations.iter().enumerate() {
        assert_backend_relation_exact_surface_contract(
            relation,
            &format!("two-step side-aware relation #{step_index}"),
        );
    }
}

#[test]
fn rv32im_main_recursion_step_spartan_two_step_surface_contract_holds_for_every_step() {
    let (_, backend_relations) = backend_relations_prefix_fixture(2);
    assert_eq!(backend_relations.len(), 2, "expected a two-step backend fixture");
    for (step_index, relation) in backend_relations.iter().enumerate() {
        assert_backend_relation_exact_surface_contract(relation, &format!("two-step relation #{step_index}"));
    }
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv32im_main_recursion_step_spartan_single_step_circuit_is_satisfied() {
    let (cover_shape, backend_relations) = backend_relations_fixture();
    let first = backend_relations.first().expect("first backend relation");
    debug_check_rv32im_main_recursion_step_spartan_circuit(&cover_shape, first)
        .expect("single-step recursive-step circuit should synthesize cleanly");
}

#[test]
fn rv32im_main_recursion_step_spartan_chunk_replay_surface_is_covered() {
    let (_, backend_relations) = single_relation_backend_fixture();
    let first = backend_relations.first().expect("first backend relation");
    debug_check_rv32im_main_recursion_step_spartan_chunk_replay_surface(first)
        .expect("single-step recursive-step replay surface should fit the carried chunk cover");
}

#[test]
fn rv32im_main_recursion_step_spartan_pi_ccs_replay_lengths_match_dims() {
    let (_, backend_relations) = single_relation_backend_fixture();
    let first = backend_relations.first().expect("first backend relation");
    debug_check_rv32im_main_recursion_step_spartan_pi_ccs_replay_lengths(first)
        .expect("single-step recursive-step Pi_CCS replay lengths should match the native dims");
}

#[test]
fn rv32im_main_recursion_step_spartan_published_target_is_authoritative() {
    let (_, backend_relations) = single_relation_backend_fixture();
    let first = backend_relations.first().expect("first backend relation");
    let canonical_target = build_rv32im_main_recursion_step_spartan_published_target(first)
        .expect("build canonical recursive-step published target");

    assert_eq!(
        canonical_target.output_statement(),
        first.spartan_statement,
        "honest recursive-step published target must expose the same public output statement"
    );
    assert_eq!(
        canonical_target.terminal_f_prime_r2_public_values().len(),
        288,
        "terminal F' R2 public IO arity drifted unexpectedly"
    );
    assert_eq!(
        canonical_target.vk_fs_digest,
        first.f_prime_advice.verifier_key_fs().expected_digest(),
        "published target must bind the authoritative Construction-2 verifier key digest"
    );
    assert_eq!(
        canonical_target.chunk_count,
        first.f_prime_advice.chunk_count_in() + 1,
        "published target must expose the verifier's final iteration count"
    );
    assert_eq!(
        canonical_target.z_0,
        *first.f_prime_advice.z_0(),
        "published target must expose the verifier's initial state digest"
    );
    assert_eq!(
        canonical_target.z_i,
        first
            .f_prime_advice
            .fresh_state_out()
            .carry
            .terminal_handle
            .0,
        "published target must expose the verifier's final state digest"
    );
    assert_eq!(
        canonical_target.pc,
        first.payload.pc_next(),
        "published target must expose the verifier's final program counter"
    );

    let mut tampered = first.clone();
    tampered.spartan_statement.x_out =
        neo_fold_prototype::rv32im::Rv32imEncodedPublicInput::from_digest_bytes([7u8; 32]);
    tampered.spartan_statement.folded_accumulator_digest = [9u8; 32];
    let tampered_target = build_rv32im_main_recursion_step_spartan_published_target(&tampered)
        .expect("published target builder should ignore tampered statement shell bytes");

    assert_eq!(
        tampered_target, canonical_target,
        "published target must rebuild from authoritative F' surfaces, not from the carried step statement shell"
    );
}

#[test]
fn rv32im_main_recursion_step_spartan_rejects_stale_construction2_u_next_shell() {
    let (spartan_shape, backend_relations) = single_relation_backend_fixture();
    let first = backend_relations.first().expect("first backend relation");
    let full_width = neo_fold_prototype::rv32im::build_rv32im_main_recursion_construction2_canonical_full_width(
        first.f_prime_advice.verifier_key_fs(),
        first.f_prime_advice.phi_side(),
    )
    .expect("build canonical Construction-2 full width");
    let stale_u_i = neo_fold_prototype::rv32im::build_rv32im_main_recursion_construction2_default_fresh_instance(
        first.f_prime_advice.verifier_key_fs(),
        full_width,
    )
    .expect("build stale default Construction-2 u_i");
    assert_ne!(
        first.construction2_u_next, stale_u_i,
        "fixture must expose a non-default Construction-2 committed-step instance"
    );

    let canonical_target = build_rv32im_main_recursion_step_spartan_published_target(first)
        .expect("build canonical recursive-step published target");
    let mut tampered = first.clone();
    tampered.construction2_u_next = stale_u_i;

    let tampered_target = build_rv32im_main_recursion_step_spartan_published_target(&tampered)
        .expect("published target builder should recompute Construction-2 u_i from F' advice");
    assert_eq!(
        tampered_target, canonical_target,
        "published target must not trust the carried Construction-2 u_i shell"
    );

    let err = debug_check_rv32im_main_recursion_step_spartan_circuit(&spartan_shape, &tampered)
        .expect_err("recursive-step circuit must reject a stale Construction-2 u_i shell");
    assert!(
        format!("{err}").contains("construction2_u_next"),
        "unexpected stale Construction-2 shell error: {err}"
    );
}

#[test]
fn rv32im_main_recursion_step_spartan_published_targets_match_native_f_prime_across_chain() {
    let (_, backend_relations) = backend_relations_fixture();

    for (step_index, relation) in backend_relations.iter().enumerate() {
        let canonical_target = build_rv32im_main_recursion_step_spartan_published_target(relation)
            .unwrap_or_else(|err| panic!("step {step_index}: build canonical published target: {err}"));
        let step_image = evaluate_rv32im_main_recursion_f_prime_advice(&relation.f_prime_advice)
            .unwrap_or_else(|err| panic!("step {step_index}: evaluate native F' advice: {err}"));
        let output_statement = canonical_target.output_statement();

        assert_eq!(
            canonical_target.vk_fs_digest,
            relation.f_prime_advice.verifier_key_fs().expected_digest(),
            "step {step_index}: published target verifier-key digest drifted from native F' advice"
        );
        assert_eq!(
            canonical_target.chunk_count,
            step_image.chunk_count(),
            "step {step_index}: published target iteration count drifted from native F' image"
        );
        assert_eq!(
            canonical_target.z_0,
            *relation.f_prime_advice.z_0(),
            "step {step_index}: published target z_0 drifted from native F' advice"
        );
        assert_eq!(
            canonical_target.z_i,
            *step_image.z_next(),
            "step {step_index}: published target z_i drifted from native F' image"
        );
        assert_eq!(
            canonical_target.pc,
            step_image.pc_next(),
            "step {step_index}: published target pc drifted from native F' image"
        );
        assert_eq!(
            output_statement, relation.spartan_statement,
            "step {step_index}: published target output statement drifted from the authoritative recursive-step statement"
        );
        assert_eq!(
            output_statement.x_out,
            *step_image.x_out(),
            "step {step_index}: published target x_out drifted from the native F' image"
        );
        assert_eq!(
            output_statement.folded_accumulator_digest,
            step_image.folded_accumulator_digest(),
            "step {step_index}: published target folded accumulator drifted from the native F' image"
        );
    }
}

#[test]
fn rv32im_main_recursion_step_spartan_published_targets_match_construction2_state_images_across_chain() {
    let (_, backend_relations) = backend_relations_fixture();

    for (step_index, relation) in backend_relations.iter().enumerate() {
        audit_rv32im_main_recursion_step_spartan_published_target_matches_construction2_state_images(relation)
            .unwrap_or_else(|err| {
                panic!(
                    "step {step_index}: recursive-step published target must match the canonical Construction-2 carrier: {err}"
                )
            });
    }
}

#[test]
#[ignore = "diagnostic baseline during compressed-chain embedding refactor; re-enable after the embedded main-relation chunk verifier body stops going unsat under the internal-step witness seam"]
fn rv32im_main_recursion_step_spartan_embedded_body_is_satisfied() {
    let (cover_shape, backend_relations) = backend_relations_fixture();
    let first = backend_relations.first().expect("first backend relation");
    debug_check_rv32im_main_recursion_step_spartan_embedded_body(&cover_shape, first)
        .expect("embedded recursive-step body should synthesize cleanly");
}

#[test]
#[ignore = "temporary during main-recursion F'/NIFS.V refactor; re-enable after the owner/runtime seam stabilizes and transition_steps cleanup lands"]
fn rv32im_main_recursion_step_spartan_shape_fingerprints_match_counts_across_first_and_last_step() {
    let (cover_shape, backend_relations) = backend_relations_fixture();
    let first = backend_relations.first().expect("first backend relation");
    let last = backend_relations.last().expect("last backend relation");

    let first_shape = debug_measure_rv32im_main_recursion_step_spartan_circuit_shape(&cover_shape, first)
        .expect("measure first recursive-step circuit");
    let last_shape = debug_measure_rv32im_main_recursion_step_spartan_circuit_shape(&cover_shape, last)
        .expect("measure last recursive-step circuit");

    println!("first fixed-step shape: {first_shape:?}");
    println!(
        "first metadata: chunk_count_in={} halted_out={} step_shape={:?} cover_shape={:?} running_claims={} state_in_claims={} state_out_claims={} ccs_outputs={} children={}",
        first.f_prime_advice.chunk_count_in(),
        first.payload.step_shape.terminal_step,
        first.payload.step_shape,
        first.payload.cover_shape,
        first.f_prime_advice.running_state().carry.main.claims.len(),
        first.payload.state_in_claims.len(),
        first.payload.state_out_claims.len(),
        first.payload.pi_ccs.ccs_outputs.len(),
        first.payload.pi_dec.children.len(),
    );
    println!("last fixed-step shape: {last_shape:?}");
    println!(
        "last metadata: chunk_count_in={} halted_out={} step_shape={:?} cover_shape={:?} running_claims={} state_in_claims={} state_out_claims={} ccs_outputs={} children={}",
        last.f_prime_advice.chunk_count_in(),
        last.payload.step_shape.terminal_step,
        last.payload.step_shape,
        last.payload.cover_shape,
        last.f_prime_advice.running_state().carry.main.claims.len(),
        last.payload.state_in_claims.len(),
        last.payload.state_out_claims.len(),
        last.payload.pi_ccs.ccs_outputs.len(),
        last.payload.pi_dec.children.len(),
    );

    assert_eq!(first_shape.num_inputs, last_shape.num_inputs);
    assert_eq!(first_shape.num_aux, last_shape.num_aux);
    assert_eq!(first_shape.num_constraints, last_shape.num_constraints);
}

#[test]
#[ignore = "diagnostic metadata dump for recursive-step shape drift"]
fn rv32im_main_recursion_step_spartan_print_first_last_metadata() {
    let (cover_shape, backend_relations) = backend_relations_fixture();
    let first = backend_relations.first().expect("first backend relation");
    let last = backend_relations.last().expect("last backend relation");
    let first_live_state_in_shapes = first
        .f_prime_advice
        .running_state()
        .carry
        .main
        .claims
        .iter()
        .map(neo_fold_prototype::rv32im::audit::Rv32imCeClaimDigestShape::from_claim)
        .collect::<Vec<_>>();
    let last_live_state_in_shapes = last
        .f_prime_advice
        .running_state()
        .carry
        .main
        .claims
        .iter()
        .map(neo_fold_prototype::rv32im::audit::Rv32imCeClaimDigestShape::from_claim)
        .collect::<Vec<_>>();
    let first_padded_state_in_shapes = first
        .payload
        .state_in_claims
        .iter()
        .map(neo_fold_prototype::rv32im::audit::Rv32imCeClaimDigestShape::from_claim)
        .collect::<Vec<_>>();
    let last_padded_state_in_shapes = last
        .payload
        .state_in_claims
        .iter()
        .map(neo_fold_prototype::rv32im::audit::Rv32imCeClaimDigestShape::from_claim)
        .collect::<Vec<_>>();
    let first_output_shapes = first
        .payload
        .pi_ccs
        .ccs_outputs
        .iter()
        .map(neo_fold_prototype::rv32im::audit::Rv32imCeClaimDigestShape::from_claim)
        .collect::<Vec<_>>();
    let last_output_shapes = last
        .payload
        .pi_ccs
        .ccs_outputs
        .iter()
        .map(neo_fold_prototype::rv32im::audit::Rv32imCeClaimDigestShape::from_claim)
        .collect::<Vec<_>>();
    let first_child_shapes = first
        .payload
        .pi_dec
        .children
        .iter()
        .map(neo_fold_prototype::rv32im::audit::Rv32imCeClaimDigestShape::from_claim)
        .collect::<Vec<_>>();
    let last_child_shapes = last
        .payload
        .pi_dec
        .children
        .iter()
        .map(neo_fold_prototype::rv32im::audit::Rv32imCeClaimDigestShape::from_claim)
        .collect::<Vec<_>>();
    let first_fresh_claim_shapes = first
        .payload
        .fresh_claims
        .iter()
        .map(neo_fold_prototype::rv32im::audit::Rv32imCcsClaimShape::from_claim)
        .collect::<Vec<_>>();
    let last_fresh_claim_shapes = last
        .payload
        .fresh_claims
        .iter()
        .map(neo_fold_prototype::rv32im::audit::Rv32imCcsClaimShape::from_claim)
        .collect::<Vec<_>>();
    let first_fresh_witness_shapes = first
        .payload
        .fresh_witnesses
        .iter()
        .map(neo_fold_prototype::rv32im::audit::Rv32imCcsWitnessShape::from_witness)
        .collect::<Vec<_>>();
    let last_fresh_witness_shapes = last
        .payload
        .fresh_witnesses
        .iter()
        .map(neo_fold_prototype::rv32im::audit::Rv32imCcsWitnessShape::from_witness)
        .collect::<Vec<_>>();
    let first_parent_shape =
        neo_fold_prototype::rv32im::audit::Rv32imCeClaimDigestShape::from_claim(&first.payload.pi_rlc.parent);
    let last_parent_shape =
        neo_fold_prototype::rv32im::audit::Rv32imCeClaimDigestShape::from_claim(&last.payload.pi_rlc.parent);

    let live_state_in_drift = first_live_state_in_shapes
        .iter()
        .zip(last_live_state_in_shapes.iter())
        .enumerate()
        .filter(|(_, (left, right))| left != right)
        .collect::<Vec<_>>();
    let padded_state_in_drift = first_padded_state_in_shapes
        .iter()
        .zip(last_padded_state_in_shapes.iter())
        .enumerate()
        .filter(|(_, (left, right))| left != right)
        .collect::<Vec<_>>();
    let output_drift = first_output_shapes
        .iter()
        .zip(last_output_shapes.iter())
        .enumerate()
        .filter(|(_, (left, right))| left != right)
        .collect::<Vec<_>>();
    let child_drift = first_child_shapes
        .iter()
        .zip(last_child_shapes.iter())
        .enumerate()
        .filter(|(_, (left, right))| left != right)
        .collect::<Vec<_>>();
    let fresh_claim_drift = first_fresh_claim_shapes
        .iter()
        .zip(last_fresh_claim_shapes.iter())
        .enumerate()
        .filter(|(_, (left, right))| left != right)
        .collect::<Vec<_>>();
    let fresh_witness_drift = first_fresh_witness_shapes
        .iter()
        .zip(last_fresh_witness_shapes.iter())
        .enumerate()
        .filter(|(_, (left, right))| left != right)
        .collect::<Vec<_>>();
    let parent_drift = first_parent_shape != last_parent_shape;

    println!("cover_shape: {cover_shape:?}");
    println!(
        "first metadata: chunk_count_in={} halted_out={} step_shape={:?} running_claims={} state_in_claims={} state_out_claims={} ccs_outputs={} children={}",
        first.f_prime_advice.chunk_count_in(),
        first.payload.step_shape.terminal_step,
        first.payload.step_shape,
        first.f_prime_advice.running_state().carry.main.claims.len(),
        first.payload.state_in_claims.len(),
        first.payload.state_out_claims.len(),
        first.payload.pi_ccs.ccs_outputs.len(),
        first.payload.pi_dec.children.len(),
    );
    println!(
        "last metadata: chunk_count_in={} halted_out={} step_shape={:?} running_claims={} state_in_claims={} state_out_claims={} ccs_outputs={} children={}",
        last.f_prime_advice.chunk_count_in(),
        last.payload.step_shape.terminal_step,
        last.payload.step_shape,
        last.f_prime_advice.running_state().carry.main.claims.len(),
        last.payload.state_in_claims.len(),
        last.payload.state_out_claims.len(),
        last.payload.pi_ccs.ccs_outputs.len(),
        last.payload.pi_dec.children.len(),
    );
    println!("first parent shape: {:?}", first_parent_shape);
    println!("last parent shape: {:?}", last_parent_shape);
    println!(
        "first live state_in[0]: {:?}",
        first
            .f_prime_advice
            .running_state()
            .carry
            .main
            .claims
            .first()
            .map(neo_fold_prototype::rv32im::audit::Rv32imCeClaimDigestShape::from_claim)
    );
    println!(
        "last live state_in[0]: {:?}",
        last.f_prime_advice
            .running_state()
            .carry
            .main
            .claims
            .first()
            .map(neo_fold_prototype::rv32im::audit::Rv32imCeClaimDigestShape::from_claim)
    );
    println!(
        "first padded state_in[0]: {:?}",
        first
            .payload
            .state_in_claims
            .first()
            .map(neo_fold_prototype::rv32im::audit::Rv32imCeClaimDigestShape::from_claim)
    );
    println!(
        "last padded state_in[0]: {:?}",
        last.payload
            .state_in_claims
            .first()
            .map(neo_fold_prototype::rv32im::audit::Rv32imCeClaimDigestShape::from_claim)
    );
    println!(
        "live state_in drift indices: {:?}",
        live_state_in_drift
            .iter()
            .map(|(idx, _)| *idx)
            .collect::<Vec<_>>()
    );
    println!(
        "padded state_in drift indices: {:?}",
        padded_state_in_drift
            .iter()
            .map(|(idx, _)| *idx)
            .collect::<Vec<_>>()
    );
    println!(
        "ccs_output drift indices: {:?}",
        output_drift.iter().map(|(idx, _)| *idx).collect::<Vec<_>>()
    );
    println!(
        "child drift indices: {:?}",
        child_drift.iter().map(|(idx, _)| *idx).collect::<Vec<_>>()
    );
    println!(
        "fresh claim drift indices: {:?}",
        fresh_claim_drift
            .iter()
            .map(|(idx, _)| *idx)
            .collect::<Vec<_>>()
    );
    println!(
        "fresh witness drift indices: {:?}",
        fresh_witness_drift
            .iter()
            .map(|(idx, _)| *idx)
            .collect::<Vec<_>>()
    );
    println!("parent drift: {parent_drift}");
    if let Some((idx, (left, right))) = live_state_in_drift.first() {
        println!("first live state_in drift[{idx}]: left={left:?} right={right:?}");
    }
    if let Some((idx, (left, right))) = padded_state_in_drift.first() {
        println!("first padded state_in drift[{idx}]: left={left:?} right={right:?}");
    }
    if let Some((idx, (left, right))) = output_drift.first() {
        println!("first ccs_output drift[{idx}]: left={left:?} right={right:?}");
    }
    if let Some((idx, (left, right))) = child_drift.first() {
        println!("first child drift[{idx}]: left={left:?} right={right:?}");
    }
    if let Some((idx, (left, right))) = fresh_claim_drift.first() {
        println!("first fresh claim drift[{idx}]: left={left:?} right={right:?}");
    }
    if let Some((idx, (left, right))) = fresh_witness_drift.first() {
        println!("first fresh witness drift[{idx}]: left={left:?} right={right:?}");
    }
    if parent_drift {
        println!(
            "first parent drift: left={:?} right={:?}",
            first_parent_shape, last_parent_shape,
        );
    }
}

#[test]
#[ignore = "diagnostic probe for the remaining multi-step recursive-step failure"]
fn rv32im_main_recursion_step_spartan_second_step_circuit_is_satisfied_under_prefix_shape() {
    let (cover_shape, backend_relations) = backend_relations_prefix_fixture(2);
    let second = backend_relations.get(1).expect("second backend relation");
    debug_check_rv32im_main_recursion_step_spartan_circuit(&cover_shape, second)
        .expect("second recursive-step circuit should synthesize cleanly under the shared prefix shape");
}
