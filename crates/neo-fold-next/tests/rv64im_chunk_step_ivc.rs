#![allow(dead_code)]

#[path = "support/rv64im_n2.rs"]
mod rv64im_n2_support;

use neo_fold_next::rv64im::audit::{
    build_rv64im_chunk_step_ivc_relations, rv64im_bridge_handoff_chain_digest, rv64im_bridge_handoff_chain_digest_init,
    rv64im_bridge_handoff_chain_digest_step, rv64im_chunk_step_ivc_initial_state, rv64im_step_statement_chain_digest,
    rv64im_step_statement_chain_digest_init, rv64im_step_statement_chain_digest_step, verify_rv64im_chunk_step_ivc,
    verify_rv64im_chunk_step_ivc_chain,
};
use p3_field::PrimeCharacteristicRing;

fn tamper_state_out_claim_projection_shell(
    claim: &mut neo_ccs::CeClaim<neo_ajtai::Commitment, neo_math::F, neo_math::K>,
) {
    if claim.X.rows() > 1 && claim.X.cols() > 0 {
        claim.X[(1, 0)] += neo_math::F::ONE;
    }
    if let Some(first) = claim.s_col.first_mut() {
        *first += neo_math::K::ONE;
    }
    if let Some(first) = claim.ct.first_mut() {
        *first += neo_math::K::ONE;
    }
    if let Some(first) = claim.aux_openings.first_mut() {
        *first += neo_math::K::ONE;
    }
    if let Some(first) = claim.y_zcol.first_mut() {
        *first += neo_math::K::ONE;
    }
    if let Some(first) = claim.c_step_coords.first_mut() {
        *first += neo_math::F::ONE;
    }
    claim.fold_digest[0] ^= 1;
}

#[test]
fn rv64im_chunk_step_ivc_round_trip() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step IVC relations");

    assert_eq!(
        relations.len(),
        fixture.final_proof.steps.len(),
        "chunk-step IVC relation count must match the carried replay witness count"
    );
    assert!(
        !relations.is_empty(),
        "expected at least one RV64IM chunk-step IVC relation"
    );

    for relation in &relations {
        let state_out =
            verify_rv64im_chunk_step_ivc(&relation.statement, &relation.witness).expect("verify chunk-step IVC");
        assert_eq!(
            state_out.carry.main.claims, relation.witness.state_out.carry.main.claims,
            "verified chunk-step IVC must recover the carried private next-main claims"
        );
        assert_eq!(
            state_out.carry.main.witnesses, relation.witness.state_out.carry.main.witnesses,
            "verified chunk-step IVC must recover the carried private next-main witnesses"
        );
        assert_eq!(
            state_out.transcript, relation.witness.state_out.transcript,
            "verified chunk-step IVC must recover the carried private next transcript state"
        );
        assert_eq!(
            state_out.carry.terminal_handle.0, relation.statement.step_public.state_out,
            "verified chunk-step IVC must recover the carried terminal handle"
        );
    }
}

#[test]
fn rv64im_chunk_step_ivc_rejects_tampered_private_transcript_out() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let mut relation = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step IVC relations")
        .into_iter()
        .next()
        .expect("first relation");

    relation.witness.state_out.transcript.state[0] += neo_math::F::ONE;

    let err = verify_rv64im_chunk_step_ivc(&relation.statement, &relation.witness)
        .expect_err("tampered private transcript_out must fail");
    assert!(format!("{err}").contains("transcript_out"), "unexpected error: {err}");
}

#[test]
fn rv64im_chunk_step_ivc_rejects_tampered_private_next_carry() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let mut relation = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step IVC relations")
        .into_iter()
        .next()
        .expect("first relation");

    if let Some(first_witness) = relation.witness.state_out.carry.main.witnesses.first_mut() {
        if first_witness.rows() > 0 && first_witness.cols() > 0 {
            first_witness[(0, 0)] += neo_math::F::ONE;
        }
    } else if let Some(first_claim) = relation.witness.state_out.carry.main.claims.first_mut() {
        first_claim.y_ring[0][1] += neo_math::K::ONE;
    } else {
        panic!("expected non-empty carried next-main state");
    }

    let err = verify_rv64im_chunk_step_ivc(&relation.statement, &relation.witness)
        .expect_err("tampered private next carry must fail");
    assert!(format!("{err}").contains("next carry"), "unexpected error: {err}");
}

#[test]
fn rv64im_chunk_step_ivc_ignores_state_out_claim_projection_shell() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let mut relation = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step IVC relations")
        .into_iter()
        .next()
        .expect("first relation");

    let first_claim = relation
        .witness
        .state_out
        .carry
        .main
        .claims
        .first_mut()
        .expect("state_out claim");
    tamper_state_out_claim_projection_shell(first_claim);

    let state_out =
        verify_rv64im_chunk_step_ivc(&relation.statement, &relation.witness).expect("projection shell must be ignored");
    assert_eq!(
        state_out.carry.main.claims, relation.witness.state_out.carry.main.claims,
        "verified chunk-step IVC should preserve the carried shell while checking only CE projection authority",
    );
}

#[test]
fn rv64im_chunk_step_ivc_chain_round_trip() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step IVC relations");
    let final_state = verify_rv64im_chunk_step_ivc_chain(&relations).expect("verify chunk-step IVC chain");
    let expected_final_state = relations
        .last()
        .map(|relation| &relation.witness.state_out)
        .unwrap_or_else(|| panic!("expected at least one chunk-step IVC relation"));

    assert_eq!(
        final_state.carry.main.claims, expected_final_state.carry.main.claims,
        "chunk-step IVC chain must recover the final carried claims"
    );
    assert_eq!(
        final_state.carry.main.witnesses, expected_final_state.carry.main.witnesses,
        "chunk-step IVC chain must recover the final carried witnesses"
    );
    assert_eq!(
        final_state.transcript, expected_final_state.transcript,
        "chunk-step IVC chain must recover the final carried transcript state"
    );
    assert_eq!(
        final_state.carry.terminal_handle.0, expected_final_state.carry.terminal_handle.0,
        "chunk-step IVC chain must recover the final carried terminal handle"
    );
}

#[test]
fn rv64im_chunk_step_ivc_chain_rejects_tampered_initial_private_state() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let mut relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step IVC relations");

    let relation = relations.first_mut().expect("first relation");
    relation.witness.state_in = rv64im_chunk_step_ivc_initial_state();
    relation.witness.state_in.transcript.state[0] += neo_math::F::ONE;

    let err =
        verify_rv64im_chunk_step_ivc_chain(&relations).expect_err("tampered initial private recursive state must fail");
    assert!(format!("{err}").contains("state_in"), "unexpected error: {err}");
}

#[test]
fn rv64im_chunk_step_ivc_chain_digests_match_stepwise_accumulation() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let relations = build_rv64im_chunk_step_ivc_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step IVC relations");

    let mut step_statement_chain = rv64im_step_statement_chain_digest_init();
    let mut bridge_handoff_chain = rv64im_bridge_handoff_chain_digest_init();
    for relation in &relations {
        step_statement_chain =
            rv64im_step_statement_chain_digest_step(step_statement_chain, relation.statement.expected_digest());
        bridge_handoff_chain = rv64im_bridge_handoff_chain_digest_step(
            bridge_handoff_chain,
            relation.witness.handoff.bridge_handoff.digest,
        );
    }

    assert_eq!(
        step_statement_chain,
        rv64im_step_statement_chain_digest(&relations),
        "stepwise statement accumulator must match the batch chain digest",
    );
    assert_eq!(
        bridge_handoff_chain,
        rv64im_bridge_handoff_chain_digest(&relations),
        "stepwise bridge-handoff accumulator must match the batch chain digest",
    );
}
