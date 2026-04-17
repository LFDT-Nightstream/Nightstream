#![allow(dead_code)]

#[path = "support/rv64im_n2.rs"]
mod rv64im_n2_support;

use neo_fold_next::rv64im::audit::{build_rv64im_chunk_step_relations, verify_rv64im_chunk_step_relation};
use p3_field::PrimeCharacteristicRing;

#[test]
fn rv64im_chunk_step_relation_round_trip() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let relations = build_rv64im_chunk_step_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step relations");

    assert_eq!(
        relations.len(),
        fixture.final_proof.steps.len(),
        "chunk-step relation count must match the carried replay witness count"
    );
    assert!(
        !relations.is_empty(),
        "expected at least one RV64IM chunk-step relation"
    );

    for relation in &relations {
        let state_out = verify_rv64im_chunk_step_relation(&relation.statement, &relation.witness)
            .expect("verify chunk-step relation");
        assert_eq!(
            state_out.carry.main.claims, relation.witness.carry_out.claims,
            "verified one-step relation must recover the carried next-main claims"
        );
        assert_eq!(
            state_out.carry.main.witnesses, relation.witness.carry_out.witnesses,
            "verified one-step relation must recover the carried next-main witnesses"
        );
        assert_eq!(
            state_out.transcript, relation.statement.transcript_out,
            "verified one-step relation must recover the carried transcript_out snapshot"
        );
        assert_eq!(
            state_out.carry.terminal_handle.0, relation.statement.step_public.state_out,
            "verified one-step relation must recover the carried terminal handle"
        );
    }
}

#[test]
fn rv64im_chunk_step_relation_rejects_tampered_transcript_out() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let mut relation = build_rv64im_chunk_step_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step relations")
        .into_iter()
        .next()
        .expect("first relation");

    relation.statement.transcript_out.state[0] += neo_math::F::ONE;

    let err = verify_rv64im_chunk_step_relation(&relation.statement, &relation.witness)
        .expect_err("tampered transcript_out must fail");
    assert!(format!("{err}").contains("transcript_out"), "unexpected error: {err}");
}

#[test]
fn rv64im_chunk_step_relation_rejects_tampered_next_carry() {
    let fixture = rv64im_n2_support::build_rv64im_n2_fixture().expect("build rv64im n=2 fixture");
    let mut relation = build_rv64im_chunk_step_relations(&fixture.final_statement, &fixture.final_proof)
        .expect("build chunk-step relations")
        .into_iter()
        .next()
        .expect("first relation");

    if let Some(first_witness) = relation.witness.carry_out.witnesses.first_mut() {
        if first_witness.rows() > 0 && first_witness.cols() > 0 {
            first_witness[(0, 0)] += neo_math::F::ONE;
        }
    } else if let Some(first_claim) = relation.witness.carry_out.claims.first_mut() {
        first_claim.fold_digest[0] ^= 1;
    } else {
        panic!("expected non-empty carried next-main state");
    }

    let err = verify_rv64im_chunk_step_relation(&relation.statement, &relation.witness)
        .expect_err("tampered next carry must fail");
    assert!(format!("{err}").contains("next carry"), "unexpected error: {err}");
}
