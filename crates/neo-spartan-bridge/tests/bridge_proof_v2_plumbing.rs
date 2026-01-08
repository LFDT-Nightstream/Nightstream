#![allow(non_snake_case)]

use neo_closure_proof::{prove_test_only_v1, verify_closure_v1, ClosureProofV1};
use neo_spartan_bridge::api::SpartanProof;
use neo_spartan_bridge::bridge_proof_v2::compute_closure_statement_v1;
use neo_spartan_bridge::statement::SpartanShardStatement;
use neo_spartan_bridge::BridgeProofV2;

fn byte32(seed: u8) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = seed.wrapping_add(i as u8);
    }
    out
}

#[test]
fn bridge_proof_v2_closure_statement_is_deterministic() {
    let stmt = SpartanShardStatement::new(
        byte32(1),
        byte32(2),
        byte32(3),
        byte32(4),
        byte32(5),
        byte32(6),
        byte32(7),
        byte32(8),
        byte32(9),
        byte32(10),
        byte32(11),
        2,
        true,
        false,
    );

    let expected_closure_stmt = compute_closure_statement_v1(&stmt);
    let closure = prove_test_only_v1(&expected_closure_stmt);

    let spartan = SpartanProof {
        proof_data: vec![1, 2, 3],
        statement: stmt,
    };
    let bridge = BridgeProofV2::new(spartan, closure);

    assert_eq!(bridge.closure_stmt, expected_closure_stmt);
    verify_closure_v1(&bridge.closure_stmt, &bridge.closure).expect("closure proof must verify");
}

#[test]
fn bridge_proof_v2_test_only_closure_tampering_is_rejected() {
    let stmt = SpartanShardStatement::new(
        byte32(21),
        byte32(22),
        byte32(23),
        byte32(24),
        byte32(25),
        byte32(26),
        byte32(27),
        byte32(28),
        byte32(29),
        byte32(30),
        byte32(31),
        1,
        false,
        true,
    );

    let closure_stmt = compute_closure_statement_v1(&stmt);
    let mut closure = prove_test_only_v1(&closure_stmt);
    match &mut closure {
        ClosureProofV1::TestOnlyDigest { digest } => {
            digest[0] ^= 1;
        }
        ClosureProofV1::OpaqueBytes { .. } => unreachable!("prove_test_only_v1 returns TestOnlyDigest"),
    }

    assert!(
        verify_closure_v1(&closure_stmt, &closure).is_err(),
        "tampering must be rejected"
    );
}

