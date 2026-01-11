#![allow(non_snake_case)]

use neo_closure_proof::ClosureProofV1;
use neo_spartan_bridge::api::SpartanProof;
use neo_spartan_bridge::bridge_proof_v2::compute_closure_statement_v1;
use neo_spartan_bridge::statement::SpartanShardStatement;
use neo_spartan_bridge::{deserialize_bridge_proof_v2, serialize_bridge_proof_v2, BridgeProofV2};

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
    let closure = ClosureProofV1::OpaqueBytes {
        proof_bytes: vec![1, 2, 3, 4],
    };

    let spartan = SpartanProof {
        proof_data: vec![1, 2, 3],
        statement: stmt,
    };
    let bridge = BridgeProofV2::new(spartan, closure);

    assert_eq!(bridge.closure_statement(), expected_closure_stmt);
}

#[test]
fn bridge_proof_v2_codec_roundtrip_and_rejects_trailing_bytes() {
    let stmt = SpartanShardStatement::new(
        byte32(41),
        byte32(42),
        byte32(43),
        byte32(44),
        byte32(45),
        byte32(46),
        byte32(47),
        byte32(48),
        byte32(49),
        byte32(50),
        byte32(51),
        3,
        true,
        true,
    );

    let closure_stmt = compute_closure_statement_v1(&stmt);
    let closure = ClosureProofV1::OpaqueBytes {
        proof_bytes: vec![9, 9, 9, closure_stmt.version as u8],
    };

    let spartan = SpartanProof {
        proof_data: vec![9, 8, 7, 6, 5],
        statement: stmt.clone(),
    };
    let bridge = BridgeProofV2::new(spartan, closure);

    let bytes = serialize_bridge_proof_v2(&bridge).expect("serialize");
    let roundtrip = deserialize_bridge_proof_v2(&bytes).expect("deserialize");
    assert_eq!(roundtrip.spartan.proof_data, bridge.spartan.proof_data);
    assert_eq!(roundtrip.spartan.statement, bridge.spartan.statement);
    assert_eq!(roundtrip.closure, bridge.closure);

    let mut bytes_with_trailing = bytes;
    bytes_with_trailing.push(0);
    assert!(deserialize_bridge_proof_v2(&bytes_with_trailing).is_err());
}
