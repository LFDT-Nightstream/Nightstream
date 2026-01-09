#![cfg(feature = "whir-p3-backend")]

use neo_closure_proof::{prove_whir_p3_placeholder_v1, verify_closure_v1, ClosureProofV1, ClosureStatementV1};

#[test]
fn whir_p3_placeholder_roundtrip() {
    let stmt = ClosureStatementV1::new([1u8; 32], [2u8; 32], [3u8; 32]);
    let proof = prove_whir_p3_placeholder_v1(&stmt);
    verify_closure_v1(&stmt, &proof).expect("WHIR placeholder proof should verify");
}

#[test]
fn whir_p3_placeholder_rejects_wrong_statement() {
    let stmt_good = ClosureStatementV1::new([1u8; 32], [2u8; 32], [3u8; 32]);
    let proof = prove_whir_p3_placeholder_v1(&stmt_good);

    let stmt_bad = ClosureStatementV1::new([9u8; 32], [2u8; 32], [3u8; 32]);
    assert!(
        verify_closure_v1(&stmt_bad, &proof).is_err(),
        "proof must be bound to the exact statement digests"
    );
}

#[test]
fn whir_p3_placeholder_rejects_corrupted_bytes() {
    let stmt = ClosureStatementV1::new([1u8; 32], [2u8; 32], [3u8; 32]);
    let mut proof = prove_whir_p3_placeholder_v1(&stmt);

    if let ClosureProofV1::OpaqueBytes { proof_bytes } = &mut proof {
        // Flip a byte late in the encoding to avoid trivial header-only corruption.
        if let Some(b) = proof_bytes.last_mut() {
            *b ^= 0x01;
        } else {
            panic!("expected non-empty proof_bytes");
        }
    } else {
        panic!("expected OpaqueBytes proof");
    }

    assert!(verify_closure_v1(&stmt, &proof).is_err());
}

