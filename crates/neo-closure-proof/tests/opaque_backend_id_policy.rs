use neo_closure_proof::{
    verify_closure_v1, verify_closure_v1_production_with_context_and_bus, ClosureProofError, ClosureProofV1,
    ClosureStatementV1,
};

fn encode_envelope(backend_id: u32, payload: &[u8]) -> Vec<u8> {
    let payload_len: u32 = payload.len().try_into().unwrap();
    let mut out = Vec::with_capacity(16 + payload.len());
    out.extend_from_slice(b"NCLP");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&backend_id.to_le_bytes());
    out.extend_from_slice(&payload_len.to_le_bytes());
    out.extend_from_slice(payload);
    out
}

#[test]
fn whir_backend_ids_require_verification_context() {
    let stmt = ClosureStatementV1::new([1u8; 32], [2u8; 32], [3u8; 32]);
    for backend_id in [5u32, 6u32] {
        let proof = ClosureProofV1::OpaqueBytes {
            proof_bytes: encode_envelope(backend_id, &[]),
        };
        let err = verify_closure_v1(&stmt, &proof).expect_err("missing context must be rejected");
        assert!(
            matches!(err, ClosureProofError::MissingVerificationContext),
            "expected MissingVerificationContext, got {err:?}"
        );
    }
}

#[test]
fn rejects_unknown_backend_id() {
    let stmt = ClosureStatementV1::new([1u8; 32], [2u8; 32], [3u8; 32]);
    let proof = ClosureProofV1::OpaqueBytes {
        proof_bytes: encode_envelope(/*backend_id=*/ 99, &[]),
    };

    let err = verify_closure_v1(&stmt, &proof).expect_err("unknown backend id must be rejected");
    assert!(
        matches!(err, ClosureProofError::InvalidOpaqueProofEncoding),
        "expected InvalidOpaqueProofEncoding, got {err:?}"
    );
}

#[test]
fn production_rejects_dev_backend_id_5() {
    let stmt = ClosureStatementV1::new([1u8; 32], [2u8; 32], [3u8; 32]);
    let proof = ClosureProofV1::OpaqueBytes {
        proof_bytes: encode_envelope(/*backend_id=*/ 5, &[]),
    };

    let err = verify_closure_v1_production_with_context_and_bus(&stmt, &proof, None, None, None)
        .expect_err("dev backend id 5 must be rejected in production");
    assert!(
        matches!(err, ClosureProofError::BackendNotImplemented),
        "expected BackendNotImplemented, got {err:?}"
    );
}

#[test]
fn production_private_backend_requires_context() {
    let stmt = ClosureStatementV1::new([1u8; 32], [2u8; 32], [3u8; 32]);
    let proof = ClosureProofV1::OpaqueBytes {
        proof_bytes: encode_envelope(/*backend_id=*/ 6, &[]),
    };

    let err = verify_closure_v1_production_with_context_and_bus(&stmt, &proof, None, None, None)
        .expect_err("missing context must be rejected");
    assert!(
        matches!(err, ClosureProofError::MissingVerificationContext),
        "expected MissingVerificationContext, got {err:?}"
    );
}

#[test]
fn production_rejects_private_backend_id_6_even_with_context() {
    let stmt = ClosureStatementV1::new([1u8; 32], [2u8; 32], [3u8; 32]);
    let proof = ClosureProofV1::OpaqueBytes {
        proof_bytes: encode_envelope(/*backend_id=*/ 6, &[]),
    };

    // Dummy context is enough to get past MissingVerificationContext; production then fails closed.
    let m = 1usize;
    let params = neo_params::NeoParams::goldilocks_auto_r1cs_ccs(m).expect("params");
    let ccs = neo_ccs::CcsStructure::new(
        vec![neo_ccs::Mat::identity(m)],
        neo_ccs::SparsePoly::new(1, vec![]),
    )
    .expect("ccs");

    let err = verify_closure_v1_production_with_context_and_bus(&stmt, &proof, Some(&params), Some(&ccs), None)
        .expect_err("private backend must be rejected in production for now");
    assert!(
        matches!(err, ClosureProofError::BackendNotImplemented),
        "expected BackendNotImplemented, got {err:?}"
    );
}
