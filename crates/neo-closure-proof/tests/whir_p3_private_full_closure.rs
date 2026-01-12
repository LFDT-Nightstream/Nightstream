use neo_ajtai::{set_global_pp_seeded, AjtaiSModule};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsStructure, Mat, SparsePoly};
use neo_closure_proof::{
    compute_accumulator_digest_v2, compute_obligations_digest_v2, prove_whir_p3_private_full_closure_v1,
    verify_closure_v1_production_with_context_and_bus, verify_closure_v1_with_context_and_bus, ClosureProofError,
    ClosureProofV1, ClosureStatementV1,
};
use neo_fold::shard::ShardObligations;
use neo_math::{D, F, K};
use bincode::Options;
use p3_field::PrimeCharacteristicRing;

fn identity_ccs(m: usize) -> CcsStructure<F> {
    let mat = Mat::identity(m);
    let f = SparsePoly::new(1, vec![]);
    CcsStructure::new(vec![mat], f).expect("CCS")
}

fn x_prefix(z: &Mat<F>, m_in: usize) -> Mat<F> {
    let mut out = Mat::zero(D, m_in, F::ZERO);
    for r in 0..D {
        for c in 0..m_in {
            out[(r, c)] = z[(r, c)];
        }
    }
    out
}

#[test]
fn whir_p3_private_full_closure_roundtrip_and_tamper_reject() {
    let m = 16usize;
    let m_in = 4usize;
    let ccs = identity_ccs(m);
    let params = neo_params::NeoParams::goldilocks_auto_r1cs_ccs(m).expect("params");

    let seed = [11u8; 32];
    set_global_pp_seeded(D, params.kappa as usize, m, seed).expect("set_global_pp_seeded");
    let l = AjtaiSModule::from_global_for_dims(D, m).expect("committer");

    // Small bounded witness Z (digits in {-1,0,1} for b=2).
    let mut z = Mat::zero(D, m, F::ZERO);
    for r in 0..D {
        for c in 0..m {
            let v = ((r * 31 + c * 17) % 3) as u64;
            z[(r, c)] = match v {
                0 => F::ZERO,
                1 => F::ONE,
                _ => F::ZERO - F::ONE,
            };
        }
    }
    let cmt = l.commit(&z);

    // r point for CCS ME relation (ell_n = log2(n)=4 for identity CCS with n=m=16).
    let r_point = vec![
        K::from(F::from_u64(3)),
        K::from(F::from_u64(5)),
        K::from(F::from_u64(7)),
        K::from(F::from_u64(11)),
    ];
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let (y, y_scalars) = neo_reductions::common::compute_y_from_Z_and_r(&ccs, &z, &r_point, ell_d, params.b);

    let me = neo_ccs::MeInstance {
        c: cmt,
        X: x_prefix(&z, m_in),
        r: r_point,
        y,
        y_scalars: y_scalars.clone(),
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    };

    let obligations = ShardObligations {
        main: vec![me],
        val: vec![],
    };

    let pp_id_digest = neo_ajtai::compute_pp_id_digest_v1(D, m, params.kappa as usize, seed);
    let acc_main = compute_accumulator_digest_v2(params.b, obligations.main.as_slice());
    let acc_val = compute_accumulator_digest_v2(params.b, obligations.val.as_slice());
    let obligations_digest = compute_obligations_digest_v2(acc_main, acc_val, pp_id_digest);

    let stmt = ClosureStatementV1::new([1u8; 32], pp_id_digest, obligations_digest);
    let proof =
        prove_whir_p3_private_full_closure_v1(&stmt, &params, &ccs, &obligations, &[z.clone()], &[], None)
            .expect("prove");

    // Basic roundtrip.
    verify_closure_v1_with_context_and_bus(&stmt, &proof, Some(&params), Some(&ccs), None).expect("verify");

    // Production verifier currently fails closed until the obligations→weights/claims binding is implemented.
    let err = verify_closure_v1_production_with_context_and_bus(&stmt, &proof, Some(&params), Some(&ccs), None)
        .expect_err("production verifier must fail closed");
    assert!(
        matches!(err, ClosureProofError::BackendNotImplemented),
        "expected BackendNotImplemented, got {err:?}"
    );

    // Regression guard: proof size stays bounded for this tiny instance.
    let ClosureProofV1::OpaqueBytes { proof_bytes } = &proof;
    assert!(
        proof_bytes.len() < 20 * 1024 * 1024,
        "unexpectedly large proof: {} bytes",
        proof_bytes.len()
    );

    // Tamper statement digest: must fail (digest-binding is statement-bound).
    let mut stmt_bad = stmt.clone();
    stmt_bad.obligations_digest[0] ^= 1;
    assert!(
        verify_closure_v1_with_context_and_bus(&stmt_bad, &proof, Some(&params), Some(&ccs), None).is_err(),
        "tampered statement must be rejected"
    );

    // Tamper proof bytes: must fail.
    let mut proof_bad = proof.clone();
    let ClosureProofV1::OpaqueBytes { proof_bytes } = &mut proof_bad;
    let idx = proof_bytes.len() / 2;
    proof_bytes[idx] ^= 1;
    assert!(
        verify_closure_v1_with_context_and_bus(&stmt, &proof_bad, Some(&params), Some(&ccs), None).is_err(),
        "tampered proof bytes must be rejected"
    );

    // Tamper the sumcheck claimed_sum (keeping the digest-binding proof untouched) and ensure
    // the verifier rejects the mismatch.
    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    struct SumcheckProofV2Mirror {
        claimed_sum_u64: u64,
        round_evals_u64: Vec<Vec<u64>>,
        z_r_u64: u64,
        w_r_u64: u64,
    }

    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    struct PrivatePayloadV1Mirror {
        digest_binding_proof: Vec<u8>,
        sumcheck: SumcheckProofV2Mirror,
        whir_proof_data_u64: Vec<u64>,
    }

    fn decode_envelope(bytes: &[u8]) -> (u32, &[u8]) {
        assert!(bytes.len() >= 16, "envelope too short");
        assert_eq!(&bytes[0..4], b"NCLP", "bad magic");
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        assert_eq!(version, 1, "unexpected envelope version");
        let backend_id = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let payload_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        assert_eq!(bytes.len(), 16 + payload_len, "bad payload length");
        (backend_id, &bytes[16..])
    }

    fn encode_envelope(backend_id: u32, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(16 + payload.len());
        out.extend_from_slice(b"NCLP");
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&backend_id.to_le_bytes());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    fn bincode_opts() -> impl Options {
        bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .reject_trailing_bytes()
            .with_limit(64 * 1024 * 1024)
    }

    let ClosureProofV1::OpaqueBytes { proof_bytes } = &proof;
    let (backend_id, payload_bytes) = decode_envelope(proof_bytes);
    assert_eq!(backend_id, 6, "expected private backend id 6");

    let mut payload: PrivatePayloadV1Mirror = bincode_opts().deserialize(payload_bytes).expect("payload decode");
    payload.sumcheck.claimed_sum_u64 ^= 1;
    let payload_bytes_bad = bincode_opts().serialize(&payload).expect("payload encode");
    let proof_bytes_bad = encode_envelope(backend_id, &payload_bytes_bad);

    let proof_bad = ClosureProofV1::OpaqueBytes { proof_bytes: proof_bytes_bad };
    let err = verify_closure_v1_with_context_and_bus(&stmt, &proof_bad, Some(&params), Some(&ccs), None)
        .expect_err("tampered claimed_sum must be rejected");
    assert!(
        matches!(err, ClosureProofError::WhirP3(ref s) if s.contains("claimed_sum mismatch")),
        "expected claimed_sum mismatch error, got {err:?}"
    );
}
