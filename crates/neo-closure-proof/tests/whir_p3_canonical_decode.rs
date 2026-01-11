use bincode::Options;
use neo_ajtai::{set_global_pp_seeded, AjtaiSModule};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsStructure, Mat, SparsePoly};
use neo_closure_proof::{
    compute_accumulator_digest_v2, compute_obligations_digest_v2, prove_whir_p3_full_closure_v1,
    verify_closure_v1_with_context_and_bus, ClosureProofError, ClosureProofV1, ClosureStatementV1,
};
use neo_fold::shard::ShardObligations;
use neo_math::{D, F, K};
use p3_field::PrimeCharacteristicRing;

fn bincode_opts() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .reject_trailing_bytes()
}

fn decode_envelope(bytes: &[u8]) -> (u32, Vec<u8>) {
    const HEADER_LEN: usize = 16;
    assert!(bytes.len() >= HEADER_LEN);
    assert_eq!(&bytes[0..4], b"NCLP");
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    assert_eq!(version, 1);
    let backend_id = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let payload_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    assert_eq!(bytes.len(), HEADER_LEN + payload_len);
    (backend_id, bytes[HEADER_LEN..].to_vec())
}

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

#[allow(dead_code)]
mod payload_codec {
    #[derive(Clone, serde::Serialize, serde::Deserialize)]
    pub(super) struct EncodedK {
        pub c0: u64,
        pub c1: u64,
    }

    #[derive(Clone, serde::Serialize, serde::Deserialize)]
    pub(super) struct EncodedCommitment {
        pub d: u32,
        pub kappa: u32,
        pub data: Vec<u64>,
    }

    #[derive(Clone, serde::Serialize, serde::Deserialize)]
    pub(super) struct EncodedMatF {
        pub rows: u32,
        pub cols: u32,
        pub data: Vec<u64>,
    }

    #[derive(Clone, serde::Serialize, serde::Deserialize)]
    pub(super) struct EncodedMeInstance {
        pub c: EncodedCommitment,
        pub x: EncodedMatF,
        pub r: Vec<EncodedK>,
        pub y: Vec<Vec<EncodedK>>,
        pub y_scalars: Vec<EncodedK>,
        pub m_in: u32,
    }

    #[derive(Clone, serde::Serialize, serde::Deserialize)]
    pub(super) struct EncodedObligations {
        pub main: Vec<EncodedMeInstance>,
        pub val: Vec<EncodedMeInstance>,
    }

    #[derive(Clone, serde::Serialize, serde::Deserialize)]
    pub(super) struct SumcheckProofV2 {
        pub round_evals_u64: Vec<Vec<u64>>,
        pub z_r_u64: u64,
        pub w_r_u64: u64,
    }

    #[derive(Clone, serde::Serialize, serde::Deserialize)]
    pub(super) struct FullClosurePayloadV1 {
        pub obligations: EncodedObligations,
        pub sumcheck: SumcheckProofV2,
        pub whir_proof_data_u64: Vec<u64>,
    }
}

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
fn whir_full_closure_rejects_non_canonical_u64_encoding() {
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
        y_scalars,
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
    let proof = prove_whir_p3_full_closure_v1(&stmt, &params, &ccs, &obligations, &[z], &[], None).expect("prove");
    verify_closure_v1_with_context_and_bus(&stmt, &proof, Some(&params), Some(&ccs), None).expect("verify");

    let ClosureProofV1::OpaqueBytes { proof_bytes } = proof;

    let (backend_id, payload_bytes) = decode_envelope(&proof_bytes);
    let mut payload: payload_codec::FullClosurePayloadV1 = bincode_opts()
        .deserialize(&payload_bytes)
        .expect("decode full-closure payload");
    assert!(
        !payload.whir_proof_data_u64.is_empty(),
        "expected non-empty whir_proof_data_u64"
    );
    payload.whir_proof_data_u64[0] = u64::MAX;
    let tampered_payload = bincode_opts()
        .serialize(&payload)
        .expect("encode tampered payload");

    let tampered_proof = ClosureProofV1::OpaqueBytes {
        proof_bytes: encode_envelope(backend_id, &tampered_payload),
    };

    let err = verify_closure_v1_with_context_and_bus(&stmt, &tampered_proof, Some(&params), Some(&ccs), None)
        .expect_err("must reject non-canonical encoding");
    assert!(
        matches!(err, ClosureProofError::InvalidOpaqueProofEncoding),
        "expected InvalidOpaqueProofEncoding, got {err:?}"
    );
}
