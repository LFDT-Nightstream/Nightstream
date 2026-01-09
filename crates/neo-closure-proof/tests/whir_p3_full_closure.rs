#![cfg(all(feature = "whir-p3-backend", feature = "whir-p3-obligations-public"))]

use neo_ajtai::{set_global_pp_seeded, AjtaiSModule};
use neo_ccs::{CcsStructure, Mat, SparsePoly};
use neo_ccs::traits::SModuleHomomorphism;
use neo_closure_proof::{
    compute_accumulator_digest_v2, compute_obligations_digest_v1, prove_whir_p3_full_closure_v1,
    verify_closure_v1_with_context_and_bus, ClosureStatementV1,
};
use neo_fold::shard::ShardObligations;
use neo_math::{D, F, K};
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
fn whir_p3_full_closure_roundtrip_and_rejects_bad_y_scalars() {
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
    let (y, y_scalars) =
        neo_reductions::common::compute_y_from_Z_and_r(&ccs, &z, &r_point, ell_d, params.b);

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
    let obligations_digest = compute_obligations_digest_v1(acc_main, acc_val, pp_id_digest);

    let stmt = ClosureStatementV1::new([1u8; 32], pp_id_digest, obligations_digest);
    let proof =
        prove_whir_p3_full_closure_v1(&stmt, &params, &ccs, &obligations, &[z.clone()], &[], None)
            .expect("prove");
    verify_closure_v1_with_context_and_bus(&stmt, &proof, Some(&params), Some(&ccs), None)
        .expect("verify");

    // Corrupt y_scalars but keep y (and thus statement digest) unchanged. Verification must reject.
    let mut obligations_bad = obligations.clone();
    obligations_bad.main[0].y_scalars = y_scalars;
    obligations_bad.main[0].y_scalars[0] += K::ONE;

    let proof_bad = prove_whir_p3_full_closure_v1(
        &stmt,
        &params,
        &ccs,
        &obligations_bad,
        &[z],
        &[],
        None,
    )
    .expect("prove with bad y_scalars (still allowed by statement digest)");

    assert!(
        verify_closure_v1_with_context_and_bus(&stmt, &proof_bad, Some(&params), Some(&ccs), None).is_err(),
        "bad y_scalars must be rejected"
    );
}

#[test]
fn whir_p3_full_closure_rejects_out_of_range_and_bad_y() {
    let m = 16usize;
    let m_in = 4usize;
    let ccs = identity_ccs(m);
    let params = neo_params::NeoParams::goldilocks_auto_r1cs_ccs(m).expect("params");

    let seed = [11u8; 32];
    set_global_pp_seeded(D, params.kappa as usize, m, seed).expect("set_global_pp_seeded");
    let l = AjtaiSModule::from_global_for_dims(D, m).expect("committer");

    let r_point = vec![
        K::from(F::from_u64(3)),
        K::from(F::from_u64(5)),
        K::from(F::from_u64(7)),
        K::from(F::from_u64(11)),
    ];
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let pp_id_digest = neo_ajtai::compute_pp_id_digest_v1(D, m, params.kappa as usize, seed);

    // Out-of-range witness: contains the digit 2 when b=2.
    let mut z_bad = Mat::zero(D, m, F::ZERO);
    z_bad[(0, 0)] = F::from_u64(2);
    let cmt_bad = l.commit(&z_bad);
    let (y_bad, y_scalars_bad) =
        neo_reductions::common::compute_y_from_Z_and_r(&ccs, &z_bad, &r_point, ell_d, params.b);
    let me_bad = neo_ccs::MeInstance {
        c: cmt_bad,
        X: x_prefix(&z_bad, m_in),
        r: r_point.clone(),
        y: y_bad,
        y_scalars: y_scalars_bad,
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    };
    let obligations_bad = ShardObligations {
        main: vec![me_bad],
        val: vec![],
    };
    let acc_main = compute_accumulator_digest_v2(params.b, obligations_bad.main.as_slice());
    let acc_val = compute_accumulator_digest_v2(params.b, obligations_bad.val.as_slice());
    let obligations_digest = compute_obligations_digest_v1(acc_main, acc_val, pp_id_digest);
    let stmt = ClosureStatementV1::new([9u8; 32], pp_id_digest, obligations_digest);

    assert!(
        prove_whir_p3_full_closure_v1(&stmt, &params, &ccs, &obligations_bad, &[z_bad], &[], None).is_err(),
        "out-of-range Z must not be provable"
    );

    // Bad y: keep Z in-range, but corrupt y so ME consistency fails.
    let mut z_ok = Mat::zero(D, m, F::ZERO);
    z_ok[(0, 0)] = F::ONE;
    let cmt_ok = l.commit(&z_ok);
    let (mut y_ok, y_scalars_ok) =
        neo_reductions::common::compute_y_from_Z_and_r(&ccs, &z_ok, &r_point, ell_d, params.b);
    y_ok[0][0] += K::ONE;

    let me_bad_y = neo_ccs::MeInstance {
        c: cmt_ok,
        X: x_prefix(&z_ok, m_in),
        r: r_point,
        y: y_ok,
        y_scalars: y_scalars_ok,
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    };
    let obligations_bad_y = ShardObligations {
        main: vec![me_bad_y],
        val: vec![],
    };
    let acc_main = compute_accumulator_digest_v2(params.b, obligations_bad_y.main.as_slice());
    let acc_val = compute_accumulator_digest_v2(params.b, obligations_bad_y.val.as_slice());
    let obligations_digest = compute_obligations_digest_v1(acc_main, acc_val, pp_id_digest);
    let stmt_bad_y = ClosureStatementV1::new([10u8; 32], pp_id_digest, obligations_digest);

    assert!(
        prove_whir_p3_full_closure_v1(&stmt_bad_y, &params, &ccs, &obligations_bad_y, &[z_ok], &[], None).is_err(),
        "bad y must not be provable"
    );
}
