#![cfg(all(feature = "whir-p3-backend", feature = "whir-p3-obligations-public"))]

use neo_ajtai::{set_global_pp_seeded, AjtaiSModule};
use neo_ccs::{CcsStructure, Mat, SparsePoly};
use neo_ccs::traits::SModuleHomomorphism;
use neo_closure_proof::{
    compute_accumulator_digest_v2, compute_obligations_digest_v1, prove_explicit_obligation_closure_v1,
    prove_whir_p3_full_closure_v1, verify_closure_v1_with_context_and_bus, ClosureStatementV1,
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
fn explicit_and_whir_full_closure_accept_same_instances() {
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
    let obligations_digest = compute_obligations_digest_v1(acc_main, acc_val, pp_id_digest);
    let stmt = ClosureStatementV1::new([1u8; 32], pp_id_digest, obligations_digest);

    let proof_explicit = prove_explicit_obligation_closure_v1(&stmt, &params, &obligations, &[z.clone()], &[])
        .expect("explicit prove");
    verify_closure_v1_with_context_and_bus(&stmt, &proof_explicit, Some(&params), Some(&ccs), None)
        .expect("explicit verify");

    let proof_whir =
        prove_whir_p3_full_closure_v1(&stmt, &params, &ccs, &obligations, &[z.clone()], &[], None)
            .expect("whir prove");
    verify_closure_v1_with_context_and_bus(&stmt, &proof_whir, Some(&params), Some(&ccs), None)
        .expect("whir verify");

    // Mutate the witness while keeping obligations/statement fixed. Both backends must reject.
    let mut z_tampered = z;
    z_tampered[(0, 0)] = if z_tampered[(0, 0)] == F::ZERO { F::ONE } else { F::ZERO };

    let proof_explicit_bad =
        prove_explicit_obligation_closure_v1(&stmt, &params, &obligations, &[z_tampered.clone()], &[])
            .expect("explicit prove (tampered witness)");
    assert!(
        verify_closure_v1_with_context_and_bus(&stmt, &proof_explicit_bad, Some(&params), Some(&ccs), None).is_err(),
        "explicit backend must reject tampered witness"
    );

    match prove_whir_p3_full_closure_v1(&stmt, &params, &ccs, &obligations, &[z_tampered], &[], None) {
        Ok(proof_whir_bad) => assert!(
            verify_closure_v1_with_context_and_bus(&stmt, &proof_whir_bad, Some(&params), Some(&ccs), None).is_err(),
            "whir backend must reject tampered witness"
        ),
        Err(_) => {}
    }
}
