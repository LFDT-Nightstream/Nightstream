use neo_fold::pi_ccs::checks::validate_inputs;
use neo_params::NeoParams;
use neo_math::{F, K};
use neo_ccs::{MeInstance, Mat};
use crate::helpers::{DummyS, create_test_ccs, mk_mcs};
use p3_field::PrimeCharacteristicRing;
use neo_ccs::SModuleHomomorphism;

#[test]
fn validate_rejects_me_inputs_with_different_r() {
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let s = create_test_ccs().ensure_identity_first().unwrap();
    let l = DummyS;
    let m_in = 1;

    let (mcs1, wit1) = mk_mcs(&params, vec![F::ONE, F::from_u64(2), F::from_u64(3), F::ZERO, F::from_u64(5)], m_in, &l);

    // Two ME inputs with different r
    let dummy_me = |rbit: u64| MeInstance {
        c_step_coords: vec![], u_offset: 0, u_len: 0,
        c: mcs1.c.clone(), X: l.project_x(&wit1.Z, mcs1.m_in),
        r: vec![K::from(F::from_u64(rbit)), K::from(F::from_u64(1))],
        y: vec![vec![K::from(F::ZERO); 1<<1]; s.t()],
        y_scalars: vec![K::from(F::ZERO); s.t()],
        m_in: mcs1.m_in, fold_digest: [0u8;32],
    };

    let me_inputs = vec![dummy_me(0), dummy_me(1)];
    let me_wits: Vec<Mat<F>> = vec![wit1.Z.clone(), wit1.Z.clone()];
    let err = validate_inputs(&s, &[mcs1], &[wit1], &me_inputs, &me_wits).unwrap_err();
    assert!(format!("{:?}", err).contains("all ME inputs must share the same r"));
}
