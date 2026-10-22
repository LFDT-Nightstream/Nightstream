//! One honest zero running claim for direct PiCCS tests.

use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CeClaim, Mat};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::engines::pi_ccs_joint::build_joint_dims;
use p3_field::PrimeCharacteristicRing;

pub(crate) fn zero_running(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    fresh_count: usize,
    m_in: usize,
) -> (Vec<CeClaim<Commitment, F, K>>, Vec<Mat<F>>) {
    let variables = build_joint_dims(params, structure, fresh_count, 1)
        .expect("joint dimensions")
        .variables;
    let padded = D.next_power_of_two();
    let claim = CeClaim {
        c: Commitment::zeros(D, params.kappa() as usize),
        X: Mat::zero(D, neo_ccs::superneo_public_x_cols(m_in).unwrap(), F::ZERO),
        r: vec![K::ZERO; variables],
        eval_k: vec![K::ZERO; padded],
        eval_a: vec![vec![K::ZERO; padded]; structure.t()],
        m_in,
        fold_digest: [0; 32],
        adv: None,
    };
    (vec![claim], vec![Mat::zero(D, structure.m.div_ceil(D), F::ZERO)])
}
