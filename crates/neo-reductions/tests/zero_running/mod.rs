//! One honest zero running claim for direct PiCCS tests.
//!
//! The digest-only PiCCS statement takes its prior digest from the running
//! claims, so it needs at least one. A zero witness opens to zero at every
//! point, so this claim holds. The test is the caller that authenticates
//! its digest, as the selected lifecycle does with its prior state hash.

use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CeClaim, Mat};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::engines::pi_ccs_joint::build_joint_dims;
use p3_field::PrimeCharacteristicRing;

/// The claim and witness lists to pass as the running inputs.
pub fn zero_running(
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
        c: Commitment::zeros(D, params.kappa as usize),
        X: Mat::zero(D, neo_ccs::superneo_public_x_cols(m_in), F::ZERO),
        r: vec![K::ZERO; variables],
        eval_k: vec![K::ZERO; padded],
        eval_a: vec![vec![K::ZERO; padded]; structure.t()],
        m_in,
        fold_digest: [0; 32],
        adv: None,
    };
    (vec![claim], vec![Mat::zero(D, structure.m.div_ceil(D), F::ZERO)])
}
