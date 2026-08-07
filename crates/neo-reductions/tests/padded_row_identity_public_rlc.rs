#![allow(non_snake_case)]

use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CeClaim, Mat, SparsePoly};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::api::{rlc_public, rot_rhos_from_mats};
use neo_reductions::common::project_x_from_witness_mat;
use p3_field::PrimeCharacteristicRing;

fn selected_claim(structure: &CcsStructure<F>, params: &NeoParams) -> CeClaim<Commitment, F, K> {
    let variables = structure
        .n
        .max(structure.m.div_ceil(D) * D)
        .next_power_of_two()
        .max(2)
        .trailing_zeros() as usize;
    let matrix_count = structure.t() + 1;
    let padded_coefficients = D.next_power_of_two();
    CeClaim {
        adv: None,
        c: Commitment::zeros(params.d as usize, params.kappa as usize),
        X: Mat::zero(D, 0, F::ZERO),
        r: vec![K::ZERO; variables],
        y_ring: vec![vec![K::ZERO; padded_coefficients]; matrix_count],
        ct: vec![K::ZERO; matrix_count],
        m_in: 0,
        fold_digest: [0; 32],
    }
}

#[test]
fn public_rlc_accepts_the_selected_shape_and_rejects_a_partial_ring() {
    let params = NeoParams::goldilocks_paper_b2();
    let structure = CcsStructure::new(vec![Mat::identity(D)], SparsePoly::new(1, Vec::new())).expect("test CCS");
    let rho = Mat::identity(D);
    let rhos = rot_rhos_from_mats(&params, &[rho], "selected public RLC").expect("identity rho");
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let claim = selected_claim(&structure, &params);
    let mix_commitments = |_: &[Mat<F>], commitments: &[Commitment]| commitments[0].clone();

    rlc_public(
        &structure,
        &params,
        &rhos,
        std::slice::from_ref(&claim),
        mix_commitments,
        ell_d,
    )
    .expect("selected public RLC");

    let mut partial_ring = selected_claim(&structure, &params);
    partial_ring.m_in = 1;
    partial_ring.X = Mat::zero(D, 1, F::ZERO);
    assert!(rlc_public(&structure, &params, &rhos, &[partial_ring], mix_commitments, ell_d).is_err());

    let witness = Mat::zero(D, 1, F::ZERO);
    assert!(
        project_x_from_witness_mat(&witness, D, 1).is_err(),
        "the public-X constructor must reject a partial ring before it creates a claim"
    );
}
