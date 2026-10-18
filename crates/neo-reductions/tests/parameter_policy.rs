use neo_ccs::{poly::Term, CcsMatrix, CcsStructure, SparsePoly};
use neo_math::F;
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

#[test]
fn engine_rejects_parameters_selected_for_fewer_matrices() {
    const SHAPE: usize = 1 << 24;
    const ACTUAL_MATRIX_COUNT: usize = 100_000;
    const DEGREE: u32 = 8;

    let params = NeoParams::goldilocks_auto_ccs_with(SHAPE, 1, DEGREE, 96, 2).expect("one-matrix parameter profile");
    let mut exps = vec![0; ACTUAL_MATRIX_COUNT];
    exps[0] = DEGREE;
    let structure = CcsStructure::new_sparse(
        vec![CcsMatrix::Identity { n: SHAPE }; ACTUAL_MATRIX_COUNT],
        SparsePoly::new(ACTUAL_MATRIX_COUNT, vec![Term { coeff: F::ONE, exps }]),
    )
    .expect("identity relation with many matrices");

    assert!(
        neo_reductions::engines::pi_ccs_joint::build_joint_dims(&params, &structure, 1, params.k_rho as usize).is_err(),
        "engine preprocessing must charge the concrete matrix count"
    );
}
