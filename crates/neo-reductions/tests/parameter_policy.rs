use neo_params::NeoParams;

#[test]
fn engine_rejects_parameters_selected_for_fewer_matrices() {
    const SHAPE: usize = 1 << 24;
    const ACTUAL_MATRIX_COUNT: usize = 100_000;

    let params = NeoParams::goldilocks_auto_ccs_with(SHAPE, 1, 8, 96, 2).expect("one-matrix parameter profile");

    assert!(
        neo_reductions::engines::pi_ccs_joint::build_joint_dims_for_shape(
            &params,
            SHAPE,
            SHAPE,
            ACTUAL_MATRIX_COUNT,
            8,
            1,
            params.k_rho as usize,
        )
        .is_err(),
        "engine preprocessing must charge the concrete matrix count"
    );
}
