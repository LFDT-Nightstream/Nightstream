use neo_ccs::{poly::Term, CcsMatrix, CcsStructure, Mat, SparsePoly};
use neo_math::F;
use neo_params::{NeoParams, ParamsError};
use neo_reductions::PiCcsError;
use p3_field::PrimeCharacteristicRing;

#[test]
fn engine_rejects_parameters_selected_for_fewer_matrices() {
    const SHAPE: usize = 1 << 24;
    const ACTUAL_MATRIX_COUNT: usize = 100_000;
    const DEGREE: u32 = 8;

    let base = NeoParams::nightstream_p64_k14();
    let one_matrix = base
        .padded_row_security_summary_for_shape(
            SHAPE,
            SHAPE,
            1,
            DEGREE,
            neo_params::nightstream_p64_k14::CHALLENGE_ALPHABET.len() as u32,
        )
        .expect("one-matrix census");
    let params = base
        .with_lambda(one_matrix.security_bits)
        .expect("one-matrix parameter profile");
    let mut exps = vec![0; ACTUAL_MATRIX_COUNT];
    exps[0] = DEGREE;
    let structure = CcsStructure::new_sparse(
        vec![CcsMatrix::Identity { n: SHAPE }; ACTUAL_MATRIX_COUNT],
        SparsePoly::new(ACTUAL_MATRIX_COUNT, vec![Term { coeff: F::ONE, exps }]),
    )
    .expect("identity relation with many matrices");

    assert!(
        neo_reductions::engines::pi_ccs_joint::build_joint_dims(&params, &structure, 1, params.k_rho() as usize)
            .is_err(),
        "engine preprocessing must charge the concrete matrix count"
    );
}

#[test]
fn engine_refuses_parameters_that_no_gate_certified() {
    let ungated = NeoParams::nightstream_p64_k14();
    let structure = CcsStructure::new(vec![Mat::identity(1)], SparsePoly::new(1, Vec::new())).expect("minimal CCS");
    let dims = |params: &NeoParams| neo_reductions::engines::pi_ccs_joint::build_joint_dims(params, &structure, 1, 0);

    assert!(
        matches!(
            dims(&ungated),
            Err(PiCcsError::ExtensionPolicyFailed(message))
                if message == ParamsError::UncertifiedStatisticalSecurity.to_string()
        ),
        "the engine accepted parameters that no gate certified"
    );
    let bits = ungated
        .padded_row_security_summary_for_shape(
            1,
            1,
            1,
            0,
            neo_params::nightstream_p64_k14::CHALLENGE_ALPHABET.len() as u32,
        )
        .expect("census")
        .security_bits;
    assert!(dims(&ungated.with_lambda(bits).unwrap()).is_ok());
    assert!(dims(&ungated.with_lambda(bits + 1).unwrap()).is_err());
}
