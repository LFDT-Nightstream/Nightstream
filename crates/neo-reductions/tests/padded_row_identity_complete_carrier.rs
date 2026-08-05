use neo_ccs::{CcsStructure, Mat, SparsePoly};
use neo_math::{D, F};
use neo_params::NeoParams;
use neo_reductions::common::{validate_fresh_witness_tail_zero, validate_packed_witness_nc_alphabet};
use neo_reductions::engines::pi_ccs_joint::build_joint_dims;
use p3_field::PrimeCharacteristicRing;

#[test]
fn joint_row_domain_uses_the_complete_packed_carrier() {
    let logical_width = D + 1;
    let structure = CcsStructure::new(
        vec![Mat::zero(1, logical_width, F::ZERO)],
        SparsePoly::new(1, Vec::new()),
    )
    .expect("valid rectangular structure");
    let params = NeoParams::goldilocks_auto_r1cs_ccs(logical_width).expect("parameters");
    let dimensions = build_joint_dims(&params, &structure, 1, 0).expect("dimensions");

    assert_eq!(logical_width, 55);
    assert_eq!(logical_width.div_ceil(D) * D, 108);
    assert_eq!(
        dimensions.variables, 7,
        "the joint row cube must cover all 108 carrier coordinates"
    );
}

#[test]
fn norm_check_covers_the_first_completed_carrier_coordinate() {
    let params = NeoParams::goldilocks_paper_b2();
    let logical_width: usize = 257;
    let carrier_width = logical_width.div_ceil(D) * D;
    let mut witness = Mat::zero(D, logical_width.div_ceil(D), F::ZERO);

    witness[(logical_width % D, logical_width / D)] = F::ONE;
    assert!(
        validate_packed_witness_nc_alphabet(&params, &witness, logical_width, "running").is_ok(),
        "a valid running value in the completed carrier must be preserved"
    );

    witness[(logical_width % D, logical_width / D)] = F::from_u64(2);
    let error = validate_packed_witness_nc_alphabet(&params, &witness, logical_width, "running")
        .expect_err("b=2 must reject the value 2 in the completed carrier");
    assert!(error
        .to_string()
        .contains(&format!("carrier_col={logical_width}")));
    assert_eq!(carrier_width, 270);
}

#[test]
fn fresh_sources_must_zero_the_completed_carrier_tail() {
    let logical_width: usize = 257;
    let mut witness = Mat::zero(D, logical_width.div_ceil(D), F::ZERO);

    assert!(validate_fresh_witness_tail_zero(&witness, logical_width, "fresh").is_ok());

    witness[(logical_width % D, logical_width / D)] = F::ONE;
    let error = validate_fresh_witness_tail_zero(&witness, logical_width, "fresh")
        .expect_err("fresh sources do not own the completed carrier tail");
    assert!(error
        .to_string()
        .contains(&format!("carrier_col={logical_width}")));
}
