//! The PiCCS fresh-source gate follows the RLC guard of the selected profile.

use neo_ccs::{CcsStructure, Mat, SparsePoly};
use neo_math::{D, F};
use neo_params::NeoParams;
use neo_reductions::engines::pi_ccs_joint::build_joint_dims;
use p3_field::PrimeCharacteristicRing;

#[test]
fn production_profile_accepts_every_fresh_count_its_guard_allows() {
    let structure =
        CcsStructure::new(vec![Mat::zero(1, D, F::ZERO)], SparsePoly::new(1, Vec::new())).expect("structure");
    let mut params = NeoParams::nightstream_goldilocks_k16();
    // The production minimum; `nightstream` lowers lambda to what the census provides.
    params = params.with_lambda(114).unwrap();
    let cap = params.max_fresh_count_from_rlc_guard().expect("guard") as usize;
    assert_eq!(cap, 287);
    build_joint_dims(&params, &structure, cap, 0).expect("the guard's own maximum");
    assert!(
        build_joint_dims(&params, &structure, cap + 1, 0).is_err(),
        "one source above the guard"
    );
}
