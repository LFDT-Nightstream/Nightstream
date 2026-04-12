use bellpepper_core::{num::AllocatedNum, test_cs::TestConstraintSystem, ConstraintSystem};
use neo_ajtai::Commitment;
use neo_ccs::{CeClaim, Mat};
use neo_fold_next::rv64im::main_relation_circuit::claim::{
    alloc_ce_claim, me_digest_poseidon, me_digest_poseidon_values,
};
use neo_math::{D, F, K};
use neo_reductions::engines::utils::me_digest_poseidon as native_me_digest_poseidon;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use spartan2::provider::goldi::F as SpartanF;

fn alloc_f_constant(
    cs: &mut TestConstraintSystem<SpartanF>,
    value: SpartanF,
    label: &str,
) -> Result<AllocatedNum<SpartanF>, bellpepper_core::SynthesisError> {
    let out = AllocatedNum::alloc(cs.namespace(|| label.to_string()), || Ok(value))?;
    cs.enforce(
        || format!("{label}_constant"),
        |lc| lc + out.get_variable(),
        |lc| lc + TestConstraintSystem::<SpartanF>::one(),
        |lc| lc + (value, TestConstraintSystem::<SpartanF>::one()),
    );
    Ok(out)
}

fn toy_claim() -> CeClaim<Commitment, F, K> {
    let mut x = Mat::zero(D, 2, F::ZERO);
    x[(0, 0)] = F::from_u64(3);
    x[(1, 1)] = F::from_u64(5);
    CeClaim {
        c: Commitment {
            d: D,
            kappa: 1,
            data: vec![F::from_u64(7); D],
        },
        X: x,
        r: vec![K::from(F::from_u64(11)), K::from(F::from_u64(13))],
        s_col: vec![K::from(F::from_u64(17)), K::from(F::from_u64(19))],
        y_ring: vec![
            vec![K::from(F::from_u64(23)), K::from(F::from_u64(29))],
            vec![K::from(F::from_u64(31)), K::from(F::from_u64(37))],
        ],
        ct: vec![K::from(F::from_u64(23)), K::from(F::from_u64(31))],
        aux_openings: vec![K::from(F::from_u64(41))],
        y_zcol: vec![K::from(F::from_u64(43)); 64],
        m_in: 2,
        fold_digest: [9u8; 32],
        c_step_coords: vec![F::from_u64(47), F::from_u64(53)],
        u_offset: 1,
        u_len: 2,
    }
}

#[test]
fn rv64im_main_relation_claim_digest_matches_native_poseidon() {
    let claim = toy_claim();
    let native_digest = native_me_digest_poseidon(&claim);

    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let claim_var = alloc_ce_claim(&mut cs, &claim, "claim").expect("alloc claim");
    assert_eq!(
        me_digest_poseidon_values(&claim_var),
        native_digest.map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()))
    );
    let digest = me_digest_poseidon(&mut cs, &claim_var, "claim_digest").expect("digest claim");

    for (idx, expected) in native_digest.iter().enumerate() {
        let expected_var = alloc_f_constant(
            &mut cs,
            SpartanF::from_canonical_u64(expected.as_canonical_u64()),
            &format!("expected_digest_{idx}"),
        )
        .expect("alloc expected digest");
        cs.enforce(
            || format!("digest_eq_{idx}"),
            |lc| lc + digest[idx].get_variable(),
            |lc| lc + TestConstraintSystem::<SpartanF>::one(),
            |lc| lc + expected_var.get_variable(),
        );
    }

    assert!(cs.is_satisfied(), "{}", cs.which_is_unsatisfied().unwrap_or_default());
}
