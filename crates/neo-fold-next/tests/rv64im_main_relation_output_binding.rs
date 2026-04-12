use bellpepper_core::test_cs::TestConstraintSystem;
use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CcsStructure, CeClaim, Mat, SparsePoly, Term};
use neo_fold_next::rv64im::main_relation_circuit::claim::alloc_ce_claim;
use neo_fold_next::rv64im::main_relation_circuit::k_field::alloc_constant_k;
use neo_fold_next::rv64im::main_relation_circuit::output_binding::enforce_me_outputs_against_inputs;
use neo_math::{D, F, K};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use spartan2::provider::goldi::F as SpartanF;

fn toy_structure() -> CcsStructure<F> {
    let matrix = Mat::from_row_major(1, 3, vec![F::from_u64(3), F::from_u64(5), F::from_u64(7)]);
    CcsStructure::new(
        vec![matrix],
        SparsePoly::new(
            1,
            vec![Term {
                coeff: F::ONE,
                exps: vec![1],
            }],
        ),
    )
    .expect("toy structure")
}

fn fresh_claim(x0: i64, x1: i64) -> CcsClaim<Commitment, F> {
    CcsClaim {
        c: Commitment::zeros(D, 1),
        x: vec![F::from_i64(x0), F::from_i64(x1)],
        m_in: 2,
    }
}

fn fresh_output(fresh: &CcsClaim<Commitment, F>, r: &[K], s_col: &[K]) -> CeClaim<Commitment, F, K> {
    let mut x = Mat::zero(D, fresh.m_in, F::ZERO);
    for col in 0..fresh.m_in {
        x[(col % D, col)] = fresh.x[col];
    }
    let mut row = vec![K::ZERO; D.next_power_of_two()];
    row[0] = K::from(F::from_u64(9));
    CeClaim {
        c: fresh.c.clone(),
        X: x,
        r: r.to_vec(),
        s_col: s_col.to_vec(),
        y_ring: vec![row.clone()],
        ct: vec![row[0]],
        aux_openings: Vec::new(),
        y_zcol: vec![K::ZERO; D.next_power_of_two()],
        m_in: fresh.m_in,
        fold_digest: [0u8; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    }
}

#[test]
fn rv64im_main_relation_output_binding_accepts_fresh_and_me_inputs() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(3).expect("params");
    let structure = toy_structure();
    let fresh = fresh_claim(3, 5);
    let r = vec![K::from(F::from_u64(11))];
    let s_col = vec![K::from(F::from_u64(13)), K::from(F::from_u64(17))];
    let fresh_out = fresh_output(&fresh, &r, &s_col);
    let carried_out = fresh_output(&fresh, &r, &s_col);

    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let fresh_out_var = alloc_ce_claim(&mut cs, &fresh_out, "fresh_out").expect("alloc fresh output");
    let carried_in_var = alloc_ce_claim(&mut cs, &carried_out, "carried_in").expect("alloc carried input");
    let carried_out_var = alloc_ce_claim(&mut cs, &carried_out, "carried_out").expect("alloc carried output");
    let r_vars = r
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            alloc_constant_k(
                &mut cs,
                neo_fold_next::rv64im::main_relation_circuit::k_field::KNum::from_neo_k(*value),
                &format!("r_{idx}"),
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .expect("alloc r");
    let s_col_vars = s_col
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            alloc_constant_k(
                &mut cs,
                neo_fold_next::rv64im::main_relation_circuit::k_field::KNum::from_neo_k(*value),
                &format!("s_col_{idx}"),
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .expect("alloc s_col");

    enforce_me_outputs_against_inputs(
        &mut cs,
        &structure,
        &params,
        &[fresh],
        &[carried_in_var],
        &[fresh_out_var, carried_out_var],
        &r_vars,
        &r,
        &s_col_vars,
        &s_col,
        "outputs",
    )
    .expect("enforce outputs");

    assert!(cs.is_satisfied(), "{}", cs.which_is_unsatisfied().unwrap_or_default());
}

#[test]
fn rv64im_main_relation_output_binding_rejects_tampered_fresh_commitment() {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(3).expect("params");
    let structure = toy_structure();
    let fresh = fresh_claim(3, 5);
    let r = vec![K::from(F::from_u64(11))];
    let s_col = vec![K::from(F::from_u64(13)), K::from(F::from_u64(17))];
    let mut fresh_out = fresh_output(&fresh, &r, &s_col);
    fresh_out.c.data[0] += F::ONE;

    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let fresh_out_var = alloc_ce_claim(&mut cs, &fresh_out, "fresh_out").expect("alloc fresh output");
    let r_vars = r
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            alloc_constant_k(
                &mut cs,
                neo_fold_next::rv64im::main_relation_circuit::k_field::KNum::from_neo_k(*value),
                &format!("r_{idx}"),
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .expect("alloc r");
    let s_col_vars = s_col
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            alloc_constant_k(
                &mut cs,
                neo_fold_next::rv64im::main_relation_circuit::k_field::KNum::from_neo_k(*value),
                &format!("s_col_{idx}"),
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .expect("alloc s_col");

    enforce_me_outputs_against_inputs(
        &mut cs,
        &structure,
        &params,
        &[fresh],
        &[],
        &[fresh_out_var],
        &r_vars,
        &r,
        &s_col_vars,
        &s_col,
        "outputs",
    )
    .expect("enforce outputs");

    assert!(!cs.is_satisfied(), "tampered fresh commitment must fail");
}
