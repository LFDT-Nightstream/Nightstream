use bellpepper_core::test_cs::TestConstraintSystem;
use neo_ajtai::{set_global_pp_seeded, AjtaiSModule, Commitment};
use neo_ccs::{
    build_superneo_ring_forms, CcsStructure, CcsWitness, CeClaim, Mat, SModuleHomomorphism, SparsePoly, Term,
};
use neo_fold_next::rv64im::main_relation_circuit::ce_consistency::enforce_paper_dec_child_claim_consistency;
use neo_fold_next::rv64im::main_relation_circuit::claim::alloc_ce_claim;
use neo_fold_next::rv64im::main_relation_circuit::witness::alloc_packed_witness;
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::common::{
    compute_y_zcol_from_witness, decode_superneo_coeffs_from_witness_mat, project_x_from_witness_mat,
};
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

fn toy_witness() -> CcsWitness<F> {
    let mut z = Mat::zero(D, 1, F::ZERO);
    z[(0, 0)] = F::from_u64(1);
    z[(1, 0)] = F::from_i64(-1);
    z[(2, 0)] = F::ONE;
    CcsWitness {
        w: vec![F::from_i64(-1), F::ONE],
        Z: z,
    }
}

fn toy_child_claim(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    log: &AjtaiSModule,
    witness: &CcsWitness<F>,
) -> CeClaim<Commitment, F, K> {
    let x = project_x_from_witness_mat(&witness.Z, structure.m, 2).expect("project x");
    let r = Vec::new();
    let s_col = vec![K::ZERO, K::ZERO];
    let chi_s = neo_ccs::utils::tensor_point::<K>(&s_col);
    let y_zcol = compute_y_zcol_from_witness(params, &witness.Z, structure.m, &chi_s, D.next_power_of_two())
        .expect("compute y_zcol");
    let z_coeffs = decode_superneo_coeffs_from_witness_mat(&witness.Z, structure.m).expect("decode z coeffs");
    let ring_forms = build_superneo_ring_forms(structure, &r).expect("build ring forms");
    let y_ring = ring_forms
        .iter()
        .map(|forms| {
            let mut row = vec![K::ZERO; D.next_power_of_two()];
            for logical_col in 0..structure.m {
                for rho in 0..D {
                    row[rho] += forms[logical_col][rho] * z_coeffs[logical_col];
                }
            }
            row
        })
        .collect::<Vec<_>>();
    CeClaim {
        c: log.commit(&witness.Z),
        X: x,
        r,
        s_col,
        y_ring: y_ring.clone(),
        ct: vec![y_ring[0][0]],
        aux_openings: Vec::new(),
        y_zcol,
        m_in: 2,
        fold_digest: [0u8; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    }
}

fn ensure_toy_pp() {
    let _ = set_global_pp_seeded(D, 1, 1, [7u8; 32]);
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_main_relation_dec_child_projection_accepts_paper_ce_even_with_tampered_nc_artifacts() {
    ensure_toy_pp();
    let params = NeoParams::goldilocks_auto_r1cs_ccs(3).expect("params");
    let structure = toy_structure();
    let witness = toy_witness();
    let log = AjtaiSModule::from_global_for_dims(D, witness.Z.cols()).expect("ajtai log");
    let mut claim = toy_child_claim(&params, &structure, &log, &witness);
    claim.s_col[0] = K::from(F::from_u64(9));
    claim.y_zcol[0] += K::ONE;
    claim.ct[0] += K::ONE;

    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let witness_var = alloc_packed_witness(&mut cs, &witness, "witness").expect("alloc witness");
    let claim_var = alloc_ce_claim(&mut cs, &claim, "claim").expect("alloc claim");
    enforce_paper_dec_child_claim_consistency(
        &mut cs,
        &params,
        &structure,
        &structure,
        &witness_var,
        &claim_var,
        "child",
    )
    .expect("paper child consistency");

    assert!(cs.is_satisfied(), "{}", cs.which_is_unsatisfied().unwrap_or_default());
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_main_relation_dec_child_projection_accepts_tampered_padded_y_ring_tail() {
    ensure_toy_pp();
    let params = NeoParams::goldilocks_auto_r1cs_ccs(3).expect("params");
    let structure = toy_structure();
    let witness = toy_witness();
    let log = AjtaiSModule::from_global_for_dims(D, witness.Z.cols()).expect("ajtai log");
    let mut claim = toy_child_claim(&params, &structure, &log, &witness);
    claim.y_ring[0][D] += K::ONE;

    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let witness_var = alloc_packed_witness(&mut cs, &witness, "witness").expect("alloc witness");
    let claim_var = alloc_ce_claim(&mut cs, &claim, "claim").expect("alloc claim");
    enforce_paper_dec_child_claim_consistency(
        &mut cs,
        &params,
        &structure,
        &structure,
        &witness_var,
        &claim_var,
        "child",
    )
    .expect("paper child consistency");

    assert!(cs.is_satisfied(), "{}", cs.which_is_unsatisfied().unwrap_or_default());
}

#[test]
#[ignore = "Spartan-path tests are parked until native NIFS and F' replacement lands"]
fn rv64im_main_relation_dec_child_projection_rejects_tampered_y_ring() {
    ensure_toy_pp();
    let params = NeoParams::goldilocks_auto_r1cs_ccs(3).expect("params");
    let structure = toy_structure();
    let witness = toy_witness();
    let log = AjtaiSModule::from_global_for_dims(D, witness.Z.cols()).expect("ajtai log");
    let mut claim = toy_child_claim(&params, &structure, &log, &witness);
    claim.y_ring[0][0] += K::ONE;

    let mut cs = TestConstraintSystem::<SpartanF>::new();
    let witness_var = alloc_packed_witness(&mut cs, &witness, "witness").expect("alloc witness");
    let claim_var = alloc_ce_claim(&mut cs, &claim, "claim").expect("alloc claim");
    enforce_paper_dec_child_claim_consistency(
        &mut cs,
        &params,
        &structure,
        &structure,
        &witness_var,
        &claim_var,
        "child",
    )
    .expect("paper child consistency");

    assert!(!cs.is_satisfied(), "tampered y_ring must fail constraints");
}
