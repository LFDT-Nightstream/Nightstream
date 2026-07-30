//! Retained red-team regressions for the exported neo-ccs CE membership checker.
#![allow(non_snake_case)]

use neo_ccs::{check_ce_consistency, CcsStructure, CeClaim, CeWitness, Mat, SModuleHomomorphism, SparsePoly};
use neo_fold_clean::config;
use neo_math::{Rq, D, F, K};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

struct TransparentLog;

impl SModuleHomomorphism<F, Vec<F>> for TransparentLog {
    fn commit(&self, z: &Mat<F>) -> Vec<F> {
        z.as_slice().to_vec()
    }
}

fn zero_structure(n: usize, m: usize) -> CcsStructure<F> {
    CcsStructure::new(vec![Mat::zero(n, m, F::ZERO)], SparsePoly::new(1, vec![])).expect("zero one-matrix CCS")
}

fn zero_eval_claim(z: &Mat<F>, X: Mat<F>, m_in: usize, r: Vec<K>) -> CeClaim<Vec<F>, F, K> {
    CeClaim {
        c: TransparentLog.commit(z),
        X,
        r,
        s_col: vec![],
        y_ring: vec![vec![K::ZERO; D]],
        ct: vec![K::ZERO],
        aux_openings: vec![],
        y_zcol: vec![],
        m_in,
        fold_digest: [0; 32],
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        adv: None,
    }
}

#[test]
fn exported_ce_checker_uses_canonical_public_projection_and_shape() {
    let params = NeoParams::goldilocks_paper_b2();

    // For m_in=2, the canonical SuperNeo projection has one active packed
    // ring column: ceil(2 / D) = 1. Packed column 1 is wholly private.
    let structure = zero_structure(2, 2 * D);
    let mut Z = Mat::zero(D, 2, F::ZERO);
    for rho in 0..D {
        Z[(rho, 0)] = F::from_u64((rho + 1) as u64);
        Z[(rho, 1)] = F::from_u64((rho + 101) as u64);
    }
    let canonical_X = neo_reductions::common::project_x_from_witness_mat(&Z, structure.m, 2)
        .expect("canonical production projection");
    let honest = zero_eval_claim(&Z, canonical_X.clone(), 2, vec![K::ZERO]);

    let mut overprojected_X = canonical_X;
    for rho in 0..D {
        overprojected_X[(rho, 1)] = Z[(rho, 1)];
    }
    let overprojected = CeClaim {
        X: overprojected_X,
        ..honest.clone()
    };
    let witness = CeWitness { Z: Z.clone() };
    let honest_result = check_ce_consistency(&params, &structure, &TransparentLog, &honest, &witness);
    let overprojected_result = check_ce_consistency(&params, &structure, &TransparentLog, &overprojected, &witness);

    // The same membership helper also compares only X.as_slice(), ignoring
    // X's declared dimensions, and never checks m_in <= structure.m.
    let narrow_structure = zero_structure(2, D);
    let narrow_Z = Mat::zero(D, 1, F::ZERO);
    let narrow_X = neo_reductions::common::project_x_from_witness_mat(&narrow_Z, narrow_structure.m, 1)
        .expect("canonical narrow projection");
    let wrong_shape_X = Mat::from_row_major(1, D, narrow_X.as_slice().to_vec());
    let wrong_shape = zero_eval_claim(&narrow_Z, wrong_shape_X, 1, vec![K::ZERO]);
    let wrong_shape_result = check_ce_consistency(
        &params,
        &narrow_structure,
        &TransparentLog,
        &wrong_shape,
        &CeWitness { Z: narrow_Z.clone() },
    );

    let too_wide_m_in = D + 1;
    assert!(
        neo_reductions::common::project_x_from_witness_mat(&narrow_Z, narrow_structure.m, too_wide_m_in,).is_err(),
        "production projector rejects a public prefix wider than the CCS"
    );
    let mut too_wide_X = Mat::zero(D, too_wide_m_in, F::ZERO);
    for rho in 0..D {
        too_wide_X[(rho, 0)] = narrow_Z[(rho, 0)];
    }
    let too_wide = zero_eval_claim(&narrow_Z, too_wide_X, too_wide_m_in, vec![K::ZERO]);
    let too_wide_result = check_ce_consistency(
        &params,
        &narrow_structure,
        &TransparentLog,
        &too_wide,
        &CeWitness { Z: narrow_Z },
    );

    assert!(
        honest_result.is_ok()
            && overprojected_result.is_err()
            && wrong_shape_result.is_err()
            && too_wide_result.is_err(),
        "CE projection/shape mismatch: honest={honest_result:?}, overprojected={overprojected_result:?}, wrong_shape={wrong_shape_result:?}, too_wide={too_wide_result:?}"
    );
}

#[test]
fn exported_ce_checker_accepts_ring_action_padding() {
    let params = NeoParams::goldilocks_paper_b2();
    let structure = zero_structure(2, D + 1);

    // A canonical last block with coefficient 0 = 1 becomes coefficient 1 = 1
    // under multiplication by X. That coefficient is past logical width D+1,
    // but is legitimate closure data after Pi_RLC's ring-scalar action.
    let mut input_block = [F::ZERO; D];
    input_block[0] = F::ONE;
    let rotated = Rq(input_block).mul_by_monomial(1);
    assert_eq!(rotated.0[0], F::ZERO);
    assert_eq!(rotated.0[1], F::ONE);

    let mut Z = Mat::zero(D, 2, F::ZERO);
    for rho in 0..D {
        Z[(rho, 1)] = rotated.0[rho];
    }
    neo_reductions::common::validate_superneo_witness_mat(&Z, structure.m)
        .expect("production CE validator permits post-ring-action tail lanes");

    let r = vec![K::ZERO];
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let (y_ring, ct) = neo_reductions::common::compute_y_from_Z_and_r(&structure, &Z, &r, ell_d, params.b);
    let X = neo_reductions::common::project_x_from_witness_mat(&Z, structure.m, 1).expect("production projection");
    let mut claim = zero_eval_claim(&Z, X, 1, r);
    claim.y_ring = y_ring;
    claim.ct = ct;

    let result = check_ce_consistency(&params, &structure, &TransparentLog, &claim, &CeWitness { Z });
    assert!(
        result.is_ok(),
        "CE completeness failure: production-valid post-ring-action padding was rejected: {result:?}"
    );
}

#[test]
fn exported_ce_checker_matches_production_single_row_domain() {
    let structure = zero_structure(1, D);
    let params = config::ccs_params(structure.n, structure.m, structure.t(), structure.max_degree())
        .expect("shape-specific params");
    let dims = neo_reductions::engines::utils::build_dims_and_policy(params.inner(), &structure)
        .expect("production dimension policy");
    assert_eq!(dims.ell_n, 1, "production fixes the row domain at >=2");

    let Z = Mat::zero(D, 1, F::ZERO);
    let X = neo_reductions::common::project_x_from_witness_mat(&Z, structure.m, 1).expect("production projection");
    let production_shaped = zero_eval_claim(&Z, X.clone(), 1, vec![K::ZERO]);
    let zero_round = zero_eval_claim(&Z, X, 1, vec![]);
    let witness = CeWitness { Z };
    let production_result = check_ce_consistency(
        params.inner(),
        &structure,
        &TransparentLog,
        &production_shaped,
        &witness,
    );
    let zero_round_result = check_ce_consistency(params.inner(), &structure, &TransparentLog, &zero_round, &witness);

    assert!(
        production_result.is_ok() && zero_round_result.is_err(),
        "single-row CE language mismatch: production-shaped={production_result:?}, zero-round={zero_round_result:?}"
    );
}

#[test]
fn exported_ce_checker_enforces_the_ce_witness_norm_bound() {
    let params = NeoParams::goldilocks_paper_b2();
    let structure = zero_structure(2, D);

    // Definition 13 requires ||Z||_infinity < b.  Keep every other CE
    // equation honest: the transparent commitment opens to this exact Z,
    // X is its canonical public projection, and the zero matrix evaluates to
    // zero at every row point.
    let mut Z = Mat::zero(D, 1, F::ZERO);
    Z[(0, 0)] = F::from_u64(params.b as u64);
    assert!(
        !neo_math::balanced::within_nc_bound(Z[(0, 0)], params.b),
        "fixture must lie just outside the configured CE(b) alphabet"
    );

    let X = neo_reductions::common::project_x_from_witness_mat(&Z, structure.m, 1)
        .expect("canonical production projection");
    let claim = zero_eval_claim(&Z, X, 1, vec![K::ZERO]);
    let result = check_ce_consistency(&params, &structure, &TransparentLog, &claim, &CeWitness { Z });

    assert!(
        result.is_err(),
        "CE membership accepted a witness with ||Z||_infinity >= b"
    );
}
