#![allow(non_snake_case)]

use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CeClaim, Mat};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

fn eval_zcol(params: &NeoParams, Z: &Mat<F>, s_col: &[K], m: usize, ell_d: usize) -> Vec<K> {
    use neo_ccs::utils::tensor_point;
    let chi_s = tensor_point::<K>(s_col);
    neo_reductions::common::compute_y_zcol_from_witness(params, Z, m, &chi_s, 1usize << ell_d)
        .expect("eval_zcol: valid y_zcol")
}

#[test]
fn dec_reduction_emits_and_checks_y_zcol() {
    let params = NeoParams::goldilocks_paper_b2();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let d_pad = 1usize << ell_d;

    // CCS: n=m=D, t=1 (identity). SuperNeo-only packed shape.
    let s = CcsStructure::new(vec![Mat::identity(D)], neo_ccs::poly::SparsePoly::new(1, vec![])).unwrap();

    let m_in = 1usize;
    let r = vec![
        K::from(F::from_u64(3)),
        K::from(F::from_u64(5)),
        K::from(F::from_u64(7)),
        K::from(F::from_u64(11)),
        K::from(F::from_u64(13)),
        K::from(F::from_u64(17)),
    ]; // ell_n = 6 for n_pad=64
    let s_col = vec![
        K::from(F::from_u64(19)),
        K::from(F::from_u64(23)),
        K::from(F::from_u64(29)),
        K::from(F::from_u64(31)),
        K::from(F::from_u64(37)),
        K::from(F::from_u64(41)),
    ]; // ell_m = 6 for m_pad=64

    // Two DEC digits Z0, Z1 (D×(m/D)).
    let m = s.m;
    let m_commit = m / D;
    let mut z0 = Mat::zero(D, m_commit, F::ZERO);
    let z1 = Mat::zero(D, m_commit, F::ZERO);
    for rho in 0..D {
        for c in 0..m_commit {
            if rho == 0 {
                z0[(rho, c)] = if (rho + c) % 2 == 0 { F::ONE } else { F::ZERO };
            }
        }
    }
    let Z_split = vec![z0.clone(), z1.clone()];

    // Parent witness Z = Z0 + b·Z1.
    let bF = F::from_u64(params.b as u64);
    let mut Z_parent = Mat::zero(D, m_commit, F::ZERO);
    for rho in 0..D {
        for c in 0..m_commit {
            Z_parent[(rho, c)] = z0[(rho, c)] + bF * z1[(rho, c)];
        }
    }

    let X_parent = neo_reductions::common::project_x_from_witness_mat(&Z_parent, m, m_in).expect("X parent");
    let (y_parent, y_scalars_parent) =
        neo_reductions::common::compute_y_from_Z_and_r(&s, &Z_parent, &r, ell_d, params.b);
    assert_eq!(y_parent.len(), s.t());
    assert_eq!(y_parent[0].len(), d_pad);

    let y_zcol_parent = eval_zcol(&params, &Z_parent, &s_col, m, ell_d);
    assert_eq!(y_zcol_parent.len(), d_pad);

    let parent = CeClaim::<Commitment, F, K> {
        adv: None,
        c: Commitment::zeros(params.d as usize, 1),
        X: X_parent,
        r: r.clone(),
        s_col: s_col.clone(),
        y_ring: y_parent,
        ct: y_scalars_parent,
        aux_openings: Vec::new(),
        y_zcol: y_zcol_parent.clone(),
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
    };

    let (children, _ok_y, _ok_X) =
        neo_reductions::optimized_engine::dec_reduction_paper_exact::<F>(&s, &params, &parent, &Z_split, ell_d);
    assert_eq!(children.len(), 2);
    for (i, child) in children.iter().enumerate() {
        assert_eq!(child.s_col, s_col, "child s_col must match parent");
        assert_eq!(child.y_zcol.len(), d_pad, "child y_zcol must be padded");
        let want = eval_zcol(&params, &Z_split[i], &s_col, m, ell_d);
        assert_eq!(child.y_zcol, want, "child y_zcol must equal Z_i · chi(s_col)");
    }

    // Explicitly check y_zcol decomposition: parent = child0 + b·child1.
    let bK = K::from(bF);
    let mut lhs = vec![K::ZERO; d_pad];
    for t in 0..d_pad {
        lhs[t] = children[0].y_zcol[t] + bK * children[1].y_zcol[t];
    }
    assert_eq!(lhs, y_zcol_parent);
}

#[cfg(feature = "paper-exact")]
#[test]
fn dec_reduction_superneo_shape_optimized_matches_paper_exact() {
    let params = NeoParams::goldilocks_paper_b2();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let d_pad = 1usize << ell_d;

    // SuperNeo-compatible shape: m = D; use n = D with identity.
    let s = CcsStructure::new(vec![Mat::identity(D)], neo_ccs::poly::SparsePoly::new(1, vec![])).unwrap();
    let m = s.m;
    let m_in = 5usize;

    // Choose ell_n=6 (n_pad=64 >= D=54), ell_m=6 (m_pad=64 >= D=54).
    let r = vec![
        K::from(F::from_u64(2)),
        K::from(F::from_u64(3)),
        K::from(F::from_u64(5)),
        K::from(F::from_u64(7)),
        K::from(F::from_u64(11)),
        K::from(F::from_u64(13)),
    ];
    let s_col = vec![
        K::from(F::from_u64(17)),
        K::from(F::from_u64(19)),
        K::from(F::from_u64(23)),
        K::from(F::from_u64(29)),
        K::from(F::from_u64(31)),
        K::from(F::from_u64(37)),
    ];

    let m_commit = m / D;
    let mut z0 = Mat::zero(D, m_commit, F::ZERO);
    let mut z1 = Mat::zero(D, m_commit, F::ZERO);
    for rho in 0..D {
        for c in 0..m_commit {
            // Keep coefficients in {0,1} so packed-base decomposition remains carry-free for b=2.
            z0[(rho, c)] = if (rho + c) % 2 == 0 { F::ONE } else { F::ZERO };
            z1[(rho, c)] = if (rho + c) % 3 == 0 { F::ONE } else { F::ZERO };
        }
    }
    let Z_split = vec![z0.clone(), z1.clone()];

    let bF = F::from_u64(params.b as u64);
    let mut Z_parent = Mat::zero(D, m_commit, F::ZERO);
    for rho in 0..D {
        for c in 0..m_commit {
            Z_parent[(rho, c)] = z0[(rho, c)] + bF * z1[(rho, c)];
        }
    }

    let X_parent = neo_reductions::common::project_x_from_witness_mat(&Z_parent, m, m_in).expect("X parent");
    let (y_parent, ct_parent) = neo_reductions::common::compute_y_from_Z_and_r(&s, &Z_parent, &r, ell_d, params.b);
    assert_eq!(y_parent.len(), s.t());
    assert_eq!(y_parent[0].len(), d_pad);
    let y_zcol_parent = eval_zcol(&params, &Z_parent, &s_col, m, ell_d);

    let parent = CeClaim::<Commitment, F, K> {
        adv: None,
        c: Commitment::zeros(params.d as usize, 1),
        X: X_parent,
        r: r.clone(),
        s_col: s_col.clone(),
        y_ring: y_parent,
        ct: ct_parent,
        aux_openings: Vec::new(),
        y_zcol: y_zcol_parent,
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
    };

    let (children_opt, ok_y_opt, ok_x_opt) =
        neo_reductions::optimized_engine::dec_reduction_paper_exact::<F>(&s, &params, &parent, &Z_split, ell_d);

    let (children_paper, ok_y_paper, ok_x_paper) =
        neo_reductions::paper_exact_engine::dec_reduction_paper_exact::<F>(&s, &params, &parent, &Z_split, ell_d);
    assert_eq!(ok_y_opt, ok_y_paper, "optimized/paper y-check status mismatch");
    assert_eq!(ok_x_opt, ok_x_paper, "optimized/paper X-check status mismatch");
    assert_eq!(
        children_opt, children_paper,
        "optimized and paper DEC children must match"
    );
}
