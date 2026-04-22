#![allow(non_snake_case)]

use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CeClaim, Mat};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

#[test]
fn rlc_public_mixes_y_zcol_when_present() {
    let params = NeoParams::goldilocks_127();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let d_pad = 1usize << ell_d;

    // Minimal SuperNeo-compatible CCS structure: only `t` matters for Π_RLC public mixing.
    let s = CcsStructure::new(vec![Mat::identity(D)], neo_ccs::poly::SparsePoly::new(1, vec![])).unwrap();

    let m_in = 1usize;
    let r = vec![K::from(F::from_u64(3)), K::from(F::from_u64(5))];
    let s_col = vec![K::from(F::from_u64(7))];

    let mut X0 = Mat::zero(D, m_in, F::ZERO);
    let mut X1 = Mat::zero(D, m_in, F::ZERO);
    X0[(0, 0)] = F::from_u64(11);
    X1[(0, 0)] = F::from_u64(13);

    let mut y0 = vec![vec![K::ZERO; d_pad]];
    let mut y1 = vec![vec![K::ZERO; d_pad]];
    y0[0][0] = K::from(F::from_u64(17));
    y1[0][0] = K::from(F::from_u64(19));

    let y_scalars0 = vec![y0[0][0]];
    let y_scalars1 = vec![y1[0][0]];

    let mut y_zcol0 = vec![K::ZERO; d_pad];
    let mut y_zcol1 = vec![K::ZERO; d_pad];
    y_zcol0[0] = K::from(F::from_u64(23));
    y_zcol1[0] = K::from(F::from_u64(29));

    let aux0 = vec![K::from(F::from_u64(31)), K::from(F::from_u64(37))];
    let aux1 = vec![K::from(F::from_u64(41)), K::from(F::from_u64(43))];

    let inst0 = CeClaim::<Commitment, F, K> {
        c: Commitment::zeros(params.d as usize, 1),
        X: X0,
        r: r.clone(),
        s_col: s_col.clone(),
        y_ring: y0,
        ct: y_scalars0,
        aux_openings: aux0.clone(),
        y_zcol: y_zcol0.clone(),
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
    };
    let inst1 = CeClaim::<Commitment, F, K> {
        c: Commitment::zeros(params.d as usize, 1),
        X: X1,
        r,
        s_col,
        y_ring: y1,
        ct: y_scalars1,
        aux_openings: aux1.clone(),
        y_zcol: y_zcol1.clone(),
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
    };

    // ρ_0 = I, ρ_1 = 2·I.
    let rho0 = Mat::identity(D);
    let mut rho1 = Mat::identity(D);
    for i in 0..D {
        rho1.set(i, i, F::from_u64(2));
    }
    let rhos = vec![rho0, rho1];
    let rhos_typed =
        neo_reductions::api::rot_rhos_from_mats(&params, &rhos, "rlc_public_y_zcol:test rhos").expect("typed rhos");

    let out = neo_reductions::api::rlc_public(
        &s,
        &params,
        &rhos_typed,
        &[inst0, inst1],
        |_rhos, _cs| Commitment::zeros(params.d as usize, 1),
        ell_d,
    )
    .expect("rlc_public");

    // Expect y_zcol_out = y_zcol0 + 2·y_zcol1 (first D entries; rest padding stays 0).
    let mut expected = vec![K::ZERO; d_pad];
    for rho in 0..D {
        expected[rho] = y_zcol0[rho] + K::from(F::from_u64(2)) * y_zcol1[rho];
    }
    assert_eq!(out.y_zcol, expected);
    assert_eq!(
        out.aux_openings,
        vec![
            aux0[0] + K::from(F::from_u64(2)) * aux1[0],
            aux0[1] + K::from(F::from_u64(2)) * aux1[1]
        ]
    );
    assert_eq!(out.s_col.len(), 1);
}

#[test]
fn rlc_public_matches_ignores_y_zcol_and_s_col_shell() {
    let params = NeoParams::goldilocks_127();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let d_pad = 1usize << ell_d;
    let s = CcsStructure::new(vec![Mat::identity(D)], neo_ccs::poly::SparsePoly::new(1, vec![])).unwrap();

    let m_in = 1usize;
    let r = vec![K::from(F::from_u64(3)), K::from(F::from_u64(5))];
    let s_col = vec![K::from(F::from_u64(7))];

    let mut X0 = Mat::zero(D, m_in, F::ZERO);
    let mut X1 = Mat::zero(D, m_in, F::ZERO);
    X0[(0, 0)] = F::from_u64(11);
    X1[(0, 0)] = F::from_u64(13);

    let mut y0 = vec![vec![K::ZERO; d_pad]];
    let mut y1 = vec![vec![K::ZERO; d_pad]];
    y0[0][0] = K::from(F::from_u64(17));
    y1[0][0] = K::from(F::from_u64(19));

    let y_scalars0 = vec![y0[0][0]];
    let y_scalars1 = vec![y1[0][0]];

    let mut y_zcol0 = vec![K::ZERO; d_pad];
    let mut y_zcol1 = vec![K::ZERO; d_pad];
    y_zcol0[0] = K::from(F::from_u64(23));
    y_zcol1[0] = K::from(F::from_u64(29));

    let inst0 = CeClaim::<Commitment, F, K> {
        c: Commitment::zeros(params.d as usize, 1),
        X: X0,
        r: r.clone(),
        s_col: s_col.clone(),
        y_ring: y0,
        ct: y_scalars0,
        aux_openings: Vec::new(),
        y_zcol: y_zcol0,
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
    };
    let inst1 = CeClaim::<Commitment, F, K> {
        c: Commitment::zeros(params.d as usize, 1),
        X: X1,
        r,
        s_col: s_col.clone(),
        y_ring: y1,
        ct: y_scalars1,
        aux_openings: Vec::new(),
        y_zcol: y_zcol1,
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
    };

    let rho0 = Mat::identity(D);
    let mut rho1 = Mat::identity(D);
    for i in 0..D {
        rho1.set(i, i, F::from_u64(2));
    }
    let rhos = vec![rho0, rho1];
    let rhos_typed =
        neo_reductions::api::rot_rhos_from_mats(&params, &rhos, "rlc_public_y_zcol:test rhos").expect("typed rhos");

    let expected = neo_reductions::api::rlc_public(
        &s,
        &params,
        &rhos_typed,
        &[inst0.clone(), inst1.clone()],
        |_rhos, _cs| Commitment::zeros(params.d as usize, 1),
        ell_d,
    )
    .expect("rlc_public");

    let mut expected_shell = expected.clone();
    expected_shell.s_col[0] += K::ONE;
    expected_shell.y_zcol[0] += K::ONE;

    let (ok, _) = neo_reductions::api::rlc_public_matches_with_perf(
        &s,
        &params,
        &rhos_typed,
        &[inst0, inst1],
        &expected_shell,
        |_rhos, _cs| Commitment::zeros(params.d as usize, 1),
        ell_d,
    )
    .expect("rlc_public_matches_with_perf");
    assert!(ok, "public RLC check must ignore y_zcol and s_col shell");
}
