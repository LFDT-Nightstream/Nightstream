#![allow(non_snake_case)]

use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CeClaim, Mat};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

fn column_point_len(structure: &CcsStructure<F>) -> usize {
    structure.m.next_power_of_two().max(2).trailing_zeros() as usize
}

fn two_digit_params() -> NeoParams {
    let paper = NeoParams::goldilocks_paper_b2();
    NeoParams::new(
        paper.q,
        paper.eta,
        paper.d,
        paper.kappa,
        paper.m,
        paper.b,
        2,
        1,
        paper.s,
        paper.lambda,
    )
    .expect("valid two-digit test parameters")
}

fn scale_commitment(c: &Commitment, scale: F) -> Commitment {
    let mut out = c.clone();
    for v in out.data.iter_mut() {
        *v *= scale;
    }
    out
}

fn add_commitments(a: &Commitment, b: &Commitment) -> Commitment {
    let mut out = a.clone();
    out.add_inplace(b);
    out
}

#[test]
fn verify_dec_public_rejects_stale_ct_shell_and_checks_y_aux_and_x_entries() {
    let params = two_digit_params();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize; // 64 -> 6
    let d_pad = 1usize << ell_d;
    assert!(d_pad >= D);

    // Minimal CCS structure: only t matters for this test.
    let s = CcsStructure::new(vec![Mat::identity(D)], neo_ccs::poly::SparsePoly::new(1, vec![])).unwrap();

    let m_in = 1usize;
    let r = vec![K::from(F::from_u64(3)), K::from(F::from_u64(5))];

    let t_eff = 3usize; // pretend we have extra appended openings

    // Child 0 / child 1.
    let mut y0 = vec![vec![K::ZERO; d_pad]; t_eff];
    let mut y1 = vec![vec![K::ZERO; d_pad]; t_eff];
    for j in 0..t_eff {
        y0[j][0] = K::from(F::from_u64((10 + j) as u64));
        y0[j][1] = K::from(F::from_u64((20 + j) as u64));
        y1[j][0] = K::from(F::from_u64((30 + j) as u64));
        y1[j][1] = K::from(F::from_u64((40 + j) as u64));
    }

    let y_scalars0: Vec<K> = y0.iter().map(|row| row[0]).collect();
    let y_scalars1: Vec<K> = y1.iter().map(|row| row[0]).collect();
    let aux0 = vec![K::from(F::from_u64(101)), K::from(F::from_u64(103))];
    let aux1 = vec![K::from(F::from_u64(107)), K::from(F::from_u64(109))];

    let mut X0 = Mat::zero(D, m_in, F::ZERO);
    let X1 = Mat::zero(D, m_in, F::ZERO);
    X0[(0, 0)] = F::ONE;

    let c0 = Commitment::zeros(params.d as usize, 1);
    let mut c1 = Commitment::zeros(params.d as usize, 1);
    c1.data[0] = F::from_u64(13);

    let child0 = CeClaim::<Commitment, F, K> {
        adv: None,
        c: c0.clone(),
        X: X0.clone(),
        r: r.clone(),
        s_col: Vec::new(),
        y_ring: y0.clone(),
        ct: y_scalars0.clone(),
        aux_openings: aux0.clone(),
        y_zcol: Vec::new(),
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
    };
    let child1 = CeClaim::<Commitment, F, K> {
        adv: None,
        c: c1.clone(),
        X: X1.clone(),
        r: r.clone(),
        s_col: Vec::new(),
        y_ring: y1.clone(),
        ct: y_scalars1.clone(),
        aux_openings: aux1.clone(),
        y_zcol: Vec::new(),
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
    };

    // Parent = child0 + b * child1.
    let bF = F::from_u64(params.b as u64);
    let bK = K::from(bF);
    let mut y_parent = vec![vec![K::ZERO; d_pad]; t_eff];
    let mut y_scalars_parent = vec![K::ZERO; t_eff];
    let mut aux_parent = vec![K::ZERO; aux0.len()];
    for j in 0..t_eff {
        for t in 0..d_pad {
            y_parent[j][t] = child0.y_ring[j][t] + bK * child1.y_ring[j][t];
        }
        y_scalars_parent[j] = child0.ct[j] + bK * child1.ct[j];
    }
    for i in 0..aux_parent.len() {
        aux_parent[i] = child0.aux_openings[i] + bK * child1.aux_openings[i];
    }

    let mut X_parent = Mat::zero(D, m_in, F::ZERO);
    X_parent[(0, 0)] = X0[(0, 0)] + bF * X1[(0, 0)];

    let c_parent = add_commitments(&c0, &scale_commitment(&c1, bF));

    let parent = CeClaim::<Commitment, F, K> {
        adv: None,
        c: c_parent.clone(),
        X: X_parent,
        r: r.clone(),
        s_col: Vec::new(),
        y_ring: y_parent.clone(),
        ct: y_scalars_parent.clone(),
        aux_openings: aux_parent.clone(),
        y_zcol: Vec::new(),
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
    };

    let combine_b_pows = |cs: &[Commitment], b: u32| {
        let bF = F::from_u64(b as u64);
        let mut acc = Commitment::zeros(cs[0].d, cs[0].kappa);
        let mut pow = F::ONE;
        for c in cs {
            let term = scale_commitment(c, pow);
            acc = add_commitments(&acc, &term);
            pow *= bF;
        }
        acc
    };

    assert!(neo_reductions::api::verify_dec_public(
        &s,
        &params,
        column_point_len(&s),
        &parent,
        &[child0.clone(), child1.clone()],
        combine_b_pows,
        ell_d
    ));

    // Tamper an "extra" opening (j >= s.t()) and ensure the check fails.
    let mut parent_bad = parent.clone();
    parent_bad.y_ring[t_eff - 1][0] += K::ONE;
    assert!(!neo_reductions::api::verify_dec_public(
        &s,
        &params,
        column_point_len(&s),
        &parent_bad,
        &[child0.clone(), child1.clone()],
        combine_b_pows,
        ell_d
    ));

    // Tamper only ct shell on the parent. `ct` is denormalized from
    // `y_ring`, but it is still a public CE field consumed by downstream
    // authority digests, so DEC public verification must reject stale ct.
    let mut parent_bad2 = parent.clone();
    parent_bad2.ct[t_eff - 1] += K::ONE;
    assert!(!neo_reductions::api::verify_dec_public(
        &s,
        &params,
        column_point_len(&s),
        &parent_bad2,
        &[child0.clone(), child1.clone()],
        combine_b_pows,
        ell_d
    ));

    // Same for child ct: the shape is unchanged and `y_ring` still
    // recomposes, but the child CE claim itself is stale.
    let mut child1_bad = child1.clone();
    child1_bad.ct[t_eff - 1] += K::ONE;
    assert!(!neo_reductions::api::verify_dec_public(
        &s,
        &params,
        column_point_len(&s),
        &parent,
        &[child0.clone(), child1_bad],
        combine_b_pows,
        ell_d
    ));

    // Tamper aux_openings and ensure the aux decomposition check fails.
    let mut parent_bad3 = parent.clone();
    parent_bad3.aux_openings[0] += K::ONE;
    assert!(!neo_reductions::api::verify_dec_public(
        &s,
        &params,
        column_point_len(&s),
        &parent_bad3,
        &[child0.clone(), child1.clone()],
        combine_b_pows,
        ell_d
    ));

    // Tamper X and ensure the X decomposition check fails.
    let mut parent_bad4 = parent.clone();
    parent_bad4.X[(0, 0)] += F::ONE;
    assert!(!neo_reductions::api::verify_dec_public(
        &s,
        &params,
        column_point_len(&s),
        &parent_bad4,
        &[child0.clone(), child1.clone()],
        combine_b_pows,
        ell_d
    ));
}

#[test]
fn verify_dec_public_ignores_y_zcol_but_checks_s_col_when_present() {
    let params = two_digit_params();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let d_pad = 1usize << ell_d;

    // Minimal CCS structure: only t matters for this test.
    let s = CcsStructure::new(vec![Mat::identity(D)], neo_ccs::poly::SparsePoly::new(1, vec![])).unwrap();

    let m_in = 1usize;
    let r = vec![K::from(F::from_u64(3)), K::from(F::from_u64(5))];
    let ell_m = s.m.next_power_of_two().max(2).trailing_zeros() as usize;
    let s_col = vec![K::from(F::from_u64(7)); ell_m];
    let t_eff = 1usize;

    // Child y / scalars (any consistent values).
    let mut y0 = vec![vec![K::ZERO; d_pad]; t_eff];
    let mut y1 = vec![vec![K::ZERO; d_pad]; t_eff];
    y0[0][0] = K::from(F::from_u64(11));
    y1[0][0] = K::from(F::from_u64(13));

    let y_scalars0: Vec<K> = y0.iter().map(|row| row[0]).collect();
    let y_scalars1: Vec<K> = y1.iter().map(|row| row[0]).collect();

    // Child y_zcol (any consistent values).
    let mut y_zcol0 = vec![K::ZERO; d_pad];
    let mut y_zcol1 = vec![K::ZERO; d_pad];
    y_zcol0[0] = K::from(F::from_u64(17));
    y_zcol1[0] = K::from(F::from_u64(19));

    let mut X0 = Mat::zero(D, m_in, F::ZERO);
    let X1 = Mat::zero(D, m_in, F::ZERO);
    X0[(0, 0)] = F::ONE;

    let c0 = Commitment::zeros(params.d as usize, 1);
    let mut c1 = Commitment::zeros(params.d as usize, 1);
    c1.data[0] = F::from_u64(23);

    let child0 = CeClaim::<Commitment, F, K> {
        adv: None,
        c: c0.clone(),
        X: X0.clone(),
        r: r.clone(),
        s_col: s_col.clone(),
        y_ring: y0.clone(),
        ct: y_scalars0.clone(),
        aux_openings: Vec::new(),
        y_zcol: y_zcol0.clone(),
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
    };
    let child1 = CeClaim::<Commitment, F, K> {
        adv: None,
        c: c1.clone(),
        X: X1.clone(),
        r: r.clone(),
        s_col: s_col.clone(),
        y_ring: y1.clone(),
        ct: y_scalars1.clone(),
        aux_openings: Vec::new(),
        y_zcol: y_zcol1.clone(),
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
    };

    // Parent = child0 + b * child1.
    let bF = F::from_u64(params.b as u64);
    let bK = K::from(bF);

    let mut y_parent = vec![vec![K::ZERO; d_pad]; t_eff];
    let mut y_scalars_parent = vec![K::ZERO; t_eff];
    for j in 0..t_eff {
        for t in 0..d_pad {
            y_parent[j][t] = child0.y_ring[j][t] + bK * child1.y_ring[j][t];
        }
        y_scalars_parent[j] = child0.ct[j] + bK * child1.ct[j];
    }

    let mut y_zcol_parent = vec![K::ZERO; d_pad];
    for t in 0..d_pad {
        y_zcol_parent[t] = child0.y_zcol[t] + bK * child1.y_zcol[t];
    }

    let mut X_parent = Mat::zero(D, m_in, F::ZERO);
    X_parent[(0, 0)] = X0[(0, 0)] + bF * X1[(0, 0)];

    let c_parent = add_commitments(&c0, &scale_commitment(&c1, bF));

    let parent = CeClaim::<Commitment, F, K> {
        adv: None,
        c: c_parent.clone(),
        X: X_parent,
        r: r.clone(),
        s_col: s_col.clone(),
        y_ring: y_parent.clone(),
        ct: y_scalars_parent.clone(),
        aux_openings: Vec::new(),
        y_zcol: y_zcol_parent.clone(),
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
    };

    let combine_b_pows = |cs: &[Commitment], b: u32| {
        let bF = F::from_u64(b as u64);
        let mut acc = Commitment::zeros(cs[0].d, cs[0].kappa);
        let mut pow = F::ONE;
        for c in cs {
            let term = scale_commitment(c, pow);
            acc = add_commitments(&acc, &term);
            pow *= bF;
        }
        acc
    };

    assert!(neo_reductions::api::verify_dec_public(
        &s,
        &params,
        column_point_len(&s),
        &parent,
        &[child0.clone(), child1.clone()],
        combine_b_pows,
        ell_d
    ));

    // y_zcol is transport shell on the public DEC boundary. It is not
    // b-ary recomposed through DEC; terminal CE authority re-binds it
    // against the opened witness.
    let mut parent_bad = parent.clone();
    parent_bad.y_zcol[0] += K::ONE;
    assert!(neo_reductions::api::verify_dec_public(
        &s,
        &params,
        column_point_len(&s),
        &parent_bad,
        &[child0.clone(), child1.clone()],
        combine_b_pows,
        ell_d
    ));

    // s_col is different: it is the point at which y_zcol is later
    // interpreted, so the DEC public boundary requires the shared point
    // to agree across parent and children.
    let mut child1_bad = child1.clone();
    child1_bad.s_col[0] += K::ONE;
    assert!(!neo_reductions::api::verify_dec_public(
        &s,
        &params,
        column_point_len(&s),
        &parent,
        &[child0, child1_bad],
        combine_b_pows,
        ell_d
    ));
}
