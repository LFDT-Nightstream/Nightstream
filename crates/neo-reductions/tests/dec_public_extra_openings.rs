#![allow(non_snake_case)]

use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, Mat, MeInstance};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

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

fn recompose_base_b_digits(params: &NeoParams, digits: &[K]) -> K {
    let bK = K::from(F::from_u64(params.b as u64));
    let mut pow = K::ONE;
    let mut acc = K::ZERO;
    for &v in digits.iter().take(D) {
        acc += pow * v;
        pow *= bK;
    }
    acc
}

#[test]
fn verify_dec_public_checks_all_y_and_y_scalars_entries() {
    let params = NeoParams::goldilocks_127();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize; // 64 -> 6
    let d_pad = 1usize << ell_d;
    assert!(d_pad >= D);

    // Minimal CCS structure: only t matters for this test.
    let s = CcsStructure::new(vec![Mat::identity(1)], neo_ccs::poly::SparsePoly::new(1, vec![])).unwrap();

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

    let y_scalars0: Vec<K> = y0.iter().map(|row| recompose_base_b_digits(&params, row)).collect();
    let y_scalars1: Vec<K> = y1.iter().map(|row| recompose_base_b_digits(&params, row)).collect();

    let mut X0 = Mat::zero(D, m_in, F::ZERO);
    let mut X1 = Mat::zero(D, m_in, F::ZERO);
    X0[(0, 0)] = F::from_u64(7);
    X1[(0, 0)] = F::from_u64(9);

    let c0 = Commitment::zeros(params.d as usize, 1);
    let mut c1 = Commitment::zeros(params.d as usize, 1);
    c1.data[0] = F::from_u64(13);

    let child0 = MeInstance::<Commitment, F, K> {
        c: c0.clone(),
        X: X0.clone(),
        r: r.clone(),
        y: y0.clone(),
        y_scalars: y_scalars0.clone(),
        m_in,
        fold_digest: [0u8; 32],
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
    };
    let child1 = MeInstance::<Commitment, F, K> {
        c: c1.clone(),
        X: X1.clone(),
        r: r.clone(),
        y: y1.clone(),
        y_scalars: y_scalars1.clone(),
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
            y_parent[j][t] = child0.y[j][t] + bK * child1.y[j][t];
        }
        y_scalars_parent[j] = child0.y_scalars[j] + bK * child1.y_scalars[j];
    }

    let mut X_parent = Mat::zero(D, m_in, F::ZERO);
    X_parent[(0, 0)] = X0[(0, 0)] + bF * X1[(0, 0)];

    let c_parent = add_commitments(&c0, &scale_commitment(&c1, bF));

    let parent = MeInstance::<Commitment, F, K> {
        c: c_parent.clone(),
        X: X_parent,
        r: r.clone(),
        y: y_parent.clone(),
        y_scalars: y_scalars_parent.clone(),
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
        &parent,
        &[child0.clone(), child1.clone()],
        combine_b_pows,
        ell_d
    ));

    // Tamper an "extra" opening (j >= s.t()) and ensure the check fails.
    let mut parent_bad = parent.clone();
    parent_bad.y[t_eff - 1][0] += K::ONE;
    assert!(!neo_reductions::api::verify_dec_public(
        &s,
        &params,
        &parent_bad,
        &[child0.clone(), child1.clone()],
        combine_b_pows,
        ell_d
    ));

    // Tamper the corresponding scalar and ensure the scalar check fails.
    let mut parent_bad2 = parent;
    parent_bad2.y_scalars[t_eff - 1] += K::ONE;
    assert!(!neo_reductions::api::verify_dec_public(
        &s,
        &params,
        &parent_bad2,
        &[child0, child1],
        combine_b_pows,
        ell_d
    ));
}

