#![cfg(feature = "paper-exact")]
#![allow(non_snake_case)]

use neo_ccs::{CcsStructure, CcsWitness, Mat, SparsePoly, Term};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::optimized_engine::oracle::NcOracle;
use neo_reductions::paper_exact_engine as refimpl;
use neo_reductions::paper_exact_engine::oracle::PaperExactOracle;
use neo_reductions::sumcheck::RoundOracle;
use p3_field::PrimeCharacteristicRing;

fn sum_q_fe_over_hypercube(
    s: &CcsStructure<F>,
    params: &NeoParams,
    mcs_witnesses: &[CcsWitness<F>],
    me_witnesses: &[Mat<F>],
    ch: &neo_reductions::Challenges,
    ell_d: usize,
    ell_n: usize,
    r_inputs: Option<&[K]>,
) -> K {
    let d_sz = 1usize << ell_d;
    let n_sz = 1usize << ell_n;
    let mut total = K::ZERO;
    for xa in 0..d_sz {
        let alpha_bool: Vec<K> = (0..ell_d)
            .map(|bit| if ((xa >> bit) & 1) == 1 { K::ONE } else { K::ZERO })
            .collect();
        for xr in 0..n_sz {
            let r_bool: Vec<K> = (0..ell_n)
                .map(|bit| if ((xr >> bit) & 1) == 1 { K::ONE } else { K::ZERO })
                .collect();
            let (q_at_bool, _) = refimpl::q_eval_at_ext_point_fe_paper_exact_with_inputs(
                s,
                params,
                mcs_witnesses,
                me_witnesses,
                &alpha_bool,
                &r_bool,
                ch,
                r_inputs,
            );
            total += q_at_bool;
        }
    }
    total
}

#[inline]
fn ceil_log2_usize(x: usize) -> usize {
    if x <= 1 {
        0
    } else {
        (usize::BITS as usize) - ((x - 1).leading_zeros() as usize)
    }
}

#[inline]
fn bool_mle_weight(mask: usize, point: &[K]) -> K {
    let mut w = K::ONE;
    for (bit, &x) in point.iter().enumerate() {
        w *= if ((mask >> bit) & 1) == 1 { x } else { K::ONE - x };
    }
    w
}

#[inline]
fn range_product_symmetric(val: K, b: u32) -> K {
    let lo = -((b as i64) - 1);
    let hi = (b as i64) - 1;
    let mut prod = K::ONE;
    for t in lo..=hi {
        prod *= val - K::from(F::from_i64(t));
    }
    prod
}

fn sum_q_nc_over_hypercube(
    params: &NeoParams,
    expected_m: usize,
    mcs_witnesses: &[CcsWitness<F>],
    me_witnesses: &[Mat<F>],
    ch: &neo_reductions::Challenges,
    ell_d: usize,
    ell_m: usize,
) -> K {
    assert_eq!(ch.beta_a.len(), ell_d, "beta_a length mismatch");
    assert_eq!(ch.beta_m.len(), ell_m, "beta_m length mismatch");

    let all_witnesses: Vec<&Mat<F>> = mcs_witnesses
        .iter()
        .map(|w| &w.Z)
        .chain(me_witnesses.iter())
        .collect();
    let digits_tables: Vec<Vec<[K; D]>> = all_witnesses
        .iter()
        .map(|z| {
            neo_reductions::common::build_witness_nc_digit_table(params, z, expected_m)
                .expect("NC test fixture must decompose witness digits")
        })
        .collect();

    let m_sz = 1usize << ell_m;
    let d_sz = 1usize << ell_d;
    let mut total = K::ZERO;
    for sm in 0..m_sz {
        let w_s = bool_mle_weight(sm, &ch.beta_m);
        for am in 0..d_sz {
            let w_a = bool_mle_weight(am, &ch.beta_a);
            let mut g = ch.gamma;
            let mut nc = K::ZERO;
            for digits in digits_tables.iter() {
                let y = if sm < expected_m && am < D {
                    digits[sm][am]
                } else {
                    K::ZERO
                };
                nc += g * range_product_symmetric(y, params.b);
                g *= ch.gamma;
            }
            total += w_s * w_a * nc;
        }
    }

    total
}

fn nc_round0_sum_from_optimized_oracle(
    s: &CcsStructure<F>,
    params: &NeoParams,
    mcs_witnesses: &[CcsWitness<F>],
    me_witnesses: &[Mat<F>],
    ch: &neo_reductions::Challenges,
    ell_d: usize,
    ell_m: usize,
    d_sc: usize,
) -> K {
    let mut oracle = NcOracle::new(s, params, mcs_witnesses, me_witnesses, ch.clone(), ell_d, ell_m, d_sc);
    let g0 = oracle.evals_at(&[K::ZERO, K::ONE]);
    g0[0] + g0[1]
}

fn tiny_ccs_id(n: usize, m: usize) -> CcsStructure<F> {
    assert_eq!(n, m, "use square tiny ccs");
    let m0 = Mat::identity(n);
    // f(y0) = y0
    let f = SparsePoly::new(
        1,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1],
        }],
    );
    CcsStructure::new(vec![m0], f).unwrap()
}

fn tiny_ccs_perm(n: usize, m: usize) -> CcsStructure<F> {
    assert_eq!(n, m, "use square tiny ccs");
    // Simple non-identity M_1: swap rows/cols for n=2; generalizes to reverse permutation
    let mut data = vec![F::ZERO; n * n];
    for r in 0..n {
        data[r * n + (n - 1 - r)] = F::ONE;
    }
    let m0 = Mat::from_row_major(n, n, data);
    // f(y0) = y0
    let f = SparsePoly::new(
        1,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1],
        }],
    );
    CcsStructure::new(vec![m0], f).unwrap()
}

fn tiny_ccs_custom_m1(m1: Mat<F>) -> CcsStructure<F> {
    assert_eq!(m1.rows(), m1.cols(), "M1 must be square");
    let _n = m1.rows();
    // f(y0) = y0
    let f = SparsePoly::new(
        1,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1],
        }],
    );
    CcsStructure::new(vec![m1], f).unwrap()
}

fn make_digits_matrix(val: F, d: usize, m: usize) -> Mat<F> {
    Mat::from_row_major(d, m, vec![val; d * m])
}

#[test]
fn round0_sum_matches_hypercube_sum_k1() {
    // Small instance: n=2 (ell_n=1), m=2, t=1
    let params = NeoParams::goldilocks_paper_b2();
    let (n, m) = (2usize, 2usize);
    let s = tiny_ccs_id(n, m);

    // One MCS witness with all-ones digits
    let z = make_digits_matrix(F::ONE, D, m);
    let mcs_w = [CcsWitness { w: vec![], Z: z }];
    let me_w: [Mat<F>; 0] = [];

    // Challenges sized to the round dimensions
    let ell_n = 1usize; // since n=2
    let ell_d = ceil_log2_usize(D);
    let ch = neo_reductions::Challenges {
        alpha: vec![K::from(F::from_u64(3)); ell_d],
        beta_a: vec![K::from(F::from_u64(5)); ell_d],
        beta_r: vec![K::from(F::from_u64(7)); ell_n],
        beta_m: Vec::new(),
        gamma: K::from(F::from_u64(11)),
    };

    // Degree bound not used by this check; set a small placeholder
    let d_sc = 4usize;

    // Build oracle (no ME inputs → Eval block gated off)
    let mut oracle = PaperExactOracle::<'_, F>::new(&s, &params, &mcs_w, &me_w, ch.clone(), ell_d, ell_n, d_sc, None);

    // Left: g0(0) + g0(1)
    let g0 = oracle.evals_at(&[K::ZERO, K::ONE]);
    let lhs = g0[0] + g0[1];

    // Right: literal ∑_{X∈{0,1}^{ell_d+ell_n}} Q_fe(X)
    let rhs = sum_q_fe_over_hypercube(&s, &params, &mcs_w, &me_w, &ch, ell_d, ell_n, None);

    assert_eq!(lhs, rhs, "round-0 sum must equal hypercube sum (k=1)");
}

#[test]
fn round0_sum_matches_hypercube_sum_k2_with_eval() {
    // Small instance with one MCS and one ME witness to enable Eval block
    let params = NeoParams::goldilocks_paper_b2();
    let (n, m) = (2usize, 2usize);
    let s = tiny_ccs_id(n, m);

    let z0 = make_digits_matrix(F::from_u64(2), D, m);
    let z1 = make_digits_matrix(F::from_u64(3), D, m);
    let mcs_w = [CcsWitness { w: vec![], Z: z0 }];
    let me_w = [z1];

    let ell_n = 1usize; // n=2
    let ell_d = ceil_log2_usize(D);
    let ch = neo_reductions::Challenges {
        alpha: vec![K::from(F::from_u64(13)); ell_d],
        beta_a: vec![K::from(F::from_u64(17)); ell_d],
        beta_r: vec![K::from(F::from_u64(19)); ell_n],
        beta_m: Vec::new(),
        gamma: K::from(F::from_u64(23)),
    };

    let r_inputs = vec![K::from(F::from_u64(29)); ell_n];
    let d_sc = 4usize;

    let mut oracle = PaperExactOracle::<'_, F>::new(
        &s,
        &params,
        &mcs_w,
        &me_w,
        ch.clone(),
        ell_d,
        ell_n,
        d_sc,
        Some(&r_inputs),
    );

    // Left: round-0 sum
    let g0 = oracle.evals_at(&[K::ZERO, K::ONE]);
    let lhs = g0[0] + g0[1];

    // Right: brute-force hypercube sum with Eval block active (r_inputs provided)
    let rhs = sum_q_fe_over_hypercube(&s, &params, &mcs_w, &me_w, &ch, ell_d, ell_n, Some(&r_inputs));

    assert_eq!(lhs, rhs, "round-0 sum must equal hypercube sum (k=2 with Eval)");
}

#[test]
fn nc_sum_engine_matches_paper_nc_when_m1_not_identity() {
    // Construct a minimal CCS where M_1 ≠ I to expose NC drift
    let params = NeoParams::goldilocks_paper_b2();
    let (n, m) = (2usize, 2usize);
    let s = tiny_ccs_perm(n, m);

    // One MCS witness with constant digits (ensures y differs from a single table lookup)
    let Z = make_digits_matrix(F::ONE, D, m);
    let mcs_w = [CcsWitness { w: vec![], Z }];
    let me_w: [Mat<F>; 0] = [];

    // Challenges for NC-only parity: use the full Ajtai bit-width of D.
    let ell_m = 1usize;
    let ell_d = ceil_log2_usize(D);
    assert!(1usize << ell_d >= D, "Ajtai domain must cover D rows");
    let ch = neo_reductions::Challenges {
        alpha: Vec::new(),
        beta_a: (0..ell_d)
            .map(|i| K::from(F::from_u64(5 + i as u64)))
            .collect(),
        beta_r: Vec::new(),
        beta_m: vec![K::from(F::from_u64(7)); ell_m],
        gamma: K::from(F::from_u64(11)),
    };

    let paper_nc = sum_q_nc_over_hypercube(&params, s.m, &mcs_w, &me_w, &ch, ell_d, ell_m);
    let engine_nc = nc_round0_sum_from_optimized_oracle(&s, &params, &mcs_w, &me_w, &ch, ell_d, ell_m, 4);
    assert_eq!(
        engine_nc, paper_nc,
        "NC round-0 sum mismatch (optimized oracle vs paper loop)"
    );
}

#[test]
fn nc_sum_engine_vs_paper_drift_with_custom_m1_and_Z() {
    // Stress with custom witness values and multiple witness channels (MCS + ME).
    let params = NeoParams::goldilocks_paper_b2();
    let (n, m) = (2usize, 2usize);

    // Custom M1: first row selects second column, second row sums both columns
    // M1 = [ [0,1], [1,1] ]
    let m1 = Mat::from_row_major(n, m, vec![F::ZERO, F::ONE, F::ONE, F::ONE]);
    let s = tiny_ccs_custom_m1(m1);

    // Z with distinct larger values to avoid accidental zeros in range product
    // Z rows (Ajtai): rho=0: [100, 17], rho=1: [9, 23], remaining rows: zeros
    let mut z_data = vec![F::ZERO; D * m];
    z_data[0] = F::from_u64(100);
    z_data[1] = F::from_u64(17);
    z_data[m] = F::from_u64(9);
    z_data[m + 1] = F::from_u64(23);
    let Z = Mat::from_row_major(D, m, z_data);
    let mcs_w = [CcsWitness { w: vec![], Z }];
    let me_w = [make_digits_matrix(F::from_u64(6), D, m)];

    let ell_m = 1usize; // m=2
    let ell_d = ceil_log2_usize(D);
    assert!(1usize << ell_d >= D, "Ajtai domain must cover D rows");
    let ch = neo_reductions::Challenges {
        alpha: Vec::new(),
        beta_a: (0..ell_d)
            .map(|i| K::from(F::from_u64(17 + i as u64)))
            .collect(),
        beta_r: Vec::new(),
        beta_m: vec![K::from(F::from_u64(19)); ell_m],
        gamma: K::from(F::from_u64(23)),
    };

    let paper_nc = sum_q_nc_over_hypercube(&params, s.m, &mcs_w, &me_w, &ch, ell_d, ell_m);
    let engine_nc = nc_round0_sum_from_optimized_oracle(&s, &params, &mcs_w, &me_w, &ch, ell_d, ell_m, 4);
    assert_eq!(engine_nc, paper_nc, "NC round-0 sum mismatch on custom fixture");
}
