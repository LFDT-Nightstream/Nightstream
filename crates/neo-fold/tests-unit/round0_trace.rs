//! Write a per-X, paper-exact Round-0 trace for Sage comparison.
//!
//! Run: cargo test -p neo-fold write_round0_trace --test unit -- --ignored --nocapture

use std::fs;
use std::path::PathBuf;

use serde::Serialize;

use neo_params::NeoParams;
use neo_math::{F, K, D, KExtensions};
use neo_ccs::{CcsStructure, Mat, MatRef, MeInstance};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::helpers::{DummyS, create_test_ccs, mk_mcs};

// ---------- helpers: encoding ----------
fn f_to_str(x: F) -> String { x.as_canonical_u64().to_string() }
fn k_to_pair(k: K) -> [String; 2] { let c = k.as_coeffs(); [f_to_str(c[0]), f_to_str(c[1])] }
fn mat_f_to_rows(m: &Mat<F>) -> Vec<Vec<String>> {
    let (rows, cols) = (m.rows(), m.cols());
    (0..rows).map(|r| (0..cols).map(|c| f_to_str(m[(r,c)])).collect()).collect()
}

// ---------- JSON types ----------
#[derive(Serialize)]
struct QuadRule { u2: [String;2] }

#[derive(Serialize)]
struct SparseTerm { coeff: String, exps: Vec<usize> }

#[derive(Serialize)]
struct SparsePolyDump { arity: usize, terms: Vec<SparseTerm> }

#[derive(Serialize)]
struct NcItem { 
    i: usize, 
    #[serde(rename = "Ni")] ni: [String;2], 
    #[serde(rename = "wNi")] w_ni: [String;2] 
}

#[derive(Serialize)]
struct EvalItem { 
    i: usize, 
    j: usize, 
    w: [String;2], 
    #[serde(rename = "Eij")] eij: [String;2], 
    #[serde(rename = "wEij")] w_eij: [String;2] 
}

#[derive(Serialize)]
struct RowTrace {
    mask: u64,
    #[serde(rename = "Xa_bits")] xa_bits: Vec<u8>,
    #[serde(rename = "Xr_bits")] xr_bits: Vec<u8>,
    eq_beta: [String;2],
    #[serde(rename = "F_val")] f_val: [String;2],
    #[serde(rename = "F_contrib")] f_contrib: [String;2],
    nc_items: Vec<NcItem>,
    nc_contrib: [String;2],
    eval_terms: Vec<EvalItem>,
    eval_contrib: [String;2],
    #[serde(rename = "Qx")] qx: [String;2],
}

#[derive(Serialize)]
struct Round0Trace {
    // params
    p: String, b: u32, d: usize, n: usize, m: usize, t: usize, k_total: usize,
    #[serde(rename="M")] mats: Vec<Vec<Vec<String>>>,
    #[serde(rename="Z")] z_witnesses: Vec<Vec<Vec<String>>>,

    // field rule + challenges + f
    quad_rule: QuadRule,
    alpha: Vec<[String;2]>,
    beta_a: Vec<[String;2]>,
    beta_r: Vec<[String;2]>,
    gamma: [String;2],
    r: Vec<[String;2]>,
    f: Option<SparsePolyDump>,

    // detailed per-X trace + totals
    rows: Vec<RowTrace>,
    #[serde(rename = "S_total")] s_total: [String;2],
    p0: [String;2],
    p1: [String;2],
    #[serde(rename = "F_beta_r")] f_beta_r: [String;2],
    #[serde(rename = "NC_hypercube_sum")] nc_hypercube_sum: [String;2],
    #[serde(rename = "Eval_total")] eval_total: [String;2],
}

// ---------- math helpers (paper definitions, no shortcuts) ----------
fn eq_gate(x_bits_k: &[K], y: &[K]) -> K {
    debug_assert_eq!(x_bits_k.len(), y.len());
    let mut acc = K::ONE;
    for i in 0..x_bits_k.len() {
        acc *= (K::ONE - x_bits_k[i]) * (K::ONE - y[i]) + x_bits_k[i] * y[i];
    }
    acc
}
fn chi_at(idx: usize, ell: usize, x_bits_k: &[K]) -> K {
    let mut acc = K::ONE;
    for j in 0..ell {
        let bit = ((idx >> j) & 1) as u64;
        acc *= if bit == 1 { x_bits_k[j] } else { K::ONE - x_bits_k[j] };
    }
    acc
}
fn mle_vec_at(w_row_f: &[F], x_bits_k: &[K], ell_n: usize) -> K {
    let mut s = K::ZERO;
    for row in 0..w_row_f.len() {
        s += K::from(w_row_f[row]) * chi_at(row, ell_n, x_bits_k);
    }
    s
}
fn mle_mat_at(y: &Mat<F>, xa_bits_k: &[K], xr_bits_k: &[K], ell_d: usize, ell_n: usize) -> K {
    let (d_rows, n_cols) = (y.rows(), y.cols());
    debug_assert!(d_rows <= (1usize<<ell_d));
    debug_assert!(n_cols <= (1usize<<ell_n));
    let mut s = K::ZERO;
    for rho in 0..d_rows {
        let chi_a = chi_at(rho, ell_d, xa_bits_k);
        for row in 0..n_cols {
            s += K::from(y[(rho,row)]) * chi_a * chi_at(row, ell_n, xr_bits_k);
        }
    }
    s
}
fn range_prod(val: K, b: u32) -> K {
    let mut acc = K::ONE;
    for t in -(b as i64 - 1)..=(b as i64 - 1) { acc *= val - K::from(F::from_i64(t)); }
    acc
}

#[test]
#[ignore] // run manually: cargo test -p neo-fold --test unit write_round0_trace -- --ignored --nocapture
fn write_round0_trace() {
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let s: CcsStructure<F> = create_test_ccs().ensure_identity_first().unwrap();
    let l = DummyS;
    let m_in = 1;

    let (mcs1, wit1) =
        mk_mcs(&params, vec![F::ONE, F::from_u64(2), F::from_u64(3), F::ZERO, F::from_u64(5)], m_in, &l);
    let (_mcs2, wit2) =
        mk_mcs(&params, vec![F::ONE, F::from_u64(7), F::from_u64(11), F::ZERO, F::from_u64(18)], m_in, &l);

    // Run k=1 simple fold to produce an ME input
    let mut tr_k1 = Poseidon2Transcript::new(b"fixture/k1");
    let (me_outs, _p1) =
        neo_fold::pi_ccs_prove_simple(&mut tr_k1, &params, &s, &[_mcs2.clone()], &[wit2.clone()], &l).unwrap();
    let me_input: MeInstance<_,_,K> = me_outs[0].clone();
    let me_witness: Mat<F> = wit2.Z.clone();

    // Sample challenges like the prover
    let dims = neo_fold::pi_ccs::context::build_dims_and_policy(&params, &s).unwrap();
    let ell_d = dims.ell_d;
    let ell_n = dims.ell_n;
    let ell   = ell_d + ell_n;

    let mut tr = neo_transcript::Poseidon2Transcript::new(b"prove/k2");
    neo_fold::pi_ccs::transcript::bind_header_and_instances(&mut tr, &params, &s, &[mcs1.clone()], ell, dims.d_sc, 0).unwrap();
    neo_fold::pi_ccs::transcript::bind_me_inputs(&mut tr, &[me_input.clone()]).unwrap();
    let ch = neo_fold::pi_ccs::transcript::sample_challenges(&mut tr, ell_d, ell).unwrap();

    // Recompose z1 from Z1
    let recompose = |z_mat: &Mat<F>| -> Vec<F> {
        let mut pow = vec![F::ONE; D];
        for i in 1..D { pow[i] = pow[i-1] * F::from_u64(params.b as u64); }
        let mut z = vec![F::ZERO; s.m];
        for c in 0..s.m {
            let mut acc = F::ZERO;
            for rho in 0..D { acc += pow[rho] * z_mat[(rho,c)]; }
            z[c] = acc;
        }
        z
    };
    let z1 = recompose(&wit1.Z);

    // helpers for evaluation
    let mats = &s.matrices;
    let mut beta: Vec<K> = Vec::with_capacity(ch.beta_a.len() + ch.beta_r.len());
    beta.extend_from_slice(&ch.beta_a);
    beta.extend_from_slice(&ch.beta_r);
    let gamma = ch.gamma;

    // sparse f dump (optional)
    let f_dump = {
        let f = &s.f;
        let mut terms: Vec<SparseTerm> = Vec::new();
        for term in f.terms() {
            let exps: Vec<usize> = term.exps.iter().map(|&e| e as usize).collect();
            terms.push(SparseTerm { coeff: f_to_str(term.coeff), exps });
        }
        Some(SparsePolyDump { arity: f.arity(), terms })
    };

    // quadratic rule (u^2 = a + b u)
    let u = neo_math::from_complex(F::ZERO, F::ONE);
    let uu = u * u;
    let quad_rule = QuadRule { u2: k_to_pair(uu) };

    // dump static header
    let mut m_rows = Vec::with_capacity(s.t());
    for j in 0..s.t() { m_rows.push(mat_f_to_rows(&mats[j])); }
    let z_rows = vec![mat_f_to_rows(&wit1.Z), mat_f_to_rows(&me_witness)];

    // gamma^k_total with k_total=2 (1 MCS + 1 ME)
    let mut gamma_k = K::ONE;
    for _ in 0..2 { gamma_k *= gamma; }

    // iterate hypercube
    let mut rows: Vec<RowTrace> = Vec::new();
    let mut s_total = K::ZERO;
    let mut p0 = K::ZERO;
    let mut p1 = K::ZERO;

    for mask in 0..(1usize << ell) {
        // split bits LSB-first
        let mut xa_bits_k = vec![K::ZERO; ell_d];
        let mut xr_bits_k = vec![K::ZERO; ell_n];
        for i in 0..ell_d { xa_bits_k[i] = if ((mask >> i) & 1) == 1 { K::ONE } else { K::ZERO }; }
        for i in 0..ell_n { xr_bits_k[i] = if ((mask >> (ell_d+i)) & 1) == 1 { K::ONE } else { K::ZERO }; }
        let mut xa_bits_u8 = vec![0u8; ell_d]; for i in 0..ell_d { xa_bits_u8[i] = if xa_bits_k[i]==K::ONE {1} else {0}; }
        let mut xr_bits_u8 = vec![0u8; ell_n]; for i in 0..ell_n { xr_bits_u8[i] = if xr_bits_k[i]==K::ONE {1} else {0}; }

        let mut x_concat = xa_bits_k.clone(); x_concat.extend_from_slice(&xr_bits_k);
        let eq_beta = eq_gate(&x_concat, &beta);

        // F(X_r)
        let mut m_vals: Vec<K> = Vec::with_capacity(s.t());
        for j in 0..s.t() {
            // wj = (M_j z1)[row]
            let mr = MatRef::from_mat(&mats[j]);
            let mut wj = vec![F::ZERO; s.n];
            for row in 0..s.n {
                let mut acc = F::ZERO;
                let rref = mr.row(row);
                for c in 0..s.m { acc += rref[c] * z1[c]; }
                wj[row] = acc;
            }
            m_vals.push(mle_vec_at(&wj, &xr_bits_k, ell_n));
        }
        let f_val = s.f.eval_in_ext::<K>(&m_vals);
        let f_contrib = eq_beta * f_val;

        // NC(X): use M1^T
        let mut nc_items: Vec<NcItem> = Vec::new();
        let mut gpow = gamma; // starts at γ^1
        let mut nc_sum = K::ZERO;
        for i_inst in 0..2 { // k_total = 2
            // Y = Z_i M1^T (d×n)
            let zi = if i_inst==0 { &wit1.Z } else { &me_witness };
            let (d_rows, n_cols) = (zi.rows(), s.n);
            let mut y_mat = Mat::zero(d_rows, n_cols, F::ZERO);
            for rho in 0..d_rows {
                for row in 0..n_cols {
                    let mut acc = F::ZERO;
                    for c in 0..s.m { acc += zi[(rho,c)] * mats[0][(row,c)]; }
                    y_mat[(rho,row)] = acc;
                }
            }
            let y_mle = mle_mat_at(&y_mat, &xa_bits_k, &xr_bits_k, ell_d, ell_n);
            let ni_val = range_prod(y_mle, params.b);
            let w_ni = eq_beta * gpow * ni_val;
            nc_sum += w_ni;
            nc_items.push(NcItem { i: i_inst+1, ni: k_to_pair(ni_val), w_ni: k_to_pair(w_ni) });
            gpow *= gamma;
        }

        // Eval(X)
        let mut eval_terms: Vec<EvalItem> = Vec::new();
        let mut eval_inner = K::ZERO;
        for j in 1..=s.t() {
            for i_abs in 2..=2 { // only ME instance (i=2)
                let zi = &me_witness;
                // Y = Z_i M_j^T
                let (d_rows, n_cols) = (zi.rows(), s.n);
                let mut y_mat = Mat::zero(d_rows, n_cols, F::ZERO);
                for rho in 0..d_rows {
                    for row in 0..n_cols {
                        let mut acc = F::ZERO;
                        for c in 0..s.m { acc += zi[(rho,c)] * mats[j-1][(row,c)]; }
                        y_mat[(rho,row)] = acc;
                    }
                }
                let y_mle = mle_mat_at(&y_mat, &xa_bits_k, &xr_bits_k, ell_d, ell_n);
                // eq((α,r), X)
                let mut ar: Vec<K> = Vec::with_capacity(ch.alpha.len() + me_input.r.len());
                ar.extend_from_slice(&ch.alpha);
                ar.extend_from_slice(&me_input.r);
                let eq_ar = eq_gate(&x_concat, &ar);
                let eij = eq_ar * y_mle;
                // weight γ^{i+(j-1)k-1}, k=2 ⇒ exponent = i + 2*(j-1) - 1
                let mut w = K::ONE;
                let exponent = i_abs + (j-1)*2 - 1;
                for _ in 0..exponent { w *= gamma; }
                let w_eij = w * eij;
                eval_inner += w_eij;
                eval_terms.push(EvalItem { i: i_abs, j, w: k_to_pair(w), eij: k_to_pair(eij), w_eij: k_to_pair(w_eij) });
            }
        }
        let eval_contrib = gamma_k * eval_inner;

        let qx = f_contrib + nc_sum + eval_contrib;
        s_total += qx;

        // p(0)/p(1) split by first row bit (ell_d)
        let first_row_bit = ell_d;
        let bit = ((mask >> first_row_bit) & 1) as u8;
        if bit == 0 { p0 += qx; } else { p1 += qx; }

        rows.push(RowTrace {
            mask: mask as u64,
            xa_bits: xa_bits_u8,
            xr_bits: xr_bits_u8,
            eq_beta: k_to_pair(eq_beta),
            f_val: k_to_pair(f_val),
            f_contrib: k_to_pair(f_contrib),
            nc_items,
            nc_contrib: k_to_pair(nc_sum),
            eval_terms,
            eval_contrib: k_to_pair(eval_contrib),
            qx: k_to_pair(qx),
        });
    }

    // F(β_r)
    let mut m_vals_beta: Vec<K> = Vec::with_capacity(s.t());
    // χ_row(β_r) via eq_gate(row_bits, β_r)
    for j in 0..s.t() {
        let mr = MatRef::from_mat(&mats[j]);
        let mut wj = vec![F::ZERO; s.n];
        for row in 0..s.n {
            let mut acc = F::ZERO;
            let rref = mr.row(row);
            for c in 0..s.m { acc += rref[c] * z1[c]; }
            wj[row] = acc;
        }
        // Σ_row wj[row]·χ_row(β_r)
        let mut s_j = K::ZERO;
        for row in 0..s.n {
            let mut row_bits_k = vec![K::ZERO; ell_n];
            for i in 0..ell_n { row_bits_k[i] = if ((row >> i) & 1) == 1 { K::ONE } else { K::ZERO }; }
            s_j += K::from(wj[row]) * eq_gate(&row_bits_k, &ch.beta_r);
        }
        m_vals_beta.push(s_j);
    }
    let f_beta_r = s.f.eval_in_ext::<K>(&m_vals_beta);

    // NC hypercube sum (def)
    let mut nc_sum_def = K::ZERO;
    for r in &rows { let [re, im] = &r.nc_contrib; nc_sum_def += neo_math::from_complex(F::from_u64(re.parse().unwrap()), F::from_u64(im.parse().unwrap())); }

    // Eval total (def)
    let mut eval_total = K::ZERO;
    for r in &rows { let [re, im] = &r.eval_contrib; eval_total += neo_math::from_complex(F::from_u64(re.parse().unwrap()), F::from_u64(im.parse().unwrap())); }

    // final JSON
    let dump = Round0Trace {
        p: "18446744069414584321".to_string(),
        b: params.b, d: D, n: s.n, m: s.m, t: s.t(), k_total: 2,
        mats: m_rows, z_witnesses: z_rows,
        quad_rule,
        alpha: ch.alpha.iter().map(|&k| k_to_pair(k)).collect(),
        beta_a: ch.beta_a.iter().map(|&k| k_to_pair(k)).collect(),
        beta_r: ch.beta_r.iter().map(|&k| k_to_pair(k)).collect(),
        gamma: k_to_pair(ch.gamma),
        r: me_input.r.iter().map(|&k| k_to_pair(k)).collect(),
        f: f_dump,
        rows,
        s_total: k_to_pair(s_total),
        p0: k_to_pair(p0), p1: k_to_pair(p1),
        f_beta_r: k_to_pair(f_beta_r),
        nc_hypercube_sum: k_to_pair(nc_sum_def),
        eval_total: k_to_pair(eval_total),
    };

    let out = std::env::var("NEO_ROUND0_TRACE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("../../new-neo-sage/round0_trace.json"));
    if let Some(parent) = out.parent() { let _ = fs::create_dir_all(parent); }
    fs::write(&out, serde_json::to_string_pretty(&dump).unwrap()).unwrap();
    eprintln!("[round0_trace] wrote {}", out.display());
    assert!(out.exists());
}
