//! Write engine-backed per-(i,j) Eval breakdown (JSON) for Sage comparison.
//! Each term contains the engine weight w_ij = γ^{(i-1) + j·k} and the scalar
//! y_ij(α) = <Z_i M_j^T χ_α, χ_r>, along with wE = w_ij * y_ij(α).
//!
//! Run: cargo test -p neo-fold --test unit --features testing -- write_round0_engine_eval_breakdown -- --ignored --nocapture

use std::fs;
use std::path::PathBuf;

use serde::Serialize;

use neo_params::NeoParams;
use neo_math::{F, K, KExtensions, D};
use neo_ccs::{CcsStructure, Mat, MatRef};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use neo_fold::pi_ccs::{self, context};

use crate::helpers::{DummyS, create_test_ccs, mk_mcs};

fn f_to_str(x: F) -> String { x.as_canonical_u64().to_string() }
fn k_to_pair(k: K) -> [String; 2] { let c = k.as_coeffs(); [f_to_str(c[0]), f_to_str(c[1])] }

#[derive(Serialize)]
struct EvalTermOut { i: usize, j: usize, w: [String;2], y_scalar: [String;2], #[serde(rename = "wE")] w_e: [String;2] }

#[derive(Serialize)]
struct EvalBreakdownOut { trace_kind: String, terms: Vec<EvalTermOut>, #[serde(rename="Eval_total")] eval_total: [String;2] }

#[test]
#[ignore]
fn write_round0_engine_eval_breakdown() {
    // Setup CCS and instances
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let s: CcsStructure<F> = create_test_ccs().ensure_identity_first().unwrap();
    let l = DummyS;
    let m_in = 1;

    let (mcs1, _wit1) =
        mk_mcs(&params, vec![F::ONE, F::from_u64(2), F::from_u64(3), F::ZERO, F::from_u64(5)], m_in, &l);
    let (_mcs2, wit2) =
        mk_mcs(&params, vec![F::ONE, F::from_u64(7), F::from_u64(11), F::ZERO, F::from_u64(18)], m_in, &l);

    let mut tr_k1 = Poseidon2Transcript::new(b"fixture/k1");
    let (me_outs, _p1) =
        neo_fold::pi_ccs_prove_simple(&mut tr_k1, &params, &s, &[_mcs2.clone()], &[wit2.clone()], &l).unwrap();
    let me_input = me_outs[0].clone();
    let me_witness: Mat<F> = wit2.Z.clone();

    let dims = context::build_dims_and_policy(&params, &s).unwrap();
    let ell_d = dims.ell_d; let ell_n = dims.ell_n; let ell = ell_d + ell_n;
    let k_total = 2usize;

    let mut tr = Poseidon2Transcript::new(b"prove/k2");
    pi_ccs::transcript::bind_header_and_instances(&mut tr, &params, &s, &[mcs1.clone()], ell, dims.d_sc, 0).unwrap();
    pi_ccs::transcript::bind_me_inputs(&mut tr, &[me_input.clone()]).unwrap();
    let ch = pi_ccs::transcript::sample_challenges(&mut tr, ell_d, ell).unwrap();

    // χ_α and χ_r
    let chi_alpha: Vec<K> = neo_ccs::utils::tensor_point::<K>(&ch.alpha);
    let chi_r: Vec<K> = neo_ccs::utils::tensor_point::<K>(&me_input.r);

    // γ powers
    let mut gamma_to_k = K::ONE; for _ in 0..k_total { gamma_to_k *= ch.gamma; }

    // Iterate over ME instances (i_abs from 2..=k_total)
    let mut terms: Vec<EvalTermOut> = Vec::new();
    let me_list: Vec<Mat<F>> = vec![me_witness.clone()];
    for (i_off, zi) in me_list.iter().enumerate() {
        let i_abs = 2 + i_off;
        // u_i[c] = Σ_rho Zi[rho,c] · χ_α[rho]
        let mut u_i = vec![K::ZERO; s.m];
        for c in 0..s.m {
            let mut acc = K::ZERO;
            for rho in 0..D {
                let w = if rho < chi_alpha.len() { chi_alpha[rho] } else { K::ZERO };
                acc += K::from(zi[(rho, c)]) * w;
            }
            u_i[c] = acc;
        }
        // For each j compute g_ij = M_j^T · u_i, scalar y_ij(α) = <g_ij, χ_r>
        for j in 0..s.t() {
            let mj_ref = MatRef::from_mat(&s.matrices[j]);
            // g_ij[row]
            let mut g_ij = vec![K::ZERO; s.n];
            for row in 0..s.n {
                let mut acc = K::ZERO;
                let row_slice = mj_ref.row(row);
                for c in 0..s.m { acc += K::from(row_slice[c]) * u_i[c]; }
                g_ij[row] = acc;
            }
            // y_ij(α) scalar
            let mut y_scalar = K::ZERO;
            for row in 0..s.n { y_scalar += g_ij[row] * chi_r[row]; }
            // weight w = γ^{(i_abs-1) + j·k_total}
            let mut w = K::ONE;
            for _ in 0..(i_abs - 1) { w *= ch.gamma; }
            for _ in 0..j { w *= gamma_to_k; }
            let w_e = w * y_scalar;
            terms.push(EvalTermOut { i: i_abs, j: j+1, w: k_to_pair(w), y_scalar: k_to_pair(y_scalar), w_e: k_to_pair(w_e) });
        }
    }
    let eval_total: K = terms.iter().fold(K::ZERO, |acc, t| acc + {
        let w = K::from_coeffs([F::from_u64(t.w[0].parse::<u64>().unwrap()), F::from_u64(t.w[1].parse::<u64>().unwrap())]);
        let y = K::from_coeffs([F::from_u64(t.y_scalar[0].parse::<u64>().unwrap()), F::from_u64(t.y_scalar[1].parse::<u64>().unwrap())]);
        w * y
    });

    let out = EvalBreakdownOut { trace_kind: "engine_eval_breakdown".to_string(), terms, eval_total: k_to_pair(eval_total) };
    let out_path = std::env::var("NEO_ENGINE_EVAL_BREAKDOWN").map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target/engine_eval_breakdown.json"));
    if let Some(parent) = out_path.parent() { let _ = fs::create_dir_all(parent); }
    fs::write(&out_path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    eprintln!("[round0_engine_eval_breakdown] wrote {}", out_path.display());
    assert!(out_path.exists());
}
