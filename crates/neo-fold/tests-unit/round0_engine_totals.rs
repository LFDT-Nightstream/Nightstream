//! Write engine-backed Round-0 totals (JSON) for Sage comparison.
//! Uses the optimized oracle path to compute p(0), p(1), and totals.
//!
//! Run: cargo test -p neo-fold --test unit --features testing -- write_round0_engine_totals -- --ignored --nocapture

use std::fs;
use std::path::PathBuf;

use serde::Serialize;

use neo_params::NeoParams;
use neo_math::{F, K, KExtensions};
use neo_ccs::{CcsStructure, Mat};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use neo_fold::pi_ccs::{self, precompute, context};
use neo_fold::pi_ccs::sparse_matrix::to_csr;
use neo_fold::pi_ccs::oracle::engine::GenericCcsOracle;
use neo_fold::sumcheck::RoundOracle;

use crate::helpers::{DummyS, create_test_ccs, mk_mcs};

#[derive(Serialize)]
struct TotalsOut {
    #[serde(rename = "p0")] p0: [String; 2],
    #[serde(rename = "p1")] p1: [String; 2],
    #[serde(rename = "S_total")] s_total: [String; 2],
    #[serde(rename = "F_beta_r")] f_beta_r: [String; 2],
    #[serde(rename = "NC_hypercube_sum")] nc_hypercube_sum: [String; 2],
    #[serde(rename = "Eval_total")] eval_total: [String; 2],
    trace_kind: String,
}

fn f_to_str(x: F) -> String { x.as_canonical_u64().to_string() }
fn k_to_pair(k: K) -> [String; 2] { let c = k.as_coeffs(); [f_to_str(c[0]), f_to_str(c[1])] }

#[test]
#[ignore] // utility writer; not a regular unit test
fn write_round0_engine_totals() {
    // Parameters and CCS
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let s: CcsStructure<F> = create_test_ccs().ensure_identity_first().unwrap();
    let l = DummyS;
    let m_in = 1;

    // Instances: one MCS (i=1), one ME input via k=1 simple fold
    let (mcs1, wit1) =
        mk_mcs(&params, vec![F::ONE, F::from_u64(2), F::from_u64(3), F::ZERO, F::from_u64(5)], m_in, &l);
    let (_mcs2, wit2) =
        mk_mcs(&params, vec![F::ONE, F::from_u64(7), F::from_u64(11), F::ZERO, F::from_u64(18)], m_in, &l);

    let mut tr_k1 = Poseidon2Transcript::new(b"fixture/k1");
    let (me_outs, _p1) =
        neo_fold::pi_ccs_prove_simple(&mut tr_k1, &params, &s, &[_mcs2.clone()], &[wit2.clone()], &l).unwrap();
    let me_input = me_outs[0].clone();
    let me_witness: Mat<F> = wit2.Z.clone();

    // Dims + transcript + challenges
    let dims = context::build_dims_and_policy(&params, &s).unwrap();
    let ell_d = dims.ell_d; let ell_n = dims.ell_n; let ell = ell_d + ell_n; let d_sc = dims.d_sc;
    let k_total = 2usize;

    let mut tr = Poseidon2Transcript::new(b"prove/k2");
    pi_ccs::transcript::bind_header_and_instances(&mut tr, &params, &s, &[mcs1.clone()], ell, d_sc, 0).unwrap();
    pi_ccs::transcript::bind_me_inputs(&mut tr, &[me_input.clone()]).unwrap();
    let ch = pi_ccs::transcript::sample_challenges(&mut tr, ell_d, ell).unwrap();

    // CSR matrices
    let mats_csr = s.matrices.iter().map(|m| to_csr::<F>(m, s.n, s.m)).collect::<Vec<_>>();
    // Prepare instances
    let wit1_arr = [wit1.clone()];
    let insts = precompute::prepare_instances(&s, &params, &[mcs1.clone()], &wit1_arr, &mats_csr, &l).unwrap();
    // Partials and precomputes
    let partials_first_inst = precompute::build_mle_partials_first_inst(&s, ell_n, &insts).unwrap();
    let beta_block = precompute::precompute_beta_block(&s, &params, &insts, &[wit1.clone()], &[me_witness.clone()], &ch, ell_d, ell_n).unwrap();
    let eval_row_partial = precompute::precompute_eval_row_partial(&s, &[me_witness.clone()], &ch, k_total, ell_n).unwrap();
    let nc_y_matrices = precompute::precompute_nc_full_rows(&s, &[wit1.clone()], &[me_witness.clone()], ell_n).unwrap();

    // Weights (MLE tensor points)
    let w_beta_a_partial = neo_ccs::utils::tensor_point::<K>(&ch.beta_a);
    let w_alpha_a_partial = neo_ccs::utils::tensor_point::<K>(&ch.alpha);
    let w_beta_r_partial = neo_ccs::utils::tensor_point::<K>(&ch.beta_r);
    let w_eval_r_partial = neo_ccs::utils::tensor_point::<K>(&me_input.r);

    // γ^i over instances (i = 1..k_total)
    let mut nc_row_gamma_pows_vec: Vec<K> = Vec::with_capacity(k_total);
    { let mut g = ch.gamma; for _ in 0..k_total { nc_row_gamma_pows_vec.push(g); g *= ch.gamma; } }

    // Oracle
    let z_witness_refs: Vec<&Mat<F>> = vec![&wit1.Z, &me_witness];
    let me_offset = 1; // first is MCS
    let initial_sum_claim = precompute::compute_initial_sum_components(&beta_block, Some(&me_input.r), &eval_row_partial).unwrap();

    let mut oracle = GenericCcsOracle::new(
        &s,
        partials_first_inst,
        w_beta_a_partial.clone(),
        w_alpha_a_partial.clone(),
        w_beta_r_partial.clone(),
        w_eval_r_partial.clone(),
        eval_row_partial.clone(),
        z_witness_refs,
        &mats_csr[0],
        &mats_csr,
        nc_y_matrices,
        nc_row_gamma_pows_vec,
        ch.gamma,
        k_total,
        params.b,
        ell_d,
        ell_n,
        d_sc,
        me_offset,
        initial_sum_claim,
        beta_block.f_at_beta_r,
        beta_block.nc_sum_hypercube,
    );

    // Round-0 evaluations at x in {0,1}
    let ys = oracle.evals_at(&[K::from(F::ZERO), K::from(F::ONE)]);
    let p0 = ys[0];
    let p1 = ys[1];
    let s_total = p0 + p1;
    let f_beta_r = beta_block.f_at_beta_r;
    let nc_sum = beta_block.nc_sum_hypercube;
    let eval_total = s_total - f_beta_r - nc_sum;

    let out = TotalsOut {
        p0: k_to_pair(p0),
        p1: k_to_pair(p1),
        s_total: k_to_pair(s_total),
        f_beta_r: k_to_pair(f_beta_r),
        nc_hypercube_sum: k_to_pair(nc_sum),
        eval_total: k_to_pair(eval_total),
        trace_kind: "engine_totals".to_string(),
    };

    let out_path = std::env::var("NEO_ENGINE_TOTALS").map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target/engine_totals.json"));
    if let Some(parent) = out_path.parent() { let _ = fs::create_dir_all(parent); }
    fs::write(&out_path, serde_json::to_string_pretty(&out).unwrap()).unwrap();
    eprintln!("[round0_engine_totals] wrote {}", out_path.display());
    assert!(out_path.exists());
}
