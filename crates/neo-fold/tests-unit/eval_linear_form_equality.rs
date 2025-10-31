use neo_fold::pi_ccs::{context, transcript, precompute};
use neo_fold::pi_ccs::transcript::Challenges;
use neo_transcript::{Poseidon2Transcript, Transcript};
use neo_params::NeoParams;
use neo_math::{F, K};
use neo_ccs::{MeInstance, Mat};
use p3_field::PrimeCharacteristicRing;
use crate::helpers::{DummyS, create_test_ccs, mk_mcs};

#[test]
fn eval_linear_forms_agree_with_inputs() {
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let s = create_test_ccs().ensure_identity_first().unwrap();
    let l = DummyS;
    let m_in = 1;

    // One MCS (2+3=5), one ME input (7+11=18) produced in a separate transcript (lineage independence)
    let (mcs1, _wit1) = mk_mcs(&params, vec![F::ONE, F::from_u64(2), F::from_u64(3), F::ZERO, F::from_u64(5)], m_in, &l);
    let (mcs2, wit2) = mk_mcs(&params, vec![F::ONE, F::from_u64(7), F::from_u64(11), F::ZERO, F::from_u64(18)], m_in, &l);

    // Make an ME input from mcs2 via a k=1 fold
    let mut tr_k1 = Poseidon2Transcript::new(b"fixture/k1");
    let (me_outs, _proof_k1) = neo_fold::pi_ccs_prove_simple(&mut tr_k1, &params, &s, &[mcs2.clone()], &[wit2.clone()], &l).unwrap();
    let me_input: MeInstance<_,_,K> = me_outs[0].clone();
    let me_witness: Mat<F> = wit2.Z.clone();

    // --- Now prepare the state for the k=2 fold precomputations ---
    let dims = context::build_dims_and_policy(&params, &s).unwrap();
    let ell_d = dims.ell_d;
    let ell_n = dims.ell_n;
    let ell   = dims.ell;

    let mut tr = Poseidon2Transcript::new(b"prove/k2");
    transcript::bind_header_and_instances(&mut tr, &params, &s, &[mcs1.clone()], ell, dims.d_sc, 0).unwrap();
    transcript::bind_me_inputs(&mut tr, &[me_input.clone()]).unwrap();
    let ch: Challenges = transcript::sample_challenges(&mut tr, ell_d, ell).unwrap();

    // Precompute Eval row partial and row gate for r (from ME input)
    let eval_row_partial = precompute::precompute_eval_row_partial(&s, &[me_witness.clone()], &ch, 1+1, ell_n).unwrap();
    let mut w_eval_r = vec![K::ZERO; 1<<ell_n];
    let chi_r = neo_ccs::utils::tensor_point::<K>(&me_input.r);
    for i in 0..(1<<ell_n) { w_eval_r[i] = chi_r[i]; }

    // T_oracle = <G_eval(row), χ_r>
    let t_oracle: K = eval_row_partial.iter().zip(&w_eval_r).map(|(g,w)| *g * *w).sum();

    // T_inputs = Σ_j Σ_{ME i} γ^{(i-1)+j·k} · <y_{(i,j)}, χ_α>
    let chi_alpha = neo_ccs::utils::tensor_point::<K>(&ch.alpha);
    let k_total = 1 + 1; // one MCS + one ME input
    let mut gamma_to_k = K::ONE; for _ in 0..k_total { gamma_to_k *= ch.gamma; }
    let mut t_inputs = K::ZERO;
    for j in 0..s.t() {
        // only the single ME input
        let y_mle: K = me_input.y[j].iter().zip(&chi_alpha).map(|(&y,&w)| y*w).sum();
        // weight = γ^{(i-1)+j·k} with i=2 -> (i-1)=1
        let mut w = ch.gamma; // γ^{1}
        for _ in 0..j { w *= gamma_to_k; }
        t_inputs += w * y_mle;
    }

    assert_eq!(t_oracle, t_inputs, "Eval initial-sum mismatch between oracle and inputs");
}
