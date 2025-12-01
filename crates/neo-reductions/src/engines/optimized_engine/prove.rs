//! Paper-exact prove implementation for PiCcsEngine.
//!
//! This module contains the prove logic for the paper-exact engine,
//! which runs the sumcheck protocol using the paper-exact oracle
//! to evaluate true per-round polynomials.

#![allow(non_snake_case)]

use neo_ccs::{CcsStructure, McsInstance, McsWitness, MeInstance, Mat};
use neo_ajtai::Commitment as Cmt;
use neo_params::NeoParams;
use neo_transcript::Poseidon2Transcript;
use neo_math::{F, K};
use neo_transcript::Transcript;
use neo_math::KExtensions;
use p3_field::*;
use crate::sumcheck::RoundOracle;
use crate::error::PiCcsError;
use crate::optimized_engine::PiCcsProof;

use crate::engines::utils;

fn from_u64<F: Field>(v: u64) -> F {
    let mut res = F::ZERO;
    let mut bit = F::ONE;
    for i in 0..64 {
        if (v >> i) & 1 == 1 {
            res += bit;
        }
        bit += bit;
    }
    res
}

/// Optimized build_me_outputs using sparse matrices and efficient eq evaluation.
pub fn build_me_outputs_optimized<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    s: &CcsStructure<F>,
    params: &NeoParams,
    mcs_list: &[McsInstance<Cmt, F>],
    mcs_witnesses: &[McsWitness<F>],
    me_inputs: &[MeInstance<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    r_prime: &[K],
    ell_d: usize,
    fold_digest: [u8; 32],
    _log: &L,
) -> Vec<MeInstance<Cmt, F, K>> {
    let mut out = Vec::with_capacity(mcs_list.len() + me_inputs.len());
    
    // 1. Compute chi_r_prime = eq(r_prime, x) over the hypercube
    let ell_n = r_prime.len();
    let n_rows = 1 << ell_n;
    let mut chi_r = vec![K::ONE];
    chi_r.reserve(n_rows - 1);
    
    for &r_i in r_prime {
        let one_minus_ri = K::ONE - r_i;
        let len = chi_r.len();
        // Extend by duplicating and scaling
        // We want [chi * (1-ri), chi * ri]
        // But to avoid allocation, we can iterate and push
        for i in 0..len {
            let val = chi_r[i];
            chi_r[i] = val * one_minus_ri;
            chi_r.push(val * r_i);
        }
    }
    debug_assert_eq!(chi_r.len(), n_rows);

    // 2. Compute v_j = M_j^T * chi_r for all j
    // Use sparse matrices
    // chi_r is Vec<K>, spmv_transpose takes &[F].
    // We need to split chi_r into real/imaginary parts if K is extension.
    // Assuming K is quadratic extension of F (Goldilocks).
    // If K=F, it's simpler. But generic K.
    
    // We need to handle K components.
    // Let's assume K is degree 2 extension for now (as per codebase).
    // Or generic K. CsrMatrix is generic over F.
    // We can compute v_j_re = M^T * chi_re, v_j_im = M^T * chi_im.
    
    let mut v_js = Vec::with_capacity(s.t());
    
    // Split chi_r into coeffs
    // We need to know the degree of extension.
    // K::D is likely 2.
    // Let's use K::as_coeffs() to be generic.
    // But we need to transpose the layout: from Vec<K> to D vectors of Vec<F>.
    
    // Check if we can assume D=2.
    // The codebase uses KExtensions.
    // Let's assume D=2 for Goldilocks extension which is common here.
    // Or better, inspect K.
    // But we can just iterate.
    
    // Optimization: Pre-allocate component vectors
    // We can't easily know D at compile time here without generic consts or traits.
    // But we can use `as_coeffs` which returns `&[F]`.
    // Let's assume D=2 for now as `neo_math` defines `D=2` usually.
    // Actually `neo_math::D` is the digits base parameter, not extension degree.
    // `K` is `BinomialExtensionField<Goldilocks, 2>`.
    
    let mut chi_comps: Vec<Vec<F>> = vec![vec![F::ZERO; n_rows]; 2]; // Assume degree 2
    for (i, val) in chi_r.iter().enumerate() {
        let coeffs = val.as_coeffs();
        if coeffs.len() > chi_comps.len() {
             chi_comps.resize(coeffs.len(), vec![F::ZERO; n_rows]);
        }
        for (k, &c) in coeffs.iter().enumerate() {
            chi_comps[k][i] = c;
        }
    }
    
    for j in 0..s.t() {
        let csr = &s.sparse_matrices[j];
        // v_j components
        let mut v_j_comps = Vec::with_capacity(chi_comps.len());
        for comp in &chi_comps {
            v_j_comps.push(csr.spmv_transpose(&comp[..csr.rows]));
        }
        
        // Reassemble v_j: Vec<K>
        let m = s.m;
        let mut v_j = vec![K::ZERO; m];
        for c in 0..m {
            let mut coeffs = [F::ZERO; 2]; // Hardcoded 2 for K
            for (k, comp) in v_j_comps.iter().enumerate() {
                if k < 2 { coeffs[k] = comp[c]; }
            }
            v_j[c] = K::from_coeffs(coeffs);
        }
        v_js.push(v_j);
    }

    // 3. Compute y_{(i,j)} = Z_i * v_j
    // Helper to process witnesses
    let process_witness = |z_digits: &Mat<F>, inp: &McsInstance<Cmt, F>| -> MeInstance<Cmt, F, K> {
        let d = neo_math::D; // Digits
        let mut y = Vec::with_capacity(s.t());
        
        for j in 0..s.t() {
            let vj = &v_js[j];
            let mut yj = vec![K::ZERO; d];
            
            // yj[rho] = dot(z_digits[rho], vj)
            // z_digits is d x m
            // vj is m
            
            for rho in 0..d {
                let mut acc = K::ZERO;
                for c in 0..s.m {
                    // z_digits[(rho, c)] is F. vj[c] is K.
                    // acc += z * vj
                    // Optimization: if z is 0, skip.
                    // But z is dense usually.
                    // We can use F::from(z) * vj[c]
                    let z_val = z_digits[(rho, c)];
                    if z_val != F::ZERO {
                         acc += K::from(z_val) * vj[c];
                    }
                }
                yj[rho] = acc;
            }
            
            // Pad to power of 2 if needed
            let d_pow2 = 1 << ell_d;
            if yj.len() < d_pow2 {
                yj.resize(d_pow2, K::ZERO);
            }
            y.push(yj);
        }
        
        // Recompose y_scalars
        let bK = K::from(from_u64::<F>(params.b as u64));
        let mut pow = vec![K::ONE; d];
        for i in 1..d { pow[i] = pow[i-1] * bK; }
        
        let y_scalars: Vec<K> = y.iter().map(|yj| {
            let mut val = K::ZERO;
            for (rho, &y_rho) in yj.iter().enumerate().take(d) {
                val += y_rho * pow[rho];
            }
            val
        }).collect();

        // Extract X from Z (first m_in columns)
        let d_z = z_digits.rows();
        let m_in = inp.m_in;
        let mut x_vals = Vec::with_capacity(d_z * m_in);
        for r in 0..d_z {
            for c in 0..m_in {
                x_vals.push(z_digits[(r, c)]);
            }
        }
        let X_mat = Mat::from_row_major(d_z, m_in, x_vals);

        MeInstance {
            c_step_coords: vec![], u_offset: 0, u_len: 0,
            c: inp.c.clone(),
            X: X_mat,
            r: r_prime.to_vec(),
            y,
            y_scalars,
            m_in: inp.m_in,
            fold_digest,
        }
    };

    // Process MCS witnesses
    for (idx, wit) in mcs_witnesses.iter().enumerate() {
        // We need the corresponding instance for 'c' and 'X'
        // mcs_list[idx]
        out.push(process_witness(&wit.Z, &mcs_list[idx]));
    }
    
    // Process ME witnesses
    // For ME witnesses, we need corresponding ME inputs to get 'c' and 'X'
    // me_inputs[idx]
    for (idx, wit_mat) in me_witnesses.iter().enumerate() {
        // Convert MeInstance to McsInstance-like structure or just use fields
        let inp = &me_inputs[idx];
        // We can reuse process_witness if we construct a dummy McsInstance or just inline logic.
        // Let's inline or adapt.
        // Actually process_witness takes McsInstance just for c and X.
        // We can pass c and X directly.
        
        let d = neo_math::D;
        let mut y = Vec::with_capacity(s.t());
        for j in 0..s.t() {
            let vj = &v_js[j];
            let mut yj = vec![K::ZERO; d];
            for rho in 0..d {
                let mut acc = K::ZERO;
                for c in 0..s.m {
                    let z_val = wit_mat[(rho, c)];
                    if z_val != F::ZERO {
                         acc += K::from(z_val) * vj[c];
                    }
                }
                yj[rho] = acc;
            }
            let d_pow2 = 1 << ell_d;
            if yj.len() < d_pow2 { yj.resize(d_pow2, K::ZERO); }
            y.push(yj);
        }
        
        let bK = K::from(from_u64::<F>(params.b as u64));
        let mut pow = vec![K::ONE; d];
        for i in 1..d { pow[i] = pow[i-1] * bK; }
        
        let y_scalars: Vec<K> = y.iter().map(|yj| {
            let mut val = K::ZERO;
            for (rho, &y_rho) in yj.iter().enumerate().take(d) {
                val += y_rho * pow[rho];
            }
            val
        }).collect();
        
        out.push(MeInstance {
            c_step_coords: vec![], u_offset: 0, u_len: 0,
            c: inp.c.clone(),
            X: inp.X.clone(),
            r: r_prime.to_vec(),
            y,
            y_scalars,
            m_in: inp.m_in,
            fold_digest,
        });
    }

    out
}

/// Paper-exact prove implementation.
///
/// This function runs the sumcheck protocol using the paper-exact oracle,
/// which directly evaluates the polynomial Q over the Boolean hypercube
/// without optimizations.
pub fn optimized_prove<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    mcs_witnesses: &[McsWitness<F>],
    me_inputs: &[MeInstance<Cmt, F, K>],
    me_witnesses: &[Mat<F>],
    log: &L,
) -> Result<(Vec<MeInstance<Cmt, F, K>>, PiCcsProof), PiCcsError> {
    // Dims + transcript binding
    let dims = utils::build_dims_and_policy(params, s)?;
    utils::bind_header_and_instances(
        tr, params, s, mcs_list, dims.ell, dims.d_sc, 0,
    )?;
    utils::bind_me_inputs(tr, me_inputs)?;

    // Sample challenges
    let ch = utils::sample_challenges(tr, dims.ell_d, dims.ell)?;

    // Validate ME input r (if provided)
    for (idx, me) in me_inputs.iter().enumerate() {
        if me.r.len() != dims.ell_n {
            return Err(PiCcsError::InvalidInput(format!(
                "ME input r length mismatch at accumulator #{}: expected ell_n = {}, got {}",
                idx, dims.ell_n, me.r.len()
            )));
        }
    }

    // Initial sum: use the public T computed from ME inputs and α
    // (not the full hypercube sum Q, which includes MCS/NC terms).
    // This ensures invalid witnesses fail the first sumcheck invariant.
    let initial_sum = crate::paper_exact_engine::claimed_initial_sum_from_inputs(
        s,
        &ch,
        me_inputs,
    );
    
    #[cfg(feature = "debug-logs")]
    {
        // ... (omitted for brevity)
    }

    // Bind initial sum to transcript
    tr.append_fields(b"sumcheck/initial_sum", &initial_sum.as_coeffs());

    // Optimized oracle with cached sparse formats and factored algebra
    let mut oracle = super::oracle::OptimizedOracle::new(
        s,
        params,
        mcs_witnesses,
        me_witnesses,
        ch.clone(),
        dims.ell_d,
        dims.ell_n,
        dims.d_sc,
        me_inputs.first().map(|mi| mi.r.as_slice()),
    );

    let mut running_sum = initial_sum;
    let mut sumcheck_rounds: Vec<Vec<K>> = Vec::with_capacity(oracle.num_rounds());
    let mut sumcheck_chals: Vec<K> = Vec::with_capacity(oracle.num_rounds());

    for round_idx in 0..oracle.num_rounds() {
        let deg = oracle.degree_bound();
        let xs: Vec<K> = (0..=deg).map(|t| K::from(F::from_u64(t as u64))).collect();
        let ys = oracle.evals_at(&xs);
        
        #[cfg(feature = "debug-logs")]
        if round_idx == 0 {
            eprintln!("\n[prove] === Round 0 ===");
            eprintln!("[prove] p(0) = {:?}", ys[0]);
            eprintln!("[prove] p(1) = {:?}", ys[1]);
            eprintln!("[prove] p(0) + p(1) = {:?}", ys[0] + ys[1]);
            eprintln!("[prove] running_sum (should equal T) = {:?}", running_sum);
            if ys[0] + ys[1] != running_sum {
                eprintln!("[prove] ERROR: Sumcheck invariant violated!");
                eprintln!("[prove]   This means the witness is invalid or T is computed incorrectly");
            } else {
                eprintln!("[prove] OK: p(0) + p(1) == running_sum");
            }
        }
        
        if ys[0] + ys[1] != running_sum {
            #[cfg(feature = "debug-logs")]
            {
                eprintln!("\n[prove] SUMCHECK FAILED at round {}", round_idx);
                eprintln!("[prove] p(0)+p(1) = {:?}", ys[0] + ys[1]);
                eprintln!("[prove] running_sum = {:?}", running_sum);
                eprintln!("[prove] difference = {:?}", (ys[0] + ys[1]) - running_sum);
            }
            return Err(PiCcsError::SumcheckError(
                format!("round {} invariant failed: p(0)+p(1) ≠ running_sum (paper-exact)", round_idx),
            ));
        }
        // Interpolation may return coefficients in an unspecified order.
        // We MUST store them in low→high order (c0, c1, ..., cn) so that
        // poly_eval_k(coeffs, ·) reproduces ys at x=0,1 and the verifier's
        // invariant p(0)+p(1) == running_sum holds.
        let mut coeffs = crate::optimized_engine::interpolate_univariate(&xs, &ys);

        // If evaluating at 0/1 doesn't recreate ys, flip the order.
        // After this normalization, `coeffs` is guaranteed low→high.
        let ok_at_01 =
            crate::sumcheck::poly_eval_k(&coeffs, K::ZERO) == ys[0] &&
            crate::sumcheck::poly_eval_k(&coeffs, K::ONE)  == ys[1];
        if !ok_at_01 {
            coeffs.reverse();
        }
        
        debug_assert_eq!(crate::sumcheck::poly_eval_k(&coeffs, K::ZERO), ys[0],
            "interpolate_univariate returned coefficients in unexpected order (x=0)");
        debug_assert_eq!(crate::sumcheck::poly_eval_k(&coeffs, K::ONE), ys[1],
            "interpolate_univariate returned coefficients in unexpected order (x=1)");

        for &c in &coeffs {
            tr.append_fields(b"sumcheck/round/coeff", &c.as_coeffs());
        }
        let c0 = tr.challenge_field(b"sumcheck/challenge/0");
        let c1 = tr.challenge_field(b"sumcheck/challenge/1");
        let r_i = neo_math::from_complex(c0, c1);
        sumcheck_chals.push(r_i);
        
        // Evaluate at challenge using poly_eval_k (low→high) for consistency.
        running_sum = crate::sumcheck::poly_eval_k(&coeffs, r_i);
        
        oracle.fold(r_i);
        sumcheck_rounds.push(coeffs);
    }

    // Build outputs literally at r′ using paper-exact helper
    let fold_digest = tr.digest32();
    let (r_prime, _alpha_prime) = sumcheck_chals.split_at(dims.ell_n);
    let out_me = build_me_outputs_optimized(
        s,
        params,
        mcs_list,
        mcs_witnesses,
        me_inputs,
        me_witnesses,
        r_prime,
        dims.ell_d,
        fold_digest,
        log,
    );

    let mut proof = PiCcsProof::new(sumcheck_rounds, Some(initial_sum));
    proof.sumcheck_challenges = sumcheck_chals;
    proof.challenges_public = ch;
    proof.sumcheck_final = running_sum;
    proof.header_digest = fold_digest.to_vec();

    Ok((out_me, proof))
}

/// Simple wrapper for k=1 case (no ME inputs)
pub fn optimized_prove_simple<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
    tr: &mut Poseidon2Transcript,
    params: &NeoParams,
    s: &CcsStructure<F>,
    mcs_list: &[McsInstance<Cmt, F>],
    mcs_witnesses: &[McsWitness<F>],
    log: &L,
) -> Result<(Vec<MeInstance<Cmt, F, K>>, PiCcsProof), PiCcsError> {
    optimized_prove(tr, params, s, mcs_list, mcs_witnesses, &[], &[], log)
}

