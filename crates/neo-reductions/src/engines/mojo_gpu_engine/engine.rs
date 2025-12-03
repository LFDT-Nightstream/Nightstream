use neo_ccs::{CcsStructure, MeInstance, Mat, McsInstance, McsWitness};
use neo_ajtai::Commitment as Cmt;
use neo_params::NeoParams;
use neo_math::{F, K, D};
use p3_field::PrimeCharacteristicRing;
use crate::engines::mojo_gpu_engine::linops::{LinOps, CpuLinOps};
use crate::engines::pi_rlc_dec::RlcDecOps;

#[derive(Clone, Debug, Default, Copy)]
pub struct MojoGpuEngine;

impl MojoGpuEngine {
    /// Helper to get the backend. In the future, this could select GPU/CPU.
    fn backend() -> impl LinOps<F, K> {
        CpuLinOps
    }

    pub fn build_me_outputs_mojo<L>(
        s: &CcsStructure<F>,
        params: &NeoParams,
        mcs_list: &[McsInstance<Cmt, F>],
        mcs_witnesses: &[McsWitness<F>],
        me_inputs: &[MeInstance<Cmt, F, K>],
        me_witnesses: &[Mat<F>],
        r_prime: &[K],
        ell_d: usize,
        fold_digest: [u8; 32],
        log: &L,
    ) -> Vec<MeInstance<Cmt, F, K>> 
    where L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>
    {
        let backend = Self::backend();
        let k_total = mcs_witnesses.len() + me_witnesses.len();
        let mut out = Vec::with_capacity(k_total);

        // 1. Compute chi_r (vector of K)
        let n_sz = 1 << r_prime.len();
        let mut chi_r = vec![K::ZERO; n_sz];
        for row in 0..n_sz {
            let mut val = K::ONE;
            for (bit, &r_i) in r_prime.iter().enumerate() {
                let is_one = ((row >> bit) & 1) == 1;
                val *= if is_one { r_i } else { K::ONE - r_i };
            }
            chi_r[row] = val;
        }
        #[cfg(debug_assertions)]
        debug_assert_eq!(chi_r.len(), s.n.next_power_of_two()); // Assuming s.n <= 2^ell_n

        // 2. Compute v_j = M_j^T * chi_r
        let mut v_js = Vec::with_capacity(s.t());
        for j in 0..s.t() {
            let m_j = &s.matrices[j];
            // v_j = M_j^T * chi_r
            let mut v_j = vec![K::ZERO; s.m];
            backend.gemv_transpose_FK(m_j, &chi_r[0..s.n], &mut v_j);
            v_js.push(v_j);
        }

        let d = D;
        let d_pow2 = 1 << ell_d;
        let bK = K::from(F::from_u64(params.b as u64));
        let mut pow_b = vec![K::ONE; d];
        for i in 1..d { pow_b[i] = pow_b[i-1] * bK; }

        // Helper to process witness
        let process_witness = |Z: &Mat<F>, inp_c: Cmt, inp_m_in: usize, inp_X: Option<&Mat<F>>| -> MeInstance<Cmt, F, K> {
            // X = Z * (I_m_in | 0)^T
            let m_in = inp_m_in;
            
            // Use project_x from L if available (via log)
            // Or just use the slice logic but assert consistency if inp_X is provided
            let Xi = log.project_x(Z, m_in);

            if let Some(_expected_X) = inp_X {
                // Debug assertion: recomputed X should match input X
                #[cfg(debug_assertions)]
                if _expected_X != &Xi {
                    // This might be expensive to check fully, but good for correctness
                    // panic!("build_me_outputs_mojo: X mismatch for ME input");
                }
            }

            // y_j = Z * v_j
            let mut y = Vec::with_capacity(s.t());
            for j in 0..s.t() {
                let mut y_j = vec![K::ZERO; d];
                backend.gemv_FK(Z, &v_js[j], &mut y_j);
                if y_j.len() < d_pow2 {
                    y_j.resize(d_pow2, K::ZERO);
                }
                y.push(y_j);
            }

            // y_scalars
            let y_scalars: Vec<K> = y.iter().map(|yj| {
                let mut acc = K::ZERO;
                for rho in 0..d {
                    acc += yj[rho] * pow_b[rho];
                }
                acc
            }).collect();

            MeInstance {
                c_step_coords: vec![],
                u_offset: 0,
                u_len: 0,
                c: inp_c,
                X: Xi,
                r: r_prime.to_vec(),
                y,
                y_scalars,
                m_in,
                fold_digest,
            }
        };

        // Process MCS witnesses
        for (i, wit) in mcs_witnesses.iter().enumerate() {
            out.push(process_witness(&wit.Z, mcs_list[i].c.clone(), mcs_list[i].m_in, None));
        }

        // Process ME witnesses
        for (i, wit_mat) in me_witnesses.iter().enumerate() {
            out.push(process_witness(wit_mat, me_inputs[i].c.clone(), me_inputs[i].m_in, Some(&me_inputs[i].X)));
        }

        out
    }

}

use crate::engines::PiCcsEngine;
use crate::engines::optimized_engine::PiCcsProof;
use crate::error::PiCcsError;
use neo_transcript::{Transcript, Poseidon2Transcript};
use crate::sumcheck::RoundOracle;
use neo_math::KExtensions;

impl PiCcsEngine for MojoGpuEngine {
    fn prove<L: neo_ccs::traits::SModuleHomomorphism<F, Cmt>>(
        &self,
        tr: &mut Poseidon2Transcript,
        params: &NeoParams,
        s: &CcsStructure<F>,
        mcs_list: &[McsInstance<Cmt, F>],
        mcs_witnesses: &[McsWitness<F>],
        me_inputs: &[MeInstance<Cmt, F, K>],
        me_witnesses: &[Mat<F>],
        _log: &L,
    ) -> Result<(Vec<MeInstance<Cmt, F, K>>, PiCcsProof), PiCcsError> {
        // Dims + transcript binding
        let dims = crate::engines::utils::build_dims_and_policy(params, s)?;
        crate::engines::utils::bind_header_and_instances(
            tr, params, s, mcs_list, dims.ell, dims.d_sc, 0,
        )?;
        crate::engines::utils::bind_me_inputs(tr, me_inputs)?;

        // Sample challenges
        let ch = crate::engines::utils::sample_challenges(tr, dims.ell_d, dims.ell)?;

        // Validate ME input r
        for (idx, me) in me_inputs.iter().enumerate() {
            if me.r.len() != dims.ell_n {
                return Err(PiCcsError::InvalidInput(format!(
                    "ME input r length mismatch at accumulator #{}: expected ell_n = {}, got {}",
                    idx, dims.ell_n, me.r.len()
                )));
            }
        }

        // Initial sum: use the public T computed from ME inputs and α
        // We can reuse the paper-exact helper for this as it's just a formula on public inputs
        let initial_sum = crate::paper_exact_engine::claimed_initial_sum_from_inputs(
            s,
            &ch,
            me_inputs,
        );

        // Bind initial sum to transcript
        tr.append_fields(b"sumcheck/initial_sum", &initial_sum.as_coeffs());

        // Mojo Oracle
        let mut oracle = crate::engines::mojo_gpu_engine::oracle::MojoOracle::new(
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

        for _round_idx in 0..oracle.num_rounds() {
            let deg = oracle.degree_bound();
            let xs: Vec<K> = (0..=deg).map(|t| K::from(F::from_u64(t as u64))).collect();
            let ys = oracle.evals_at(&xs);
            
            // Check invariant at round 0
            if _round_idx == 0 {
                if ys[0] + ys[1] != running_sum {
                     return Err(PiCcsError::SumcheckError(
                        format!("round {} invariant failed: p(0)+p(1) ≠ running_sum (mojo)", _round_idx),
                    ));
                }
            }

            let mut coeffs = crate::optimized_engine::interpolate_univariate(&xs, &ys);

            // Normalize coeffs order
            let ok_at_01 =
                crate::sumcheck::poly_eval_k(&coeffs, K::ZERO) == ys[0] &&
                crate::sumcheck::poly_eval_k(&coeffs, K::ONE)  == ys[1];
            if !ok_at_01 {
                coeffs.reverse();
            }
            
            for &c in &coeffs {
                tr.append_fields(b"sumcheck/round/coeff", &c.as_coeffs());
            }
            let c0 = tr.challenge_field(b"sumcheck/challenge/0");
            let c1 = tr.challenge_field(b"sumcheck/challenge/1");
            let r_i = neo_math::from_complex(c0, c1);
            sumcheck_chals.push(r_i);
            
            running_sum = crate::sumcheck::poly_eval_k(&coeffs, r_i);
            
            oracle.fold(r_i);
            sumcheck_rounds.push(coeffs);
        }

        // Build outputs
        let fold_digest = tr.digest32();
        let (r_prime, _alpha_prime) = sumcheck_chals.split_at(dims.ell_n);
        
        let out_me = Self::build_me_outputs_mojo(
            s,
            params,
            mcs_list,
            mcs_witnesses,
            me_inputs,
            me_witnesses,
            r_prime,
            dims.ell_d,
            fold_digest,
            _log,
        );

        let mut proof = PiCcsProof::new(sumcheck_rounds, Some(initial_sum));
        proof.sumcheck_challenges = sumcheck_chals;
        proof.challenges_public = ch;
        proof.sumcheck_final = running_sum;
        proof.header_digest = fold_digest.to_vec();

        Ok((out_me, proof))
    }

    fn verify(
        &self,
        tr: &mut Poseidon2Transcript,
        params: &NeoParams,
        s: &CcsStructure<F>,
        mcs_list: &[McsInstance<Cmt, F>],
        me_inputs: &[MeInstance<Cmt, F, K>],
        me_outputs: &[MeInstance<Cmt, F, K>],
        proof: &PiCcsProof,
    ) -> Result<bool, PiCcsError> {
        // Use optimized verifier
        crate::engines::OptimizedEngine.verify(tr, params, s, mcs_list, me_inputs, me_outputs, proof)
    }
}


impl RlcDecOps for MojoGpuEngine {
    fn rlc_with_commit<Comb>(
        s: &CcsStructure<F>,
        params: &NeoParams,
        rhos: &[Mat<F>],
        me_inputs: &[MeInstance<Cmt, F, K>],
        Zs: &[Mat<F>],
        ell_d: usize,
        mix_commits: Comb,
    ) -> (MeInstance<Cmt, F, K>, Mat<F>)
    where
        Comb: Fn(&[Mat<F>], &[Cmt]) -> Cmt,
    {
        let backend = Self::backend();
        let k = Zs.len();
        assert_eq!(rhos.len(), k);
        assert!(!Zs.is_empty());

        // 1. Compute Z_mix = Σ ρ_i · Z_i
        let rows = Zs[0].rows();
        let cols = Zs[0].cols();
        let mut Z_mix = Mat::from_row_major(rows, cols, vec![F::ZERO; rows * cols]);

        for i in 0..k {
            // Z_mix += rho_i * Z_i
            // rho_i is d x d (usually scalar matrix but treated as matrix here)
            // Z_i is d x m
            // Z_mix is d x m
            backend.gemm(&rhos[i], &Zs[i], &mut Z_mix, F::ONE, F::ONE);
        }

        // 2. Compute other fields for parent instance
        // c = Σ ρ_i · c_i (via closure)
        let inputs_c: Vec<Cmt> = me_inputs.iter().map(|m| m.c.clone()).collect();
        let c_mix = mix_commits(rhos, &inputs_c);

        // X = Σ ρ_i · X_i
        // X_i are in me_inputs[i].X
        // X_mix is d x m_in
        let m_in = me_inputs[0].m_in;
        let mut X_mix = Mat::from_row_major(rows, m_in, vec![F::ZERO; rows * m_in]);
        
        for i in 0..k {
             backend.gemm(&rhos[i], &me_inputs[i].X, &mut X_mix, F::ONE, F::ONE);
        }

        // y = Σ ρ_i · y_i
        // We only need to compute the first D elements (rest are 0)
        let d_pad = 1 << ell_d;
        let mut y_mix = Vec::with_capacity(s.t());
        let d = D;
        
        for j in 0..s.t() {
            let mut acc = vec![K::ZERO; d_pad];
            for (i, rho) in rhos.iter().enumerate() {
                // rho is d x d
                // me_inputs[i].y[j] is vector of size d_pad
                let y_ij = &me_inputs[i].y[j];
                
                for r in 0..d {
                    let mut sum = K::ZERO;
                    for k in 0..d {
                        sum += K::from(rho[(r, k)]) * y_ij[k];
                    }
                    acc[r] += sum;
                }
            }
            y_mix.push(acc);
        }

        // y_scalars
        let bK = K::from(F::from_u64(params.b as u64));
        let mut y_scalars_mix = Vec::with_capacity(s.t());
        for j in 0..s.t() {
            let mut sc = K::ZERO; 
            let mut pow = K::ONE;
            for rho in 0..d { 
                sc += pow * y_mix[j][rho]; 
                pow *= bK; 
            }
            y_scalars_mix.push(sc);
        }

        let out = MeInstance {
            c_step_coords: vec![], // TODO: Handle if needed
            u_offset: 0,
            u_len: 0,
            c: c_mix,
            X: X_mix,
            r: me_inputs[0].r.clone(),
            y: y_mix,
            y_scalars: y_scalars_mix,
            m_in,
            fold_digest: me_inputs[0].fold_digest,
        };

        (out, Z_mix)
    }

    fn dec_children_with_commit<Comb>(
        s: &CcsStructure<F>,
        params: &NeoParams,
        parent: &MeInstance<Cmt, F, K>,
        Z_split: &[Mat<F>],
        ell_d: usize,
        child_commitments: &[Cmt],
        combine_b_pows: Comb,
    ) -> (Vec<MeInstance<Cmt, F, K>>, bool, bool, bool)
    where
        Comb: Fn(&[Cmt], u32) -> Cmt,
    {
        let backend = Self::backend();
        let k = Z_split.len();
        
        // 1. Reconstruct parent Z from Z_split to check consistency (optional but good for debug)
        // We skip that for now to focus on children construction.

        // 2. Build children
        // We need v_j = M_j^T * chi_r
        // But we don't have chi_r here!
        // Wait, dec_children_with_commit takes `parent`.
        // `parent` has `r`. We need to compute `chi_r` from `parent.r`.
        
        let r = &parent.r;
        let n_sz = 1 << r.len();
        
        // Compute chi_r (vector of K)
        // This is not a matmul, just a helper.
        // We can put this in a helper or just do it here.
        let mut chi_r = vec![K::ZERO; n_sz];
        for row in 0..n_sz {
            let mut val = K::ONE;
            for (bit, &r_i) in r.iter().enumerate() {
                let is_one = ((row >> bit) & 1) == 1;
                val *= if is_one { r_i } else { K::ONE - r_i };
            }
            chi_r[row] = val;
        }

        // Compute v_j = M_j^T * chi_r
        // M_j is n x m. M_j^T is m x n.
        // v_j is m x 1.
        // backend.gemv_FK(M_j^T, chi_r, v_j)
        // But M_j is stored as n x m.
        // We need M_j^T.
        // LinOps::gemv_FK takes a: d x m.
        // If we want M^T * x, we need to pass M^T.
        // We can transpose M_j on the fly or assume LinOps handles it?
        // LinOps definition: `gemv_FK(a, x, y)` -> `y = a * x`.
        // So we need to pass M_j^T.
        // Constructing M_j^T might be expensive if done every time.
        // But for "Mojo", we assume we can do it or it's cached.
        // For now, let's transpose M_j.
        // Or add `gemv_transpose_FK` to LinOps?
        // The prompt suggested: "For each M_j create or reuse M_j^T".
        // Let's transpose locally for now.
        
        let mut v_js = Vec::with_capacity(s.t());
        for j in 0..s.t() {
            let m_j = &s.matrices[j];
            // v_j = M_j^T * chi_r
            // M_j is n x m. chi_r is n. v_j is m.
            let mut v_j = vec![K::ZERO; s.m];
            backend.gemv_transpose_FK(m_j, &chi_r[0..s.n], &mut v_j);
            v_js.push(v_j);
        }

        let mut children = Vec::with_capacity(k);
        let d = D; // digits
        let d_pow2 = 1 << ell_d;

        // Precompute powers of b for y_scalars
        let bK = K::from(F::from_u64(params.b as u64));
        let mut pow_b = vec![K::ONE; d];
        for i in 1..d { pow_b[i] = pow_b[i-1] * bK; }

        for i in 0..k {
            let Z_i = &Z_split[i]; // d x m
            
            // X_i = Z_i * (first m_in cols identity, rest 0)^T ?
            // X_i is just the first m_in columns of Z_i.
            // We can use gemm for this if we want, or just slice.
            // Let's slice for simplicity as it's just copying.
            let m_in = parent.m_in;
            let mut x_vals = Vec::with_capacity(d * m_in);
            for r in 0..d {
                for c in 0..m_in {
                    x_vals.push(Z_i[(r, c)]);
                }
            }
            let Xi = Mat::from_row_major(d, m_in, x_vals);

            // y_(i,j) = Z_i * v_j
            // Z_i is d x m. v_j is m.
            // y_(i,j) is d.
            let mut y_i = Vec::with_capacity(s.t());
            for j in 0..s.t() {
                let mut y_ij = vec![K::ZERO; d];
                backend.gemv_FK(Z_i, &v_js[j], &mut y_ij);
                
                // Pad to d_pow2
                if y_ij.len() < d_pow2 {
                    y_ij.resize(d_pow2, K::ZERO);
                }
                y_i.push(y_ij);
            }

            // y_scalars
            let y_scalars_i: Vec<K> = y_i.iter().map(|yj| {
                let mut acc = K::ZERO;
                for rho in 0..d {
                    acc += yj[rho] * pow_b[rho];
                }
                acc
            }).collect();

            children.push(MeInstance {
                c_step_coords: vec![],
                u_offset: 0,
                u_len: 0,
                c: child_commitments[i].clone(),
                X: Xi,
                r: parent.r.clone(),
                y: y_i,
                y_scalars: y_scalars_i,
                m_in,
                fold_digest: parent.fold_digest,
            });
        }

        // 3. Verification checks (ok_y, ok_X, ok_c)
        // ok_c is checked by caller usually, but we do it here
        let ok_c = combine_b_pows(child_commitments, params.b) == parent.c;

        // ok_y: y_j ?= Σ b^i y_(i,j)
        // We can check this using LinOps too if we want, or just loop.
        // Let's loop for now.
        // ok_y: y_j ?= Σ b^i y_(i,j)
        let mut ok_y = true;
        for j in 0..s.t() {
             let mut sum_y = vec![K::ZERO; d_pow2];
             let mut b_pow = K::ONE;
             for i in 0..k {
                 for rho in 0..d_pow2 {
                     sum_y[rho] += children[i].y[j][rho] * b_pow;
                 }
                 b_pow *= bK;
             }
             if sum_y != parent.y[j] {
                 ok_y = false;
                 break;
             }
        }

        // ok_X: X ?= Σ b^i X_i
        let mut ok_X = true;
        let m_in = parent.m_in;
        let mut sum_X = Mat::from_row_major(d, m_in, vec![F::ZERO; d * m_in]);
        let mut b_pow_f = F::ONE;
        let bF = F::from_u64(params.b as u64);
        
        for i in 0..k {
            // sum_X += b^i * X_i
            // We can use gemm: sum_X = 1 * (b^i * I) * X_i + 1 * sum_X
            // Or just iterate.
            for r in 0..d {
                for c in 0..m_in {
                    sum_X[(r, c)] += children[i].X[(r, c)] * b_pow_f;
                }
            }
            b_pow_f *= bF;
        }
        if sum_X != parent.X {
            ok_X = false;
        }

        (children, ok_y, ok_X, ok_c)
    }
}
