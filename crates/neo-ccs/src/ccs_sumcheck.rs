use crate::{CcsInstance, CcsStructure, CcsWitness};
use neo_math::{embed_base_to_ext, from_base};
use neo_sumcheck::{
    fiat_shamir::{batch_unis, fs_absorb_poly, fs_absorb_extf, fs_absorb_u64, fs_challenge_ext},
    ExtF, Polynomial, F,
};
use thiserror::Error;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;
use rayon::prelude::*;
use std::sync::Mutex;

/// Sample blind polynomial coefficients from Fiat-Shamir transcript
fn fs_sample_blind_poly(transcript: &mut Vec<u8>, deg: usize) -> Polynomial<ExtF> {
    let mut coeffs = Vec::with_capacity(deg + 1);
    for i in 0..=deg {
        let label = format!("ccs.blind_coeff_{}", i);
        coeffs.push(fs_challenge_ext(transcript, label.as_bytes()));
    }
    Polynomial::new(coeffs)
}

/// CCS sum-check verifier
pub fn ccs_sumcheck_verifier(
    structure: &CcsStructure,
    claim: ExtF,
    msgs: &[(Polynomial<ExtF>, ExtF)],
    norm_bound: u64,
    transcript: &mut Vec<u8>,
) -> Option<(Vec<ExtF>, ExtF)> {
    let do_norm = norm_bound != 0;
    let _alpha = fs_challenge_ext(transcript, b"ccs.norm_alpha");
    let rho = fs_challenge_ext(transcript, b"ccs.norm_rho");

    let mut current = claim;
    let mut r = Vec::new();

    for (round, (uni, blind_eval)) in msgs.iter().enumerate() {
        fs_absorb_u64(transcript, b"ccs.round", round as u64);
        
        // Reconstruct blind_factor at the same transcript state as prover
        let batch_deg_bound = if structure.f.max_individual_degree() == 1 {
            // Multilinear branch
            if do_norm {
                (2 * (norm_bound as usize) + 2).max(2)
            } else {
                2
            }
        } else {
            // General-degree branch
            let deg_f = structure.max_deg;
            if do_norm {
                (deg_f + 1).max(2 * (norm_bound as usize) + 2)
            } else {
                deg_f + 1
            }
        };
        let blind_deg = batch_deg_bound.saturating_sub(2);
        let blind_poly = fs_sample_blind_poly(transcript, blind_deg);
        let x_poly = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE]);
        let xm1_poly = Polynomial::new(vec![-ExtF::ONE, ExtF::ONE]);
        let blind_factor = x_poly * xm1_poly * blind_poly;
        
        let eval_0 = uni.eval(ExtF::ZERO);
        let eval_1 = uni.eval(ExtF::ONE);
        let sum = eval_0 + eval_1;
        eprintln!("ccs_sumcheck_verifier: Round {round}: eval(0)={eval_0:?}, eval(1)={eval_1:?}, sum={sum:?}, current={current:?}");
        if sum != current {
            eprintln!("ccs_sumcheck_verifier: FAIL - sum check failed in round {round}");
            return None;
        }
        
        fs_absorb_poly(transcript, b"ccs.uni", uni);
        let challenge = fs_challenge_ext(transcript, b"ccs.challenge");
        let blind_weight = if do_norm { rho * rho } else { rho };
        let blind_eval_expected = blind_factor.eval(challenge) * blind_weight;
        
        // Security check: verify the provided blind_eval matches expected
        if *blind_eval != blind_eval_expected {
            eprintln!("ccs_sumcheck_verifier: FAIL - blind_eval mismatch in round {round}: got {:?}, expected {:?}", 
                     *blind_eval, blind_eval_expected);
            return None;
        }
        
        current = uni.eval(challenge) - blind_eval_expected;
        fs_absorb_extf(transcript, b"ccs.blind_eval", *blind_eval);
        r.push(challenge);
    }

    // NARK mode: Direct polynomial check - verify that current reduces to zero
    if current == ExtF::ZERO {
        Some((r, ExtF::ZERO))
    } else {
        None
    }
}

/// CCS sum-check prover

#[derive(Error, Debug)]
pub enum ProverError {
    #[error("Sum mismatch in round {0}")]
    SumMismatch(usize),
}

pub fn ccs_sumcheck_prover(
    structure: &CcsStructure,
    instance: &CcsInstance,
    witness: &CcsWitness,
    norm_bound: u64,
    transcript: &mut Vec<u8>,
) -> Result<Vec<(Polynomial<ExtF>, ExtF)>, ProverError> {
    // Fix 1: Gate norm computation behind norm_bound != 0
    let do_norm = norm_bound != 0;
    
    let m = structure.num_constraints;
    let l_ccs = (m as f64).log2().ceil() as usize;

    let mut full_z: Vec<ExtF> = instance
        .public_input
        .iter()
        .map(|&x| embed_base_to_ext(x))
        .collect();
    full_z.extend_from_slice(&witness.z);
    let n_w = full_z.len();
    let l_norm = (n_w as f64).log2().ceil() as usize;
    let l = l_ccs.max(l_norm);
    let padded = 1 << l;

    let s = structure.mats.len();

    let mut mjz_rows = vec![vec![ExtF::ZERO; structure.num_constraints]; s];
    for j in 0..s {
        for b in 0..structure.num_constraints {
            let mut sum = ExtF::ZERO;
            for k in 0..structure.witness_size {
                let m = structure.mats[j].get(b, k).unwrap_or(ExtF::ZERO);
                sum += m * full_z[k];
            }
            mjz_rows[j][b] = sum;
        }
    }

    let mut tables = vec![vec![ExtF::ZERO; padded]; s];
    for (j, table) in tables.iter_mut().enumerate() {
        for (b, table_entry) in table.iter_mut().enumerate().take(m) {
            *table_entry = mjz_rows[j][b];
        }
    }

    let mut witness_table = vec![ExtF::ZERO; padded];
    for (dest, &src) in witness_table.iter_mut().zip(&full_z) {
        *dest = src;
    }

    let alpha = fs_challenge_ext(transcript, b"ccs.norm_alpha");
    let mut alpha_table = vec![ExtF::ONE; padded];
    for i in 1..padded {
        alpha_table[i] = alpha_table[i - 1] * alpha;
    }

    let rho = fs_challenge_ext(transcript, b"ccs.norm_rho");

    let mut current;
    let mut msgs = Vec::with_capacity(l);

    // NARK mode: No commitments needed

    fn naive_mul(a: &[ExtF], b: &[ExtF]) -> Vec<ExtF> {
        let mut res = vec![ExtF::ZERO; a.len() + b.len() - 1];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                res[i + j] += ai * bj;
            }
        }
        res
    }

    fn add_slices(a: &[ExtF], b: &[ExtF]) -> Vec<ExtF> {
        let n = a.len().max(b.len());
        let mut out = vec![ExtF::ZERO; n];
        out[..a.len()].copy_from_slice(a);
        for (i, &bi) in b.iter().enumerate() {
            out[i] += bi;
        }
        out
    }

    fn sub_assign(target: &mut Vec<ExtF>, other: &[ExtF]) {
        if target.len() < other.len() {
            target.resize(other.len(), ExtF::ZERO);
        }
        for (i, &oi) in other.iter().enumerate() {
            target[i] -= oi;
        }
    }

    fn karatsuba_mul(a: &[ExtF], b: &[ExtF]) -> Vec<ExtF> {
        let n = a.len().max(b.len());
        if n <= 64 {
            return naive_mul(a, b);
        }
        let m = n / 2;
        let (a0, a1) = a.split_at(m.min(a.len()));
        let (b0, b1) = b.split_at(m.min(b.len()));
        let p0 = karatsuba_mul(a0, b0);
        let p2 = karatsuba_mul(a1, b1);
        let a01 = add_slices(a0, a1);
        let b01 = add_slices(b0, b1);
        let mut p1 = karatsuba_mul(&a01, &b01);
        sub_assign(&mut p1, &p0);
        sub_assign(&mut p1, &p2);
        let mut res = vec![ExtF::ZERO; a.len() + b.len() - 1];
        res[..p0.len()]
            .iter_mut()
            .zip(&p0)
            .for_each(|(r, &p)| *r += p);
        let offset_m = &mut res[m..m + p1.len()];
        offset_m.iter_mut().zip(&p1).for_each(|(r, &p)| *r += p);
        let offset_2m = &mut res[2 * m..2 * m + p2.len()];
        offset_2m.iter_mut().zip(&p2).for_each(|(r, &p)| *r += p);
        res
    }

    fn phi_of_w(w_poly: &[ExtF], b: u64) -> Vec<ExtF> {
        let mut result = w_poly.to_vec();
        let w_sq = karatsuba_mul(w_poly, w_poly);
        for k in 1..=b {
            let mut term = w_sq.clone();
            let kf = from_base(F::from_u64(k));
            term[0] -= kf * kf;
            result = karatsuba_mul(&result, &term);
        }
        result
    }

    fn phi_val(w: ExtF, b: u64) -> ExtF {
        let mut result = w;
        for k in 1..=b {
            let kf = from_base(F::from_u64(k));
            result *= w * w - kf * kf;
        }
        result
    }

    if structure.f.max_individual_degree() == 1 {
        // Separate CCS and norm claims before batching
        let mut claim_ccs = ExtF::ZERO;
        let mut claim_norm = ExtF::ZERO;
        for i in 0..padded {
            let mut inputs = vec![ExtF::ZERO; s];
            for j in 0..s {
                inputs[j] = tables[j][i];
            }
            claim_ccs += structure.f.evaluate(&inputs) * alpha_table[i];
            if do_norm {
                let w = witness_table[i];
                let phi = phi_val(w, norm_bound);
                claim_norm += rho * phi * alpha_table[i];
            }
        }
        current = if do_norm { claim_ccs + claim_norm } else { claim_ccs };

        for round in 0..l {
            fs_absorb_u64(transcript, b"ccs.round", round as u64);
            let half = witness_table.len() / 2;

            // Fold CCS tables with alpha: degree-1 f → per-pair f_poly(x) = f0 + x(f1-f0)
            // alpha is linear per pair: alpha_poly(x) = α0 + x(α1 - α0)
            // CCS univariate is Σ_i alpha_poly_i(x) * f_poly_i(x)  ⇒ degree ≤ 2
            let mut coeffs_ccs = vec![ExtF::ZERO; 3]; // 0..=2
            for i in 0..half {
                let mut inputs0 = vec![ExtF::ZERO; s];
                let mut inputs1 = vec![ExtF::ZERO; s];
                for j in 0..s {
                    inputs0[j] = tables[j][i];
                    inputs1[j] = tables[j][i + half];
                }
                let f0 = structure.f.evaluate(&inputs0);
                let f1 = structure.f.evaluate(&inputs1);
                let f_poly = vec![f0, f1 - f0]; // degree 1 in x

                let alpha_poly = vec![
                    alpha_table[i],
                    alpha_table[i + half] - alpha_table[i],
                ]; // degree 1 in x

                let pair_poly = karatsuba_mul(&alpha_poly, &f_poly); // degree ≤ 2
                for (d, &c) in pair_poly.iter().enumerate() {
                    coeffs_ccs[d] += c;
                }
            }
            let ccs_uni = Polynomial::new(coeffs_ccs);

            // Fold norms via phi polynomials (only if do_norm)
            let norm_uni = if do_norm {
                let mut coeffs = vec![ExtF::ZERO; (2 * norm_bound as usize) + 3];
                for i in 0..half {
                    let wl = witness_table[i];
                    let wh = witness_table[i + half];
                    let w_poly = vec![wl, wh - wl];
                    let phi_poly = phi_of_w(&w_poly, norm_bound);
                    let alpha_poly = vec![alpha_table[i], alpha_table[i + half] - alpha_table[i]];
                    let pair_poly = karatsuba_mul(&alpha_poly, &phi_poly);
                    for (d, &c) in pair_poly.iter().enumerate() {
                        if d < coeffs.len() {
                            coeffs[d] += c;
                        }
                    }
                }
                Polynomial::new(coeffs)
            } else {
                Polynomial::new(vec![ExtF::ZERO])
            };

            // Blind and batch - use deterministic FS sampling for security
            let batch_deg_bound = if do_norm {
                (2 * (norm_bound as usize) + 2).max(2) // norm vs CCS(≤2)
            } else {
                2
            };
            let blind_deg = batch_deg_bound.saturating_sub(2);
            let blind_poly = fs_sample_blind_poly(transcript, blind_deg);
            let x_poly = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE]);
            let xm1_poly = Polynomial::new(vec![-ExtF::ONE, ExtF::ONE]);
            let blind_factor = x_poly * xm1_poly * blind_poly;
            
            let blinded = if do_norm {
                batch_unis(&[ccs_uni, norm_uni, blind_factor.clone()], rho)
            } else {
                batch_unis(&[ccs_uni, blind_factor.clone()], rho)
            };

            if blinded.eval(ExtF::ZERO) + blinded.eval(ExtF::ONE) != current {
                return Err(ProverError::SumMismatch(round));
            }

            fs_absorb_poly(transcript, b"ccs.uni", &blinded);
            let r = fs_challenge_ext(transcript, b"ccs.challenge");
            let blind_weight = if do_norm { rho * rho } else { rho };
            let blind_eval = blind_factor.eval(r) * blind_weight;
            current = blinded.eval(r) - blind_eval;
            fs_absorb_extf(transcript, b"ccs.blind_eval", blind_eval);
            msgs.push((blinded.clone(), blind_eval));

            // Fold tables in place
            for j in 0..s {
                for i in 0..half {
                    let lval = tables[j][i];
                    let hval = tables[j][i + half];
                    tables[j][i] = (ExtF::ONE - r) * lval + r * hval;
                }
                tables[j].truncate(half);
            }
            {
                let (w_low, w_high) = witness_table.split_at_mut(half);
                let (a_low, a_high) = alpha_table.split_at_mut(half);
                for i in 0..half {
                    w_low[i] = (ExtF::ONE - r) * w_low[i] + r * w_high[i];
                    a_low[i] = (ExtF::ONE - r) * a_low[i] + r * a_high[i];
                }
            }
            witness_table.truncate(half);
            alpha_table.truncate(half);
        }
        return Ok(msgs);
    }

    // Compute initial claim for general-degree branch
    let mut claim_ccs = ExtF::ZERO;
    let mut claim_norm = ExtF::ZERO;
    for i in 0..padded {
        let mut inputs = vec![ExtF::ZERO; s];
        for j in 0..s { 
            inputs[j] = tables[j][i]; 
        }
        claim_ccs += structure.f.evaluate(&inputs) * alpha_table[i];
        if do_norm {
            claim_norm += rho * phi_val(witness_table[i], norm_bound) * alpha_table[i];
        }
    }
    current = if do_norm { claim_ccs + claim_norm } else { claim_ccs };

    for round in 0..l {
        fs_absorb_u64(transcript, b"ccs.round", round as u64);
        let half = witness_table.len() / 2;

        let deg_f = structure.max_deg;                  // degree of f along this round
        let num_points = 2 * deg_f + 1;                 // your choice; ok to keep
        let points: Vec<ExtF> = (0..num_points).map(|p| from_base(F::from_u64(p as u64))).collect();

        // Accumulate coefficients for degree ≤ deg_f + 1 (α adds +1)
        let ccs_coeffs = Mutex::new(vec![ExtF::ZERO; deg_f + 2]);

        (0..half).into_par_iter().for_each(|i| {
            // 1) interpolate f_i(x) along the line l→h
            let mut evals_f = vec![ExtF::ZERO; num_points];
            for (p_idx, &x) in points.iter().enumerate() {
                let mut inputs = vec![ExtF::ZERO; s];
                for j in 0..s {
                    let lval = tables[j][i];
                    let hval = tables[j][i + half];
                    inputs[j] = lval + x * (hval - lval);
                }
                evals_f[p_idx] = structure.f.evaluate(&inputs);
            }
            let uni_f = Polynomial::interpolate(&points, &evals_f); // deg ≤ deg_f

            // 2) alpha per pair is linear in x
            let alpha_poly = vec![
                alpha_table[i],
                alpha_table[i + half] - alpha_table[i],
            ];

            // 3) multiply and accumulate
            let pair_poly = karatsuba_mul(&alpha_poly, uni_f.coeffs()); // deg ≤ deg_f + 1
            let mut guard = ccs_coeffs.lock().unwrap();
            if guard.len() < pair_poly.len() {
                guard.resize(pair_poly.len(), ExtF::ZERO);
            }
            for (d, &c) in pair_poly.iter().enumerate() {
                guard[d] += c;
            }
        });
        let ccs_uni = Polynomial::new(ccs_coeffs.into_inner().unwrap());

        let norm_uni = if do_norm {
            let coeffs = Mutex::new(vec![ExtF::ZERO; (2 * norm_bound as usize) + 3]);
            {
                let (witness_low, witness_high) = witness_table.split_at(half);
                let (alpha_low, alpha_high) = alpha_table.split_at(half);
                (0..half).into_par_iter().for_each(|i| {
                    let l = witness_low[i];
                    let delta = witness_high[i] - l;
                    let w_poly = vec![l, delta];
                    let phi_poly = phi_of_w(&w_poly, norm_bound);
                    let alpha_poly = vec![alpha_low[i], alpha_high[i] - alpha_low[i]];
                    let pair_poly = karatsuba_mul(&alpha_poly, &phi_poly);
                    let mut guard = coeffs.lock().unwrap();
                    for (d, &c) in pair_poly.iter().enumerate() {
                        guard[d] += c;
                    }
                });
            }
            Polynomial::new(coeffs.into_inner().unwrap())
        } else {
            Polynomial::new(vec![ExtF::ZERO])
        };

        let batch_deg_bound = if do_norm {
            (deg_f + 1).max(2 * (norm_bound as usize) + 2)
        } else {
            deg_f + 1
        };
        let blind_deg = batch_deg_bound.saturating_sub(2);
        let blind_poly = fs_sample_blind_poly(transcript, blind_deg);
        let x_poly = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE]);
        let xm1_poly = Polynomial::new(vec![-ExtF::ONE, ExtF::ONE]);
        let blind_factor = x_poly * xm1_poly * blind_poly;
        
        let blinded = if do_norm {
            batch_unis(&[ccs_uni, norm_uni, blind_factor.clone()], rho)
        } else {
            batch_unis(&[ccs_uni, blind_factor.clone()], rho)
        };

        if blinded.eval(ExtF::ZERO) + blinded.eval(ExtF::ONE) != current {
            return Err(ProverError::SumMismatch(round));
        }

        fs_absorb_poly(transcript, b"ccs.uni", &blinded);
        let r = fs_challenge_ext(transcript, b"ccs.challenge");
        let blind_weight = if do_norm { rho * rho } else { rho };
        let blind_eval = blind_factor.eval(r) * blind_weight;
        current = blinded.eval(r) - blind_eval;
        fs_absorb_extf(transcript, b"ccs.blind_eval", blind_eval);
        msgs.push((blinded.clone(), blind_eval));

        for j in 0..s {
            for i in 0..half {
                let lval = tables[j][i];
                let hval = tables[j][i + half];
                tables[j][i] = (ExtF::ONE - r) * lval + r * hval;
            }
            tables[j].truncate(half);
        }
        {
            let (w_low, w_high) = witness_table.split_at_mut(half);
            let (a_low, a_high) = alpha_table.split_at_mut(half);
            for i in 0..half {
                let l_val = w_low[i];
                let h_val = w_high[i];
                w_low[i] = (ExtF::ONE - r) * l_val + r * h_val;
                a_low[i] = (ExtF::ONE - r) * a_low[i] + r * a_high[i];
            }
        }
        witness_table.truncate(half);
        alpha_table.truncate(half);
    }

    Ok(msgs)
}


