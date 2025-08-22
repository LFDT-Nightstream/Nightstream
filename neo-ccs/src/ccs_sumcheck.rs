use crate::{CcsInstance, CcsStructure, CcsWitness};
use neo_fields::{embed_base_to_ext, from_base};
use neo_sumcheck::{
    challenger::NeoChallenger,
    fiat_shamir::{batch_unis, fiat_shamir_challenge},
    ExtF, Polynomial, F,
};
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};
use thiserror::Error;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_matrix::Matrix;
use rayon::prelude::*;
use std::sync::Mutex;

/// CCS sum-check verifier
pub fn ccs_sumcheck_verifier(
    _structure: &CcsStructure,
    claim: ExtF,
    msgs: &[(Polynomial<ExtF>, ExtF)],
    _norm_bound: u64,
    transcript: &mut Vec<u8>,
) -> Option<(Vec<ExtF>, ExtF)> {
    transcript.extend(b"norm_alpha");
    let _alpha = fiat_shamir_challenge(transcript);
    transcript.extend(b"ccs_norm_rho");
    let _rho = fiat_shamir_challenge(transcript);

    let mut current = claim;
    let mut r = Vec::new();

    for (round, (uni, blind_eval)) in msgs.iter().enumerate() {
        transcript.extend(format!("neo_ccs_round_{}", round).as_bytes());
        let eval_0 = uni.eval(ExtF::ZERO);
        let eval_1 = uni.eval(ExtF::ONE);
        let sum = eval_0 + eval_1;
        eprintln!("ccs_sumcheck_verifier: Round {}: eval(0)={:?}, eval(1)={:?}, sum={:?}, current={:?}", 
                 round, eval_0, eval_1, sum, current);
        if sum != current {
            eprintln!("ccs_sumcheck_verifier: FAIL - sum check failed in round {}", round);
            return None;
        }
        for &c in uni.coeffs() {
            let arr = c.to_array();
            transcript.extend(&arr[0].as_canonical_u64().to_be_bytes());
            transcript.extend(&arr[1].as_canonical_u64().to_be_bytes());
        }
        let challenge = fiat_shamir_challenge(transcript);
        current = uni.eval(challenge) - *blind_eval;
        transcript.extend(
            blind_eval
                .to_array()
                .iter()
                .flat_map(|f| f.as_canonical_u64().to_be_bytes()),
        );
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
const ZK_SIGMA: f64 = 3.2;

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

    transcript.extend(b"norm_alpha");
    let alpha = fiat_shamir_challenge(transcript);
    let mut alpha_table = vec![ExtF::ONE; padded];
    for i in 1..padded {
        alpha_table[i] = alpha_table[i - 1] * alpha;
    }

    transcript.extend(b"ccs_norm_rho");
    let rho = fiat_shamir_challenge(transcript);

    let mut current = ExtF::ZERO;
    let mut msgs = Vec::with_capacity(l);
    // Derive prover randomness from FS via a challenger bound to transcript
    let mut ch = NeoChallenger::new("neo_ccs_sumcheck");
    ch.observe_bytes("transcript_prefix", transcript);
    let mut seed = [0u8; 32];
    for i in 0..4 {
        let limb = ch
            .challenge_base(&format!("blind_seed_{}", i))
            .as_canonical_u64();
        seed[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_be_bytes());
    }
    let mut rng = ChaCha20Rng::from_seed(seed);

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
            let w = witness_table[i];
            let phi = phi_val(w, norm_bound);
            claim_norm += rho * phi * alpha_table[i];
        }
        current = claim_ccs + claim_norm;

        for round in 0..l {
            transcript.extend(format!("neo_ccs_round_{}", round).as_bytes());
            let half = witness_table.len() / 2;

            // Fold CCS tables directly
            let mut s0_ccs = ExtF::ZERO;
            let mut s1_ccs = ExtF::ZERO;
            for i in 0..half {
                let mut inputs0 = vec![ExtF::ZERO; s];
                let mut inputs1 = vec![ExtF::ZERO; s];
                for j in 0..s {
                    inputs0[j] = tables[j][i];
                    inputs1[j] = tables[j][i + half];
                }
                s0_ccs += structure.f.evaluate(&inputs0);
                s1_ccs += structure.f.evaluate(&inputs1);
            }
            let ccs_uni = Polynomial::new(vec![s0_ccs, s1_ccs - s0_ccs]);

            // Fold norms via phi polynomials
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
            let norm_uni = Polynomial::new(coeffs);

            // Blind and batch
            let blind_deg = structure.max_deg.saturating_sub(2);
            let blind_coeffs: Vec<ExtF> = (0..=blind_deg)
                .map(|_| {
                    let sample: f64 =
                        <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut rng)
                            * ZK_SIGMA;
                    from_base(F::from_i64(sample.round() as i64))
                })
                .collect();
            let blind_poly = Polynomial::new(blind_coeffs);
            let x_poly = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE]);
            let xm1_poly = Polynomial::new(vec![-ExtF::ONE, ExtF::ONE]);
            let blind_factor = x_poly * xm1_poly * blind_poly;
            let blinded = batch_unis(&[ccs_uni, norm_uni, blind_factor.clone()], rho);

            if blinded.eval(ExtF::ZERO) + blinded.eval(ExtF::ONE) != current {
                return Err(ProverError::SumMismatch(round));
            }

            for &c in blinded.coeffs() {
                let arr = c.to_array();
                transcript.extend(&arr[0].as_canonical_u64().to_be_bytes());
                transcript.extend(&arr[1].as_canonical_u64().to_be_bytes());
            }
            let r = fiat_shamir_challenge(transcript);
            let blind_weight = rho * rho;
            let blind_eval = blind_factor.eval(r) * blind_weight;
            current = blinded.eval(r) - blind_eval;
            transcript.extend(
                blind_eval
                    .to_array()
                    .iter()
                    .flat_map(|f| f.as_canonical_u64().to_be_bytes()),
            );
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

    for round in 0..l {
        transcript.extend(format!("neo_ccs_round_{}", round).as_bytes());
        let half = witness_table.len() / 2;

        // No copying: use tables directly via indices
        let max_deg = structure.max_deg;
        let num_points = 2 * max_deg + 1;
        let points: Vec<ExtF> = (0..num_points)
            .map(|p| from_base(F::from_u64(p as u64)))
            .collect();
        let ccs_coeffs = Mutex::new(vec![ExtF::ZERO; num_points]);
        (0..half).into_par_iter().for_each(|i| {
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
            let uni_f = Polynomial::interpolate(&points, &evals_f);
            let mut guard = ccs_coeffs.lock().unwrap();
            for (d, &c) in uni_f.coeffs().iter().enumerate() {
                if d < guard.len() {
                    guard[d] += c;
                }
            }
        });
        let ccs_uni = Polynomial::new(ccs_coeffs.into_inner().unwrap());

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
        let norm_uni = Polynomial::new(coeffs.into_inner().unwrap());

        let blind_deg = structure.max_deg.saturating_sub(2);
        let blind_coeffs: Vec<ExtF> = (0..=blind_deg)
            .map(|_| {
                let sample: f64 =
                    <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut rng)
                        * ZK_SIGMA;
                from_base(F::from_i64(sample.round() as i64))
            })
            .collect();
        let blind_poly = Polynomial::new(blind_coeffs);
        let x_poly = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE]);
        let xm1_poly = Polynomial::new(vec![-ExtF::ONE, ExtF::ONE]);
        let blind_factor = x_poly * xm1_poly * blind_poly;
        let blinded = batch_unis(&[ccs_uni, norm_uni, blind_factor.clone()], rho);

        if blinded.eval(ExtF::ZERO) + blinded.eval(ExtF::ONE) != current {
            return Err(ProverError::SumMismatch(round));
        }

        for &c in blinded.coeffs() {
            let arr = c.to_array();
            transcript.extend(&arr[0].as_canonical_u64().to_be_bytes());
            transcript.extend(&arr[1].as_canonical_u64().to_be_bytes());
        }
        let r = fiat_shamir_challenge(transcript);
        let blind_weight = rho * rho;
        let blind_eval = blind_factor.eval(r) * blind_weight;
        current = blinded.eval(r) - blind_eval;
        transcript.extend(
            blind_eval
                .to_array()
                .iter()
                .flat_map(|f| f.as_canonical_u64().to_be_bytes()),
        );
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


