use crate::{CcsInstance, CcsStructure, CcsWitness};
use neo_fields::{embed_base_to_ext, from_base, ExtFieldNorm, MAX_BLIND_NORM};
use neo_sumcheck::{
    challenger::NeoChallenger,
    fiat_shamir::{batch_unis, fiat_shamir_challenge},
    oracle::serialize_comms,
    Commitment, ExtF, PolyOracle, Polynomial, F,
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
    comms: &[Commitment],
    oracle: &mut impl PolyOracle,
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
        if uni.eval(ExtF::ZERO) + uni.eval(ExtF::ONE) != current {
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

    // Toy mode (no PCS): skip openings and return the algebraic current.
    if comms.is_empty() {
        return Some((r, current));
    }

    transcript.extend(serialize_comms(comms));

    let (evals, proofs) = oracle.open_at_point(&r);
    if !oracle.verify_openings(comms, &r, &evals, &proofs) {
        return None;
    }
    if evals.iter().any(|e| e.abs_norm() > MAX_BLIND_NORM) {
        return None;
    }
    let final_eval = match evals.get(0) {
        Some(&e) => e,
        None => return Some((r, current)),
    };

    if final_eval == current {
        Some((r, final_eval))
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
    oracle: &mut impl PolyOracle,
    transcript: &mut Vec<u8>,
) -> Result<(Vec<(Polynomial<ExtF>, ExtF)>, Vec<Commitment>), ProverError> {
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

    let comms = oracle.commit();
    transcript.extend(serialize_comms(&comms));

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
        return Ok((msgs, comms));
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

    Ok((msgs, comms))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mv_poly;
    use neo_sumcheck::{
        batched_sumcheck_prover, batched_sumcheck_verifier, from_base, multilinear_sumcheck_prover,
        multilinear_sumcheck_verifier, ExtF, FnOracle, MultilinearEvals, UnivPoly, F,
    };
    use p3_field::{PrimeCharacteristicRing, PrimeField64};
    use p3_matrix::dense::RowMajorMatrix;
    use rand::Rng;

    #[test]
    fn test_multilinear_sumcheck_roundtrip() {
        if std::env::var("RUN_LONG_TESTS").is_err() {
            return;
        }
        let ell = 3;
        let n = 1 << ell;
        let original_evals: Vec<ExtF> = (0..n - 2)
            .map(|i| from_base(F::from_u64(i as u64)))
            .collect();
        let mle = MultilinearEvals::new(original_evals.clone());
        let claim = mle.evals.iter().copied().fold(ExtF::ZERO, |a, b| a + b);
        let mut oracle = FnOracle::new(|point: &[ExtF]| {
            let mle = MultilinearEvals::new(original_evals.clone());
            vec![mle.evaluate(point)]
        });
        let mut transcript = vec![];
        let (msgs, comms) = multilinear_sumcheck_prover(
            &mut mle.evals.clone(),
            claim,
            &mut oracle,
            &mut transcript,
        )
        .unwrap();
        let mut vt = vec![];
        assert!(
            multilinear_sumcheck_verifier(claim, &msgs, &comms, &mut oracle, &mut vt).is_some()
        );
    }

    #[test]
    fn test_ccs_zk_prover_hides() {
        if std::env::var("RUN_LONG_TESTS").is_err() {
            return;
        }
        let m = 2;
        let mat = RowMajorMatrix::new(vec![F::ZERO; m], 1);
        let mats = vec![mat];
        let f = mv_poly(|_: &[ExtF]| ExtF::ZERO, 1);
        let structure = CcsStructure::new(mats, f);
        let instance = CcsInstance {
            commitment: vec![],
            public_input: vec![],
            u: F::ZERO,
            e: F::ONE,
        };
        let witness = CcsWitness {
            z: vec![from_base(F::ZERO)],
        };
        let mut oracle = FnOracle::new(|_: &[ExtF]| vec![]);
        let mut prev: Option<Polynomial<ExtF>> = None;
        let mut diff = false;
        let mut rng = rand::rng();
        for _ in 0..5 {
            let mut t = vec![];
            let prefix = rng.random::<u64>().to_be_bytes().to_vec();
            t.extend(prefix);
            let (msgs, _) =
                ccs_sumcheck_prover(&structure, &instance, &witness, 1, &mut oracle, &mut t)
                    .expect("sumcheck");
            if let Some(p) = &prev {
                if *p != msgs[0].0 {
                    diff = true;
                    break;
                }
            }
            prev = Some(msgs[0].0.clone());
        }
        assert!(diff);
    }

    #[test]
    fn test_quadratic_sumcheck() {
        if std::env::var("RUN_LONG_TESTS").is_err() {
            return;
        }
        let num_vars = 3;
        struct QuadraticPoly {
            num_vars: usize,
        }

        impl UnivPoly for QuadraticPoly {
            fn evaluate(&self, point: &[ExtF]) -> ExtF {
                if point.len() != self.num_vars {
                    ExtF::ZERO
                } else {
                    point[0] * point[0] + point[1] * point[2]
                }
            }

            fn degree(&self) -> usize {
                self.num_vars
            }

            fn max_individual_degree(&self) -> usize {
                2
            }
        }

        let poly: Box<dyn UnivPoly> = Box::new(QuadraticPoly { num_vars });
        let mut claim = ExtF::ZERO;
        let domain_size = 1 << num_vars;
        for idx in 0..domain_size {
            let mut point = vec![ExtF::ZERO; num_vars];
            for (j, point_j) in point.iter_mut().enumerate() {
                *point_j = if (idx >> j) & 1 == 1 {
                    ExtF::ONE
                } else {
                    ExtF::ZERO
                };
            }
            claim += poly.evaluate(&point);
        }
        assert_eq!(claim, from_base(F::from_u64(6)));

        let mut oracle = FnOracle::new(|point: &[ExtF]| {
            let poly = QuadraticPoly { num_vars };
            vec![poly.evaluate(point)]
        });
        let mut transcript = vec![];
        let (msgs, comms) =
            batched_sumcheck_prover(&[claim], &[&*poly], &mut oracle, &mut transcript).unwrap();
        let mut vt = vec![];
        let result = batched_sumcheck_verifier(&[claim], &msgs, &comms, &mut oracle, &mut vt);
        assert!(result.is_some());
        let (r, final_evals) = result.unwrap();
        let poly = QuadraticPoly { num_vars };
        assert_eq!(final_evals[0], poly.evaluate(&r));
    }

    #[test]
    fn test_linear_sumcheck() {
        if std::env::var("RUN_LONG_TESTS").is_err() {
            return;
        }
        let num_vars = 2;

        struct LinearPoly {
            num_vars: usize,
        }

        impl UnivPoly for LinearPoly {
            fn evaluate(&self, point: &[ExtF]) -> ExtF {
                if point.len() != self.num_vars {
                    ExtF::ZERO
                } else {
                    point[0] + point[1]
                }
            }

            fn degree(&self) -> usize {
                self.num_vars
            }

            fn max_individual_degree(&self) -> usize {
                1
            }
        }

        let poly: Box<dyn UnivPoly> = Box::new(LinearPoly { num_vars });
        let correct_claim = from_base(F::from_u64(4));
        let mut oracle = FnOracle::new(|point: &[ExtF]| {
            let poly = LinearPoly { num_vars };
            vec![poly.evaluate(point)]
        });
        let mut transcript = vec![];
        let (msgs, comms) =
            batched_sumcheck_prover(&[correct_claim], &[&*poly], &mut oracle, &mut transcript)
                .unwrap();
        let mut vt = vec![];
        assert!(
            batched_sumcheck_verifier(&[correct_claim], &msgs, &comms, &mut oracle, &mut vt)
                .is_some()
        );
    }

    #[test]
    fn test_prover_rejects_invalid_claim() {
        let num_vars = 2;

        struct LinearPoly {
            num_vars: usize,
        }

        impl UnivPoly for LinearPoly {
            fn evaluate(&self, point: &[ExtF]) -> ExtF {
                if point.len() != self.num_vars {
                    ExtF::ZERO
                } else {
                    point[0] + point[1]
                }
            }

            fn degree(&self) -> usize {
                self.num_vars
            }

            fn max_individual_degree(&self) -> usize {
                1
            }
        }

        let poly: Box<dyn UnivPoly> = Box::new(LinearPoly { num_vars });
        let invalid_claim = from_base(F::from_u64(5));
        let mut transcript = vec![];

        let mut oracle = FnOracle::new(|point: &[ExtF]| {
            let poly = LinearPoly { num_vars };
            vec![poly.evaluate(point)]
        });
        let result =
            batched_sumcheck_prover(&[invalid_claim], &[&*poly], &mut oracle, &mut transcript);

        assert!(result.is_err());
    }

    #[test]
    fn test_high_degree_multivariate_sumcheck() {
        if std::env::var("RUN_LONG_TESTS").is_err() {
            return;
        }
        let num_vars = 4;

        struct HighDegreePoly {
            num_vars: usize,
        }

        impl UnivPoly for HighDegreePoly {
            fn evaluate(&self, point: &[ExtF]) -> ExtF {
                if point.len() != self.num_vars {
                    ExtF::ZERO
                } else {
                    let x0 = point[0];
                    let x1 = point[1];
                    let x2 = point[2];
                    let x3 = point[3];
                    x0 * x0 * x0 * x0 + x1 * x1 * x1 * x2 + x3 * x3
                }
            }

            fn degree(&self) -> usize {
                self.num_vars
            }

            fn max_individual_degree(&self) -> usize {
                4
            }
        }

        let poly: Box<dyn UnivPoly> = Box::new(HighDegreePoly { num_vars });

        let domain_size = 1 << num_vars;
        let mut claim = ExtF::ZERO;
        for idx in 0..domain_size {
            let mut point = vec![ExtF::ZERO; num_vars];
            for (j, point_j) in point.iter_mut().enumerate() {
                *point_j = if (idx >> j) & 1 == 1 {
                    ExtF::ONE
                } else {
                    ExtF::ZERO
                };
            }
            claim += poly.evaluate(&point);
        }

        let mut oracle = FnOracle::new(|point: &[ExtF]| {
            let poly = HighDegreePoly { num_vars };
            vec![poly.evaluate(point)]
        });
        let mut transcript = vec![];
        let (msgs, comms) =
            batched_sumcheck_prover(&[claim], &[&*poly], &mut oracle, &mut transcript).unwrap();

        let mut vt = vec![];
        let result = batched_sumcheck_verifier(&[claim], &msgs, &comms, &mut oracle, &mut vt);
        assert!(result.is_some());

        let (r, final_evals) = result.unwrap();
        let poly = HighDegreePoly { num_vars };
        assert_eq!(final_evals[0], poly.evaluate(&r));
    }

    #[test]
    fn test_r1cs_sumcheck_valid() {
        if std::env::var("RUN_LONG_TESTS").is_err() {
            return;
        }
        let witness_size = 5;

        let m0_data = vec![
            F::ZERO,
            F::from_u64(F::ORDER_U64 - 1),
            F::from_u64(F::ORDER_U64 - 1),
            F::ONE,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ONE,
            F::ZERO,
        ];
        let m0 = RowMajorMatrix::new(m0_data, witness_size);

        let m1_data = vec![
            F::ONE,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ONE,
            F::ZERO,
            F::ZERO,
        ];
        let m1 = RowMajorMatrix::new(m1_data, witness_size);

        let m2_data = vec![
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ONE,
        ];
        let m2 = RowMajorMatrix::new(m2_data, witness_size);

        let mats = vec![m0, m1, m2];

        let f = mv_poly(
            |inputs: &[ExtF]| {
                if inputs.len() == 3 {
                    inputs[0] * inputs[1] - inputs[2]
                } else {
                    ExtF::ZERO
                }
            },
            2,
        );

        let structure = CcsStructure::new(mats, f);

        let z_base = vec![
            F::ONE,
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(5),
            F::from_u64(15),
        ];
        let z: Vec<ExtF> = z_base.into_iter().map(from_base).collect();
        let witness = CcsWitness { z };

        let mut transcript = vec![];
        let mut oracle = FnOracle::new(|_: &[ExtF]| vec![]);
        let instance = CcsInstance {
            commitment: vec![],
            public_input: vec![],
            u: F::ZERO,
            e: F::ONE,
        };
        let (_msgs, _comms) = ccs_sumcheck_prover(
            &structure,
            &instance,
            &witness,
            16,
            &mut oracle,
            &mut transcript,
        )
        .expect("sumcheck");

        let mut tmp = vec![];
        tmp.extend(b"norm_alpha");
        let alpha = fiat_shamir_challenge(&tmp);
        let mut sum = ExtF::ZERO;
        for (i, &w_i) in witness.z.iter().enumerate() {
            let mut prod = w_i;
            for k in 1..=16 {
                let kf = from_base(F::from_u64(k));
                prod *= w_i * w_i - kf * kf;
            }
            let mut alpha_i = ExtF::ONE;
            for _ in 0..i {
                alpha_i *= alpha;
            }
            sum += alpha_i * prod;
        }
        assert_eq!(sum, ExtF::ZERO);
    }

    #[test]
    fn test_r1cs_sumcheck_invalid_norm() {
        let witness_size = 5;

        let m0_data = vec![
            F::ZERO,
            F::from_u64(F::ORDER_U64 - 1),
            F::from_u64(F::ORDER_U64 - 1),
            F::ONE,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ONE,
            F::ZERO,
        ];
        let m0 = RowMajorMatrix::new(m0_data, witness_size);

        let m1_data = vec![
            F::ONE,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ONE,
            F::ZERO,
            F::ZERO,
        ];
        let m1 = RowMajorMatrix::new(m1_data, witness_size);

        let m2_data = vec![
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ONE,
        ];
        let m2 = RowMajorMatrix::new(m2_data, witness_size);

        let mats = vec![m0, m1, m2];

        let f = mv_poly(
            |inputs: &[ExtF]| {
                if inputs.len() == 3 {
                    inputs[0] * inputs[1] - inputs[2]
                } else {
                    ExtF::ZERO
                }
            },
            2,
        );

        let structure = CcsStructure::new(mats, f);

        let z_base = vec![
            F::ONE,
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(5),
            F::from_u64(17),
        ];
        let z: Vec<ExtF> = z_base.into_iter().map(from_base).collect();
        let witness = CcsWitness { z };

        let mut transcript = vec![];
        let mut oracle = FnOracle::new(|_: &[ExtF]| vec![]);
        let instance = CcsInstance {
            commitment: vec![],
            public_input: vec![],
            u: F::ZERO,
            e: F::ONE,
        };
        let res = ccs_sumcheck_prover(
            &structure,
            &instance,
            &witness,
            16,
            &mut oracle,
            &mut transcript,
        );
        assert!(res.is_err());
    }

    #[test]
    fn test_r1cs_sumcheck_invalid() {
        let witness_size = 5;

        let m0_data = vec![
            F::ZERO,
            F::from_u64(F::ORDER_U64 - 1),
            F::from_u64(F::ORDER_U64 - 1),
            F::ONE,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ONE,
            F::ZERO,
        ];
        let m0 = RowMajorMatrix::new(m0_data, witness_size);

        let m1_data = vec![
            F::ONE,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ONE,
            F::ZERO,
            F::ZERO,
        ];
        let m1 = RowMajorMatrix::new(m1_data, witness_size);

        let m2_data = vec![
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ONE,
        ];
        let m2 = RowMajorMatrix::new(m2_data, witness_size);

        let mats = vec![m0, m1, m2];

        let f = mv_poly(
            |inputs: &[ExtF]| {
                if inputs.len() == 3 {
                    inputs[0] * inputs[1] - inputs[2]
                } else {
                    ExtF::ZERO
                }
            },
            2,
        );

        let structure = CcsStructure::new(mats, f);

        let z_bad_base = vec![
            F::ONE,
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(5),
            F::from_u64(16),
        ];
        let z_bad: Vec<ExtF> = z_bad_base.into_iter().map(from_base).collect();
        let witness_bad = CcsWitness { z: z_bad };

        let mut transcript = vec![];
        let mut oracle = FnOracle::new(|_: &[ExtF]| vec![]);
        let instance = CcsInstance {
            commitment: vec![],
            public_input: vec![],
            u: F::ZERO,
            e: F::ONE,
        };
        let res = ccs_sumcheck_prover(
            &structure,
            &instance,
            &witness_bad,
            0,
            &mut oracle,
            &mut transcript,
        );
        assert!(res.is_err());
    }

    #[test]
    fn test_ccs_high_deg_f() {
        let mats = vec![RowMajorMatrix::new(vec![F::ONE], 1)];
        let f = mv_poly(|inputs: &[ExtF]| inputs[0] * inputs[0] * inputs[0], 3);
        let structure = CcsStructure::new(mats, f);
        let instance = CcsInstance {
            commitment: vec![],
            public_input: vec![],
            u: F::ZERO,
            e: F::ONE,
        };
        let witness = CcsWitness {
            z: vec![from_base(F::from_u64(2))],
        };
        let mut transcript = vec![];
        let mut oracle = FnOracle::new(|_: &[ExtF]| vec![]);
        let res = ccs_sumcheck_prover(
            &structure,
            &instance,
            &witness,
            3,
            &mut oracle,
            &mut transcript,
        );
        assert!(res.is_ok());
    }

    #[test]
    fn test_sumcheck_public_inputs() {
        let m = 2;
        let mat = RowMajorMatrix::new(vec![F::ZERO, F::ZERO], m);
        let mats = vec![mat];
        let f = mv_poly(|inputs: &[ExtF]| inputs[0], 1);
        let structure = CcsStructure::new(mats, f);
        let instance = CcsInstance {
            commitment: vec![],
            public_input: vec![F::ONE],
            u: F::ZERO,
            e: F::ONE,
        };
        let witness = CcsWitness {
            z: vec![from_base(F::ZERO)],
        };
        let mut transcript = vec![];
        let mut oracle = FnOracle::new(|_: &[ExtF]| vec![]);
        let res = ccs_sumcheck_prover(
            &structure,
            &instance,
            &witness,
            3,
            &mut oracle,
            &mut transcript,
        );
        assert!(res.is_ok());
    }
}
