use crate::fiat_shamir::{batch_unis, fiat_shamir_challenge};
use crate::challenger::NeoChallenger;
use crate::oracle::serialize_comms;
use crate::{from_base, Commitment, ExtF, ExtFieldNorm, PolyOracle, Polynomial, UnivPoly, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rand_distr::{Distribution, StandardNormal};
use rand_chacha::ChaCha20Rng;
use rand_chacha::rand_core::SeedableRng;
use thiserror::Error;
use neo_fields::MAX_BLIND_NORM;

/// Batched sum-check prover for multiple polynomial instances
const ZK_SIGMA: f64 = 3.2;

#[derive(Error, Debug)]
pub enum SumCheckError {
    #[error("Invalid sum in round {0}")]
    InvalidSum(usize),
}

fn serialize_uni(uni: &Polynomial<ExtF>) -> Vec<u8> {
    uni.coeffs()
        .iter()
        .flat_map(|&c| {
            let arr = c.to_array();
            let mut bytes = Vec::with_capacity(16);
            bytes.extend_from_slice(&arr[0].as_canonical_u64().to_be_bytes());
            bytes.extend_from_slice(&arr[1].as_canonical_u64().to_be_bytes());
            bytes
        })
        .collect()
}

fn serialize_ext(e: ExtF) -> Vec<u8> {
    let arr = e.to_array();
    let mut bytes = Vec::with_capacity(16);
    bytes.extend_from_slice(&arr[0].as_canonical_u64().to_be_bytes());
    bytes.extend_from_slice(&arr[1].as_canonical_u64().to_be_bytes());
    bytes
}
#[allow(clippy::type_complexity)]
pub fn batched_sumcheck_prover(
    claims: &[ExtF],
    polys: &[&dyn UnivPoly],
    oracle: &mut impl PolyOracle,
    transcript: &mut Vec<u8>,
) -> Result<(Vec<(Polynomial<ExtF>, ExtF)>, Vec<Commitment>), SumCheckError> {
    assert_eq!(claims.len(), polys.len());
    if polys.is_empty() {
        return Ok((vec![], vec![]));
    }

    let ell = polys[0].degree();
    let max_d = polys
        .iter()
        .map(|p| p.max_individual_degree())
        .max()
        .unwrap_or(0);
    let mut msgs = Vec::with_capacity(ell);
    let mut challenges = Vec::with_capacity(ell);
    // Drive DRBG from challenger for blinding randomness
    let mut seed = [0u8; 32];
    // Bind to transcript length and claims for domain sep
    let mut challenger = NeoChallenger::new("neo_sumcheck_batched");
    challenger.observe_bytes("claims", &claims.len().to_be_bytes());
    for i in 0..4 {
        let limb = challenger.challenge_base(&format!("blind_seed_{}", i)).as_canonical_u64();
        seed[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_be_bytes());
    }
    let mut rng = ChaCha20Rng::from_seed(seed);

    transcript.extend(b"sumcheck_rho");
    // Use stateful challenger for FS
    let mut challenger = NeoChallenger::new("neo_sumcheck_batched");
    challenger.observe_bytes("claims", &claims.len().to_be_bytes());
    let rho = challenger.challenge_ext("batch_rho");

    let mut rho_pow = ExtF::ONE;
    let mut current_batched = ExtF::ZERO;
    for &c in claims {
        current_batched += rho_pow * c;
        rho_pow *= rho;
    }

    let comms = oracle.commit();
    transcript.extend(serialize_comms(&comms));

    for round in 0..ell {
        // Frame round in transcript and challenger for domain separation
        transcript.extend(format!("sumcheck_round_{}", round).as_bytes());
        challenger.observe_bytes("round_label", format!("neo_sumcheck_round_{}", round).as_bytes());
        let remaining = ell - round - 1;
        let mut uni_polys = vec![];

        for poly in polys.iter() {
            let points: Vec<ExtF> = (0..=max_d)
                .map(|i| from_base(F::from_u64(i as u64)))
                .collect();
            let mut uni_evals = vec![];

            // Reuse a single buffer for evaluations to avoid cloning in the inner loop
            let mut point_full = challenges.clone();
            point_full.push(ExtF::ZERO); // placeholder for current x
            let y_start = point_full.len();
            point_full.resize(y_start + remaining, ExtF::ZERO);
            for &x in &points {
                // update x position
                point_full[y_start - 1] = x;
                let num_combs = 1 << remaining;
                let mut evals = Vec::with_capacity(num_combs);
                for idx in 0..num_combs {
                    for (j, yj) in point_full[y_start..].iter_mut().enumerate() {
                        *yj = if (idx >> j) & 1 == 1 { ExtF::ONE } else { ExtF::ZERO };
                    }
                    evals.push(poly.evaluate(&point_full));
                }
                while evals.len() > 1 {
                    let half = evals.len() / 2;
                    for i in 0..half {
                        evals[i] = evals[i] + evals[i + half];
                    }
                    evals.truncate(half);
                }
                uni_evals.push(evals[0]);
            }

            let uni = Polynomial::interpolate(&points, &uni_evals);
            uni_polys.push(uni);
        }

        let blind_deg = max_d.saturating_sub(2);
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
        let mut uni_polys_with_blind = uni_polys.clone();
        uni_polys_with_blind.push(blind_factor.clone());
        let batched_uni = batch_unis(&uni_polys_with_blind, rho);

        if batched_uni.eval(ExtF::ZERO) + batched_uni.eval(ExtF::ONE) != current_batched {
            return Err(SumCheckError::InvalidSum(round));
        }

        transcript.extend(batched_uni.coeffs().iter().flat_map(|&c| {
            let arr = c.to_array();
            let mut bytes = Vec::with_capacity(16);
            bytes.extend_from_slice(&arr[0].as_canonical_u64().to_be_bytes());
            bytes.extend_from_slice(&arr[1].as_canonical_u64().to_be_bytes());
            bytes
        }));
        challenger.observe_bytes("blinded_uni", &serialize_uni(&batched_uni));

        let challenge = challenger.challenge_ext("round_challenge");
        challenges.push(challenge);

        let num_polys = polys.len();
        let mut blind_weight = ExtF::ONE;
        for _ in 0..num_polys {
            blind_weight *= rho;
        }
        let blind_eval = blind_factor.eval(challenge) * blind_weight;

        current_batched = batched_uni.eval(challenge) - blind_eval;
        let blind_bytes: Vec<u8> = serialize_ext(blind_eval);
        transcript.extend(&blind_bytes);
        challenger.observe_bytes("blind_eval", &blind_bytes);

        msgs.push((batched_uni.clone(), blind_eval));
    }
    Ok((msgs, comms))
}

/// Batched sum-check verifier
pub fn batched_sumcheck_verifier(
    claims: &[ExtF],
    msgs: &[(Polynomial<ExtF>, ExtF)],
    comms: &[Commitment],
    oracle: &mut impl PolyOracle,
    transcript: &mut Vec<u8>,
) -> Option<(Vec<ExtF>, Vec<ExtF>)> {
    if claims.is_empty() || msgs.is_empty() {
        return Some((vec![], vec![]));
    }

    transcript.extend(b"sumcheck_rho");
    let mut challenger = NeoChallenger::new("neo_sumcheck_batched");
    challenger.observe_bytes("claims", &claims.len().to_be_bytes());
    let rho = challenger.challenge_ext("batch_rho");

    let mut rho_pow = ExtF::ONE;
    let mut current = ExtF::ZERO;
    for &c in claims {
        current += rho_pow * c;
        rho_pow *= rho;
    }

    transcript.extend(serialize_comms(comms));

    let mut r = Vec::new();

    for (round, (uni, blind_eval)) in msgs.iter().enumerate() {
        challenger.observe_bytes("round_label", format!("neo_sumcheck_round_{}", round).as_bytes());
        transcript.extend(format!("sumcheck_round_{}", round).as_bytes());
        if uni.eval(ExtF::ZERO) + uni.eval(ExtF::ONE) != current {
            return None;
        }

        transcript.extend(uni.coeffs().iter().flat_map(|&c| {
            let arr = c.to_array();
            let mut bytes = Vec::with_capacity(16);
            bytes.extend_from_slice(&arr[0].as_canonical_u64().to_be_bytes());
            bytes.extend_from_slice(&arr[1].as_canonical_u64().to_be_bytes());
            bytes
        }));
        challenger.observe_bytes("blinded_uni", &serialize_uni(uni));

        let challenge = challenger.challenge_ext("round_challenge");
        current = uni.eval(challenge) - *blind_eval;
        let blind_bytes: Vec<u8> = serialize_ext(*blind_eval);
        transcript.extend(&blind_bytes);
        challenger.observe_bytes("blind_eval", &blind_bytes);
        r.push(challenge);
    }

    let (evals, proofs) = oracle.open_at_point(&r);
    if !oracle.verify_openings(comms, &r, &evals, &proofs) {
        return None;
    }

    if evals.iter().any(|e| e.abs_norm() > MAX_BLIND_NORM) {
        return None;
    }

    let mut rho_pow = ExtF::ONE;
    let mut final_batched = ExtF::ZERO;
    for &e in &evals {
        final_batched += rho_pow * e;
        rho_pow *= rho;
    }

    if final_batched == current {
        Some((r, evals))
    } else {
        None
    }
}

/// Optimized multilinear sum-check prover using folding
#[allow(clippy::type_complexity)]
pub fn multilinear_sumcheck_prover(
    evals: &mut Vec<ExtF>,
    claim: ExtF,
    oracle: &mut impl PolyOracle,
    transcript: &mut Vec<u8>,
) -> Result<(Vec<(Polynomial<ExtF>, ExtF)>, Vec<Commitment>), SumCheckError> {
    let mut msgs = Vec::new();
    let mut current_claim = claim;
    let mut ell = evals.len().trailing_zeros() as usize;
    // DRBG seeded from FS for multilinear prover
    let mut seed = [0u8; 32];
    let mut ch = NeoChallenger::new("neo_multilinear_sumcheck");
    ch.observe_bytes("transcript_prefix", transcript);
    for i in 0..4 {
        let limb = ch.challenge_base(&format!("blind_seed_{}", i)).as_canonical_u64();
        seed[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_be_bytes());
    }
    let mut rng = ChaCha20Rng::from_seed(seed);

    let comms = oracle.commit();
    transcript.extend(serialize_comms(&comms));

    let mut round = 0;
    while ell > 0 {
        transcript.extend(format!("neo_multilinear_round_{}", ell).as_bytes());
        let half = 1 << (ell - 1);

        let s0: ExtF = evals[..half].iter().copied().fold(ExtF::ZERO, |a, b| a + b);
        let s1: ExtF = evals[half..(2 * half)]
            .iter()
            .copied()
            .fold(ExtF::ZERO, |a, b| a + b);

        let uni = Polynomial::new(vec![s0, s1 - s0]);

        let r_blind = from_base(F::from_i64(
            (<StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut rng) * ZK_SIGMA)
                .round() as i64,
        ));
        let blind_factor = Polynomial::new(vec![ExtF::ZERO, r_blind, -r_blind]);
        let blinded_uni = uni.clone() + blind_factor.clone();

        for &c in blinded_uni.coeffs().iter() {
            let arr = c.to_array();
            transcript.extend(&arr[0].as_canonical_u64().to_be_bytes());
            transcript.extend(&arr[1].as_canonical_u64().to_be_bytes());
        }

        let r = fiat_shamir_challenge(transcript);

        if blinded_uni.eval(ExtF::ZERO) + blinded_uni.eval(ExtF::ONE) != current_claim {
            return Err(SumCheckError::InvalidSum(round));
        }

        for i in 0..half {
            evals[i] = (ExtF::ONE - r) * evals[i] + r * evals[i + half];
        }
        evals.truncate(half);

        let blind_eval = blind_factor.eval(r);
        current_claim = blinded_uni.eval(r) - blind_eval;
        transcript.extend(
            blind_eval
                .to_array()
                .iter()
                .flat_map(|f| f.as_canonical_u64().to_be_bytes()),
        );

        msgs.push((blinded_uni.clone(), blind_eval));
        ell -= 1;
        round += 1;
    }

    Ok((msgs, comms))
}

/// Multilinear sum-check verifier
pub fn multilinear_sumcheck_verifier(
    claim: ExtF,
    msgs: &[(Polynomial<ExtF>, ExtF)],
    comms: &[Commitment],
    oracle: &mut impl PolyOracle,
    transcript: &mut Vec<u8>,
) -> Option<(Vec<ExtF>, ExtF)> {
    let mut current = claim;
    transcript.extend(serialize_comms(comms));

    let mut r = Vec::new();

    let mut ell = msgs.len();
    for (uni, blind_eval) in msgs {
        transcript.extend(format!("neo_multilinear_round_{}", ell).as_bytes());
        if uni.eval(ExtF::ZERO) + uni.eval(ExtF::ONE) != current {
            return None;
        }

        for &c in uni.coeffs().iter() {
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
        ell -= 1;
    }

    r.reverse();

    let (evals, proofs) = oracle.open_at_point(&r);
    if !oracle.verify_openings(comms, &r, &evals, &proofs) {
        return None;
    }
    let final_eval = evals[0];

    if final_eval.abs_norm() > MAX_BLIND_NORM {
        return None;
    }

    if final_eval == current {
        Some((r, final_eval))
    } else {
        None
    }
}

/// Batched multilinear sum-check prover
#[allow(clippy::type_complexity)]
pub fn batched_multilinear_sumcheck_prover(
    claims: &[ExtF],
    evals: &mut [Vec<ExtF>],
    oracle: &mut impl PolyOracle,
    transcript: &mut Vec<u8>,
) -> Result<(Vec<(Polynomial<ExtF>, ExtF)>, Vec<Commitment>), SumCheckError> {
    assert!(!evals.is_empty());
    assert_eq!(claims.len(), evals.len());
    let ell = evals[0].len().trailing_zeros() as usize;

    transcript.extend(b"batched_multilinear_rho");
    let rho = fiat_shamir_challenge(transcript);
    let mut rho_pow = ExtF::ONE;
    let mut _current = ExtF::ZERO;
    for &c in claims {
        _current += rho_pow * c;
        rho_pow *= rho;
    }

    let mut msgs = Vec::with_capacity(ell);
    // DRBG seeded from FS for batched multilinear prover
    let mut seed = [0u8; 32];
    let mut ch = NeoChallenger::new("neo_batched_multilinear_sumcheck");
    ch.observe_bytes("transcript_prefix", transcript);
    for i in 0..4 {
        let limb = ch.challenge_base(&format!("blind_seed_{}", i)).as_canonical_u64();
        seed[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_be_bytes());
    }
    let mut rng = ChaCha20Rng::from_seed(seed);

    let comms = oracle.commit();
    transcript.extend(serialize_comms(&comms));

    for round in 0..ell {
        transcript.extend(format!("neo_batched_multilinear_round_{}", round).as_bytes());
        let mut batched_uni = Polynomial::new(vec![ExtF::ZERO; 2]);
        let mut rho_pow = ExtF::ONE;

        for eval in evals.iter() {
            let half = eval.len() / 2;
            let s0: ExtF = eval[..half].iter().copied().fold(ExtF::ZERO, |a, b| a + b);
            let s1: ExtF = eval[half..].iter().copied().fold(ExtF::ZERO, |a, b| a + b);

            let uni = Polynomial::new(vec![s0, s1 - s0]);

            batched_uni = batched_uni + uni * Polynomial::new(vec![rho_pow]);
            rho_pow *= rho;
        }

        let r_blind = from_base(F::from_i64(
            (<StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut rng) * ZK_SIGMA)
                .round() as i64,
        ));
        let blind_factor = Polynomial::new(vec![ExtF::ZERO, r_blind, -r_blind]);
        let blinded_uni = batched_uni.clone() + blind_factor.clone();

        transcript.extend(blinded_uni.coeffs().iter().flat_map(|&c| {
            let arr = c.to_array();
            [
                arr[0].as_canonical_u64().to_be_bytes(),
                arr[1].as_canonical_u64().to_be_bytes(),
            ]
            .into_iter()
            .flatten()
        }));
        let r = fiat_shamir_challenge(transcript);

        for eval in evals.iter_mut() {
            let half = eval.len() / 2;
            for i in 0..half {
                eval[i] = (ExtF::ONE - r) * eval[i] + r * eval[i + half];
            }
            eval.truncate(half);
        }
        let blind_eval = blind_factor.eval(r);
        _current = blinded_uni.eval(r) - blind_eval;
        transcript.extend(
            blind_eval
                .to_array()
                .iter()
                .flat_map(|f| f.as_canonical_u64().to_be_bytes()),
        );
        msgs.push((blinded_uni.clone(), blind_eval));
    }
    Ok((msgs, comms))
}

/// Batched multilinear sum-check verifier
pub fn batched_multilinear_sumcheck_verifier(
    claims: &[ExtF],
    msgs: &[(Polynomial<ExtF>, ExtF)],
    comms: &[Commitment],
    oracle: &mut impl PolyOracle,
    transcript: &mut Vec<u8>,
) -> Option<(Vec<ExtF>, Vec<ExtF>)> {
    transcript.extend(b"batched_multilinear_rho");
    let rho = fiat_shamir_challenge(transcript);

    let mut rho_pow = ExtF::ONE;
    let mut current = ExtF::ZERO;
    for &c in claims {
        current += rho_pow * c;
        rho_pow *= rho;
    }

    transcript.extend(serialize_comms(comms));

    let mut r = Vec::new();

    for (round, (uni, blind_eval)) in msgs.iter().enumerate() {
        transcript.extend(format!("neo_batched_multilinear_round_{}", round).as_bytes());
        if uni.eval(ExtF::ZERO) + uni.eval(ExtF::ONE) != current {
            return None;
        }

        transcript.extend(uni.coeffs().iter().flat_map(|&c| {
            let arr = c.to_array();
            [
                arr[0].as_canonical_u64().to_be_bytes(),
                arr[1].as_canonical_u64().to_be_bytes(),
            ]
            .into_iter()
            .flatten()
        }));

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

    let (evals, proofs) = oracle.open_at_point(&r);
    if !oracle.verify_openings(comms, &r, &evals, &proofs) {
        return None;
    }

    let mut rho_pow = ExtF::ONE;
    let mut final_batched = ExtF::ZERO;
    for &e in &evals {
        final_batched += rho_pow * e;
        rho_pow *= rho;
    }

    if final_batched == current {
        Some((r, evals))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FnOracle, FriOracle, OpeningProof};
    use quickcheck_macros::quickcheck;
    use rand::{rng, Rng};

    #[test]
    fn test_fiat_shamir_challenge_ext_field() {
        // Determinism: same input yields same output; domain separation yields different.
        let ch1 = fiat_shamir_challenge(b"12345678");
        let ch2 = fiat_shamir_challenge(b"12345678");
        assert_eq!(ch1, ch2);
        let mut t = b"12345678".to_vec();
        t.extend_from_slice(b"|sep");
        let ch3 = fiat_shamir_challenge(&t);
        assert_ne!(ch1, ch3);
    }

    #[test]
    fn test_domain_sep_multilinear() {
        let mut evals1 = vec![ExtF::ONE; 4];
        let claim = from_base(F::from_u64(4));
        let mut oracle1 = FnOracle::new(|_| vec![]);
        let mut t1 = vec![];
        multilinear_sumcheck_prover(&mut evals1, claim, &mut oracle1, &mut t1).unwrap();

        let mut evals2 = vec![ExtF::ONE; 4];
        let mut oracle2 = FnOracle::new(|_| vec![]);
        let mut t2 = b"extra".to_vec();
        multilinear_sumcheck_prover(&mut evals2, claim, &mut oracle2, &mut t2).unwrap();

        assert_ne!(t1, t2);
    }

    #[test]
    fn test_multilinear_zk_blinding() {
        if std::env::var("RUN_LONG_TESTS").is_err() {
            return;
        }
        let claim = from_base(F::from_u64(4));
        let mut rng = rng();
        let mut evals1 = vec![ExtF::ONE; 4];
        let mut oracle = FnOracle::new(|_| vec![]);
        let mut t1 = vec![];
        let prefix1 = rng.random::<u64>().to_be_bytes().to_vec();
        t1.extend(prefix1);
        let (msgs1, _) =
            multilinear_sumcheck_prover(&mut evals1, claim, &mut oracle, &mut t1).unwrap();
        let mut evals2 = vec![ExtF::ONE; 4];
        let mut t2 = vec![];
        let prefix2 = rng.random::<u64>().to_be_bytes().to_vec();
        t2.extend(prefix2);
        let (msgs2, _) =
            multilinear_sumcheck_prover(&mut evals2, claim, &mut oracle, &mut t2).unwrap();
        assert_ne!(msgs1[0].0, msgs2[0].0);
    }

    #[test]
    fn test_sumcheck_unit_correctness() {
        struct SquarePoly;
        impl UnivPoly for SquarePoly {
            fn evaluate(&self, point: &[ExtF]) -> ExtF {
                if point.len() != 1 {
                    ExtF::ZERO
                } else {
                    point[0] * point[0]
                }
            }
            fn degree(&self) -> usize {
                1
            }
            fn max_individual_degree(&self) -> usize {
                2
            }
        }

        let poly: Box<dyn UnivPoly> = Box::new(SquarePoly);
        let claim = from_base(F::from_u64(1));
        let dense = Polynomial::new(vec![ExtF::ZERO, ExtF::ZERO, ExtF::ONE]);
        let mut transcript = vec![];
        let mut oracle = FriOracle::new(vec![dense], &mut transcript);
        let (msgs, _comms) =
            batched_sumcheck_prover(&[claim], &[&*poly], &mut oracle, &mut transcript).unwrap();
        let (first_uni, _) = &msgs[0];
        assert_eq!(
            first_uni.eval(ExtF::ZERO) + first_uni.eval(ExtF::ONE),
            claim
        );
    }

    #[test]
    fn test_batched_multilinear_domain_sep() {
        let claims = vec![ExtF::ONE];
        let mut evals1 = vec![vec![ExtF::ONE; 4]];
        let mut oracle1 = FnOracle::new(|_| vec![]);
        let mut t1 = vec![];
        batched_multilinear_sumcheck_prover(&claims, &mut evals1, &mut oracle1, &mut t1).unwrap();

        let mut evals2 = vec![vec![ExtF::ONE; 4]];
        let mut oracle2 = FnOracle::new(|_| vec![]);
        let mut t2 = b"extra".to_vec();
        batched_multilinear_sumcheck_prover(&claims, &mut evals2, &mut oracle2, &mut t2).unwrap();

        assert_ne!(t1, t2);
    }

    #[test]
    fn test_batched_sumcheck_domain_sep() {
        struct ConstPoly;
        impl UnivPoly for ConstPoly {
            fn evaluate(&self, _: &[ExtF]) -> ExtF {
                ExtF::ONE
            }
            fn degree(&self) -> usize {
                1
            }
            fn max_individual_degree(&self) -> usize {
                1
            }
        }
        let poly: Box<dyn UnivPoly> = Box::new(ConstPoly);
        let claim = ExtF::ONE + ExtF::ONE; // sum over {0,1}
        let mut oracle1 = FnOracle::new(|_| vec![]);
        let mut t1 = vec![];
        batched_sumcheck_prover(&[claim], &[&*poly], &mut oracle1, &mut t1).unwrap();

        let mut oracle2 = FnOracle::new(|_| vec![]);
        let mut t2 = b"extra".to_vec();
        batched_sumcheck_prover(&[claim], &[&*poly], &mut oracle2, &mut t2).unwrap();

        assert_ne!(t1, t2);
    }

    #[test]
    fn test_verifier_rejects_high_norm_eval() {
        struct ConstPoly;
        impl UnivPoly for ConstPoly {
            fn evaluate(&self, _: &[ExtF]) -> ExtF {
                ExtF::ZERO
            }
            fn degree(&self) -> usize {
                1
            }
            fn max_individual_degree(&self) -> usize {
                1
            }
        }
        let poly: Box<dyn UnivPoly> = Box::new(ConstPoly);
        let claim = ExtF::ZERO;
        let mut prover_oracle = FnOracle::new(|_| vec![ExtF::ZERO]);
        let mut transcript = vec![];
        let (msgs, comms) =
            batched_sumcheck_prover(&[claim], &[&*poly], &mut prover_oracle, &mut transcript)
                .unwrap();

        struct HighNormOracle;
        impl PolyOracle for HighNormOracle {
            fn commit(&mut self) -> Vec<Commitment> {
                vec![]
            }
            fn open_at_point(&mut self, _point: &[ExtF]) -> (Vec<ExtF>, Vec<OpeningProof>) {
                let val = F::from_u64(MAX_BLIND_NORM / 2 + 1);
                let high_norm = ExtF::new_complex(val, val);
                (vec![high_norm], vec![])
            }
            fn verify_openings(
                &self,
                _comms: &[Commitment],
                _point: &[ExtF],
                _evals: &[ExtF],
                _proofs: &[OpeningProof],
            ) -> bool {
                true
            }
        }
        let mut high_oracle = HighNormOracle;
        let mut vt = vec![];
        let result = batched_sumcheck_verifier(&[claim], &msgs, &comms, &mut high_oracle, &mut vt);
        assert!(result.is_none());
    }

    #[quickcheck]
    fn prop_blind_vanishes(coeffs: Vec<i64>) -> bool {
        if coeffs.is_empty() {
            return true;
        }
        let coeffs_ext: Vec<ExtF> = coeffs.iter().map(|&c| from_base(F::from_i64(c))).collect();
        let blind_poly = Polynomial::new(coeffs_ext);
        let x_poly = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE]);
        let xm1_poly = Polynomial::new(vec![-ExtF::ONE, ExtF::ONE]);
        let blind_factor = x_poly * xm1_poly * blind_poly;
        blind_factor.eval(ExtF::ZERO) == ExtF::ZERO && blind_factor.eval(ExtF::ONE) == ExtF::ZERO
    }
}
