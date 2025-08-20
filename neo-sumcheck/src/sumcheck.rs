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

pub fn serialize_uni(uni: &Polynomial<ExtF>) -> Vec<u8> {
    // CRITICAL FIX: Prefix degree as u8 to match verifier expectation
    let mut bytes = vec![uni.degree() as u8];
    
    // CRITICAL: The verifier expects degree+1 coefficients, but Polynomial::new trims zeros
    // If the polynomial is empty (trimmed), we need to serialize it as a degree-0 polynomial with a zero coefficient
    if uni.coeffs().is_empty() {
        // Serialize as degree 0 with one zero coefficient
        bytes[0] = 0; // Explicitly set degree to 0
        let zero_arr = ExtF::ZERO.to_array();
        bytes.extend_from_slice(&zero_arr[0].as_canonical_u64().to_be_bytes());
        bytes.extend_from_slice(&zero_arr[1].as_canonical_u64().to_be_bytes());
    } else {
        // Normal case: serialize all coefficients
        for &c in uni.coeffs().iter() {
            let arr = c.to_array();
            bytes.extend_from_slice(&arr[0].as_canonical_u64().to_be_bytes());
            bytes.extend_from_slice(&arr[1].as_canonical_u64().to_be_bytes());
        }
    }
    bytes
}

pub fn serialize_ext(e: ExtF) -> Vec<u8> {
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
    // Bind to transcript content and claims for domain separation (CRITICAL: for computational hiding)
    let mut challenger = NeoChallenger::new("neo_sumcheck_batched");
    challenger.observe_bytes("transcript_prefix", transcript);  // HIDING FIX: Observe transcript for unique seeds
    challenger.observe_bytes("claims", &claims.len().to_be_bytes());
    for i in 0..4 {
        let limb = challenger.challenge_base(&format!("blind_seed_{}", i)).as_canonical_u64();
        seed[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_be_bytes());
    }
    let mut rng = ChaCha20Rng::from_seed(seed);

    // CRITICAL FIX: Add rho domain separator BEFORE deriving challenge (match verifier)
    transcript.extend(b"sumcheck_rho");
    let rho = fiat_shamir_challenge(transcript);

    let mut rho_pow = ExtF::ONE;
    let mut current_batched = ExtF::ZERO;
    for &c in claims {
        current_batched += rho_pow * c;
        rho_pow *= rho;
    }

    let comms = oracle.commit();
    transcript.extend(serialize_comms(&comms));
    eprintln!("SUMCHECK_PROVER_DEBUG: transcript.len()={} after serialize_comms", transcript.len());

    for round in 0..ell {
        // Frame round in transcript and challenger for domain separation
        transcript.extend(format!("sumcheck_round_{}", round).as_bytes());
        eprintln!("SUMCHECK_PROVER_DEBUG: transcript.len()={} after round {} info", transcript.len(), round);
        challenger.observe_bytes("round_label", format!("neo_sumcheck_round_{}", round).as_bytes());
        let remaining = ell - round - 1;
        let mut uni_polys = vec![];

        for (poly_idx, poly) in polys.iter().enumerate() {
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
                
                // DEBUG: Show what evaluations we're getting for zero polynomials
                if round == 0 && poly_idx == 0 { // Focus on first polynomial in first round
                    eprintln!("SUMCHECK_PROVER_DEBUG: poly[{}] eval at x={:?}, point_full={:?} -> sum={:?}", poly_idx, x, point_full, evals[0]);
                }
            }

            let uni = Polynomial::interpolate(&points, &uni_evals);
            
            // DEBUG: Show interpolated univariate for zero polynomials  
            if round == 0 && poly_idx == 0 {
                eprintln!("SUMCHECK_PROVER_DEBUG: poly[{}] uni_evals = {:?}", poly_idx, uni_evals);
                eprintln!("SUMCHECK_PROVER_DEBUG: poly[{}] interpolated uni coeffs = {:?}", poly_idx, uni.coeffs());
            }
            
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
        
        // DEBUG: Show the individual univariates before batching
        eprintln!("SUMCHECK_PROVER_DEBUG: Round {} univariate construction:", round);
        for (i, uni) in uni_polys.iter().enumerate() {
            eprintln!("SUMCHECK_PROVER_DEBUG: uni[{}] coeffs = {:?}", i, uni.coeffs());
            eprintln!("SUMCHECK_PROVER_DEBUG: uni[{}] eval(0) = {:?}, eval(1) = {:?}", i, uni.eval(ExtF::ZERO), uni.eval(ExtF::ONE));
        }
        eprintln!("SUMCHECK_PROVER_DEBUG: blind_factor coeffs = {:?}", blind_factor.coeffs());
        eprintln!("SUMCHECK_PROVER_DEBUG: blind_factor eval(0) = {:?}, eval(1) = {:?}", blind_factor.eval(ExtF::ZERO), blind_factor.eval(ExtF::ONE));
        
        let batched_uni = batch_unis(&uni_polys_with_blind, rho);
        
        eprintln!("SUMCHECK_PROVER_DEBUG: batched_uni coeffs = {:?}", batched_uni.coeffs());
        eprintln!("SUMCHECK_PROVER_DEBUG: batched_uni eval(0) = {:?}, eval(1) = {:?}", batched_uni.eval(ExtF::ZERO), batched_uni.eval(ExtF::ONE));

        if batched_uni.eval(ExtF::ZERO) + batched_uni.eval(ExtF::ONE) != current_batched {
            return Err(SumCheckError::InvalidSum(round));
        }

        // CRITICAL FIX: Use serialize_uni to include degree prefix (matches verifier expectation)
        transcript.extend(serialize_uni(&batched_uni));
        eprintln!("SUMCHECK_PROVER_DEBUG: Round {} - transcript.len()={} after serialize_uni", round, transcript.len());
        challenger.observe_bytes("blinded_uni", &serialize_uni(&batched_uni));

        let challenge = fiat_shamir_challenge(transcript); // Direct FS to match verifier
        challenges.push(challenge);

        let num_polys = polys.len();
        let mut blind_weight = ExtF::ONE;
        for _ in 0..num_polys {
            blind_weight *= rho;
        }
        let blind_eval = blind_factor.eval(challenge) * blind_weight;

        // CRITICAL DEBUG: This is where the zero polynomial issue occurs
        let batched_eval = batched_uni.eval(challenge);
        eprintln!("SUMCHECK_PROVER_DEBUG: Round {}", round);
        eprintln!("SUMCHECK_PROVER_DEBUG: challenge = {:?}", challenge);
        eprintln!("SUMCHECK_PROVER_DEBUG: batched_uni.eval(challenge) = {:?}", batched_eval);
        eprintln!("SUMCHECK_PROVER_DEBUG: blind_eval = {:?}", blind_eval);
        eprintln!("SUMCHECK_PROVER_DEBUG: current_batched (before) = {:?}", current_batched);
        
        // For a zero polynomial Q, we expect:
        // - batched_uni should evaluate to 0 at challenge (since Q is identically 0)
        // - blind_eval should be exactly the blinding factor evaluation
        // - So: batched_eval - blind_eval should be 0
        // But we're seeing batched_eval != blind_eval for zero polynomials!
        
        current_batched = batched_eval - blind_eval;
        eprintln!("SUMCHECK_PROVER_DEBUG: current_batched (after) = {:?}", current_batched);
        
        // This should be 0 for zero polynomials, but it's not!
        if current_batched != ExtF::ZERO {
            eprintln!("SUMCHECK_PROVER_DEBUG: ⚠️  PROBLEM: Zero polynomial producing non-zero current!");
            eprintln!("SUMCHECK_PROVER_DEBUG: batched_uni coeffs = {:?}", batched_uni.coeffs());
            eprintln!("SUMCHECK_PROVER_DEBUG: blind_factor coeffs = {:?}", blind_factor.coeffs());
        }
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
    pre_transcript: &[u8],  // NEW: Pass pre-sumcheck transcript for blind recomputation
) -> Option<(Vec<ExtF>, Vec<ExtF>)> {
    if claims.is_empty() || msgs.is_empty() {
        return Some((vec![], vec![]));
    }
    
    // CRITICAL FIX: Add rho domain separator BEFORE deriving challenge
    transcript.extend(b"sumcheck_rho");
    let rho = fiat_shamir_challenge(transcript);
    eprintln!("VERIFIER_DEBUG: TRANSCRIPT_TRACE: Initial transcript.len()={}, using direct fiat_shamir rho={:?}", transcript.len(), rho);
    
    let mut rho_pow = ExtF::ONE;
    let mut current = ExtF::ZERO;
    for &c in claims {
        current += rho_pow * c;
        rho_pow *= rho;
    }
    
    transcript.extend(serialize_comms(comms)); // Now matches prover
    eprintln!("VERIFIER_DEBUG: TRANSCRIPT_TRACE: Extended transcript with serialize_comms, new len={}", transcript.len());
    
    let mut r = Vec::new();
    
    for (round, (uni, blind_eval)) in msgs.iter().enumerate() {
        transcript.extend(format!("sumcheck_round_{}", round).as_bytes());
        eprintln!("VERIFIER_DEBUG: TRANSCRIPT_TRACE: Round {} - extended transcript with round info", round);
        if uni.eval(ExtF::ZERO) + uni.eval(ExtF::ONE) != current {
            eprintln!("VERIFIER_DEBUG: ❌ Round {} FAILED sum check: {} != {}", round, uni.eval(ExtF::ZERO) + uni.eval(ExtF::ONE), current);
            return None;
        }
        eprintln!("VERIFIER_DEBUG: ✅ Round {} passed sum check", round);
        
        // CRITICAL FIX: Extend transcript with received uni to match prover
        transcript.extend(serialize_uni(uni)); // Now matches prover's extension
        eprintln!("VERIFIER_DEBUG: TRANSCRIPT_TRACE: Round {} - extended transcript with serialize_uni", round);
        let challenge = fiat_shamir_challenge(transcript); // Direct FS to match prover
        eprintln!("VERIFIER_DEBUG: Round {} - using direct fiat_shamir challenge={:?}", round, challenge);
        eprintln!("VERIFIER_DEBUG: Round {} - challenge={:?}", round, challenge);
        eprintln!("VERIFIER_DEBUG: Round {} - uni.eval(challenge)={:?}, blind_eval={:?}", round, uni.eval(challenge), *blind_eval);
        current = uni.eval(challenge) - *blind_eval;
        eprintln!("VERIFIER_DEBUG: Round {} - updated current={:?}", round, current);
        let blind_bytes: Vec<u8> = serialize_ext(*blind_eval);
        transcript.extend(&blind_bytes); // Now matches prover
        eprintln!("VERIFIER_DEBUG: TRANSCRIPT_TRACE: Round {} - extended transcript with blind_bytes", round);
        r.push(challenge);
    }

    eprintln!("VERIFIER_DEBUG: About to open at point r={:?}", r);
    let (evals, proofs) = oracle.open_at_point(&r);
    if !oracle.verify_openings(comms, &r, &evals, &proofs) {
        return None;
    }

    eprintln!("VERIFIER_DEBUG: Opened evals (raw from oracle): {:?}", evals);
    eprintln!("VERIFIER_DEBUG: Expected current from sumcheck: {:?}", current);
    
    // CRITICAL FIX: Only subtract blinds if we have commitments (FRI oracle)
    // FnOracle has no commitments and adds no blinds, so skip blind subtraction
    let mut evals = evals;
    if !comms.is_empty() {
        // FRI oracle case - subtract recomputed blinds
        eprintln!("VERIFIER_DEBUG: FRI oracle detected (comms.len()={}), subtracting blinds", comms.len());
        
        // Use pre-sumcheck transcript state for blind recomputation
        let mut blind_trans = pre_transcript.to_vec();
        blind_trans.extend(b"fri_blind_seed");
        eprintln!("VERIFIER_DEBUG: Using pre_transcript.len()={} for blind computation", pre_transcript.len());
        let hash_result = crate::fiat_shamir_challenge_base(&blind_trans);
        let mut seed = [0u8; 32];
        seed[0..8].copy_from_slice(&hash_result.as_canonical_u64().to_le_bytes());
        for i in 8..32 {
            seed[i] = ((hash_result.as_canonical_u64() >> (i % 8)) ^ (i as u64)) as u8;
        }
        let mut rng = rand_chacha::ChaCha20Rng::from_seed(seed);
        let mut blinds = vec![ExtF::ZERO; evals.len()];
        for blind in &mut blinds {
            *blind = crate::oracle::FriOracle::sample_discrete_gaussian(&mut rng, 3.2);
        }
        eprintln!("VERIFIER_DEBUG: Recomputed blinds: {:?}", blinds);
        
        // Check norm of blinds (the noise), not the unblinded evaluations
        if blinds.iter().any(|b| b.abs_norm() > MAX_BLIND_NORM) {
            eprintln!("VERIFIER_DEBUG: ❌ HIGH BLIND NORM DETECTED - blind norms: {:?}", blinds.iter().map(|b| b.abs_norm()).collect::<Vec<_>>());
            return None;
        }

        // Subtract FRI blinds to get unblinded evaluations
        for (i, e) in evals.iter_mut().enumerate() {
            *e -= blinds[i];
        }
        eprintln!("VERIFIER_DEBUG: Unblinded evals: {:?}", evals);
    } else {
        // FnOracle case - no blinds to subtract
        eprintln!("VERIFIER_DEBUG: FnOracle detected (no commitments), using evals directly");
    }

    // Debug the batching computation step by step
    let mut rho_pow = ExtF::ONE;
    let mut final_batched = ExtF::ZERO;
    eprintln!("VERIFIER_DEBUG: Starting final batching with rho={:?}", rho);
    for (i, &e) in evals.iter().enumerate() {
        let contribution = rho_pow * e;
        final_batched += contribution;
        eprintln!("VERIFIER_DEBUG: eval[{}]={:?}, rho^{}={:?}, contribution={:?}, running_total={:?}", 
                 i, e, i, rho_pow, contribution, final_batched);
        rho_pow *= rho;
    }

    eprintln!("VERIFIER_DEBUG: FINAL COMPARISON: final_batched={:?}, current={:?}", final_batched, current);
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

    // Note: We don't check norm of final_eval here because it's the actual polynomial evaluation,
    // not a blinding value. The norm check should only apply to blinding noise.
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
