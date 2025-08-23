use crate::fiat_shamir::{batch_unis, fiat_shamir_challenge};
use crate::fiat_shamir::{fs_absorb_poly, fs_absorb_extf, fs_absorb_u64, fs_challenge_ext};
use crate::fiat_shamir::{fs_challenge_base_labeled, NEO_FS_DOMAIN};
use crate::challenger::NeoChallenger;
use crate::{from_base, ExtF, Polynomial, UnivPoly, F};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rand_distr::{Distribution, StandardNormal};
use rand_chacha::ChaCha20Rng;
use rand_chacha::rand_core::SeedableRng;
use thiserror::Error;
// Unused imports removed for NARK mode

/// Batched sum-check prover for multiple polynomial instances
/// Enhanced with full ZK blinding using Fiat-Shamir derived parameters
const ZK_SIGMA: f64 = 3.2;

// No BLAKE3 here; use canonical helpers via fiat_shamir.rs when we need seeds.

// NARK mode: Knowledge soundness verified through direct polynomial checks
// extract_sumcheck_witness function removed - no longer needed

/// Serialize a univariate polynomial for transcript
pub fn serialize_uni(uni: &Polynomial<ExtF>) -> Vec<u8> {
    let mut bytes = vec![uni.degree() as u8];
    let coeffs = uni.coeffs();
    if coeffs.is_empty() {
        // Serialize zero coeff for zero poly
        bytes.extend_from_slice(&[0u8; 8]); // real = 0
        bytes.extend_from_slice(&[0u8; 8]); // imag = 0
    } else {
        for &c in coeffs {
            let arr = c.to_array();
            bytes.extend_from_slice(&arr[0].as_canonical_u64().to_be_bytes());
            bytes.extend_from_slice(&arr[1].as_canonical_u64().to_be_bytes());
        }
    }
    bytes
}

/// Serialize an ExtF for transcript
pub fn serialize_ext(e: ExtF) -> Vec<u8> {
    let arr = e.to_array();
    let mut bytes = Vec::with_capacity(16);
    bytes.extend_from_slice(&arr[0].as_canonical_u64().to_be_bytes());
    bytes.extend_from_slice(&arr[1].as_canonical_u64().to_be_bytes());
    bytes
}

// NARK mode: simulate_sumcheck_opening function removed - no longer needed

// NARK mode: extractor tests removed - no longer needed

#[derive(Error, Debug)]
pub enum SumCheckError {
    #[error("Invalid sum in round {0}")]
    InvalidSum(usize),
}


#[allow(clippy::type_complexity)]
pub fn batched_sumcheck_prover(
    claims: &[ExtF],
    polys: &[&dyn UnivPoly],
    transcript: &mut Vec<u8>,
) -> Result<Vec<(Polynomial<ExtF>, ExtF)>, SumCheckError> {
    assert_eq!(claims.len(), polys.len());
    if polys.is_empty() {
        return Ok(vec![]);
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

    // Structured, domain-separated challenge for the rho used to batch claims
    let rho = fs_challenge_ext(transcript, b"sumcheck.rho");

    let mut rho_pow = ExtF::ONE;
    let mut current_batched = ExtF::ZERO;
    for &c in claims {
        current_batched += rho_pow * c;
        rho_pow *= rho;
    }

    // NARK mode: No commitments needed - prover sends polynomials directly
    eprintln!("SUMCHECK_PROVER_DEBUG: transcript.len()={} before rounds", transcript.len());

    for round in 0..ell {
        // Bind the round number into the transcript with length-delimited framing
        fs_absorb_u64(transcript, b"sumcheck.round", round as u64);
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

        // Enhanced ZK blinding with Fiat-Shamir derived parameters
        let blind_deg = max_d.saturating_sub(2);

        let mut zk_sigma_transcript = transcript.clone();
        zk_sigma_transcript.extend_from_slice(NEO_FS_DOMAIN);
        zk_sigma_transcript.extend_from_slice(b"|neo.sumcheck.zk_sigma");
        zk_sigma_transcript.extend_from_slice(&round.to_le_bytes());
        let zk_sigma_seed = fs_challenge_base_labeled(&zk_sigma_transcript, "neo.sumcheck.zk_sigma.seed").as_canonical_u64();
        let zk_sigma = ZK_SIGMA + (zk_sigma_seed as f64 / u64::MAX as f64) * 2.0; // Range [3.2, 5.2]

        let mut blind_coeffs: Vec<ExtF> = (0..=blind_deg)
            .map(|_| {
                let sample: f64 =
                    <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut rng)
                        * zk_sigma;
                from_base(F::from_i64(sample.round() as i64))
            })
            .collect();
        let mut blind_poly = Polynomial::new(blind_coeffs.clone());
        let x_poly = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE]);
        let xm1_poly = Polynomial::new(vec![-ExtF::ONE, ExtF::ONE]);
        let mut blind_factor = x_poly.clone() * xm1_poly.clone() * blind_poly.clone();
        
        // TRANSCRIPT FIX: Ensure blind_factor maintains full degree to prevent cursor position mismatches
        let mut attempts = 0;
        while blind_factor.coeffs().is_empty() || blind_factor.coeffs().last() == Some(&ExtF::ZERO) {
            if attempts >= 100 {
                return Err(SumCheckError::InvalidSum(round));
            }
            blind_coeffs = (0..=blind_deg)
                .map(|_| {
                    let sample: f64 =
                        <StandardNormal as Distribution<f64>>::sample(&StandardNormal, &mut rng)
                            * zk_sigma;
                    from_base(F::from_i64(sample.round() as i64))
                })
                .collect();
            blind_poly = Polynomial::new(blind_coeffs.clone());
            blind_factor = x_poly.clone() * xm1_poly.clone() * blind_poly.clone();
            attempts += 1;
        }
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

        // Absorb the univariate with structured framing
        fs_absorb_poly(transcript, b"sumcheck.uni", &batched_uni);
        eprintln!("SUMCHECK_PROVER_DEBUG: Round {} - transcript.len()={} after fs_absorb_poly", round, transcript.len());
        challenger.observe_bytes("blinded_uni", &serialize_uni(&batched_uni));

        // Round challenge with domain separation
        let challenge = fs_challenge_ext(transcript, b"sumcheck.challenge");
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
        // Absorb blind evaluation in structured form
        fs_absorb_extf(transcript, b"sumcheck.blind_eval", blind_eval);
        let blind_bytes: Vec<u8> = serialize_ext(blind_eval);
        challenger.observe_bytes("blind_eval", &blind_bytes);

        msgs.push((batched_uni.clone(), blind_eval));
    }
    Ok(msgs)
}

/// Batched sum-check verifier
pub fn batched_sumcheck_verifier(
    claims: &[ExtF],
    msgs: &[(Polynomial<ExtF>, ExtF)],
    transcript: &mut Vec<u8>,
) -> Option<(Vec<ExtF>, ExtF)> {  // Updated return type
    if claims.is_empty() {
        return Some((vec![], ExtF::ZERO));
    }
    
    if msgs.is_empty() {
        // Trivial case: no rounds, return the combined claim as final_current
        let rho = fs_challenge_ext(transcript, b"sumcheck.rho");
        eprintln!("VERIFIER_DEBUG: TRANSCRIPT_TRACE: Trivial case - transcript.len()={}, using structured FS rho={:?}", transcript.len(), rho);
        
        let mut rho_pow = ExtF::ONE;
        let mut current = ExtF::ZERO;
        for &c in claims {
            current += rho_pow * c;
            rho_pow *= rho;
        }
        eprintln!("VERIFIER_DEBUG: Trivial case - final current (combined claim): {:?}", current);
        return Some((vec![], current));
    }
    
    let rho = fs_challenge_ext(transcript, b"sumcheck.rho");
    eprintln!("VERIFIER_DEBUG: TRANSCRIPT_TRACE: Initial transcript.len()={}, using structured FS rho={:?}", transcript.len(), rho);
    
    let mut rho_pow = ExtF::ONE;
    let mut current = ExtF::ZERO;
    for &c in claims {
        current += rho_pow * c;
        rho_pow *= rho;
    }
    
    // NARK mode: No commitments to serialize
    eprintln!("VERIFIER_DEBUG: NARK mode - no commitments to serialize");
    
    let mut r = Vec::new();
    
    for (round, (uni, blind_eval)) in msgs.iter().enumerate() {
        fs_absorb_u64(transcript, b"sumcheck.round", round as u64);
        eprintln!("VERIFIER_DEBUG: TRANSCRIPT_TRACE: Round {} - absorbed round with structured FS", round);
        if uni.eval(ExtF::ZERO) + uni.eval(ExtF::ONE) != current {
            eprintln!("VERIFIER_DEBUG: ❌ Round {} FAILED sum check: {} != {}", round, uni.eval(ExtF::ZERO) + uni.eval(ExtF::ONE), current);
            return None;
        }
        eprintln!("VERIFIER_DEBUG: ✅ Round {} passed sum check", round);
        
        // Absorb the received uni exactly as the prover absorbed it
        fs_absorb_poly(transcript, b"sumcheck.uni", uni);
        eprintln!("VERIFIER_DEBUG: TRANSCRIPT_TRACE: Round {} - absorbed uni with structured FS", round);
        let challenge = fs_challenge_ext(transcript, b"sumcheck.challenge");
        eprintln!("VERIFIER_DEBUG: Round {} - using structured FS challenge={:?}", round, challenge);
        eprintln!("VERIFIER_DEBUG: Round {} - challenge={:?}", round, challenge);
        eprintln!("VERIFIER_DEBUG: Round {} - uni.eval(challenge)={:?}, blind_eval={:?}", round, uni.eval(challenge), *blind_eval);
        current = uni.eval(challenge) - *blind_eval;
        eprintln!("VERIFIER_DEBUG: Round {} - updated current={:?}", round, current);
        fs_absorb_extf(transcript, b"sumcheck.blind_eval", *blind_eval);
        eprintln!("VERIFIER_DEBUG: TRANSCRIPT_TRACE: Round {} - absorbed blind_eval with structured FS", round);
        r.push(challenge);
    }

    // NARK mode: Return challenges and final current (no zero-check; caller verifies against reconstructed Q(r))
    eprintln!("VERIFIER_DEBUG: Final current (claimed Q(r)): {:?}", current);
    
    Some((r, current))  // Always return if rounds passed; caller checks current == f(ys)
}

/// Optimized multilinear sum-check prover using folding
#[allow(clippy::type_complexity)]
pub fn multilinear_sumcheck_prover(
    evals: &mut Vec<ExtF>,
    claim: ExtF,
    transcript: &mut Vec<u8>,
) -> Result<Vec<(Polynomial<ExtF>, ExtF)>, SumCheckError> {
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

    // NARK mode: No commitments needed

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

    Ok(msgs)
}

/// Multilinear sum-check verifier
pub fn multilinear_sumcheck_verifier(
    claim: ExtF,
    msgs: &[(Polynomial<ExtF>, ExtF)],
    transcript: &mut Vec<u8>,
) -> Option<Vec<ExtF>> {
    let mut current = claim;
    // NARK mode: No commitments to serialize

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

    // NARK mode: Direct polynomial check - verify that current reduces to zero
    if current == ExtF::ZERO {
        Some(r)
    } else {
        None
    }
}

/// Batched multilinear sum-check prover
#[allow(clippy::type_complexity)]
pub fn batched_multilinear_sumcheck_prover(
    claims: &[ExtF],
    evals: &mut [Vec<ExtF>],
    transcript: &mut Vec<u8>,
) -> Result<Vec<(Polynomial<ExtF>, ExtF)>, SumCheckError> {
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

    // NARK mode: No commitments needed

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
    Ok(msgs)
}

/// Batched multilinear sum-check verifier
pub fn batched_multilinear_sumcheck_verifier(
    claims: &[ExtF],
    msgs: &[(Polynomial<ExtF>, ExtF)],
    transcript: &mut Vec<u8>,
) -> Option<Vec<ExtF>> {
    transcript.extend(b"batched_multilinear_rho");
    let rho = fiat_shamir_challenge(transcript);

    let mut rho_pow = ExtF::ONE;
    let mut current = ExtF::ZERO;
    for &c in claims {
        current += rho_pow * c;
        rho_pow *= rho;
    }

    // NARK mode: No commitments to serialize

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

    // NARK mode: Direct polynomial check - verify that current reduces to zero
    if current == ExtF::ZERO {
        Some(r)
    } else {
        None
    }
}


