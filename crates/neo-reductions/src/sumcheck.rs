//! Sumcheck protocol interface

use neo_math::{from_complex, Fq, KExtensions, K};
use neo_transcript::Transcript;
use p3_field::PrimeCharacteristicRing;
use thiserror::Error;

/// Format K value compactly for logging
#[cfg(feature = "debug-logs")]
fn format_k(k: &K) -> String {
    use p3_field::PrimeField64;
    let coeffs = k.as_coeffs();
    format!("K[{}, {}]", coeffs[0].as_canonical_u64(), coeffs[1].as_canonical_u64())
}

/// Trait for round oracles in the sumcheck protocol
pub trait RoundOracle {
    /// Evaluate the oracle at multiple points for the current round
    fn evals_at(&mut self, points: &[K]) -> Vec<K>;

    /// Get the number of rounds in the sumcheck protocol
    fn num_rounds(&self) -> usize;

    /// Get the degree bound for each round
    fn degree_bound(&self) -> usize;

    /// Fold the oracle with the given challenge
    fn fold(&mut self, r: K);

    /// Alias for fold - bind to a specific value and advance to the next round
    fn bind(&mut self, r: K) {
        self.fold(r);
    }
}

/// Errors that can occur while running the prover side of the sumcheck.
#[derive(Debug, Error)]
pub enum SumcheckError {
    #[error("round {round} invariant failed: expected p(0)+p(1)={expected:?}, got {actual:?}")]
    Invariant {
        round: usize,
        expected: K,
        actual: K,
    },
}

/// Evaluate a polynomial (given as coefficients) at a point
pub fn poly_eval_k(coeffs: &[K], x: K) -> K {
    if coeffs.is_empty() {
        return K::ZERO;
    }
    // Horner's method: p(x) = c_0 + x*(c_1 + x*(c_2 + ...))
    let mut result = coeffs[coeffs.len() - 1];
    for &c in coeffs.iter().rev().skip(1) {
        result = result * x + c;
    }
    result
}

/// Evaluate a polynomial at a base-field point `x ∈ Fq ⊂ K` (imag=0).
///
/// This is a hot path for sumcheck oracles which evaluate at `x = 0..deg` and can use the
/// specialized `KExtensions::scale_base` instead of a full extension-field multiplication.
#[inline]
pub fn poly_eval_k_base(coeffs: &[K], x: Fq) -> K {
    if coeffs.is_empty() {
        return K::ZERO;
    }
    let mut result = coeffs[coeffs.len() - 1];
    for &c in coeffs.iter().rev().skip(1) {
        result = result.scale_base(x) + c;
    }
    result
}

/// Lagrange-interpolate a univariate polynomial from evaluations.
///
/// Returns coefficients in low→high order so that `poly_eval_k(&coeffs, x)`
/// reproduces the provided `(xs, ys)` pairs.
pub fn interpolate_from_evals(xs: &[K], ys: &[K]) -> Vec<K> {
    assert_eq!(xs.len(), ys.len(), "xs/ys length mismatch");
    let n = xs.len();
    let mut coeffs = vec![K::ZERO; n];

    for i in 0..n {
        // Build numerator of ℓ_i(x) = Π_{j≠i} (x - x_j)
        let mut numer = vec![K::ZERO; n];
        numer[0] = K::ONE;
        let mut cur_deg = 0usize;
        for j in 0..n {
            if i == j {
                continue;
            }
            let xj = xs[j];
            let mut next = vec![K::ZERO; n];
            for d in 0..=cur_deg {
                next[d + 1] += numer[d];
                next[d] += -xj * numer[d];
            }
            numer = next;
            cur_deg += 1;
        }

        // Denominator of ℓ_i(x) = Π_{j≠i} (x_i - x_j)
        let mut denom = K::ONE;
        for j in 0..n {
            if i != j {
                denom *= xs[i] - xs[j];
            }
        }
        let scale = ys[i] * denom.inv();
        for d in 0..=cur_deg {
            coeffs[d] += scale * numer[d];
        }
    }

    coeffs
}

/// Run the sumcheck prover against a [`RoundOracle`].
///
/// This mirrors the verifier in `verify_sumcheck_rounds`, interpolating the
/// univariate sent each round and sampling challenges from the transcript.
pub fn run_sumcheck_prover<O: RoundOracle, Tr: Transcript>(
    tr: &mut Tr,
    oracle: &mut O,
    initial_sum: K,
) -> Result<(Vec<Vec<K>>, Vec<K>), SumcheckError> {
    let total_rounds = oracle.num_rounds();
    let mut running_sum = initial_sum;
    let mut rounds = Vec::with_capacity(total_rounds);
    let mut challenges = Vec::with_capacity(total_rounds);

    #[cfg(feature = "debug-logs")]
    eprintln!(
        "PROVER: Starting sumcheck with {} rounds, initial_sum={}, degree_bound={}",
        total_rounds,
        format_k(&initial_sum),
        oracle.degree_bound()
    );

    for round_idx in 0..total_rounds {
        let deg = oracle.degree_bound();
        let xs: Vec<K> = (0..=deg).map(|t| K::from(Fq::from_u64(t as u64))).collect();
        let ys = oracle.evals_at(&xs);

        #[cfg(feature = "debug-logs")]
        if round_idx < 3 {
            eprintln!("PROVER Round {}:", round_idx);
            eprintln!("  degree_bound={}", deg);
            eprintln!(
                "  evals: [{}]",
                ys.iter()
                    .take(5)
                    .map(format_k)
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            eprintln!(
                "  p(0)={}, p(1)={}, sum={}",
                format_k(&ys[0]),
                format_k(&ys[1]),
                format_k(&(ys[0] + ys[1]))
            );
            eprintln!("  expected running_sum={}", format_k(&running_sum));
        }

        let sum_at_01 = ys[0] + ys[1];
        if sum_at_01 != running_sum {
            #[cfg(feature = "debug-logs")]
            eprintln!(
                "PROVER ERROR: round {} invariant failed!\n  expected={}\n  actual={}",
                round_idx,
                format_k(&running_sum),
                format_k(&sum_at_01)
            );
            return Err(SumcheckError::Invariant {
                round: round_idx,
                expected: running_sum,
                actual: sum_at_01,
            });
        }

        // Interpolate and normalize to low→high coefficient order.
        let coeffs = interpolate_from_evals(&xs, &ys);
        debug_assert!(xs
            .iter()
            .zip(ys.iter())
            .all(|(&x, &y)| poly_eval_k(&coeffs, x) == y));

        // Commit coefficients to the transcript
        for &coeff in coeffs.iter() {
            tr.append_fields(b"sumcheck/round/coeff", &coeff.as_coeffs());
        }

        // Sample challenge as an extension-field element
        let c = tr.challenge_field(b"sumcheck/challenge/0");
        let d = tr.challenge_field(b"sumcheck/challenge/1");
        let challenge = from_complex(c, d);
        challenges.push(challenge);

        // Advance state
        running_sum = poly_eval_k(&coeffs, challenge);
        oracle.fold(challenge);
        rounds.push(coeffs);
    }

    Ok((rounds, challenges))
}

/// Verify sumcheck rounds against a transcript
///
/// Returns (challenges, running_sum, is_valid)
pub fn verify_sumcheck_rounds<Tr: Transcript>(
    tr: &mut Tr,
    degree_bound: usize,
    initial_sum: K,
    rounds: &[Vec<K>],
) -> (Vec<K>, K, bool) {
    let mut challenges = Vec::with_capacity(rounds.len());
    let mut running_sum = initial_sum;

    #[cfg(feature = "debug-logs")]
    eprintln!(
        "VERIFIER: Starting sumcheck with initial_sum={}",
        format_k(&initial_sum)
    );

    for (i, round_poly) in rounds.iter().enumerate() {
        // Check degree bound
        if round_poly.len() > degree_bound + 1 {
            eprintln!(
                "Round {} failed: degree check. len={}, degree_bound={}",
                i,
                round_poly.len(),
                degree_bound
            );
            return (challenges, running_sum, false);
        }

        // Verify that round_poly(0) + round_poly(1) = running_sum
        let eval_0 = poly_eval_k(round_poly, K::ZERO);
        let eval_1 = poly_eval_k(round_poly, K::ONE);

        #[cfg(feature = "debug-logs")]
        if i <= 1 {
            eprintln!("VERIFIER Round {}:", i);
            eprintln!("  Received {} coefficients", round_poly.len());
            if i == 0 {
                eprintln!(
                    "  coeffs=[{}]",
                    round_poly
                        .iter()
                        .map(format_k)
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
            eprintln!(
                "  eval_0={}, eval_1={}, sum={}",
                format_k(&eval_0),
                format_k(&eval_1),
                format_k(&(eval_0 + eval_1))
            );
            eprintln!("  expected running_sum={}", format_k(&running_sum));
        }

        if eval_0 + eval_1 != running_sum {
            eprintln!(
                "Round {} failed: invariant check. eval_0={:?}, eval_1={:?}, sum={:?}, running_sum={:?}",
                i,
                eval_0,
                eval_1,
                eval_0 + eval_1,
                running_sum
            );
            return (challenges, running_sum, false);
        }

        // Append round polynomial to transcript
        for &coeff in round_poly.iter() {
            tr.append_fields(b"sumcheck/round/coeff", &coeff.as_coeffs());
        }

        // Sample challenge for this round: extension field element
        // Sample 2 base field elements and combine them
        let c = tr.challenge_field(b"sumcheck/challenge/0");
        let d = tr.challenge_field(b"sumcheck/challenge/1");
        let challenge = neo_math::from_complex(c, d);
        challenges.push(challenge);

        // Update running sum: running_sum := round_poly(challenge)
        running_sum = poly_eval_k(round_poly, challenge);

        #[cfg(feature = "debug-logs")]
        if i <= 1 {
            eprintln!("  challenge={}", format_k(&challenge));
            eprintln!("  new_running_sum={}", format_k(&running_sum));
        }
    }

    (challenges, running_sum, true)
}

// ============================================================================
// Batched Sum-Check (Shared Challenges)
// ============================================================================
//
// For Route A soundness: run multiple sum-checks with shared transcript-derived
// challenges. This enables r-alignment across CCS and Twist/Shout while maintaining
// sum-check soundness (challenges unpredictable to prover).

/// A claim participating in batched sum-check.
///
/// Each claim has an oracle, degree bound, and claimed sum. During batched sum-check,
/// all oracles receive the same transcript-derived challenges.
pub struct BatchedClaim<'a> {
    /// The oracle for this claim
    pub oracle: &'a mut dyn RoundOracle,
    /// Claimed sum for this protocol
    pub claimed_sum: K,
    /// Label for this claim (for transcript domain separation)
    pub label: &'static [u8],
}

/// Result of batched sum-check for a single claim
#[derive(Debug, Clone)]
pub struct BatchedClaimResult {
    /// Round polynomials (coefficients per round)
    pub round_polys: Vec<Vec<K>>,
    /// Final oracle value after all rounds
    pub final_value: K,
}

/// Run batched sum-check prover with shared transcript-derived challenges.
///
/// This is the SOUND version of sum-check that fixes the fixed-challenge issue.
/// All oracles in `claims` receive the same challenges derived from the transcript,
/// ensuring:
/// 1. Challenges are unpredictable when the prover produces round polynomials
/// 2. The resulting `r` is shared across all protocols (enables r-alignment for RLC)
///
/// # Arguments
/// - `tr`: Transcript for Fiat-Shamir
/// - `claims`: Vector of claims participating in the batched sum-check
///
/// # Returns
/// - Shared challenges (the `r` vector for ME claims)
/// - Per-claim results (round polynomials and final values)
pub fn run_batched_sumcheck_prover<Tr: Transcript>(
    tr: &mut Tr,
    claims: &mut [BatchedClaim<'_>],
) -> Result<(Vec<K>, Vec<BatchedClaimResult>), SumcheckError> {
    if claims.is_empty() {
        return Ok((vec![], vec![]));
    }

    // All oracles must have the same number of rounds (they operate over the same domain)
    let num_rounds = claims[0].oracle.num_rounds();
    for (i, claim) in claims.iter().enumerate() {
        if claim.oracle.num_rounds() != num_rounds {
            panic!(
                "Batched sum-check: claim {} has {} rounds, expected {} (all must match)",
                i,
                claim.oracle.num_rounds(),
                num_rounds
            );
        }
    }

    let mut shared_challenges = Vec::with_capacity(num_rounds);
    let mut per_claim_results: Vec<BatchedClaimResult> = claims
        .iter()
        .map(|_| BatchedClaimResult {
            round_polys: Vec::with_capacity(num_rounds),
            final_value: K::ZERO,
        })
        .collect();

    // Track running sums per claim
    let mut running_sums: Vec<K> = claims.iter().map(|c| c.claimed_sum).collect();

    #[cfg(feature = "debug-logs")]
    eprintln!(
        "BATCHED SUMCHECK PROVER: {} claims, {} rounds",
        claims.len(),
        num_rounds
    );

    for round_idx in 0..num_rounds {
        tr.append_message(b"batched/round_idx", &(round_idx as u64).to_le_bytes());

        // 1. Collect round polynomials from ALL claims
        let mut all_round_polys: Vec<Vec<K>> = Vec::with_capacity(claims.len());

        for (claim_idx, claim) in claims.iter_mut().enumerate() {
            let deg = claim.oracle.degree_bound();
            let xs: Vec<K> = (0..=deg).map(|t| K::from(Fq::from_u64(t as u64))).collect();
            let ys = claim.oracle.evals_at(&xs);

            // Check invariant: p(0) + p(1) = running_sum
            let sum_at_01 = ys[0] + ys[1];
            if sum_at_01 != running_sums[claim_idx] {
                return Err(SumcheckError::Invariant {
                    round: round_idx,
                    expected: running_sums[claim_idx],
                    actual: sum_at_01,
                });
            }

            // Interpolate to get polynomial coefficients
            let coeffs = interpolate_from_evals(&xs, &ys);
            all_round_polys.push(coeffs);
        }

        // 2. Append ALL round polynomials to transcript (domain separated by claim)
        for (claim_idx, (claim, coeffs)) in claims.iter().zip(all_round_polys.iter()).enumerate() {
            tr.append_message(b"batched/claim_label", claim.label);
            tr.append_message(b"batched/claim_idx", &(claim_idx as u64).to_le_bytes());
            for &coeff in coeffs.iter() {
                tr.append_fields(b"batched/round/coeff", &coeff.as_coeffs());
            }
        }

        // 3. Derive ONE shared challenge from transcript
        let c = tr.challenge_field(b"batched/challenge/0");
        let d = tr.challenge_field(b"batched/challenge/1");
        let shared_challenge = from_complex(c, d);
        shared_challenges.push(shared_challenge);

        #[cfg(feature = "debug-logs")]
        if round_idx < 3 {
            eprintln!(
                "BATCHED Round {}: shared_challenge={}",
                round_idx,
                format_k(&shared_challenge)
            );
        }

        // 4. Fold ALL oracles with the shared challenge and update running sums
        for (claim_idx, (claim, coeffs)) in claims.iter_mut().zip(all_round_polys.iter()).enumerate() {
            running_sums[claim_idx] = poly_eval_k(coeffs, shared_challenge);
            claim.oracle.fold(shared_challenge);
            per_claim_results[claim_idx]
                .round_polys
                .push(coeffs.clone());
        }
    }

    // Store final values
    for (claim_idx, result) in per_claim_results.iter_mut().enumerate() {
        result.final_value = running_sums[claim_idx];
    }

    Ok((shared_challenges, per_claim_results))
}

/// Verify batched sum-check rounds with shared challenges.
///
/// # Arguments
/// - `tr`: Transcript (must be in same state as prover's)
/// - `per_claim_rounds`: Round polynomials per claim
/// - `per_claim_sums`: Claimed sums per claim
/// - `per_claim_labels`: Labels per claim (for transcript domain separation)
/// - `degree_bounds`: Degree bounds per claim
///
/// # Returns
/// - Shared challenges
/// - Final values per claim
/// - Whether verification passed
pub fn verify_batched_sumcheck_rounds<Tr: Transcript>(
    tr: &mut Tr,
    per_claim_rounds: &[Vec<Vec<K>>],
    per_claim_sums: &[K],
    per_claim_labels: &[&[u8]],
    degree_bounds: &[usize],
) -> (Vec<K>, Vec<K>, bool) {
    let num_claims = per_claim_rounds.len();
    if num_claims == 0 {
        return (vec![], vec![], true);
    }

    // All claims must have the same number of rounds
    let num_rounds = per_claim_rounds[0].len();
    for (i, rounds) in per_claim_rounds.iter().enumerate() {
        if rounds.len() != num_rounds {
            eprintln!(
                "Batched verify: claim {} has {} rounds, expected {}",
                i,
                rounds.len(),
                num_rounds
            );
            return (vec![], vec![], false);
        }
    }

    let mut shared_challenges = Vec::with_capacity(num_rounds);
    let mut running_sums = per_claim_sums.to_vec();

    for round_idx in 0..num_rounds {
        tr.append_message(b"batched/round_idx", &(round_idx as u64).to_le_bytes());

        // Verify each claim's round polynomial
        for (claim_idx, rounds) in per_claim_rounds.iter().enumerate() {
            let round_poly = &rounds[round_idx];
            let degree_bound = degree_bounds[claim_idx];

            // Check degree bound
            if round_poly.len() > degree_bound + 1 {
                eprintln!(
                    "Claim {} round {} failed: degree check. len={}, degree_bound={}",
                    claim_idx,
                    round_idx,
                    round_poly.len(),
                    degree_bound
                );
                return (shared_challenges, running_sums, false);
            }

            // Check invariant: p(0) + p(1) = running_sum
            let eval_0 = poly_eval_k(round_poly, K::ZERO);
            let eval_1 = poly_eval_k(round_poly, K::ONE);

            if eval_0 + eval_1 != running_sums[claim_idx] {
                eprintln!(
                    "Claim {} round {} failed: invariant. sum={:?}, running_sum={:?}",
                    claim_idx,
                    round_idx,
                    eval_0 + eval_1,
                    running_sums[claim_idx]
                );
                return (shared_challenges, running_sums, false);
            }
        }

        // Append ALL round polynomials to transcript (same order as prover)
        for (claim_idx, rounds) in per_claim_rounds.iter().enumerate() {
            let round_poly = &rounds[round_idx];
            tr.append_message(b"batched/claim_label", per_claim_labels[claim_idx]);
            tr.append_message(b"batched/claim_idx", &(claim_idx as u64).to_le_bytes());
            for &coeff in round_poly.iter() {
                tr.append_fields(b"batched/round/coeff", &coeff.as_coeffs());
            }
        }

        // Derive shared challenge (must match prover)
        let c = tr.challenge_field(b"batched/challenge/0");
        let d = tr.challenge_field(b"batched/challenge/1");
        let shared_challenge = from_complex(c, d);
        shared_challenges.push(shared_challenge);

        // Update running sums
        for (claim_idx, rounds) in per_claim_rounds.iter().enumerate() {
            running_sums[claim_idx] = poly_eval_k(&rounds[round_idx], shared_challenge);
        }
    }

    (shared_challenges, running_sums, true)
}
