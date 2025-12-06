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
