//! Generic Sum-check engine over extension field K.
//!
//! This module encapsulates the ℓ-round driver and the verifier-side
//! checks. Problem-specific logic (how to build the round evaluations
//! and how to fold state) is supplied via the `RoundOracle` trait.
//!
//! Transcript binding matches the existing Π_CCS code: for each round we
//! absorb all coefficients under the label `neo/ccs/round`.

use crate::error::PiCcsError;
use crate::transcript::FoldTranscript;
use neo_math::K;
use p3_field::{Field, PrimeCharacteristicRing};

/// Trait implemented by problem-specific provers (e.g., generic CCS, R1CS fast-path).
pub trait RoundOracle {
    /// Number of rounds ℓ (equals the arity of the boolean hypercube).
    fn num_rounds(&self) -> usize;
    /// Degree bound d for each univariate round polynomial.
    fn degree_bound(&self) -> usize;

    /// Return evaluations of the current round polynomial at the provided points.
    /// Caller (engine) will Lagrange-interpolate these into coefficients.
    /// Must not mutate the internal folded state.
    fn evals_at(&mut self, sample_xs: &[K]) -> Vec<K>;

    /// Fold all internal vectors with the sampled challenge r_i:
    /// S[k] <- (1-r_i)*S[2k] + r_i*S[2k+1].
    fn fold(&mut self, r_i: K);
}

/// Output of the prover-side engine.
pub struct SumcheckOutput {
    pub rounds: Vec<Vec<K>>,   // coefficients per round, low→high
    pub challenges: Vec<K>,    // sampled r_0..r_{ℓ-1}
    pub final_sum: K,          // p_ℓ(r_ℓ) (the reduced claim Q(r))
}

/// Evaluate polynomial at x using Horner (coeffs low→high).
#[inline]
fn poly_eval_k(coeffs: &[K], x: K) -> K {
    let mut acc = K::ZERO;
    for &c in coeffs.iter().rev() { acc = acc * x + c; }
    acc
}

#[inline]
fn poly_mul_k(a: &[K], b: &[K]) -> Vec<K> {
    let mut out = vec![K::ZERO; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() { out[i + j] += ai * bj; }
    }
    out
}

/// Lagrange interpolation at distinct points xs with values ys (returns coeffs low→high).
fn lagrange_interpolate_k(xs: &[K], ys: &[K]) -> Vec<K> {
    debug_assert_eq!(xs.len(), ys.len());
    let m = xs.len();
    let mut coeffs = vec![K::ZERO; m];
    for i in 0..m {
        let (xi, yi) = (xs[i], ys[i]);
        let mut denom = K::ONE;
        let mut numer = vec![K::ONE]; // polynomial 1
        for j in 0..m {
            if i == j { continue; }
            denom *= xi - xs[j];
            numer = poly_mul_k(&numer, &[-xs[j], K::ONE]); // (X - x_j)
        }
        let scale = yi * denom.inverse();
        for k in 0..numer.len() { coeffs[k] += numer[k] * scale; }
    }
    coeffs
}

/// Prover: run ℓ rounds of sum-check driven by `oracle`.
/// - `initial_sum` is the public sum claim (0 in Π_CCS use).
/// - `sample_xs` are the interpolation points (e.g., 0..=d).
pub fn run_sumcheck(
    tr: &mut FoldTranscript,
    oracle: &mut dyn RoundOracle,
    initial_sum: K,
    sample_xs: &[K],
) -> Result<SumcheckOutput, PiCcsError> {
    let ell = oracle.num_rounds();
    let d_sc = oracle.degree_bound();

    let mut rounds = Vec::with_capacity(ell);
    let mut challenges = Vec::with_capacity(ell);
    let mut running_sum = initial_sum;

    for i in 0..ell {
        let sample_ys = oracle.evals_at(sample_xs);
        if sample_ys.len() != sample_xs.len() {
            return Err(PiCcsError::SumcheckError(format!(
                "round {i}: evals_at returned {} values for {} points",
                sample_ys.len(), sample_xs.len()
            )));
        }

        let coeffs = lagrange_interpolate_k(sample_xs, &sample_ys);
        if coeffs.len() > d_sc + 1 {
            return Err(PiCcsError::SumcheckError(format!(
                "round {i}: degree {} exceeds bound {d_sc}", coeffs.len() - 1
            )));
        }

        // Standard consistency check: p(0) + p(1) == running_sum
        let p0 = poly_eval_k(&coeffs, K::ZERO);
        let p1 = poly_eval_k(&coeffs, K::ONE);
        if p0 + p1 != running_sum {
            return Err(PiCcsError::SumcheckError(format!(
                "round {i}: p(0)+p(1) mismatch"
            )));
        }

        // Bind polynomial to transcript and sample r_i
        tr.absorb_ext_as_base_fields_k(b"neo/ccs/round", coeffs[0]);
        for c in coeffs.iter().skip(1) {
            tr.absorb_ext_as_base_fields_k(b"neo/ccs/round", *c);
        }
        let r_i = tr.challenge_k();

        running_sum = poly_eval_k(&coeffs, r_i);
        oracle.fold(r_i);

        challenges.push(r_i);
        rounds.push(coeffs);
    }

    Ok(SumcheckOutput { rounds, challenges, final_sum: running_sum })
}

/// Verifier: check the provided round polynomials and derive the same r-vector.
/// Returns (r, final_sum, ok).
pub fn verify_sumcheck_rounds(
    tr: &mut FoldTranscript,
    d_sc: usize,
    initial_sum: K,
    rounds: &[Vec<K>],
) -> (Vec<K>, K, bool) {
    let mut running_sum = initial_sum;
    let mut r = Vec::with_capacity(rounds.len());

    for coeffs in rounds.iter() {
        if coeffs.is_empty() || coeffs.len() > d_sc + 1 {
            return (Vec::new(), K::ZERO, false);
        }
        let p0 = poly_eval_k(coeffs, K::ZERO);
        let p1 = poly_eval_k(coeffs, K::ONE);
        if p0 + p1 != running_sum {
            return (Vec::new(), K::ZERO, false);
        }
        tr.absorb_ext_as_base_fields_k(b"neo/ccs/round", coeffs[0]);
        for c in coeffs.iter().skip(1) {
            tr.absorb_ext_as_base_fields_k(b"neo/ccs/round", *c);
        }
        let r_i = tr.challenge_k();
        running_sum = poly_eval_k(coeffs, r_i);
        r.push(r_i);
    }
    (r, running_sum, true)
}
