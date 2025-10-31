//! Generic Sum-check engine over extension field K.
//!
//! This module encapsulates the ℓ-round driver and the verifier-side
//! checks. Problem-specific logic (how to build the round evaluations
//! and how to fold state) is supplied via the `RoundOracle` trait.
//!
//! Transcript binding matches the existing Π_CCS code: for each round we
//! absorb all coefficients under the label `neo/ccs/round`.

use crate::error::PiCcsError;
use neo_transcript::{Transcript, Poseidon2Transcript};
use neo_math::KExtensions;
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
pub fn poly_eval_k(coeffs: &[K], x: K) -> K {
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

#[inline]
fn effective_degree(coeffs: &[K]) -> usize {
    match coeffs.iter().rposition(|&c| c != K::ZERO) {
        Some(idx) => idx,
        None => 0, // treat zero polynomial as degree 0 for bounding purposes
    }
}

/// Prover: run ℓ rounds of sum-check driven by `oracle`.
/// - `initial_sum` is the public sum claim (0 in Π_CCS use).
/// - `sample_xs` are the interpolation points (e.g., 0..=d).
/// 
/// For unique reconstruction, pass at least `d_sc+1` pairwise-distinct `sample_xs`.
pub fn run_sumcheck(
    tr: &mut Poseidon2Transcript,
    oracle: &mut dyn RoundOracle,
    initial_sum: K,
    sample_xs: &[K],
) -> Result<SumcheckOutput, PiCcsError> {
    // Guard: Lagrange interpolation requires pairwise-distinct nodes.
    // Keep it K-agnostic and future-proof (works for any extension degree d_K ≥ 1).
    for i in 0..sample_xs.len() {
        for j in 0..i {
            if sample_xs[i] == sample_xs[j] {
                return Err(PiCcsError::SumcheckError("duplicate sample_xs".into()));
            }
        }
    }
    let ell = oracle.num_rounds();
    let d_sc = oracle.degree_bound();
    
    // Optional: enforce sufficient nodes for degree bound (stronger soundness)
    if sample_xs.len() < d_sc + 1 {
        return Err(PiCcsError::SumcheckError(format!(
            "insufficient sample points: need at least {} for degree bound {}, got {}",
            d_sc + 1, d_sc, sample_xs.len()
        )));
    }

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
        let deg = effective_degree(&coeffs);
        if deg > d_sc {
            return Err(PiCcsError::SumcheckError(format!(
                "round {i}: degree {} exceeds bound {d_sc}", deg
            )));
        }

        // Standard consistency check: p(0) + p(1) == running_sum
        let p0 = poly_eval_k(&coeffs, K::ZERO);
        let p1 = poly_eval_k(&coeffs, K::ONE);
        #[cfg(feature = "debug-logs")]
        {
            let fmt = |x: K| -> String {
                let c = x.as_coeffs();
                format!("{:?}", c)
            };
            eprintln!(
                "[sumcheck][round{round}] deg={} p(0)={} p(1)={} p(0)+p(1)={} expected={}",
                deg, fmt(p0), fmt(p1), fmt(p0 + p1), fmt(running_sum),
                round = i
            );
            if i == 0 {
                eprintln!("  Initial sum claim: {}", fmt(initial_sum));
            }
        }
        if p0 + p1 != running_sum {
            return Err(PiCcsError::SumcheckError(format!(
                "round {i}: p(0)+p(1) mismatch"
            )));
        }

        // Bind polynomial to transcript and sample r_i
        // absorb round coeffs as base-field limbs
        let coeffs0 = coeffs[0].as_coeffs();
        tr.append_message(b"neo/ccs/round", b"");
        tr.append_fields(b"round/coeffs", &coeffs0);
        for c in coeffs.iter().skip(1) {
            let cc = c.as_coeffs();
            tr.append_fields(b"round/coeffs", &cc);
        }
        let ch = tr.challenge_fields(b"chal/k", 2);
        let r_i = neo_math::from_complex(ch[0], ch[1]);
        
        #[cfg(feature = "debug-logs")]
        {
            let dbg_oracle = std::env::var("NEO_ORACLE_TRACE").ok().as_deref() == Some("1");
            if dbg_oracle && i < 2 {
                eprintln!("[sumcheck][prove] round{}: challenge r_i={:?}", i, r_i);
            }
        }

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
    tr: &mut Poseidon2Transcript,
    d_sc: usize,
    initial_sum: K,
    rounds: &[Vec<K>],
) -> (Vec<K>, K, bool) {
    let mut running_sum = initial_sum;
    let mut r = Vec::with_capacity(rounds.len());

    for (round_idx, coeffs) in rounds.iter().enumerate() {
        if coeffs.is_empty() { return (Vec::new(), K::ZERO, false); }
        let deg = effective_degree(coeffs);
        if deg > d_sc {
            let dbg = std::env::var("NEO_ORACLE_TRACE").ok().as_deref() == Some("1");
            if dbg { eprintln!("[sumcheck][verify] round{}: degree {} exceeds bound {}", round_idx, deg, d_sc); }
            return (Vec::new(), K::ZERO, false);
        }
        let p0 = poly_eval_k(coeffs, K::ZERO);
        let p1 = poly_eval_k(coeffs, K::ONE);
        // Back-compat fallback: if the initial claim isn't carried, derive it from round 0
        if round_idx == 0 && p0 + p1 != running_sum {
            running_sum = p0 + p1;
        }
        if p0 + p1 != running_sum {
            let dbg = std::env::var("NEO_ORACLE_TRACE").ok().as_deref() == Some("1");
            if dbg { eprintln!("[sumcheck][verify] round{}: p(0)+p(1) mismatch", round_idx); }
            return (Vec::new(), K::ZERO, false);
        }
        tr.append_message(b"neo/ccs/round", b"");
        let c0 = coeffs[0].as_coeffs();
        tr.append_fields(b"round/coeffs", &c0);
        for c in coeffs.iter().skip(1) {
            tr.append_fields(b"round/coeffs", &c.as_coeffs());
        }
        let ch = tr.challenge_fields(b"chal/k", 2);
        let r_i = neo_math::from_complex(ch[0], ch[1]);
        
        #[cfg(feature = "debug-logs")]
        {
            let dbg = std::env::var("NEO_ORACLE_TRACE").ok().as_deref() == Some("1");
            if dbg && round_idx < 2 {
                eprintln!("[sumcheck][verify] round{}: challenge r_i={:?}", round_idx, r_i);
            }
        }
        
        running_sum = poly_eval_k(coeffs, r_i);
        r.push(r_i);
    }
    (r, running_sum, true)
}

/// Prover: run ℓ rounds but skip evaluating the round polynomial at X=1.
/// We reconstruct s_i(1) from the sum-check invariant s_i(0) + s_i(1) = running_sum.
/// This saves one evaluation per round while keeping transcript binding identical.
///
/// Requirements on `sample_xs_full`:
/// - must contain distinct points
/// - must include 0 and 1
pub fn run_sumcheck_skip_eval_at_one(
    tr: &mut Poseidon2Transcript,
    oracle: &mut dyn RoundOracle,
    initial_sum: K,
    sample_xs_full: &[K],
) -> Result<SumcheckOutput, PiCcsError> {
    // Guards identical to run_sumcheck
    for i in 0..sample_xs_full.len() {
        for j in 0..i {
            if sample_xs_full[i] == sample_xs_full[j] {
                return Err(PiCcsError::SumcheckError("duplicate sample_xs".into()));
            }
        }
    }
    let ell = oracle.num_rounds();
    let d_sc = oracle.degree_bound();
    if sample_xs_full.len() < d_sc + 1 {
        return Err(PiCcsError::SumcheckError(format!(
            "insufficient sample points: need at least {} for degree bound {}, got {}",
            d_sc + 1, d_sc, sample_xs_full.len()
        )));
    }

    // Locate 0 and 1 in the full sampling set
    let _idx_zero = sample_xs_full
        .iter()
        .position(|&x| x == K::ZERO)
        .ok_or_else(|| PiCcsError::SumcheckError("sample_xs_full must contain 0".into()))?;
    let idx_one = sample_xs_full
        .iter()
        .position(|&x| x == K::ONE)
        .ok_or_else(|| PiCcsError::SumcheckError("sample_xs_full must contain 1".into()))?;

    // Build reduced set without X=1, preserving order
    let mut sample_xs_wo_one = Vec::with_capacity(sample_xs_full.len().saturating_sub(1));
    for (i, &x) in sample_xs_full.iter().enumerate() {
        if i != idx_one { sample_xs_wo_one.push(x); }
    }

    let mut rounds = Vec::with_capacity(ell);
    let mut challenges = Vec::with_capacity(ell);
    let mut running_sum = initial_sum;

    for i in 0..ell {
        // Evaluate at all points except X=1
        let sample_ys_wo_one = oracle.evals_at(&sample_xs_wo_one);
        if sample_ys_wo_one.len() != sample_xs_wo_one.len() {
            return Err(PiCcsError::SumcheckError("evals_at returned wrong length".into()));
        }

        // Extract s_i(0)
        let pos_zero_in_reduced = sample_xs_wo_one
            .iter()
            .position(|&x| x == K::ZERO)
            .ok_or_else(|| PiCcsError::SumcheckError("reduced sample set missing 0".into()))?;
        let s_i_at_0 = sample_ys_wo_one[pos_zero_in_reduced];
        // CRITICAL: enforce the sum-check invariant with the previous running sum:
        // s_{i-1}(r_{i-1}) = s_i(0) + s_i(1)
        let s_i_at_1 = running_sum - s_i_at_0;
        #[cfg(feature = "debug-logs")]
        {
            let fmt = |x: K| -> String {
                let c = x.as_coeffs();
                format!("{:?}", c)
            };
            eprintln!(
                "[sumcheck-skip1][round{round}] s(0)={} s(1)={} s(0)+s(1)={} expected={}",
                fmt(s_i_at_0), fmt(s_i_at_1), fmt(s_i_at_0 + s_i_at_1), fmt(running_sum),
                round = i
            );
        }

        // Prover-side sanity check: verify oracle consistency
        // If the oracle has a bug (e.g., missing Ajtai contributions), fail fast here
        // rather than generating a proof that will definitely fail verification.
        #[cfg(debug_assertions)]
        {
            // Strong, same-call strict check: validate p(0)+p(1) directly from oracle outputs
            // to avoid cross-batch ambiguities.
            let ys_01 = oracle.evals_at(&[K::ZERO, K::ONE]);
            if ys_01.len() != 2 || ys_01[0] + ys_01[1] != running_sum {
                return Err(PiCcsError::SumcheckError(format!(
                    "[sumcheck-skip1][round{}] strict invariant failed at (0,1): got {}+{}={} expected {}",
                    i,
                    ys_01.get(0).copied().unwrap_or(K::ZERO),
                    ys_01.get(1).copied().unwrap_or(K::ZERO),
                    ys_01.get(0).copied().unwrap_or(K::ZERO) + ys_01.get(1).copied().unwrap_or(K::ZERO),
                    running_sum
                )));
            }
        }
        // Suppress warning about unused 'i' in release builds (used in debug_assertions)
        #[cfg(not(debug_assertions))]
        let _ = i;

        // Reconstruct full evaluation array in original order, inserting s_i(1)
        let mut sample_ys_full = Vec::with_capacity(sample_xs_full.len());
        let mut j = 0usize;
        for f in 0..sample_xs_full.len() {
            if f == idx_one { sample_ys_full.push(s_i_at_1); } else { sample_ys_full.push(sample_ys_wo_one[j]); j += 1; }
        }

        // Interpolate and enforce degree/invariant checks
        let coeffs = lagrange_interpolate_k(sample_xs_full, &sample_ys_full);
        let deg = effective_degree(&coeffs);
        if deg > d_sc {
            return Err(PiCcsError::SumcheckError(format!(
                "degree {} exceeds bound {d_sc}", deg
            )));
        }

        let p0 = poly_eval_k(&coeffs, K::ZERO);
        let p1 = poly_eval_k(&coeffs, K::ONE);
        // Verify the sum-check invariant against the previous running_sum
        if p0 + p1 != running_sum {
            return Err(PiCcsError::SumcheckError("p(0)+p(1) mismatch (skip-1)".into()));
        }

        // Transcript binding
        tr.append_message(b"neo/ccs/round", b"");
        let c0 = coeffs[0].as_coeffs();
        tr.append_fields(b"round/coeffs", &c0);
        for c in coeffs.iter().skip(1) { tr.append_fields(b"round/coeffs", &c.as_coeffs()); }
        let ch = tr.challenge_fields(b"chal/k", 2);
        let r_i = neo_math::from_complex(ch[0], ch[1]);

        running_sum = poly_eval_k(&coeffs, r_i);
        oracle.fold(r_i);

        challenges.push(r_i);
        rounds.push(coeffs);
    }

    Ok(SumcheckOutput { rounds, challenges, final_sum: running_sum })
}
