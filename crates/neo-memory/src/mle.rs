//! Multilinear-extension helpers for Twist/Shout oracles.
use p3_field::Field;

/// Evaluate the less-than multilinear extension:
/// LT(j', j) = 1 if int(j') < int(j) else 0, with bit-vectors interpreted
/// little-endian. Valid over any field since it is a multilinear polynomial.
pub fn lt_eval<Kf: Field>(j_prime: &[Kf], j: &[Kf]) -> Kf {
    assert_eq!(j_prime.len(), j.len(), "lt_eval: length mismatch");
    let ell = j.len();

    // suffix[i] = Π_{k≥i} eq(j'_k, j_k)
    let mut suffix = vec![Kf::ONE; ell + 1];
    for i in (0..ell).rev() {
        let eq = eq_single(j_prime[i], j[i]);
        suffix[i] = suffix[i + 1] * eq;
    }

    let mut acc = Kf::ZERO;
    for i in 0..ell {
        let tail = suffix[i + 1];
        acc += (Kf::ONE - j_prime[i]) * j[i] * tail;
    }
    acc
}

/// Build the χ table for a point `r ∈ K^ℓ`, returning length `2^ℓ`.
///
/// χ_r[i] = Π_bit (r_bit if i_bit else 1-r_bit), little-endian bits.
pub fn build_chi_table<Kf: Field>(r: &[Kf]) -> Vec<Kf> {
    let ell = r.len();
    let n = 1usize << ell;
    let mut out = vec![Kf::ONE; n];

    // Gray-code style expansion: at step i, split every block into low/high halves.
    for (i, &ri) in r.iter().enumerate() {
        let stride = 1usize << i;
        let block = 1usize << (ell - i - 1);
        let one_minus = Kf::ONE - ri;

        let mut idx = 0usize;
        for _ in 0..block {
            for j in 0..stride {
                let a = out[idx + j];
                out[idx + j] = a * one_minus;
            }
            for j in 0..stride {
                let a = out[idx + stride + j];
                out[idx + stride + j] = a * ri;
            }
            idx += 2 * stride;
        }
    }

    out
}

/// Evaluate the multilinear extension of a vector `v` at point `r`.
///
/// `v` is interpreted over the Boolean hypercube of dimension `r.len()`.
pub fn mle_eval<F: Field, Kf: Field + From<F>>(v: &[F], r: &[Kf]) -> Kf {
    let chi = build_chi_table(r);
    debug_assert_eq!(v.len(), chi.len(), "mle_eval: dimension mismatch");
    let mut acc = Kf::ZERO;
    for (val, weight) in v.iter().zip(chi.iter()) {
        acc += Kf::from(*val) * *weight;
    }
    acc
}

#[inline]
fn eq_single<Kf: Field>(a: Kf, b: Kf) -> Kf {
    (Kf::ONE - a) * (Kf::ONE - b) + a * b
}

/// Re-export the eq polynomial for convenience.
pub use neo_reductions::engines::paper_exact_engine::eq_points;
