use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks as Fq;
use p3_matrix::dense::RowMajorMatrix;

use super::{StrongSetConfig, Rho};

#[derive(Clone, Debug)]
pub struct ExpansionStats {
    pub trials: usize,
    pub max_ratio: f64,
    pub avg_ratio: f64,
}

/// Empirically estimate expansion ‖ρ·v‖∞/‖v‖∞ across random v.
/// This should never exceed T (Theorem 3); useful in tests/benchmarks.
pub fn empirical_expansion_stats(rhos: &[Rho], cfg: &StrongSetConfig, trials: usize) -> ExpansionStats {
    use rand::{Rng, SeedableRng};
    use rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(0x5eed);
    let d = cfg.d;

    let mut max_ratio = 0f64;
    let mut sum_ratio = 0f64;

    for _ in 0..trials {
        // random v ∈ F_q^d with entries in [-H..H]
        let mut v = vec![Fq::ZERO; d];
        for x in v.iter_mut() {
            let z = rng.random_range(-cfg.coeff_bound..=cfg.coeff_bound);
            *x = if z >= 0 { Fq::from_u64(z as u64) } else { Fq::ZERO - Fq::from_u64((-z) as u64) };
        }
        let v_inf = l_inf(&v);

        for rho in rhos {
            let w = mul_rho_vec(&rho.matrix, &v, d);
            let w_inf = l_inf(&w);
            let ratio = if v_inf == 0f64 { 0f64 } else { w_inf / v_inf };
            if ratio > max_ratio { max_ratio = ratio; }
            sum_ratio += ratio;
        }
    }

    ExpansionStats { trials, max_ratio, avg_ratio: sum_ratio / (trials as f64 * rhos.len() as f64) }
}

fn l_inf(v: &[Fq]) -> f64 {
    // Convert to small signed repr; good enough for metrics.
    let q: u128 = neo_math::GOLDILOCKS_MODULUS; // 2^64 - 2^32 + 1
    let half_q = (q / 2) as i128;
    v.iter().map(|x| {
        let mut u = x.as_canonical_u64() as i128;
        // center in [-q/2, q/2] for heuristic readability
        if u > half_q { u -= q as i128; }
        u.unsigned_abs() as f64
    }).fold(0.0, f64::max)
}

fn mul_rho_vec(m: &RowMajorMatrix<Fq>, v: &[Fq], d: usize) -> Vec<Fq> {
    let mut out = vec![Fq::ZERO; d];
    out.iter_mut().enumerate().for_each(|(r, out_elem)| {
        let mut acc = Fq::ZERO;
        v.iter().enumerate().for_each(|(c, &v_c)| {
            acc += m.values[r * d + c] * v_c;
        });
        *out_elem = acc;
    });
    out
}
