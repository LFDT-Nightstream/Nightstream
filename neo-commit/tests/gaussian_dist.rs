use neo_commit::{AjtaiCommitter, TOY_PARAMS};
use neo_modint::ModInt;
use neo_ring::RingElement;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;

#[cfg_attr(miri, ignore)]
#[test]
fn test_gaussian_dist() {
    if std::env::var("RUN_LONG_TESTS").is_err() {
        return;
    }
    let comm = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
    let center = RingElement::from_scalar(ModInt::from_u64(0), TOY_PARAMS.n);
    let mut rng = StdRng::seed_from_u64(42);
    let samples: u32 = std::env::var("NEO_GAUSS_SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);
    let mut counts: HashMap<i64, u32> = HashMap::new();

    for _ in 0..samples {
        let sample = comm
            .sample_gaussian_ring(&center, TOY_PARAMS.sigma, &mut rng)
            .unwrap();
        for &c in sample.coeffs() {
            let val = {
                let v = c.as_canonical_u64() as i64;
                let q = TOY_PARAMS.q as i64;
                if v > q / 2 {
                    v - q
                } else {
                    v
                }
            };
            *counts.entry(val).or_insert(0) += 1;
        }
    }

    let total = (samples as usize * TOY_PARAMS.n) as f64;
    let mean: f64 = counts
        .iter()
        .map(|(&v, &cnt)| v as f64 * cnt as f64)
        .sum::<f64>()
        / total;
    assert!(mean.abs() < 0.1);
    let var: f64 = counts
        .iter()
        .map(|(&v, &cnt)| (v as f64 - mean).powi(2) * cnt as f64)
        .sum::<f64>()
        / total;
    assert!((var - 10.24).abs() < 1.0);
    let tail_count: f64 = counts
        .iter()
        .filter(|(&v, _)| v.abs() > (3.0 * TOY_PARAMS.sigma) as i64)
        .map(|(_, &cnt)| cnt as f64)
        .sum::<f64>()
        / total;
    assert!(tail_count < 0.01);
}

#[test]
fn test_gpv_retry_limit() {
    let mut params = TOY_PARAMS;
    params.norm_bound = 1;
    params.e_bound = 1000; // avoid internal Gaussian failure
    let comm = AjtaiCommitter::setup_unchecked(params);
    let mut rng = StdRng::seed_from_u64(1);
    let target = vec![RingElement::zero(params.n); params.k];
    let res = comm.gpv_trapdoor_sample(&target, params.sigma, &mut rng);
    assert!(res.is_err());
    let err = res.unwrap_err();
    assert!(
        err == "GPV sampling failed after 100 retries"
            || err == "Gaussian sampling failed after 100 retries"
    );
}

#[test]
fn test_sample_gaussian_returns_err_on_low_sigma() {
    let comm = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
    let center = RingElement::from_scalar(ModInt::from_u64(0), TOY_PARAMS.n);
    let mut rng = StdRng::seed_from_u64(42);
    let res = comm.sample_gaussian_ring(&center, 0.001, &mut rng);
    assert!(res.is_err());
}
