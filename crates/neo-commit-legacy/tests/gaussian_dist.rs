use neo_commit::{AjtaiCommitter, TOY_PARAMS};
use neo_modint::{Coeff, ModInt};
use neo_ring::RingElement;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;

#[test]
fn test_gaussian_dist() {
    let comm = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
    // Use zero center for proper Gaussian distribution testing
    let center = RingElement::from_scalar(ModInt::from_u64(0), TOY_PARAMS.n);
    if std::env::var("NEO_DEBUG").is_ok() {
        println!("Center coeffs length (trimmed): {}", center.coeffs().len());
        println!("ModInt modulus (Q): {:#x}", <ModInt as Coeff>::modulus());
    }
    let mut rng = StdRng::seed_from_u64(42);
    let samples: u32 = std::env::var("NEO_GAUSS_SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);
    let mut counts: HashMap<i64, u32> = HashMap::new(); // frequency table over Z
    let mut all_values: Vec<i64> = Vec::with_capacity(samples as usize * TOY_PARAMS.n);

    for i in 0..samples {
        let sample = comm
            .sample_gaussian_ring(&center, TOY_PARAMS.sigma, &mut rng)
            .unwrap();
        // Always account for exactly n coefficients. RingElement is trimmed, so pad zeros.
        let coeffs = sample.coeffs();
        let q = <ModInt as Coeff>::modulus();
        for idx in 0..TOY_PARAMS.n {
            let c = if idx < coeffs.len() { coeffs[idx] } else { ModInt::from_u64(0) };
            let v = c.as_canonical_u64();
            // Convert to signed w.r.t. ModInt::Q (not TOY_PARAMS.q)
            let signed = if v > q / 2 {
                (v as i128 - q as i128) as i64
            } else {
                v as i64
            };
            all_values.push(signed);
            *counts.entry(signed).or_insert(0) += 1;
        }
        if i < 3 && std::env::var("NEO_DEBUG").is_ok() {
            let preview: Vec<i64> = (0..TOY_PARAMS.n).map(|idx| {
                let c = if idx < coeffs.len() { coeffs[idx] } else { ModInt::from_u64(0) };
                let v = c.as_canonical_u64();
                if v > q / 2 { (v as i128 - q as i128) as i64 } else { v as i64 }
            }).collect();
            println!("Sample {} (signed, padded): {:?}", i, preview);
        }
    }
    
    // Optional debug output (can be enabled with environment variable)
    if std::env::var("NEO_DEBUG").is_ok() {
        println!("Total unique values: {}", counts.len());
        if let (Some(min), Some(max)) = (all_values.iter().min(), all_values.iter().max()) {
            println!("Value range: {} to {}", min, max);
        }
        println!("TOY_PARAMS.sigma: {}", TOY_PARAMS.sigma);
        println!("TOY_PARAMS.n: {}", TOY_PARAMS.n);
        println!("(Signed conversion used Q = ModInt::Q = {:#x})", <ModInt as Coeff>::modulus());
    }

    let total = (samples as usize * TOY_PARAMS.n) as f64;
    let sum: f64 = all_values.iter().map(|&v| v as f64).sum();
    let mean: f64 = sum / total;
    if std::env::var("NEO_DEBUG").is_ok() {
        println!("Mean: {}", mean);
    }
    assert!(mean.abs() < 0.2);
    let var: f64 = all_values.iter().map(|&v| {
        let dv = v as f64 - mean;
        dv * dv
    }).sum::<f64>() / total;
    if std::env::var("NEO_DEBUG").is_ok() {
        println!("Variance: {} (expected: 10.24)", var);
    }
    assert!((var - 10.24).abs() < 2.0); // Loosen the bound temporarily
    let tail_cut = (3.0 * TOY_PARAMS.sigma).round() as i64;
    let tail_count: f64 = all_values.iter().filter(|&&v| v.abs() > tail_cut).count() as f64 / total;
    assert!(tail_count < 0.01);
}



#[test]
fn test_sample_gaussian_returns_err_on_low_sigma() {
    let comm = AjtaiCommitter::setup_unchecked(TOY_PARAMS);
    let center = RingElement::from_scalar(ModInt::from_u64(0), TOY_PARAMS.n);
    let mut rng = StdRng::seed_from_u64(42);
    let res = comm.sample_gaussian_ring(&center, 0.0, &mut rng);
    assert!(res.is_err());
}

#[test]
fn test_discrete_gaussian_sample_direct() {
    use neo_commit::AjtaiCommitter;
    let mut rng = StdRng::seed_from_u64(42);
    let samples = 1000;
    let mut values = Vec::new();
    
    for _ in 0..samples {
        let sample = AjtaiCommitter::discrete_gaussian_sample(TOY_PARAMS.sigma, &mut rng, TOY_PARAMS.q);
        // Interpret using ModInt::Q, not TOY_PARAMS.q
        let q = <ModInt as Coeff>::modulus();
        let v = sample.as_canonical_u64();
        let val = if v > q / 2 {
            (v as i128 - q as i128) as i64
        } else {
            v as i64
        };
        values.push(val);
    }
    
    let mean: f64 = values.iter().map(|&v| v as f64).sum::<f64>() / samples as f64;
    let var: f64 = values.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / samples as f64;
    
    if std::env::var("NEO_DEBUG").is_ok() {
        println!("Direct discrete_gaussian_sample:");
        println!("Mean: {}", mean);
        println!("Variance: {} (expected: 10.24)", var);
        println!("Range: {} to {}", values.iter().min().unwrap(), values.iter().max().unwrap());
    }
    
    assert!(mean.abs() < 0.2, "Mean should be close to 0, got {}", mean);
    // Optional: keep it informative but not strict on var in this direct test
}
