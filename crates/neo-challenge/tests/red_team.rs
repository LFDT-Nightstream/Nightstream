use neo_challenge::{StrongSetConfig, DEFAULT_STRONGSET};

#[test]
fn config_zero_coeff_bound_invalid() {
    // coeff_bound=0 => every coefficient is zero => differences are zero => not invertible
    let cfg = StrongSetConfig{ eta: 81, d: 54, coeff_bound: 0, domain_sep: b"rt", max_resamples: 3 };
    // This is a degenerate case that should be avoided
    assert_eq!(cfg.expansion_upper_bound(), 0); // 2 * phi(81) * 0 = 0
    println!("✅ RED TEAM: Zero coefficient bound produces zero expansion bound as expected");
}

#[test]
fn config_validation_edge_cases() {
    
    // Test config with very small d (should still work but might be insecure)
    let small_cfg = StrongSetConfig {
        eta: 81,
        d: 1, // Minimal dimension
        coeff_bound: 2,
        domain_sep: b"test",
        max_resamples: 16,
    };
    
    // This might work but expansion bound should reflect the small dimension
    let bound = small_cfg.expansion_upper_bound();
    assert_eq!(bound, 2 * 54 * 2); // 2 * phi(81) * 2 = 216, regardless of d setting
    println!("✅ RED TEAM: Small dimension config has expected expansion bound: {}", bound);
    
    // Test config with large coefficient bound
    let large_cfg = StrongSetConfig {
        eta: 81,
        d: 54,
        coeff_bound: 1000, // Large bound
        domain_sep: b"test",
        max_resamples: 16,
    };
    
    let large_bound = large_cfg.expansion_upper_bound();
    assert_eq!(large_bound, 2 * 54 * 1000); // Should scale with H
    println!("✅ RED TEAM: Large coefficient bound scales expansion correctly: {}", large_bound);
}

#[test] 
fn resampling_config_properties() {
    // Test config that would likely fail invertibility with very limited resamples
    let strict_cfg = StrongSetConfig {
        eta: 81,
        d: 54,
        coeff_bound: 1, // Very small range [-1,0,1] 
        domain_sep: b"strict",
        max_resamples: 1, // Only one attempt
    };
    
    // Verify the configuration is mathematically reasonable
    assert_eq!(strict_cfg.phi_eta(), 54);
    assert_eq!(strict_cfg.expansion_upper_bound(), 2 * 54 * 1); // 108
    println!("✅ RED TEAM: Strict config has expected mathematical properties");
}

#[test]
fn domain_separation_configs_differ() {
    // Different domain separators should be distinct even with same other params
    let cfg1 = StrongSetConfig {
        eta: 81,
        d: 54,
        coeff_bound: 2,
        domain_sep: b"domain1",
        max_resamples: 16,
    };
    
    let cfg2 = StrongSetConfig {
        domain_sep: b"domain2", // Different domain
        ..cfg1
    };
    
    // The configs should have different domain separators but same mathematical properties
    assert_ne!(cfg1.domain_sep, cfg2.domain_sep);
    assert_eq!(cfg1.phi_eta(), cfg2.phi_eta());
    assert_eq!(cfg1.expansion_upper_bound(), cfg2.expansion_upper_bound());
    println!("✅ RED TEAM: Domain separation configs have distinct separators but same math properties");
}

#[test]
fn phi_eta_calculation_accuracy() {
    let cfg = DEFAULT_STRONGSET;
    
    // φ(81) should equal 54
    // 81 = 3^4, so φ(81) = 81 * (1 - 1/3) = 81 * 2/3 = 54
    assert_eq!(cfg.phi_eta(), 54);
    
    // Test some other values
    let cfg2 = StrongSetConfig {
        eta: 12, // 12 = 2^2 * 3, φ(12) = 12 * (1-1/2) * (1-1/3) = 12 * 1/2 * 2/3 = 4
        d: 4,
        coeff_bound: 2,
        domain_sep: b"test",
        max_resamples: 16,
    };
    assert_eq!(cfg2.phi_eta(), 4);
    
    println!("✅ RED TEAM: Euler totient function calculated correctly");
}

#[test] 
fn default_config_properties() {
    let cfg = DEFAULT_STRONGSET;
    
    // Verify expected values from the paper
    assert_eq!(cfg.eta, 81);
    assert_eq!(cfg.d, 54);  
    assert_eq!(cfg.coeff_bound, 2);
    assert_eq!(cfg.max_resamples, 16);
    
    // Mathematical properties
    assert_eq!(cfg.phi_eta(), 54); // φ(81) = 54
    assert_eq!(cfg.expansion_upper_bound(), 2 * 54 * 2); // 216
    
    // Domain separator should be reasonable
    assert!(!cfg.domain_sep.is_empty());
    assert!(cfg.domain_sep.len() < 100); // Not too long
    
    println!("✅ RED TEAM: Default configuration matches expected paper values");
}
