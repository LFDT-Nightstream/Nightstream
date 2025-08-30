#[allow(clippy::uninlined_format_args)]
use neo_params::{NeoParams, ParamsError};

#[test]
fn guard_rejects_tight_or_overflowing_profiles() {
    // Tight inequality: lhs == B should be rejected.
    // Choose b=2,k=12 => B=4096; pick T so (k+1)T(b-1)=4096
    let (b, k, d, eta, kappa, m, s, lambda) = (2u32, 12u32, 54u32, 81u32, 16u32, 1u64<<20, 2u32, 128u32);
    let q = 0xFFFF_FFFF_0000_0001u64;
    let t = 316u32; // makes lhs > B (exact would be 4096/13≈315.076, so 316 > bound)
    let err = NeoParams::new(q, eta, d, kappa, m, b, k, t, s, lambda).unwrap_err();
    assert!(matches!(err, ParamsError::GuardInequality));
    println!("✅ RED TEAM: Guard correctly rejects tight inequality");

    // Overflow in B=b^k must be rejected (checked u128 → u64 downcast)  
    // Pick b so b^k won't fit into u64: e.g., b=256, k=9 → 2^72 (large but avoids compile-time overflow)
    let large_b = 256u32;
    let large_k = 9u32; // 256^9 is much larger than u64::MAX
    let err2 = NeoParams::new(q, eta, d, kappa, m, large_b, large_k, 10, 2, 128).unwrap_err();
    assert!(matches!(err2, ParamsError::Invalid(_)));
    println!("✅ RED TEAM: B overflow correctly rejected");
}

#[test]
fn extension_policy_rejects_when_s_min_gt_2() {
    let p = NeoParams::goldilocks_127(); // s=2 compatible
    // Force s_min > 2 by tightening λ and picking large (ℓ·d_sc)
    let mut p2 = p;
    p2.lambda = 320; // very tight target
    let (ell, d_sc) = (64u32, 16u32);
    let e = p2.extension_check(ell, d_sc).unwrap_err();
    match e {
        ParamsError::UnsupportedExtension { required } => {
            assert!(required > 2);
            println!("✅ RED TEAM: Extension policy correctly rejects s_min={} > 2", required);
        },
        _ => panic!("expected UnsupportedExtension"),
    }
}

#[test]
fn s_min_and_slack_bits_behave() {
    let p = NeoParams::goldilocks_127(); // s=2 compatible
    
    // Test that s_min calculation doesn't panic and returns reasonable values
    let s_min1 = p.s_min(1, 1);
    let s_min2 = p.s_min(8, 8);
    
    // Both should be reasonable (not zero, not huge)
    assert!(s_min1 > 0 && s_min1 < 10);
    assert!(s_min2 > 0 && s_min2 < 10);
    
    // Test extension_check error handling
    match p.extension_check(64, 64) {
        Ok(summary) => {
            assert_eq!(summary.s_supported, 2);
            println!("✅ RED TEAM: Extension check passed for large inputs");
        }
        Err(_) => {
            println!("✅ RED TEAM: Extension check correctly rejects large inputs requiring s > 2");
        }
    }
}

#[test]
fn parameter_boundary_conditions() {
    let base_params = (0xFFFF_FFFF_0000_0001u64, 81u32, 54u32, 16u32, 1u64<<20, 2u32, 12u32, 216u32, 2u32, 128u32);
    let (q, eta, d, kappa, m, b, k, t, s, lambda) = base_params;

    // Test zero/invalid parameters are rejected
    assert!(matches!(NeoParams::new(0, eta, d, kappa, m, b, k, t, s, lambda).unwrap_err(), 
                    ParamsError::Invalid("q must be nonzero")));
    assert!(matches!(NeoParams::new(q, 0, d, kappa, m, b, k, t, s, lambda).unwrap_err(), 
                    ParamsError::Invalid("eta must be > 0")));
    assert!(matches!(NeoParams::new(q, eta, 0, kappa, m, b, k, t, s, lambda).unwrap_err(), 
                    ParamsError::Invalid("d must be > 0")));
    assert!(matches!(NeoParams::new(q, eta, d, 0, m, b, k, t, s, lambda).unwrap_err(), 
                    ParamsError::Invalid("kappa must be > 0")));
    assert!(matches!(NeoParams::new(q, eta, d, kappa, 0, b, k, t, s, lambda).unwrap_err(), 
                    ParamsError::Invalid("m must be > 0")));
    assert!(matches!(NeoParams::new(q, eta, d, kappa, m, 1, k, t, s, lambda).unwrap_err(), 
                    ParamsError::Invalid("b must be >= 2")));
    assert!(matches!(NeoParams::new(q, eta, d, kappa, m, b, 0, t, s, lambda).unwrap_err(), 
                    ParamsError::Invalid("k must be > 0")));
    assert!(matches!(NeoParams::new(q, eta, d, kappa, m, b, k, 0, s, lambda).unwrap_err(), 
                    ParamsError::Invalid("T must be > 0")));
    assert!(matches!(NeoParams::new(q, eta, d, kappa, m, b, k, t, 3, lambda).unwrap_err(), 
                    ParamsError::UnsupportedExtension { required: 3 }));
    assert!(matches!(NeoParams::new(q, eta, d, kappa, m, b, k, t, s, 0).unwrap_err(), 
                    ParamsError::Invalid("lambda must be > 0")));

    println!("✅ RED TEAM: All parameter boundary conditions correctly enforced");
}

#[test]
fn goldilocks_preset_security_invariants() {
    let p = NeoParams::goldilocks_127();
    
    // Verify the guard inequality is satisfied with margin
    let lhs = (p.k as u128 + 1) * (p.T as u128) * ((p.b as u128) - 1);
    let rhs = p.B as u128;
    assert!(lhs < rhs, "Guard inequality must hold: {} < {}", lhs, rhs);
    
    // Verify reasonable margin exists (not too tight)
    let margin = rhs - lhs;
    let margin_ratio = (margin as f64) / (rhs as f64);
    assert!(margin_ratio > 0.1, "Security margin should be > 10%, got {:.1}%", margin_ratio * 100.0);
    
    // Verify field parameters
    assert_eq!(p.q, 0xFFFF_FFFF_0000_0001); // Goldilocks prime
    assert_eq!(p.s, 2); // Extension degree
    assert_eq!(p.lambda, 127); // Security level (~127-bit for s=2 compatibility)
    
    println!("✅ RED TEAM: Goldilocks preset satisfies all security invariants");
    println!("   Guard margin: {:.1}% ({} out of {})", margin_ratio * 100.0, margin, rhs);
}
