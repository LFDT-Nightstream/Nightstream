//! Extension policy enforcement tests
//! 
//! Tests that validate the extension-degree policy actually enforces security requirements:
//! 1. Rejects unsafe parameter combinations where s_min > 2  
//! 2. Computes correct s_min from circuit parameters
//! 3. Binds extension policy to transcript for FS soundness
//! 4. Different policies produce different transcript digests

use neo_transcript::{Poseidon2Transcript, Transcript};
use neo_params::NeoParams;

#[test]
fn extension_policy_enforces_s_min_limit() {
    let params = NeoParams::goldilocks_127();
    
    // Test case 1: Minimal circuit that should pass (only ell=1, d=1 with Goldilocks-127)
    let ell_minimal = 1u32;   // log2(2) - absolute minimum
    let d_minimal = 1u32;     // degree 1 (minimal)
    
    let result_pass = params.extension_check(ell_minimal, d_minimal);
    assert!(result_pass.is_ok(), "Minimal circuit should pass extension policy check");
    
    let summary = result_pass.unwrap();
    assert_eq!(summary.s_supported, 2, "v1 policy should always use s=2");
    assert_eq!(summary.s_min, 2, "Minimal circuit should have s_min=2 with Goldilocks-127");
    assert_eq!(summary.slack_bits, 0, "Minimal circuit should have exactly 0 slack bits with exact integer arithmetic");
    
    println!("✅ Minimal circuit: ell={}, d={}, s_min={}, slack_bits={}", 
             ell_minimal, d_minimal, summary.s_min, summary.slack_bits);
}

#[test] 
fn extension_policy_rejects_non_minimal_circuits() {
    let params = NeoParams::goldilocks_127();
    
    // With exact bit-length arithmetic, these cases now correctly compute s_min:
    // ell=2,d=1 -> s_min=2 (now passes!), ell=1,d=2 -> s_min=2 (now passes!)
    // But larger circuits still require s_min=3+
    let test_cases = [
        (4u32, 2u32),  // ell=4,d=2: ld=8, s_min=3 (should reject)
        (8u32, 2u32),  // ell=8,d=2: ld=16, s_min=3 (should reject) 
        (25u32, 8u32), // Large circuit: s_min >> 2 (should reject)
        (16u32, 1u32), // ell=16: s_min=3 (should reject)
        (1u32, 16u32), // d=16: s_min=3 (should reject) 
    ];
    
    for (ell, d) in test_cases {
        let result_fail = params.extension_check(ell, d);
        assert!(result_fail.is_err(), "Circuit (ell={}, d={}) should fail extension policy check", ell, d);
        
        match result_fail {
            Err(neo_params::ParamsError::UnsupportedExtension { required }) => {
                assert!(required > 2, "Should require s > 2 for circuit (ell={}, d={})", ell, d);
                println!("✅ Circuit (ell={}, d={}) correctly rejected: required s={} > 2", ell, d, required);
            },
            _ => panic!("Expected UnsupportedExtension error for circuit (ell={}, d={})", ell, d),
        }
    }
}

#[test]
fn transcript_binding_changes_with_policy_parameters() {
    // Test that different extension policies produce different transcript digests
    
    // Create two transcripts with different circuit parameters
    let mut tr1 = Poseidon2Transcript::new(b"extension_test_1");
    let mut tr2 = Poseidon2Transcript::new(b"extension_test_1"); // Same seed
    
    // Bind different extension policies (mirror pi_ccs header framing)
    tr1.append_message(b"neo/ccs/header/v1", b"");
    tr1.append_u64s(b"ccs/header", &[64, 2, 127, 4, 2, 10]);
    tr2.append_message(b"neo/ccs/header/v1", b"");
    tr2.append_u64s(b"ccs/header", &[64, 2, 127, 8, 2, 5]);
    
    let digest1 = tr1.digest32();
    let digest2 = tr2.digest32();
    
    assert_ne!(digest1, digest2, 
               "Different extension policies must produce different transcript digests");
    
    println!("✅ Transcript binding verified: different policies → different digests");
    println!("   Policy 1 digest: {:?}", &digest1[..8]);
    println!("   Policy 2 digest: {:?}", &digest2[..8]);
}

#[test]
fn transcript_binding_is_deterministic() {
    // Test that the same policy parameters always produce the same digest
    let create_bound_transcript = || {
        let mut tr = Poseidon2Transcript::new(b"deterministic_test");
        tr.append_message(b"neo/ccs/header/v1", b"");
        tr.append_u64s(b"ccs/header", &[64, 2, 127, 6, 3, 8]);
        tr.digest32()
    };

    let digest1 = create_bound_transcript();
    let digest2 = create_bound_transcript();
    
    assert_eq!(digest1, digest2,
               "Same policy parameters must produce deterministic transcript digest");
    
    println!("✅ Transcript binding is deterministic");
}

#[test]
fn extension_policy_computes_correct_s_min() {
    let params = NeoParams::goldilocks_127();
    
    // Test s_min computation for known values
    // For Goldilocks (q = 2^64), lambda=127, the formula is:
    // s_min = ceil((127 + log2(ell*d)) / 64)
    // Since 127/64 ≈ 1.98, we need s_min >= 2 for any non-zero circuit
    
    let test_cases = [
        (1u32, 1u32, 2u32),    // Minimal: ceil((127 + 0) / 64) = ceil(1.98) = 2
        (2u32, 1u32, 3u32),    // 2^127 * 2 overflows u128, so s_min = 3 (exact calculation)
        (4u32, 2u32, 3u32),    // ceil((127 + ~3) / 64) = ceil(2.05) = 3  
        (8u32, 4u32, 3u32),    // ceil((127 + ~5) / 64) = ceil(2.08) = 3
    ];
    
    for (ell, d, expected_s_min) in test_cases {
        let actual_s_min = params.s_min(ell, d);
        assert_eq!(actual_s_min, expected_s_min,
                   "s_min mismatch for ell={}, d={}: expected {}, got {}",
                   ell, d, expected_s_min, actual_s_min);
        
        println!("✅ s_min({}, {}) = {} ✓ (exact calculation)", ell, d, actual_s_min);
    }
}

#[test] 
fn goldilocks_preset_satisfies_security_requirements() {
    // Verify the Goldilocks preset actually meets its security claims
    let params = NeoParams::goldilocks_127();
    
    // Check basic parameter validity
    assert_eq!(params.q, 0xFFFF_FFFF_0000_0001, "Goldilocks prime");
    assert_eq!(params.s, 2, "v1 policy: s=2");
    assert_eq!(params.lambda, 127, "Target ~127-bit security");
    
    // Check guard inequality: (k+1)*T*(b-1) < B
    let guard_lhs = (params.k as u128 + 1) * (params.T as u128) * ((params.b as u128) - 1);
    assert!(guard_lhs < params.B as u128, 
            "Guard inequality failed: {} ≥ {}", guard_lhs, params.B);
    
    println!("✅ Goldilocks preset validated:");
    println!("   Guard: ({}+1)*{}*({}-1) = {} < {}", 
             params.k, params.T, params.b, guard_lhs, params.B);
    
    // Test extension policy behavior: Goldilocks-127 is very conservative
    // Only the minimal circuit (ell=1, d=1) should pass
    let minimal_result = params.extension_check(1, 1);
    assert!(minimal_result.is_ok(), "Minimal circuit should be supported");
    
    let minimal_summary = minimal_result.unwrap();
    println!("   Minimal circuit (ell=1, d=1): s_min={}, slack_bits={}", 
             minimal_summary.s_min, minimal_summary.slack_bits);
    
    // Larger circuits should be rejected (s_min > 2)
    for ell in [2, 4, 8, 12, 16] {
        let result = params.extension_check(ell, 2);
        assert!(result.is_err(), "Circuit ell={} should be rejected (requires s>2)", ell);
        
        if let Err(neo_params::ParamsError::UnsupportedExtension { required }) = result {
            println!("   ell={}: correctly rejected (requires s={} > 2)", ell, required);
        }
    }
}
