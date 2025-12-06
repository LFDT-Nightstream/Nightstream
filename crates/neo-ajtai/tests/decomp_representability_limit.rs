//! Test for decomposition representability limit bug.
//!
//! This test exposes the issue where base-b decomposition with D=54 digits
//! cannot faithfully represent all Goldilocks field elements >= 2^54.
//!
//! When b^D < q (the field modulus), large values cannot round-trip through
//! decomposition and recomposition, causing CCS constraint violations.
//!
//! Expected behavior:
//! - For Goldilocks (64-bit prime), we need b^D >= 2^64 for full representation
//! - With b=2 and D=54, we have 2^54 < 2^64, so values >= 2^54 will fail
//! - This shows up in multiplication constraints (x*x) when x >= 2^54

#![allow(non_snake_case)]
use neo_ajtai::{decomp_b, DecompStyle};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as Fq;

/// Test that decomposition round-trips correctly for all representable values.
///
/// This test checks the fundamental requirement: if we decompose a field element
/// into base-b digits and recompose it, we should get back the original value.
///
/// For base b=2 and D=54 digits, we can only represent values up to 2^54.
/// Beyond that, the decomposition cannot faithfully encode the value, and
/// recomposition yields a different field element.
///
/// This manifests as:
/// 1. Addition constraints (x+x) work fine since sums grow slowly
/// 2. Multiplication constraints (x*x) fail when x >= ~2^27 because x^2 >= 2^54
/// 3. Poseidon2 (x^7) fails even sooner
///
/// This test is marked as should_panic because with current parameters (D=54),
/// the representability limit IS insufficient for Goldilocks. When D is increased
/// to 64 or higher, this test should be updated to pass.
#[test]
#[should_panic(expected = "Decomposition representability limit confirmed")]
fn decomp_recomp_roundtrip_limit_base2() {
    let d = 54; // D from neo_math::ring
    let b = 2u32; // Base-2 decomposition (standard in Neo)

    // The theoretical limit: b^d
    // For b=2, d=54: limit = 2^54 = 18_014_398_509_481_984
    let theoretical_limit = 1u64 << 54;

    println!("Testing decomposition round-trip with b={}, d={}", b, d);
    println!("Theoretical representability limit: 2^{} = {}", d, theoretical_limit);

    // Test strategy: sample points across the range
    let test_ranges = vec![
        // Well within safe range
        (0u64, 1000u64, "small values"),
        // Approaching 2^27 (where x*x starts to hit 2^54)
        ((1u64 << 27) - 100, (1u64 << 27) + 100, "near 2^27"),
        // Approaching 2^54 (the critical boundary)
        (theoretical_limit - 1000, theoretical_limit + 1000, "near 2^54 boundary"),
        // Well beyond 2^54
        ((1u64 << 56), (1u64 << 56) + 1000, "beyond 2^54 at 2^56"),
    ];

    let mut first_mismatch: Option<u64> = None;

    for (start, end, label) in test_ranges {
        println!("\nTesting range: {} ({} to {})", label, start, end);

        for x_u in start..end {
            let x = Fq::from_u64(x_u);

            // Decompose using NonNegative style (matches the recomposed_z_from_Z logic)
            let digits = decomp_b(&[x], b, d, DecompStyle::NonNegative);

            // Recompose using the same logic as recomposed_z_from_Z in the engine
            let mut pow = Fq::ONE;
            let mut acc = Fq::ZERO;
            let bFq = Fq::from_u64(b as u64);

            for &digit in &digits[0..d] {
                acc += digit * pow;
                pow *= bFq;
            }

            if acc != x && first_mismatch.is_none() {
                first_mismatch = Some(x_u);
                println!("  MISMATCH at x = {} (0x{:x})", x_u, x_u);
                println!("    original:    {:?}", x);
                println!("    recomposed:  {:?}", acc);
                println!("    difference:  {} from 2^54", x_u as i128 - theoretical_limit as i128);
            }
        }
    }

    match first_mismatch {
        Some(x_u) => {
            println!("\n=== REPRESENTABILITY LIMIT DETECTED ===");
            println!("First mismatch at: x = {} (0x{:x})", x_u, x_u);
            println!("Theoretical limit: 2^{} = {}", d, theoretical_limit);
            println!("Difference: {} values", x_u as i128 - theoretical_limit as i128);
            println!("\nThis demonstrates that b^D < q:");
            println!("  b^D = 2^{} = {} (representability limit)", d, theoretical_limit);
            println!("  q = 2^64 - 2^32 + 1 ≈ 2^64 (Goldilocks modulus)");
            println!("\nIMPACT:");
            println!(
                "  - Multiplication constraints fail when intermediate values >= 2^{}",
                d
            );
            println!("  - Example: x*x constraint fails when x >= 2^{}", d / 2);
            println!("  - Poseidon2 (x^7) fails when x >= 2^{}", d / 7);
            println!("\nFIX:");
            println!("  - Increase D so that b^D >= q");
            println!("  - For b=2, need D >= 64 to represent all Goldilocks elements");
            println!("  - Or increase b and adjust D accordingly");

            panic!(
                "Decomposition representability limit confirmed: values >= 2^{} cannot round-trip",
                d
            );
        }
        None => {
            println!("\n=== ALL TESTS PASSED ===");
            println!("No mismatches detected in tested ranges.");
            println!("b^D = 2^{} appears sufficient for the field.", d);
        }
    }
}

/// Test specifically for the x*x constraint case that triggers the bug.
///
/// This simulates what happens in a CCS constraint y = x*x when x >= 2^27:
/// - The host computes y_expected = x*x correctly in the field
/// - The decomposition/recomposition loses precision for large y
/// - The CCS constraint checks the recomposed values, which don't match
#[test]
#[should_panic(expected = "multiplication constraint violated")]
fn multiplication_constraint_fails_for_large_values() {
    let d = 54;
    let b = 2u32;

    // Choose x such that x*x >= 2^54
    // With x = 2^27, we have x*x = 2^54 exactly
    let x_u = 1u64 << 27; // 2^27 = 134_217_728
    let x = Fq::from_u64(x_u);
    let y_expected = x * x; // Correct field multiplication

    println!("Testing multiplication constraint: y = x*x");
    println!("  x = 2^27 = {}", x_u);
    println!("  y_expected (x*x in field) = {:?}", y_expected);

    // Decompose and recompose y (simulating what the CCS engine does)
    let y_digits = decomp_b(&[y_expected], b, d, DecompStyle::NonNegative);

    let mut pow = Fq::ONE;
    let mut y_recomposed = Fq::ZERO;
    let bFq = Fq::from_u64(b as u64);

    for &digit in &y_digits[0..d] {
        y_recomposed += digit * pow;
        pow *= bFq;
    }

    println!("  y_recomposed (from decomp) = {:?}", y_recomposed);

    if y_recomposed != y_expected {
        println!("\nERROR: Recomposed value doesn't match!");
        println!("  This is why CCS F(constraints) != 0 in the sumcheck");
        panic!("multiplication constraint violated: y_recomposed != x*x");
    }
}

/// Diagnostic test that reports the representability limit without failing.
///
/// This test performs the same checks as decomp_recomp_roundtrip_limit_base2
/// but reports findings as warnings instead of panicking. Useful for checking
/// parameter changes.
#[test]
fn diagnostic_representability_report() {
    let d = 54; // D from neo_math::ring
    let b = 2u32; // Base-2 decomposition (standard in Neo)

    let theoretical_limit = 1u64 << 54;

    println!("\n=== DECOMPOSITION REPRESENTABILITY DIAGNOSTIC ===");
    println!("Parameters: b={}, d={}", b, d);
    println!("Theoretical limit: b^d = 2^{} = {}", d, theoretical_limit);
    println!("Field modulus (Goldilocks): q = 2^64 - 2^32 + 1 ≈ 2^64");

    // Quick check: test a value just below and at the limit
    let test_below = theoretical_limit - 1;
    let test_at = theoretical_limit;

    let check_roundtrip = |x_u: u64| -> bool {
        let x = Fq::from_u64(x_u);
        let digits = decomp_b(&[x], b, d, DecompStyle::NonNegative);

        let mut pow = Fq::ONE;
        let mut acc = Fq::ZERO;
        let bFq = Fq::from_u64(b as u64);

        for &digit in &digits[0..d] {
            acc += digit * pow;
            pow *= bFq;
        }

        acc == x
    };

    let below_ok = check_roundtrip(test_below);
    let at_ok = check_roundtrip(test_at);

    println!("\nRound-trip test results:");
    println!("  x = 2^{} - 1: {}", d, if below_ok { "✓ OK" } else { "✗ FAIL" });
    println!("  x = 2^{}:     {}", d, if at_ok { "✓ OK" } else { "✗ FAIL" });

    if !at_ok {
        println!("\nWARNING: Representability limit detected at 2^{}", d);
        println!("IMPACT:");
        println!("  - Values >= 2^{} cannot be faithfully represented", d);
        println!("  - Multiplication x*x fails when x >= 2^{}", d / 2);
        println!("  - This causes CCS constraint violations in the prover");
        println!("\nRECOMMENDED FIX:");
        println!("  - Increase D from {} to at least 64 (for Goldilocks)", d);
        println!("  - Or increase base b and adjust D to ensure b^D >= q");
    } else {
        println!("\n✓ Representability check passed for 2^{}", d);
        println!("  All field elements should be representable.");
    }
}

/// Test that addition constraints work even for large values.
///
/// This demonstrates that addition (x+x) doesn't hit the representability limit
/// as quickly as multiplication, explaining why "addition works but multiplication fails".
#[test]
fn addition_constraint_works_for_moderately_large_values() {
    let d = 54;
    let b = 2u32;

    // Test values that would cause multiplication to fail
    let test_values = vec![
        1u64 << 26, // 2^26
        1u64 << 27, // 2^27 (x*x hits 2^54)
        1u64 << 28, // 2^28 (x*x exceeds 2^54)
    ];

    for x_u in test_values {
        let x = Fq::from_u64(x_u);
        let sum_expected = x + x; // Correct field addition

        // Decompose and recompose the sum
        let sum_digits = decomp_b(&[sum_expected], b, d, DecompStyle::NonNegative);

        let mut pow = Fq::ONE;
        let mut sum_recomposed = Fq::ZERO;
        let bFq = Fq::from_u64(b as u64);

        for &digit in &sum_digits[0..d] {
            sum_recomposed += digit * pow;
            pow *= bFq;
        }

        println!(
            "x = 2^{}: sum round-trip OK? {}",
            x_u.trailing_zeros(),
            sum_recomposed == sum_expected
        );

        assert_eq!(
            sum_recomposed, sum_expected,
            "Addition should round-trip even when multiplication fails"
        );
    }

    println!("\nAddition constraints work because x+x grows linearly,");
    println!("while x*x grows quadratically and hits 2^54 much sooner.");
}
