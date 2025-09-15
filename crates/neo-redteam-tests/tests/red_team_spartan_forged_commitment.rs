// neo-tests/tests/red_team_spartan_forged_commitment.rs
#![cfg(feature = "redteam")]
#![allow(deprecated)] // Tests use legacy MEInstance/MEWitness for backward compatibility

//! Red-team test to exploit missing Ajtai binding in Spartan-bridge layer
//! 
//! When MEWitness.ajtai_rows is None, MeCircuit skips the ⟨Lᵢ,Z⟩ = cᵢ checks, 
//! so the Spartan proof binds only to the supplied Ajtai coordinates c_coords 
//! without verifying they derive from the witness digits z_digits.
//! 
//! A test can exploit this by forging c_coords while leaving ME constraints satisfied.

use neo_spartan_bridge::{compress_me_to_spartan, verify_me_spartan};
use neo_ccs::{MEInstance, MEWitness};
use p3_goldilocks::Goldilocks as F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

/// Create a minimal ME instance for testing with consistent Ajtai rows
fn tiny_me_instance() -> (MEInstance, MEWitness) {
    // Witness digits (len = 8 = 2^3)
    let z = vec![1i64, 2, 3, 0, -1, 1, 0, 2];

    // Ajtai rows: simple unit vectors so c0=z0, c1=z1
    let ajtai_rows = vec![
        vec![F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO],
        vec![F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO],
    ];
    let dot_f_z = |row: &[F]| -> F {
        row.iter().zip(z.iter()).fold(F::ZERO, |acc, (a, &zi)| {
            let zf = if zi >= 0 { F::from_u64(zi as u64) } else { -F::from_u64((-zi) as u64) };
            acc + *a * zf
        })
    };
    let c0 = dot_f_z(&ajtai_rows[0]); // = 1
    let c1 = dot_f_z(&ajtai_rows[1]); // = 2
    let c_coords = vec![c0, c1]; // Match the number of Ajtai rows

    // ME weights: w0 sums z0..z3; w1 sums z5+z7
    let w0 = vec![F::ONE, F::ONE, F::ONE, F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO];
    let w1 = vec![F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ZERO, F::ONE, F::ZERO, F::ONE];
    let y0 = dot_f_z(&w0); // 1+2+3+0 = 6
    let y1 = dot_f_z(&w1); // 1 + 2 = 3 (z5 + z7)
    let y_outputs = vec![y0, y1]; // Match the number of weight vectors

    let me = MEInstance {
        c_coords,
        y_outputs,
        r_point: vec![F::from_u64(3); 2],
        base_b: 4,
        header_digest: [0u8; 32],
    };
    let wit = MEWitness {
        z_digits: z,
        weight_vectors: vec![w0, w1],
        ajtai_rows: Some(ajtai_rows),
    };
    (me, wit)
}

#[test]
fn forged_commitment_should_fail_without_ajtai_rows() {
    // Red-team test: Verify that forged commitments are rejected even without Ajtai binding
    // If this test fails with a panic, there's a SECURITY VULNERABILITY
    
    // 1. Start from a valid instance/witness pair.
    let (mut me, mut wit) = tiny_me_instance();
    
    // 2. Drop Ajtai rows (simulates weak security configuration)
    wit.ajtai_rows = None;
    
    // 3. Forge the commitment coordinates (c_coords ≠ L·Z)
    for c in &mut me.c_coords {
        *c = *c + F::ONE;
    }
    
    // 4. Try to verify the forged proof - this MUST fail for security
    let bundle_result = compress_me_to_spartan(&me, &wit);
    
    match bundle_result {
        Ok(bundle) => {
            let verify_result = verify_me_spartan(&bundle);
            
            match verify_result {
                Ok(true) => {
                    panic!("🚨 SECURITY VULNERABILITY: Forged commitment passed verification without Ajtai binding! This should never happen.");
                }
                Ok(false) => {
                    println!("✅ Bridge correctly rejects forged commitment");
                }
                Err(e) => {
                    println!("✅ Bridge correctly fails with forged commitment: {}", e);
                }
            }
        }
        Err(e) => {
            println!("✅ SNARK generation correctly fails with forged commitment: {}", e);
        }
    }
}

#[test]
fn honest_commitment_passes_with_ajtai_rows() {
    // Control test: verify that honest commitment works
    let (me, wit) = tiny_me_instance();
    
    // Optional simple diagnostic (no D/m gymnastics needed here)
    if let Some(rows) = &wit.ajtai_rows {
        let eval = |row: &[F]| -> F {
            row.iter().zip(wit.z_digits.iter()).fold(F::ZERO, |acc, (a,&zi)| {
                let zf = if zi >= 0 { F::from_u64(zi as u64) } else { -F::from_u64((-zi) as u64) };
                acc + *a * zf
            })
        };
        for i in 0..me.c_coords.len().min(rows.len()) {
            let lhs = eval(&rows[i]);
            eprintln!("🔍 Ajtai row {i}: c = {}, dot = {}",
                      me.c_coords[i].as_canonical_u64(),
                      lhs.as_canonical_u64());
        }
    }
    
    // Keep Ajtai rows for proper binding check
    let bundle = compress_me_to_spartan(&me, &wit).expect("SNARK generation should succeed");
    let verification_result = verify_me_spartan(&bundle);
    
    match verification_result {
        Ok(true) => {
            println!("✅ Honest commitment with Ajtai rows verifies correctly");
        }
        Ok(false) => {
            panic!("Honest commitment should verify successfully");
        }
        Err(e) => {
            panic!("Honest commitment verification should not error: {}", e);
        }
    }
}

#[test]
fn forged_commitment_fails_with_ajtai_rows() {
    // Test that forged commitment fails when Ajtai rows are present
    let (mut me, wit) = tiny_me_instance();
    
    // Keep Ajtai rows (wit.ajtai_rows = Some(...))
    // Forge the commitment coordinates
    for c in &mut me.c_coords {
        *c = *c + F::ONE;
    }
    
    // This should fail because Ajtai binding checks are enforced
    let result = compress_me_to_spartan(&me, &wit);
    
    match result {
        Ok(bundle) => {
            // If SNARK generation succeeds, verification should fail
            let verification_result = verify_me_spartan(&bundle);
            match verification_result {
                Ok(false) => {
                    println!("✅ Bridge correctly rejects forged commitment when Ajtai rows are present");
                }
                Err(e) if e.to_string().contains("InvalidSumcheckProof") => {
                    println!("✅ SNARK correctly fails with InvalidSumcheckProof for forged commitment");
                }
                Ok(true) => {
                    panic!("Forged commitment should not verify when Ajtai rows are present");
                }
                Err(e) => {
                    println!("✅ Bridge correctly fails with forged commitment: {}", e);
                }
            }
        }
        Err(e) => {
            println!("✅ SNARK generation correctly fails with forged commitment when Ajtai rows are present: {}", e);
        }
    }
}

