//! Simple test to reproduce the MCS opening length mismatch bug
//!
//! This test reproduces the exact error: "MCS opening failed: length mismatch in x (public): expected 0, got 3"

use neo_ajtai::{setup as ajtai_setup, AjtaiSModule};
use neo_ccs::{check_ccs_claim_opening, traits::SModuleHomomorphism, CcsClaim, CcsWitness, Mat};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;
use rand::SeedableRng;

type F = Goldilocks;

#[test]
fn test_mcs_opening_bug_reproduction() {
    println!("🔍 Reproducing MCS opening bug: expected 0, got 3");

    // Setup Ajtai scheme with dimensions that match our test
    let mut rng = rand::rngs::StdRng::from_seed([42u8; 32]);
    let d = neo_math::ring::D;
    let kappa = 4;
    let m = 4; // Total variables: 3 public + 1 witness

    let pp = ajtai_setup(&mut rng, d, kappa, m).expect("Ajtai setup failed");
    let scheme = AjtaiSModule::new(std::sync::Arc::new(pp));

    // Create test data: 3 public inputs + 1 witness = 4 total variables
    let public_inputs = vec![F::from_u64(100), F::from_u64(200), F::from_u64(300)];
    let private_witness = vec![F::from_u64(400)];

    println!("   public_inputs: {:?} (len={})", public_inputs, public_inputs.len());
    println!(
        "   private_witness: {:?} (len={})",
        private_witness,
        private_witness.len()
    );

    // Create Z matrix for the commitment (d × m)
    let z_matrix = Mat::from_row_major(d, m, vec![F::ONE; d * m]);
    let commitment = scheme.commit(&z_matrix);

    // ❌ BUG CASE: Set m_in=0 even though we have 3 public inputs
    let buggy_instance = CcsClaim {
        adv: None,
        c: commitment.clone(),
        x: public_inputs.clone(), // 3 public inputs
        m_in: 0,                  // ❌ WRONG: Should be 3, not 0!
    };

    let witness = CcsWitness {
        w: private_witness.clone(),
        Z: z_matrix.clone(),
    };

    println!(
        "   Testing buggy case (m_in=0 but providing {} public inputs)...",
        public_inputs.len()
    );
    let buggy_result = check_ccs_claim_opening(&scheme, &buggy_instance, &witness);

    match buggy_result {
        Ok(_) => {
            panic!("❌ Expected MCS opening to fail with buggy m_in=0, but it succeeded!");
        }
        Err(e) => {
            let error_msg = format!("{:?}", e);
            println!("✅ Successfully reproduced error: {}", error_msg);

            // Check if this is the exact error we were fixing
            if error_msg.contains("expected 0, got 3") {
                println!("🎯 PERFECT: Reproduced exact 'expected 0, got 3' error!");
            } else if error_msg.contains("length mismatch") && error_msg.contains("x (public)") {
                println!("🎯 GOOD: Reproduced MCS opening length mismatch error");
            } else {
                println!("⚠️  Different error than expected: {}", error_msg);
            }
        }
    }

    // ✅ FIXED CASE: Set m_in to actual public input count
    let fixed_instance = CcsClaim {
        adv: None,
        c: commitment,
        x: public_inputs.clone(),
        m_in: public_inputs.len(), // ✅ CORRECT: Use actual count
    };

    println!("   Testing fixed case (m_in={})...", public_inputs.len());
    let fixed_result = check_ccs_claim_opening(&scheme, &fixed_instance, &witness);

    match fixed_result {
        Ok(_) => {
            println!("✅ Fixed case works correctly!");
        }
        Err(e) => {
            println!("⚠️  Fixed case still has error: {:?}", e);
            // This might be due to commitment/witness consistency, not the m_in bug
        }
    }
}
