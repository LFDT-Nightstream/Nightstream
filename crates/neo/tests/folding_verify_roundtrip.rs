#![cfg(test)]

use neo::{ivc, ivc_chain};
use neo::{F, CcsStructure, NeoParams};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

fn trivial_step_ccs(y_len: usize) -> CcsStructure<F> {
    // Simple identity CCS with (m = 1 + y_len) so witness has:
    //  - index 0 = const 1
    //  - indices [1..=y_len] = y_step we bind into the augmentation
    let m = 1 + y_len;
    let rows = 1; // At least one constraint for valid CCS
    
    // Create identity constraint: 1 * 1 = 1 (always satisfied)
    let mut a_data = vec![F::ZERO; rows * m];
    let mut b_data = vec![F::ZERO; rows * m];
    let mut c_data = vec![F::ZERO; rows * m];
    
    // Row 0: constraint "1 * 1 = 1" (always true)
    a_data[0] = F::ONE;  // A[0,0] = 1 (constant column)
    b_data[0] = F::ONE;  // B[0,0] = 1 (constant column)
    c_data[0] = F::ONE;  // C[0,0] = 1 (result)
    
    let a = neo_ccs::Mat::from_row_major(rows, m, a_data);
    let b = neo_ccs::Mat::from_row_major(rows, m, b_data);
    let c = neo_ccs::Mat::from_row_major(rows, m, c_data);
    
    neo_ccs::r1cs_to_ccs(a, b, c)
}

#[test]
fn fold_roundtrip_chain_ok() -> anyhow::Result<()> {
    // Arrange
    std::env::set_var("NEO_DETERMINISTIC", "1"); // avoid RNG-induced PP mismatches
    let params = NeoParams::goldilocks_small_circuits();
    let y_len = 2usize;
    let step_ccs = trivial_step_ccs(y_len);
    let binding = ivc::StepBindingSpec {
        y_step_offsets: (1..=y_len).collect(),
        x_witness_indices: vec![],          // no app x in this trivial example
        y_prev_witness_indices: vec![],     // not needed for this test
        const1_witness_index: 0,
    };
    let y0 = vec![F::from_u64(10), F::from_u64(20)];
    let mut state = ivc_chain::State::new(params.clone(), step_ccs.clone(), y0.clone(), binding.clone())?;

    // Prove 3 steps with arbitrary y_step embedded at offsets [1,2]
    let steps = [
        vec![F::ONE, F::from_u64(3),  F::from_u64(5)],
        vec![F::ONE, F::from_u64(7),  F::from_u64(1)],
        vec![F::ONE, F::from_u64(11), F::from_u64(2)],
    ];
    for w in steps {
        state = ivc_chain::step(state, &[], &w)?;
    }
    let chain = ivc::IvcChainProof {
        steps: state.ivc_proofs.clone(),
        final_accumulator: state.accumulator.clone(),
        chain_length: state.ivc_proofs.len() as u64,
    };
    let initial_acc = ivc::Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: y0,
        step: 0,
    };

    // Verify (strict: step checks + folding)
    let ok = ivc::verify_ivc_chain(
        &step_ccs,
        &chain,
        &initial_acc,
        &binding,
        &params,
    ).map_err(|e| anyhow::anyhow!("Verification failed: {}", e))?;
    
    if !ok {
        // Debug: Let's try verifying each step individually to see where it fails
        let mut current_acc = initial_acc.clone();
        let mut prev_step_x: Option<Vec<F>> = None;
        
        for (i, step_proof) in chain.steps.iter().enumerate() {
            println!("🔍 DEBUG: Verifying step {} individually", i);
            println!("  prev_step_x: {:?}", prev_step_x.as_ref().map(|x| x.len()));
            println!("  step_public_input: {:?}", step_proof.step_public_input.len());
            println!("  step_rho: {}", step_proof.step_rho.as_canonical_u64());
            println!("  step_y_prev: {:?}", step_proof.step_y_prev.len());
            println!("  step_y_next: {:?}", step_proof.step_y_next.len());
                
                // Verify step (now includes folding verification)
                let step_ok = ivc::verify_ivc_step(&step_ccs, step_proof, &current_acc, &binding, &params, prev_step_x.as_deref())
                    .map_err(|e| anyhow::anyhow!("Step {} verification failed: {}", i, e))?;
                println!("  Step verification: {}", step_ok);
                
                println!("  Step {} result: {}", i, step_ok);
                if !step_ok {
                    println!("❌ Step {} failed verification", i);
                    break;
                }
                
                current_acc = step_proof.next_accumulator.clone();
                // Reconstruct the full augmented public input for the next step
                prev_step_x = Some(step_proof.step_augmented_public_input.clone());
        }
    }
    
    assert!(ok, "strict chain verification should pass");
    Ok(())
}

#[test]
fn fold_roundtrip_rejects_mutated_rhs_commitment() -> anyhow::Result<()> {
    std::env::set_var("NEO_DETERMINISTIC", "1");
    let params = NeoParams::goldilocks_small_circuits();
    let y_len = 1usize;
    let step_ccs = trivial_step_ccs(y_len);
    let binding = ivc::StepBindingSpec {
        y_step_offsets: vec![1],
        x_witness_indices: vec![],
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };
    let y0 = vec![F::from_u64(1)];
    let mut state = ivc_chain::State::new(params.clone(), step_ccs.clone(), y0, binding.clone())?;
    state = ivc_chain::step(state, &[], &[F::ONE, F::from_u64(7)])?;
    let mut step_proof = state.ivc_proofs[0].clone();

    // Corrupt the RHS commitment the verifier will absorb into Pi-CCS transcript
    if let Some(fold) = &mut step_proof.folding_proof {
        if let Some(c) = fold.pi_ccs_outputs.get_mut(1) {
            if !c.c.data.is_empty() {
                c.c.data[0] += F::ONE; // flip one coordinate
            }
        }
    }

    let initial_acc = ivc::Accumulator { c_z_digest: [0u8; 32], c_coords: vec![], y_compact: vec![F::ZERO], step: 0 };
    let ok = ivc::verify_ivc_step(
        &step_ccs,
        &step_proof,
        &initial_acc,
        &binding,
        &params,
        None,
    ).map_err(|e| anyhow::anyhow!("Verification failed: {}", e))?;
    assert!(!ok, "strict step verification must fail if commitments don't match");
    Ok(())
}
