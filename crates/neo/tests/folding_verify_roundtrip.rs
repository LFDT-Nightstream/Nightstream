#![cfg(test)]

use neo::{
    F, CcsStructure, NeoParams,
    StepBindingSpec, Accumulator, IvcProof, IvcStepInput, IvcChainProof,
    prove_ivc_step_chained, verify_ivc_chain_legacy, verify_ivc_step_legacy,
};
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
    let binding = StepBindingSpec {
        y_step_offsets: (1..=y_len).collect(),
        step_program_input_witness_indices: vec![],          // no app x in this trivial example
        y_prev_witness_indices: vec![],     // not needed for this test
        const1_witness_index: 0,
    };
    let y0 = vec![F::from_u64(10), F::from_u64(20)];
    // Manual chained proving (replaces ivc_chain)
    let mut acc = Accumulator { c_z_digest: [0u8;32], c_coords: vec![], y_compact: y0.clone(), step: 0 };
    let mut proofs: Vec<IvcProof> = Vec::new();
    let mut prev_me = None;
    let mut prev_me_wit = None;
    let mut prev_lhs = None;

    // Prove 3 steps with arbitrary y_step embedded at offsets [1,2]
    let steps = [
        vec![F::ONE, F::from_u64(3),  F::from_u64(5)],
        vec![F::ONE, F::from_u64(7),  F::from_u64(1)],
        vec![F::ONE, F::from_u64(11), F::from_u64(2)],
    ];
    for (i, w) in steps.into_iter().enumerate() {
        let y_step = w[1..=y_len].to_vec();
        let input = IvcStepInput {
            params: &params,
            step_ccs: &step_ccs,
            step_witness: &w,
            prev_accumulator: &acc,
            step: i as u64,
            public_input: Some(&[]),
            y_step: &y_step,
            binding_spec: &binding,
            transcript_only_app_inputs: false,
            prev_augmented_x: proofs.last().map(|p| p.public_inputs.step_augmented_public_input()),
        };
        let (res, me, wit, lhs_next) = prove_ivc_step_chained(
            input,
            prev_me.take(),
            prev_me_wit.take(),
            prev_lhs.take(),
        ).map_err(|e| anyhow::anyhow!("prove_ivc_step_chained failed: {}", e))?;
        acc = res.proof.next_accumulator.clone();
        prev_me = Some(me);
        prev_me_wit = Some(wit);
        prev_lhs = Some(lhs_next);
        proofs.push(res.proof);
    }
    let chain = IvcChainProof { steps: proofs.clone(), final_accumulator: acc.clone(), chain_length: proofs.len() as u64 };
    let initial_acc = Accumulator {
        c_z_digest: [0u8; 32],
        c_coords: vec![],
        y_compact: y0,
        step: 0,
    };

    // Verify (strict: step checks + folding)
    let ok = verify_ivc_chain_legacy(
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
            println!("  step_public_input: {:?}", step_proof.public_inputs.wrapper_public_input_x().len());
            println!("  step_rho: {}", step_proof.public_inputs.rho().as_canonical_u64());
            println!("  step_y_prev: {:?}", step_proof.public_inputs.y_prev().len());
            println!("  step_y_next: {:?}", step_proof.public_inputs.y_next().len());
                
                // Verify step (now includes folding verification)
                let step_ok = verify_ivc_step_legacy(&step_ccs, step_proof, &current_acc, &binding, &params, prev_step_x.as_deref())
                    .map_err(|e| anyhow::anyhow!("Step {} verification failed: {}", i, e))?;
                println!("  Step verification: {}", step_ok);
                
                println!("  Step {} result: {}", i, step_ok);
                if !step_ok {
                    println!("❌ Step {} failed verification", i);
                    break;
                }
                
                current_acc = step_proof.next_accumulator.clone();
                // Reconstruct the full augmented public input for the next step
                prev_step_x = Some(step_proof.public_inputs.step_augmented_public_input().to_vec());
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
    let binding = StepBindingSpec {
        y_step_offsets: vec![1],
        step_program_input_witness_indices: vec![],
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };
    let y0 = vec![F::from_u64(1)];
    // manual single-step proof
    let acc = Accumulator { c_z_digest: [0u8;32], c_coords: vec![], y_compact: y0.clone(), step: 0 };
    let w = vec![F::ONE, F::from_u64(7)];
    let y_step = w[1..=y_len].to_vec();
    let input = IvcStepInput { params: &params, step_ccs: &step_ccs, step_witness: &w, prev_accumulator: &acc, step: 0, public_input: Some(&[]), y_step: &y_step, binding_spec: &binding, transcript_only_app_inputs: false, prev_augmented_x: None };
    let (res, _, _, _) = prove_ivc_step_chained(input, None, None, None)
        .map_err(|e| anyhow::anyhow!("prove_ivc_step_chained failed: {}", e))?;
    let _acc_unused = res.proof.next_accumulator.clone();
    let mut step_proof = res.proof.clone();

    // Corrupt the RHS commitment the verifier will absorb into Pi-CCS transcript
    if let Some(fold) = &mut step_proof.folding_proof {
        if let Some(c) = fold.pi_ccs_outputs.get_mut(1) {
            if !c.c.data.is_empty() {
                c.c.data[0] += F::ONE; // flip one coordinate
            }
        }
    }

    let initial_acc = Accumulator { c_z_digest: [0u8; 32], c_coords: vec![], y_compact: y0, step: 0 };
    let ok = verify_ivc_step_legacy(
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
