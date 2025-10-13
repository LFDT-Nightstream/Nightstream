#![cfg(test)]

use neo::{self, F, CcsStructure, NeoParams};
use p3_field::PrimeCharacteristicRing;

fn trivial_step_ccs(y_len: usize) -> CcsStructure<F> {
    // Simple CCS: identity 1*1=1 constraint with witness layout:
    // [1, y_step[0..y_len]] so we can bind y_step from the witness tail.
    let rows = 1usize;
    let cols = 1 + y_len;
    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let mut c = vec![F::ZERO; rows * cols];
    a[0] = F::ONE; b[0] = F::ONE; c[0] = F::ONE;
    let a_mat = neo_ccs::Mat::from_row_major(rows, cols, a);
    let b_mat = neo_ccs::Mat::from_row_major(rows, cols, b);
    let c_mat = neo_ccs::Mat::from_row_major(rows, cols, c);
    neo_ccs::r1cs_to_ccs(a_mat, b_mat, c_mat)
}

#[test]
fn ivc_linking_rejects_mismatched_prev_augmented_x() -> anyhow::Result<()> {
    std::env::set_var("NEO_DETERMINISTIC", "1");
    let params = NeoParams::goldilocks_small_circuits();
    let y_len = 1usize;
    let step_ccs = trivial_step_ccs(y_len);
    let binding = neo::StepBindingSpec {
        y_step_offsets: vec![1],
        step_program_input_witness_indices: vec![],
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };

    // Build a small two-step chain via manual chained proving
    let y0 = vec![F::from_u64(10)];
    let mut acc = neo::Accumulator { c_z_digest: [0u8;32], c_coords: vec![], y_compact: y0.clone(), step: 0 };
    let mut proofs: Vec<neo::IvcProof> = Vec::new();
    let mut prev_lhs: Option<(neo_ccs::McsInstance<neo_ajtai::Commitment, F>, neo_ccs::McsWitness<F>)> = None;
    for (i, w) in [
        vec![F::ONE, F::from_u64(3)],
        vec![F::ONE, F::from_u64(5)],
    ].into_iter().enumerate() {
        let y_step = w[1..=y_len].to_vec();
        let input = neo::IvcStepInput { params: &params, step_ccs: &step_ccs, step_witness: &w, prev_accumulator: &acc, step: i as u64, public_input: Some(&[]), y_step: &y_step, binding_spec: &binding, transcript_only_app_inputs: false, prev_augmented_x: proofs.last().map(|p| p.public_inputs.step_augmented_public_input()) };
        let (res, _me, _wit, lhs_next) = neo::prove_ivc_step_chained(input, None, None, prev_lhs.take())
            .map_err(|e| anyhow::anyhow!("prove_ivc_step_chained failed: {}", e))?;
        acc = res.proof.next_accumulator.clone();
        proofs.push(res.proof);
        prev_lhs = Some(lhs_next);
    }

    let mut chain = neo::IvcChainProof { steps: proofs.clone(), final_accumulator: acc.clone(), chain_length: proofs.len() as u64 };

    // Tamper: mutate step 1's prev_step_augmented_public_input (linking LHS)
    assert!(chain.steps.len() >= 2);
    let prev_aug = &mut chain.steps[1].prev_step_augmented_public_input;
    if !prev_aug.is_empty() { prev_aug[0] += F::ONE; } else { prev_aug.push(F::ONE); }

    // Verify chain should now fail due to linking check
    let initial_acc = neo::Accumulator { c_z_digest: [0u8;32], c_coords: vec![], y_compact: y0, step: 0 };
    match neo::verify_ivc_chain_legacy(&step_ccs, &chain, &initial_acc, &binding, &params) {
        Ok(ok) => assert!(!ok, "linking violation should be rejected"),
        Err(_) => { /* also acceptable: verifier detected linking failure */ }
    }
    Ok(())
}

#[test]
fn nivc_lane_local_linking_rejects_mismatch() -> anyhow::Result<()> {
    std::env::set_var("NEO_DETERMINISTIC", "1");
    let params = NeoParams::goldilocks_small_circuits();
    let y_len = 1usize;
    let ccs = trivial_step_ccs(y_len);
    let binding = neo::StepBindingSpec {
        y_step_offsets: vec![1],
        step_program_input_witness_indices: vec![],
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };

    let program = neo::NivcProgram::new(vec![neo::NivcStepSpec { ccs: ccs.clone(), binding }]);
    let y0 = vec![F::from_u64(7)];
    let mut st = neo::NivcState::new(params.clone(), program.clone(), y0.clone())?;

    // Two steps on the same lane (0)
    st.step(0, &[], &[F::ONE, F::from_u64(2)])?;
    st.step(0, &[], &[F::ONE, F::from_u64(4)])?;
    let mut chain = st.into_proof();

    // Tamper: mutate second step's prev_step_augmented_public_input
    assert!(chain.steps.len() >= 2);
    let prev_aug2 = &mut chain.steps[1].inner.prev_step_augmented_public_input;
    if !prev_aug2.is_empty() { prev_aug2[0] += F::ONE; } else { prev_aug2.push(F::ONE); }

    // Verify NIVC chain should fail due to lane-local linking
    match neo::verify_nivc_chain(&program, &params, &chain, &y0) {
        Ok(ok) => assert!(!ok, "lane-local linking violation should be rejected"),
        Err(_) => { /* acceptable: verifier errored on linking mismatch */ }
    }
    Ok(())
}

#[test]
fn ivc_linking_accepts_matched_prev_augmented_x() -> anyhow::Result<()> {
    // Deterministic setup to stabilize PP and transcripts
    std::env::set_var("NEO_DETERMINISTIC", "1");
    let params = NeoParams::goldilocks_small_circuits();
    let y_len = 1usize;
    let step_ccs = trivial_step_ccs(y_len);
    let binding = neo::StepBindingSpec {
        y_step_offsets: vec![1],
        step_program_input_witness_indices: vec![],
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };

    // Build a small two-step chain via manual chained proving
    let y0 = vec![F::from_u64(10)];
    let mut acc = neo::Accumulator { c_z_digest: [0u8;32], c_coords: vec![], y_compact: y0.clone(), step: 0 };
    let mut proofs: Vec<neo::IvcProof> = Vec::new();
    let mut prev_lhs: Option<(neo_ccs::McsInstance<neo_ajtai::Commitment, F>, neo_ccs::McsWitness<F>)> = None;
    for (i, w) in [
        vec![F::ONE, F::from_u64(3)],
        vec![F::ONE, F::from_u64(5)],
    ].into_iter().enumerate() {
        let y_step = w[1..=y_len].to_vec();
        let input = neo::IvcStepInput { params: &params, step_ccs: &step_ccs, step_witness: &w, prev_accumulator: &acc, step: i as u64, public_input: Some(&[]), y_step: &y_step, binding_spec: &binding, transcript_only_app_inputs: false, prev_augmented_x: proofs.last().map(|p| p.public_inputs.step_augmented_public_input()) };
        let (res, _me, _wit, lhs_next) = neo::prove_ivc_step_chained(input, None, None, prev_lhs.take())
            .map_err(|e| anyhow::anyhow!("prove_ivc_step_chained failed: {}", e))?;
        acc = res.proof.next_accumulator.clone();
        proofs.push(res.proof);
        prev_lhs = Some(lhs_next);
    }

    let chain = neo::IvcChainProof { steps: proofs.clone(), final_accumulator: acc.clone(), chain_length: proofs.len() as u64 };

    // Assert positive property: for non-base step (index 1),
    // prev_step_augmented_public_input must equal previous step's step_augmented_public_input
    assert!(chain.steps.len() >= 2, "need at least 2 steps for linkage check");
    let prev_aug = chain.steps[0].public_inputs.step_augmented_public_input();
    let lhs = &chain.steps[1].prev_step_augmented_public_input;
    assert_eq!(lhs, prev_aug, "LHS augmented x must equal previous step's augmented x");

    // Verify the entire chain passes with strict verification
    let initial_acc = neo::Accumulator { c_z_digest: [0u8; 32], c_coords: vec![], y_compact: y0, step: 0 };
    match neo::verify_ivc_chain_legacy(&step_ccs, &chain, &initial_acc, &binding, &params) {
        Ok(ok) => assert!(ok, "strict chain verification should succeed for matched linkage"),
        Err(e) => panic!("verification failed unexpectedly: {}", e),
    }
    Ok(())
}
