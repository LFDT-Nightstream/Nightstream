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
    let binding = neo::ivc::StepBindingSpec {
        y_step_offsets: vec![1],
        x_witness_indices: vec![],
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };

    // Build a small two-step chain via ivc_chain
    let y0 = vec![F::from_u64(10)];
    let mut state = neo::ivc_chain::State::new(params.clone(), step_ccs.clone(), y0.clone(), binding.clone())?;
    // Two steps with simple witnesses: [1, y_step]
    state = neo::ivc_chain::step(state, &[], &[F::ONE, F::from_u64(3)])?;
    state = neo::ivc_chain::step(state, &[], &[F::ONE, F::from_u64(5)])?;

    let mut chain = neo::ivc::IvcChainProof {
        steps: state.ivc_proofs.clone(),
        final_accumulator: state.accumulator.clone(),
        chain_length: state.ivc_proofs.len() as u64,
    };

    // Tamper: mutate step 1's prev_step_augmented_public_input (linking LHS)
    assert!(chain.steps.len() >= 2);
    let prev_aug = &mut chain.steps[1].prev_step_augmented_public_input;
    if !prev_aug.is_empty() { prev_aug[0] += F::ONE; } else { prev_aug.push(F::ONE); }

    // Verify chain should now fail due to linking check
    let initial_acc = neo::ivc::Accumulator { c_z_digest: [0u8;32], c_coords: vec![], y_compact: y0, step: 0 };
    match neo::ivc::verify_ivc_chain(&step_ccs, &chain, &initial_acc, &binding, &params) {
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
    let binding = neo::ivc::StepBindingSpec {
        y_step_offsets: vec![1],
        x_witness_indices: vec![],
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
    let binding = neo::ivc::StepBindingSpec {
        y_step_offsets: vec![1],
        x_witness_indices: vec![],
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };

    // Build a small two-step chain via ivc_chain
    let y0 = vec![F::from_u64(10)];
    let mut state = neo::ivc_chain::State::new(params.clone(), step_ccs.clone(), y0.clone(), binding.clone())?;
    // Two steps with simple witnesses: [1, y_step]
    state = neo::ivc_chain::step(state, &[], &[F::ONE, F::from_u64(3)])?;
    state = neo::ivc_chain::step(state, &[], &[F::ONE, F::from_u64(5)])?;

    let chain = neo::ivc::IvcChainProof {
        steps: state.ivc_proofs.clone(),
        final_accumulator: state.accumulator.clone(),
        chain_length: state.ivc_proofs.len() as u64,
    };

    // Assert positive property: for non-base step (index 1),
    // prev_step_augmented_public_input must equal previous step's step_augmented_public_input
    assert!(chain.steps.len() >= 2, "need at least 2 steps for linkage check");
    let prev_aug = &chain.steps[0].step_augmented_public_input;
    let lhs = &chain.steps[1].prev_step_augmented_public_input;
    assert_eq!(lhs, prev_aug, "LHS augmented x must equal previous step's augmented x");

    // Verify the entire chain passes with strict verification
    let initial_acc = neo::ivc::Accumulator { c_z_digest: [0u8; 32], c_coords: vec![], y_compact: y0, step: 0 };
    match neo::ivc::verify_ivc_chain(&step_ccs, &chain, &initial_acc, &binding, &params) {
        Ok(ok) => assert!(ok, "strict chain verification should succeed for matched linkage"),
        Err(e) => panic!("verification failed unexpectedly: {}", e),
    }
    Ok(())
}
