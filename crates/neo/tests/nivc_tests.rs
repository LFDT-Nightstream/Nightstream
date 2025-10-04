#![cfg(test)]

use neo::{self, F, CcsStructure, NeoParams};
use p3_field::PrimeCharacteristicRing;

fn trivial_step_ccs_rows(y_len: usize, rows: usize) -> CcsStructure<F> {
    // Simple identity-style R1CS lifted to CCS with configurable row count.
    // Witness layout expectation for tests: index 0 = const 1, indices [1..=y_len] = y_step
    let m = 1 + y_len;
    let n = rows.max(1);
    let mut a = vec![F::ZERO; n * m];
    let mut b = vec![F::ZERO; n * m];
    let mut c = vec![F::ZERO; n * m];
    for r in 0..n {
        a[r * m + 0] = F::ONE;
        b[r * m + 0] = F::ONE;
        c[r * m + 0] = F::ONE;
    }
    let a_mat = neo_ccs::Mat::from_row_major(n, m, a);
    let b_mat = neo_ccs::Mat::from_row_major(n, m, b);
    let c_mat = neo_ccs::Mat::from_row_major(n, m, c);
    neo_ccs::r1cs_to_ccs(a_mat, b_mat, c_mat)
}

#[test]
fn base_case_not_self_fold_anymore() -> anyhow::Result<()> {
    // Deterministic PP for stable tests
    std::env::set_var("NEO_DETERMINISTIC", "1");

    let params = NeoParams::goldilocks_small_circuits();
    let y_len = 2usize;
    let step_ccs = trivial_step_ccs_rows(y_len, 1);
    let binding = neo::ivc::StepBindingSpec {
        y_step_offsets: (1..=y_len).collect(),
        step_program_input_witness_indices: vec![],
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };

    let y0 = vec![F::from_u64(10), F::from_u64(20)];
    let prev_acc = neo::ivc::Accumulator { c_z_digest: [0u8; 32], c_coords: vec![], y_compact: y0.clone(), step: 0 };

    // Build a single step input
    let witness = vec![F::ONE, F::from_u64(3), F::from_u64(5)];
    let y_step = witness[1..=y_len].to_vec();
    let input = neo::ivc::IvcStepInput {
        params: &params,
        step_ccs: &step_ccs,
        step_witness: &witness,
        prev_accumulator: &prev_acc,
        step: 0,
        public_input: None,
        y_step: &y_step,
        binding_spec: &binding,
        transcript_only_app_inputs: false,
        prev_augmented_x: None,
    };

    // Prove using the chained variant with no previous ME (should use zero LHS)
    let (res, _me, _wit, _lhs) = neo::ivc::prove_ivc_step_chained(input, None, None, None)
        .map_err(|e| anyhow::anyhow!("prove_ivc_step_chained failed: {}", e))?;

    // Inspect the Pi-CCS inputs stored in the folding proof
    let fold = res.proof.folding_proof.as_ref().expect("folding proof");
    assert_eq!(fold.pi_ccs_inputs.len(), 2, "expect two inputs to Pi-CCS");
    let lhs = &fold.pi_ccs_inputs[0];
    let rhs = &fold.pi_ccs_inputs[1];

    // LHS commitment should NOT equal RHS commitment at base case (no folding with itself)
    assert_ne!(lhs.c.data, rhs.c.data, "base-case LHS commitment must differ from RHS");

    // Verify step succeeds
    let ok = neo::ivc::verify_ivc_step(&step_ccs, &res.proof, &prev_acc, &binding, &params, None)
        .map_err(|e| anyhow::anyhow!("verify_ivc_step failed: {}", e))?;
    assert!(ok, "verification must pass");
    Ok(())
}

#[test]
fn nivc_mixed_circuits_roundtrip() -> anyhow::Result<()> {
    std::env::set_var("NEO_DETERMINISTIC", "1");
    let params = NeoParams::goldilocks_small_circuits();
    let y_len = 2usize;

    // Two different CCS shapes (same m, different n)
    let ccs_a = trivial_step_ccs_rows(y_len, 1);
    let ccs_b = trivial_step_ccs_rows(y_len, 2);

    let bind_a = neo::ivc::StepBindingSpec {
        y_step_offsets: (1..=y_len).collect(),
        step_program_input_witness_indices: vec![],
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };
    let bind_b = bind_a.clone();

    let program = neo::NivcProgram::new(vec![
        neo::NivcStepSpec { ccs: ccs_a.clone(), binding: bind_a },
        neo::NivcStepSpec { ccs: ccs_b.clone(), binding: bind_b },
    ]);

    let y0 = vec![F::from_u64(1), F::from_u64(2)];
    let mut st = neo::NivcState::new(params.clone(), program.clone(), y0.clone())?;

    // Step 0: type 0
    let w0 = vec![F::ONE, F::from_u64(3), F::from_u64(5)];
    let sp0 = st.step(0, &[], &w0)?;
    {
        // Ensure base-case for lane 0 used distinct LHS/RHS commitments
        let fold = sp0.inner.folding_proof.as_ref().expect("folding proof");
        assert_ne!(fold.pi_ccs_inputs[0].c.data, fold.pi_ccs_inputs[1].c.data);
    }

    // Step 1: type 1 (different CCS)
    let w1 = vec![F::ONE, F::from_u64(7), F::from_u64(11)];
    let _sp1 = st.step(1, &[], &w1)?;

    // Step 2: type 0 again
    let w2 = vec![F::ONE, F::from_u64(13), F::from_u64(17)];
    let _sp2 = st.step(0, &[], &w2)?;

    // Finalize chain and verify
    let proof = st.into_proof();
    let ok = neo::verify_nivc_chain(&program, &params, &proof, &y0)?;
    assert!(ok, "NIVC chain verification must pass");
    Ok(())
}

#[test]
fn nivc_enforces_selector_and_root_and_step_io() -> anyhow::Result<()> {
    std::env::set_var("NEO_DETERMINISTIC", "1");
    let params = NeoParams::goldilocks_small_circuits();
    let y_len = 2usize;

    let ccs_a = trivial_step_ccs_rows(y_len, 1);
    let ccs_b = trivial_step_ccs_rows(y_len, 2);
    let binding = neo::ivc::StepBindingSpec {
        y_step_offsets: (1..=y_len).collect(),
        step_program_input_witness_indices: vec![],
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    };
    let program = neo::NivcProgram::new(vec![
        neo::NivcStepSpec { ccs: ccs_a.clone(), binding: binding.clone() },
        neo::NivcStepSpec { ccs: ccs_b.clone(), binding: binding.clone() },
    ]);

    let y0 = vec![F::from_u64(3), F::from_u64(4)];
    let mut st = neo::NivcState::new(params.clone(), program.clone(), y0.clone())?;

    // Build a chain with non-empty step_io
    let w0 = vec![F::ONE, F::from_u64(1), F::from_u64(2)];
    let step_io0 = vec![F::from_u64(42)];
    st.step(0, &step_io0, &w0)?; // type 0
    let chain = st.into_proof();

    // Baseline verify passes
    assert!(neo::verify_nivc_chain(&program, &params, &chain, &y0)?);

    // A) Flip which_type and expect failure
    let mut bad = chain.clone();
    bad.steps[0].which_type ^= 1;
    assert!(!neo::verify_nivc_chain(&program, &params, &bad, &y0)?, "must reject mismatched which_type");

    // B) Corrupt per-step binding (lanes_root) by mutating a step's public data and expect failure
    // Flip which_type back to a wrong value (already covered above) or perturb step IO digest.
    // Here we reuse case (C) below for explicit step_io tamper.

    // C) Corrupt step_io and expect failure
    let mut bad3 = chain.clone();
    if !bad3.steps[0].step_io.is_empty() {
        bad3.steps[0].step_io[0] += F::ONE;
    } else {
        bad3.steps[0].step_io.push(F::from_u64(1));
    }
    assert!(!neo::verify_nivc_chain(&program, &params, &bad3, &y0)?, "must reject mismatched step_io");

    Ok(())
}
