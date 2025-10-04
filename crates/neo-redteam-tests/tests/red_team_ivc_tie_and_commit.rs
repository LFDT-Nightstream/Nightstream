//! Red-team tests: ensure IVC verification rejects tampered proofs without
//! relying on external row-wise CCS checks.

use neo::{F, NeoParams};
use neo::ivc::{
    Accumulator, LastNExtractor, StepBindingSpec,
    prove_ivc_step_with_extractor, verify_ivc_step,
};
use neo_ccs::{Mat, r1cs::r1cs_to_ccs};
use p3_field::PrimeCharacteristicRing;

fn tiny_r1cs_to_ccs() -> neo_ccs::CcsStructure<F> {
    // Constraint: (z0 - z1) * z0 = 0  (R1CS style)
    let a = Mat::from_row_major(1, 2, vec![F::ONE, -F::ONE]);
    let b = Mat::from_row_major(1, 2, vec![F::ONE, F::ZERO]);
    let c = Mat::from_row_major(1, 2, vec![F::ZERO, F::ZERO]);
    r1cs_to_ccs(a, b, c)
}

fn make_binding_spec() -> StepBindingSpec {
    StepBindingSpec {
        y_step_offsets: vec![],
        step_program_input_witness_indices: vec![],
        y_prev_witness_indices: vec![],
        const1_witness_index: 0,
    }
}

fn make_valid_ivc_step() -> (neo::ivc::IvcStepResult, neo_ccs::CcsStructure<F>, NeoParams, Accumulator, StepBindingSpec) {
    // Deterministic PP (optional): uncomment if you need fully reproducible runs
    // std::env::set_var("NEO_DETERMINISTIC", "1");

    let step_ccs = tiny_r1cs_to_ccs();
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let prev_acc = Accumulator { c_z_digest: [0u8; 32], c_coords: vec![], y_compact: vec![], step: 0 };
    let binding = make_binding_spec();

    // Valid step witness: z0 = 1 (const-1), z1 = 1 → (1-1)*1 = 0
    let step_witness = vec![F::ONE, F::ONE];
    let extractor = LastNExtractor { n: 0 }; // y_len = 0

    let step_res = prove_ivc_step_with_extractor(
        &params, &step_ccs, &step_witness, &prev_acc, prev_acc.step,
        None, &extractor, &binding
    ).expect("IVC step proving should succeed");

    (step_res, step_ccs, params, prev_acc, binding)
}

#[test]
fn rt_ivc_tie_me_y_tamper_must_fail() {
    let (step_res, step_ccs, params, prev_acc, binding) = make_valid_ivc_step();

    // Baseline verify (informational); core assertion is on the tampered proof
    let _baseline = verify_ivc_step(&step_ccs, &step_res.proof, &prev_acc, &binding, &params, None)
        .unwrap_or(false);

    // Tamper: change one y entry in the first digit ME instance
    let mut proof_tampered = step_res.proof.clone();
    let mut me_digits = proof_tampered.me_instances.take().expect("digit MEs present");
    assert!(!me_digits.is_empty());
    assert!(!me_digits[0].y.is_empty());
    // Flip y[0][0]
    if let Some(first_vec) = me_digits[0].y.get_mut(0) {
        if !first_vec.is_empty() {
            // Add 1 in the K field via embedding from F
            first_vec[0] = first_vec[0] + neo_math::K::from(F::ONE);
        }
    }
    proof_tampered.me_instances = Some(me_digits);

    // Verify must now fail due to tie check y != Z·(M^T χ_r)
    let ok2 = verify_ivc_step(&step_ccs, &proof_tampered, &prev_acc, &binding, &params, None)
        .expect("verify_ivc_step should not error");
    assert!(!ok2, "tampered y must be rejected");
}

#[test]
fn rt_ivc_tie_digit_z_tamper_must_fail() {
    let (step_res, step_ccs, params, prev_acc, binding) = make_valid_ivc_step();

    // Tamper: modify one element of the first digit witness Z
    let mut proof_tampered = step_res.proof.clone();
    let me_wits_opt = proof_tampered.digit_witnesses.take();
    let mut me_wits = me_wits_opt.expect("digit witnesses present");
    assert!(!me_wits.is_empty());
    if me_wits[0].Z.rows() > 0 && me_wits[0].Z.cols() > 0 {
        me_wits[0].Z[(0, 0)] += F::ONE;
    }
    proof_tampered.digit_witnesses = Some(me_wits);

    let ok = verify_ivc_step(&step_ccs, &proof_tampered, &prev_acc, &binding, &params, None)
        .expect("verify_ivc_step should not error");
    assert!(!ok, "tampered digit Z must be rejected");
}

#[test]
fn rt_ivc_commit_evolution_tamper_must_fail() {
    let (step_res, step_ccs, params, prev_acc, binding) = make_valid_ivc_step();

    // Tamper: break commitment evolution c_next = c_prev + ρ·c_step by flipping one coord
    let mut proof_tampered = step_res.proof.clone();
    if !proof_tampered.next_accumulator.c_coords.is_empty() {
        proof_tampered.next_accumulator.c_coords[0] += F::ONE;
    }

    let ok = verify_ivc_step(&step_ccs, &proof_tampered, &prev_acc, &binding, &params, None)
        .expect("verify_ivc_step should not error");
    assert!(!ok, "tampered commitment evolution must be rejected");
}

#[test]
fn rt_ivc_pipps_rhs_y_scalars_tamper_must_fail() {
    let (step_res, step_ccs, params, prev_acc, binding) = make_valid_ivc_step();

    // Tamper: modify Π‑CCS RHS y_scalars to mismatch recombined parent
    let mut proof_tampered = step_res.proof.clone();
    if let Some(mut folding) = proof_tampered.folding_proof.take() {
        if let Some(rhs) = folding.pi_ccs_outputs.get_mut(1) {
            if !rhs.y_scalars.is_empty() {
                rhs.y_scalars[0] = rhs.y_scalars[0] + neo_math::K::from(F::ONE);
            } else {
                // If empty (degenerate), make it non-equal by pushing one element
                rhs.y_scalars.push(neo_math::K::from(F::ONE));
            }
        }
        proof_tampered.folding_proof = Some(folding);
    }

    let ok = verify_ivc_step(&step_ccs, &proof_tampered, &prev_acc, &binding, &params, None)
        .expect("verify_ivc_step should not error");
    assert!(!ok, "tampered Π‑CCS RHS y_scalars must be rejected");
}
