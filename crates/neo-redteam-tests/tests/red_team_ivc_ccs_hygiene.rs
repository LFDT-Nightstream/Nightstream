//! Red-team: Π‑CCS hygiene & shape guards must fail closed.

use neo::{F, NeoParams};
use neo::ivc::{Accumulator, LastNExtractor, StepBindingSpec,
    prove_ivc_step_with_extractor, verify_ivc_step};
use neo_ccs::{Mat, r1cs::r1cs_to_ccs};
use p3_field::PrimeCharacteristicRing;

fn tiny_r1cs_to_ccs() -> neo_ccs::CcsStructure<F> {
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

fn make_valid() -> (neo::ivc::IvcStepResult, neo_ccs::CcsStructure<F>, NeoParams, Accumulator, StepBindingSpec) {
    let step_ccs = tiny_r1cs_to_ccs();
    let params = NeoParams::goldilocks_autotuned_s2(3, 2, 2);
    let prev_acc = Accumulator { c_z_digest: [0u8; 32], c_coords: vec![], y_compact: vec![], step: 0 };
    let binding = make_binding_spec();
    let witness = vec![F::ONE, F::ONE];
    let extractor = LastNExtractor { n: 0 };

    let step_res = prove_ivc_step_with_extractor(
        &params, &step_ccs, &witness, &prev_acc, prev_acc.step, None, &extractor, &binding
    ).expect("prove");

    (step_res, step_ccs, params, prev_acc, binding)
}

#[test]
fn hygiene_rho_outputs_count_mismatch_is_error() {
    let (step_res, step_ccs, params, prev_acc, binding) = make_valid();
    let mut proof = step_res.proof.clone();
    let mut folding = proof.folding_proof.take().expect("folding");
    assert!(!folding.pi_rlc_proof.rho_elems.is_empty());
    folding.pi_rlc_proof.rho_elems.pop(); // mismatch
    proof.folding_proof = Some(folding);

    let res = verify_ivc_step(&step_ccs, &proof, &prev_acc, &binding, &params, None);
    assert!(res.is_err(), "expected Err(..) for |ρ| != |outputs|, got {:?}", res);
}

#[test]
fn hygiene_inconsistent_t_across_outputs_is_error() {
    let (step_res, step_ccs, params, prev_acc, binding) = make_valid();
    let mut proof = step_res.proof.clone();
    let mut folding = proof.folding_proof.take().expect("folding");
    assert!(!folding.pi_ccs_outputs.is_empty());
    folding.pi_ccs_outputs[0].y.push(vec![neo_math::K::ZERO; neo_math::D]); // change t for first output
    proof.folding_proof = Some(folding);

    let res = verify_ivc_step(&step_ccs, &proof, &prev_acc, &binding, &params, None);
    assert!(res.is_err(), "expected Err(..) for inconsistent t, got {:?}", res);
}

#[test]
fn hygiene_y_vector_wrong_length_is_error() {
    let (step_res, step_ccs, params, prev_acc, binding) = make_valid();
    let mut proof = step_res.proof.clone();
    let mut folding = proof.folding_proof.take().expect("folding");
    assert!(!folding.pi_ccs_outputs[0].y.is_empty());
    assert_eq!(folding.pi_ccs_outputs[0].y[0].len(), neo_math::D);
    folding.pi_ccs_outputs[0].y[0].pop(); // y[j].len() = D-1
    proof.folding_proof = Some(folding);

    let res = verify_ivc_step(&step_ccs, &proof, &prev_acc, &binding, &params, None);
    assert!(res.is_err(), "expected Err(..) for |y[j]| != D, got {:?}", res);
}
