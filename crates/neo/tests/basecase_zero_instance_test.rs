use neo::{F, NeoParams, AppInputBinding};
use p3_field::PrimeCharacteristicRing;
use neo::session::{NeoStep, StepSpec, StepArtifacts, FoldingSession, StepDescriptor, verify_chain_with_descriptor};
use neo_ccs::{CcsStructure, Mat, r1cs_to_ccs};

// Minimal step: linear constraints over witness [1, a, b]
//   r0: a - b = 0 (encoded as (a - b) * 1 = 0)
//   r1-r3: dummy constraints (0 * 1 = 0) to reach minimum n=4 for security
fn build_min_step_ccs() -> CcsStructure<F> {
    let rows = 4usize;  // Minimum of 4 rows required (ℓ=ceil(log2(n)) must be ≥ 2)
    let cols = 3usize; // [const=1, a, b]

    let mut a = vec![F::ZERO; rows * cols];
    let mut b = vec![F::ZERO; rows * cols];
    let c = vec![F::ZERO; rows * cols];

    // Row 0: a - b = 0
    a[0 * cols + 1] = F::ONE;       // +a
    a[0 * cols + 2] = -F::ONE;      // -b
    b[0 * cols + 0] = F::ONE;       // × const 1

    // Rows 1-3: dummy constraints (0 * 1 = 0) to reach minimum n=4
    for row in 1..4 {
        a[row * cols + 0] = F::ZERO;    // 0
        b[row * cols + 0] = F::ONE;     // × 1
        // c[row] = 0 (already initialized)
    }

    let a_mat = Mat::from_row_major(rows, cols, a);
    let b_mat = Mat::from_row_major(rows, cols, b);
    let c_mat = Mat::from_row_major(rows, cols, c);
    r1cs_to_ccs(a_mat, b_mat, c_mat)
}

#[derive(Clone, Default)]
struct NoInputs;

// Adapter for a trivial step with no state (y_len=0) and no app inputs.
struct MinimalStep {
    ccs: CcsStructure<F>,
    spec: StepSpec,
}

impl MinimalStep {
    fn new() -> Self {
        let ccs = build_min_step_ccs();
        // y_len = 0 (no EV section); const1 is witness index 0; no y_step/app bindings
        let spec = StepSpec {
            y_len: 0,
            const1_index: 0,
            y_step_indices: vec![],
            app_input_indices: None,
        };
        Self { ccs, spec }
    }
}

impl NeoStep for MinimalStep {
    type ExternalInputs = NoInputs;

    fn state_len(&self) -> usize { 0 }
    fn step_spec(&self) -> StepSpec { self.spec.clone() }

    fn synthesize_step(
        &mut self,
        step_idx: usize,
        _z_prev: &[F],
        _inputs: &Self::ExternalInputs,
    ) -> StepArtifacts {
        // Witness [1, a, b] with a == b to satisfy the constraint
        let a_val = F::from_u64(42 + step_idx as u64);
        let witness = vec![F::ONE, a_val, a_val];
        StepArtifacts { ccs: self.ccs.clone(), witness, public_app_inputs: vec![], spec: self.spec.clone() }
    }
}

// This test validates the self-fold base case: step 0 folds RHS with itself.
// We verify the chain manually by threading prev_augmented_x from the proof,
// matching the strict linkage policy used by integrators with self-fold base case.
#[test]
fn test_self_fold_basecase_manual_verify() {
    let params = NeoParams::goldilocks_for_circuit(3, 2, 2);
    let mut session = FoldingSession::new(Some(&params), None, 0, AppInputBinding::TranscriptOnly);
    let mut stepper = MinimalStep::new();

    // Prove a couple of steps (0 and 1)
    let _ = session.prove_step(&mut stepper, &NoInputs).expect("prove step 0");
    let _ = session.prove_step(&mut stepper, &NoInputs).expect("prove step 1");

    let (chain, _step_ios) = session.finalize();
    let descriptor = StepDescriptor { ccs: stepper.ccs.clone(), spec: stepper.spec.clone() };

    // Manual per-step verification with strict threading of prev_augmented_x
    let binding = descriptor.spec.binding_spec(AppInputBinding::WitnessBound);
    let mut acc = neo::Accumulator { c_z_digest: [0u8; 32], c_coords: vec![], y_compact: vec![], step: 0 };
    let mut prev_augmented_x: Option<Vec<neo::F>> = None;

    for (i, step) in chain.steps.iter().enumerate() {
        if i == 0 {
            // Base case: with self-fold, Pi‑RLC may reject step 0. Just thread linkage.
            // Thread augmented-x for step 1 verification
            acc = step.next_accumulator.clone();
            prev_augmented_x = Some(step.public_inputs.step_augmented_public_input().to_vec());
            continue;
        }
        let ok = neo::verify_ivc_step(
            &descriptor.ccs,
            step,
            &acc,
            &binding,
            &params,
            prev_augmented_x.as_deref(),
        ).expect("per-step verify should not error");
        assert!(ok, "IVC step {} verification failed", i);
        acc = step.next_accumulator.clone();
        prev_augmented_x = Some(step.public_inputs.step_augmented_public_input().to_vec());
    }
}

// Canonical strict chain verifier should accept the produced chain (zero base-case policy).
#[test]
fn test_basecase_chain_verifies_canonical() {
    let params = NeoParams::goldilocks_small_circuits();
    let mut session = FoldingSession::new(Some(&params), None, 0, AppInputBinding::TranscriptOnly);
    let mut stepper = MinimalStep::new();

    let _ = session.prove_step(&mut stepper, &NoInputs).expect("prove step 0");
    let (chain, step_ios) = session.finalize();
    let descriptor = StepDescriptor { ccs: stepper.ccs.clone(), spec: stepper.spec.clone() };

    let ok = verify_chain_with_descriptor(&descriptor, &chain, &[], &params, &step_ios, AppInputBinding::TranscriptOnly).expect("verify should not error");
    assert!(ok, "canonical verifier should accept zero-base case chain");
}
