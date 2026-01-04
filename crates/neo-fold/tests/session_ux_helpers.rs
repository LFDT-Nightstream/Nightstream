#![allow(non_snake_case)]

use std::sync::Arc;

use neo_ajtai::AjtaiSModule;
use neo_ccs::{r1cs_to_ccs, Mat};
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::session::{FoldingSession, NeoStep, StepArtifacts, StepSpec};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;

fn boolean_ccs(n: usize) -> neo_ccs::CcsStructure<F> {
    // z_i^2 = z_i for i=0..n-1
    let mut a = Mat::zero(n, n, F::ZERO);
    let mut b = Mat::zero(n, n, F::ZERO);
    let mut c = Mat::zero(n, n, F::ZERO);
    for i in 0..n {
        a[(i, i)] = F::ONE;
        b[(i, i)] = F::ONE;
        c[(i, i)] = F::ONE;
    }
    r1cs_to_ccs(a, b, c)
}

#[test]
fn test_session_new_ajtai_seeded_add_step_io_and_prove_and_verify() {
    let ccs = boolean_ccs(3);
    let seed = [3u8; 32];
    let mut session =
        FoldingSession::<AjtaiSModule>::new_ajtai_seeded(FoldingMode::Optimized, &ccs, seed).expect("new_ajtai_seeded");

    let x = vec![F::ONE, F::ZERO];
    let w = vec![F::ONE];
    session.add_step_io(&ccs, &x, &w).expect("add_step_io");

    let _run = session.prove_and_verify_collected(&ccs).expect("prove_and_verify_collected");
}

#[test]
fn test_session_enable_step_linking_from_step_spec_ivc_happy_path() {
    // x layout: [const1] ++ y_step ++ y_prev  (y_len=1, y_step_len=1, y_prev_len=1)
    let ccs = boolean_ccs(3);
    let mut session = FoldingSession::<AjtaiSModule>::new_ajtai(FoldingMode::Optimized, &ccs).expect("new_ajtai");

    let spec = StepSpec {
        y_len: 1,
        const1_index: 0,
        y_step_indices: vec![1],
        app_input_indices: Some(vec![2]),
        m_in: 3,
    };
    assert_eq!(spec.ivc_step_linking_pairs().expect("pairs"), vec![(1, 2)]);
    session
        .enable_step_linking_from_step_spec(&spec)
        .expect("enable_step_linking_from_step_spec");

    let x0 = vec![F::ONE, F::ZERO, F::ZERO]; // y_next=0, y_prev=0
    let x1 = vec![F::ONE, F::ONE, F::ZERO]; // y_prev=0 matches previous y_next
    session.add_step_io(&ccs, &x0, &[]).expect("step0");
    session.add_step_io(&ccs, &x1, &[]).expect("step1");

    let _run = session.prove_and_verify_collected(&ccs).expect("prove_and_verify_collected");
}

struct IvcHappyPathStepper {
    ccs: Arc<neo_ccs::CcsStructure<F>>,
}

impl NeoStep for IvcHappyPathStepper {
    type ExternalInputs = ();

    fn state_len(&self) -> usize {
        2
    }

    fn step_spec(&self) -> StepSpec {
        StepSpec {
            y_len: 2,
            const1_index: 0,
            y_step_indices: vec![1, 2],
            app_input_indices: Some(vec![3, 4]),
            m_in: 5,
        }
    }

    fn synthesize_step(&mut self, step_idx: usize, y_prev: &[F], _inputs: &Self::ExternalInputs) -> StepArtifacts {
        let (next0, next1) = match step_idx {
            0 => (F::ONE, F::ZERO),
            1 => (F::ZERO, F::ONE),
            _ => (F::ONE, F::ONE),
        };

        StepArtifacts {
            ccs: self.ccs.clone(),
            witness: vec![F::ONE, next0, next1, y_prev[0], y_prev[1]],
            public_app_inputs: vec![],
            spec: self.step_spec(),
        }
    }
}

#[test]
fn test_session_neo_step_auto_enables_step_linking_ivc_happy_path() {
    let ccs = boolean_ccs(5);
    let mut stepper = IvcHappyPathStepper { ccs: Arc::new(ccs.clone()) };

    let mut session = FoldingSession::<AjtaiSModule>::new_ajtai(FoldingMode::Optimized, &ccs).expect("new_ajtai");
    session.add_step(&mut stepper, &()).expect("add_step 0");
    session.add_step(&mut stepper, &()).expect("add_step 1");

    let _run = session.prove_and_verify_collected(&ccs).expect("prove_and_verify_collected");
}

#[test]
fn test_session_new_ajtai_seeded_is_deterministic_for_commitment() {
    let ccs = boolean_ccs(3);
    let seed = [7u8; 32];

    let mut s1 =
        FoldingSession::<AjtaiSModule>::new_ajtai_seeded(FoldingMode::Optimized, &ccs, seed).expect("new_ajtai_seeded");
    let mut s2 =
        FoldingSession::<AjtaiSModule>::new_ajtai_seeded(FoldingMode::Optimized, &ccs, seed).expect("new_ajtai_seeded");

    let x = vec![F::ONE, F::ZERO];
    let w = vec![F::ONE];
    s1.add_step_io(&ccs, &x, &w).expect("add_step_io");
    s2.add_step_io(&ccs, &x, &w).expect("add_step_io");

    let c1 = s1.mcss_public()[0].c.clone();
    let c2 = s2.mcss_public()[0].c.clone();
    assert_eq!(c1, c2, "new_ajtai_seeded should be deterministic");
}

#[test]
fn test_session_rectangular_ccs_normalization_pads_witness_and_verifies() {
    // Rectangular R1CS: n=4 constraints, m=2 variables.
    // Each constraint enforces z0^2 = z0, so z0=1 is a satisfying assignment.
    let n = 4usize;
    let m = 2usize;
    let mut a = Mat::zero(n, m, F::ZERO);
    let mut b = Mat::zero(n, m, F::ZERO);
    let mut c = Mat::zero(n, m, F::ZERO);
    for i in 0..n {
        a[(i, 0)] = F::ONE;
        b[(i, 0)] = F::ONE;
        c[(i, 0)] = F::ONE;
    }
    let ccs = r1cs_to_ccs(a, b, c);

    // new_ajtai_seeded must pick committer dims for m=max(n,m)=4 and the session must pad the witness.
    let seed = [11u8; 32];
    let mut session =
        FoldingSession::<AjtaiSModule>::new_ajtai_seeded(FoldingMode::Optimized, &ccs, seed).expect("new_ajtai_seeded");

    // Provide a witness of length m=2; the session should zero-extend to length 4 internally.
    let x = vec![F::ONE]; // z0=1
    let w = vec![F::ZERO]; // z1=0
    session.add_step_io(&ccs, &x, &w).expect("add_step_io");

    let _run = session.prove_and_verify_collected(&ccs).expect("prove_and_verify_collected");
}
