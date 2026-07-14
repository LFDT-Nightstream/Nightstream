use neo_reductions::cuda_backend::{
    CudaReductionPhase, CudaReductionStepPlan, CudaResidencyPolicy, ResidentCudaNifsSession,
};

#[test]
fn resident_superneo_chain_keeps_repeated_loop_on_device() {
    let policy = CudaResidencyPolicy::RESIDENT_SUPERNEO_CHAIN;

    assert!(policy.setup_h2d());
    assert!(policy.final_d2h());
    assert!(!policy.repeated_loop_h2d());
    assert!(!policy.repeated_loop_d2h());
    assert!(!policy.repeated_loop_host_sync());
    assert!(policy.keeps_repeated_loop_resident());
}

#[test]
fn cuda_reduction_phase_order_matches_superneo_nifs() {
    assert_eq!(
        CudaReductionPhase::NIFS_ORDER,
        [
            CudaReductionPhase::PiCcs,
            CudaReductionPhase::PiRlc,
            CudaReductionPhase::PiDec,
        ]
    );
}

#[test]
fn resident_step_plan_is_reductions_level_and_resident() {
    let plan = CudaReductionStepPlan::RESIDENT_SUPERNEO_NIFS;

    assert_eq!(plan.phases(), CudaReductionPhase::NIFS_ORDER);
    assert!(plan.residency().keeps_repeated_loop_resident());
}

#[test]
fn resident_session_runs_pi_ccs_pi_rlc_pi_dec_before_retaining_children() {
    let mut session = MockResidentSession::default();

    session.run_resident_step(MockPiCcsInput).unwrap();

    assert_eq!(
        session.events,
        [
            MockEvent::PiCcs,
            MockEvent::PiRlc,
            MockEvent::PiDec,
            MockEvent::RetainDecChildren,
        ]
    );
}

#[test]
fn resident_session_exports_only_at_explicit_final_boundary() {
    let mut session = MockResidentSession::default();

    session.run_resident_step(MockPiCcsInput).unwrap();
    assert!(!session.exported);

    session.export_final_proof().unwrap();
    assert!(session.exported);
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct MockPiCcsInput;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MockPiCcsOutput;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MockPiRlcOutput;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MockPiDecOutput;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MockFinalProof;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MockEvent {
    PiCcs,
    PiRlc,
    PiDec,
    RetainDecChildren,
}

#[derive(Default)]
struct MockResidentSession {
    events: Vec<MockEvent>,
    exported: bool,
}

impl ResidentCudaNifsSession for MockResidentSession {
    type Error = &'static str;
    type PiCcsInput = MockPiCcsInput;
    type PiCcsOutput = MockPiCcsOutput;
    type PiRlcOutput = MockPiRlcOutput;
    type PiDecOutput = MockPiDecOutput;
    type FinalProof = MockFinalProof;

    fn launch_pi_ccs(&mut self, _input: Self::PiCcsInput) -> Result<Self::PiCcsOutput, Self::Error> {
        self.events.push(MockEvent::PiCcs);
        Ok(MockPiCcsOutput)
    }

    fn launch_pi_rlc(&mut self, _input: Self::PiCcsOutput) -> Result<Self::PiRlcOutput, Self::Error> {
        self.events.push(MockEvent::PiRlc);
        Ok(MockPiRlcOutput)
    }

    fn launch_pi_dec(&mut self, _input: Self::PiRlcOutput) -> Result<Self::PiDecOutput, Self::Error> {
        self.events.push(MockEvent::PiDec);
        Ok(MockPiDecOutput)
    }

    fn retain_dec_children(&mut self, _output: Self::PiDecOutput) -> Result<(), Self::Error> {
        self.events.push(MockEvent::RetainDecChildren);
        Ok(())
    }

    fn export_final_proof(&mut self) -> Result<Self::FinalProof, Self::Error> {
        self.exported = true;
        Ok(MockFinalProof)
    }
}
