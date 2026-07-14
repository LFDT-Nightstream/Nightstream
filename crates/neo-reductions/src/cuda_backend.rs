//! CUDA-facing reduction backend contracts.
//!
//! This module owns the crate-level shape of a CUDA reduction backend. It does
//! not own CUDA kernels, buffers, cuda-oxide dependencies, or host orchestration.
//! A CUDA implementation lives in a prover crate and implements these contracts
//! around the same reduction chain as the CPU engines:
//!
//! ```text
//! Pi_CCS.prove -> Pi_RLC.prove -> Pi_DEC.prove
//! ```
//!
//! The important boundary is reductions-level ownership. `neo-fold-clean` may
//! orchestrate IVC state, but it should not be the place where CUDA substitutes
//! for the SuperNeo reductions.

/// SuperNeo NIFS.P reduction phases in the order a resident CUDA backend runs
/// them for each fold step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaReductionPhase {
    PiCcs,
    PiRlc,
    PiDec,
}

impl CudaReductionPhase {
    pub const NIFS_ORDER: [Self; 3] = [Self::PiCcs, Self::PiRlc, Self::PiDec];
}

/// A resident CUDA step plan. The phase order is intentionally fixed to the
/// SuperNeo prover order; a backend can choose different kernels internally,
/// but it may not expose a different protocol order at this boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaReductionStepPlan {
    phases: [CudaReductionPhase; 3],
    residency: CudaResidencyPolicy,
}

impl CudaReductionStepPlan {
    pub const RESIDENT_SUPERNEO_NIFS: Self = Self {
        phases: CudaReductionPhase::NIFS_ORDER,
        residency: CudaResidencyPolicy::RESIDENT_SUPERNEO_CHAIN,
    };

    pub fn phases(self) -> [CudaReductionPhase; 3] {
        self.phases
    }

    pub fn residency(self) -> CudaResidencyPolicy {
        self.residency
    }
}

/// Host/device transfer policy for a resident CUDA NIFS session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CudaResidencyPolicy {
    setup_h2d: bool,
    repeated_loop_h2d: bool,
    repeated_loop_d2h: bool,
    repeated_loop_host_sync: bool,
    final_d2h: bool,
}

impl CudaResidencyPolicy {
    /// The intended SuperNeo CUDA policy: upload static/input state once, keep
    /// Pi_CCS -> Pi_RLC -> Pi_DEC and carried children on device across repeated
    /// folds, then export canonical proof material at the end.
    pub const RESIDENT_SUPERNEO_CHAIN: Self = Self {
        setup_h2d: true,
        repeated_loop_h2d: false,
        repeated_loop_d2h: false,
        repeated_loop_host_sync: false,
        final_d2h: true,
    };

    pub fn setup_h2d(self) -> bool {
        self.setup_h2d
    }

    pub fn repeated_loop_h2d(self) -> bool {
        self.repeated_loop_h2d
    }

    pub fn repeated_loop_d2h(self) -> bool {
        self.repeated_loop_d2h
    }

    pub fn repeated_loop_host_sync(self) -> bool {
        self.repeated_loop_host_sync
    }

    pub fn final_d2h(self) -> bool {
        self.final_d2h
    }

    pub fn keeps_repeated_loop_resident(self) -> bool {
        !self.repeated_loop_h2d && !self.repeated_loop_d2h && !self.repeated_loop_host_sync
    }
}

/// Session contract for a CUDA implementation of `Pi_CCS -> Pi_RLC -> Pi_DEC`.
///
/// Associated input/output types are intentionally opaque handles. In a real
/// implementation they should be device-resident buffers, not CPU proof
/// structures. The contract only says how the protocol phases compose:
///
/// - [`Self::launch_pi_ccs`] consumes the current CCS/fresh+carried surface and
///   produces the CE claims/proof material needed by Pi_RLC.
/// - [`Self::launch_pi_rlc`] consumes Pi_CCS output and produces the mixed parent
///   CE claim plus mixed witness state needed by Pi_DEC.
/// - [`Self::launch_pi_dec`] consumes Pi_RLC output and produces low-norm child
///   CE claims/witnesses for the next running accumulator.
/// - [`Self::retain_dec_children`] keeps the Pi_DEC children on device for the
///   next step. This is the repeated-loop residency boundary.
/// - [`Self::export_final_proof`] is the first required D2H proof-material
///   boundary for the normal prover path.
pub trait ResidentCudaNifsSession {
    type Error;
    type PiCcsInput;
    type PiCcsOutput;
    type PiRlcOutput;
    type PiDecOutput;
    type FinalProof;

    fn step_plan(&self) -> CudaReductionStepPlan {
        CudaReductionStepPlan::RESIDENT_SUPERNEO_NIFS
    }

    fn residency_policy(&self) -> CudaResidencyPolicy {
        self.step_plan().residency()
    }

    fn launch_pi_ccs(&mut self, input: Self::PiCcsInput) -> Result<Self::PiCcsOutput, Self::Error>;

    fn launch_pi_rlc(&mut self, input: Self::PiCcsOutput) -> Result<Self::PiRlcOutput, Self::Error>;

    fn launch_pi_dec(&mut self, input: Self::PiRlcOutput) -> Result<Self::PiDecOutput, Self::Error>;

    fn retain_dec_children(&mut self, output: Self::PiDecOutput) -> Result<(), Self::Error>;

    fn export_final_proof(&mut self) -> Result<Self::FinalProof, Self::Error>;

    /// Run one resident prover step without exporting intermediate proof
    /// material to the host.
    fn run_resident_step(&mut self, input: Self::PiCcsInput) -> Result<(), Self::Error> {
        let pi_ccs = self.launch_pi_ccs(input)?;
        let pi_rlc = self.launch_pi_rlc(pi_ccs)?;
        let pi_dec = self.launch_pi_dec(pi_rlc)?;
        self.retain_dec_children(pi_dec)
    }
}
