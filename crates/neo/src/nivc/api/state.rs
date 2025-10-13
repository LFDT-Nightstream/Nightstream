//! NIVC state management: accumulators, lane states, and prover context

use crate::{F, NeoParams};
use super::program::NivcProgram;

/// Maintains the running "ME / accumulator state" for a particular lane (function F_j)
#[derive(Clone, Default)]
pub struct LaneRunningState {
    /// The ME instance (folded up to now) for this lane j; optional if no steps used yet
    pub me: Option<neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>>,
    /// Witness associated with the ME instance (for proving)
    pub wit: Option<neo_ccs::MeWitness<F>>,
    /// Coordinates c_coords of the lane's current accumulator commitment
    pub c_coords: Vec<F>,
    /// Digest (e.g. compressed hash) of the lane's current accumulator state
    pub c_digest: [u8; 32],
    /// (Optional) left-hand side MCS instance, used for linking multiple ME instances
    pub lhs_mcs: Option<neo_ccs::McsInstance<neo_ajtai::Commitment, F>>,
    /// Witness for the above MCS instance
    pub lhs_mcs_wit: Option<neo_ccs::McsWitness<F>>,
}

/// The NIVC accumulator: global state plus per-lane running states (reflects "U vector + y")
#[derive(Clone)]
pub struct NivcAccumulators {
    /// Running state for each lane j (i.e. U_j)
    pub lanes: Vec<LaneRunningState>,
    /// The shared compact state y (this is `z_i` in the paper, or the global accumulator state)
    pub global_y: Vec<F>,
    /// Step counter: number of total steps executed so far
    pub step: u64,
}

impl NivcAccumulators {
    /// Initialize with `num_lanes` and starting state `y0`
    pub fn new(num_lanes: usize, y0: Vec<F>) -> Self {
        Self {
            lanes: vec![LaneRunningState::default(); num_lanes],
            global_y: y0,
            step: 0,
        }
    }
}

/// Proof of one NIVC step: i.e., that one application of F_j from state z_i yields z_{i+1}
#[derive(Clone)]
pub struct NivcStepProof {
    /// Which lane / function index `j` was used (this corresponds to φ choice)
    pub lane_idx: usize,
    /// Public application-level inputs bound into the transcript (step_io)
    pub step_io: Vec<F>,
    /// The inner IVC proof (folding the chosen lane) that enforces zᵢ → zᵢ₊₁
    pub inner: crate::ivc::IvcProof,
}

/// Full NIVC chain proof: sequence of step proofs + final accumulator snapshot
#[derive(Clone)]
pub struct NivcChainProof {
    /// The sequence of step proofs produced by the prover
    pub steps: Vec<NivcStepProof>,
    /// The final accumulator (global_y and all lanes) after all steps
    pub final_acc: NivcAccumulators,
}

/// Prover-side state / context for building an NIVC proof
pub struct NivcState {
    /// Public params (e.g. commitment parameters, field parameters)
    pub params: NeoParams,
    /// The NIVC program (available step types F_j)
    pub program: NivcProgram,
    /// The current accumulator state (U vector + y)
    pub acc: NivcAccumulators,
    /// Collected step proofs (in order) as the proof is built
    pub(crate) steps: Vec<NivcStepProof>,
    /// For each lane j, the previous *augmented public input* X (used to enforce left-linking)
    pub(crate) prev_aug_x_by_lane: Vec<Option<Vec<F>>>,
}

impl NivcState {
    /// Create a fresh NIVC prover state with initial y = y₀
    pub fn new(params: NeoParams, program: NivcProgram, y0: Vec<F>) -> anyhow::Result<Self> {
        if program.is_empty() { 
            anyhow::bail!("NIVC program has no step types"); 
        }
        let lanes = program.len();
        Ok(Self { 
            params, 
            program, 
            acc: NivcAccumulators::new(lanes, y0), 
            steps: Vec::new(), 
            prev_aug_x_by_lane: vec![None; lanes] 
        })
    }
    
    /// Finalize and return the NIVC chain proof (no outer SNARK compression).
    pub fn into_proof(self) -> NivcChainProof {
        NivcChainProof { 
            steps: self.steps, 
            final_acc: self.acc 
        }
    }
    
    /// Get a reference to the collected step proofs
    pub fn steps(&self) -> &[NivcStepProof] {
        &self.steps
    }
    
    /// Get a reference to the current accumulator state
    pub fn accumulator(&self) -> &NivcAccumulators {
        &self.acc
    }
    
    /// Get a reference to the program
    pub fn program(&self) -> &NivcProgram {
        &self.program
    }
}

