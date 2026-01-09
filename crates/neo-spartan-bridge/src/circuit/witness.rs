//! Witness and instance types for FoldRun circuit
//!
//! These types describe the public inputs and private witness that the
//! FoldRun circuit expects. They mirror the structures used by neo-fold
//! and neo-reductions, but are tailored for circuit synthesis.

use crate::statement::SpartanShardStatement;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::MeInstance;
use neo_fold::output_binding::OutputBindingConfig;
use neo_fold::shard::ShardProof as FoldRun;
use neo_math::{F, K};
use neo_memory::witness::StepInstanceBundle;

/// Public inputs to the FoldRun circuit
#[derive(Clone, Debug)]
pub struct FoldRunInstance {
    /// Public statement (encoded as Spartan public IO).
    pub statement: SpartanShardStatement,
}

/// Private witness for the FoldRun circuit
#[derive(Clone, Debug)]
pub struct FoldRunWitness {
    /// The complete shard proof (FoldRun)
    pub fold_run: FoldRun,
    /// Per-step public bundles (MCS + optional Twist/Shout instances).
    pub steps_public: Vec<StepInstanceBundle<Cmt, F, K>>,
    /// Initial accumulator (ME inputs to step 0).
    pub initial_accumulator: Vec<MeInstance<Cmt, F, K>>,
    /// Digest binding the VM/program identity (e.g., ROM/ELF bytes).
    pub vm_digest: [u8; 32],
    /// Optional output binding configuration (public claim set) used when `fold_run.output_proof` is present.
    pub output_binding: Option<OutputBindingConfig>,
    /// Optional step-to-step linking constraints pinned into the circuit.
    ///
    /// Each pair `(prev_idx, next_idx)` enforces `steps[i].x[prev_idx] == steps[i+1].x[next_idx]`.
    pub step_linking: Vec<(usize, usize)>,
}

impl FoldRunInstance {
    pub fn public_io(&self) -> Vec<crate::CircuitF> {
        self.statement.public_io()
    }
}

impl FoldRunWitness {
    pub fn new(
        fold_run: FoldRun,
        steps_public: Vec<StepInstanceBundle<Cmt, F, K>>,
        initial_accumulator: Vec<MeInstance<Cmt, F, K>>,
        vm_digest: [u8; 32],
        output_binding: Option<OutputBindingConfig>,
    ) -> Self {
        Self {
            fold_run,
            steps_public,
            initial_accumulator,
            vm_digest,
            output_binding,
            step_linking: Vec::new(),
        }
    }

    pub fn with_step_linking(mut self, pairs: Vec<(usize, usize)>) -> Self {
        self.step_linking = pairs;
        self
    }
}
