//! IVC (Incrementally Verifiable Computation) engine
//!
//! This module implements IVC with two layers:
//! 1. Core folding primitives (internal, used by NIVC)
//! 2. Public IVC API (pipeline, exported via lib.rs)
//!
//! The pipeline module provides the main IVC proving and verification functions.

pub(crate) mod internal;
pub(crate) mod pipeline;

use crate::F;

/// Crate-private engine interface that NIVC uses to call into IVC
///
/// This trait provides a clean abstraction boundary between IVC and NIVC,
/// allowing IVC internals to be refactored without affecting NIVC.
#[allow(dead_code)]
pub(crate) trait FoldStepEngine {
    /// Prove a single IVC step with chained state
    ///
    /// This is the main entry point for proving an IVC step. It takes the current
    /// step input and previous folding state, and produces updated state.
    fn prove_step_chained(
        input: &IvcStepInput<'_>,
        prev_me: Option<neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>>,
        prev_me_wit: Option<neo_ccs::MeWitness<F>>,
        prev_lhs_mcs: Option<(neo_ccs::McsInstance<neo_ajtai::Commitment, F>, neo_ccs::McsWitness<F>)>,
    ) -> anyhow::Result<(
        IvcStepResult,
        neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>,
        neo_ccs::MeWitness<F>,
        (neo_ccs::McsInstance<neo_ajtai::Commitment, F>, neo_ccs::McsWitness<F>)
    )>;

    /// Verify a single IVC step
    fn verify_step(
        step_ccs: &neo_ccs::CcsStructure<F>,
        proof: &IvcProof,
        prev_acc: &Accumulator,
        binding: &StepBindingSpec,
        params: &crate::NeoParams,
        prev_augmented_x: Option<&[F]>,
    ) -> anyhow::Result<bool>;

    /// Build augmented CCS for a step with the given parameters
    fn augmented_ccs_for(
        step_ccs: &neo_ccs::CcsStructure<F>,
        step_x_len: usize,
        binding: &StepBindingSpec,
        y_len: usize,
        rlc_binder: Option<(Vec<F>, F)>,
    ) -> anyhow::Result<neo_ccs::CcsStructure<F>>;
}

/// Concrete IVC engine implementation
#[allow(dead_code)]
pub(crate) struct IvcEngine;

impl FoldStepEngine for IvcEngine {
    fn prove_step_chained(
        input: &IvcStepInput<'_>,
        prev_me: Option<neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>>,
        prev_me_wit: Option<neo_ccs::MeWitness<F>>,
        prev_lhs_mcs: Option<(neo_ccs::McsInstance<neo_ajtai::Commitment, F>, neo_ccs::McsWitness<F>)>,
    ) -> anyhow::Result<(
        IvcStepResult,
        neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>,
        neo_ccs::MeWitness<F>,
        (neo_ccs::McsInstance<neo_ajtai::Commitment, F>, neo_ccs::McsWitness<F>)
    )> {
        // Delegate to pipeline prover
        let input_owned = IvcStepInput {
            params: input.params,
            step_ccs: input.step_ccs,
            step_witness: input.step_witness,
            prev_accumulator: input.prev_accumulator,
            step: input.step,
            public_input: input.public_input,
            y_step: input.y_step,
            binding_spec: input.binding_spec,
            app_input_binding: input.app_input_binding,
            prev_augmented_x: input.prev_augmented_x,
        };
        pipeline::prover::prove_ivc_step_chained(input_owned, prev_me, prev_me_wit, prev_lhs_mcs)
            .map_err(|e| anyhow::anyhow!("{}", e))
    }

    fn verify_step(
        step_ccs: &neo_ccs::CcsStructure<F>,
        proof: &IvcProof,
        prev_acc: &Accumulator,
        binding: &StepBindingSpec,
        params: &crate::NeoParams,
        prev_augmented_x: Option<&[F]>,
    ) -> anyhow::Result<bool> {
        pipeline::verifier::verify_ivc_step(step_ccs, proof, prev_acc, binding, params, prev_augmented_x)
            .map_err(|e| anyhow::anyhow!("{}", e))
    }

    fn augmented_ccs_for(
        step_ccs: &neo_ccs::CcsStructure<F>,
        step_x_len: usize,
        binding: &StepBindingSpec,
        y_len: usize,
        rlc_binder: Option<(Vec<F>, F)>,
    ) -> anyhow::Result<neo_ccs::CcsStructure<F>> {
        internal::augmented::build_augmented_ccs_linked_with_rlc(
            step_ccs,
            step_x_len,
            &binding.y_step_offsets,
            &binding.y_prev_witness_indices,
            &binding.step_program_input_witness_indices,
            y_len,
            binding.const1_witness_index,
            rlc_binder,
        ).map_err(|e| anyhow::anyhow!("{}", e))
    }
}

// Public API exports (for backwards compatibility until migration is complete)
// These will eventually be removed as we migrate to the engine interface
pub use internal::basecase::zero_mcs_instance_for_shape;

// Re-exports from pipeline for NIVC to use
pub(crate) use pipeline::prover::prove_ivc_step_chained;
pub(crate) use pipeline::verifier::verify_ivc_step;
pub(crate) use internal::augmented::{build_augmented_ccs_linked_with_rlc, build_final_snark_public_input};

// Re-export from shared for NIVC
pub(crate) use crate::shared::digest::compute_accumulator_digest_fields;
pub(crate) use crate::shared::types::{
    Accumulator, IvcStepInput, IvcProof, StepBindingSpec, IvcStepResult,
    IvcChainProof,
};

