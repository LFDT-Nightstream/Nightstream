//! Chain-style step/verify API over Neo IVC
//!
//! This module provides a minimal ergonomic wrapper around the production IVC
//! pipeline so examples can look like:
//!
//!   - `step(state, io, witness) -> State`
//!   - `finalize_and_prove(state) -> Proof`
//!
//! Internally it uses per-step Nova folding (Stages 1-4) and generates
//! a final SNARK proof (Stage 5) on demand.

use crate::{F, NeoParams};
use crate::ivc::{Accumulator, IvcProof, StepBindingSpec, prove_ivc_step_chained, LastNExtractor, IvcStepInput, StepOutputExtractor};
use p3_field::PrimeCharacteristicRing;
use neo_ccs::CcsStructure;

/// Minimal state wrapper for the simple API
pub struct State {
    params: NeoParams,
    step_ccs: CcsStructure<F>,
    binding: StepBindingSpec,
    pub accumulator: Accumulator,
    initial_y: Vec<F>,
    /// Running folded ME instance (None for first step, Some after first fold)
    running_me: Option<neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>>,
    /// Running folded ME witness (None for first step, Some after first fold)
    running_me_wit: Option<neo_ccs::MeWitness<F>>,
    pub ivc_proofs: Vec<IvcProof>,
}

impl State {
    /// Initialize a new State for a given step CCS and initial y-state.
    ///
    /// - `y0` is the initial compact y (the running state exposed by IVC folding)
    /// - `binding` must be a trusted binding specification for the step circuit
    pub fn new(
        params: NeoParams,
        step_ccs: CcsStructure<F>,
        y0: Vec<F>,
        binding: StepBindingSpec,
    ) -> anyhow::Result<Self> {
        let acc = Accumulator {
            c_z_digest: [0u8; 32],
            c_coords: vec![],
            y_compact: y0.clone(),
            step: 0,
        };

        Ok(Self { 
            params, 
            step_ccs, 
            binding, 
            accumulator: acc,
            initial_y: y0,
            running_me: None,
            running_me_wit: None,
            ivc_proofs: Vec::new(),
        })
    }
}

/// Advance one step of the IVC chain using per-step Nova folding.
///
/// - `io` are per-step public inputs (can be empty)
/// - `witness` is the step circuit witness
///
/// This performs real Nova folding (Stages 1-4) for each step.
pub fn step(mut state: State, io: &[F], witness: &[F]) -> anyhow::Result<State> {
    let extractor = LastNExtractor { n: state.binding.y_step_offsets.len() };
    
    // Extract y_step from witness
    let y_step = extractor.extract_y_step(witness);
    
    // Build input for chained proving
    let input = IvcStepInput {
        params: &state.params,
        step_ccs: &state.step_ccs,
        step_witness: witness,
        prev_accumulator: &state.accumulator,
        step: state.accumulator.step,
        public_input: if io.is_empty() { None } else { Some(io) },
        y_step: &y_step,
        binding_spec: &state.binding,
    };
    
    // Perform chained Nova folding
    let (step_res, new_me, new_me_wit) = prove_ivc_step_chained(
        input,
        state.running_me.clone(),
        state.running_me_wit.clone(),
    ).map_err(|e| anyhow::anyhow!("IVC step proving failed: {}", e))?;

    // Update state with new running fold state
    state.accumulator = step_res.proof.next_accumulator.clone();
    state.running_me = Some(new_me);
    state.running_me_wit = Some(new_me_wit);
    state.ivc_proofs.push(step_res.proof);

    Ok(state)
}

/// Finalize the IVC chain and generate the final SNARK proof (Stage 5).
///
/// This generates a succinct proof that attests to the entire IVC chain.
/// Returns `Ok(Some((proof, augmented_ccs, final_public_input)))` if there were steps, `Ok(None)` if no steps.
pub fn finalize_and_prove(state: State) -> anyhow::Result<Option<(crate::Proof, neo_ccs::CcsStructure<F>, Vec<F>)>> {
    if state.ivc_proofs.is_empty() {
        return Ok(None);
    }

    // 🔒 SECURITY FIX: Build correct augmented CCS public input format
    // Get the final step's data to construct [step_x || ρ || y_prev || y_next]
    let final_ivc_proof = state.ivc_proofs.last().unwrap();
    let step_x = &final_ivc_proof.step_public_input; // The step's public input
    
    // Extract ρ by recomputing it from the accumulator and step digest
    // This is the same way it was computed during proving
    let prev_accumulator = if state.ivc_proofs.len() > 1 {
        &state.ivc_proofs[state.ivc_proofs.len() - 2].next_accumulator
    } else {
        // For single step, use initial accumulator
        &Accumulator {
            c_z_digest: [0u8; 32],
            c_coords: vec![],
            y_compact: vec![F::ZERO; state.accumulator.y_compact.len()],
            step: 0,
        }
    };
    
    // Use stored ρ when available; recompute as fallback
    let rho = if final_ivc_proof.step_rho != F::ZERO { final_ivc_proof.step_rho } else {
        let step_data = crate::ivc::build_step_data_with_x(
            prev_accumulator, 
            final_ivc_proof.step, 
            step_x
        );
        let step_digest = crate::ivc::create_step_digest(&step_data);
        let (rho, _td) = crate::ivc::rho_from_transcript(prev_accumulator, step_digest);
        rho
    };
    
    // Get y_prev and y_next from the accumulator progression
    let y_prev = if state.ivc_proofs.len() > 1 {
        &state.ivc_proofs[state.ivc_proofs.len() - 2].next_accumulator.y_compact
    } else {
        &state.initial_y
    };
    let y_next = &state.accumulator.y_compact;
    
    // Get the augmented CCS from the final proof for verification
    let augmented_ccs = state.ivc_proofs.last().unwrap()
        .augmented_ccs.as_ref()
        .expect("Final proof missing augmented CCS")
        .clone();

    // Build the correct public input format: [step_x || ρ || y_prev || y_next]
    let final_public_input = crate::ivc::build_final_snark_public_input(
        step_x, rho, y_prev, y_next
    );

    // Generate Stage 5 final SNARK proof using the final ME instance
    let proof = if let (Some(final_me), Some(final_me_wit)) = (&state.running_me, &state.running_me_wit) {
        // Use adapt_from_modern to properly populate ajtai_rows for Spartan bridge
        let (mut legacy_me, legacy_wit) = crate::adapt_from_modern(
            std::slice::from_ref(final_me),
            std::slice::from_ref(final_me_wit),
            &augmented_ccs,
            &state.params,
            &[],  // no extra OutputClaim for final proof
            None, // no vjs needed for final proof
        ).map_err(|e| anyhow::anyhow!("Bridge adapter failed: {}", e))?;
        
        // Bind proof to the augmented CCS + public input
        let context_digest = crate::context_digest_v1(&augmented_ccs, &final_public_input);
        #[allow(deprecated)]
        { legacy_me.header_digest = context_digest; }
        
        let lean = neo_spartan_bridge::compress_me_to_lean_proof(&legacy_me, &legacy_wit)?;
        
        crate::Proof {
            v: 2,
            circuit_key: lean.circuit_key,
            vk_digest: lean.vk_digest,
            public_io: lean.public_io_bytes,
            proof_bytes: lean.proof_bytes,
            public_results: vec![], // IVC chains have no separate application outputs
            meta: crate::ProofMeta { 
                num_y_compact: state.ivc_proofs.last().unwrap().step_proof.meta.num_y_compact,
                num_app_outputs: 0, // IVC chains have no separate application outputs
            },
        }
    } else {
        return Err(anyhow::anyhow!("No running ME instance available for final proof"));
    };

    Ok(Some((proof, augmented_ccs, final_public_input)))
}