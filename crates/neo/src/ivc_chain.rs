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
use crate::{OutputClaim, expose_z_component};
use crate::ivc::{Accumulator, IvcProof, StepBindingSpec, prove_ivc_step_chained, LastNExtractor, IvcStepInput, StepOutputExtractor};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use neo_ccs::CcsStructure;

/// Minimal state wrapper for the simple API
#[derive(Clone)]
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

    /// Set the running ME instance and witness (for compatibility with old API)
    pub fn set_running_me(
        &mut self,
        me: neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>,
        wit: neo_ccs::MeWitness<F>,
    ) {
        self.running_me = Some(me);
        self.running_me_wit = Some(wit);
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
    finalize_and_prove_with_options(state, FinalizeOptions { embed_ivc_ev: false })
}

/// Options to control final proof generation
pub struct FinalizeOptions { pub embed_ivc_ev: bool }

/// Same as `finalize_and_prove` but with options (e.g., embed IVC EV inside Spartan).
pub fn finalize_and_prove_with_options(
    state: State,
    opts: FinalizeOptions,
) -> anyhow::Result<Option<(crate::Proof, neo_ccs::CcsStructure<F>, Vec<F>)>> {
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
    let rho = if final_ivc_proof.step_rho != F::ZERO { 
        final_ivc_proof.step_rho 
    } else {
        let step_data = crate::ivc::build_step_data_with_x(
            prev_accumulator, 
            final_ivc_proof.step, 
            step_x
        );
        let step_digest = crate::ivc::create_step_digest(&step_data);
        // Use stored step commitment coordinates for consistent rho derivation
        let c_step_coords = &final_ivc_proof.c_step_coords;
        let (rho, _td) = crate::ivc::rho_from_transcript(prev_accumulator, step_digest, c_step_coords);
        rho
    };
    
    // Get y_prev and y_next from the accumulator progression
    let y_prev = if state.ivc_proofs.len() > 1 {
        &state.ivc_proofs[state.ivc_proofs.len() - 2].next_accumulator.y_compact
    } else {
        &state.initial_y
    };
    let y_next = &state.accumulator.y_compact;
    
    // Reconstruct the exact augmented CCS used for folding at the final step
    // (same logic as verifier; binds to transcript via RLC binder)
    let final_proof = state.ivc_proofs.last().unwrap();
    let step_data = crate::ivc::build_step_data_with_x(prev_accumulator, final_proof.step, step_x);
    let step_digest = crate::ivc::create_step_digest(&step_data);
    let num_coords = final_proof.c_step_coords.len();
    let rlc_coeffs = crate::ivc::generate_rlc_coefficients(prev_accumulator, step_digest, &final_proof.c_step_coords, num_coords);

    // Build augmented public input with recomputed ρ
    let step_public_input_aug = crate::ivc::build_linked_augmented_public_input(
        step_x,
        rho,
        y_prev,
        y_next,
    );
    // Dummy witness (zeros) to size the augmented witness tail
    let step_witness_aug = crate::ivc::build_linked_augmented_witness(
        &vec![F::ZERO; state.step_ccs.m],
        &state.binding.y_step_offsets,
        rho,
    );
    // Full z = [public || witness]
    let mut full_step_z = step_public_input_aug.clone();
    full_step_z.extend_from_slice(&step_witness_aug);
    let d = neo_math::ring::D;
    let decomp_z = crate::decomp_b(&full_step_z, 2, d, crate::DecompStyle::Balanced);
    let m_step = decomp_z.len() / d;
    
    // Ensure Ajtai PP and compute aggregated row g_full
    crate::ensure_ajtai_pp_for_dims(d, m_step, || {
        use rand::{SeedableRng, RngCore};
        use rand::rngs::StdRng;
        let mut rng = if std::env::var("NEO_DETERMINISTIC").is_ok() {
            StdRng::from_seed([42u8; 32])
        } else {
            let mut seed = [0u8; 32];
            rand::rng().fill_bytes(&mut seed);
            StdRng::from_seed(seed)
        };
        let pp = crate::ajtai_setup(&mut rng, d, 16, m_step)?;
        neo_ajtai::set_global_pp(pp).map_err(anyhow::Error::from)
    })?;
    let pp_full = neo_ajtai::get_global_pp_for_dims(d, m_step)
        .map_err(|e| anyhow::anyhow!("Failed to get Ajtai PP for full witness: {}", e))?;
    let z_len_digits = d * m_step;
    let g_full = neo_ajtai::compute_aggregated_ajtai_row(&*pp_full, &rlc_coeffs, z_len_digits, num_coords)
        .map_err(|e| anyhow::anyhow!("compute_aggregated_ajtai_row failed: {}", e))?;

    // Extract U = ρ·y_step digit slice and compute rhs_diff to match prover
    let y_len = y_prev.len();
    let u_offset = step_public_input_aug.len() + state.step_ccs.m;
    let u_digits_start = u_offset * d;
    let u_digits_len = y_len * d;
    let mut g_u = vec![F::ZERO; z_len_digits];
    g_u[u_digits_start .. u_digits_start + u_digits_len]
        .copy_from_slice(&g_full[u_digits_start .. u_digits_start + u_digits_len]);

    let rhs_full = rlc_coeffs.iter().zip(final_proof.folding_proof.as_ref().ok_or_else(|| anyhow::anyhow!("missing folding_proof"))?.pi_ccs_inputs[1].c.data.iter())
        .fold(F::ZERO, |acc, (ri, cf)| acc + *ri * F::from_u64(cf.as_canonical_u64()));
    let rhs_step = rlc_coeffs.iter().zip(final_proof.c_step_coords.iter())
        .fold(F::ZERO, |acc, (ri, cs)| acc + *ri * *cs);
    let rhs_diff = rhs_full - rhs_step;
    let rlc_binder = Some((g_u, rhs_diff));

    let augmented_ccs = crate::ivc::build_augmented_ccs_linked_with_rlc(
        &state.step_ccs,
        step_x.len(),
        &state.binding.y_step_offsets,
        &state.binding.y_prev_witness_indices,
        &state.binding.x_witness_indices,
        y_len,
        state.binding.const1_witness_index,
        rlc_binder,
    ).map_err(|e| anyhow::anyhow!("Failed to build final augmented CCS: {}", e))?;

    // Build the correct public input format: [step_x || ρ || y_prev || y_next]
    let final_public_input = crate::ivc::build_final_snark_public_input(
        step_x, rho, y_prev, y_next
    );

    // Generate Stage 5 final SNARK proof using the final ME instance
    let proof = if let (Some(final_me), Some(final_me_wit)) = (&state.running_me, &state.running_me_wit) {
        // Use adapt_from_modern to properly populate ajtai_rows for Spartan bridge
        let (mut legacy_me, legacy_wit, _ajtai_pp) = crate::adapt_from_modern(
            std::slice::from_ref(final_me),
            std::slice::from_ref(final_me_wit),
            &augmented_ccs,
            &state.params,
            &[],  // possibly replaced below when embedding EV
            None, // no vjs needed for final proof
        ).map_err(|e| anyhow::anyhow!("Bridge adapter failed: {}", e))?;
        
        // Bind proof to the augmented CCS + public input
        let context_digest = crate::context_digest_v1(&augmented_ccs, &final_public_input);
        #[allow(deprecated)]
        { legacy_me.header_digest = context_digest; }
        
        let ajtai_pp_arc = std::sync::Arc::new(_ajtai_pp);
        let lean = if opts.embed_ivc_ev {
            // Build OutputClaims to expose y_step components from Z
            let pub_cols = final_public_input.len();
            let y_len = y_prev.len();
            let m = final_me_wit.Z.cols();
            let mut claims: Vec<OutputClaim<F>> = Vec::with_capacity(y_len);
            // Compute y_step = (y_next - y_prev) / rho (if rho != 0)
            if rho == F::ZERO { anyhow::bail!("ρ derived from transcript is zero; EV embedding not supported for this rare case"); }
            let rho_inv = F::ONE / rho;
            for (i, &off) in state.binding.y_step_offsets.iter().enumerate().take(y_len) {
                let expected = (y_next[i] - y_prev[i]) * rho_inv;
                let k_index = pub_cols + off; // position in [public || witness]
                let weight = expose_z_component(&state.params, m, k_index);
                claims.push(OutputClaim { weight, expected });
            }

            // Re-run adapter with claims to append weight vectors and outputs
            let (mut legacy_me2, legacy_wit2, _pp2) = crate::adapt_from_modern(
                std::slice::from_ref(final_me),
                std::slice::from_ref(final_me_wit),
                &augmented_ccs,
                &state.params,
                &claims,
                None,
            ).map_err(|e| anyhow::anyhow!("Bridge adapter (with EV claims) failed: {}", e))?;
            #[allow(deprecated)]
            { legacy_me2.header_digest = context_digest; }

            let ev_embed = neo_spartan_bridge::IvcEvEmbed { rho, y_prev: y_prev.clone(), y_next: y_next.clone() };
            neo_spartan_bridge::compress_me_to_lean_proof_with_pp_and_ev(&legacy_me2, &legacy_wit2, Some(ajtai_pp_arc), Some(ev_embed))?
        } else {
            neo_spartan_bridge::compress_me_to_lean_proof_with_pp(&legacy_me, &legacy_wit, Some(ajtai_pp_arc))?
        };
        
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
