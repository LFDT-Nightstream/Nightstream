//! NIVC (Non-Uniform IVC) support à la HyperNova
//!
//! This module provides a pragmatic NIVC driver on top of the existing IVC folding
//! implementation. It allows selecting one of multiple step CCS relations per step
//! and folds only that lane's running instance, achieving an "à‑la‑carte" cost profile.
//!
//! Design highlights:
//! - Keeps a per‑type ("lane") running ME instance and witness.
//! - Maintains a global y (compact state) shared across lanes.
//! - Binds the selected lane index into the step public input (and thus the FS transcript).
//! - Reuses `prove_ivc_step_chained`/`verify_ivc_step` for per‑step proving and verification.
//!
//! NOTE: For production‑grade scalability, consider switching the "lanes state" to a
//! Merkle tree and proving a single leaf update in‑circuit. This initial driver does not
//! add in‑circuit constraints for unchanged lanes; it preserves à‑la‑carte cost by
//! only folding the chosen lane each step.

use crate::{F, NeoParams};
use p3_field::{PrimeField64, PrimeCharacteristicRing};

use neo_ccs::CcsStructure;

/// Import IVC helpers
use crate::ivc::{
    Accumulator,
    IvcProof,
    IvcStepInput,
    StepBindingSpec,
    StepOutputExtractor,
    IndexExtractor,
    prove_ivc_step_chained,
};

/// Poseidon2 (same configuration used across the repo)
use neo_ccs::crypto::poseidon2_goldilocks as p2;

/// One step specification in an NIVC program
#[derive(Clone)]
pub struct NivcStepSpec {
    pub ccs: CcsStructure<F>,
    pub binding: StepBindingSpec,
}

/// Program registry of all step types
#[derive(Clone)]
pub struct NivcProgram {
    pub steps: Vec<NivcStepSpec>,
}

impl NivcProgram {
    pub fn new(steps: Vec<NivcStepSpec>) -> Self { Self { steps } }
    pub fn len(&self) -> usize { self.steps.len() }
    pub fn is_empty(&self) -> bool { self.steps.is_empty() }
}

/// Running ME state per lane
#[derive(Clone, Default)]
pub struct LaneRunningState {
    pub me: Option<neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>>,
    pub wit: Option<neo_ccs::MeWitness<F>>,
    pub c_coords: Vec<F>,
    pub c_digest: [u8; 32],
    pub lhs_mcs: Option<neo_ccs::McsInstance<neo_ajtai::Commitment, F>>,
    pub lhs_mcs_wit: Option<neo_ccs::McsWitness<F>>,
}

/// NIVC accumulator: per‑lane commitment state + global y and step counter
#[derive(Clone)]
pub struct NivcAccumulators {
    pub lanes: Vec<LaneRunningState>,
    pub global_y: Vec<F>,
    pub step: u64,
}

impl NivcAccumulators {
    pub fn new(num_lanes: usize, y0: Vec<F>) -> Self {
        Self {
            lanes: vec![LaneRunningState::default(); num_lanes],
            global_y: y0,
            step: 0,
        }
    }
}

/// NIVC step proof: identify which lane was executed and carry the inner IVC proof
#[derive(Clone)]
pub struct NivcStepProof {
    pub which_type: usize,
    /// Application-level public inputs bound into the transcript for this step
    pub step_io: Vec<F>,
    pub inner: IvcProof,
}

/// NIVC chain proof: sequence of step proofs and the final accumulator snapshot
#[derive(Clone)]
pub struct NivcChainProof {
    pub steps: Vec<NivcStepProof>,
    pub final_acc: NivcAccumulators,
}

/// NIVC driver state for proving
pub struct NivcState {
    pub params: NeoParams,
    pub program: NivcProgram,
    pub acc: NivcAccumulators,
    steps: Vec<NivcStepProof>,
    prev_aug_x_by_lane: Vec<Option<Vec<F>>>,
}

impl NivcState {
    pub fn new(params: NeoParams, program: NivcProgram, y0: Vec<F>) -> anyhow::Result<Self> {
        if program.is_empty() { anyhow::bail!("NIVC program has no step types"); }
        let lanes = program.len();
        Ok(Self { params, program, acc: NivcAccumulators::new(lanes, y0), steps: Vec::new(), prev_aug_x_by_lane: vec![None; lanes] })
    }

    /// Compute a compact digest of all lane digests for transcript binding.
    /// Returns 4 field elements (32 bytes) in Goldilocks packed form.
    fn lanes_root_fields(&self) -> Vec<F> {
        // Concatenate per‑lane c_digest bytes
        let mut bytes = Vec::with_capacity(self.acc.lanes.len() * 32 + 16);
        for lane in &self.acc.lanes {
            bytes.extend_from_slice(&lane.c_digest);
        }
        // Domain separate
        bytes.extend_from_slice(b"neo/nivc/lanes_root/v1");
        let digest = p2::poseidon2_hash_packed_bytes(&bytes);
        digest.into_iter().map(|g| F::from_u64(g.as_canonical_u64())).collect()
    }

    /// Execute one NIVC step for lane `which` with given step IO and witness.
    /// Returns the step proof and updates internal state.
    pub fn step(
        &mut self,
        which: usize,
        step_io: &[F],
        step_witness: &[F],
    ) -> anyhow::Result<NivcStepProof> {
        if which >= self.program.len() { anyhow::bail!("which_type out of bounds"); }
        let spec = &self.program.steps[which];

        // Build a lane‑scoped Accumulator view for the existing IVC prover
        let lane = &self.acc.lanes[which];
        let prev_acc_lane = Accumulator {
            c_z_digest: lane.c_digest,
            c_coords: lane.c_coords.clone(),
            y_compact: self.acc.global_y.clone(),
            step: self.acc.step,
        };

        // Public input: bind which_type and lanes_root to the FS transcript via step_x
        let mut app_inputs = Vec::with_capacity(1 + step_io.len() + 4);
        app_inputs.push(F::from_u64(which as u64));
        app_inputs.extend_from_slice(step_io);
        app_inputs.extend_from_slice(&self.lanes_root_fields());

        // Extract y_step from witness using binding spec offsets
        let extractor = IndexExtractor { indices: spec.binding.y_step_offsets.clone() };
        let y_step = extractor.extract_y_step(step_witness);

        // Thread the running ME for this lane (if any)
        let prev_me = self.acc.lanes[which].me.clone();
        let prev_wit = self.acc.lanes[which].wit.clone();
        let prev_mcs = self.acc.lanes[which].lhs_mcs.clone().zip(self.acc.lanes[which].lhs_mcs_wit.clone());

        // Prove the step using the existing chained IVC helper
        let input = IvcStepInput {
            params: &self.params,
            step_ccs: &spec.ccs,
            step_witness,
            prev_accumulator: &prev_acc_lane,
            step: self.acc.step,
            public_input: Some(&app_inputs),
            y_step: &y_step,
            binding_spec: &spec.binding,
            transcript_only_app_inputs: true,
            prev_augmented_x: self.prev_aug_x_by_lane[which].as_deref(),
        };
        let (res, me_out, wit_out, lhs_next) = prove_ivc_step_chained(input, prev_me, prev_wit, prev_mcs)
            .map_err(|e| anyhow::anyhow!("prove_ivc_step_chained failed: {}", e))?;

        // Update lane state: carry ME forward and refresh commitment coords/digest
        let lane_mut = &mut self.acc.lanes[which];
        lane_mut.me = Some(me_out);
        lane_mut.wit = Some(wit_out);
        lane_mut.c_coords = res.proof.next_accumulator.c_coords.clone();
        lane_mut.c_digest = res.proof.next_accumulator.c_z_digest;
        lane_mut.lhs_mcs = Some(lhs_next.0);
        lane_mut.lhs_mcs_wit = Some(lhs_next.1);

        // Update global state
        self.acc.global_y = res.proof.next_accumulator.y_compact.clone();
        self.acc.step += 1;

        // Update lane-local previous augmented X for linking next time this lane is used
        self.prev_aug_x_by_lane[which] = Some(res.proof.step_augmented_public_input.clone());

        let sp = NivcStepProof { which_type: which, step_io: step_io.to_vec(), inner: res.proof };
        self.steps.push(sp.clone());
        Ok(sp)
    }

    /// Finalize and return the NIVC chain proof (no outer SNARK compression).
    pub fn into_proof(self) -> NivcChainProof {
        NivcChainProof { steps: self.steps, final_acc: self.acc }
    }
}

/// Verify an NIVC chain given the program, initial y, and parameter set.
pub fn verify_nivc_chain(
    program: &NivcProgram,
    params: &NeoParams,
    chain: &NivcChainProof,
    initial_y: &[F],
) -> anyhow::Result<bool> {
    if program.is_empty() { return Ok(false); }

    // Initialize verifier‑side accumulators (no ME state needed; we rely on inner proofs)
    let mut acc = NivcAccumulators::new(program.len(), initial_y.to_vec());
    acc.step = 0;
    for lane in &mut acc.lanes {
        lane.c_coords.clear();
        lane.c_digest = [0u8; 32];
    }

    // Helper: compute lanes root fields from current accumulator snapshot
    fn lanes_root_fields_from(acc: &NivcAccumulators) -> Vec<F> {
        let mut bytes = Vec::with_capacity(acc.lanes.len() * 32 + 16);
        for lane in &acc.lanes { bytes.extend_from_slice(&lane.c_digest); }
        bytes.extend_from_slice(b"neo/nivc/lanes_root/v1");
        let digest = p2::poseidon2_hash_packed_bytes(&bytes);
        digest.into_iter().map(|g| F::from_u64(g.as_canonical_u64())).collect()
    }

    // Maintain lane-local previous augmented X to enforce LHS linking on repeated lane usage
    let mut prev_aug_x_by_lane: Vec<Option<Vec<F>>> = vec![None; program.len()];

    for sp in &chain.steps {
        let j = sp.which_type;
        if j >= program.len() { return Ok(false); }

        // Lane‑scoped accumulator to feed the existing IVC verifier
        let lane = &acc.lanes[j];
        let prev_acc_lane = Accumulator {
            c_z_digest: lane.c_digest,
            c_coords: lane.c_coords.clone(),
            y_compact: acc.global_y.clone(),
            step: acc.step,
        };

        // Build expected step_x = [H(prev_acc_lane) || which || step_io || lanes_root]
        let acc_prefix = crate::ivc::compute_accumulator_digest_fields(&prev_acc_lane)
            .map_err(|e| anyhow::anyhow!("compute_accumulator_digest_fields failed: {}", e))?;
        let lanes_root = lanes_root_fields_from(&acc);
        let mut expected_app_inputs = Vec::with_capacity(1 + sp.step_io.len() + lanes_root.len());
        expected_app_inputs.push(F::from_u64(j as u64));
        expected_app_inputs.extend_from_slice(&sp.step_io);
        expected_app_inputs.extend_from_slice(&lanes_root);
        // Build expected step_x = [acc_prefix || which || step_io || lanes_root]
        let mut expected_step_x = acc_prefix.clone();
        expected_step_x.extend_from_slice(&expected_app_inputs);

        // Enforce prefix/suffix equality:
        // - prefix must equal H(prev_acc_lane)
        // - suffix must equal [which_type || step_io || lanes_root]
        let step_x = &sp.inner.step_public_input;
        let digest_len = acc_prefix.len();
        if step_x.len() != digest_len + expected_app_inputs.len() {
            return Ok(false);
        }
        if &step_x[..digest_len] != acc_prefix.as_slice() {
            return Ok(false);
        }
        if &step_x[digest_len..] != expected_app_inputs.as_slice() {
            return Ok(false);
        }
        // Redundant but explicit: selector in suffix must match `which`
        let which_in_x = step_x[digest_len].as_canonical_u64() as usize;
        if which_in_x != j { return Ok(false); }

        let ok = crate::ivc::verify_ivc_step(
            &program.steps[j].ccs,
            &sp.inner,
            &prev_acc_lane,
            &program.steps[j].binding,
            params,
            prev_aug_x_by_lane[j].as_deref(),
        ).map_err(|e| anyhow::anyhow!("verify_ivc_step failed: {}", e))?;
        if !ok { return Ok(false); }

        // Update lane commitment and global y from the proof
        let lane_mut = &mut acc.lanes[j];
        lane_mut.c_coords = sp.inner.next_accumulator.c_coords.clone();
        lane_mut.c_digest = sp.inner.next_accumulator.c_z_digest;
        acc.global_y = sp.inner.next_accumulator.y_compact.clone();
        acc.step += 1;

        // Update lane-local previous augmented X for linking next time this lane is used
        prev_aug_x_by_lane[j] = Some(sp.inner.step_augmented_public_input.clone());
    }

    // Final snapshot minimal check (global y and step)
    Ok(acc.global_y == chain.final_acc.global_y && acc.step == chain.final_acc.step)
}

/// Options for NIVC final proof
pub struct NivcFinalizeOptions { pub embed_ivc_ev: bool }

/// Generate a succinct final SNARK proof for the NIVC chain (Stage 5), analogous to ivc_chain.
///
/// Returns: (lean proof, augmented CCS, final public input)
pub fn finalize_nivc_chain_with_options(
    program: &NivcProgram,
    params: &NeoParams,
    chain: NivcChainProof,
    opts: NivcFinalizeOptions,
) -> anyhow::Result<Option<(crate::Proof, neo_ccs::CcsStructure<F>, Vec<F>)>> {
    if chain.steps.is_empty() { return Ok(None); }
    let last = chain.steps.last().unwrap();
    let j = last.which_type;
    anyhow::ensure!(j < program.len(), "invalid which_type in last step");
    let spec = &program.steps[j];

    // Gather data from last step proof
    let rho = if last.inner.step_rho != F::ZERO { last.inner.step_rho } else { F::ONE }; // fallback
    let y_prev = last.inner.step_y_prev.clone();
    let y_next = last.inner.step_y_next.clone();
    let step_x = last.inner.step_public_input.clone();
    let y_len = y_prev.len();

    // Reconstruct augmented CCS used for folding with the chosen step
    let augmented_ccs = crate::ivc::build_augmented_ccs_linked_with_rlc(
        &spec.ccs,
        step_x.len(),
        &spec.binding.y_step_offsets,
        &spec.binding.y_prev_witness_indices,
        &spec.binding.x_witness_indices,
        y_len,
        spec.binding.const1_witness_index,
        None,
    ).map_err(|e| anyhow::anyhow!("Failed to build augmented CCS: {}", e))?;

    // Build final public input for the final SNARK
    let final_public_input = crate::ivc::build_final_snark_public_input(&step_x, rho, &y_prev, &y_next);

    // Extract final running ME for the chosen lane
    let lane = &chain.final_acc.lanes[j];
    let (final_me, final_me_wit) = match (&lane.me, &lane.wit) {
        (Some(me), Some(wit)) => (me, wit),
        _ => anyhow::bail!("No running ME instance available on the chosen lane for final proof"),
    };

    // Bridge adapter: modern → legacy
    let (mut legacy_me, legacy_wit, _pp) = crate::adapt_from_modern(
        std::slice::from_ref(final_me),
        std::slice::from_ref(final_me_wit),
        &augmented_ccs,
        params,
        &[],
        None,
    ).map_err(|e| anyhow::anyhow!("Bridge adapter failed: {}", e))?;

    // Bind proof to augmented CCS + public input
    let context_digest = crate::context_digest_v1(&augmented_ccs, &final_public_input);
    #[allow(deprecated)]
    { legacy_me.header_digest = context_digest; }

    // Compress to lean proof, optionally embedding full IVC verifier checks (EV + linkage + commit-evo)
    let ajtai_pp_arc = std::sync::Arc::new(_pp);
    let lean = if opts.embed_ivc_ev {
        // Expose y_step via linear claims (stable, working path) and embed EV in-circuit.
        anyhow::ensure!(rho != F::ZERO, "ρ is zero; EV embedding not supported");
        let rho_inv = F::ONE / rho;
        let pub_cols = final_public_input.len();
        let m = final_me_wit.Z.cols();
        let mut claims: Vec<crate::OutputClaim<F>> = Vec::with_capacity(y_len);
        for (i, &off) in spec.binding.y_step_offsets.iter().enumerate().take(y_len) {
            let expected = (y_next[i] - y_prev[i]) * rho_inv;
            let k_index = pub_cols + off; // position in [public || witness]
            let weight = crate::expose_z_component(params, m, k_index);
            claims.push(crate::OutputClaim { weight, expected });
        }
        let (mut legacy_me2, legacy_wit2, _pp2) = crate::adapt_from_modern(
            std::slice::from_ref(final_me),
            std::slice::from_ref(final_me_wit),
            &augmented_ccs,
            params,
            &claims,
            None,
        ).map_err(|e| anyhow::anyhow!("Bridge adapter (with EV claims) failed: {}", e))?;
        #[allow(deprecated)]
        { legacy_me2.header_digest = context_digest; }
        let ev_embed = neo_spartan_bridge::IvcEvEmbed { rho, y_prev: y_prev.clone(), y_next: y_next.clone(), y_step_public: None };
        neo_spartan_bridge::compress_me_to_lean_proof_with_pp_and_ev(&legacy_me2, &legacy_wit2, Some(ajtai_pp_arc), Some(ev_embed))?
    } else {
        neo_spartan_bridge::compress_me_to_lean_proof_with_pp(&legacy_me, &legacy_wit, Some(ajtai_pp_arc))?
    };

    let proof = crate::Proof {
        v: 2,
        circuit_key: lean.circuit_key,
        vk_digest: lean.vk_digest,
        public_io: lean.public_io_bytes,
        proof_bytes: lean.proof_bytes,
        public_results: vec![],
        meta: crate::ProofMeta { num_y_compact: last.inner.step_proof.meta.num_y_compact, num_app_outputs: 0 },
    };
    Ok(Some((proof, augmented_ccs, final_public_input)))
}

pub fn finalize_nivc_chain(
    program: &NivcProgram,
    params: &NeoParams,
    chain: NivcChainProof,
) -> anyhow::Result<Option<(crate::Proof, neo_ccs::CcsStructure<F>, Vec<F>)>> {
    finalize_nivc_chain_with_options(program, params, chain, NivcFinalizeOptions { embed_ivc_ev: true })
}

// (Track A combined proof path removed; single-SNARK finalize path remains.)
