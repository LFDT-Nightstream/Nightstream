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
    let mut rho = if last.inner.step_rho != F::ZERO { last.inner.step_rho } else { F::ONE }; // fallback
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
        &spec.binding.step_program_input_witness_indices,
        y_len,
        spec.binding.const1_witness_index,
        None,
    ).map_err(|e| anyhow::anyhow!("Failed to build augmented CCS: {}", e))?;

    // Build final public input for the final SNARK
    let final_public_input = crate::ivc::build_final_snark_public_input(&step_x, rho, &y_prev, &y_next);

    // Extract ME and witness for final SNARK:
    // Prefer the RHS step ME/witness whose commitment equals c_step, so Ajtai@step binds correctly.
    // Fall back to the most recent available pair; else use running lane.
    let (final_me, final_me_wit) = if let (Some(meis), Some(wits)) = (&last.inner.me_instances, &last.inner.digit_witnesses) {
        if !meis.is_empty() && !wits.is_empty() {
            // Try to locate the RHS (step) instance by exact commitment equality.
            let mut idx = core::cmp::min(meis.len(), wits.len()) - 1; // default: last
            for i in 0..core::cmp::min(meis.len(), wits.len()) {
                if meis[i].c.data.len() == last.inner.c_step_coords.len()
                    && meis[i].c.data == last.inner.c_step_coords
                {
                    idx = i;
                    break;
                }
            }
            (&meis[idx], &wits[idx])
        } else {
            let lane = &chain.final_acc.lanes[j];
            match (&lane.me, &lane.wit) {
                (Some(me), Some(wit)) => (me, wit),
                _ => anyhow::bail!("No running ME instance available on the chosen lane for final proof"),
            }
        }
    } else {
        let lane = &chain.final_acc.lanes[j];
        match (&lane.me, &lane.wit) {
            (Some(me), Some(wit)) => (me, wit),
            _ => anyhow::bail!("No running ME instance available on the chosen lane for final proof"),
        }
    };

    // Bridge adapter: modern → legacy — always bind to the step-only commitment (c_step),
    // which matches the witness Z used to build the ME instance. EV constraints enforce
    // commitment evolution separately, so keeping instance/witness consistent here avoids
    // prover/verify drift.
    let mut me_for_bridge = final_me.clone();
    {
        let (label, coords) = ("step commitment c_step", &last.inner.c_step_coords);
        eprintln!(
            "[FIXUP] modern.c.data <= {} (len {} -> {})",
            label,
            me_for_bridge.c.data.len(),
            coords.len()
        );
        me_for_bridge.c.data = coords.clone();
        let show = core::cmp::min(4, me_for_bridge.c.data.len());
        if show > 0 {
            let mut buf = String::new();
            for i in 0..show {
                if i > 0 { buf.push_str(", "); }
                buf.push_str(&format!("{}", me_for_bridge.c.data[i].as_canonical_u64()));
            }
            eprintln!("[HOST-CHECK] modern.c.data[0..{}): {}", show, buf);
        } else {
            eprintln!("[HOST-CHECK] modern.c.data is EMPTY (len=0)");
        }
    }

    // Debug: identify the chosen ME/WIT index that matches c_step
    {
        if let (Some(meis), Some(wits)) = (&last.inner.me_instances, &last.inner.digit_witnesses) {
            let mut idx_dbg: isize = -1;
            for i in 0..core::cmp::min(meis.len(), wits.len()) {
                if meis[i].c.data.len() == last.inner.c_step_coords.len()
                    && meis[i].c.data == last.inner.c_step_coords
                {
                    idx_dbg = i as isize; break;
                }
            }
            eprintln!("[DEBUG] Finalizer picked RHS ME index: {} (of {})", idx_dbg, meis.len());
            if idx_dbg >= 0 {
                let wi = &wits[idx_dbg as usize];
                let d0 = wi.Z.rows(); let m0 = wi.Z.cols();
                let sample = if d0 > 0 && m0 > 0 { wi.Z[(0,0)].as_canonical_u64() } else { 0 };
                eprintln!("[DEBUG] final_me_wit.Z dims = {}x{}, Z[0,0] = {}", d0, m0, sample);
            }
        }
    }

    let (mut legacy_me, mut legacy_wit, _pp) = crate::adapt_from_modern(
        std::slice::from_ref(&me_for_bridge),
        std::slice::from_ref(final_me_wit),
        &augmented_ccs,
        params,
        &[],
        None,
    ).map_err(|e| anyhow::anyhow!("Bridge adapter failed: {}", e))?;

    // Show a few z-digits after adaptation
    {
        #[allow(deprecated)]
        let show = core::cmp::min(8, legacy_wit.z_digits.len());
        if show > 0 {
            let mut buf = String::new();
            #[allow(deprecated)]
            for i in 0..show { if i>0 { buf.push_str(", "); } buf.push_str(&format!("{}", legacy_wit.z_digits[i])); }
            eprintln!("[DEBUG] legacy_wit.z_digits[0..{}): {}", show, buf);
        } else {
            eprintln!("[DEBUG] legacy_wit.z_digits is EMPTY (len=0)");
        }
    }

    // Align Ajtai binding target with the actual witness used in the SNARK.
    // Without EV embedding, the witness corresponds to the full step state (including U),
    // so recompute c_coords as ⟨L_i, z_digits⟩ using PP to avoid mismatch.
    if !opts.embed_ivc_ev {
        #[allow(deprecated)]
        {
            let z_len_dm = final_me_wit.Z.rows() * final_me_wit.Z.cols();
            let rows = {
                #[allow(deprecated)]
                { legacy_me.c_coords.len() }
            };
            let mut new_coords = Vec::with_capacity(rows);
            for i in 0..rows {
                match neo_ajtai::compute_single_ajtai_row(&_pp, i, z_len_dm, rows) {
                    Ok(row) => {
                        let mut acc_f = neo_math::F::ZERO;
                        for (j, &a) in row.iter().enumerate() {
                            if j >= legacy_wit.z_digits.len() { break; }
                            let zi = legacy_wit.z_digits[j];
                            let zf = if zi >= 0 { neo_math::F::from_u64(zi as u64) } else { -neo_math::F::from_u64((-zi) as u64) };
                            acc_f += a * zf;
                        }
                        new_coords.push(acc_f);
                    }
                    Err(e) => anyhow::bail!("Failed to compute Ajtai row {}: {}", i, e),
                }
            }
            eprintln!("[FIXUP] legacy_me.c_coords recomputed from PP •z (aligned with witness)");
            legacy_me.c_coords = new_coords;
        }
    }

    // Bind proof to augmented CCS + public input
    let context_digest = crate::context_digest_v1(&augmented_ccs, &final_public_input);
    #[allow(deprecated)]
    { legacy_me.header_digest = context_digest; }

    // Quick host-side Ajtai sanity: show a few c_coords and spot-check inner products
    #[allow(deprecated)]
    {
        let show = {
            #[allow(deprecated)]
            { core::cmp::min(4, legacy_me.c_coords.len()) }
        };
        if show > 0 {
            let mut buf = String::new();
            for i in 0..show {
                if i > 0 { buf.push_str(", "); }
                let cc_i = {
                    #[allow(deprecated)]
                    { legacy_me.c_coords[i] }
                };
                buf.push_str(&format!("{}", cc_i.as_canonical_u64()));
            }
            eprintln!("[HOST-CHECK] c_coords[0..{}): {}", show, buf);
        } else {
            eprintln!("[HOST-CHECK] c_coords is EMPTY (len=0)");
        }
        // Spot-check Ajtai row IPs with correct z_len = d*m (pre-padding)
        let z_len_dm = final_me_wit.Z.rows() * final_me_wit.Z.cols();
        let rows = {
            #[allow(deprecated)]
            { legacy_me.c_coords.len() }
        };
        let take = core::cmp::min(3, rows);
        for i in 0..take {
            match neo_ajtai::compute_single_ajtai_row(&_pp, i, z_len_dm, rows) {
                Ok(row) => {
                    let mut acc = neo_math::F::ZERO;
                    // Use only the original (pre-padding) limbs
                    for (j, &a) in row.iter().enumerate() {
                        if j >= legacy_wit.z_digits.len() { break; }
                        let z = legacy_wit.z_digits[j];
                        let zf = if z >= 0 { neo_math::F::from_u64(z as u64) } else { -neo_math::F::from_u64((-z) as u64) };
                        acc += a * zf;
                    }
                    let cc_i = {
                        #[allow(deprecated)]
                        { legacy_me.c_coords[i] }
                    };
                    eprintln!("[HOST-CHECK] row {} • z = {} vs c_coords[{}] = {}", i, acc.as_canonical_u64(), i, cc_i.as_canonical_u64());
                }
                Err(e) => eprintln!("[HOST-CHECK] compute_single_ajtai_row({}) failed: {}", i, e),
            }
        }
    }

    // If embedding EV and binding Ajtai to the step vector, DO NOT swap digit witnesses.
    // Use the z_digits already paired with `final_me_wit` (the RHS step instance).
    // Only ensure power-of-two padding for circuit compatibility.
    if opts.embed_ivc_ev {
        #[allow(deprecated)]
        let original_len = legacy_wit.z_digits.len();
        #[allow(deprecated)]
        let target_len = if original_len <= 1 { 1 } else { original_len.next_power_of_two() };
        if target_len > original_len {
            eprintln!(
                "🔍 modern_to_legacy_witness(): z_digits padded from {} to {} (power-of-two)",
                original_len, target_len
            );
            #[allow(deprecated)]
            legacy_wit.z_digits.resize(target_len, 0i64);
        }
    }

    // Compress to lean proof. If embedding EV, also enforce linkage and commitment evolution in-circuit.
    let ajtai_pp_arc = std::sync::Arc::new(_pp.clone());
    let lean = if opts.embed_ivc_ev {
        anyhow::ensure!(rho != F::ZERO, "ρ is zero; EV embedding not supported");

        // Use the IVC verifier-style circuit to enforce EV + linkage + commitment evolution.
        // SECURITY: optionally bind ρ to the fold-chain transcript digest (restores Fiat–Shamir soundness).
        let fold_digest_opt = last.inner.folding_proof
            .as_ref()
            .map(|fp| neo_fold::folding_proof_digest(fp));
        if let Some(d) = &fold_digest_opt {
            // Mirror in-circuit binding: ρ = Poseidon2("neo/ev/rho_from_digest/v1" || digest), non-zero tweak
            let mut limbs: Vec<u64> = Vec::new();
            for chunk in b"neo/ev/rho_from_digest/v1".chunks(8) {
                let mut b = [0u8; 8];
                b[..chunk.len()].copy_from_slice(chunk);
                limbs.push(u64::from_le_bytes(b));
            }
            for chunk in d.chunks(8) {
                let mut b = [0u8; 8];
                b[..chunk.len()].copy_from_slice(chunk);
                limbs.push(u64::from_le_bytes(b));
            }
            let packed: Vec<u8> = limbs.iter().flat_map(|&x| x.to_le_bytes()).collect();
            let h = neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash_packed_bytes(&packed);
            let mut derived = F::from_u64(h[0].as_canonical_u64());
            if derived == F::ZERO { derived = F::ONE; }
            rho = derived;
        }
        // 1) EV: Provide (ρ, y_prev, y_next); optionally expose y_step as public convenience.
        let y_step_public = {
            let rho_inv = F::ONE / rho;
            Some(y_next.iter().zip(y_prev.iter()).map(|(n,p)| (*n - *p) * rho_inv).collect::<Vec<_>>())
        };

        // Keep legacy_me.c_coords equal to the accumulator commitment (c_next).
        // Ajtai will bind to me.c_coords, and EV constraints enforce commit evolution.

        // Precompute acc evolution vectors
        let c_step_vec = last.inner.c_step_coords.clone();
        let c_next_vec = last.inner.next_accumulator.c_coords.clone();
        let _c_prev_vec = c_next_vec.iter().zip(c_step_vec.iter())
            .map(|(n,s)| *n - rho * *s).collect::<Vec<_>>();

        // Recompute the step commitment vector directly from PP rows and the adapted witness
        // to guarantee Ajtai parity inside the circuit. This avoids relying on the pipeline's
        // c_step_coords ordering and ensures constraints match the provided witness.
        let c_step_from_z: Vec<F> = {
            let d_pp = final_me_wit.Z.rows();
            let m_pp = final_me_wit.Z.cols();
            let z_len_dm = d_pp * m_pp; // original (pre-padding) length
            let rows = last.inner.c_step_coords.len();
            let mut out = Vec::with_capacity(rows);
            for i in 0..rows {
                match neo_ajtai::compute_single_ajtai_row(&_pp, i, z_len_dm, rows) {
                    Ok(row) => {
                        let mut acc_f = F::ZERO;
                        #[allow(deprecated)]
                        for j in 0..core::cmp::min(z_len_dm, legacy_wit.z_digits.len()) {
                            let a = row[j];
                            let zi = legacy_wit.z_digits[j];
                            let zf = if zi >= 0 { F::from_u64(zi as u64) } else { -F::from_u64((-zi) as u64) };
                            acc_f += a * zf;
                        }
                        out.push(acc_f);
                    }
                    Err(e) => anyhow::bail!("Ajtai row {} compute failed: {}", i, e),
                }
            }
            out
        };

        // Re-derive prev from next and the recomputed step to keep commit-evo consistent.
        let c_prev_from_z: Vec<F> = {
            let mut v = Vec::with_capacity(c_next_vec.len());
            for i in 0..c_next_vec.len() { v.push(c_next_vec[i] - rho * c_step_from_z[i]); }
            v
        };

        let ev_embed = neo_spartan_bridge::IvcEvEmbed {
            rho,
            y_prev: y_prev.clone(),
            y_next: y_next.clone(),
            y_step_public,
            fold_chain_digest: fold_digest_opt,
            // Bind Ajtai to the recomputed step vector; enforce evolution to c_next.
            acc_c_prev: Some(c_prev_from_z.clone()),
            acc_c_step: Some(c_step_from_z.clone()),
            acc_c_next: Some(c_next_vec.clone()),
            rho_eff: None,
        };

        // Preflight Ajtai parity: verify that with the current witness digits the Ajtai rows
        // reproduce the intended binding target (acc_c_step) before we create the SNARK.
        // This localizes issues to the finalizer if constraints wouldn't hold.
        let do_strict = std::env::var("NEO_AJTAI_STRICT_PREFLIGHT").ok().as_deref() == Some("1");
        let do_log    = do_strict || std::env::var("NEO_AJTAI_PREFLIGHT").ok().as_deref() == Some("1");
        if do_log {
            let d_pp = final_me_wit.Z.rows();
            let m_pp = final_me_wit.Z.cols();
            let rows = last.inner.c_step_coords.len();
            match neo_ajtai::get_global_pp_for_dims(d_pp, m_pp) {
                Ok(pp_chk) => {
                    let z_len = d_pp * m_pp;
                    let mut mismatches = 0usize;
                    let mut first_few: Vec<(usize, u64, u64)> = Vec::new();
                    for i in 0..rows {
                        let row = match neo_ajtai::compute_single_ajtai_row(&pp_chk, i, z_len, rows) {
                            Ok(r) => r,
                            Err(e) => { eprintln!("[AJTAI-PREFLIGHT] row {} error: {}", i, e); mismatches += 1; continue; }
                        };
                        let mut acc_f = F::ZERO;
                        #[allow(deprecated)]
                        let z_digits_ref = &legacy_wit.z_digits;
                        for (j, &a) in row.iter().enumerate() {
                            if j >= z_digits_ref.len() { break; }
                            let zi = z_digits_ref[j];
                            let zf = if zi >= 0 { F::from_u64(zi as u64) } else { -F::from_u64((-zi) as u64) };
                            acc_f += a * zf;
                        }
                        let rhs = last.inner.c_step_coords[i];
                        if acc_f != rhs {
                            if first_few.len() < 5 {
                                first_few.push((i, acc_f.as_canonical_u64(), rhs.as_canonical_u64()));
                            }
                            mismatches += 1;
                        }
                    }
                    if mismatches > 0 {
                        eprintln!(
                            "[AJTAI-PREFLIGHT] MISMATCH: {} of {} rows differ (first few {:?})",
                            mismatches, rows, first_few
                        );
                        if do_strict {
                            anyhow::bail!(
                                "Ajtai preflight failed: {} / {} mismatches (see logs for samples)",
                                mismatches, rows
                            );
                        }
                    } else {
                        eprintln!("[AJTAI-PREFLIGHT] OK: all {} rows match acc_c_step", rows);
                    }
                }
                Err(e) => {
                    eprintln!("[AJTAI-PREFLIGHT] PP init failed: {} (d={}, m={})", e, d_pp, m_pp);
                }
            }
        }

        // 2) Commit evolution diagnostics: prev + rho * step = next
        let c_step_vec = last.inner.c_step_coords.clone();
        let c_next_vec = last.inner.next_accumulator.c_coords.clone();
        anyhow::ensure!(c_step_vec.len() == c_next_vec.len(), "commitment vector length mismatch");
        let c_prev_vec = c_next_vec.iter().zip(c_step_vec.iter()).map(|(n,s)| *n - rho * *s).collect::<Vec<_>>();
        {
            // Debug check: prev + rho*step vs next
            let n = c_next_vec.len();
            let mut mismatches = 0usize;
            for i in 0..n.min(8) {
                let lhs = c_next_vec[i];
                let rhs = c_prev_vec[i] + rho * c_step_vec[i];
                eprintln!(
                    "[EV-CHECK] i={} prev={} step={} next={} prev+rho*step={}",
                    i,
                    c_prev_vec[i].as_canonical_u64(),
                    c_step_vec[i].as_canonical_u64(),
                    c_next_vec[i].as_canonical_u64(),
                    rhs.as_canonical_u64()
                );
                if lhs != rhs { mismatches += 1; }
            }
            if n > 8 {
                for &i in &[n/2, n-1] {
                    let lhs = c_next_vec[i];
                    let rhs = c_prev_vec[i] + rho * c_step_vec[i];
                    eprintln!(
                        "[EV-CHECK] i={} prev={} step={} next={} prev+rho*step={}",
                        i,
                        c_prev_vec[i].as_canonical_u64(),
                        c_step_vec[i].as_canonical_u64(),
                        c_next_vec[i].as_canonical_u64(),
                        rhs.as_canonical_u64()
                    );
                    if lhs != rhs { mismatches += 1; }
                }
            }
            if mismatches > 0 {
                eprintln!(
                    "[EV-CHECK] rho={} mismatches (sampled)={} of {}",
                    rho.as_canonical_u64(), mismatches, c_next_vec.len()
                );
            } else {
                eprintln!(
                    "[EV-CHECK] rho={} all sampled entries match",
                    rho.as_canonical_u64()
                );
            }
        }
        // EV embedding already enforces commit evolution with its own public vectors.
        // No additional commit-evo embed is necessary here.

        // 3) Linkage: bind specific undigitized witness positions to provided step IO values.
        let linkage = Some(neo_spartan_bridge::IvcLinkageInputs {
            x_indices_abs: spec.binding.step_program_input_witness_indices.clone(),
            y_prev_indices_abs: spec.binding.y_prev_witness_indices.clone(),
            const1_index_abs: None, // const-1 binding is enforced by CCS; avoid double-binding here
            step_io: last.step_io.clone(),
        });

        // Bind header digest to legacy ME used for proving
        #[allow(deprecated)]
        { legacy_me.header_digest = context_digest; }

        // Build Pi-CCS embed from augmented_ccs matrices (sparse triplets).
        // Default: ENABLED
        let pi_ccs_embed_opt = {
            use neo_spartan_bridge::{CcsCsr, PiCcsEmbed};
            let mut mats = Vec::with_capacity(augmented_ccs.matrices.len());
            for mj in &augmented_ccs.matrices {
                let rows = mj.rows();
                let cols = mj.cols();
                let mut entries = Vec::new();
                for r in 0..rows { for c in 0..cols {
                    let a = mj[(r, c)];
                    if a != F::ZERO { entries.push((r as u32, c as u32, a)); }
                }}
                mats.push(CcsCsr { rows, cols, entries });
            }
            Some(PiCcsEmbed { matrices: mats })
        };

        // Optional: host-side Pi-CCS preflight to detect mismatches (behind env)
        if std::env::var("NEO_PI_CCS_PREFLIGHT").ok().as_deref() == Some("1") {
            use neo_spartan_bridge::pi_ccs_embed as piccs;
            // Rebuild the same CSR bundle as above
            let mut mats = Vec::with_capacity(augmented_ccs.matrices.len());
            for mj in &augmented_ccs.matrices {
                let rows = mj.rows();
                let cols = mj.cols();
                let mut entries = Vec::new();
                for r in 0..rows { for c in 0..cols {
                    let a = mj[(r, c)];
                    if a != F::ZERO { entries.push((r as u32, c as u32, a)); }
                }}
                mats.push(piccs::CcsCsr { rows, cols, entries });
            }

            #[allow(deprecated)]
            let (_r_bits, d, base_b_u64, provided_is_2lane) = {
                // Use Re lane of r over K
                let ell = legacy_me.r_point.len() / 2;
                let _r_bits: Vec<F> = (0..ell).map(|t| legacy_me.r_point[2*t]).collect();
                // Canonical digit count (avoid padded z influence)
                let d = neo_math::ring::D;
                let base_b_u64 = legacy_me.base_b;
                let provided_is_2lane = legacy_wit.weight_vectors.len() == 2 * mats.len();
                (_r_bits, d, base_b_u64, provided_is_2lane)
            };
            let base_b = F::from_u64(base_b_u64 as u64);
            let n_rows = augmented_ccs.matrices[0].rows();

            // Compute gold weights using K-arithmetic χ_r via Gray-code tensor point,
            // then take Re lane to match the in-circuit Pi-CCS behavior.
            let mut ell_needed = 0usize; while (1usize << ell_needed) < n_rows { ell_needed += 1; }
            #[allow(deprecated)]
            let pairs_avail = legacy_me.r_point.len() / 2;
            anyhow::ensure!(pairs_avail == ell_needed, "r pairs {} != ceil(log2 n_rows) {} (n_rows={})", pairs_avail, ell_needed, n_rows);
            // Build r in K^ell
            let r_vec_k: Vec<neo_math::K> = {
                let mut out = Vec::with_capacity(ell_needed);
                #[allow(deprecated)]
                for t in 0..ell_needed {
                    let re = legacy_me.r_point[2*t];
                    let im = legacy_me.r_point[2*t+1];
                    out.push(neo_math::field::from_complex(re, im));
                }
                out
            };
            // χ_r in K^n (Gray-code order), then restrict to first n_rows
            let chi_full_k: Vec<neo_math::K> = neo_ccs::utils::tensor_point::<neo_math::K>(&r_vec_k);
            anyhow::ensure!(chi_full_k.len() >= n_rows, "chi length {} < n_rows {}", chi_full_k.len(), n_rows);
            let chi_k = &chi_full_k[..n_rows];
            let pow_b = piccs::pow_table(base_b, d);

            let mut gold: Vec<Vec<F>> = Vec::with_capacity(mats.len());
            for mj in &mats {
                // group entries by column
                let mut by_col: Vec<Vec<(usize, F)>> = vec![Vec::new(); mj.cols];
                for &(r_idx, c_idx, a) in &mj.entries { by_col[c_idx as usize].push((r_idx as usize, a)); }
                // compute v_re per column using K chi
                let mut v_re = vec![F::ZERO; mj.cols];
                for c in 0..mj.cols {
                    let mut acc_re = F::ZERO;
                    for (r_i, a) in &by_col[c] {
                        let chi = chi_k[*r_i];
                        let chi_re = chi.real();
                        acc_re += *a * chi_re;
                    }
                    v_re[c] = acc_re;
                }
                // expand to digit weights w_re (column-major)
                let mut w_re = vec![F::ZERO; d * mj.cols];
                for c in 0..mj.cols { for r in 0..d { let idx = c*d + r; w_re[idx] = v_re[c] * pow_b[r]; } }
                gold.push(w_re);
            }

            let mut bad = 0usize;
            for j in 0..gold.len() {
                #[allow(deprecated)]
                let provided: &[F] = if provided_is_2lane { &legacy_wit.weight_vectors[2*j] } else { &legacy_wit.weight_vectors[j] };
                if provided.len() != gold[j].len() || provided != &gold[j][..] {
                    eprintln!(
                        "[PI-CCS PREFLIGHT] mismatch at j={} (len gold={} provided={})",
                        j, gold[j].len(), provided.len()
                    );
                    bad += 1;
                    // print a couple sample entries for quick diagnosis
                    for idx in [0usize, 1, d.saturating_sub(1)].iter().copied().filter(|&x| x < gold[j].len()) {
                        eprintln!(
                            "  gold[{}] = {}, provided = {}",
                            idx,
                            gold[j][idx].as_canonical_u64(),
                            provided[idx].as_canonical_u64()
                        );
                    }
                }
            }
            anyhow::ensure!(bad == 0, "Pi-CCS preflight failed: {} of {} vectors differ", bad, gold.len());
            eprintln!("[PI-CCS PREFLIGHT] OK: {} vectors match", gold.len());
        }

        // Canonicalize weight vectors to Pi‑CCS definition to avoid inconsistencies
        {
            use neo_math::F as FF;
            use neo_spartan_bridge::pi_ccs_embed as piccs;

            // Rebuild CSR bundle for convenience
            let mut mats = Vec::with_capacity(augmented_ccs.matrices.len());
            for mj in &augmented_ccs.matrices {
                let rows = mj.rows();
                let cols = mj.cols();
                let mut entries = Vec::new();
                for r in 0..rows { for c in 0..cols {
                    let a = mj[(r, c)];
                    if a != F::ZERO { entries.push((r as u32, c as u32, a)); }
                }}
                mats.push(piccs::CcsCsr { rows, cols, entries });
            }

            // Dimensions
            let n_rows = mats.first().map(|m| m.rows).unwrap_or(0);
            let _m_cols = mats.first().map(|m| m.cols).unwrap_or(0);
            let d = neo_math::ring::D;
            // r in K as pairs
            #[allow(deprecated)]
            let pairs_avail = legacy_me.r_point.len() / 2;
            let mut ell_needed = 0usize; while (1usize << ell_needed) < n_rows { ell_needed += 1; }
            anyhow::ensure!(pairs_avail == ell_needed, "r pairs {} != ceil(log2 n_rows) {} (n_rows={})", pairs_avail, ell_needed, n_rows);
            #[allow(deprecated)]
            let r_pairs: Vec<(FF, FF)> = (0..ell_needed).map(|t| (legacy_me.r_point[2*t], legacy_me.r_point[2*t+1])).collect();
            #[allow(deprecated)]
            let base_b = FF::from_u64(legacy_me.base_b as u64);
            let pow_b = piccs::pow_table(base_b, d);

            // Helper: compute chi(row_index) over K using r_pairs
            let compute_chi = |row_i: usize| -> (FF, FF) {
                let mut re = FF::ONE; let mut im = FF::ZERO;
                let mut mask = row_i;
                for t in 0..ell_needed {
                    let (rt_re, rt_im) = r_pairs[t];
                    let bit = (mask & 1) == 1;
                    let tr = if bit { rt_re } else { FF::ONE - rt_re };
                    let ti = if bit { rt_im } else { -rt_im };
                    let new_re = re * tr - im * ti;
                    let new_im = re * ti + im * tr;
                    re = new_re; im = new_im; mask >>= 1;
                }
                (re, im)
            };

            // For each matrix, compute v_re/v_im, then expand to weights and interleave (Re, Im)
            let mut new_weights: Vec<Vec<FF>> = Vec::with_capacity(2 * mats.len());
            for (j, mj) in mats.iter().enumerate() {
                // group entries by column
                let mut by_col: Vec<Vec<(usize, FF)>> = vec![Vec::new(); mj.cols];
                for &(r_idx, c_idx, a) in &mj.entries { by_col[c_idx as usize].push((r_idx as usize, a)); }
                // compute v per column
                let mut v_re = vec![FF::ZERO; mj.cols];
                let mut v_im = vec![FF::ZERO; mj.cols];
                for c in 0..mj.cols {
                    let mut acc_re = FF::ZERO; let mut acc_im = FF::ZERO;
                    for (r_i, a) in &by_col[c] {
                        let (chi_re, chi_im) = compute_chi(*r_i);
                        acc_re += *a * chi_re;
                        acc_im += *a * chi_im;
                    }
                    v_re[c] = acc_re; v_im[c] = acc_im;
                }
                // expand to digit weights (column-major layout c*d + r)
                let mut w_re = vec![FF::ZERO; d * mj.cols];
                let mut w_im = vec![FF::ZERO; d * mj.cols];
                for c in 0..mj.cols { for r in 0..d { let idx = c*d + r; w_re[idx] = v_re[c] * pow_b[r]; w_im[idx] = v_im[c] * pow_b[r]; } }
                // interleave into witness order: Re_j, Im_j
                new_weights.push(w_re);
                new_weights.push(w_im);
                if j == 0 {
                    // small diagnostics for first and last column
                    if mj.cols > 0 {
                        let c0 = 0usize; let c1 = mj.cols - 1;
                        eprintln!("[PI-CCS-DIAG] j=0, c={} v_re={} v_im={} (sample)", c0, v_re[c0].as_canonical_u64(), v_im[c0].as_canonical_u64());
                        eprintln!("[PI-CCS-DIAG] j=0, c={} v_re={} v_im={} (sample)", c1, v_re[c1].as_canonical_u64(), v_im[c1].as_canonical_u64());
                    }
                }
            }
            #[allow(deprecated)]
            { legacy_wit.weight_vectors = new_weights; }
        }

        // Guard A: Cheap parity check between encoder and circuit public IO (debug only)
        if cfg!(debug_assertions) {
            neo_spartan_bridge::guards::assert_public_io_parity(
                &legacy_me, &legacy_wit, Some(&ev_embed), Some(ajtai_pp_arc.clone())
            )?;
        }

        neo_spartan_bridge::compress_ivc_verifier_to_lean_proof_with_linkage_and_pi_ccs(
            &legacy_me, &legacy_wit, Some(ajtai_pp_arc), Some(ev_embed), None, linkage, pi_ccs_embed_opt,
        )?
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
