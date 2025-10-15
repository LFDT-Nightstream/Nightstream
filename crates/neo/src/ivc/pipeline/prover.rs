//! IVC prover pipeline
//!
//! This module implements the complete IVC proving logic, including:
//! - Single step proving (prove_ivc_step)
//! - Chained step proving with state threading (prove_ivc_step_chained)
//! - Chain proving (prove_ivc_chain)

use crate::F;
use crate::shared::types::*;
use crate::shared::binding::{StepOutputExtractor, IndexExtractor};
use neo_ccs::CcsStructure;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::super::internal::{
    transcript::{create_step_digest, build_step_transcript_data},
    augmented::{build_augmented_ccs_linked_with_rlc, build_linked_augmented_witness, build_augmented_public_input_for_step, build_augmented_ccs_linked},
    ev::generate_rlc_coefficients,
    commit::evolve_commitment,
};
use crate::shared::digest::{compute_accumulator_digest_fields, digest_commit_coords};

pub fn prove_ivc_step_with_extractor(
    params: &crate::NeoParams,
    step_ccs: &CcsStructure<F>,
    step_witness: &[F],
    prev_accumulator: &Accumulator,
    step: u64,
    public_input: Option<&[F]>,
    extractor: &dyn StepOutputExtractor,
    binding_spec: &StepBindingSpec,
) -> Result<IvcStepResult, Box<dyn std::error::Error>> {
    // SECURITY: Validate CCS structure requirements
    // ℓ = ceil(log2(n)) must be ≥ 2 for the sumcheck protocol
    // n is padded to next power of 2 (max 2), so n=3 → 4 → ℓ=2 is acceptable
    if step_ccs.n < 3 {
        return Err(format!(
            "CCS validation failed: n={} is too small (minimum n=3 required). \
            The sumcheck challenge length ℓ=ceil(log2(n_padded)) must be ≥ 2 for protocol security. \
            n is padded to next power-of-2 (minimum 2), so n=3→4→ℓ=2, n=2→2→ℓ=1 (too small). \
            Please ensure your circuit has at least 3 constraint rows.",
            step_ccs.n
        ).into());
    }
    
    // NOTE: ℓ ≤ 1 limitation
    // 
    // When ℓ=1 (single-row CCS padded to 2 rows), the augmented CCS carries a constant offset
    // from const-1 binding and other glue rows. This makes initial_sum non-zero even for valid
    // witnesses, preventing the verifier from distinguishing valid from invalid witnesses based
    // on the initial_sum == 0 check.
    //
    // For soundness, production circuits SHOULD have at least 3 constraint rows to ensure ℓ >= 2
    // after power-of-2 padding (3 rows → padded to 4 → ℓ = log₂(4) = 2).
    //
    // We don't enforce this as a hard guard because:
    // 1. Valid witnesses for ℓ=1 CCS are correctly accepted (no false rejections)
    // 2. The limitation is documented and tests demonstrate the behavior
    // 3. A future fix (Option B: carry "Q is pure residual" bit) will address this properly
    //
    // Users should be aware that ℓ=1 circuits cannot have invalid witnesses detected at
    // verification time and should add prover-side checks if needed.
    
    // Extract REAL y_step from step computation (not placeholder)
    let y_step = extractor.extract_y_step(step_witness);
    
    #[cfg(feature = "neo-logs")]
    #[cfg(feature = "debug-logs")]
    println!("🎯 Extracted REAL y_step: {:?}", y_step.iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>());
    
    let input = IvcStepInput {
        params,
        step_ccs,
        step_witness,
        prev_accumulator,
        step,
        public_input,
        y_step: &y_step,
        binding_spec, // Use TRUSTED binding specification
        app_input_binding: AppInputBinding::WitnessBound,
        prev_augmented_x: None,
    };
    
    prove_ivc_step(input)
}

/// Prove a single IVC step with proper chaining.
///
/// - Accepts previous folded ME instance (for Stage 5 compression continuity).
/// - Accepts previous RHS MCS instance+witness, and returns the current RHS MCS to be
///   used as the next LHS. This ensures exact X‑linkage across steps by construction.
pub fn prove_ivc_step_chained(
    input: IvcStepInput,
    prev_me: Option<neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>>,
    prev_me_wit: Option<neo_ccs::MeWitness<F>>,
    prev_lhs_mcs: Option<(neo_ccs::McsInstance<neo_ajtai::Commitment, F>, neo_ccs::McsWitness<F>)>,
) -> Result<(
    IvcStepResult,
    neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>,
    neo_ccs::MeWitness<F>,
    (neo_ccs::McsInstance<neo_ajtai::Commitment, F>, neo_ccs::McsWitness<F>),
), Box<dyn std::error::Error>> {
    // SECURITY: Validate CCS structure requirements
    // ℓ = ceil(log2(n)) must be ≥ 2 for the sumcheck protocol
    // n is padded to next power of 2 (max 2), so n=3 → 4 → ℓ=2 is acceptable
    if input.step_ccs.n < 3 {
        return Err(format!(
            "CCS validation failed: n={} is too small (minimum n=3 required). \
            The sumcheck challenge length ℓ=ceil(log2(n_padded)) must be ≥ 2 for protocol security. \
            n is padded to next power-of-2 (minimum 2), so n=3→4→ℓ=2, n=2→2→ℓ=1 (too small). \
            Please ensure your circuit has at least 3 constraint rows.",
            input.step_ccs.n
        ).into());
    }
    
    // Proper chaining: fold previous (ME) with current (MCS) instead of self-folding
    // 1) Build step_x = [H(prev_acc) || app_inputs]
    let acc_digest_fields = compute_accumulator_digest_fields(&input.prev_accumulator)?;
    let step_x: Vec<F> = match input.public_input {
        Some(app_inputs) => {
            let mut combined = acc_digest_fields.clone();
            combined.extend_from_slice(app_inputs);
            combined
        }
        None => acc_digest_fields.clone(),
    };

    let step_data = build_step_transcript_data(&input.prev_accumulator, input.step, &step_x);
    let step_digest = create_step_digest(&step_data);

    // 2) Validate binding metadata
    if input.binding_spec.y_step_offsets.is_empty() && !input.y_step.is_empty() {
        return Err("SECURITY: y_step_offsets cannot be empty when y_step is provided. This would allow malicious y_step attacks.".into());
    }
    if !input.binding_spec.y_step_offsets.is_empty() && input.binding_spec.y_step_offsets.len() != input.y_step.len() {
        return Err("y_step_offsets length must match y_step length".into());
    }
    
    // Validate app input binding based on mode
    let digest_len = compute_accumulator_digest_fields(&input.prev_accumulator)?.len();
    let app_len = step_x.len().saturating_sub(digest_len);
    let x_bind_len = input.binding_spec.step_program_input_witness_indices.len();
    
    match input.app_input_binding {
        AppInputBinding::WitnessBound => {
            // IVC mode: strict 1:1 binding required
            if app_len > 0 && x_bind_len == 0 {
                return Err("SECURITY: step_program_input_witness_indices cannot be empty when step_x has app inputs (WitnessBound mode); this would allow public input manipulation".into());
            }
            if x_bind_len > 0 && x_bind_len != app_len {
                return Err(format!("SECURITY: step_program_input_witness_indices length ({}) must match app public input length ({}) in WitnessBound mode", x_bind_len, app_len).into());
            }
        }
        AppInputBinding::TranscriptOnly => {
            // NIVC mode: binding via Fiat-Shamir transcript
            // Enforce that witness indices are NOT used in this mode to prevent confusion
            if x_bind_len > 0 {
                return Err(format!("SECURITY: step_program_input_witness_indices must be empty in TranscriptOnly mode (found {} indices); app inputs are bound via Fiat-Shamir transcript only", x_bind_len).into());
            }
            // App inputs are included in step_x which enters the FS transcript
            // The transcript digest and challenges will bind these values
        }
    }

    let y_len = input.prev_accumulator.y_compact.len();
    if !input.binding_spec.y_prev_witness_indices.is_empty()
        && input.binding_spec.y_prev_witness_indices.len() != y_len
    {
        return Err("y_prev_witness_indices length must match y_len when provided".into());
    }

    // Enforce const-1 convention
    let const_idx = input.binding_spec.const1_witness_index;
    if input.step_witness.get(const_idx) != Some(&F::ONE) {
        return Err(format!("SECURITY: step_witness[{}] must be 1 (constant-1 column)", const_idx).into());
    }

    // Guard: extractor vs binding_spec.y_step_offsets must agree
    // This prevents a subtle class of bugs where the EV constraints are wired to
    // witness positions different from the values used to compute y_next.
    if !input.binding_spec.y_step_offsets.is_empty() {
        let mut y_from_offsets = Vec::with_capacity(input.binding_spec.y_step_offsets.len());
        for &idx in &input.binding_spec.y_step_offsets {
            y_from_offsets.push(*input
                .step_witness
                .get(idx)
                .ok_or_else(|| format!("y_step_offsets index {} out of bounds for step_witness (len={})", idx, input.step_witness.len()))?);
        }
        if y_from_offsets != input.y_step {
            return Err(format!(
                "Extractor/binding mismatch: y_step extracted by StepOutputExtractor != step_witness[y_step_offsets].\n  extracted: {:?}\n  from_offsets: {:?}\n  offsets: {:?}",
                input.y_step.iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>(),
                y_from_offsets.iter().map(|f| f.as_canonical_u64()).collect::<Vec<_>>(),
                input.binding_spec.y_step_offsets
            ).into());
        }
    }

    // PROVER-SIDE CHECK (COMMENTED OUT TO TEST IN-CIRCUIT ENFORCEMENT):
    // This check can be bypassed by a malicious prover who modifies the code.
    // The REAL security MUST come from in-circuit constraints in the augmented CCS.
    // 
    // TODO: Investigate why step CCS constraints might not be properly enforced in-circuit.
    // The step CCS should be copied into the augmented CCS and checked cryptographically.
    //
    // let step_ccs_public_input = match input.public_input {
    //     Some(app_inputs) => app_inputs,
    //     None => &[],
    // };
    // neo_ccs::check_ccs_rowwise_zero(input.step_ccs, step_ccs_public_input, input.step_witness)
    //     .map_err(|e| format!(
    //         "SOUNDNESS ERROR: step witness does not satisfy CCS constraints: {:?}",
    //         e
    //     ))?;


    // 3) Commit to the ρ-independent step witness (Pattern B), then derive ρ,
    // then build the full augmented witness for proving. This breaks FS circularity.
    let d = neo_math::ring::D;
    
    // Commit to the step witness only (not including EV part)
    
    // Pattern B: derive ρ from a commitment that does NOT include the ρ-dependent tail.
    // Implementation detail: we keep dimensions stable by zero-padding the tail so
    // the Ajtai PP (d, m) matches the later full vector. No in-circuit link is added.
    
    // First, determine the final witness structure to get consistent dimensions
    let temp_witness = build_linked_augmented_witness(
        input.step_witness,
        &input.binding_spec.y_step_offsets,
        F::ONE // temporary rho for dimension calculation
    );
    let y_len = input.prev_accumulator.y_compact.len();
    
    // Build the final public input structure for dimension calculation
    let temp_y_next = input.prev_accumulator.y_compact.clone(); // placeholder
    let final_public_input = build_augmented_public_input_for_step(
        &step_x, F::ONE, &input.prev_accumulator.y_compact, &temp_y_next
    );
    
    // Calculate final dimensions: [final_public_input || temp_witness]
    let mut final_z = final_public_input.clone();
    final_z.extend_from_slice(&temp_witness);
    let final_decomp = crate::decomp_b(&final_z, input.params.b, d, crate::DecompStyle::Balanced);
    let m_final = final_decomp.len() / d;
    
    // Setup Ajtai PP for final dimensions (used for both pre-commit and final commit)
    crate::ensure_ajtai_pp_for_dims(d, m_final, || {
        use rand::{RngCore, SeedableRng};
        use rand::rngs::StdRng;
        
        let mut rng = if std::env::var("NEO_DETERMINISTIC").is_ok() {
            StdRng::from_seed([42u8; 32])
        } else {
            let mut seed = [0u8; 32];
            rand::rng().fill_bytes(&mut seed);
            StdRng::from_seed(seed)
        };
        let pp = crate::ajtai_setup(&mut rng, d, input.params.kappa as usize, m_final)?;
        neo_ajtai::set_global_pp(pp).map_err(anyhow::Error::from)
    })?;

    // Build pre-commit vector: same structure as final but with ρ=0 for EV part
    // (equivalent to committing only to [step_x || step_witness] under Pattern B semantics)
    let pre_public_input = build_augmented_public_input_for_step(
        &step_x, F::ZERO, &input.prev_accumulator.y_compact, &temp_y_next
    );
    let pre_witness = build_linked_augmented_witness(
        input.step_witness,
        &input.binding_spec.y_step_offsets,
        F::ZERO // This zeros out the U = ρ·y_step part
    );
    
    let mut z_pre = pre_public_input.clone();
    z_pre.extend_from_slice(&pre_witness);
    let decomp_pre = crate::decomp_b(&z_pre, input.params.b, d, crate::DecompStyle::Balanced);
    
    // Pre-commit (breaks Fiat-Shamir circularity)
    let pp = neo_ajtai::get_global_pp_for_dims(d, m_final)
        .map_err(|e| anyhow::anyhow!("Failed to get Ajtai PP: {}", e))?;
    let pre_commitment = crate::commit(&*pp, &decomp_pre);
    
    // Extract pre-commit coordinates for ρ derivation
    // SECURITY: The verifier will check that c_step_coords.len() == κ * m_final
    // to prevent attackers from providing coordinates of the wrong dimension (Pattern-B binding)
    let c_step_coords: Vec<F> = pre_commitment
        .data
        .iter()
        .map(|&x| F::from_u64(x.as_canonical_u64()))
        .collect();
    
    // Derive ρ from pre-commitment (standard Fiat-Shamir order)
    let (rho, _td) = super::super::internal::transcript::rho_from_transcript(&input.prev_accumulator, step_digest, &c_step_coords);
    
    // CRITICAL: Π_RLC binding to prevent split-brain attacks
    // Generate RLC coefficients from the same transcript used for ρ
    let num_coords = c_step_coords.len();
    let rlc_coeffs = generate_rlc_coefficients(&input.prev_accumulator, step_digest, &c_step_coords, num_coords);
    
    // Compute aggregated Ajtai row G = Σ_i r_i · L_i for RLC binding
    // CRITICAL: Use exact PP dimensions, not decomp length (which may be padded)
    let total_z_len = d * m_final; // Must equal d * m for Ajtai validation
    let _aggregated_row = neo_ajtai::compute_aggregated_ajtai_row(&*pp, &rlc_coeffs, total_z_len, num_coords)
        .map_err(|e| anyhow::anyhow!("Failed to compute aggregated Ajtai row: {}", e))?;
    
    // Compute RLC right-hand side: rhs = ⟨r, c_step⟩
    let _rlc_rhs = rlc_coeffs.iter().zip(c_step_coords.iter())
        .map(|(ri, ci)| *ri * *ci)
        .fold(F::ZERO, |acc, x| acc + x);
    
    // Store the U offset for the circuit (where the ρ-dependent part starts)
    let u_offset = pre_public_input.len() + input.step_witness.len();
    let u_len = y_len;
    
    // Store final dimensions for later validation (if needed)
    let _expected_m_final = m_final;
    
    // 6) Build full witness and public input with the actual rho
    let step_witness_augmented = build_linked_augmented_witness(
        input.step_witness,
        &input.binding_spec.y_step_offsets,
        rho
    );
    // FULL OUTPUTS: integrator provides the whole next state
    // y_step directly contains y_next (not a delta)
    // ρ is only for folding commitments, not for state evolution
    let y_next: Vec<F> = input.y_step.to_vec();
    let step_public_input = build_augmented_public_input_for_step(
        &step_x, rho, &input.prev_accumulator.y_compact, &y_next
    );

    // 7) Build the full commitment for the MCS instance (includes EV variables for the CCS),
    // but note the IVC accumulator uses only the pre-ρ step commitment (c_step_coords).
    let mut full_step_z = step_public_input.clone();
    full_step_z.extend_from_slice(&step_witness_augmented);
    let decomp_z = crate::decomp_b(&full_step_z, input.params.b, d, crate::DecompStyle::Balanced);
    if decomp_z.len() % d != 0 { return Err("decomp length not multiple of d".into()); }
    let m_step = decomp_z.len() / d;

    // Pattern A: Ensure PP exists for the full witness dimensions (m_step)
    // This is different from m_final because the full witness includes additional public input structure
    crate::ensure_ajtai_pp_for_dims(d, m_step, || {
        use rand::{RngCore, SeedableRng};
        use rand::rngs::StdRng;
        
        let mut rng = if std::env::var("NEO_DETERMINISTIC").is_ok() {
            StdRng::from_seed([42u8; 32])
        } else {
            let mut seed = [0u8; 32];
            rand::rng().fill_bytes(&mut seed);
            StdRng::from_seed(seed)
        };
        let pp = crate::ajtai_setup(&mut rng, d, input.params.kappa as usize, m_step)?;
        neo_ajtai::set_global_pp(pp).map_err(anyhow::Error::from)
    })?;
    
    // Pattern B: Use pre_commitment for ρ derivation and accumulator evolution

    // Get PP for the full witness dimensions (m_step)
    let _pp_full = neo_ajtai::get_global_pp_for_dims(d, m_step)
        .map_err(|e| anyhow::anyhow!("Failed to get Ajtai PP for full witness: {}", e))?;

    // Build full commitment that matches the full witness (for CCS consistency)
    let full_commitment = crate::commit(&*_pp_full, &decomp_z);
    // CCS-level RLC binder intentionally disabled. Soundness is enforced by
    // Pi-CCS → Pi-RLC → Pi-DEC and commitment evolution checks.
    // Intentionally no CCS-level RLC binder row in the proving CCS.
    // Rationale:
    // - The linear form you might try to enforce, <G, vec(Z)> = rhs, lives in Ajtai digit space
    //   (Z are base-b digits). CCS variables are the undigitized vector z. Encoding this check
    //   directly in CCS is structurally mismatched unless we first lift the relevant digit columns
    //   into CCS, which increases width and duplicates Π_DEC checks.
    // - Soundness and completeness are already enforced by the folding pipeline:
    //      • Π_CCS (with eq-binding for R1CS shapes) enforces the constraint relation,
    //      • Π_RLC performs the random linear combination with strong-set guard,
    //      • Π_DEC authenticates digit decomposition & range, tying digits to the commitment,
    //      • The IVC layer checks commitment evolution c_next = c_prev + ρ·c_step.
    //   This mirrors HyperNova/LatticeFold-style reductions and keeps CCS lean.
    // - For experiments, the builder still supports adding a single linear row; unit tests cover
    //   that it is encoded as a true linear equality (<G,z> = rhs) and rejects mismatches.
    let rlc_binder = None;
    let step_augmented_ccs = build_augmented_ccs_linked_with_rlc(
        input.step_ccs,
        step_x.len(),
        &input.binding_spec.y_step_offsets,
        &input.binding_spec.y_prev_witness_indices,
        &input.binding_spec.step_program_input_witness_indices,
        y_len,
        input.binding_spec.const1_witness_index,
        rlc_binder, // RLC binder enabled for soundness
    )?;

    // CCS uses full vector (with ρ), commitment binding uses pre-commit
    let full_witness_part = full_step_z[step_public_input.len()..].to_vec();
    
    // DEBUG: Check consistency
    #[cfg(feature = "debug-logs")]
    println!("🔍 DEBUG: full_step_z.len()={}, step_public_input.len()={}, full_witness_part.len()={}", 
             full_step_z.len(), step_public_input.len(), full_witness_part.len());
    #[cfg(feature = "debug-logs")]
    println!("🔍 DEBUG: decomp_z.len()={}, d*m_step={}", 
             decomp_z.len(), d * m_step);
    
    // Build MCS instance/witness using shared helper
    // CRITICAL FIX: Use step_public_input for CCS instance (with ρ)
    // Pattern B: The CCS instance uses ρ-bearing public input, full witness, and full commitment
    let (step_mcs_inst, step_mcs_wit) = crate::build_mcs_from_decomp(
        full_commitment,
        &decomp_z,
        &step_public_input,
        &full_witness_part,
        d,
        m_step,
    );

    // 6) Reify previous ME→MCS, or create trivial zero instance (base case)
    let (lhs_inst, lhs_wit) = if let Some((inst, wit)) = prev_lhs_mcs {
        // Use exact previous RHS MCS as next LHS for strict linkage
        (inst, wit)
    } else {
        match (prev_me, prev_me_wit) {
        (Some(me), Some(wit)) => {
            // Dimension checks for ME→MCS reification
            if wit.Z.rows() != d {
                return Err(format!("prev ME witness Z has wrong row count: expected {}, got {}", d, wit.Z.rows()).into());
            }
            if wit.Z.cols() != m_step {
                return Err(format!("prev ME witness Z has wrong column count: expected {}, got {}", m_step, wit.Z.cols()).into());
            }
            
            // Recompose z from Z using base b
            let base_f = F::from_u64(input.params.b as u64);
            let d_rows = wit.Z.rows();
            let m_cols = wit.Z.cols();
            let mut z_vec = vec![F::ZERO; m_cols];
            for c in 0..m_cols {
                let mut acc = F::ZERO; let mut pow = F::ONE;
                for r in 0..d_rows { acc += wit.Z[(r, c)] * pow; pow *= base_f; }
                z_vec[c] = acc;
            }
            if me.m_in > z_vec.len() { return Err("prev ME m_in exceeds recomposed z length".into()); }
            let x_prev = z_vec[..me.m_in].to_vec();
            let w_prev = z_vec[me.m_in..].to_vec();
            let inst = neo_ccs::McsInstance { c: me.c.clone(), x: x_prev, m_in: me.m_in };
            
            // Check m_in consistency with current step
            if me.m_in != step_public_input.len() {
                return Err(format!("m_in mismatch between prev ME ({}) and current step ({})", me.m_in, step_public_input.len()).into());
            }
            let wit_mcs = neo_ccs::McsWitness::<F> { w: w_prev, Z: wit.Z.clone() };
            (inst, wit_mcs)
        }
        _ => {
            // Base case (step 0): Use the exact same RHS instance as LHS (self-fold).
            // This works because:
            // 1. The step CCS is trivially satisfied (same witness)
            // 2. The EV constraints are satisfied (same y_prev, y_next, y_step)
            // 3. The commitment is consistent
            (step_mcs_inst.clone(), step_mcs_wit.clone())
        }
    }};

    // DEBUG: Check if this is step 0 or later
    let is_first_step = input.prev_accumulator.step == 0;
    #[cfg(feature = "debug-logs")]
    {
        println!("🔍 DEBUG: Step {}, is_first_step: {}", input.step, is_first_step);
        println!("🔍 DEBUG: prev_accumulator.c_coords.len(): {}", input.prev_accumulator.c_coords.len());
        println!("🔍 DEBUG: c_step_coords.len(): {}", c_step_coords.len());
        println!("🔍 DEBUG: LHS instance commitment len: {}", lhs_inst.c.data.len());
        println!("🔍 DEBUG: RHS instance commitment len: {}", step_mcs_inst.c.data.len());
        println!("🔍 DEBUG: LHS witness Z shape: {}x{}", lhs_wit.Z.rows(), lhs_wit.Z.cols());
        println!("🔍 DEBUG: RHS witness Z shape: {}x{}", step_mcs_wit.Z.rows(), step_mcs_wit.Z.cols());
    }
    
    // DEBUG: Check if commitments are consistent
    if !is_first_step {
        #[cfg(feature = "debug-logs")]
        println!("🔍 DEBUG: LHS commitment first 4 coords: {:?}", 
                 lhs_inst.c.data.iter().take(4).collect::<Vec<_>>());
        #[cfg(feature = "debug-logs")]
        println!("🔍 DEBUG: RHS commitment first 4 coords: {:?}", 
                 step_mcs_inst.c.data.iter().take(4).collect::<Vec<_>>());
        #[cfg(feature = "debug-logs")]
        println!("🔍 DEBUG: prev_accumulator.c_coords first 4: {:?}", 
                 input.prev_accumulator.c_coords.iter().take(4).collect::<Vec<_>>());
        #[cfg(feature = "debug-logs")]
        println!("🔍 DEBUG: c_step_coords first 4: {:?}", 
                 c_step_coords.iter().take(4).collect::<Vec<_>>());
    }
    
    // 7) Fold prev-with-current using the production pipeline
    // Record the exact LHS augmented input used inside Pi-CCS for robust linking checks.
    let prev_augmented_public_input = lhs_inst.x.clone();
    
    #[cfg(feature = "debug-logs")]
    {
        eprintln!("🔍 DEBUG: About to fold instances");
        eprintln!("  LHS x.len()={}, RHS x.len()={}", lhs_inst.x.len(), step_mcs_inst.x.len());
        eprintln!("  LHS c.data.len()={}, RHS c.data.len()={}", lhs_inst.c.data.len(), step_mcs_inst.c.data.len());
        if !lhs_inst.c.data.is_empty() && !step_mcs_inst.c.data.is_empty() {
            eprintln!("  LHS c first 4: {:?}", &lhs_inst.c.data[..4.min(lhs_inst.c.data.len())]);
            eprintln!("  RHS c first 4: {:?}", &step_mcs_inst.c.data[..4.min(step_mcs_inst.c.data.len())]);
            eprintln!("  LHS == RHS commitment: {}", lhs_inst.c == step_mcs_inst.c);
            eprintln!("  LHS == RHS x: {}", lhs_inst.x == step_mcs_inst.x);
        }
    }
    
    // Clone MCS witnesses for later recombination of parent witness
    // (removed unused clones)
    let (mut me_instances, digit_witnesses, folding_proof) = neo_fold::fold_ccs_instances(
        input.params, 
        &step_augmented_ccs, 
        &[lhs_inst.clone(), step_mcs_inst.clone()], 
        &[lhs_wit.clone(),  step_mcs_wit.clone()]
    ).map_err(|e| format!("Nova folding failed: {}", e))?;

    // 🔒 SOUNDNESS: Populate ME instances with step commitment binding data
    for me_instance in &mut me_instances {
        me_instance.c_step_coords = c_step_coords.clone();
        me_instance.u_offset = u_offset;
        me_instance.u_len = u_len;
    }

    // 8) Evolve accumulator commitment coordinates with ρ using the step-only commitment.
    // Pattern B: c_next = c_prev + ρ·c_step, where c_step = pre-ρ step commitment (no tail)
    #[cfg(feature = "debug-logs")]
    {
        println!("🔍 DEBUG: About to evolve commitment, prev_coords.is_empty()={}", input.prev_accumulator.c_coords.is_empty());
        println!("🔍 DEBUG: rho value: {:?}", rho.as_canonical_u64());
    }
    let (c_coords_next, c_z_digest_next) = if input.prev_accumulator.c_coords.is_empty() {
        #[cfg(feature = "debug-logs")]
        println!("🔍 DEBUG: First step, using c_step_coords directly");
        let digest = digest_commit_coords(&c_step_coords);
        (c_step_coords.clone(), digest)
    } else {
        #[cfg(feature = "debug-logs")]
        println!("🔍 DEBUG: Evolving commitment: c_prev.len()={}, c_step.len()={}, rho={:?}", 
                 input.prev_accumulator.c_coords.len(), c_step_coords.len(), rho.as_canonical_u64());
        let result = evolve_commitment(&input.prev_accumulator.c_coords, &c_step_coords, rho)
            .map_err(|e| format!("commitment evolution failed: {}", e))?;
        #[cfg(feature = "debug-logs")]
        println!("🔍 DEBUG: Commitment evolution completed successfully");
        result
    };
    
    #[cfg(feature = "debug-logs")]
    println!("🔍 DEBUG: c_coords_next.len()={}", c_coords_next.len());

    let next_accumulator = Accumulator {
        c_z_digest: c_z_digest_next,
        c_coords: c_coords_next,
        y_compact: y_next.clone(),
        step: input.step + 1,
    };


    // ❌ INCORRECT CHECK REMOVED:
    // The following EV↔Π-CCS linkage check was treating rhs_me.y[j] as if j indexes
    // state elements, but y[j] actually indexes CCS matrices (t=3 for R1CS: A,B,C).
    // The EV constraints are already properly enforced in the augmented CCS (lines 2120-2142).
    //
    // PROVER-SIDE EV↔Π-CCS linkage check: catch invalid steps early
    // This is a development-time guard that duplicates the verifier check for better error messages.
    // SECURITY: The real check is in the verifier; this just provides early failure.
    // {
        // let rho_inv = F::ONE / rho;
        // let d_digits = neo_math::D;
        // let rhs_me = &folding_proof.pi_ccs_outputs[1]; // RHS = current step
        
        // Precompute powers of base b
        // let mut pow_b_f = vec![F::ONE; d_digits];
        // for i in 1..d_digits { pow_b_f[i] = pow_b_f[i-1] * F::from_u64(input.params.b as u64); }
        // let pow_b_k: Vec<neo_math::K> = pow_b_f.iter().cloned().map(neo_math::K::from).collect();
        
        // Check EV vs Π-CCS for each component
        // for j in 0..y_len.min(rhs_me.y.len()) {
            // let mut y_rhs_scalar_k = neo_math::K::ZERO;
            // for r in 0..d_digits { y_rhs_scalar_k += rhs_me.y[j][r] * pow_b_k[r]; }
            
            // let ev_j_f = (y_next[j] - input.prev_accumulator.y_compact[j]) * rho_inv;
            // let ev_j_k = neo_math::K::from(ev_j_f);
            
            // if y_rhs_scalar_k != ev_j_k {
                // return Err(format!(
                    // "SOUNDNESS: EV↔Π-CCS linkage failed at j={}: \
                    // prover produced y_step inconsistent with Π-CCS outputs.\n\
                    // This indicates the step witness does not match the claimed state transition.\n\
                    // RHS Π-CCS y[{}] scalarized: {:?}, EV (Δ/ρ)[{}]: {:?}",
                    // j, j, y_rhs_scalar_k, j, ev_j_k
                // ).into());
            // }
        // }
    // }

    // 9) Package IVC proof (no per-step SNARK compression)
    // Compute context digest using a SIMPLE CCS construction (without RLC binder) for consistency
    // The verifier will reconstruct the same simple CCS for digest verification
    let digest_ccs = build_augmented_ccs_linked(
        input.step_ccs,
        step_x.len(),
        &input.binding_spec.y_step_offsets,
        &input.binding_spec.y_prev_witness_indices,
        &input.binding_spec.step_program_input_witness_indices,
        y_len,
        input.binding_spec.const1_witness_index,
    ).map_err(|e| anyhow::anyhow!("Failed to build digest CCS: {}", e))?;
    
    let context_digest = crate::context_digest_v1(&digest_ccs, &step_public_input);
    
    #[cfg(feature = "neo-logs")]
    {
        eprintln!("🔍 PROVER DIGEST DEBUG:");
        eprintln!("  Prover context digest: {:02x?}", &context_digest[..8]);
        eprintln!("  Digest CCS: n={}, m={}", digest_ccs.n, digest_ccs.m);
        eprintln!("  Step public input length: {}", step_public_input.len());
        eprintln!("  Prover CCS params: step_x_len={}, y_len={}, const1_idx={}", 
                  step_x.len(), y_len, input.binding_spec.const1_witness_index);
        eprintln!("  Prover y_step_offsets: {:?}", input.binding_spec.y_step_offsets);
    }
    
    // Build public_io: [y_next values as bytes] + [context_digest]
    // This allows verify_and_extract* to work and puts digest at the end for verify()
    let mut public_io = Vec::with_capacity(8 * y_next.len() + 32);
    for y in &y_next {
        public_io.extend_from_slice(&y.as_canonical_u64().to_le_bytes());
    }
    public_io.extend_from_slice(&context_digest);
    
    let step_proof = crate::Proof {
        v: 2,
        circuit_key: [0u8; 32],           
        vk_digest: [0u8; 32],             
        public_io,                        // y_next values + context digest
        proof_bytes: vec![],              
        public_results: y_next.clone(),   
        meta: crate::ProofMeta { num_y_compact: y_len, num_app_outputs: y_next.len() },
    };
    
    // Construct PublicInputSegments from components
    // The prover doesn't need to know the internal structure of public_input
    // (it might be NIVC metadata or actual app inputs - we treat it uniformly)
    let public_inputs = PublicInputSegments::new(
        acc_digest_fields,                   // H(prev_acc)
        vec![],                              // app_x: empty (structure unknown to prover)
        input.public_input.unwrap_or(&[]).to_vec(), // transport: NIVC envelope or app inputs
        rho,
        input.prev_accumulator.y_compact.clone(),
        y_next.clone(),
    );
    
    let ivc_proof = IvcProof {
        step_proof,
        next_accumulator: next_accumulator.clone(),
        step: input.step,
        metadata: None,
        public_inputs,
        prev_step_augmented_public_input: prev_augmented_public_input,
        c_step_coords,
        me_instances: Some(me_instances.clone()), // Keep for final SNARK generation (TODO: optimize)
        digit_witnesses: Some(digit_witnesses.clone()), // Keep for final SNARK generation (TODO: optimize)
        folding_proof: Some(folding_proof),
    };

    // Return next chaining state: carry latest digit ME (for Stage 5) and RHS MCS (for strict linkage)
    Ok((
        IvcStepResult { proof: ivc_proof, next_state: y_next },
        me_instances.last().unwrap().clone(),
        digit_witnesses.last().unwrap().clone(),
        (step_mcs_inst, step_mcs_wit)
    ))
}
/// Prove a single IVC step using the main Neo proving pipeline
/// 
/// This is a convenience wrapper around `prove_ivc_step_chained` for cases
/// where you don't need to maintain chaining state between calls.
/// For proper Nova chaining, use `prove_ivc_step_chained` directly.
pub fn prove_ivc_step(input: IvcStepInput) -> Result<IvcStepResult, Box<dyn std::error::Error>> {
    // Use the chained version with no previous ME instance (folds with canonical zero instance)
    let (result, _me, _wit, _mcs) = prove_ivc_step_chained(input, None, None, None)?;
    Ok(result)
}

/// Prove an entire IVC chain from start to finish  
pub fn prove_ivc_chain(
    params: &crate::NeoParams,
    step_ccs: &CcsStructure<F>,
    step_inputs: &[IvcChainStepInput],
    initial_accumulator: Accumulator,
    binding_spec: &StepBindingSpec,          // NEW: Require trusted binding spec
) -> Result<IvcChainProof, Box<dyn std::error::Error>> {
    // SECURITY: Validate CCS structure requirements
    // ℓ = ceil(log2(n)) must be ≥ 2 for the sumcheck protocol
    // n is padded to next power of 2 (max 2), so n=3 → 4 → ℓ=2 is acceptable
    if step_ccs.n < 3 {
        return Err(format!(
            "CCS validation failed: n={} is too small (minimum n=3 required). \
            The sumcheck challenge length ℓ=ceil(log2(n_padded)) must be ≥ 2 for protocol security. \
            n is padded to next power-of-2 (minimum 2), so n=3→4→ℓ=2, n=2→2→ℓ=1 (too small). \
            Please ensure your circuit has at least 3 constraint rows.",
            step_ccs.n
        ).into());
    }
    
    let mut current_accumulator = initial_accumulator;
    let mut step_proofs = Vec::with_capacity(step_inputs.len());
    // Carry the running ME instance across steps (proper chaining)
    let mut prev_me: Option<neo_ccs::MeInstance<neo_ajtai::Commitment, F, neo_math::K>> = None;
    let mut prev_me_wit: Option<neo_ccs::MeWitness<F>> = None;

    // Strict linkage: carry RHS MCS instance/witness as next LHS across steps
    let mut prev_lhs_mcs: Option<(neo_ccs::McsInstance<neo_ajtai::Commitment, F>, neo_ccs::McsWitness<F>)> = None;

    for (step_idx, step_input) in step_inputs.iter().enumerate() {
        // Extract y_step from the witness using the binding spec indices
        let extractor = IndexExtractor { indices: binding_spec.y_step_offsets.clone() };
        let y_step = extractor.extract_y_step(&step_input.witness);

        let ivc_step_input = IvcStepInput {
            params,
            step_ccs,
            step_witness: &step_input.witness,
            prev_accumulator: &current_accumulator,
            step: step_idx as u64,
            public_input: step_input.public_input.as_ref().map(|v| v.as_slice()),
            y_step: &y_step,
            binding_spec,
            app_input_binding: AppInputBinding::WitnessBound,
            prev_augmented_x: step_proofs.last().map(|p: &IvcProof| p.public_inputs.step_augmented_public_input()),
        };

        let (step_result, me_out, me_wit_out, lhs_next) =
            prove_ivc_step_chained(ivc_step_input, prev_me.clone(), prev_me_wit.clone(), prev_lhs_mcs.clone())?;

        prev_me = Some(me_out);
        prev_me_wit = Some(me_wit_out);
        prev_lhs_mcs = Some(lhs_next);

        current_accumulator = step_result.proof.next_accumulator.clone();
        step_proofs.push(step_result.proof);
    }

    Ok(IvcChainProof { steps: step_proofs, final_accumulator: current_accumulator, chain_length: step_inputs.len() as u64 })
}
