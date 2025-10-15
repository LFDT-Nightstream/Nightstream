//! Augmented CCS builders for IVC
//!
//! This module builds the augmented CCS that includes:
//! - The base step CCS (user's computation)
//! - Embedded verifier (EV) constraints
//! - Cross-step bindings (y_prev, y_step)
//! - Optional RLC commitment bindings

use super::prelude::*;
use crate::shared::types::{AugmentConfig, IvcProof};
use super::ev::ev_with_public_rho_ccs;
use super::transcript::{build_step_transcript_data, create_step_digest};

/// Build augmented CCS with trusted binding specification
pub fn build_augmented_ccs_linked(
    step_ccs: &CcsStructure<F>,
    step_x_len: usize,
    y_step_offsets: &[usize],
    y_prev_witness_indices: &[usize],   
    step_program_input_witness_indices: &[usize],        
    y_len: usize,
    const1_witness_index: usize,
) -> Result<CcsStructure<F>, String> {
    build_augmented_ccs_linked_with_rlc(
        step_ccs,
        step_x_len,
        y_step_offsets,
        y_prev_witness_indices,
        step_program_input_witness_indices,
        y_len,
        const1_witness_index,
        None, // No RLC binder by default
    )
}

/// Build augmented CCS with optional RLC binder for step commitment binding
pub fn build_augmented_ccs_linked_with_rlc(
    step_ccs: &CcsStructure<F>,
    step_x_len: usize,
    y_step_offsets: &[usize],
    y_prev_witness_indices: &[usize],   
    step_program_input_witness_indices: &[usize],        
    y_len: usize,
    const1_witness_index: usize,
    rlc_binder: Option<(Vec<F>, F)>, // (aggregated_row, rhs) for RLC constraint
) -> Result<CcsStructure<F>, String> {
    // 🛡️ SECURITY: Validate matrix count assumptions
    let t = step_ccs.matrices.len();
    if t < 3 {
        return Err(format!(
            "augmented CCS requires at least 3 matrices (A,B,C). Got t={}", t
        ));
    }
    
    if y_step_offsets.len() != y_len {
        return Err(format!("y_step_offsets length {} must equal y_len {}", y_step_offsets.len(), y_len));
    }
    // y_prev_witness_indices are optional now (used later for cross-step stitching)
    // Only require equal length if provided.
    if !y_prev_witness_indices.is_empty() && y_prev_witness_indices.len() != y_len {
        return Err(format!(
            "y_prev_witness_indices length {} must equal y_len {} when provided",
            y_prev_witness_indices.len(), y_len
        ));
    }
    // Allow binding only the app input tail of step_x; not the digest prefix
    if !step_program_input_witness_indices.is_empty() && step_program_input_witness_indices.len() > step_x_len {
        return Err(format!("step_program_input_witness_indices length {} cannot exceed step_public_input_len {}", step_program_input_witness_indices.len(), step_x_len));
    }
    if const1_witness_index >= step_ccs.m {
        return Err(format!("const1_witness_index {} out of range (m={})", const1_witness_index, step_ccs.m));
    }
    for &o in y_step_offsets.iter().chain(y_prev_witness_indices).chain(step_program_input_witness_indices) {
        if o >= step_ccs.m {
            return Err(format!("witness offset {} out of range (m={})", o, step_ccs.m));
        }
    }

    // Public input: [ step_x || ρ || y_prev || y_next ]
    let pub_cols = step_x_len + 1 + 2 * y_len;

    // Row accounting (no preset cap):
    //  - step_rows                              (copy step CCS)
    //  - EV rows                                (see below; production: 2*y_len, testing: 2)
    //  - step_x_len binder rows (optional)      (step_x[i] - step_witness[x_i] = 0)
    //  - y_len prev binder rows                 (y_prev[k] - step_witness[prev_k] = 0)
    //  - 1 RLC binder row (optional)            (⟨G, z⟩ = Σ r_i * c_step[i])
    //  - 1 const-1 enforcement row              (w_const1 * ρ = ρ, forces w_const1 = 1)
    let step_rows = step_ccs.n;
    // EV rows in production encoding: two per state element (u = ρ·y_step; y_next − y_prev − u)
    let ev_rows = 2 * y_len;
    let x_bind_rows = if step_program_input_witness_indices.is_empty() { 0 } else { step_x_len };
    let prev_bind_rows = if y_prev_witness_indices.is_empty() { 0 } else { y_len };
    let rlc_rows = if rlc_binder.is_some() { 1 } else { 0 };
    // SOUNDNESS FIX: Enforce const-1 column is actually 1 using public ρ
    let const1_enforce_rows = 1;
    let total_rows = step_rows + ev_rows + x_bind_rows + prev_bind_rows + rlc_rows + const1_enforce_rows;
    // Pre-pad to next power-of-two for clean χ_r wiring (optional but stable).
    // This keeps augmented CCS shape fixed and matches Π_CCS's ℓ computation.
    // Avoid n=1 degeneracy (0-round sum-check) which can misalign R1CS terminal checks.
    let mut target_rows = total_rows.next_power_of_two();
    if target_rows < 2 { target_rows = 2; }

    // Witness: [ step_witness || u ]
    let step_wit_cols = step_ccs.m;
    // Witness columns added for EV section: u has length y_len
    let ev_wit_cols = y_len; // u
    let total_wit_cols = step_wit_cols + ev_wit_cols;
    let total_cols = pub_cols + total_wit_cols;

    let mut combined_mats = Vec::new();
    for matrix_idx in 0..step_ccs.matrices.len() {
        let mut data = vec![F::ZERO; target_rows * total_cols];

        // Copy step CCS at the top
        let step_matrix = &step_ccs.matrices[matrix_idx];
        for r in 0..step_rows {
            for c in 0..step_ccs.m {
                let col = pub_cols + c;                  // step witness lives after public block
                data[r * total_cols + col] = step_matrix[(r, c)];
            }
        }

        // Offsets
        let col_rho     = step_x_len;
        let col_y_prev0 = col_rho + 1;
        let col_y_next0 = col_y_prev0 + y_len;
        // absolute column for the constant-1 witness (within the *augmented* z = [public | witness])
        let col_const1_abs = pub_cols + const1_witness_index;
        
        let col_u0 = pub_cols + step_wit_cols;
        // EV: u[k] = y_step[k]  (delta semantics, no rho)
        // ρ is only for folding instances/commitments, NOT for scaling state evolution
        for k in 0..y_len {
            let r = step_rows + k;
            match matrix_idx {
                // A: pick y_step[k]
                0 => data[r * total_cols + (pub_cols + y_step_offsets[k])] = F::ONE,
                // B: multiply by 1 (const-1 witness col)
                1 => data[r * total_cols + col_const1_abs] = F::ONE,
                // C: equals u[k]
                2 => data[r * total_cols + (col_u0 + k)] = F::ONE,
                _ => {}
            }
        }
        // EV: y_next[k] - y_prev[k] - u[k] = 0  (× 1 via step_witness[0] == 1)
        for k in 0..y_len {
            let r = step_rows + y_len + k;
            match matrix_idx {
                0 => {
                    data[r * total_cols + (col_y_next0 + k)] = F::ONE;
                    data[r * total_cols + (col_y_prev0 + k)] = -F::ONE;
                    data[r * total_cols + (col_u0 + k)]      = -F::ONE;
                }
                1 => data[r * total_cols + col_const1_abs] = F::ONE,
                _ => {}
            }
        }

        // Binder X: step_x[i] - step_witness[x_i] = 0  (if any)
        if !step_program_input_witness_indices.is_empty() {
            // Bind only the last step_program_input_witness_indices.len() elements of step_x (the app inputs)
            let bind_len = step_program_input_witness_indices.len();
            let bind_start = step_x_len - bind_len;
            for i in 0..bind_len {
                let r = step_rows + ev_rows + i;
                match matrix_idx {
                    0 => {
                        data[r * total_cols + (bind_start + i)] = F::ONE;                         // + step_x[bind_start + i]
                        data[r * total_cols + (pub_cols + step_program_input_witness_indices[i])] = -F::ONE;      // - step_witness[x_i]
                    }
                    1 => data[r * total_cols + col_const1_abs] = F::ONE,                        // × 1
                    _ => {}
                }
            }
        }

        // Binder Y_prev: y_prev[k] - step_witness[y_prev_witness_indices[k]] = 0  (if any)
        // SECURITY FIX: Enforce that step circuit reads of y_prev match the accumulator's y_prev
        if !y_prev_witness_indices.is_empty() {
            for k in 0..y_len {
                let r = step_rows + ev_rows + x_bind_rows + k;
                match matrix_idx {
                    0 => {
                        data[r * total_cols + (col_y_prev0 + k)] = F::ONE;                           // + y_prev[k]
                        data[r * total_cols + (pub_cols + y_prev_witness_indices[k])] = -F::ONE;    // - step_witness[y_prev_witness_indices[k]]
                    }
                    1 => data[r * total_cols + col_const1_abs] = F::ONE,                           // × 1
                    _ => {}
                }
            }
        }

        // RLC Binder: enforce linear equality ⟨G, z⟩ = rhs where
        // G = aggregated_row over witness coordinates and rhs = Σ r_i * c_step[i] (or diff variant)
        // Encode in R1CS as: <A,z> * <B,z> = <C,z>
        //   A row = G · z, B row selects const-1 (== 1), C puts rhs in const-1 column
        if let Some((ref aggregated_row, rhs)) = rlc_binder {
            let r = step_rows + ev_rows + x_bind_rows + prev_bind_rows;
            match matrix_idx {
                0 => {
                    // A matrix: ⟨G, z⟩ where z = [public || witness]
                    // G covers the entire witness (step_witness || u)
                    for (j, &g_j) in aggregated_row.iter().enumerate() {
                        if j < total_wit_cols {
                            let col = pub_cols + j;  // witness starts after public inputs
                            data[r * total_cols + col] = g_j;
                        }
                    }
                }
                1 => {
                    // B matrix: multiply by 1 (const-1 witness column)
                    data[r * total_cols + col_const1_abs] = F::ONE;
                }
                2 => {
                    // C matrix: place rhs on const-1 column so equality is linear
                    data[r * total_cols + col_const1_abs] = rhs;
                }
                _ => {}
            }
        }

        // SOUNDNESS FIX: Enforce w_const1 * ρ = ρ (forces w_const1 = 1 since ρ ≠ 0)
        // This prevents malicious provers from setting const-1 to 0, which would turn
        // all linear constraints (that multiply by const-1) into trivial 0=0 identities.
        // Many rows above (EV, X/Y binders) rely on B matrix selecting the "1" column;
        // without this constraint, those rows can be zeroed out by a malicious prover.
        {
            let r = step_rows + ev_rows + x_bind_rows + prev_bind_rows + rlc_rows;
            
            // SECURITY: Bounds checks to prevent silent corruption of the constraint
            debug_assert!(r < target_rows, "const-1 enforcement row {} must be within target_rows {}", r, target_rows);
            debug_assert!(col_const1_abs < total_cols, "const-1 column {} must be within total_cols {}", col_const1_abs, total_cols);
            debug_assert!(col_rho < total_cols, "rho column {} must be within total_cols {}", col_rho, total_cols);
            
            match matrix_idx {
                0 => {
                    // A matrix: select w_const1 (the witness column that should be 1)
                    debug_assert!(r * total_cols + col_const1_abs < data.len(), "A matrix index out of bounds");
                    data[r * total_cols + col_const1_abs] = F::ONE;
                }
                1 => {
                    // B matrix: select ρ (public, from Fiat-Shamir, guaranteed non-zero)
                    debug_assert!(r * total_cols + col_rho < data.len(), "B matrix index out of bounds");
                    data[r * total_cols + col_rho] = F::ONE;
                }
                2 => {
                    // C matrix: also select ρ (public)
                    // This enforces: w_const1 * ρ = ρ  =>  w_const1 = 1
                    debug_assert!(r * total_cols + col_rho < data.len(), "C matrix index out of bounds");
                    data[r * total_cols + col_rho] = F::ONE;
                }
                _ => {}
            }
        }

        // Remaining rows (from total_rows..target_rows) remain zero — they encode 0 == 0
        combined_mats.push(Mat::from_row_major(target_rows, total_cols, data));
    }

    let f = step_ccs.f.clone();
    CcsStructure::new(combined_mats, f).map_err(|e| format!("Failed to create CCS: {:?}", e))
}

/// Build witness for linked augmented CCS.
/// 
/// This creates the combined witness [step_witness || u] where u = y_step (delta)
/// and y_step is extracted from the step_witness at the specified offsets.
/// Note: ρ is NOT used here - it's only for folding commitments, not state evolution.
pub fn build_linked_augmented_witness(
    step_witness: &[F],
    y_step_offsets: &[usize],
    _rho: F,
) -> Vec<F> {
    // Extract y_step values from step witness
    let mut y_step = Vec::with_capacity(y_step_offsets.len());
    for &offset in y_step_offsets {
        y_step.push(step_witness[offset]);
    }
    
    // Compute u = y_step (delta semantics, no rho multiplier)
    let u: Vec<F> = y_step.iter().map(|&ys| ys).collect();
    
    // Combined witness: [step_witness || u]
    let mut combined_witness = step_witness.to_vec();
    combined_witness.extend_from_slice(&u);
    
    combined_witness
}

/// Build public input for linked augmented CCS.
/// 
/// Public input layout: [step_x || ρ || y_prev || y_next]
pub fn build_augmented_public_input_for_step(
    step_x: &[F],
    rho: F,
    y_prev: &[F],
    y_next: &[F],
) -> Vec<F> {
    let mut public_input = Vec::new();
    public_input.extend_from_slice(step_x);
    public_input.push(rho);
    public_input.extend_from_slice(y_prev);
    public_input.extend_from_slice(y_next);
    public_input
}

/// Build public input for linked augmented CCS with identity semantics.
/// 
/// This is used for the LHS (identity/running) instance in folding where
/// no state change occurs: y_next = y_prev.
/// 
/// Public input layout: [step_x || ρ || y_prev || y_prev]
pub fn build_augmented_public_input_identity(
    step_x: &[F],
    rho: F,
    y: &[F],
) -> Vec<F> {
    let mut public_input = Vec::new();
    public_input.extend_from_slice(step_x);
    public_input.push(rho);
    public_input.extend_from_slice(y);  // y_prev
    public_input.extend_from_slice(y);  // y_next = y_prev (identity)
    public_input
}

/// **UNIFIED NOVA AUGMENTATION**: Build the complete Nova embedded verifier CCS
/// 
/// This composes all the Nova/HyperNova components into a single augmented CCS:
/// 1. **Step CCS**: User's computation relation
/// 2. **EV-hash**: In-circuit ρ derivation + folding verification (with public y)
/// 3. **Commitment opening**: Ajtai commitment verification constraints
/// 4. **Commitment lincomb**: In-circuit commitment folding (c_next = c_prev + ρ * c_step)
/// 
/// **Public Input Structure**: [ step_X || y_prev || y_next || c_open || c_prev || c_step || c_next ]
/// **Witness Structure**: [ step_witness || ev_witness || ajtai_opening_witness || lincomb_witness ]
/// 
/// All components share the same in-circuit derived challenge ρ, ensuring consistency
/// across the folding verification process.
/// 
/// This satisfies Las's requirement for "folding verifier expressed as a CCS structure."
pub fn augmentation_ccs(
    step_ccs: &CcsStructure<F>,
    cfg: AugmentConfig,
    step_digest: [u8; 32],
) -> Result<CcsStructure<F>, Box<dyn std::error::Error>> {
    // 1) EV (public-ρ) over y
    let ev = ev_with_public_rho_ccs(cfg.y_len);
    let a1 = neo_ccs::direct_sum_transcript_mixed(step_ccs, &ev, step_digest)?;

    // 2) Ajtai opening: build fixed rows from PP and bake as CCS constants
    //    msg_len = d * m  (digits)
    let (kappa, m, d) = cfg.ajtai_pp;
    let msg_len = d * m;

    // Ensure PP present for (d, m)
    crate::ensure_ajtai_pp_for_dims(d, m, || {
        use rand::{RngCore, SeedableRng};
        use rand::rngs::StdRng;
        
        let mut rng = if std::env::var("NEO_DETERMINISTIC").is_ok() {
            StdRng::from_seed([42u8; 32])
        } else {
            let mut seed = [0u8; 32];
            rand::rng().fill_bytes(&mut seed);
            StdRng::from_seed(seed)
        };
        let pp = neo_ajtai::setup(&mut rng, d, kappa, m)?;
        neo_ajtai::set_global_pp(pp).map_err(anyhow::Error::from)
    })?;

    let pp = neo_ajtai::get_global_pp_for_dims(d, m)
        .map_err(|e| format!("Ajtai PP unavailable for (d={}, m={}): {}", d, m, e))?;

    // Bake L_i rows as constants
    let rows: Vec<Vec<F>> = {
        let l = cfg.commit_len; // number of coordinates to open
        neo_ajtai::rows_for_coords(&*pp, msg_len, l)
            .map_err(|e| format!("rows_for_coords failed: {}", e))?
    };

    let open = neo_ccs::gadgets::commitment_opening::commitment_opening_from_rows_ccs(&rows, msg_len);
    let a2 = neo_ccs::direct_sum_transcript_mixed(&a1, &open, step_digest)?;

    // 3) Commitment lincomb with public ρ
    let clin = neo_ccs::gadgets::commitment_opening::commitment_lincomb_ccs(cfg.commit_len);
    let augmented = neo_ccs::direct_sum_transcript_mixed(&a2, &clin, step_digest)?;

    Ok(augmented)
}

impl Default for AugmentConfig {
    fn default() -> Self {
        Self {
            hash_input_len: 4,      // Common hash input size
            y_len: 2,               // Typical compact accumulator size  
            ajtai_pp: (4, 8, 32),   // Example Ajtai parameters (kappa=4, m=8, d=32)
            commit_len: 128,        // d * kappa = 32 * 4 = 128
        }
    }
}
/// Build the correct public input format for the final SNARK
/// 
/// **SECURITY FIX**: This constructs the proper augmented CCS public input format:
/// `[step_x || ρ || y_prev || y_next]` instead of arbitrary formats like `[x]`.
/// 
/// This prevents the vulnerability where wrong formats were accepted.
pub fn build_final_snark_public_input(
    step_x: &[F],
    rho: F,
    y_prev: &[F],
    y_next: &[F],
) -> Vec<F> {
    let mut public_input = Vec::new();
    public_input.extend_from_slice(step_x);  // step_x
    public_input.push(rho);                  // ρ 
    public_input.extend_from_slice(y_prev);  // y_prev
    public_input.extend_from_slice(y_next);  // y_next
    public_input
}
/// Recompute the augmented public input used by the prover for this step:
/// X = [step_x || ρ || y_prev || y_next]. Returns (X, ρ).
pub(crate) fn compute_augmented_public_input_for_step(
    prev_acc: &Accumulator,
    proof: &IvcProof,
) -> Result<(Vec<F>, F), Box<dyn std::error::Error>> {
    let step_data = build_step_transcript_data(prev_acc, proof.step, proof.public_inputs.wrapper_public_input_x());
    let step_digest = create_step_digest(&step_data);
    let (rho_calc, _td) = super::transcript::rho_from_transcript(prev_acc, step_digest, &proof.c_step_coords);
    
    // SECURITY: Always use recalculated ρ from transcript, never trust prover's value
    // The rho_from_transcript uses challenge_nonzero_field which guarantees ρ ≠ 0
    let rho = rho_calc;
    debug_assert_ne!(rho, F::ZERO, "ρ must be non-zero for const-1 enforcement soundness");

    let x_aug = build_augmented_public_input_for_step(
        proof.public_inputs.wrapper_public_input_x(),
        rho,
        &prev_acc.y_compact,
        &proof.next_accumulator.y_compact,
    );
    Ok((x_aug, rho))
}
