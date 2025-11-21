//! Main FoldRun circuit implementation
//!
//! This synthesizes R1CS constraints to verify an entire FoldRun:
//! - For each fold step:
//!   - Verify Π-CCS terminal identity
//!   - Verify sumcheck rounds
//!   - Verify RLC equalities
//!   - Verify DEC equalities
//! - Verify accumulator chaining between steps

use bellpepper_core::{ConstraintSystem, SynthesisError};
use crate::circuit::witness::{FoldRunWitness, FoldRunInstance};
use crate::gadgets::k_field::{KNum, KNumVar, alloc_k, k_add as k_add_raw};
use crate::gadgets::pi_ccs::{sumcheck_round_gadget, sumcheck_eval_gadget};
use crate::error::{Result, SpartanBridgeError};
use neo_fold::folding::FoldStep;
use neo_math::F as NeoF;
use neo_ccs::Mat;
use p3_field::PrimeCharacteristicRing;
use crate::CircuitF;

// Import helper functions from separate module
use super::fold_circuit_helpers as helpers;

// Spartan2 integration: implement SpartanCircuit over Goldilocks + Hash-MLE PCS.
use bellpepper_core::num::AllocatedNum;
use spartan2::provider::GoldilocksP3MerkleMleEngine;
use spartan2::traits::circuit::SpartanCircuit as SpartanCircuitTrait;

/// Sparse representation of the CCS polynomial f in the circuit field.
///
/// Each term is coeff * ∏_j m_j^{exps[j]}.
#[derive(Clone, Debug)]
pub struct CircuitPolyTerm {
    pub coeff: CircuitF,
    pub exps: Vec<u32>,
}

/// Main circuit for verifying a FoldRun
#[derive(Clone)]
pub struct FoldRunCircuit {
    /// Public instance
    pub instance: FoldRunInstance,
    
    /// Private witness
    pub witness: Option<FoldRunWitness>,
    
    /// Delta constant for K-field multiplication (u^2 = δ)
    /// For Goldilocks K, δ = 7
    pub delta: CircuitF,
    
    /// Base parameter b for DEC decomposition
    pub base_b: u32,

    /// CCS polynomial f, converted to circuit field coefficients.
    pub poly_f: Vec<CircuitPolyTerm>,
}

impl FoldRunCircuit {
    pub fn new(
        instance: FoldRunInstance,
        witness: Option<FoldRunWitness>,
        delta: CircuitF,
        base_b: u32,
        poly_f: Vec<CircuitPolyTerm>,
    ) -> Self {
        Self {
            instance,
            witness,
            delta,
            base_b,
            poly_f,
        }
    }

    /// Synthesize the full FoldRun circuit
    ///
    /// This is the main entry point for constraint generation
    pub fn synthesize<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
    ) -> Result<()> {
        // Allocate public inputs
        self.allocate_public_inputs(cs)?;

        // Get witness or error
        let witness = self.witness.as_ref()
            .ok_or_else(|| SpartanBridgeError::InvalidInput("Missing witness".into()))?;

        // Verify each fold step
        for (step_idx, step) in witness.fold_run.steps.iter().enumerate() {
            self.verify_fold_step(cs, step_idx, step, witness)?;
        }

        // Verify accumulator chaining across all steps
        self.verify_accumulator_chaining(cs, witness)?;

        Ok(())
    }

    /// Allocate and constrain public inputs
    /// 
    /// Currently we expose a minimal set of public inputs:
    /// - Params digest
    /// - CCS digest
    /// - MCS digest
    ///
    /// The full accumulator and Π-CCS challenges remain private witness data for now.
    fn allocate_public_inputs<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
    ) -> Result<()> {
        // Helper to expose a 32-byte digest as 4 field elements by chunking into u64.
        let mut alloc_digest =
            |label: &str, digest: &[u8; 32]| -> Result<()> {
                for (i, chunk) in digest.chunks(8).enumerate() {
                    let mut limb_bytes = [0u8; 8];
                    limb_bytes.copy_from_slice(chunk);
                    let limb_u64 = u64::from_le_bytes(limb_bytes);
                    let value = CircuitF::from(limb_u64);
                    // Allocate as public input.
                    let _ = cs.alloc_input(
                        || format!("{}_limb_{}", label, i),
                        || Ok(value),
                    )?;
                }
                Ok(())
            };

        alloc_digest("params_digest", &self.instance.params_digest)?;
        alloc_digest("ccs_digest", &self.instance.ccs_digest)?;
        alloc_digest("mcs_digest", &self.instance.mcs_digest)?;

        Ok(())
    }

    /// Verify a single fold step
    fn verify_fold_step<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        step: &neo_fold::folding::FoldStep,
        witness: &FoldRunWitness,
    ) -> Result<()> {
        // Get the Π-CCS proof and challenges for this step
        let pi_ccs_proof = &witness.pi_ccs_proofs[step_idx];
        let challenges = &self.instance.pi_ccs_challenges[step_idx];

        // 1a. Bind the public claimed initial sum T from ME inputs and challenges.
        let initial_sum_var = self.verify_initial_sum_binding(
            cs,
            step_idx,
            pi_ccs_proof,
            challenges,
            witness,
        )?;

        // 1b. Run sumcheck rounds algebra starting from T to obtain the final running sum.
        let sumcheck_final_var = self.verify_sumcheck_rounds(
            cs,
            step_idx,
            pi_ccs_proof,
            challenges,
            &initial_sum_var,
        )?;

        // Pre-allocate y-tables for Π-CCS outputs once so they can be shared
        // between the terminal identity and RLC gadgets.
        let mut ccs_out_y_vars: Vec<Vec<Vec<KNumVar>>> =
            Vec::with_capacity(step.ccs_out.len());
        for (i, child) in step.ccs_out.iter().enumerate() {
            let y_table = helpers::alloc_y_table_from_neo(
                cs,
                &child.y,
                &format!("step_{}_ccs_out_child_{}_y", step_idx, i),
            )?;
            ccs_out_y_vars.push(y_table);
        }

        // 1c. Verify Π-CCS terminal identity: final running sum == RHS(α,β,γ,r',α', out_me, inputs).
        self.verify_terminal_identity(
            cs,
            step_idx,
            step,
            pi_ccs_proof,
            challenges,
            witness,
            &sumcheck_final_var,
            &ccs_out_y_vars,
        )?;

        // 2. Verify RLC equalities
        self.verify_rlc(cs, step_idx, step, witness, &ccs_out_y_vars)?;

        // 3. Verify DEC equalities
        self.verify_dec(cs, step_idx, step, witness)?;

        Ok(())
    }

    /// Verify sumcheck rounds for a step
    fn verify_sumcheck_rounds<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        proof: &neo_reductions::PiCcsProof,
        challenges: &crate::circuit::witness::PiCcsChallenges,
        initial_sum: &KNumVar,
    ) -> Result<KNumVar> {
        // Sanity: number of rounds in the proof must match the number of
        // advertised challenges for this step.
        if proof.sumcheck_rounds.len() != challenges.sumcheck_challenges.len() {
            return Err(SpartanBridgeError::InvalidInput(
                format!(
                    "Sumcheck rounds/challenges length mismatch at step {}: rounds={}, challenges={}",
                    step_idx,
                    proof.sumcheck_rounds.len(),
                    challenges.sumcheck_challenges.len()
                ),
            ));
        }

        // Start running sum from the in-circuit initial sum variable (T).
        let mut claimed_sum = initial_sum.clone();

        // For each sumcheck round
        for (round_idx, round_poly) in proof.sumcheck_rounds.iter().enumerate() {
            // Allocate polynomial coefficients
            let mut coeffs = Vec::new();
            for (coeff_idx, coeff) in round_poly.iter().enumerate() {
            let coeff_var = helpers::alloc_k_from_neo(
                cs,
                *coeff,
                &format!("step_{}_round_{}_coeff_{}", step_idx, round_idx, coeff_idx),
            ).map_err(|e| SpartanBridgeError::SynthesisError(format!("{:?}", e)))?;
                coeffs.push(coeff_var);
            }

            // Verify p(0) + p(1) = claimed_sum
            sumcheck_round_gadget(
                cs,
                &coeffs,
                round_poly,
                &claimed_sum,
                self.delta,
                &format!("step_{}_round_{}", step_idx, round_idx),
            ).map_err(|e| SpartanBridgeError::from(e))?;

            // Allocate challenge for this round
            let challenge = helpers::alloc_k_from_neo(
                cs,
                challenges.sumcheck_challenges[round_idx],
                &format!("step_{}_round_{}_challenge", step_idx, round_idx),
            )?;

            // Compute next claimed sum = p(challenge) via gadget (returns KNumVar)
            let next_sum = sumcheck_eval_gadget(
                cs,
                &coeffs,
                round_poly,
                &challenge,
                challenges.sumcheck_challenges[round_idx],
                self.delta,
                &format!("step_{}_round_{}_eval", step_idx, round_idx),
            ).map_err(|e| SpartanBridgeError::from(e))?;

            // Update claimed_sum for next round
            claimed_sum = next_sum;
        }

        // After all rounds, optionally check that the prover's scalar
        // sumcheck_final matches the in-circuit running sum. This is a
        // consistency check only; the algebraic binding is via the KNumVar
        // returned from this method.
        let final_sum_expected = helpers::alloc_k_from_neo(
            cs,
            proof.sumcheck_final,
            &format!("step_{}_final_sum", step_idx),
        )?;

        helpers::enforce_k_eq(
            cs,
            &claimed_sum,
            &final_sum_expected,
            &format!("step_{}_final_sum_matches_scalar", step_idx),
        );

        Ok(claimed_sum)
    }

    /// Verify that sc_initial_sum equals the public claimed initial sum T
    /// computed from ME inputs and α, mirroring `claimed_initial_sum_from_inputs`.
    fn verify_initial_sum_binding<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        proof: &neo_reductions::PiCcsProof,
        challenges: &crate::circuit::witness::PiCcsChallenges,
        witness: &FoldRunWitness,
    ) -> Result<KNumVar> {
        // 1. Allocate α and γ challenges as in-circuit K variables so that
        //    the initial sum T is derived inside the circuit from these
        //    challenges and the ME inputs' y-vectors.

        let mut alpha_vars = Vec::with_capacity(challenges.alpha.len());
        for (i, &val) in challenges.alpha.iter().enumerate() {
            alpha_vars.push(helpers::alloc_k_from_neo(
                cs,
                val,
                &format!("step_{}_alpha_{}", step_idx, i),
            )?);
        }

        let gamma_var = helpers::alloc_k_from_neo(
            cs,
            challenges.gamma,
            &format!("step_{}_gamma", step_idx),
        )?;

        // 2. Select ME inputs for this step (same choice as the native
        //    verifier): step 0 uses the public initial accumulator; later
        //    steps use DEC children from the previous step.
        let me_inputs = if step_idx == 0 {
            &self.instance.initial_accumulator
        } else {
            &witness.fold_run.steps[step_idx - 1].dec_children
        };

        // 3. Allocate the ME inputs' y-tables as KNumVar arrays so they can
        //    be fed into the in-circuit claimed_initial_sum_gadget.
        let mut me_inputs_y_vars: Vec<Vec<Vec<KNumVar>>> =
            Vec::with_capacity(me_inputs.len());
        let mut me_inputs_y_vals: Vec<Vec<Vec<neo_math::K>>> =
            Vec::with_capacity(me_inputs.len());
        for (i, input) in me_inputs.iter().enumerate() {
            let y_table = helpers::alloc_y_table_from_neo(
                cs,
                &input.y,
                &format!("step_{}_me_input_{}_y", step_idx, i),
            )?;
            me_inputs_y_vars.push(y_table);
            me_inputs_y_vals.push(input.y.clone());
        }

        // 4. Derive T in-circuit from (α, γ, ME y) using the gadget version
        //    of `claimed_initial_sum_from_inputs`.
        let t_var = self.claimed_initial_sum_gadget(
            cs,
            step_idx,
            &alpha_vars,
            &challenges.alpha,
            &gamma_var,
            challenges.gamma,
            &me_inputs_y_vars,
            &me_inputs_y_vals,
        )?;

        // 5. If the proof provided a scalar sc_initial_sum, enforce that it
        //    matches the in-circuit T. This mirrors the native verifier's
        //    optional tightness check.
        if let Some(sc_initial) = proof.sc_initial_sum {
            let sc_initial_var = helpers::alloc_k_from_neo(
                cs,
                sc_initial,
                &format!("step_{}_sc_initial_sum_binding", step_idx),
            )?;

            helpers::enforce_k_eq(
                cs,
                &sc_initial_var,
                &t_var,
                &format!("step_{}_initial_sum_matches_T", step_idx),
            );
        }

        Ok(t_var)
    }

    /// KNumVar version of `claimed_initial_sum_from_inputs`, using only the ME
    /// inputs' y-vectors and α, γ from the challenges.
    fn claimed_initial_sum_gadget<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        alpha_vars: &[KNumVar],
        alpha_vals: &[neo_math::K],
        gamma_var: &KNumVar,
        gamma_val: neo_math::K,
        me_inputs_y_vars: &[Vec<Vec<KNumVar>>],
        me_inputs_y_vals: &[Vec<Vec<neo_math::K>>],
    ) -> Result<KNumVar> {
        use core::cmp::min;
        use neo_math::K as NeoK;

        if alpha_vars.len() != alpha_vals.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "claimed_initial_sum_gadget alpha length mismatch at step {}: vars={}, vals={}",
                step_idx,
                alpha_vars.len(),
                alpha_vals.len()
            )));
        }
        if me_inputs_y_vars.len() != me_inputs_y_vals.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "claimed_initial_sum_gadget me_inputs length mismatch at step {}: vars={}, vals={}",
                step_idx,
                me_inputs_y_vars.len(),
                me_inputs_y_vals.len()
            )));
        }

        let k_total = 1 + me_inputs_y_vars.len(); // 1 MCS + |ME|
        if k_total < 2 {
            // No Eval block when k=1 → T = 0.
            return helpers::k_zero(cs, &format!("step_{}_T_zero", step_idx));
        }

        // Build χ_α over Ajtai domain.
        let d_sz = 1usize << alpha_vars.len();
        let mut chi_alpha_vars: Vec<KNumVar> = Vec::with_capacity(d_sz);
        let mut chi_alpha_vals: Vec<NeoK> = Vec::with_capacity(d_sz);

        // χ_α[ρ] = ∏_bit (α_bit if ρ_bit=1 else 1-α_bit)
        for rho in 0..d_sz {
            // Start with w = 1 in both the native and in-circuit representation.
            let mut w_val = NeoK::ONE;
            let w_hint = KNum::<CircuitF>::from_neo_k(w_val);
            let mut w_var = alloc_k(
                cs,
                Some(w_hint),
                &format!("step_{}_chi_alpha_{}_init", step_idx, rho),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;

            for (bit, (a_var, &a_val)) in alpha_vars.iter().zip(alpha_vals.iter()).enumerate() {
                let bit_is_one = ((rho >> bit) & 1) == 1;
                // Factor value in K.
                let (factor_var, factor_val) = if bit_is_one {
                    (a_var.clone(), a_val)
                } else {
                    // factor = 1 - α_bit, enforced via an explicit relation
                    // factor + α_bit = 1 in K.
                    let factor_val = NeoK::ONE - a_val;
                    let factor_hint = KNum::<CircuitF>::from_neo_k(factor_val);
                    let factor_var = alloc_k(
                        cs,
                        Some(factor_hint),
                        &format!(
                            "step_{}_chi_alpha_{}_bit{}_factor",
                            step_idx, rho, bit
                        ),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;

                    // one_var with explicit K value 1.
                    let one_hint = KNum::<CircuitF>::from_neo_k(NeoK::ONE);
                    let one_var = alloc_k(
                        cs,
                        Some(one_hint),
                        &format!(
                            "step_{}_chi_alpha_{}_bit{}_one",
                            step_idx, rho, bit
                        ),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;

                    // sum = factor + α_bit with native hint, enforce sum == 1.
                    let sum_val = factor_val + a_val;
                    let sum_hint = KNum::<CircuitF>::from_neo_k(sum_val);
                    let sum_var = k_add_raw(
                        cs,
                        &factor_var,
                        a_var,
                        Some(sum_hint),
                        &format!(
                            "step_{}_chi_alpha_{}_bit{}_sum",
                            step_idx, rho, bit
                        ),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;
                    helpers::enforce_k_eq(
                        cs,
                        &sum_var,
                        &one_var,
                        &format!(
                            "step_{}_chi_alpha_{}_bit{}_one_minus",
                            step_idx, rho, bit
                        ),
                    );
                    (factor_var, factor_val)
                };

                // w <- w * factor with native hint.
                let (new_w_var, new_w_val) = helpers::k_mul_with_hint(
                    cs,
                    &w_var,
                    w_val,
                    &factor_var,
                    factor_val,
                    self.delta,
                    &format!("step_{}_chi_alpha_{}_bit{}", step_idx, rho, bit),
                )?;
                w_var = new_w_var;
                w_val = new_w_val;
            }

            chi_alpha_vars.push(w_var);
            chi_alpha_vals.push(w_val);
        }

        // γ^k_total (used in the weights)
        let mut gamma_to_k_val = NeoK::ONE;
        for _ in 0..k_total {
            gamma_to_k_val *= gamma_val;
        }

        // Inner weighted sum over (j, i>=2).
        let t = if me_inputs_y_vars.is_empty() {
            0
        } else {
            me_inputs_y_vars[0].len()
        };

        let mut inner_val = NeoK::ZERO;
        let mut inner = helpers::k_zero(
            cs,
            &format!("step_{}_T_inner_init", step_idx),
        )?;

        for j in 0..t {
            // (γ^k_total)^j – shared across all i for this j, tracked natively and then lifted.
            let mut gamma_k_j_val = NeoK::ONE;
            for _ in 0..j {
                gamma_k_j_val *= gamma_to_k_val;
            }
            let gamma_k_j = helpers::alloc_k_from_neo(
                cs,
                gamma_k_j_val,
                &format!("step_{}_T_gamma_k_j_j{}", step_idx, j),
            )?;

            for (idx, (y_table_vars, y_table_vals)) in
                me_inputs_y_vars.iter().zip(me_inputs_y_vals.iter()).enumerate()
            {
                // me_inputs[idx] corresponds to instance i = idx + 2 in the paper.
                let i_abs = idx + 2;
                let row_vars = &y_table_vars[j];
                let row_vals = &y_table_vals[j];
                let limit = min(d_sz, min(row_vars.len(), row_vals.len()));

                // y_eval = ⟨ y_{(i,j)}, χ_α ⟩
                let mut y_eval_val = NeoK::ZERO;
                let mut y_eval = helpers::k_zero(
                    cs,
                    &format!("step_{}_T_y_eval_j{}_i{}", step_idx, j, i_abs),
                )?;
                for rho in 0..limit {
                    let (prod, prod_val) = helpers::k_mul_with_hint(
                        cs,
                        &row_vars[rho],
                        row_vals[rho],
                        &chi_alpha_vars[rho],
                        chi_alpha_vals[rho],
                        self.delta,
                        &format!(
                            "step_{}_T_y_eval_j{}_i{}_rho{}",
                            step_idx, j, i_abs, rho
                        ),
                    )?;

                    y_eval_val += prod_val;
                    let y_eval_hint = KNum::<CircuitF>::from_neo_k(y_eval_val);
                    y_eval = k_add_raw(
                        cs,
                        &y_eval,
                        &prod,
                        Some(y_eval_hint),
                        &format!(
                            "step_{}_T_y_eval_acc_j{}_i{}_rho{}",
                            step_idx, j, i_abs, rho
                        ),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;
                }

                // γ^{i-1}
                let mut gamma_i_val = NeoK::ONE;
                let mut gamma_i = helpers::k_one(
                    cs,
                    &format!("step_{}_T_gamma_i_init_j{}_i{}", step_idx, j, i_abs),
                )?;
                for pow_idx in 0..(i_abs - 1) {
                    let (new_gamma_i, new_gamma_i_val) = helpers::k_mul_with_hint(
                        cs,
                        &gamma_i,
                        gamma_i_val,
                        gamma_var,
                        gamma_val,
                        self.delta,
                        &format!(
                            "step_{}_T_gamma_i_step_j{}_i{}_{}",
                            step_idx, j, i_abs, pow_idx
                        ),
                    )?;
                    gamma_i = new_gamma_i;
                    gamma_i_val = new_gamma_i_val;
                }

                let (weight, weight_val) = helpers::k_mul_with_hint(
                    cs,
                    &gamma_i,
                    gamma_i_val,
                    &gamma_k_j,
                    gamma_k_j_val,
                    self.delta,
                    &format!("step_{}_T_weight_j{}_i{}", step_idx, j, i_abs),
                )?;

                let (contrib, contrib_val) = helpers::k_mul_with_hint(
                    cs,
                    &weight,
                    weight_val,
                    &y_eval,
                    y_eval_val,
                    self.delta,
                    &format!("step_{}_T_contrib_j{}_i{}", step_idx, j, i_abs),
                )?;

                inner_val += contrib_val;
                let inner_hint = KNum::<CircuitF>::from_neo_k(inner_val);
                inner = k_add_raw(
                    cs,
                    &inner,
                    &contrib,
                    Some(inner_hint),
                    &format!("step_{}_T_inner_acc_j{}_i{}", step_idx, j, i_abs),
                )
                .map_err(SpartanBridgeError::BellpepperError)?;
            }
        }

        // T = inner (weights already include the γ^k factor as in the paper-exact engine).
        Ok(inner)
    }

    /// Build χ-table for an Ajtai challenge vector (α or α') using native K values.
    ///
    /// χ_α[ρ] = ∏_bit (α_bit if ρ_bit=1 else 1-α_bit), computed in the host field and
    /// lifted into K variables. This avoids needing in-circuit K arithmetic for χ.
    #[allow(dead_code)]
    fn build_chi_table<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        alpha_vars: &[KNumVar],
        alpha_values: &[neo_math::K],
        label: &str,
    ) -> Result<Vec<KNumVar>> {
        if alpha_vars.len() != alpha_values.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "chi-table alpha length mismatch at step {}: vars={}, values={}",
                step_idx,
                alpha_vars.len(),
                alpha_values.len()
            )));
        }

        let d_sz = 1usize << alpha_values.len();
        let mut chi: Vec<KNumVar> = Vec::with_capacity(d_sz);

        for rho in 0..d_sz {
            // Host-side χ value for this rho.
            let mut w_native = neo_math::K::ONE;
            for (bit, &a_val) in alpha_values.iter().enumerate() {
                let bit_is_one = ((rho >> bit) & 1) == 1;
                let term = if bit_is_one {
                    a_val
                } else {
                    neo_math::K::ONE - a_val
                };
                w_native *= term;
            }

            let w_var = helpers::alloc_k_from_neo(
                cs,
                w_native,
                &format!("step_{}_{}_chi_{}", step_idx, label, rho),
            )?;
            chi.push(w_var);
        }

        Ok(chi)
    }

    /// Equality polynomial eq_points over K, using the same formula as the
    /// native `eq_points`: ∏_i [(1-p_i)*(1-q_i) + p_i*q_i].
    ///
    /// The value is computed natively from `p_vals`/`q_vals` and lifted into K;
    /// the circuit treats it as a derived witness scalar.
    #[allow(dead_code)]
    fn eq_points<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        p: &[KNumVar],
        q: &[KNumVar],
        p_vals: &[neo_math::K],
        q_vals: &[neo_math::K],
        label: &str,
    ) -> Result<KNumVar> {
        if p.len() != q.len() || p_vals.len() != q_vals.len() || p.len() != p_vals.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "eq_points length mismatch at step {}: p_vars={}, q_vars={}, p_vals={}, q_vals={}",
                step_idx,
                p.len(),
                q.len(),
                p_vals.len(),
                q_vals.len(),
            )));
        }

        // Native eq value.
        let mut acc_native = neo_math::K::ONE;
        for (&pi, &qi) in p_vals.iter().zip(q_vals.iter()) {
            let term = (neo_math::K::ONE - pi) * (neo_math::K::ONE - qi) + pi * qi;
            acc_native *= term;
        }

        let acc_var = helpers::alloc_k_from_neo(
            cs,
            acc_native,
            &format!("step_{}_{}_eq", step_idx, label),
        )?;

        Ok(acc_var)
    }

    /// Range product gadget: ∏_{t=-(b-1)}^{b-1} (val - t) over K.
    #[allow(dead_code)]
    fn range_product<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        val: &KNumVar,
        label: &str,
    ) -> Result<KNumVar> {
        let mut acc = helpers::k_one(
            cs,
            &format!("step_{}_{}_range_init", step_idx, label),
        )?;

        let b = self.base_b as i32;
        for t in (-(b - 1))..=(b - 1) {
            let abs = t.abs() as u64;
            let base = if t >= 0 {
                CircuitF::from(abs)
            } else {
                CircuitF::from(0u64) - CircuitF::from(abs)
            };
            let t_k = helpers::k_const(
                cs,
                base,
                &format!("step_{}_{}_t_{}", step_idx, label, t),
            )?;

            let diff = helpers::k_sub(
                cs,
                val,
                &t_k,
                &format!("step_{}_{}_val_minus_t_{}", step_idx, label, t),
            )?;

            acc = helpers::k_mul(
                cs,
                &acc,
                &diff,
                self.delta,
                &format!("step_{}_{}_range_acc_{}", step_idx, label, t),
            )?;
        }

        Ok(acc)
    }

    /// Evaluate the CCS polynomial f at the given m-values in K.
    #[allow(dead_code)]
    fn eval_poly_f_in_k<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        m_vals: &[KNumVar],
    ) -> Result<KNumVar> {
        let mut acc = helpers::k_zero(
            cs,
            &format!("step_{}_F_prime_init", step_idx),
        )?;

        for (term_idx, term) in self.poly_f.iter().enumerate() {
            // term_val = ∏_j m_j^{exp_j}
            let mut term_val = helpers::k_one(
                cs,
                &format!("step_{}_F_term{}_init", step_idx, term_idx),
            )?;

            for (var_idx, &exp) in term.exps.iter().enumerate() {
                if exp == 0 {
                    continue;
                }
                let base = &m_vals[var_idx];
                let pow = helpers::k_pow(
                    cs,
                    base,
                    exp,
                    self.delta,
                    &format!(
                        "step_{}_F_term{}_var{}_pow",
                        step_idx, term_idx, var_idx
                    ),
                )?;
                term_val = helpers::k_mul(
                    cs,
                    &term_val,
                    &pow,
                    self.delta,
                    &format!(
                        "step_{}_F_term{}_var{}_mul",
                        step_idx, term_idx, var_idx
                    ),
                )?;
            }

            // Scale by coefficient.
            let scaled = helpers::k_scalar_mul(
                cs,
                term.coeff,
                &term_val,
                &format!("step_{}_F_term{}_scaled", step_idx, term_idx),
            )?;

            acc = helpers::k_add(
                cs,
                &acc,
                &scaled,
                &format!("step_{}_F_acc_term{}", step_idx, term_idx),
            )?;
        }

        Ok(acc)
    }

    /// Verify Π-CCS terminal identity for a step:
    /// sumcheck_final == rhs_terminal_identity_paper_exact(α,β,γ,r',α', outputs, inputs.r).
    fn verify_terminal_identity<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        step: &FoldStep,
        _proof: &neo_reductions::PiCcsProof,
        challenges: &crate::circuit::witness::PiCcsChallenges,
        witness: &FoldRunWitness,
        sumcheck_final: &KNumVar,
        out_y_vars: &[Vec<Vec<KNumVar>>],
    ) -> Result<()> {
        use neo_math::{D, K as NeoK};

        // ME inputs for this step (needed only for r in eq((α',r'),(α,r))).
        let me_inputs = if step_idx == 0 {
            &self.instance.initial_accumulator
        } else {
            &witness.fold_run.steps[step_idx - 1].dec_children
        };

        // Outputs y' for this step.
        let out_me = &step.ccs_out;
        if out_me.is_empty() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "Terminal identity at step {}: empty outputs",
                step_idx
            )));
        }

        // Allocate α' and r' (used to tie them to sumcheck challenges).
        let mut alpha_prime_vars = Vec::with_capacity(challenges.alpha_prime.len());
        for (i, &k) in challenges.alpha_prime.iter().enumerate() {
            alpha_prime_vars.push(helpers::alloc_k_from_neo(
                cs,
                k,
                &format!("step_{}_alpha_prime_{}", step_idx, i),
            )?);
        }

        let mut r_prime_vars = Vec::with_capacity(challenges.r_prime.len());
        for (i, &k) in challenges.r_prime.iter().enumerate() {
            r_prime_vars.push(helpers::alloc_k_from_neo(
                cs,
                k,
                &format!("step_{}_r_prime_{}", step_idx, i),
            )?);
        }

        // Enforce that (r_prime, alpha_prime) is exactly the split of the
        // per-round sumcheck challenges vector.
        let total_chals = challenges.sumcheck_challenges.len();
        let rows = r_prime_vars.len();
        let ajtai = alpha_prime_vars.len();
        if total_chals != rows + ajtai {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "sumcheck_challenges length mismatch at step {}: expected {} (rows + ajtai), got {}",
                step_idx,
                rows + ajtai,
                total_chals
            )));
        }
        // r' == prefix of sumcheck_challenges
        for i in 0..rows {
            let sc_val = helpers::alloc_k_from_neo(
                cs,
                challenges.sumcheck_challenges[i],
                &format!("step_{}_sc_round_chal_row_{}", step_idx, i),
            )?;
            helpers::enforce_k_eq(
                cs,
                &r_prime_vars[i],
                &sc_val,
                &format!("step_{}_r_prime_matches_sc_{}", step_idx, i),
            );
        }
        // α' == suffix of sumcheck_challenges
        for j in 0..ajtai {
            let idx = rows + j;
            let sc_val = helpers::alloc_k_from_neo(
                cs,
                challenges.sumcheck_challenges[idx],
                &format!("step_{}_sc_round_chal_ajtai_{}", step_idx, j),
            )?;
            helpers::enforce_k_eq(
                cs,
                &alpha_prime_vars[j],
                &sc_val,
                &format!("step_{}_alpha_prime_matches_sc_{}", step_idx, j),
            );
        }

        // --- Scalar equality polynomials eq((α',r'),β) and eq((α',r'),(α,r)) ---
        //
        // These are still computed natively in K from the challenge values
        // and then lifted into the circuit as K scalars.

        // eq((α',r'), β) = eq(α', β_a) * eq(r', β_r), in K.
        let eq_points_native = |p: &[NeoK], q: &[NeoK]| -> NeoK {
            if p.len() != q.len() {
                return NeoK::ZERO;
            }
            let mut acc = NeoK::ONE;
            for (&pi, &qi) in p.iter().zip(q.iter()) {
                let term = (NeoK::ONE - pi) * (NeoK::ONE - qi) + pi * qi;
                acc *= term;
            }
                acc
        };

        let eq_aprp_beta_native = if challenges.alpha_prime.is_empty() && challenges.r_prime.is_empty() {
            NeoK::ONE
        } else {
            let e1 = eq_points_native(&challenges.alpha_prime, &challenges.beta_a);
            let e2 = eq_points_native(&challenges.r_prime, &challenges.beta_r);
            e1 * e2
        };

        // eq((α',r'),(α,r)) if we have ME inputs; else 0 (Eval' block vanishes).
        let eq_aprp_ar_native = if let Some(first_input) = me_inputs.first() {
            let e1 = eq_points_native(&challenges.alpha_prime, &challenges.alpha);
            let e2 = eq_points_native(&challenges.r_prime, &first_input.r);
            e1 * e2
        } else {
            NeoK::ZERO
        };

        // Lift eq scalars into KNumVars.
        let eq_aprp_beta = helpers::alloc_k_from_neo(
            cs,
            eq_aprp_beta_native,
            &format!("step_{}_eq_aprp_beta", step_idx),
        )?;
        let eq_aprp_ar = helpers::alloc_k_from_neo(
            cs,
            eq_aprp_ar_native,
            &format!("step_{}_eq_aprp_ar", step_idx),
        )?;

        // --- Allocate γ and precompute γ^k_total in K ---
        let gamma_var = helpers::alloc_k_from_neo(
            cs,
            challenges.gamma,
            &format!("step_{}_gamma_terminal", step_idx),
        )?;

        let k_total = out_me.len();
        // Compute γ^k_total natively and lift as a single K constant, to avoid
        // introducing additional K-multiplication constraints here.
        let mut gamma_k_total_val = neo_math::K::ONE;
        for _ in 0..k_total {
            gamma_k_total_val *= challenges.gamma;
        }
        let gamma_k_total = helpers::alloc_k_from_neo(
            cs,
            gamma_k_total_val,
            &format!("step_{}_gamma_k_total", step_idx),
        )?;

        // --- F' from first output's y'[i=1] in-circuit ---
        //
        // Recompose m_j from Ajtai digits with base-b using only the first D
        // digits, then evaluate f via poly_f in K.
        let t = out_me[0].y.len();
        let F_prime = if t == 0 {
            helpers::k_zero(cs, &format!("step_{}_F_prime_zero", step_idx))?
        } else {
            // Use the shared y-table allocation for the first output.
            let first_out_y = &out_y_vars[0];
            let d_pad = first_out_y[0].len();
            let d_ring = D.min(d_pad);

            let mut m_vals: Vec<KNumVar> = Vec::with_capacity(t);
            for j in 0..t {
                if first_out_y[j].len() != d_pad {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "Terminal identity at step {}: inconsistent y row length in output 0 (j={})",
                        step_idx, j
                    )));
                }
                let m_j = crate::gadgets::pi_ccs::base_b_recompose_k(
                    cs,
                    &first_out_y[j][..d_ring],
                    self.base_b,
                    self.delta,
                    &format!("step_{}_F_m_j{}", step_idx, j),
                )?;
                m_vals.push(m_j);
            }

            self.eval_poly_f_in_k(
                cs,
                step_idx,
                &m_vals,
            )?
        };

        // --- χ_{α'} table in K for Ajtai domain ---
        let d_sz = 1usize << challenges.alpha_prime.len();
        let mut chi_alpha_prime: Vec<KNumVar> = Vec::with_capacity(d_sz);
        for rho in 0..d_sz {
            let mut w = helpers::k_one(
                cs,
                &format!("step_{}_chi_alpha_prime_{}_init", step_idx, rho),
            )?;
            for (bit, a_bit) in alpha_prime_vars.iter().enumerate() {
                let bit_is_one = ((rho >> bit) & 1) == 1;
                let factor = if bit_is_one {
                    a_bit.clone()
                } else {
                    helpers::k_one_minus(
                        cs,
                        a_bit,
                        &format!(
                            "step_{}_chi_alpha_prime_{}_bit{}_one_minus",
                            step_idx, rho, bit
                        ),
                    )?
                };
                w = helpers::k_mul(
                    cs,
                    &w,
                    &factor,
                    self.delta,
                    &format!(
                        "step_{}_chi_alpha_prime_{}_bit{}",
                        step_idx, rho, bit
                    ),
                )?;
            }
            chi_alpha_prime.push(w);
        }

        // --- Σ γ^i · N_i' over outputs, in K ---
        //
        // N_i' = ∏_{t} ( ẏ'_{(i,1)}(α') - t ), with ẏ' evaluated at α' as MLE.
        let mut nc_prime_sum = helpers::k_zero(
            cs,
            &format!("step_{}_N_prime_sum_init", step_idx),
        )?;

        // g = γ^1
        let mut g = gamma_var.clone();
        for (i_idx, out_y) in out_y_vars.iter().enumerate() {
            // ẏ'_{(i,1)}(α') uses j = 0 row.
            if out_y.is_empty() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "Terminal identity at step {}: empty y table for output {}",
                    step_idx, i_idx
                )));
            }
            let y1 = &out_y[0];
            let limit = core::cmp::min(chi_alpha_prime.len(), y1.len());

            let mut y_eval = helpers::k_zero(
                cs,
                &format!("step_{}_N_y_eval_i{}", step_idx, i_idx),
            )?;
            for rho in 0..limit {
                let prod = helpers::k_mul(
                    cs,
                    &y1[rho],
                    &chi_alpha_prime[rho],
                    self.delta,
                    &format!(
                        "step_{}_N_y_eval_i{}_rho{}",
                        step_idx, i_idx, rho
                    ),
                )?;
                y_eval = helpers::k_add(
                    cs,
                    &y_eval,
                    &prod,
                    &format!(
                        "step_{}_N_y_eval_acc_i{}_rho{}",
                        step_idx, i_idx, rho
                    ),
                )?;
            }

            let Ni = self.range_product(
                cs,
                step_idx,
                &y_eval,
                &format!("step_{}_N_range_i{}", step_idx, i_idx),
            )?;

            let gNi = helpers::k_mul(
                cs,
                &g,
                &Ni,
                self.delta,
                &format!("step_{}_N_weighted_i{}", step_idx, i_idx),
            )?;

            nc_prime_sum = helpers::k_add(
                cs,
                &nc_prime_sum,
                &gNi,
                &format!("step_{}_N_acc_i{}", step_idx, i_idx),
            )?;

            // g <- g * γ
            g = helpers::k_mul(
                cs,
                &g,
                &gamma_var,
                self.delta,
                &format!("step_{}_N_gamma_step_i{}", step_idx, i_idx),
            )?;
        }

        // --- Eval' block in K ---
        //
        // γ^k · Σ_{j=1,i=2}^{t,k} γ^{i+(j-1)k-1} · E_{(i,j)} with
        // E_{(i,j)} = eq((α',r'),(α,r)) · ẏ'_{(i,j)}(α').
        let mut eval_sum = helpers::k_zero(
            cs,
            &format!("step_{}_Eval_sum_init", step_idx),
        )?;

        if me_inputs.first().is_some() && k_total >= 2 {
            for j in 0..t {
                // Precompute (γ^k_total)^j once per j and reuse across outputs.
                let mut gamma_k_j = helpers::k_one(
                    cs,
                    &format!("step_{}_Eval_gamma_k_j_init_j{}", step_idx, j),
                )?;
                for pow_idx in 0..j {
                    gamma_k_j = helpers::k_mul(
                        cs,
                        &gamma_k_j,
                        &gamma_k_total,
                        self.delta,
                        &format!(
                            "step_{}_Eval_gamma_k_j_step_j{}_{}",
                            step_idx,
                            j,
                            pow_idx
                        ),
                    )?;
                }

                for (i_abs, out_y) in out_y_vars.iter().enumerate().skip(1) {
                    if out_y.len() != t {
                        return Err(SpartanBridgeError::InvalidInput(format!(
                            "Terminal identity at step {}: y length mismatch in output {}",
                            step_idx, i_abs
                        )));
                    }
                    let row = &out_y[j];
                    let limit = core::cmp::min(chi_alpha_prime.len(), row.len());

                    let mut y_eval = helpers::k_zero(
                        cs,
                        &format!("step_{}_Eval_y_eval_j{}_i{}", step_idx, j, i_abs),
                    )?;
                    for rho in 0..limit {
                        let prod = helpers::k_mul(
                            cs,
                            &row[rho],
                            &chi_alpha_prime[rho],
                            self.delta,
                            &format!(
                                "step_{}_Eval_y_eval_j{}_i{}_rho{}",
                                step_idx, j, i_abs, rho
                            ),
                        )?;
                        y_eval = helpers::k_add(
                            cs,
                            &y_eval,
                            &prod,
                            &format!(
                                "step_{}_Eval_y_eval_acc_j{}_i{}_rho{}",
                                step_idx, j, i_abs, rho
                            ),
                        )?;
                    }

                    // weight = γ^{i_abs} * (γ^k_total)^j  (0-based indices)
                    let mut gamma_i = helpers::k_one(
                        cs,
                        &format!("step_{}_Eval_gamma_i_init_j{}_i{}", step_idx, j, i_abs),
                    )?;
                    for pow_idx in 0..i_abs {
                        gamma_i = helpers::k_mul(
                            cs,
                            &gamma_i,
                            &gamma_var,
                            self.delta,
                            &format!(
                                "step_{}_Eval_gamma_i_step_j{}_i{}_{}",
                                step_idx,
                                j,
                                i_abs,
                                pow_idx
                            ),
                        )?;
                    }

                    let weight = helpers::k_mul(
                        cs,
                        &gamma_i,
                        &gamma_k_j,
                        self.delta,
                        &format!("step_{}_Eval_weight_j{}_i{}", step_idx, j, i_abs),
                    )?;

                    let contrib = helpers::k_mul(
                        cs,
                        &weight,
                        &y_eval,
                        self.delta,
                        &format!("step_{}_Eval_contrib_j{}_i{}", step_idx, j, i_abs),
                    )?;

                    eval_sum = helpers::k_add(
                        cs,
                        &eval_sum,
                        &contrib,
                        &format!("step_{}_Eval_acc_j{}_i{}", step_idx, j, i_abs),
                    )?;
                }
            }
        }

        // Assemble RHS in K:
        // v = eq((α',r'), β)·(F' + Σ γ^i N_i') + γ^k · eq((α',r'), (α,r)) · Eval'.
        let F_plus_N = helpers::k_add(
            cs,
            &F_prime,
            &nc_prime_sum,
            &format!("step_{}_RHS_F_plus_N", step_idx),
        )?;
        let left = helpers::k_mul(
            cs,
            &eq_aprp_beta,
            &F_plus_N,
            self.delta,
            &format!("step_{}_RHS_left", step_idx),
        )?;

        // Eval' := γ^k · Σ_{j,i} γ^{i-1 + j·k} · ẏ'_{(i,j)}(α').
        let eval_sum_scaled = helpers::k_mul(
            cs,
            &gamma_k_total,
            &eval_sum,
            self.delta,
            &format!("step_{}_Eval_sum_scaled", step_idx),
        )?;
        let right = helpers::k_mul(
            cs,
            &eq_aprp_ar,
            &eval_sum_scaled,
            self.delta,
            &format!("step_{}_RHS_right", step_idx),
        )?;

        let rhs = helpers::k_add(
            cs,
            &left,
            &right,
            &format!("step_{}_RHS_total", step_idx),
        )?;

        // Enforce that the in-circuit final running sum from sumcheck
        // rounds equals the RHS terminal identity.
        helpers::enforce_k_eq(
            cs,
            sumcheck_final,
            &rhs,
            &format!("step_{}_final_sum_matches_rhs", step_idx),
        );

        Ok(())
    }

    /// Verify RLC equalities for a step
    ///
    /// Enforces the public RLC relations used in `rlc_public`:
    /// - r is preserved: parent.r == child.r for all inputs
    /// - X_parent = Σ_i ρ_i · X_i
    /// - y_parent[j] = Σ_i ρ_i · y_(i,j) (first D digits)
    fn verify_rlc<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        step: &FoldStep,
        _witness: &FoldRunWitness,
        children_y_vars: &[Vec<Vec<KNumVar>>],
    ) -> Result<()> {
        let parent = &step.rlc_parent;
        let children = &step.ccs_out;
        let rhos: &[Mat<NeoF>] = &step.rlc_rhos;

        if children.is_empty() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "RLC at step {} has no children",
                step_idx
            )));
        }
        if children.len() != rhos.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "RLC at step {}: rhos/children length mismatch (rhos={}, children={})",
                step_idx,
                rhos.len(),
                children.len()
            )));
        }

        let d = parent.X.rows();
        let m_in = parent.m_in;

        // Dimension sanity checks
        for (i, child) in children.iter().enumerate() {
            if child.X.rows() != d || child.X.cols() != parent.X.cols() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "RLC X dimension mismatch at step {}, child {}",
                    step_idx, i
                )));
            }
            if child.m_in != m_in {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "RLC m_in mismatch at step {}, child {}",
                    step_idx, i
                )));
            }
        }
        for (i, rho) in rhos.iter().enumerate() {
            if rho.rows() != d || rho.cols() != d {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "RLC ρ dimension mismatch at step {}, matrix {}",
                    step_idx, i
                )));
            }
        }

        // Allocate X matrices for parent and children
        let parent_X_vars = helpers::alloc_matrix_from_neo(
            cs,
            &parent.X,
            &format!("step_{}_rlc_parent_X", step_idx),
        )?;

        let mut child_X_vars = Vec::with_capacity(children.len());
        for (i, child) in children.iter().enumerate() {
            child_X_vars.push(helpers::alloc_matrix_from_neo(
                cs,
                &child.X,
                &format!("step_{}_rlc_child_{}_X", step_idx, i),
            )?);
        }

        // Enforce X_parent = Σ_i ρ_i · X_i
        for row in 0usize..d {
            for col in 0usize..m_in {
                cs.enforce(
                    || format!("step_{}_rlc_X_{}_{}", step_idx, row, col),
                    |lc| {
                        let mut res = lc;
                        for (i, rho) in rhos.iter().enumerate() {
                            for k in 0..d {
                                let coeff = helpers::neo_f_to_circuit(&rho[(row, k)]);
                                if coeff != CircuitF::from(0u64) {
                                    res = res + (coeff, child_X_vars[i][k][col]);
                                }
                            }
                        }
                        res
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + parent_X_vars[row][col],
                );
            }
        }

        // Enforce r preservation: parent.r == child.r for all inputs
        let r_len = parent.r.len();
        for (i, child) in children.iter().enumerate() {
            if child.r.len() != r_len {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "RLC r length mismatch at step {}, child {}",
                    step_idx, i
                )));
            }
            for idx in 0..r_len {
                let parent_r_var = helpers::alloc_k_from_neo(
                    cs,
                    parent.r[idx],
                    &format!("step_{}_rlc_parent_r_{}_{}", step_idx, i, idx),
                )?;
                let child_r_var = helpers::alloc_k_from_neo(
                    cs,
                    child.r[idx],
                    &format!("step_{}_rlc_child_{}_r_{}", step_idx, i, idx),
                )?;
                helpers::enforce_k_eq(
                    cs,
                    &parent_r_var,
                    &child_r_var,
                    &format!("step_{}_rlc_r_eq_child_{}_idx_{}", step_idx, i, idx),
                );
            }
        }

        // Enforce y_parent[j] = Σ_i ρ_i · y_(i,j) on the first D digits.
        // Use the ring's D dimension and y-vector lengths.
        let t = parent.y.len();
        if t == 0 {
            return Ok(());
        }
        let d_pad = parent.y[0].len();
        if d_pad == 0 {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "RLC at step {}: empty y vectors in parent",
                step_idx
            )));
        }
        let d_ring = neo_math::D.min(d_pad);

        let parent_y_vars = helpers::alloc_y_table_from_neo(
            cs,
            &parent.y,
            &format!("step_{}_rlc_parent_y", step_idx),
        )?;

        if children_y_vars.len() != children.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "RLC at step {}: children_y_vars/children length mismatch (vars={}, children={})",
                step_idx,
                children_y_vars.len(),
                children.len()
            )));
        }
        for (i, child) in children.iter().enumerate() {
            if child.y.len() != t {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "RLC y length mismatch at step {}, child {}",
                    step_idx, i
                )));
            }
        }

        for j in 0..t {
            if parent.y[j].len() != d_pad {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "RLC parent y[j] length mismatch at step {}, j={}",
                    step_idx, j
                )));
            }
            for r_idx in 0..d_ring {
                // c0 component
                cs.enforce(
                    || format!("step_{}_rlc_y_c0_j{}_r{}", step_idx, j, r_idx),
                    |lc| {
                        let mut res = lc;
                        for (i, rho) in rhos.iter().enumerate() {
                            for k in 0..d_ring {
                                let coeff = helpers::neo_f_to_circuit(&rho[(r_idx, k)]);
                                if coeff != CircuitF::from(0u64) {
                                    res = res + (coeff, children_y_vars[i][j][k].c0);
                                }
                            }
                        }
                        res
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + parent_y_vars[j][r_idx].c0,
                );

                // c1 component
                cs.enforce(
                    || format!("step_{}_rlc_y_c1_j{}_r{}", step_idx, j, r_idx),
                    |lc| {
                        let mut res = lc;
                        for (i, rho) in rhos.iter().enumerate() {
                            for k in 0..d_ring {
                                let coeff = helpers::neo_f_to_circuit(&rho[(r_idx, k)]);
                                if coeff != CircuitF::from(0u64) {
                                    res = res + (coeff, children_y_vars[i][j][k].c1);
                                }
                            }
                        }
                        res
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + parent_y_vars[j][r_idx].c1,
                );
            }
        }

        Ok(())
    }

    /// Verify DEC equalities for a step
    ///
    /// Enforces the public DEC relations (ignoring commitments for now):
    /// - r is preserved: parent.r == child.r for all children
    /// - X_parent = Σ_i b^i · X_i
    /// - y_parent[j] = Σ_i b^i · y_(i,j)
    fn verify_dec<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        step: &FoldStep,
        _witness: &FoldRunWitness,
    ) -> Result<()> {
        let parent = &step.rlc_parent;
        let children = &step.dec_children;

        if children.is_empty() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "DEC at step {} has no children",
                step_idx
            )));
        }

        let d = parent.X.rows();
        let m_in = parent.m_in;

        for (i, child) in children.iter().enumerate() {
            if child.X.rows() != d || child.X.cols() != parent.X.cols() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "DEC X dimension mismatch at step {}, child {}",
                    step_idx, i
                )));
            }
            if child.m_in != m_in {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "DEC m_in mismatch at step {}, child {}",
                    step_idx, i
                )));
            }
        }

        // Allocate X matrices
        let parent_X_vars = helpers::alloc_matrix_from_neo(
            cs,
            &parent.X,
            &format!("step_{}_dec_parent_X", step_idx),
        )?;

        let mut child_X_vars = Vec::with_capacity(children.len());
        for (i, child) in children.iter().enumerate() {
            child_X_vars.push(helpers::alloc_matrix_from_neo(
                cs,
                &child.X,
                &format!("step_{}_dec_child_{}_X", step_idx, i),
            )?);
        }

        // X_parent = Σ b^i · X_i
        for row in 0usize..d {
            for col in 0usize..m_in {
                cs.enforce(
                    || format!("step_{}_dec_X_{}_{}", step_idx, row, col),
                    |lc| {
                        let mut res = lc;
                        let mut pow = CircuitF::from(1u64);
                        for child_vars in child_X_vars.iter() {
                            res = res + (pow, child_vars[row][col]);
                            pow *= CircuitF::from(self.base_b as u64);
                        }
                        res
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + parent_X_vars[row][col],
                );
            }
        }

        // r preservation: parent.r == child.r for all children
        let r_len = parent.r.len();
        for (i, child) in children.iter().enumerate() {
            if child.r.len() != r_len {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "DEC r length mismatch at step {}, child {}",
                    step_idx, i
                )));
            }
            for idx in 0..r_len {
                let parent_r_var = helpers::alloc_k_from_neo(
                    cs,
                    parent.r[idx],
                    &format!("step_{}_dec_parent_r_{}_{}", step_idx, i, idx),
                )?;
                let child_r_var = helpers::alloc_k_from_neo(
                    cs,
                    child.r[idx],
                    &format!("step_{}_dec_child_{}_r_{}", step_idx, i, idx),
                )?;
                helpers::enforce_k_eq(
                    cs,
                    &parent_r_var,
                    &child_r_var,
                    &format!("step_{}_dec_r_eq_child_{}_idx_{}", step_idx, i, idx),
                );
            }
        }

        // y_parent[j] = Σ b^i · y_(i,j)
        let t = parent.y.len();
        if t == 0 {
            return Ok(());
        }
        let d_pad = parent.y[0].len();
        if d_pad == 0 {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "DEC at step {}: empty y vectors in parent",
                step_idx
            )));
        }

        let parent_y_vars = helpers::alloc_y_table_from_neo(
            cs,
            &parent.y,
            &format!("step_{}_dec_parent_y", step_idx),
        )?;

        let mut children_y_vars = Vec::with_capacity(children.len());
        for (i, child) in children.iter().enumerate() {
            if child.y.len() != t {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "DEC y length mismatch at step {}, child {}",
                    step_idx, i
                )));
            }
            for y_j in &child.y {
                if y_j.len() != d_pad {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "DEC child y[j] length mismatch at step {}, child {}",
                        step_idx, i
                    )));
                }
            }
            children_y_vars.push(helpers::alloc_y_table_from_neo(
                cs,
                &child.y,
                &format!("step_{}_dec_child_{}_y", step_idx, i),
            )?);
        }

        for j in 0..t {
            for r_idx in 0..d_pad {
                // c0 component
                cs.enforce(
                    || format!("step_{}_dec_y_c0_j{}_r{}", step_idx, j, r_idx),
                    |lc| {
                        let mut res = lc;
                        let mut pow = CircuitF::from(1u64);
                        for child_y in children_y_vars.iter() {
                            res = res + (pow, child_y[j][r_idx].c0);
                            pow *= CircuitF::from(self.base_b as u64);
                        }
                        res
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + parent_y_vars[j][r_idx].c0,
                );

                // c1 component
                cs.enforce(
                    || format!("step_{}_dec_y_c1_j{}_r{}", step_idx, j, r_idx),
                    |lc| {
                        let mut res = lc;
                        let mut pow = CircuitF::from(1u64);
                        for child_y in children_y_vars.iter() {
                            res = res + (pow, child_y[j][r_idx].c1);
                            pow *= CircuitF::from(self.base_b as u64);
                        }
                        res
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + parent_y_vars[j][r_idx].c1,
                );
            }
        }

        Ok(())
    }

    /// Verify accumulator chaining across steps.
    ///
    /// Currently enforces that the final accumulator claimed in the public
    /// instance matches the `FoldRun`'s final outputs. Full cross-step chaining
    /// (linking DEC children to the next step's inputs) is left for future work
    /// once those inputs are explicitly exposed in the witness.
    fn verify_accumulator_chaining<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        witness: &FoldRunWitness,
    ) -> Result<()> {
        let final_public = &self.instance.final_accumulator;
        let final_witness = &witness.fold_run.final_outputs;

        if final_public.len() != final_witness.len() {
            return Err(SpartanBridgeError::InvalidInput(
                "Final accumulator length mismatch".into(),
            ));
        }

        for (idx, (pub_me, wit_me)) in final_public.iter().zip(final_witness.iter()).enumerate() {
            // Enforce X equality
            if pub_me.X.rows() != wit_me.X.rows() || pub_me.X.cols() != wit_me.X.cols() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "Final accumulator X dimension mismatch at index {}",
                    idx
                )));
            }

            let pub_X_vars = helpers::alloc_matrix_from_neo(
                cs,
                &pub_me.X,
                &format!("acc_final_pub_{}_X", idx),
            )?;
            let wit_X_vars = helpers::alloc_matrix_from_neo(
                cs,
                &wit_me.X,
                &format!("acc_final_wit_{}_X", idx),
            )?;

            for r in 0..pub_me.X.rows() {
                for c in 0..pub_me.X.cols() {
                    cs.enforce(
                        || format!("acc_final_X_eq_{}_{}_{}", idx, r, c),
                        |lc| lc + pub_X_vars[r][c],
                        |lc| lc + CS::one(),
                        |lc| lc + wit_X_vars[r][c],
                    );
                }
            }

            // Enforce r equality
            if pub_me.r.len() != wit_me.r.len() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "Final accumulator r length mismatch at index {}",
                    idx
                )));
            }
            for j in 0..pub_me.r.len() {
                let pub_r = helpers::alloc_k_from_neo(
                    cs,
                    pub_me.r[j],
                    &format!("acc_final_pub_{}_r_{}", idx, j),
                )?;
                let wit_r = helpers::alloc_k_from_neo(
                    cs,
                    wit_me.r[j],
                    &format!("acc_final_wit_{}_r_{}", idx, j),
                )?;
                helpers::enforce_k_eq(
                    cs,
                    &pub_r,
                    &wit_r,
                    &format!("acc_final_r_eq_{}_{}", idx, j),
                );
            }

            // Enforce y equality
            if pub_me.y.len() != wit_me.y.len() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "Final accumulator y length mismatch at index {}",
                    idx
                )));
            }

            let pub_y_vars = helpers::alloc_y_table_from_neo(
                cs,
                &pub_me.y,
                &format!("acc_final_pub_{}_y", idx),
            )?;
            let wit_y_vars = helpers::alloc_y_table_from_neo(
                cs,
                &wit_me.y,
                &format!("acc_final_wit_{}_y", idx),
            )?;

            for j in 0..pub_me.y.len() {
                if pub_me.y[j].len() != wit_me.y[j].len() {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "Final accumulator y[j] length mismatch at index {}, j={}",
                        idx, j
                    )));
                }
                for r_idx in 0..pub_me.y[j].len() {
                    helpers::enforce_k_eq(
                        cs,
                        &pub_y_vars[j][r_idx],
                        &wit_y_vars[j][r_idx],
                        &format!("acc_final_y_eq_{}_{}_{}", idx, j, r_idx),
                    );
                }
            }
        }

        Ok(())
    }

}

/// Implement Spartan2's `SpartanCircuit` trait for `FoldRunCircuit` using the
/// Goldilocks + Hash-MLE PCS engine. This lets Spartan2 treat the FoldRun
/// circuit as an R1CS provider.
impl SpartanCircuitTrait<GoldilocksP3MerkleMleEngine> for FoldRunCircuit {
    fn public_values(&self) -> std::result::Result<Vec<CircuitF>, SynthesisError> {
        // Must mirror `allocate_public_inputs`: expose params/CCS/MCS digests as
        // 4 little-endian u64 limbs each, in that order.
        fn append_digest(out: &mut Vec<CircuitF>, digest: &[u8; 32]) {
            for chunk in digest.chunks(8) {
                let mut limb_bytes = [0u8; 8];
                limb_bytes.copy_from_slice(chunk);
                let limb_u64 = u64::from_le_bytes(limb_bytes);
                out.push(CircuitF::from(limb_u64));
            }
        }

        let mut vals = Vec::with_capacity(12);
        append_digest(&mut vals, &self.instance.params_digest);
        append_digest(&mut vals, &self.instance.ccs_digest);
        append_digest(&mut vals, &self.instance.mcs_digest);
        Ok(vals)
    }

    fn shared<CS: ConstraintSystem<CircuitF>>(
        &self,
        _cs: &mut CS,
    ) -> std::result::Result<Vec<AllocatedNum<CircuitF>>, SynthesisError> {
        // This circuit does not use "shared" variables across multiple Spartan
        // circuits; all variables are allocated inside `synthesize`.
        Ok(Vec::new())
    }

    fn precommitted<CS: ConstraintSystem<CircuitF>>(
        &self,
        _cs: &mut CS,
        _shared: &[AllocatedNum<CircuitF>],
    ) -> std::result::Result<Vec<AllocatedNum<CircuitF>>, SynthesisError> {
        // We do not distinguish precommitted variables; everything is handled
        // directly in `synthesize`.
        Ok(Vec::new())
    }

    fn num_challenges(&self) -> usize {
        // Spartan's transcript-driven "challenges" are not used by this
        // circuit; all randomness comes from the Neo folding transcript
        // (already baked into `FoldRunInstance` / `FoldRunWitness`).
        0
    }

    fn synthesize<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        _shared: &[AllocatedNum<CircuitF>],
        _precommitted: &[AllocatedNum<CircuitF>],
        _challenges: Option<&[CircuitF]>,
    ) -> std::result::Result<(), SynthesisError> {
        // Delegate to the existing FoldRunCircuit::synthesize and map any
        // bridge errors back into a SynthesisError understood by Spartan2.
        self.synthesize(cs).map_err(|e| match e {
            SpartanBridgeError::BellpepperError(inner) => inner,
            // For higher-level bridge errors (invalid input, etc.), treat them
            // as unsatisfiable circuits from the SNARK's perspective.
            _ => SynthesisError::Unsatisfiable,
        })
    }
}
