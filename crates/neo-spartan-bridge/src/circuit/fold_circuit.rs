//! Main FoldRun circuit implementation
//!
//! This synthesizes R1CS constraints to verify an entire FoldRun:
//! - For each fold step:
//!   - Verify Π-CCS terminal identity
//!   - Verify sumcheck rounds
//!   - Verify RLC equalities
//!   - Verify DEC equalities
//! - Verify accumulator chaining between steps

use crate::circuit::witness::{FoldRunInstance, FoldRunWitness};
use crate::error::{Result, SpartanBridgeError};
use crate::gadgets::k_field::{alloc_k, k_add as k_add_raw, KNum, KNumVar};
use crate::gadgets::pi_ccs::{sumcheck_eval_gadget, sumcheck_round_gadget};
use crate::gadgets::sponge::Poseidon2Sponge;
use crate::gadgets::sumcheck::{verify_batched_sumcheck_rounds_ds, verify_sumcheck_rounds_ds};
use crate::gadgets::transcript::Poseidon2TranscriptVar;
use crate::CircuitF;
use crate::statement::{
    STATEMENT_IO_ACC_FINAL_MAIN_DIGEST_OFFSET, STATEMENT_IO_ACC_FINAL_VAL_DIGEST_OFFSET, STATEMENT_IO_ACC_INIT_DIGEST_OFFSET,
    STATEMENT_IO_PROGRAM_IO_DIGEST_OFFSET, STATEMENT_IO_STEP_LINKING_DIGEST_OFFSET, STATEMENT_IO_STEPS_DIGEST_OFFSET,
};
use bellpepper_core::{ConstraintSystem, SynthesisError};
use neo_ccs::Mat;
use neo_fold::output_binding::OutputBindingConfig;
use neo_fold::shard::{FoldStep, StepProof};
use neo_math::{F as NeoF, K as NeoK};
use neo_math::KExtensions;
use neo_memory::output_check::OutputBindingProof;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

// Import helper functions from separate module
use super::fold_circuit_helpers as helpers;
use super::route_a_memory_terminal::{self, ShoutAddrPreCircuitData, TwistAddrPreCircuitData, TwistValEvalCircuitData};
use super::route_a_time::verify_route_a_batched_time_step;

// Spartan2 integration: implement SpartanCircuit over Goldilocks + Hash-MLE PCS.
use bellpepper_core::num::AllocatedNum;
use spartan2::provider::GoldilocksMerkleMleEngine;
use spartan2::traits::circuit::SpartanCircuit as SpartanCircuitTrait;

/// Sparse representation of the CCS polynomial f in the circuit field.
///
/// Each term is coeff * ∏_j m_j^{exps[j]}.
#[derive(Clone, Debug)]
pub struct CircuitPolyTerm {
    /// Coefficient in the circuit field (Spartan's Goldilocks).
    pub coeff: CircuitF,
    /// Same coefficient in Neo's base field, for native K computations.
    pub coeff_native: NeoF,
    pub exps: Vec<u32>,
}

/// Main circuit for verifying a FoldRun
#[derive(Clone)]
pub struct FoldRunCircuit {
    /// Public instance
    pub instance: FoldRunInstance,

    /// Private witness
    pub witness: Option<FoldRunWitness>,

    /// CCS header constants (bound into the native transcript).
    pub ccs_n: usize,
    pub ccs_m: usize,
    pub ccs_t: usize,
    pub poly_arity: usize,
    pub ell_d: usize,
    pub ell_n: usize,
    pub ell: usize,
    pub d_sc: usize,
    pub lambda: u32,
    pub s_supported: u32,
    pub slack_bits_abs: u32,
    pub slack_sign: u8,
    pub ccs_mat_digest: [u64; 4],

    /// Delta constant for K-field multiplication (u^2 = δ)
    /// For Goldilocks K, δ = 7
    pub delta: CircuitF,

    /// Base parameter b for DEC decomposition
    pub base_b: u32,

    /// CCS polynomial f, converted to circuit field coefficients.
    pub poly_f: Vec<CircuitPolyTerm>,

    /// Optional step-linking pairs (prev_idx, next_idx) enforced across step boundaries.
    ///
    /// This mirrors `neo_fold`'s `check_step_linking` logic, but is pinned into the circuit shape.
    pub step_linking: Vec<(usize, usize)>,
}

#[derive(Clone, Debug)]
struct PublicInputsVars {
    statement_io: Vec<bellpepper_core::Variable>,
}

#[derive(Clone, Debug)]
pub(crate) struct McsInstanceVars {
    pub(crate) x: Vec<bellpepper_core::Variable>,
    pub(crate) c_data: Vec<bellpepper_core::Variable>,
}

#[derive(Clone, Debug)]
pub(crate) struct MeInstanceVars {
    pub(crate) c_data: Vec<bellpepper_core::Variable>,
    pub(crate) X: Vec<Vec<bellpepper_core::Variable>>,
    pub(crate) r: Vec<KNumVar>,
    pub(crate) y: Vec<Vec<KNumVar>>,
}

#[derive(Clone, Debug)]
struct OutputSumcheckStateVars {
    pub r_prime_vars: Vec<KNumVar>,
    pub r_prime_vals: Vec<neo_math::K>,
    pub output_final_var: KNumVar,
    pub eq_eval_var: KNumVar,
    pub eq_eval_val: neo_math::K,
    pub io_mask_eval_var: KNumVar,
    pub io_mask_eval_val: neo_math::K,
    pub val_io_eval_var: KNumVar,
    pub val_io_eval_val: neo_math::K,
}

impl FoldRunCircuit {
    pub fn new(
        instance: FoldRunInstance,
        witness: Option<FoldRunWitness>,
        ccs_n: usize,
        ccs_m: usize,
        ccs_t: usize,
        poly_arity: usize,
        ell_d: usize,
        ell_n: usize,
        ell: usize,
        d_sc: usize,
        lambda: u32,
        s_supported: u32,
        slack_bits_abs: u32,
        slack_sign: u8,
        ccs_mat_digest: [u64; 4],
        delta: CircuitF,
        base_b: u32,
        poly_f: Vec<CircuitPolyTerm>,
        step_linking: Vec<(usize, usize)>,
    ) -> Self {
        Self {
            instance,
            witness,
            ccs_n,
            ccs_m,
            ccs_t,
            poly_arity,
            ell_d,
            ell_n,
            ell,
            d_sc,
            lambda,
            s_supported,
            slack_bits_abs,
            slack_sign,
            ccs_mat_digest,
            delta,
            base_b,
            poly_f,
            step_linking,
        }
    }

    /// Synthesize the full FoldRun circuit
    ///
    /// This is the main entry point for constraint generation
    pub fn synthesize<CS: ConstraintSystem<CircuitF>>(&self, cs: &mut CS) -> Result<()> {
        // Allocate public inputs
        let public_inputs = self.allocate_public_inputs(cs)?;

        // Get witness or error
        let witness = self
            .witness
            .as_ref()
            .ok_or_else(|| SpartanBridgeError::InvalidInput("Missing witness".into()))?;

        // Host-side shape checks (keep dimension logic out of R1CS).
        self.validate_witness_structure(witness)?;

        // Native session transcript (Fiat–Shamir source of truth).
        let mut tr = Poseidon2TranscriptVar::new(cs, b"neo.fold/session", "neo_fold_session")
            .map_err(SpartanBridgeError::BellpepperError)?;

        // Public step metadata digest transcript (statement binding).
        let mut steps_tr = Poseidon2TranscriptVar::new(
            cs,
            b"neo/spartan-bridge/steps_digest/v3",
            "steps_digest_v3",
        )
        .map_err(SpartanBridgeError::BellpepperError)?;
        steps_tr
            .append_message(
                cs,
                b"steps/len",
                &(witness.steps_public.len() as u64).to_le_bytes(),
                "steps_digest_len",
            )
            .map_err(SpartanBridgeError::BellpepperError)?;

        // Allocate the initial accumulator once and thread it through all steps.
        let mut current_inputs_vals = witness.initial_accumulator.clone();
        let mut current_inputs_vars: Vec<MeInstanceVars> = current_inputs_vals
            .iter()
            .enumerate()
            .map(|(i, me)| self.alloc_me_instance_vars(cs, me, &format!("acc_init_{}", i)))
            .collect::<Result<Vec<_>>>()?;

        self.enforce_accumulator_digest_v2(
            cs,
            "acc_init_digest",
            current_inputs_vals.as_slice(),
            &current_inputs_vars,
            &public_inputs.statement_io,
            STATEMENT_IO_ACC_INIT_DIGEST_OFFSET,
        )?;

        // Bind the program I/O digest when output binding is enabled. This binds the public
        // output claim set (addresses + values + mem_idx + num_bits) to the statement.
        let program_io_claims: Option<Vec<(u64, bellpepper_core::Variable, NeoF)>> =
            if self.instance.statement.output_binding_enabled {
                let cfg = witness.output_binding.as_ref().ok_or_else(|| {
                    SpartanBridgeError::InvalidInput("output binding enabled but witness.output_binding is None".into())
                })?;

                cfg.program_io
                    .validate(cfg.num_bits)
                    .map_err(|e| SpartanBridgeError::InvalidInput(format!("invalid ProgramIO for output binding: {e:?}")))?;

                let mut claims: Vec<(u64, bellpepper_core::Variable, NeoF)> =
                    Vec::with_capacity(cfg.program_io.num_claims());
                for (idx, (addr, val)) in cfg.program_io.claims().enumerate() {
                    let v = CircuitF::from(val.as_canonical_u64());
                    let var = cs.alloc(|| format!("program_io_claim_{idx}_val"), || Ok(v))?;
                    claims.push((addr, var, val));
                }

                let mut io_tr = Poseidon2TranscriptVar::new(
                    cs,
                    b"neo/spartan-bridge/program_io_digest/v1",
                    "program_io_digest_v1",
                )
                .map_err(SpartanBridgeError::BellpepperError)?;
                io_tr
                    .append_u64s(
                        cs,
                        b"output_binding/mem_idx",
                        &[cfg.mem_idx as u64],
                        "program_io_digest_mem_idx",
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;
                io_tr
                    .append_u64s(
                        cs,
                        b"output_binding/num_bits",
                        &[cfg.num_bits as u64],
                        "program_io_digest_num_bits",
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;
                io_tr
                    .append_message(
                        cs,
                        b"output_check/num_claims",
                        &(claims.len() as u64).to_le_bytes(),
                        "program_io_digest_num_claims",
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;
                for (idx, (addr, var, val)) in claims.iter().enumerate() {
                    io_tr
                        .append_message(
                            cs,
                            b"output_check/addr",
                            &addr.to_le_bytes(),
                            &format!("program_io_digest_addr_{idx}"),
                        )
                        .map_err(SpartanBridgeError::BellpepperError)?;
                    let v = CircuitF::from(val.as_canonical_u64());
                    io_tr
                        .append_fields_vars(
                            cs,
                            b"output_check/value",
                            &[*var],
                            &[v],
                            &format!("program_io_digest_val_{idx}"),
                        )
                        .map_err(SpartanBridgeError::BellpepperError)?;
                }

                let digest_limbs = io_tr
                    .digest32(cs, "program_io_digest_v1_digest32")
                    .map_err(SpartanBridgeError::BellpepperError)?;
                for limb_idx in 0..4 {
                    cs.enforce(
                        || format!("program_io_digest_v1_limb_{limb_idx}_eq_public"),
                        |lc| lc + digest_limbs[limb_idx].get_variable(),
                        |lc| lc + CS::one(),
                        |lc| lc + public_inputs.statement_io[STATEMENT_IO_PROGRAM_IO_DIGEST_OFFSET + limb_idx],
                    );
                }

                Some(claims)
            } else {
                None
            };

        // Accumulate all val-lane obligations across steps (Phase 2 semantics, Phase 1 plumbing).
        let mut val_obligations_vals: Vec<neo_ccs::MeInstance<neo_ajtai::Commitment, NeoF, neo_math::K>> = Vec::new();
        let mut val_obligations_vars: Vec<MeInstanceVars> = Vec::new();

        // Verify each fold step and extract its DEC children as the next step's inputs.
        let mut prev_mcs_vars: Option<McsInstanceVars> = None;
        for (step_idx, step_proof) in witness.fold_run.steps.iter().enumerate() {
            let is_last = step_idx + 1 == witness.fold_run.steps.len();
            let output_binding = if self.instance.statement.output_binding_enabled && is_last {
                Some((
                    witness
                        .output_binding
                        .as_ref()
                        .ok_or_else(|| SpartanBridgeError::InvalidInput("output binding enabled but witness.output_binding is None".into()))?,
                    witness.fold_run.output_proof.as_ref().ok_or_else(|| {
                        SpartanBridgeError::InvalidInput("output binding enabled but fold_run.output_proof is None".into())
                    })?,
                    program_io_claims
                        .as_ref()
                        .ok_or_else(|| SpartanBridgeError::InvalidInput("output binding enabled but program_io_claims is None".into()))?
                        .as_slice(),
                ))
            } else {
                None
            };

            let step_public = witness
                .steps_public
                .get(step_idx)
                .ok_or_else(|| SpartanBridgeError::InvalidInput(format!("missing steps_public[{step_idx}]")))?;
            let prev_step_public = if step_idx == 0 {
                None
            } else {
                Some(
                    witness
                        .steps_public
                        .get(step_idx - 1)
                        .ok_or_else(|| SpartanBridgeError::InvalidInput(format!("missing steps_public[{}]", step_idx - 1)))?,
                )
            };

            let (next_inputs_vars, mcs_vars, val_children_vars) = self.verify_fold_step_route_a(
                cs,
                &mut tr,
                &mut steps_tr,
                step_idx,
                step_proof,
                step_public,
                prev_step_public,
                &current_inputs_vals,
                &current_inputs_vars,
                prev_mcs_vars.as_ref(),
                output_binding,
            )?;
            current_inputs_vals = step_proof.fold.dec_children.clone();
            current_inputs_vars = next_inputs_vars;
            prev_mcs_vars = Some(mcs_vars);

            if let Some(val_fold) = &step_proof.val_fold {
                val_obligations_vals.extend_from_slice(&val_fold.dec_children);
                val_obligations_vars.extend(val_children_vars);
            } else if !val_children_vars.is_empty() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "step {step_idx}: internal error: got val_children_vars but step_proof.val_fold is None"
                )));
            }
        }

        // Bind steps_digest to the public statement.
        let steps_digest_limbs = steps_tr
            .digest32(cs, "steps_digest_v3_digest32")
            .map_err(SpartanBridgeError::BellpepperError)?;
        for limb_idx in 0..4 {
            cs.enforce(
                || format!("steps_digest_v3_limb_{limb_idx}_eq_public"),
                |lc| lc + steps_digest_limbs[limb_idx].get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + public_inputs.statement_io[STATEMENT_IO_STEPS_DIGEST_OFFSET + limb_idx],
            );
        }

        // Bind the step-linking policy digest to the public statement.
        //
        // This commits the verifier-visible policy (pairs of x-coordinate equality constraints)
        // so the proof meaning is non-malleable even if multiple step-linking profiles share the
        // same circuit shape.
        {
            let mut link_tr = Poseidon2TranscriptVar::new(
                cs,
                b"neo/spartan-bridge/step_linking_digest/v1",
                "step_linking_digest_v1",
            )
            .map_err(SpartanBridgeError::BellpepperError)?;
            link_tr
                .append_message(
                    cs,
                    b"pairs/len",
                    &(self.step_linking.len() as u64).to_le_bytes(),
                    "step_linking_len",
                )
                .map_err(SpartanBridgeError::BellpepperError)?;
            for (i, (prev, next)) in self.step_linking.iter().enumerate() {
                link_tr
                    .append_message(
                        cs,
                        b"pair/idx",
                        &(i as u64).to_le_bytes(),
                        &format!("step_linking_pair_{i}_idx"),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;
                link_tr
                    .append_message(
                        cs,
                        b"pair/prev",
                        &(*prev as u64).to_le_bytes(),
                        &format!("step_linking_pair_{i}_prev"),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;
                link_tr
                    .append_message(
                        cs,
                        b"pair/next",
                        &(*next as u64).to_le_bytes(),
                        &format!("step_linking_pair_{i}_next"),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;
            }
            let link_limbs = link_tr
                .digest32(cs, "step_linking_digest_v1_digest32")
                .map_err(SpartanBridgeError::BellpepperError)?;
            for limb_idx in 0..4 {
                cs.enforce(
                    || format!("step_linking_digest_v1_limb_{limb_idx}_eq_public"),
                    |lc| lc + link_limbs[limb_idx].get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + public_inputs.statement_io[STATEMENT_IO_STEP_LINKING_DIGEST_OFFSET + limb_idx],
                );
            }
        }

        self.enforce_accumulator_digest_v2(
            cs,
            "acc_final_main_digest",
            current_inputs_vals.as_slice(),
            &current_inputs_vars,
            &public_inputs.statement_io,
            STATEMENT_IO_ACC_FINAL_MAIN_DIGEST_OFFSET,
        )?;

        self.enforce_accumulator_digest_v2(
            cs,
            "acc_final_val_digest",
            val_obligations_vals.as_slice(),
            val_obligations_vars.as_slice(),
            &public_inputs.statement_io,
            STATEMENT_IO_ACC_FINAL_VAL_DIGEST_OFFSET,
        )?;

        Ok(())
    }

    fn alloc_me_instance_vars<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        me: &neo_ccs::MeInstance<neo_ajtai::Commitment, NeoF, neo_math::K>,
        label: &str,
    ) -> Result<MeInstanceVars> {
        let mut c_data = Vec::with_capacity(me.c.data.len());
        for (i, c) in me.c.data.iter().enumerate() {
            let v = helpers::neo_f_to_circuit(c);
            let var = cs.alloc(|| format!("{}_c_data_{}", label, i), || Ok(v))?;
            c_data.push(var);
        }

        let X = helpers::alloc_matrix_from_neo(cs, &me.X, &format!("{}_X", label))?;

        let mut r = Vec::with_capacity(me.r.len());
        for (i, &k) in me.r.iter().enumerate() {
            r.push(helpers::alloc_k_from_neo(cs, k, &format!("{}_r_{}", label, i))?);
        }

        let y = helpers::alloc_y_table_from_neo(cs, &me.y, &format!("{}_y", label))?;

        Ok(MeInstanceVars {
            c_data,
            X,
            r,
            y,
        })
    }

    fn enforce_accumulator_digest_v2<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        label: &str,
        acc_vals: &[neo_ccs::MeInstance<neo_ajtai::Commitment, NeoF, neo_math::K>],
        acc_vars: &[MeInstanceVars],
        statement_io: &[bellpepper_core::Variable],
        statement_offset: usize,
    ) -> Result<()> {
        use p3_field::PrimeField64;

        if acc_vals.len() != acc_vars.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "{label}: accumulator length mismatch (vals={}, vars={})",
                acc_vals.len(),
                acc_vars.len()
            )));
        }
        if statement_io.len() < statement_offset + 4 {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "{label}: statement_io too short (len={}, need={})",
                statement_io.len(),
                statement_offset + 4
            )));
        }

        let mut sponge =
            Poseidon2Sponge::new(cs, &format!("{label}_sponge")).map_err(SpartanBridgeError::BellpepperError)?;

        for (i, &b) in b"neo/spartan-bridge/acc_digest/v2".iter().enumerate() {
            let x_val = CircuitF::from(b as u64);
            let x = AllocatedNum::alloc(cs.namespace(|| format!("{label}_ds_{i}")), || Ok(x_val))
                .map_err(SpartanBridgeError::BellpepperError)?;
            cs.enforce(
                || format!("{label}_ds_{i}_is_const"),
                |lc| lc + x.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + (x_val, CS::one()),
            );
            sponge.absorb(cs, x).map_err(SpartanBridgeError::BellpepperError)?;
        }

        let len_val = CircuitF::from(acc_vals.len() as u64);
        let len_num =
            AllocatedNum::alloc(cs.namespace(|| format!("{label}_len")), || Ok(len_val)).map_err(SpartanBridgeError::BellpepperError)?;
        cs.enforce(
            || format!("{label}_len_is_const"),
            |lc| lc + len_num.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + (len_val, CS::one()),
        );
        sponge.absorb(cs, len_num).map_err(SpartanBridgeError::BellpepperError)?;

        for (me_idx, (me_val, me_var)) in acc_vals.iter().zip(acc_vars.iter()).enumerate() {
            let m_in_val = CircuitF::from(me_val.m_in as u64);
            let m_in = AllocatedNum::alloc(cs.namespace(|| format!("{label}_me_{me_idx}_m_in")), || Ok(m_in_val))
                .map_err(SpartanBridgeError::BellpepperError)?;
            cs.enforce(
                || format!("{label}_me_{me_idx}_m_in_is_const"),
                |lc| lc + m_in.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + (m_in_val, CS::one()),
            );
            sponge.absorb(cs, m_in).map_err(SpartanBridgeError::BellpepperError)?;

            // Commitment data (c_data)
            let c_len_val = CircuitF::from(me_val.c.data.len() as u64);
            let c_len = AllocatedNum::alloc(cs.namespace(|| format!("{label}_me_{me_idx}_c_len")), || Ok(c_len_val))
                .map_err(SpartanBridgeError::BellpepperError)?;
            cs.enforce(
                || format!("{label}_me_{me_idx}_c_len_is_const"),
                |lc| lc + c_len.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + (c_len_val, CS::one()),
            );
            sponge.absorb(cs, c_len).map_err(SpartanBridgeError::BellpepperError)?;

            if me_val.c.data.len() != me_var.c_data.len() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "{label}: ME[{me_idx}] c_data length mismatch (vals={}, vars={})",
                    me_val.c.data.len(),
                    me_var.c_data.len()
                )));
            }
            for (i, c) in me_val.c.data.iter().enumerate() {
                let v = CircuitF::from(c.as_canonical_u64());
                let num = AllocatedNum::alloc(cs.namespace(|| format!("{label}_me_{me_idx}_c_{i}")), || Ok(v))
                    .map_err(SpartanBridgeError::BellpepperError)?;
                cs.enforce(
                    || format!("{label}_me_{me_idx}_c_{i}_eq"),
                    |lc| lc + num.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + me_var.c_data[i],
                );
                sponge.absorb(cs, num).map_err(SpartanBridgeError::BellpepperError)?;
            }

            // X matrix
            let x_rows = me_val.X.rows();
            let x_cols = me_val.X.cols();
            let x_rows_val = CircuitF::from(x_rows as u64);
            let x_cols_val = CircuitF::from(x_cols as u64);
            let x_rows_num =
                AllocatedNum::alloc(cs.namespace(|| format!("{label}_me_{me_idx}_X_rows")), || Ok(x_rows_val))
                    .map_err(SpartanBridgeError::BellpepperError)?;
            cs.enforce(
                || format!("{label}_me_{me_idx}_X_rows_is_const"),
                |lc| lc + x_rows_num.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + (x_rows_val, CS::one()),
            );
            sponge.absorb(cs, x_rows_num).map_err(SpartanBridgeError::BellpepperError)?;

            let x_cols_num =
                AllocatedNum::alloc(cs.namespace(|| format!("{label}_me_{me_idx}_X_cols")), || Ok(x_cols_val))
                    .map_err(SpartanBridgeError::BellpepperError)?;
            cs.enforce(
                || format!("{label}_me_{me_idx}_X_cols_is_const"),
                |lc| lc + x_cols_num.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + (x_cols_val, CS::one()),
            );
            sponge.absorb(cs, x_cols_num).map_err(SpartanBridgeError::BellpepperError)?;

            if me_var.X.len() != x_rows || (!me_var.X.is_empty() && me_var.X[0].len() != x_cols) {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "{label}: ME[{me_idx}] X vars shape mismatch"
                )));
            }
            for r in 0..x_rows {
                for c in 0..x_cols {
                    let v = CircuitF::from(me_val.X[(r, c)].as_canonical_u64());
                    let num =
                        AllocatedNum::alloc(cs.namespace(|| format!("{label}_me_{me_idx}_X_{r}_{c}")), || Ok(v))
                            .map_err(SpartanBridgeError::BellpepperError)?;
                    cs.enforce(
                        || format!("{label}_me_{me_idx}_X_{r}_{c}_eq"),
                        |lc| lc + num.get_variable(),
                        |lc| lc + CS::one(),
                        |lc| lc + me_var.X[r][c],
                    );
                    sponge.absorb(cs, num).map_err(SpartanBridgeError::BellpepperError)?;
                }
            }

            let r_len_val = CircuitF::from(me_val.r.len() as u64);
            let r_len = AllocatedNum::alloc(cs.namespace(|| format!("{label}_me_{me_idx}_r_len")), || Ok(r_len_val))
                .map_err(SpartanBridgeError::BellpepperError)?;
            cs.enforce(
                || format!("{label}_me_{me_idx}_r_len_is_const"),
                |lc| lc + r_len.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + (r_len_val, CS::one()),
            );
            sponge.absorb(cs, r_len).map_err(SpartanBridgeError::BellpepperError)?;

            if me_val.r.len() != me_var.r.len() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "{label}: ME[{me_idx}] r length mismatch (vals={}, vars={})",
                    me_val.r.len(),
                    me_var.r.len()
                )));
            }
            for (i, limb) in me_val.r.iter().enumerate() {
                let coeffs = limb.as_coeffs();

                let c0_val = CircuitF::from(coeffs[0].as_canonical_u64());
                let c0 = AllocatedNum::alloc(cs.namespace(|| format!("{label}_me_{me_idx}_r_{i}_c0")), || Ok(c0_val))
                    .map_err(SpartanBridgeError::BellpepperError)?;
                cs.enforce(
                    || format!("{label}_me_{me_idx}_r_{i}_c0_eq"),
                    |lc| lc + c0.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + me_var.r[i].c0,
                );
                sponge.absorb(cs, c0).map_err(SpartanBridgeError::BellpepperError)?;

                let c1_val = CircuitF::from(coeffs[1].as_canonical_u64());
                let c1 = AllocatedNum::alloc(cs.namespace(|| format!("{label}_me_{me_idx}_r_{i}_c1")), || Ok(c1_val))
                    .map_err(SpartanBridgeError::BellpepperError)?;
                cs.enforce(
                    || format!("{label}_me_{me_idx}_r_{i}_c1_eq"),
                    |lc| lc + c1.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + me_var.r[i].c1,
                );
                sponge.absorb(cs, c1).map_err(SpartanBridgeError::BellpepperError)?;
            }

            let y_len_val = CircuitF::from(me_val.y.len() as u64);
            let y_len = AllocatedNum::alloc(cs.namespace(|| format!("{label}_me_{me_idx}_y_len")), || Ok(y_len_val))
                .map_err(SpartanBridgeError::BellpepperError)?;
            cs.enforce(
                || format!("{label}_me_{me_idx}_y_len_is_const"),
                |lc| lc + y_len.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + (y_len_val, CS::one()),
            );
            sponge.absorb(cs, y_len).map_err(SpartanBridgeError::BellpepperError)?;

            if me_val.y.len() != me_var.y.len() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "{label}: ME[{me_idx}] y length mismatch (vals={}, vars={})",
                    me_val.y.len(),
                    me_var.y.len()
                )));
            }

            for (j, (row_val, row_var)) in me_val.y.iter().zip(me_var.y.iter()).enumerate() {
                let row_len_val = CircuitF::from(row_val.len() as u64);
                let row_len = AllocatedNum::alloc(cs.namespace(|| format!("{label}_me_{me_idx}_y_{j}_len")), || Ok(row_len_val))
                    .map_err(SpartanBridgeError::BellpepperError)?;
                cs.enforce(
                    || format!("{label}_me_{me_idx}_y_{j}_len_is_const"),
                    |lc| lc + row_len.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + (row_len_val, CS::one()),
                );
                sponge.absorb(cs, row_len).map_err(SpartanBridgeError::BellpepperError)?;

                let limit = core::cmp::min(row_val.len(), row_var.len());
                for rho in 0..limit {
                    let coeffs = row_val[rho].as_coeffs();

                    let c0_val = CircuitF::from(coeffs[0].as_canonical_u64());
                    let c0 = AllocatedNum::alloc(
                        cs.namespace(|| format!("{label}_me_{me_idx}_y_{j}_{rho}_c0")),
                        || Ok(c0_val),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;
                    cs.enforce(
                        || format!("{label}_me_{me_idx}_y_{j}_{rho}_c0_eq"),
                        |lc| lc + c0.get_variable(),
                        |lc| lc + CS::one(),
                        |lc| lc + row_var[rho].c0,
                    );
                    sponge.absorb(cs, c0).map_err(SpartanBridgeError::BellpepperError)?;

                    let c1_val = CircuitF::from(coeffs[1].as_canonical_u64());
                    let c1 = AllocatedNum::alloc(
                        cs.namespace(|| format!("{label}_me_{me_idx}_y_{j}_{rho}_c1")),
                        || Ok(c1_val),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;
                    cs.enforce(
                        || format!("{label}_me_{me_idx}_y_{j}_{rho}_c1_eq"),
                        |lc| lc + c1.get_variable(),
                        |lc| lc + CS::one(),
                        |lc| lc + row_var[rho].c1,
                    );
                    sponge.absorb(cs, c1).map_err(SpartanBridgeError::BellpepperError)?;
                }

                if row_val.len() != row_var.len() {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "{label}: ME[{me_idx}] y[{j}] length mismatch (vals={}, vars={})",
                        row_val.len(),
                        row_var.len()
                    )));
                }
            }

            // Canonical y_scalars: base-b recomposition from the first D digits of y[j].
            //
            // IMPORTANT: in shared-bus mode, `me_val.y_scalars` may include extra scalars appended
            // after the core `t=s.t()` entries (bus openings). Those must not affect the public
            // accumulator digest, so we derive scalars from `y` directly.
            let base_circ = CircuitF::from(self.base_b as u64);
            let bK = neo_math::K::from(NeoF::from_u64(self.base_b as u64));
            for (j, yj) in me_val.y.iter().enumerate() {
                if yj.len() < neo_math::D {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "{label}: ME[{me_idx}] y[{j}] too short for y_scalars (have {}, need >= {})",
                        yj.len(),
                        neo_math::D
                    )));
                }
                if me_var.y[j].len() < neo_math::D {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "{label}: ME[{me_idx}] y[{j}] vars too short for y_scalars (have {}, need >= {})",
                        me_var.y[j].len(),
                        neo_math::D
                    )));
                }

                let mut acc_k = neo_math::K::ZERO;
                let mut pw = neo_math::K::ONE;
                for rho in 0..neo_math::D {
                    acc_k += pw * yj[rho];
                    pw *= bK;
                }

                let y_scalar_var = helpers::alloc_k_from_neo(
                    cs,
                    acc_k,
                    &format!("{label}_me_{me_idx}_y_scalar_{j}"),
                )?;

                // Enforce recomposition from digits (first D entries).
                cs.enforce(
                    || format!("{label}_me_{me_idx}_y_scalar_{j}_c0"),
                    |lc| {
                        let mut res = lc;
                        let mut pow = CircuitF::from(1u64);
                        for rho in 0..neo_math::D {
                            res = res + (pow, me_var.y[j][rho].c0);
                            pow *= base_circ;
                        }
                        res
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + y_scalar_var.c0,
                );
                cs.enforce(
                    || format!("{label}_me_{me_idx}_y_scalar_{j}_c1"),
                    |lc| {
                        let mut res = lc;
                        let mut pow = CircuitF::from(1u64);
                        for rho in 0..neo_math::D {
                            res = res + (pow, me_var.y[j][rho].c1);
                            pow *= base_circ;
                        }
                        res
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + y_scalar_var.c1,
                );

                let coeffs = acc_k.as_coeffs();
                let c0_val = CircuitF::from(coeffs[0].as_canonical_u64());
                let c0 = AllocatedNum::alloc(
                    cs.namespace(|| format!("{label}_me_{me_idx}_y_scalar_{j}_c0_abs")),
                    || Ok(c0_val),
                )
                .map_err(SpartanBridgeError::BellpepperError)?;
                cs.enforce(
                    || format!("{label}_me_{me_idx}_y_scalar_{j}_c0_eq"),
                    |lc| lc + c0.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + y_scalar_var.c0,
                );
                sponge.absorb(cs, c0).map_err(SpartanBridgeError::BellpepperError)?;

                let c1_val = CircuitF::from(coeffs[1].as_canonical_u64());
                let c1 = AllocatedNum::alloc(
                    cs.namespace(|| format!("{label}_me_{me_idx}_y_scalar_{j}_c1_abs")),
                    || Ok(c1_val),
                )
                .map_err(SpartanBridgeError::BellpepperError)?;
                cs.enforce(
                    || format!("{label}_me_{me_idx}_y_scalar_{j}_c1_eq"),
                    |lc| lc + c1.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + y_scalar_var.c1,
                );
                sponge.absorb(cs, c1).map_err(SpartanBridgeError::BellpepperError)?;
            }
        }

        let digest = sponge
            .digest32(cs, &format!("{label}_digest32"))
            .map_err(SpartanBridgeError::BellpepperError)?;

        for i in 0..4 {
            cs.enforce(
                || format!("{label}_digest_limb_{i}_matches_statement"),
                |lc| lc + digest[i].get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + statement_io[statement_offset + i],
            );
        }

        Ok(())
    }

    fn validate_witness_structure(&self, witness: &FoldRunWitness) -> Result<()> {
        let steps_len = witness.fold_run.steps.len();
        let step_count = usize::try_from(self.instance.statement.step_count).map_err(|_| {
            SpartanBridgeError::InvalidInput(format!(
                "statement.step_count does not fit usize: {}",
                self.instance.statement.step_count
            ))
        })?;
        if steps_len != step_count {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "statement.step_count mismatch: statement={}, witness={}",
                step_count, steps_len
            )));
        }
        if witness.steps_public.len() != steps_len {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "steps_public length mismatch: instances={}, steps={}",
                witness.steps_public.len(),
                steps_len
            )));
        }

        let mem_enabled = self.instance.statement.mem_enabled;
        let output_binding_enabled = self.instance.statement.output_binding_enabled;

        match (
            output_binding_enabled,
            witness.fold_run.output_proof.as_ref(),
            witness.output_binding.as_ref(),
        ) {
            (true, None, _) => {
                return Err(SpartanBridgeError::InvalidInput(
                    "output binding enabled but fold_run.output_proof is None".into(),
                ))
            }
            (true, _, None) => {
                return Err(SpartanBridgeError::InvalidInput(
                    "output binding enabled but witness.output_binding is None".into(),
                ))
            }
            (false, Some(_), _) => {
                return Err(SpartanBridgeError::InvalidInput(
                    "fold_run.output_proof is Some but statement.output_binding_enabled=false".into(),
                ))
            }
            (false, _, Some(_)) => {
                return Err(SpartanBridgeError::InvalidInput(
                    "witness.output_binding is Some but statement.output_binding_enabled=false".into(),
                ))
            }
            _ => {}
        }

        if output_binding_enabled {
            if !mem_enabled {
                return Err(SpartanBridgeError::InvalidInput(
                    "output binding requires mem_enabled=true".into(),
                ));
            }
            if steps_len == 0 {
                return Err(SpartanBridgeError::InvalidInput(
                    "output binding requires at least one step".into(),
                ));
            }
            let cfg = witness
                .output_binding
                .as_ref()
                .expect("checked above");
            cfg.program_io
                .validate(cfg.num_bits)
                .map_err(|e| SpartanBridgeError::InvalidInput(format!("invalid ProgramIO for output binding: {e:?}")))?;

            let last_step_public = witness
                .steps_public
                .last()
                .ok_or_else(|| SpartanBridgeError::InvalidInput("steps_public is empty".into()))?;
            if cfg.mem_idx >= last_step_public.mem_insts.len() {
                return Err(SpartanBridgeError::InvalidInput(
                    "output binding mem_idx out of range for last step".into(),
                ));
            }
            let mem_inst = last_step_public
                .mem_insts
                .get(cfg.mem_idx)
                .ok_or_else(|| SpartanBridgeError::InvalidInput("output binding mem_idx out of range".into()))?;
            let expected_k = 1usize
                .checked_shl(cfg.num_bits as u32)
                .ok_or_else(|| SpartanBridgeError::InvalidInput("output binding: 2^num_bits overflow".into()))?;
            if mem_inst.k != expected_k {
                return Err(SpartanBridgeError::InvalidInput(
                    "output binding: cfg.num_bits inconsistent with mem_inst.k".into(),
                ));
            }
            let ell_addr = mem_inst.d * mem_inst.ell;
            if ell_addr != cfg.num_bits {
                return Err(SpartanBridgeError::InvalidInput(
                    "output binding: cfg.num_bits inconsistent with twist_layout.ell_addr".into(),
                ));
            }

            let ob = witness
                .fold_run
                .output_proof
                .as_ref()
                .expect("checked above");
            if ob.output_sc.round_polys.len() != cfg.num_bits {
                return Err(SpartanBridgeError::InvalidInput(
                    "output binding: output_sc.round_polys.len() mismatch".into(),
                ));
            }
            for (round_idx, coeffs) in ob.output_sc.round_polys.iter().enumerate() {
                if coeffs.len() != 4 {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "output binding: output_sc round {round_idx} wrong degree (len={}, expected 4)",
                        coeffs.len()
                    )));
                }
            }

            let last_step_proof = witness
                .fold_run
                .steps
                .last()
                .ok_or_else(|| SpartanBridgeError::InvalidInput("fold_run.steps is empty".into()))?;
            let inc_idx = last_step_proof
                .batched_time
                .labels
                .len()
                .checked_sub(1)
                .ok_or_else(|| SpartanBridgeError::InvalidInput("output binding: missing inc_total claim".into()))?;
            if last_step_proof
                .batched_time
                .labels
                .get(inc_idx)
                .copied()
                != Some(neo_fold::output_binding::OB_INC_TOTAL_LABEL)
            {
                return Err(SpartanBridgeError::InvalidInput(
                    "output binding: inc_total claim not last".into(),
                ));
            }
        }

        for step_idx in 0..steps_len {
            let step_public = witness
                .steps_public
                .get(step_idx)
                .ok_or_else(|| SpartanBridgeError::InvalidInput(format!("missing steps_public[{step_idx}]")))?;
            let step_proof = &witness.fold_run.steps[step_idx];

            if !mem_enabled {
                if !step_public.lut_insts.is_empty() || !step_public.mem_insts.is_empty() {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "step {step_idx}: mem_enabled=false requires empty StepInstanceBundle lut/mem vectors"
                    )));
                }
                if !step_proof.mem.cpu_me_claims_val.is_empty()
                    || !step_proof.mem.proofs.is_empty()
                    || !step_proof.mem.shout_addr_pre.claimed_sums.is_empty()
                    || !step_proof.mem.shout_addr_pre.round_polys.is_empty()
                    || !step_proof.mem.shout_addr_pre.r_addr.is_empty()
                {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "step {step_idx}: mem_enabled=false requires empty mem sidecars"
                    )));
                }
                if step_proof.val_fold.is_some() {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "step {step_idx}: mem_enabled=false forbids val lane proofs"
                    )));
                }
            } else {
                let n_lut = step_public.lut_insts.len();
                let n_mem = step_public.mem_insts.len();
                if step_proof.mem.proofs.len() != n_lut + n_mem {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "step {step_idx}: mem.proofs.len() mismatch (got {}, expected n_lut+n_mem={})",
                        step_proof.mem.proofs.len(),
                        n_lut + n_mem
                    )));
                }
                if n_mem == 0 {
                    if !step_proof.mem.cpu_me_claims_val.is_empty() || step_proof.val_fold.is_some() {
                        return Err(SpartanBridgeError::InvalidInput(format!(
                            "step {step_idx}: unexpected val-lane artifacts with no mem instances"
                        )));
                    }
                } else {
                    let expected_cpu_me = 1 + usize::from(step_idx > 0);
                    if step_proof.mem.cpu_me_claims_val.len() != expected_cpu_me {
                        return Err(SpartanBridgeError::InvalidInput(format!(
                            "step {step_idx}: cpu_me_claims_val.len() mismatch (got {}, expected {expected_cpu_me})",
                            step_proof.mem.cpu_me_claims_val.len()
                        )));
                    }
                    if step_proof.val_fold.is_none() {
                        return Err(SpartanBridgeError::InvalidInput(format!(
                            "step {step_idx}: missing val_fold proof (mem instances present)"
                        )));
                    }
                }
            }

            let expected_inputs = if step_idx == 0 {
                witness.initial_accumulator.len()
            } else {
                witness.fold_run.steps[step_idx - 1].fold.dec_children.len()
            };
            let expected_ccs_out = expected_inputs
                .checked_add(1)
                .ok_or_else(|| SpartanBridgeError::InvalidInput("ccs_out length overflow".into()))?;
            let actual_ccs_out = step_proof.fold.ccs_out.len();
            if actual_ccs_out != expected_ccs_out {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "step {step_idx}: ccs_out.len()={actual_ccs_out}, expected {expected_ccs_out} (= 1 + inputs {expected_inputs})",
                )));
            }

            // Phase 1 requires Pattern-B only for RLC transcript binding.
            let header_digest = &step_proof.fold.ccs_proof.header_digest;
            if header_digest.len() != 32 {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "step {step_idx}: ccs_proof.header_digest must be 32 bytes, got {}",
                    header_digest.len()
                )));
            }
            let mut header_digest_arr = [0u8; 32];
            header_digest_arr.copy_from_slice(header_digest.as_slice());

            for (me_idx, me) in step_proof.fold.ccs_out.iter().enumerate() {
                if me.fold_digest != header_digest_arr {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "step {step_idx}: ccs_out[{me_idx}].fold_digest must equal ccs_proof.header_digest in Phase 1"
                    )));
                }
                if !me.c_step_coords.is_empty() || me.u_offset != 0 || me.u_len != 0 {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "step {step_idx}: ccs_out[{me_idx}] requires Pattern-B (empty c_step_coords, u_offset=0, u_len=0) in Phase 1"
                    )));
                }
                for (j, yj) in me.y.iter().enumerate() {
                    if yj.len() < neo_math::D {
                        return Err(SpartanBridgeError::InvalidInput(format!(
                            "step {step_idx}: ccs_out[{me_idx}] y[{j}] too short (have {}, need >= {})",
                            yj.len(),
                            neo_math::D
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    /// Allocate 32 digest bytes as circuit variables and constrain that they encode the provided
    /// digest limbs as canonical Goldilocks u64s (little-endian), matching `neo_transcript::digest32`.
    fn alloc_digest32_bytes_from_limbs<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        digest_limbs: &[AllocatedNum<CircuitF>; 4],
        hint_bytes: Option<&[u8]>,
        label: &str,
    ) -> Result<Vec<AllocatedNum<CircuitF>>> {
        use ff::Field;

        if let Some(h) = hint_bytes {
            if h.len() != 32 {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "{label}: digest hint must have len 32, got {}",
                    h.len()
                )));
            }
        }

        let mut bytes: Vec<AllocatedNum<CircuitF>> = Vec::with_capacity(32);

        // Allocate bytes with 8-bit range constraints (bit decomposition).
        for byte_idx in 0..32 {
            let hint = hint_bytes.map(|h| h[byte_idx]).unwrap_or(0u8);
            let byte_val = CircuitF::from(hint as u64);
            let byte = AllocatedNum::alloc(cs.namespace(|| format!("{label}_byte_{byte_idx}")), || Ok(byte_val))?;

            let mut bits: Vec<AllocatedNum<CircuitF>> = Vec::with_capacity(8);
            for bit_idx in 0..8 {
                let bit_hint = ((hint >> bit_idx) & 1) as u64;
                let bit_val = CircuitF::from(bit_hint);
                let bit =
                    AllocatedNum::alloc(cs.namespace(|| format!("{label}_byte_{byte_idx}_bit_{bit_idx}")), || Ok(bit_val))?;
                // bit is boolean: bit*(bit-1)=0
                cs.enforce(
                    || format!("{label}_byte_{byte_idx}_bit_{bit_idx}_bool"),
                    |lc| lc + bit.get_variable(),
                    |lc| lc + bit.get_variable() - CS::one(),
                    |lc| lc,
                );
                bits.push(bit);
            }

            // Enforce: byte == Σ 2^i * bit_i
            cs.enforce(
                || format!("{label}_byte_{byte_idx}_recompose"),
                |lc| {
                    let mut acc = lc + byte.get_variable();
                    for (bit_idx, bit) in bits.iter().enumerate() {
                        let coeff_u64 = 1u64 << bit_idx;
                        let neg_coeff = CircuitF::from(0u64) - CircuitF::from(coeff_u64);
                        acc = acc + (neg_coeff, bit.get_variable());
                    }
                    acc
                },
                |lc| lc + CS::one(),
                |lc| lc,
            );

            bytes.push(byte);
        }

        // Constrain each u64 limb to match the digest limb, and enforce canonical u64 encoding (< p).
        // For Goldilocks p = 2^64 - 2^32 + 1, the only non-canonical u64 representations are those
        // with hi32 = 0xFFFF_FFFF and lo32 != 0.
        for limb_idx in 0..4 {
            let limb_bytes = &bytes[limb_idx * 8..(limb_idx + 1) * 8];
            let digest_limb = &digest_limbs[limb_idx];

            // Enforce packed u64 == digest_limb (as a field element).
            cs.enforce(
                || format!("{label}_limb_{limb_idx}_pack_eq"),
                |lc| {
                    let mut acc = lc;
                    let mut pow = CircuitF::from(1u64);
                    for b in limb_bytes {
                        acc = acc + (pow, b.get_variable());
                        pow *= CircuitF::from(256u64);
                    }
                    acc
                },
                |lc| lc + CS::one(),
                |lc| lc + digest_limb.get_variable(),
            );

            // Compute hint hi32 for (optional) inverse witness assignment.
            let (hi32_hint_u64, lo32_hint_u64) = if let Some(h) = hint_bytes {
                let limb_off = limb_idx * 8;
                let lo = u32::from_le_bytes([h[limb_off], h[limb_off + 1], h[limb_off + 2], h[limb_off + 3]]) as u64;
                let hi = u32::from_le_bytes([h[limb_off + 4], h[limb_off + 5], h[limb_off + 6], h[limb_off + 7]]) as u64;
                (hi, lo)
            } else {
                (0u64, 0u64)
            };

            let is_hi_ff_hint = if hi32_hint_u64 == 0xFFFF_FFFF { 1u64 } else { 0u64 };
            let is_hi_ff =
                AllocatedNum::alloc(cs.namespace(|| format!("{label}_limb_{limb_idx}_is_hi_ff")), || {
                    Ok(CircuitF::from(is_hi_ff_hint))
                })?;
            // boolean
            cs.enforce(
                || format!("{label}_limb_{limb_idx}_is_hi_ff_bool"),
                |lc| lc + is_hi_ff.get_variable(),
                |lc| lc + is_hi_ff.get_variable() - CS::one(),
                |lc| lc,
            );

            let hi32_lc = |lc: bellpepper_core::LinearCombination<CircuitF>| {
                let mut acc = lc;
                let mut pow = CircuitF::from(1u64);
                for b in limb_bytes.iter().skip(4) {
                    acc = acc + (pow, b.get_variable());
                    pow *= CircuitF::from(256u64);
                }
                acc
            };
            let lo32_lc = |lc: bellpepper_core::LinearCombination<CircuitF>| {
                let mut acc = lc;
                let mut pow = CircuitF::from(1u64);
                for b in limb_bytes.iter().take(4) {
                    acc = acc + (pow, b.get_variable());
                    pow *= CircuitF::from(256u64);
                }
                acc
            };

            // Inverse witness for diff = (hi32 - 0xFFFF_FFFF), with 0 when diff == 0.
            let diff_hint = CircuitF::from(hi32_hint_u64) - CircuitF::from(0xFFFF_FFFFu64);
            let inv_hint: CircuitF = Option::from(diff_hint.invert()).unwrap_or(CircuitF::from(0u64));
            let inv = AllocatedNum::alloc(cs.namespace(|| format!("{label}_limb_{limb_idx}_hi_diff_inv")), || Ok(inv_hint))?;

            // Enforce: (hi32 - 0xFFFF_FFFF) * inv = 1 - is_hi_ff
            let minus_one = CircuitF::from(0u64) - CircuitF::from(1u64);
            let neg_const_ff = CircuitF::from(0u64) - CircuitF::from(0xFFFF_FFFFu64);
            cs.enforce(
                || format!("{label}_limb_{limb_idx}_is_hi_ff_inverse_gate"),
                |lc| {
                    let mut acc = hi32_lc(lc);
                    acc = acc + (neg_const_ff, CS::one());
                    acc
                },
                |lc| lc + inv.get_variable(),
                |lc| {
                    let mut acc = lc + CS::one();
                    acc = acc + (minus_one, is_hi_ff.get_variable());
                    acc
                },
            );

            // Enforce: (hi32 - 0xFFFF_FFFF) * is_hi_ff = 0
            cs.enforce(
                || format!("{label}_limb_{limb_idx}_is_hi_ff_zero"),
                |lc| {
                    let mut acc = hi32_lc(lc);
                    acc = acc + (neg_const_ff, CS::one());
                    acc
                },
                |lc| lc + is_hi_ff.get_variable(),
                |lc| lc,
            );

            // Canonical u64 check: if hi32 == 0xFFFF_FFFF then lo32 == 0.
            cs.enforce(
                || format!("{label}_limb_{limb_idx}_canonical_u64"),
                |lc| lo32_lc(lc),
                |lc| lc + is_hi_ff.get_variable(),
                |lc| lc,
            );

            // Extra safety: if hint had hi32=0xFFFF_FFFF and lo32 != 0, reject early.
            if hi32_hint_u64 == 0xFFFF_FFFF && lo32_hint_u64 != 0 {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "{label}: limb {limb_idx} digest bytes are not canonical Goldilocks u64 (hi32=0xFFFF_FFFF, lo32={lo32_hint_u64})",
                )));
            }
        }

        Ok(bytes)
    }

    /// Allocate and constrain public inputs
    ///
    /// Currently we expose:
    /// - the `SpartanShardStatement` (see `crate::statement`), and
    ///
    /// Everything else (fold run, accumulators, transcript) is private witness data.
    fn allocate_public_inputs<CS: ConstraintSystem<CircuitF>>(&self, cs: &mut CS) -> Result<PublicInputsVars> {
        let mut statement_io = Vec::with_capacity(self.instance.statement.public_io().len());
        for (i, value) in self.instance.statement.public_io().into_iter().enumerate() {
            let v = cs.alloc_input(|| format!("statement_io_{i}"), || Ok(value))?;
            statement_io.push(v);
        }
        Ok(PublicInputsVars { statement_io })
    }

    fn absorb_step_memory_empty<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        tr: &mut Poseidon2TranscriptVar,
        step_idx: usize,
    ) -> Result<()> {
        let ctx = format!("step_{step_idx}_absorb_step_memory");
        tr.append_message(cs, b"step/absorb_memory_start", &[], &ctx)
            .map_err(SpartanBridgeError::BellpepperError)?;
        tr.append_message(cs, b"step/lut_count", &0u64.to_le_bytes(), &ctx)
            .map_err(SpartanBridgeError::BellpepperError)?;
        tr.append_message(cs, b"step/mem_count", &0u64.to_le_bytes(), &ctx)
            .map_err(SpartanBridgeError::BellpepperError)?;
        tr.append_message(cs, b"step/absorb_memory_done", &[], &ctx)
            .map_err(SpartanBridgeError::BellpepperError)?;
        Ok(())
    }

    fn alloc_u64_le_bytes_witness<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        v: u64,
        label: &str,
    ) -> Result<[AllocatedNum<CircuitF>; 8]> {
        let bytes = v.to_le_bytes();
        let mut out = Vec::with_capacity(8);
        for (i, b) in bytes.iter().enumerate() {
            let num = AllocatedNum::alloc(cs.namespace(|| format!("{label}_byte_{i}")), || Ok(CircuitF::from(*b as u64)))
                .map_err(SpartanBridgeError::BellpepperError)?;
            out.push(num);
        }
        out.try_into()
            .map_err(|_| SpartanBridgeError::InvalidInput(format!("{label}: u64 byte allocation failed")))
    }

    fn append_message_u64_le_bytes_multi<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        tr: &mut Poseidon2TranscriptVar,
        mut tr2: Option<&mut Poseidon2TranscriptVar>,
        mut tr3: Option<&mut Poseidon2TranscriptVar>,
        label: &'static [u8],
        v: u64,
        ctx: &str,
    ) -> Result<()> {
        let bytes = self.alloc_u64_le_bytes_witness(cs, v, ctx)?;
        tr.append_message_bytes_allocated(cs, label, bytes.as_slice(), ctx)
            .map_err(SpartanBridgeError::BellpepperError)?;
        if let Some(tr2) = tr2.as_mut() {
            tr2.append_message_bytes_allocated(cs, label, bytes.as_slice(), ctx)
                .map_err(SpartanBridgeError::BellpepperError)?;
        }
        if let Some(tr3) = tr3.as_mut() {
            tr3.append_message_bytes_allocated(cs, label, bytes.as_slice(), ctx)
                .map_err(SpartanBridgeError::BellpepperError)?;
        }
        Ok(())
    }

    fn digest_fields_bytes32<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        label: &'static [u8],
        fs: &[NeoF],
        ctx: &str,
    ) -> Result<Vec<AllocatedNum<CircuitF>>> {
        // Native: `neo_fold::memory_sidecar::transcript::digest_fields`.
        let hint = neo_fold::memory_sidecar::transcript::digest_fields(label, fs);

        let mut h = Poseidon2TranscriptVar::new(cs, b"memory/public_digest", &format!("{ctx}_digest_fields"))
            .map_err(SpartanBridgeError::BellpepperError)?;
        h.append_message(cs, b"digest/label", label, &format!("{ctx}_digest_label"))
            .map_err(SpartanBridgeError::BellpepperError)?;
        h.append_message(
            cs,
            b"digest/len",
            &(fs.len() as u64).to_le_bytes(),
            &format!("{ctx}_digest_len"),
        )
        .map_err(SpartanBridgeError::BellpepperError)?;

        let mut fs_alloc = Vec::with_capacity(fs.len());
        for (i, f) in fs.iter().enumerate() {
            use p3_field::PrimeField64;
            let v = CircuitF::from(f.as_canonical_u64());
            let num = AllocatedNum::alloc(cs.namespace(|| format!("{ctx}_digest_field_{i}")), || Ok(v))
                .map_err(SpartanBridgeError::BellpepperError)?;
            fs_alloc.push(num);
        }
        h.append_fields_allocated(cs, b"digest/fields", &fs_alloc, &format!("{ctx}_digest_fields"))
            .map_err(SpartanBridgeError::BellpepperError)?;

        let limbs = h.digest32(cs, &format!("{ctx}_digest32")).map_err(SpartanBridgeError::BellpepperError)?;
        self.alloc_digest32_bytes_from_limbs(cs, &limbs, Some(&hint), ctx)
    }

    fn bind_shout_table_spec<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        tr: &mut Poseidon2TranscriptVar,
        mut tr2: Option<&mut Poseidon2TranscriptVar>,
        mut tr3: Option<&mut Poseidon2TranscriptVar>,
        spec: &Option<neo_memory::witness::LutTableSpec>,
        ctx: &str,
    ) -> Result<()> {
        let Some(spec) = spec else {
            return Ok(());
        };

        tr.append_message(cs, b"shout/table_spec/tag", &[1u8], ctx)
            .map_err(SpartanBridgeError::BellpepperError)?;
        if let Some(tr2) = tr2.as_mut() {
            tr2.append_message(cs, b"shout/table_spec/tag", &[1u8], ctx)
                .map_err(SpartanBridgeError::BellpepperError)?;
        }
        if let Some(tr3) = tr3.as_mut() {
            tr3.append_message(cs, b"shout/table_spec/tag", &[1u8], ctx)
                .map_err(SpartanBridgeError::BellpepperError)?;
        }

        match spec {
            neo_memory::witness::LutTableSpec::RiscvOpcode { opcode, xlen } => {
                tr.append_message(cs, b"shout/table_spec/riscv/tag", &[1u8], ctx)
                    .map_err(SpartanBridgeError::BellpepperError)?;
                if let Some(tr2) = tr2.as_mut() {
                    tr2.append_message(cs, b"shout/table_spec/riscv/tag", &[1u8], ctx)
                        .map_err(SpartanBridgeError::BellpepperError)?;
                }
                if let Some(tr3) = tr3.as_mut() {
                    tr3.append_message(cs, b"shout/table_spec/riscv/tag", &[1u8], ctx)
                        .map_err(SpartanBridgeError::BellpepperError)?;
                }

                // Stable numeric encoding: align with `neo_memory::riscv::lookups::tables::RiscvShoutTables::opcode_to_id`.
                let opcode_id: u64 = match opcode {
                    neo_memory::riscv::lookups::RiscvOpcode::And => 0,
                    neo_memory::riscv::lookups::RiscvOpcode::Xor => 1,
                    neo_memory::riscv::lookups::RiscvOpcode::Or => 2,
                    neo_memory::riscv::lookups::RiscvOpcode::Add => 3,
                    neo_memory::riscv::lookups::RiscvOpcode::Sub => 4,
                    neo_memory::riscv::lookups::RiscvOpcode::Slt => 5,
                    neo_memory::riscv::lookups::RiscvOpcode::Sltu => 6,
                    neo_memory::riscv::lookups::RiscvOpcode::Sll => 7,
                    neo_memory::riscv::lookups::RiscvOpcode::Srl => 8,
                    neo_memory::riscv::lookups::RiscvOpcode::Sra => 9,
                    neo_memory::riscv::lookups::RiscvOpcode::Eq => 10,
                    neo_memory::riscv::lookups::RiscvOpcode::Neq => 11,
                    neo_memory::riscv::lookups::RiscvOpcode::Mul => 12,
                    neo_memory::riscv::lookups::RiscvOpcode::Mulh => 13,
                    neo_memory::riscv::lookups::RiscvOpcode::Mulhu => 14,
                    neo_memory::riscv::lookups::RiscvOpcode::Mulhsu => 15,
                    neo_memory::riscv::lookups::RiscvOpcode::Div => 16,
                    neo_memory::riscv::lookups::RiscvOpcode::Divu => 17,
                    neo_memory::riscv::lookups::RiscvOpcode::Rem => 18,
                    neo_memory::riscv::lookups::RiscvOpcode::Remu => 19,
                    neo_memory::riscv::lookups::RiscvOpcode::Addw => 20,
                    neo_memory::riscv::lookups::RiscvOpcode::Subw => 21,
                    neo_memory::riscv::lookups::RiscvOpcode::Sllw => 22,
                    neo_memory::riscv::lookups::RiscvOpcode::Srlw => 23,
                    neo_memory::riscv::lookups::RiscvOpcode::Sraw => 24,
                    neo_memory::riscv::lookups::RiscvOpcode::Mulw => 25,
                    neo_memory::riscv::lookups::RiscvOpcode::Divw => 26,
                    neo_memory::riscv::lookups::RiscvOpcode::Divuw => 27,
                    neo_memory::riscv::lookups::RiscvOpcode::Remw => 28,
                    neo_memory::riscv::lookups::RiscvOpcode::Remuw => 29,
                    neo_memory::riscv::lookups::RiscvOpcode::Andn => 30,
                };

                self.append_message_u64_le_bytes_multi(
                    cs,
                    tr,
                    tr2.as_deref_mut(),
                    tr3.as_deref_mut(),
                    b"shout/table_spec/riscv/opcode_id",
                    opcode_id,
                    &format!("{ctx}_opcode_id"),
                )?;
                self.append_message_u64_le_bytes_multi(
                    cs,
                    tr,
                    tr2.as_deref_mut(),
                    tr3.as_deref_mut(),
                    b"shout/table_spec/riscv/xlen",
                    *xlen as u64,
                    &format!("{ctx}_xlen"),
                )?;
            }
        }

        Ok(())
    }

    fn absorb_step_memory<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        tr: &mut Poseidon2TranscriptVar,
        mut tr2: Option<&mut Poseidon2TranscriptVar>,
        mut tr3: Option<&mut Poseidon2TranscriptVar>,
        step_idx: usize,
        step_public: &neo_memory::witness::StepInstanceBundle<neo_ajtai::Commitment, NeoF, neo_math::K>,
    ) -> Result<()> {
        let ctx = format!("step_{step_idx}_absorb_step_memory");

        tr.append_message(cs, b"step/absorb_memory_start", &[], &ctx)
            .map_err(SpartanBridgeError::BellpepperError)?;
        if let Some(tr2) = tr2.as_mut() {
            tr2.append_message(cs, b"step/absorb_memory_start", &[], &ctx)
                .map_err(SpartanBridgeError::BellpepperError)?;
        }
        if let Some(tr3) = tr3.as_mut() {
            tr3.append_message(cs, b"step/absorb_memory_start", &[], &ctx)
                .map_err(SpartanBridgeError::BellpepperError)?;
        }

        // LUTs
        let lut_count_bytes = &(step_public.lut_insts.len() as u64).to_le_bytes();
        tr.append_message(cs, b"step/lut_count", lut_count_bytes, &format!("{ctx}_lut_count"))
            .map_err(SpartanBridgeError::BellpepperError)?;
        if let Some(tr2) = tr2.as_mut() {
            tr2.append_message(cs, b"step/lut_count", lut_count_bytes, &format!("{ctx}_lut_count"))
                .map_err(SpartanBridgeError::BellpepperError)?;
        }
        if let Some(tr3) = tr3.as_mut() {
            tr3.append_message(cs, b"step/lut_count", lut_count_bytes, &format!("{ctx}_lut_count"))
                .map_err(SpartanBridgeError::BellpepperError)?;
        }

        for (lut_idx, inst) in step_public.lut_insts.iter().enumerate() {
            let lut_idx_bytes = &(lut_idx as u64).to_le_bytes();
            tr.append_message(cs, b"step/lut_idx", lut_idx_bytes, &format!("{ctx}_lut_{lut_idx}_idx"))
                .map_err(SpartanBridgeError::BellpepperError)?;
            if let Some(tr2) = tr2.as_mut() {
                tr2.append_message(cs, b"step/lut_idx", lut_idx_bytes, &format!("{ctx}_lut_{lut_idx}_idx"))
                    .map_err(SpartanBridgeError::BellpepperError)?;
            }
            if let Some(tr3) = tr3.as_mut() {
                tr3.append_message(cs, b"step/lut_idx", lut_idx_bytes, &format!("{ctx}_lut_{lut_idx}_idx"))
                    .map_err(SpartanBridgeError::BellpepperError)?;
            }

            self.append_message_u64_le_bytes_multi(
                cs,
                tr,
                tr2.as_deref_mut(),
                tr3.as_deref_mut(),
                b"shout/k",
                inst.k as u64,
                &format!("{ctx}_lut_{lut_idx}_k"),
            )?;
            self.append_message_u64_le_bytes_multi(
                cs,
                tr,
                tr2.as_deref_mut(),
                tr3.as_deref_mut(),
                b"shout/d",
                inst.d as u64,
                &format!("{ctx}_lut_{lut_idx}_d"),
            )?;
            self.append_message_u64_le_bytes_multi(
                cs,
                tr,
                tr2.as_deref_mut(),
                tr3.as_deref_mut(),
                b"shout/n_side",
                inst.n_side as u64,
                &format!("{ctx}_lut_{lut_idx}_n_side"),
            )?;
            self.append_message_u64_le_bytes_multi(
                cs,
                tr,
                tr2.as_deref_mut(),
                tr3.as_deref_mut(),
                b"shout/steps",
                inst.steps as u64,
                &format!("{ctx}_lut_{lut_idx}_steps"),
            )?;
            self.append_message_u64_le_bytes_multi(
                cs,
                tr,
                tr2.as_deref_mut(),
                tr3.as_deref_mut(),
                b"shout/ell",
                inst.ell as u64,
                &format!("{ctx}_lut_{lut_idx}_ell"),
            )?;
            self.append_message_u64_le_bytes_multi(
                cs,
                tr,
                tr2.as_deref_mut(),
                tr3.as_deref_mut(),
                b"shout/lanes",
                inst.lanes.max(1) as u64,
                &format!("{ctx}_lut_{lut_idx}_lanes"),
            )?;

            self.bind_shout_table_spec(
                cs,
                tr,
                tr2.as_deref_mut(),
                tr3.as_deref_mut(),
                &inst.table_spec,
                &format!("{ctx}_lut_{lut_idx}_table_spec"),
            )?;

            let table_digest_bytes =
                self.digest_fields_bytes32(cs, b"shout/table", inst.table.as_slice(), &format!("{ctx}_lut_{lut_idx}_table_digest"))?;
            tr.append_message_bytes_allocated(
                cs,
                b"shout/table_digest",
                table_digest_bytes.as_slice(),
                &format!("{ctx}_lut_{lut_idx}_table_digest_bind"),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;
            if let Some(tr2) = tr2.as_mut() {
                tr2.append_message_bytes_allocated(
                    cs,
                    b"shout/table_digest",
                    table_digest_bytes.as_slice(),
                    &format!("{ctx}_lut_{lut_idx}_table_digest_bind"),
                )
                .map_err(SpartanBridgeError::BellpepperError)?;
            }
            if let Some(tr3) = tr3.as_mut() {
                tr3.append_message_bytes_allocated(
                    cs,
                    b"shout/table_digest",
                    table_digest_bytes.as_slice(),
                    &format!("{ctx}_lut_{lut_idx}_table_digest_bind"),
                )
                .map_err(SpartanBridgeError::BellpepperError)?;
            }
        }

        // MEMs
        let mem_count_bytes = &(step_public.mem_insts.len() as u64).to_le_bytes();
        tr.append_message(cs, b"step/mem_count", mem_count_bytes, &format!("{ctx}_mem_count"))
            .map_err(SpartanBridgeError::BellpepperError)?;
        if let Some(tr2) = tr2.as_mut() {
            tr2.append_message(cs, b"step/mem_count", mem_count_bytes, &format!("{ctx}_mem_count"))
                .map_err(SpartanBridgeError::BellpepperError)?;
        }
        if let Some(tr3) = tr3.as_mut() {
            tr3.append_message(cs, b"step/mem_count", mem_count_bytes, &format!("{ctx}_mem_count"))
                .map_err(SpartanBridgeError::BellpepperError)?;
        }

        for (mem_idx, inst) in step_public.mem_insts.iter().enumerate() {
            let mem_idx_bytes = &(mem_idx as u64).to_le_bytes();
            tr.append_message(cs, b"step/mem_idx", mem_idx_bytes, &format!("{ctx}_mem_{mem_idx}_idx"))
                .map_err(SpartanBridgeError::BellpepperError)?;
            if let Some(tr2) = tr2.as_mut() {
                tr2.append_message(cs, b"step/mem_idx", mem_idx_bytes, &format!("{ctx}_mem_{mem_idx}_idx"))
                    .map_err(SpartanBridgeError::BellpepperError)?;
            }
            if let Some(tr3) = tr3.as_mut() {
                tr3.append_message(cs, b"step/mem_idx", mem_idx_bytes, &format!("{ctx}_mem_{mem_idx}_idx"))
                    .map_err(SpartanBridgeError::BellpepperError)?;
            }

            self.append_message_u64_le_bytes_multi(
                cs,
                tr,
                tr2.as_deref_mut(),
                tr3.as_deref_mut(),
                b"twist/k",
                inst.k as u64,
                &format!("{ctx}_mem_{mem_idx}_k"),
            )?;
            self.append_message_u64_le_bytes_multi(
                cs,
                tr,
                tr2.as_deref_mut(),
                tr3.as_deref_mut(),
                b"twist/d",
                inst.d as u64,
                &format!("{ctx}_mem_{mem_idx}_d"),
            )?;
            self.append_message_u64_le_bytes_multi(
                cs,
                tr,
                tr2.as_deref_mut(),
                tr3.as_deref_mut(),
                b"twist/n_side",
                inst.n_side as u64,
                &format!("{ctx}_mem_{mem_idx}_n_side"),
            )?;
            self.append_message_u64_le_bytes_multi(
                cs,
                tr,
                tr2.as_deref_mut(),
                tr3.as_deref_mut(),
                b"twist/steps",
                inst.steps as u64,
                &format!("{ctx}_mem_{mem_idx}_steps"),
            )?;
            self.append_message_u64_le_bytes_multi(
                cs,
                tr,
                tr2.as_deref_mut(),
                tr3.as_deref_mut(),
                b"twist/ell",
                inst.ell as u64,
                &format!("{ctx}_mem_{mem_idx}_ell"),
            )?;
            self.append_message_u64_le_bytes_multi(
                cs,
                tr,
                tr2.as_deref_mut(),
                tr3.as_deref_mut(),
                b"twist/lanes",
                inst.lanes.max(1) as u64,
                &format!("{ctx}_mem_{mem_idx}_lanes"),
            )?;

            let (init_label, init_fields): (&'static [u8], Vec<NeoF>) = match &inst.init {
                neo_memory::MemInit::Zero => (b"twist/init/zero", Vec::new()),
                neo_memory::MemInit::Sparse(pairs) => {
                    let mut fs = Vec::with_capacity(2 * pairs.len());
                    for (addr, val) in pairs.iter() {
                        fs.push(NeoF::from_u64(*addr));
                        fs.push(*val);
                    }
                    (b"twist/init/sparse", fs)
                }
            };

            let init_digest_bytes =
                self.digest_fields_bytes32(cs, init_label, init_fields.as_slice(), &format!("{ctx}_mem_{mem_idx}_init_digest"))?;
            tr.append_message_bytes_allocated(
                cs,
                b"twist/init_digest",
                init_digest_bytes.as_slice(),
                &format!("{ctx}_mem_{mem_idx}_init_digest_bind"),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;
            if let Some(tr2) = tr2.as_mut() {
                tr2.append_message_bytes_allocated(
                    cs,
                    b"twist/init_digest",
                    init_digest_bytes.as_slice(),
                    &format!("{ctx}_mem_{mem_idx}_init_digest_bind"),
                )
                .map_err(SpartanBridgeError::BellpepperError)?;
            }
            if let Some(tr3) = tr3.as_mut() {
                tr3.append_message_bytes_allocated(
                    cs,
                    b"twist/init_digest",
                    init_digest_bytes.as_slice(),
                    &format!("{ctx}_mem_{mem_idx}_init_digest_bind"),
                )
                .map_err(SpartanBridgeError::BellpepperError)?;
            }
        }

        tr.append_message(cs, b"step/absorb_memory_done", &[], &ctx)
            .map_err(SpartanBridgeError::BellpepperError)?;
        if let Some(tr2) = tr2.as_mut() {
            tr2.append_message(cs, b"step/absorb_memory_done", &[], &ctx)
                .map_err(SpartanBridgeError::BellpepperError)?;
        }
        if let Some(tr3) = tr3.as_mut() {
            tr3.append_message(cs, b"step/absorb_memory_done", &[], &ctx)
                .map_err(SpartanBridgeError::BellpepperError)?;
        }
        Ok(())
    }

    fn bind_header_and_mcs_with_digest<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        tr: &mut Poseidon2TranscriptVar,
        mcs_inst: &neo_ccs::McsInstance<neo_ajtai::Commitment, NeoF>,
        step_idx: usize,
    ) -> Result<McsInstanceVars> {
        let ctx = format!("step_{step_idx}_bind_header");

        // Phase marker.
        tr.append_message(cs, neo_transcript::labels::PI_CCS, &[], &ctx)
            .map_err(SpartanBridgeError::BellpepperError)?;

        // Header (matches `neo_reductions::engines::utils::bind_header_and_instances_with_digest`).
        tr.append_message(cs, b"neo/ccs/header/v1", &[], &ctx)
            .map_err(SpartanBridgeError::BellpepperError)?;
        tr.append_u64s(
            cs,
            b"ccs/header",
            &[
                64,
                self.s_supported as u64,
                self.lambda as u64,
                self.ell as u64,
                self.d_sc as u64,
                self.slack_bits_abs as u64,
            ],
            &ctx,
        )
        .map_err(SpartanBridgeError::BellpepperError)?;
        tr.append_message(cs, b"ccs/slack_sign", &[self.slack_sign], &ctx)
            .map_err(SpartanBridgeError::BellpepperError)?;

        // Instances (CCS dims + matrix digest + polynomial + MCS instance).
        tr.append_message(cs, b"neo/ccs/instances", &[], &ctx)
            .map_err(SpartanBridgeError::BellpepperError)?;
        tr.append_u64s(
            cs,
            b"dims",
            &[self.ccs_n as u64, self.ccs_m as u64, self.ccs_t as u64],
            &ctx,
        )
        .map_err(SpartanBridgeError::BellpepperError)?;

        for (i, &digest_elem) in self.ccs_mat_digest.iter().enumerate() {
            tr.append_fields_u64s(cs, b"mat_digest", &[digest_elem], &format!("{ctx}_mat_digest_{i}"))
                .map_err(SpartanBridgeError::BellpepperError)?;
        }

        // CCS sparse polynomial (sorted by exponent vector).
        tr.append_message(cs, b"neo/ccs/poly", &[], &ctx)
            .map_err(SpartanBridgeError::BellpepperError)?;
        tr.append_u64s(cs, b"arity", &[self.poly_arity as u64], &ctx)
            .map_err(SpartanBridgeError::BellpepperError)?;
        tr.append_u64s(cs, b"terms_len", &[self.poly_f.len() as u64], &ctx)
            .map_err(SpartanBridgeError::BellpepperError)?;

        for (term_idx, term) in self.poly_f.iter().enumerate() {
            use p3_field::PrimeField64;

            tr.append_fields_u64s(
                cs,
                b"coeff",
                &[term.coeff_native.as_canonical_u64()],
                &format!("{ctx}_term_{term_idx}_coeff"),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;
            let exps_u64: Vec<u64> = term.exps.iter().map(|&e| e as u64).collect();
            tr.append_u64s(cs, b"exps", &exps_u64, &format!("{ctx}_term_{term_idx}_exps"))
                .map_err(SpartanBridgeError::BellpepperError)?;
        }

        // MCS instance fields are witness-provided (absorb as vars, not constants).
        use p3_field::PrimeField64;
        let x_vals: Vec<CircuitF> = mcs_inst.x.iter().map(|x| CircuitF::from(x.as_canonical_u64())).collect();
        let mut x_vars = Vec::with_capacity(x_vals.len());
        for (i, &xv) in x_vals.iter().enumerate() {
            x_vars.push(cs.alloc(|| format!("step_{step_idx}_mcs_x_{i}"), || Ok(xv))?);
        }
        tr.append_fields_vars(cs, b"x", &x_vars, &x_vals, &format!("{ctx}_mcs_x"))
            .map_err(SpartanBridgeError::BellpepperError)?;

        tr.append_u64s(cs, b"m_in", &[mcs_inst.m_in as u64], &format!("{ctx}_mcs_m_in"))
            .map_err(SpartanBridgeError::BellpepperError)?;

        let c_vals: Vec<CircuitF> = mcs_inst
            .c
            .data
            .iter()
            .map(|c| CircuitF::from(c.as_canonical_u64()))
            .collect();
        let mut c_vars = Vec::with_capacity(c_vals.len());
        for (i, &cv) in c_vals.iter().enumerate() {
            c_vars.push(cs.alloc(|| format!("step_{step_idx}_mcs_c_{i}"), || Ok(cv))?);
        }
        tr.append_fields_vars(cs, b"c_data", &c_vars, &c_vals, &format!("{ctx}_mcs_c"))
            .map_err(SpartanBridgeError::BellpepperError)?;

        Ok(McsInstanceVars { x: x_vars, c_data: c_vars })
    }

    fn bind_me_inputs_v2<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        tr: &mut Poseidon2TranscriptVar,
        me_inputs_vals: &[neo_ccs::MeInstance<neo_ajtai::Commitment, NeoF, neo_math::K>],
        me_inputs_vars: &[MeInstanceVars],
        step_idx: usize,
    ) -> Result<()> {
        if me_inputs_vals.len() != me_inputs_vars.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "step {step_idx}: bind_me_inputs_v2 length mismatch (vals={}, vars={})",
                me_inputs_vals.len(),
                me_inputs_vars.len()
            )));
        }

        let ctx = format!("step_{step_idx}_bind_me_inputs");
        tr.append_message(cs, b"neo/ccs/me_inputs/v2", &[], &ctx)
            .map_err(SpartanBridgeError::BellpepperError)?;
        tr.append_u64s(cs, b"me_count", &[me_inputs_vals.len() as u64], &ctx)
            .map_err(SpartanBridgeError::BellpepperError)?;

        for (me_idx, (me_val, me_var)) in me_inputs_vals.iter().zip(me_inputs_vars.iter()).enumerate() {
            // c_data_in
            use p3_field::PrimeField64;
            let c_vals: Vec<CircuitF> = me_val.c.data.iter().map(|c| CircuitF::from(c.as_canonical_u64())).collect();
            tr.append_fields_vars(
                cs,
                b"c_data_in",
                &me_var.c_data,
                &c_vals,
                &format!("{ctx}_me_{me_idx}_c_data"),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;

            // m_in_in
            tr.append_u64s(
                cs,
                b"m_in_in",
                &[me_val.m_in as u64],
                &format!("{ctx}_me_{me_idx}_m_in"),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;

            // r_in: flatten coeffs (c0,c1) for each limb.
            let mut r_vars: Vec<bellpepper_core::Variable> = Vec::with_capacity(me_var.r.len() * 2);
            let mut r_vals: Vec<CircuitF> = Vec::with_capacity(me_var.r.len() * 2);
            for (limb_idx, limb) in me_val.r.iter().enumerate() {
                let coeffs = limb.as_coeffs();
                r_vars.push(me_var.r[limb_idx].c0);
                r_vals.push(CircuitF::from(coeffs[0].as_canonical_u64()));
                r_vars.push(me_var.r[limb_idx].c1);
                r_vals.push(CircuitF::from(coeffs[1].as_canonical_u64()));
            }
            tr.append_fields_vars(
                cs,
                b"r_in",
                &r_vars,
                &r_vals,
                &format!("{ctx}_me_{me_idx}_r_in"),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;

            // y_elem: flatten coeffs for all elements in all rows.
            let mut y_vars: Vec<bellpepper_core::Variable> = Vec::new();
            let mut y_vals: Vec<CircuitF> = Vec::new();
            for (j, yj) in me_val.y.iter().enumerate() {
                for (rho, y_elem) in yj.iter().enumerate() {
                    let coeffs = y_elem.as_coeffs();
                    y_vars.push(me_var.y[j][rho].c0);
                    y_vals.push(CircuitF::from(coeffs[0].as_canonical_u64()));
                    y_vars.push(me_var.y[j][rho].c1);
                    y_vals.push(CircuitF::from(coeffs[1].as_canonical_u64()));
                }
            }
            tr.append_fields_vars(
                cs,
                b"y_elem",
                &y_vars,
                &y_vals,
                &format!("{ctx}_me_{me_idx}_y_elem"),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;
        }

        Ok(())
    }

    fn sample_pi_ccs_challenges_from_transcript<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        tr: &mut Poseidon2TranscriptVar,
        step_idx: usize,
    ) -> Result<(Vec<KNumVar>, Vec<KNumVar>, Vec<KNumVar>, KNumVar)> {
        let ctx = format!("step_{step_idx}_sample_pi_ccs_chals");
        tr.append_message(cs, b"neo/ccs/chals/v1", &[], &ctx)
            .map_err(SpartanBridgeError::BellpepperError)?;

        let mut alpha = Vec::with_capacity(self.ell_d);
        for i in 0..self.ell_d {
            let fs = tr
                .challenge_fields(cs, b"chal/k", 2, &format!("{ctx}_alpha_{i}"))
                .map_err(SpartanBridgeError::BellpepperError)?;
            alpha.push(KNumVar {
                c0: fs[0].get_variable(),
                c1: fs[1].get_variable(),
            });
        }

        let mut beta = Vec::with_capacity(self.ell);
        for i in 0..self.ell {
            let fs = tr
                .challenge_fields(cs, b"chal/k", 2, &format!("{ctx}_beta_{i}"))
                .map_err(SpartanBridgeError::BellpepperError)?;
            beta.push(KNumVar {
                c0: fs[0].get_variable(),
                c1: fs[1].get_variable(),
            });
        }
        let beta_a = beta[..self.ell_d].to_vec();
        let beta_r = beta[self.ell_d..].to_vec();

        let g = tr
            .challenge_fields(cs, b"chal/k", 2, &format!("{ctx}_gamma"))
            .map_err(SpartanBridgeError::BellpepperError)?;
        let gamma = KNumVar {
            c0: g[0].get_variable(),
            c1: g[1].get_variable(),
        };

        Ok((alpha, beta_a, beta_r, gamma))
    }

    fn sample_ext_point<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        tr: &mut Poseidon2TranscriptVar,
        label: &'static [u8],
        coord0_label: &'static [u8],
        coord1_label: &'static [u8],
        len: usize,
        ctx: &str,
    ) -> std::result::Result<(Vec<KNumVar>, Vec<neo_math::K>), SynthesisError> {
        use neo_math::{F as NeoF, KExtensions};

        let mut out_vars = Vec::with_capacity(len);
        let mut out_vals = Vec::with_capacity(len);
        for i in 0..len {
            tr.append_message(cs, label, &(i as u64).to_le_bytes(), ctx)?;
            let c0 = tr.challenge_field(cs, coord0_label, ctx)?;
            let c1 = tr.challenge_field(cs, coord1_label, ctx)?;

            let c0_u64 = c0.get_value().unwrap_or(CircuitF::from(0u64)).to_canonical_u64();
            let c1_u64 = c1.get_value().unwrap_or(CircuitF::from(0u64)).to_canonical_u64();
            out_vals.push(neo_math::K::from_coeffs([NeoF::from_u64(c0_u64), NeoF::from_u64(c1_u64)]));

            out_vars.push(KNumVar {
                c0: c0.get_variable(),
                c1: c1.get_variable(),
            });
        }
        Ok((out_vars, out_vals))
    }

    fn k_mul_by_f_var_with_hint<CS: ConstraintSystem<CircuitF>>(
        cs: &mut CS,
        k_var: &KNumVar,
        k_val: neo_math::K,
        f_var: bellpepper_core::Variable,
        f_val: NeoF,
        label: &str,
    ) -> Result<(KNumVar, neo_math::K)> {
        use neo_math::{from_complex, KExtensions};
        use p3_field::PrimeField64;

        let coeffs = k_val.as_coeffs();
        let out0 = coeffs[0] * f_val;
        let out1 = coeffs[1] * f_val;
        let out_val = from_complex(out0, out1);

        let out0_c = CircuitF::from(out0.as_canonical_u64());
        let out1_c = CircuitF::from(out1.as_canonical_u64());
        let out0_var = cs.alloc(|| format!("{label}_c0"), || Ok(out0_c))?;
        let out1_var = cs.alloc(|| format!("{label}_c1"), || Ok(out1_c))?;

        cs.enforce(
            || format!("{label}_c0_mul"),
            |lc| lc + k_var.c0,
            |lc| lc + f_var,
            |lc| lc + out0_var,
        );
        cs.enforce(
            || format!("{label}_c1_mul"),
            |lc| lc + k_var.c1,
            |lc| lc + f_var,
            |lc| lc + out1_var,
        );

        Ok((KNumVar { c0: out0_var, c1: out1_var }, out_val))
    }

    fn verify_output_sumcheck_rounds_get_state<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        tr: &mut Poseidon2TranscriptVar,
        step_idx: usize,
        cfg: &OutputBindingConfig,
        program_io_claims: &[(u64, bellpepper_core::Variable, NeoF)],
        round_polys: &[Vec<neo_math::K>],
    ) -> Result<OutputSumcheckStateVars> {
        use neo_math::{from_complex, F as NeoF, K as NeoK, KExtensions};
        use p3_field::PrimeField64;

        let ctx = format!("step_{step_idx}_output_sumcheck");

        if round_polys.len() != cfg.num_bits {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "{ctx}: output_sc.round_polys.len()={}, expected num_bits={}",
                round_polys.len(),
                cfg.num_bits
            )));
        }
        if program_io_claims.len() != cfg.program_io.num_claims() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "{ctx}: program_io_claims.len()={}, expected num_claims={}",
                program_io_claims.len(),
                cfg.program_io.num_claims()
            )));
        }

        // OutputSumcheckParams::sample_from_transcript:
        // - absorb program_io
        // - sample r_addr
        tr.append_message(
            cs,
            b"output_check/num_claims",
            &(program_io_claims.len() as u64).to_le_bytes(),
            &format!("{ctx}_num_claims"),
        )
        .map_err(SpartanBridgeError::BellpepperError)?;
        for (claim_idx, (addr, var, val)) in program_io_claims.iter().enumerate() {
            tr.append_message(
                cs,
                b"output_check/addr",
                &addr.to_le_bytes(),
                &format!("{ctx}_addr_{claim_idx}"),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;
            let v = CircuitF::from(val.as_canonical_u64());
            tr.append_fields_vars(
                cs,
                b"output_check/value",
                &[*var],
                &[v],
                &format!("{ctx}_val_{claim_idx}"),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;
        }

        let (r_addr_vars, r_addr_vals) = self
            .sample_ext_point(
                cs,
                tr,
                b"output_check/r_addr/idx",
                b"output_check/chal/re",
                b"output_check/chal/im",
                cfg.num_bits,
                &format!("{ctx}_r_addr"),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;

        // Verify output sumcheck rounds, returning (output_final, r_prime).
        let mut running_sum_var = helpers::k_zero(cs, &format!("{ctx}_claim_init"))?;
        let mut r_prime_vars = Vec::with_capacity(round_polys.len());
        let mut r_prime_vals = Vec::with_capacity(round_polys.len());

        for (round_idx, coeffs) in round_polys.iter().enumerate() {
            if coeffs.len() != 4 {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "{ctx}: output round {round_idx} wrong degree (len={}, expected 4)",
                    coeffs.len()
                )));
            }

            let mut coeff_vars = Vec::with_capacity(coeffs.len());
            for (coeff_idx, &coeff) in coeffs.iter().enumerate() {
                coeff_vars.push(helpers::alloc_k_from_neo(
                    cs,
                    coeff,
                    &format!("{ctx}_round_{round_idx}_coeff_{coeff_idx}"),
                )?);
            }

            sumcheck_round_gadget(
                cs,
                coeff_vars.as_slice(),
                coeffs.as_slice(),
                &running_sum_var,
                self.delta,
                &format!("{ctx}_round_{round_idx}_invariant"),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;

            // Absorb coeffs into transcript under `output_check/round_coeff` (one append_fields per coeff).
            for (coeff_idx, (coeff_var, &coeff_val)) in coeff_vars.iter().zip(coeffs.iter()).enumerate() {
                let c = coeff_val.as_coeffs();
                let vals = [CircuitF::from(c[0].as_canonical_u64()), CircuitF::from(c[1].as_canonical_u64())];
                tr.append_fields_vars(
                    cs,
                    b"output_check/round_coeff",
                    &[coeff_var.c0, coeff_var.c1],
                    &vals,
                    &format!("{ctx}_round_{round_idx}_absorb_{coeff_idx}"),
                )
                .map_err(SpartanBridgeError::BellpepperError)?;
            }

            // sample_k_challenge(tr)
            let c0 = tr
                .challenge_field(cs, b"output_check/chal/re", &format!("{ctx}_round_{round_idx}_chal"))
                .map_err(SpartanBridgeError::BellpepperError)?;
            let c1 = tr
                .challenge_field(cs, b"output_check/chal/im", &format!("{ctx}_round_{round_idx}_chal"))
                .map_err(SpartanBridgeError::BellpepperError)?;
            let r_var = KNumVar {
                c0: c0.get_variable(),
                c1: c1.get_variable(),
            };
            let c0_u64 = c0.get_value().unwrap_or(CircuitF::from(0u64)).to_canonical_u64();
            let c1_u64 = c1.get_value().unwrap_or(CircuitF::from(0u64)).to_canonical_u64();
            let r_val = from_complex(NeoF::from_u64(c0_u64), NeoF::from_u64(c1_u64));
            r_prime_vars.push(r_var.clone());
            r_prime_vals.push(r_val);

            running_sum_var = sumcheck_eval_gadget(
                cs,
                coeff_vars.as_slice(),
                coeffs.as_slice(),
                &r_var,
                r_val,
                self.delta,
                &format!("{ctx}_round_{round_idx}_eval"),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;
        }

        let output_final_var = running_sum_var;

        // eq_eval = eq_points(r_addr, r_prime)
        let (eq_eval_var, eq_eval_val) = self.eq_points(
            cs,
            step_idx,
            r_addr_vars.as_slice(),
            r_prime_vars.as_slice(),
            r_addr_vals.as_slice(),
            r_prime_vals.as_slice(),
            &format!("{ctx}_eq_eval"),
        )?;

        // io_mask_eval and val_io_eval at r_prime.
        let one_var = helpers::k_one(cs, &format!("{ctx}_one"))?;
        let mut io_mask_var = helpers::k_zero(cs, &format!("{ctx}_io_mask_init"))?;
        let mut io_mask_val = NeoK::ZERO;
        let mut val_io_var = helpers::k_zero(cs, &format!("{ctx}_val_io_init"))?;
        let mut val_io_val = NeoK::ZERO;

        for (claim_idx, (addr, val_var, val_f)) in program_io_claims.iter().enumerate() {
            // chi_at_point(r_prime, addr)
            let mut chi_var = one_var.clone();
            let mut chi_val = NeoK::ONE;
            for bit_idx in 0..cfg.num_bits {
                let bit = ((*addr >> bit_idx) & 1) as u8;
                let (factor_var, factor_val) = if bit == 1 {
                    (r_prime_vars[bit_idx].clone(), r_prime_vals[bit_idx])
                } else {
                    let one_minus_val = NeoK::ONE - r_prime_vals[bit_idx];
                    let one_minus_var = alloc_k(
                        cs,
                        Some(KNum::<CircuitF>::from_neo_k(one_minus_val)),
                        &format!("{ctx}_chi_{claim_idx}_1_minus_r_{bit_idx}"),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;
                    // one_minus + r = 1
                    cs.enforce(
                        || format!("{ctx}_chi_{claim_idx}_1_minus_r_{bit_idx}_c0"),
                        |lc| lc + one_minus_var.c0 + r_prime_vars[bit_idx].c0,
                        |lc| lc + CS::one(),
                        |lc| lc + one_var.c0,
                    );
                    cs.enforce(
                        || format!("{ctx}_chi_{claim_idx}_1_minus_r_{bit_idx}_c1"),
                        |lc| lc + one_minus_var.c1 + r_prime_vars[bit_idx].c1,
                        |lc| lc + CS::one(),
                        |lc| lc + one_var.c1,
                    );
                    (one_minus_var, one_minus_val)
                };

                let (new_chi, new_chi_val) = helpers::k_mul_with_hint(
                    cs,
                    &chi_var,
                    chi_val,
                    &factor_var,
                    factor_val,
                    self.delta,
                    &format!("{ctx}_chi_{claim_idx}_mul_{bit_idx}"),
                )?;
                chi_var = new_chi;
                chi_val = new_chi_val;
            }

            io_mask_val += chi_val;
            io_mask_var = k_add_raw(
                cs,
                &io_mask_var,
                &chi_var,
                Some(KNum::<CircuitF>::from_neo_k(io_mask_val)),
                &format!("{ctx}_io_mask_add_{claim_idx}"),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;

            let (scaled, scaled_val) =
                Self::k_mul_by_f_var_with_hint(cs, &chi_var, chi_val, *val_var, *val_f, &format!("{ctx}_val_io_mul_{claim_idx}"))?;
            val_io_val += scaled_val;
            val_io_var = k_add_raw(
                cs,
                &val_io_var,
                &scaled,
                Some(KNum::<CircuitF>::from_neo_k(val_io_val)),
                &format!("{ctx}_val_io_add_{claim_idx}"),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;
        }

        Ok(OutputSumcheckStateVars {
            r_prime_vars,
            r_prime_vals,
            output_final_var,
            eq_eval_var,
            eq_eval_val,
            io_mask_eval_var: io_mask_var,
            io_mask_eval_val: io_mask_val,
            val_io_eval_var: val_io_var,
            val_io_eval_val: val_io_val,
        })
    }

    fn claimed_initial_sum_value_host(
        &self,
        alpha: &[neo_math::K],
        gamma: neo_math::K,
        me_inputs: &[neo_ccs::MeInstance<neo_ajtai::Commitment, NeoF, neo_math::K>],
    ) -> neo_math::K {
        use core::cmp::min;
        use neo_math::K as NeoK;

        let k_total = 1 + me_inputs.len();
        if k_total < 2 || me_inputs.is_empty() {
            return NeoK::ZERO;
        }

        let d_sz = 1usize << alpha.len();
        let mut chi_a = vec![NeoK::ZERO; d_sz];
        for rho in 0..d_sz {
            let mut w = NeoK::ONE;
            for (bit, &a) in alpha.iter().enumerate() {
                let is_one = ((rho >> bit) & 1) == 1;
                w *= if is_one { a } else { NeoK::ONE - a };
            }
            chi_a[rho] = w;
        }

        let t = me_inputs[0].y.len();

        let mut gamma_to_k = NeoK::ONE;
        for _ in 0..k_total {
            gamma_to_k *= gamma;
        }

        let mut inner = NeoK::ZERO;
        for j in 0..t {
            for (idx, out) in me_inputs.iter().enumerate() {
                let i_abs = idx + 2;

                let yj = &out.y[j];
                let mut y_eval = NeoK::ZERO;
                let limit = min(d_sz, yj.len());
                for rho in 0..limit {
                    y_eval += yj[rho] * chi_a[rho];
                }

                let mut weight = NeoK::ONE;
                for _ in 0..(i_abs - 1) {
                    weight *= gamma;
                }
                for _ in 0..j {
                    weight *= gamma_to_k;
                }
                inner += weight * y_eval;
            }
        }

        gamma_to_k * inner
    }

    fn eval_poly_k(coeffs: &[neo_math::K], x: neo_math::K) -> neo_math::K {
        let mut acc = neo_math::K::ZERO;
        for &c in coeffs.iter().rev() {
            acc = acc * x + c;
        }
        acc
    }

    fn enforce_rot_rho_sampling_no_reject_mod5<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        tr: &mut Poseidon2TranscriptVar,
        lane: &'static [u8],
        rho_vars: &[Vec<Vec<bellpepper_core::Variable>>],
        step_idx: usize,
    ) -> Result<()> {
        use neo_math::D;

        if rho_vars.is_empty() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "step {step_idx}: rlc_rhos must be non-empty"
            )));
        }
        for (i, rho) in rho_vars.iter().enumerate() {
            if rho.len() != D {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "step {step_idx}: rlc_rhos[{i}] row count mismatch (got {}, expected {D})",
                    rho.len()
                )));
            }
            for (r, row) in rho.iter().enumerate() {
                if row.len() != D {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "step {step_idx}: rlc_rhos[{i}][{r}] col count mismatch (got {}, expected {D})",
                        row.len()
                    )));
                }
            }
        }

        // Goldilocks ring alphabet size (RotRing::goldilocks): [-2,-1,0,1,2].
        const ALPHABET_M: u64 = 5;
        const DIGEST_U16S: usize = 16; // 32 bytes = 16 u16 chunks
        let digests_per_rho = (D + (DIGEST_U16S - 1)) / DIGEST_U16S; // ceil(D/16) = 4 for D=54

        let lane_str: &str = match lane {
            b"main" => "main",
            b"val" => "val",
            _ => "lane",
        };
        let ctx = format!("step_{step_idx}_rho_fs_{lane_str}");

        let minus_one = CircuitF::from(0u64) - CircuitF::from(1u64);

        let alloc_bit = |cs: &mut CS, bit: bool, label: String| -> std::result::Result<bellpepper_core::Variable, SynthesisError> {
            let v = CircuitF::from(if bit { 1u64 } else { 0u64 });
            let var = cs.alloc(|| label.clone(), || Ok(v))?;
            // var * (var - 1) == 0
            cs.enforce(
                || format!("{label}_is_bit"),
                |lc| lc + var,
                |lc| lc + var + (minus_one, CS::one()),
                |lc| lc,
            );
            Ok(var)
        };

        // Special case: Π_RLC(k=1) uses identity, but still consumes transcript randomness.
        if rho_vars.len() == 1 {
            tr.append_message(cs, b"rlc/rot/index", &0u64.to_le_bytes(), &ctx)
                .map_err(SpartanBridgeError::BellpepperError)?;
            for k in 0..digests_per_rho {
                let ctr = k as u64;
                tr.append_message(cs, b"rlc/rot/chunk", &ctr.to_le_bytes(), &ctx)
                    .map_err(SpartanBridgeError::BellpepperError)?;
                let _ = tr
                    .digest32(cs, &format!("{ctx}_digest_0_{k}"))
                    .map_err(SpartanBridgeError::BellpepperError)?;
            }

            // Enforce ρ = I_D.
            for r in 0..D {
                for c in 0..D {
                    let want = if r == c { CircuitF::from(1u64) } else { CircuitF::from(0u64) };
                    cs.enforce(
                        || format!("{ctx}_rho0_identity_r{r}_c{c}"),
                        |lc| lc + rho_vars[0][r][c],
                        |lc| lc + CS::one(),
                        |lc| lc + (want, CS::one()),
                    );
                }
            }
            return Ok(());
        }

        // For k>1, enforce: ρ_i == rot(a_i) where a_i coeffs are sampled from transcript.
        for (rho_idx, rho) in rho_vars.iter().enumerate() {
            tr.append_message(
                cs,
                b"rlc/rot/index",
                &(rho_idx as u64).to_le_bytes(),
                &ctx,
            )
            .map_err(SpartanBridgeError::BellpepperError)?;

            // 1) Enforce column 0 coefficients a[0..D) match transcript-derived u16 % 5 mapping.
            //
            // Native: idx = (u16 % m), coeff = alphabet[idx] with alphabet=[-2,-1,0,1,2] => coeff = idx - 2.
            let mut row_idx = 0usize;
            for digest_idx in 0..digests_per_rho {
                let ctr = (rho_idx as u64).wrapping_add(digest_idx as u64);
                tr.append_message(cs, b"rlc/rot/chunk", &ctr.to_le_bytes(), &ctx)
                    .map_err(SpartanBridgeError::BellpepperError)?;
                let limbs = tr
                    .digest32(cs, &format!("{ctx}_digest_{rho_idx}_{digest_idx}"))
                    .map_err(SpartanBridgeError::BellpepperError)?;

                for limb_idx in 0..4 {
                    if row_idx >= D {
                        break;
                    }
                    let limb_val = limbs[limb_idx].get_value().unwrap_or(CircuitF::from(0u64));
                    let limb_u64 = limb_val.to_canonical_u64();

                    // Allocate 64 little-endian bits for the digest limb.
                    let mut bits = Vec::with_capacity(64);
                    for bit_idx in 0..64 {
                        let bit = ((limb_u64 >> bit_idx) & 1) == 1;
                        let v = alloc_bit(
                            cs,
                            bit,
                            format!("{ctx}_rho{rho_idx}_dig{digest_idx}_limb{limb_idx}_bit{bit_idx}"),
                        )
                        .map_err(SpartanBridgeError::BellpepperError)?;
                        bits.push(v);
                    }

                    // Enforce limb == Σ 2^i · bit_i.
                    cs.enforce(
                        || format!("{ctx}_rho{rho_idx}_dig{digest_idx}_limb{limb_idx}_recompose"),
                        |lc| {
                            let mut acc = lc;
                            for (i, &b) in bits.iter().enumerate() {
                                acc = acc + (CircuitF::from(1u64 << i), b);
                            }
                            acc
                        },
                        |lc| lc + CS::one(),
                        |lc| lc + limbs[limb_idx].get_variable(),
                    );

                    // Consume 4 u16 chunks from this limb (little-endian).
                    for chunk_idx in 0..4 {
                        if row_idx >= D {
                            break;
                        }
                        let chunk_bits = &bits[chunk_idx * 16..(chunk_idx + 1) * 16];
                        let chunk_u16 = ((limb_u64 >> (chunk_idx * 16)) & 0xFFFF) as u16;
                        let n0 = (chunk_u16 & 0xF) as u64;
                        let n1 = ((chunk_u16 >> 4) & 0xF) as u64;
                        let n2 = ((chunk_u16 >> 8) & 0xF) as u64;
                        let n3 = ((chunk_u16 >> 12) & 0xF) as u64;
                        let sum_nibbles = n0 + n1 + n2 + n3;
                        let q = (sum_nibbles / ALPHABET_M) as u8;
                        let r = (sum_nibbles % ALPHABET_M) as u8;

                        // Remainder r in [0..5): 3 bits with an extra constraint to exclude 5..7.
                        let r0 = alloc_bit(
                            cs,
                            (r & 1) != 0,
                            format!("{ctx}_rho{rho_idx}_row{row_idx}_r0"),
                        )
                        .map_err(SpartanBridgeError::BellpepperError)?;
                        let r1 = alloc_bit(
                            cs,
                            ((r >> 1) & 1) != 0,
                            format!("{ctx}_rho{rho_idx}_row{row_idx}_r1"),
                        )
                        .map_err(SpartanBridgeError::BellpepperError)?;
                        let r2 = alloc_bit(
                            cs,
                            ((r >> 2) & 1) != 0,
                            format!("{ctx}_rho{rho_idx}_row{row_idx}_r2"),
                        )
                        .map_err(SpartanBridgeError::BellpepperError)?;

                        // Quotient q for sum_nibbles / 5: 4 bits (0..12).
                        let q0 = alloc_bit(
                            cs,
                            (q & 1) != 0,
                            format!("{ctx}_rho{rho_idx}_row{row_idx}_q0"),
                        )
                        .map_err(SpartanBridgeError::BellpepperError)?;
                        let q1 = alloc_bit(
                            cs,
                            ((q >> 1) & 1) != 0,
                            format!("{ctx}_rho{rho_idx}_row{row_idx}_q1"),
                        )
                        .map_err(SpartanBridgeError::BellpepperError)?;
                        let q2 = alloc_bit(
                            cs,
                            ((q >> 2) & 1) != 0,
                            format!("{ctx}_rho{rho_idx}_row{row_idx}_q2"),
                        )
                        .map_err(SpartanBridgeError::BellpepperError)?;
                        let q3 = alloc_bit(
                            cs,
                            ((q >> 3) & 1) != 0,
                            format!("{ctx}_rho{rho_idx}_row{row_idx}_q3"),
                        )
                        .map_err(SpartanBridgeError::BellpepperError)?;

                        // r2 * (r0 + r1) == 0 enforces r ∈ {0,1,2,3,4}.
                        cs.enforce(
                            || format!("{ctx}_rho{rho_idx}_row{row_idx}_r_lt5"),
                            |lc| lc + r2,
                            |lc| lc + r0 + r1,
                            |lc| lc,
                        );

                        // sum_nibbles = Σ_{g=0..3} Σ_{j=0..3} 2^j * bit_{4g+j}.
                        // Enforce: sum_nibbles == 5*q + r with q,r derived from their bits.
                        let five = CircuitF::from(ALPHABET_M);
                        cs.enforce(
                            || format!("{ctx}_rho{rho_idx}_row{row_idx}_u16_mod5"),
                            |lc| {
                                let mut acc = lc;
                                for nibble in 0..4 {
                                    for j in 0..4 {
                                        let bit = chunk_bits[nibble * 4 + j];
                                        acc = acc + (CircuitF::from(1u64 << j), bit);
                                    }
                                }
                                acc
                            },
                            |lc| lc + CS::one(),
                            |lc| {
                                let mut rhs = lc;
                                // 5*q
                                rhs = rhs + (five, q0);
                                rhs = rhs + (five * CircuitF::from(2u64), q1);
                                rhs = rhs + (five * CircuitF::from(4u64), q2);
                                rhs = rhs + (five * CircuitF::from(8u64), q3);
                                // + r
                                rhs = rhs + r0;
                                rhs = rhs + (CircuitF::from(2u64), r1);
                                rhs = rhs + (CircuitF::from(4u64), r2);
                                rhs
                            },
                        );

                        // Enforce column-0 entry: ρ[row,0] = r - 2.
                        //
                        // i.e. ρ[row,0] + 2 == r0 + 2*r1 + 4*r2.
                        cs.enforce(
                            || format!("{ctx}_rho{rho_idx}_a_coeff_row{row_idx}"),
                            |lc| {
                                let mut acc = lc + rho[row_idx][0] + (CircuitF::from(2u64), CS::one());
                                acc = acc + (minus_one, r0);
                                acc = acc + (CircuitF::from(0u64) - CircuitF::from(2u64), r1);
                                acc = acc + (CircuitF::from(0u64) - CircuitF::from(4u64), r2);
                                acc
                            },
                            |lc| lc + CS::one(),
                            |lc| lc,
                        );

                        row_idx += 1;
                    }
                }
            }

            if row_idx != D {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "step {step_idx}: rho sampling produced {row_idx} coeffs, expected {D}"
                )));
            }

            // 2) Enforce the shift recurrence for all columns (Definition 7, Remark 1).
            //
            // For Goldilocks: Φ(X)=X^54 + X^27 + 1 so c0=1, c27=1, other c_r=0.
            // Let neg_c[r] = -c_r.
            // next[0] = last * neg_c[0]
            // next[r] = col[r-1] + last * neg_c[r]   for r>=1
            for col in 0..(D - 1) {
                let last = rho[D - 1][col];
                // next[0] = -last
                cs.enforce(
                    || format!("{ctx}_rho{rho_idx}_shift_col{col}_r0"),
                    |lc| lc + rho[0][col + 1] + last,
                    |lc| lc + CS::one(),
                    |lc| lc,
                );
                for r in 1..D {
                    // Enforce: next[r] = prev[r-1] - c_r * last, where c_r is the Φ coefficient.
                    // For Φ(X)=X^54 + X^27 + 1: c_27 = 1, others 0.
                    let c_r = if r == 27 { CircuitF::from(1u64) } else { CircuitF::from(0u64) };
                    cs.enforce(
                        || format!("{ctx}_rho{rho_idx}_shift_col{col}_r{r}"),
                        |lc| {
                            let mut acc = lc + rho[r][col + 1];
                            acc = acc + (minus_one, rho[r - 1][col]);
                            if c_r != CircuitF::from(0u64) {
                                acc = acc + (c_r, last);
                            }
                            acc
                        },
                        |lc| lc + CS::one(),
                        |lc| lc,
                    );
                }
            }
        }

        Ok(())
    }

    fn bind_rlc_inputs_lane<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        tr: &mut Poseidon2TranscriptVar,
        lane: &'static [u8],
        step_idx: usize,
        me_inputs: &[neo_ccs::MeInstance<neo_ajtai::Commitment, NeoF, neo_math::K>],
        me_inputs_vars: &[MeInstanceVars],
        fold_digest_bytes: &[AllocatedNum<CircuitF>],
    ) -> Result<()> {
        if me_inputs.len() != me_inputs_vars.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "step {step_idx}: bind_rlc_inputs_lane length mismatch (vals={}, vars={})",
                me_inputs.len(),
                me_inputs_vars.len()
            )));
        }
        if fold_digest_bytes.len() != 32 {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "step {step_idx}: fold_digest_bytes must have len 32, got {}",
                fold_digest_bytes.len()
            )));
        }

        let lane_str: &str = match lane {
            b"main" => "main",
            b"val" => "val",
            _ => "lane",
        };
        let ctx = format!("step_{step_idx}_bind_rlc_inputs_{lane_str}");

        tr.append_message(cs, b"fold/rlc_inputs/v1", lane, &ctx)
            .map_err(SpartanBridgeError::BellpepperError)?;
        tr.append_u64s(cs, b"step_idx", &[step_idx as u64], &ctx)
            .map_err(SpartanBridgeError::BellpepperError)?;
        tr.append_u64s(cs, b"me_count", &[me_inputs.len() as u64], &ctx)
            .map_err(SpartanBridgeError::BellpepperError)?;

        use neo_math::{F as NeoF, K as NeoK, KExtensions};
        use p3_field::PrimeField64;
        for (me_idx, (me, me_var)) in me_inputs.iter().zip(me_inputs_vars.iter()).enumerate() {
            let c_vals: Vec<CircuitF> = me.c.data.iter().map(|c| CircuitF::from(c.as_canonical_u64())).collect();
            tr.append_fields_vars(
                cs,
                b"c_data",
                &me_var.c_data,
                &c_vals,
                &format!("{ctx}_me_{me_idx}_c_data"),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;

            tr.append_u64s(cs, b"m_in", &[me.m_in as u64], &format!("{ctx}_me_{me_idx}_m_in"))
                .map_err(SpartanBridgeError::BellpepperError)?;

            // fold_digest is transcript-derived (header digest). Never absorb prover-chosen
            // `me.fold_digest` here, or ρ sampling becomes prover-influenced.
            tr.append_message_bytes_allocated(
                cs,
                b"me_fold_digest",
                fold_digest_bytes,
                &format!("{ctx}_me_{me_idx}_fd"),
            )
                .map_err(SpartanBridgeError::BellpepperError)?;

            for (limb_idx, limb) in me.r.iter().enumerate() {
                let coeffs = limb.as_coeffs();
                let coeff_vals = [CircuitF::from(coeffs[0].as_canonical_u64()), CircuitF::from(coeffs[1].as_canonical_u64())];
                if limb_idx >= me_var.r.len() {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "step {step_idx}: ME[{me_idx}] r limb out of bounds (have {}, need {})",
                        me_var.r.len(),
                        limb_idx + 1
                    )));
                }
                let vars = [me_var.r[limb_idx].c0, me_var.r[limb_idx].c1];
                tr.append_fields_vars(
                    cs,
                    b"r_limb",
                    &vars,
                    &coeff_vals,
                    &format!("{ctx}_me_{me_idx}_r_{limb_idx}"),
                )
                .map_err(SpartanBridgeError::BellpepperError)?;
            }

            // X flattened row-major.
            let x_vals: Vec<CircuitF> = me.X.as_slice().iter().map(|x| CircuitF::from(x.as_canonical_u64())).collect();
            let mut x_vars = Vec::with_capacity(x_vals.len());
            for row in &me_var.X {
                for &v in row {
                    x_vars.push(v);
                }
            }
            tr.append_fields_vars(cs, b"X", &x_vars, &x_vals, &format!("{ctx}_me_{me_idx}_X"))
                .map_err(SpartanBridgeError::BellpepperError)?;

            for (j, yj) in me.y.iter().enumerate() {
                for (rho, y_elem) in yj.iter().enumerate() {
                    let coeffs = y_elem.as_coeffs();
                    let coeff_vals = [
                        CircuitF::from(coeffs[0].as_canonical_u64()),
                        CircuitF::from(coeffs[1].as_canonical_u64()),
                    ];
                    if j >= me_var.y.len() || rho >= me_var.y[j].len() {
                        return Err(SpartanBridgeError::InvalidInput(format!(
                            "step {step_idx}: ME[{me_idx}] y index out of bounds (j={j}, rho={rho})"
                        )));
                    }
                    let vars = [me_var.y[j][rho].c0, me_var.y[j][rho].c1];
                    tr.append_fields_vars(
                        cs,
                        b"y_elem",
                        &vars,
                        &coeff_vals,
                        &format!("{ctx}_me_{me_idx}_y_{j}_{rho}"),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;
                }
            }

            // y_scalars must be canonical: base-b recomposition from the first D digits of y[j].
            let base_circ = CircuitF::from(self.base_b as u64);
            let bK = NeoK::from(NeoF::from_u64(self.base_b as u64));
            for (j, yj) in me.y.iter().enumerate() {
                let take = core::cmp::min(neo_math::D, yj.len());
                if j >= me_var.y.len() {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "step {step_idx}: ME[{me_idx}] y index out of bounds at j={j}"
                    )));
                }
                if me_var.y[j].len() < take {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "step {step_idx}: ME[{me_idx}] y[{j}] too short for y_scalars (have {}, need {})",
                        me_var.y[j].len(),
                        take
                    )));
                }

                // Native value (for witness assignment).
                let mut acc = NeoK::ZERO;
                let mut pw_k = NeoK::ONE;
                for rho in 0..take {
                    acc += pw_k * yj[rho];
                    pw_k *= bK;
                }

                // Allocate y_scalar[j] and constrain it to the recomposition from digits.
                let y_scalar_var = helpers::alloc_k_from_neo(cs, acc, &format!("{ctx}_me_{me_idx}_ys_{j}"))?;

                // c0: Σ b^rho * y[j][rho].c0 == y_scalar.c0
                cs.enforce(
                    || format!("{ctx}_me_{me_idx}_ys_{j}_c0"),
                    |lc| {
                        let mut res = lc;
                        let mut pow = CircuitF::from(1u64);
                        for rho in 0..take {
                            res = res + (pow, me_var.y[j][rho].c0);
                            pow *= base_circ;
                        }
                        res
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + y_scalar_var.c0,
                );

                // c1: Σ b^rho * y[j][rho].c1 == y_scalar.c1
                cs.enforce(
                    || format!("{ctx}_me_{me_idx}_ys_{j}_c1"),
                    |lc| {
                        let mut res = lc;
                        let mut pow = CircuitF::from(1u64);
                        for rho in 0..take {
                            res = res + (pow, me_var.y[j][rho].c1);
                            pow *= base_circ;
                        }
                        res
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + y_scalar_var.c1,
                );

                let coeffs = acc.as_coeffs();
                let coeff_vals = [
                    CircuitF::from(coeffs[0].as_canonical_u64()),
                    CircuitF::from(coeffs[1].as_canonical_u64()),
                ];
                let vars = [y_scalar_var.c0, y_scalar_var.c1];
                tr.append_fields_vars(cs, b"y_scalar", &vars, &coeff_vals, &format!("{ctx}_me_{me_idx}_ys_bind_{j}"))
                    .map_err(SpartanBridgeError::BellpepperError)?;
            }

            // Folding-only Phase 1 uses Pattern-B only: these fields must be canonical zeros.
            tr.append_u64s(cs, b"c_step_coords_len", &[0u64], &format!("{ctx}_me_{me_idx}_c_step_coords_len"))
                .map_err(SpartanBridgeError::BellpepperError)?;
            tr.append_fields_vars(cs, b"c_step_coords", &[], &[], &format!("{ctx}_me_{me_idx}_c_step_coords_empty"))
                .map_err(SpartanBridgeError::BellpepperError)?;
            tr.append_u64s(cs, b"u_offset", &[0u64], &format!("{ctx}_me_{me_idx}_u_offset_zero"))
                .map_err(SpartanBridgeError::BellpepperError)?;
            tr.append_u64s(cs, b"u_len", &[0u64], &format!("{ctx}_me_{me_idx}_u_len_zero"))
                .map_err(SpartanBridgeError::BellpepperError)?;
        }

        Ok(())
    }

    fn verify_fold_step_route_a<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        tr: &mut Poseidon2TranscriptVar,
        steps_tr: &mut Poseidon2TranscriptVar,
        step_idx: usize,
        step_proof: &StepProof,
        step_public: &neo_memory::witness::StepInstanceBundle<neo_ajtai::Commitment, NeoF, neo_math::K>,
        prev_step_public: Option<&neo_memory::witness::StepInstanceBundle<neo_ajtai::Commitment, NeoF, neo_math::K>>,
        me_inputs_vals: &[neo_ccs::MeInstance<neo_ajtai::Commitment, NeoF, neo_math::K>],
        me_inputs_vars: &[MeInstanceVars],
        prev_mcs_vars: Option<&McsInstanceVars>,
        output_binding: Option<(&OutputBindingConfig, &OutputBindingProof, &[(u64, bellpepper_core::Variable, NeoF)])>,
    ) -> Result<(Vec<MeInstanceVars>, McsInstanceVars, Vec<MeInstanceVars>)> {
        let mcs_inst = &step_public.mcs_inst;
        let mem_enabled = self.instance.statement.mem_enabled;
        let include_ob = self.instance.statement.output_binding_enabled
            && step_idx + 1 == usize::try_from(self.instance.statement.step_count).map_err(|_| {
                SpartanBridgeError::InvalidInput(format!(
                    "statement.step_count does not fit usize: {}",
                    self.instance.statement.step_count
                ))
            })?;
        let mut ob_state: Option<OutputSumcheckStateVars> = None;
        let mut ob_inc_total_degree_bound: Option<usize> = None;

        let step = &step_proof.fold;
        let proof = &step.ccs_proof;

        // Bind step_idx into the public steps_digest transcript.
        steps_tr
            .append_message(
                cs,
                b"step/idx",
                &(step_idx as u64).to_le_bytes(),
                &format!("steps_digest_step_{step_idx}_idx"),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;

        // 0) absorb_step_memory (Route-A memory meta).
        if mem_enabled {
            self.absorb_step_memory(cs, tr, None, Some(steps_tr), step_idx, step_public)?;
        } else {
            if !step_public.lut_insts.is_empty() || !step_public.mem_insts.is_empty() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "step {step_idx}: mem_enabled=false requires empty StepInstanceBundle lut/mem vectors"
                )));
            }
            if !step_proof.mem.cpu_me_claims_val.is_empty()
                || !step_proof.mem.proofs.is_empty()
                || !step_proof.mem.shout_addr_pre.claimed_sums.is_empty()
                || !step_proof.mem.shout_addr_pre.round_polys.is_empty()
                || !step_proof.mem.shout_addr_pre.r_addr.is_empty()
            {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "step {step_idx}: mem_enabled=false requires empty mem sidecars"
                )));
            }
            if step_proof.val_fold.is_some() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "step {step_idx}: mem_enabled=false forbids val lane proofs"
                )));
            }
            self.absorb_step_memory_empty(cs, tr, step_idx)?;
            self.absorb_step_memory_empty(cs, steps_tr, step_idx)?;
        }

        // Output binding (last step only): happens immediately after `absorb_step_memory` and before
        // CCS header/instance binding, matching `neo_fold::shard::fold_shard_verify_impl`.
        if include_ob {
            if !mem_enabled {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "step {step_idx}: output binding requires mem_enabled=true"
                )));
            }
            let Some((cfg, ob_proof, program_io_claims)) = output_binding else {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "step {step_idx}: output binding enabled but missing witness config/proof"
                )));
            };
            if cfg.mem_idx >= step_public.mem_insts.len() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "step {step_idx}: output binding mem_idx out of range (mem_idx={}, mem_insts={})",
                    cfg.mem_idx,
                    step_public.mem_insts.len()
                )));
            }
            let mem_inst = step_public
                .mem_insts
                .get(cfg.mem_idx)
                .ok_or_else(|| SpartanBridgeError::InvalidInput("output binding mem_idx out of range".into()))?;
            let expected_k = 1usize
                .checked_shl(cfg.num_bits as u32)
                .ok_or_else(|| SpartanBridgeError::InvalidInput("output binding: 2^num_bits overflow".into()))?;
            if mem_inst.k != expected_k {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "step {step_idx}: output binding cfg.num_bits implies k={expected_k}, but mem_inst.k={}",
                    mem_inst.k
                )));
            }
            let ell_addr = mem_inst.d * mem_inst.ell;
            if ell_addr != cfg.num_bits {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "step {step_idx}: output binding cfg.num_bits={}, but twist_layout.ell_addr={ell_addr}",
                    cfg.num_bits
                )));
            }
            ob_inc_total_degree_bound = Some(2 + ell_addr);

            tr.append_message(
                cs,
                b"shard/output_binding_start",
                &(step_idx as u64).to_le_bytes(),
                &format!("step_{step_idx}_output_binding_start"),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;
            tr.append_u64s(
                cs,
                b"output_binding/mem_idx",
                &[cfg.mem_idx as u64],
                &format!("step_{step_idx}_output_binding_mem_idx"),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;
            tr.append_u64s(
                cs,
                b"output_binding/num_bits",
                &[cfg.num_bits as u64],
                &format!("step_{step_idx}_output_binding_num_bits"),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;

            let state = self.verify_output_sumcheck_rounds_get_state(
                cs,
                tr,
                step_idx,
                cfg,
                program_io_claims,
                &ob_proof.output_sc.round_polys,
            )?;
            ob_state = Some(state);
        } else if output_binding.is_some() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "step {step_idx}: unexpected output_binding data on non-final step"
            )));
        }

        // 1) bind header + mcs + polynomial
        let mcs_vars = self.bind_header_and_mcs_with_digest(cs, tr, mcs_inst, step_idx)?;

        // Bind the MCS instance into the public steps_digest transcript using the same vars.
        {
            use p3_field::PrimeField64;

            let x_vals: Vec<CircuitF> = mcs_inst
                .x
                .iter()
                .map(|f| CircuitF::from(f.as_canonical_u64()))
                .collect();
            steps_tr
                .append_fields_vars(
                    cs,
                    b"x",
                    mcs_vars.x.as_slice(),
                    x_vals.as_slice(),
                    &format!("steps_digest_step_{step_idx}_x"),
                )
                .map_err(SpartanBridgeError::BellpepperError)?;
            steps_tr
                .append_u64s(
                    cs,
                    b"m_in",
                    &[mcs_inst.m_in as u64],
                    &format!("steps_digest_step_{step_idx}_m_in"),
                )
                .map_err(SpartanBridgeError::BellpepperError)?;
            let c_vals: Vec<CircuitF> = mcs_inst
                .c
                .data
                .iter()
                .map(|f| CircuitF::from(f.as_canonical_u64()))
                .collect();
            steps_tr
                .append_fields_vars(
                    cs,
                    b"c_data",
                    mcs_vars.c_data.as_slice(),
                    c_vals.as_slice(),
                    &format!("steps_digest_step_{step_idx}_c_data"),
                )
                .map_err(SpartanBridgeError::BellpepperError)?;
        }

        // Optional step-linking constraints across the boundary (step_idx-1 -> step_idx).
        if let Some(prev) = prev_mcs_vars {
            let prev_x = prev.x.as_slice();
            for (pair_idx, (prev_idx, next_idx)) in self.step_linking.iter().enumerate() {
                if *prev_idx >= prev_x.len() {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "step {step_idx}: step_linking pair {pair_idx} prev_idx out of range (prev_idx={}, prev_x.len()={})",
                        prev_idx,
                        prev_x.len()
                    )));
                }
                if *next_idx >= mcs_vars.x.len() {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "step {step_idx}: step_linking pair {pair_idx} next_idx out of range (next_idx={}, cur_x.len()={})",
                        next_idx,
                        mcs_vars.x.len()
                    )));
                }
                cs.enforce(
                    || format!("step_{step_idx}_step_link_pair_{pair_idx}"),
                    |lc| lc + prev_x[*prev_idx],
                    |lc| lc + CS::one(),
                    |lc| lc + mcs_vars.x[*next_idx],
                );
            }
        }

        // 2) bind ME inputs (v2 framing)
        self.bind_me_inputs_v2(cs, tr, me_inputs_vals, me_inputs_vars, step_idx)?;

        // 3) sample challenges from transcript and enforce equality to proof.challenges_public
        let (alpha_tr, beta_a_tr, beta_r_tr, gamma_tr) =
            self.sample_pi_ccs_challenges_from_transcript(cs, tr, step_idx)?;

        if proof.challenges_public.alpha.len() != alpha_tr.len()
            || proof.challenges_public.beta_a.len() != beta_a_tr.len()
            || proof.challenges_public.beta_r.len() != beta_r_tr.len()
        {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "step {step_idx}: Π-CCS challenge length mismatch"
            )));
        }

        let mut alpha_vars = Vec::with_capacity(alpha_tr.len());
        for (i, &k) in proof.challenges_public.alpha.iter().enumerate() {
            let v = helpers::alloc_k_from_neo(cs, k, &format!("step_{step_idx}_alpha_{i}"))?;
            helpers::enforce_k_eq(cs, &v, &alpha_tr[i], &format!("step_{step_idx}_alpha_fs_{i}"));
            alpha_vars.push(v);
        }

        let mut beta_a_vars = Vec::with_capacity(beta_a_tr.len());
        for (i, &k) in proof.challenges_public.beta_a.iter().enumerate() {
            let v = helpers::alloc_k_from_neo(cs, k, &format!("step_{step_idx}_beta_a_{i}"))?;
            helpers::enforce_k_eq(cs, &v, &beta_a_tr[i], &format!("step_{step_idx}_beta_a_fs_{i}"));
            beta_a_vars.push(v);
        }

        let mut beta_r_vars = Vec::with_capacity(beta_r_tr.len());
        for (i, &k) in proof.challenges_public.beta_r.iter().enumerate() {
            let v = helpers::alloc_k_from_neo(cs, k, &format!("step_{step_idx}_beta_r_{i}"))?;
            helpers::enforce_k_eq(cs, &v, &beta_r_tr[i], &format!("step_{step_idx}_beta_r_fs_{i}"));
            beta_r_vars.push(v);
        }

        let gamma_var = helpers::alloc_k_from_neo(cs, proof.challenges_public.gamma, &format!("step_{step_idx}_gamma"))?;
        helpers::enforce_k_eq(cs, &gamma_var, &gamma_tr, &format!("step_{step_idx}_gamma_fs"));

        // 4) derive claimed initial sum T and bind optional sc_initial_sum
        let t_var = self.claimed_initial_sum_gadget(
            cs,
            step_idx,
            &alpha_vars,
            &proof.challenges_public.alpha,
            &gamma_var,
            proof.challenges_public.gamma,
            me_inputs_vars,
            me_inputs_vals,
        )?;
        if let Some(sc_initial) = proof.sc_initial_sum {
            let sc_initial_var =
                helpers::alloc_k_from_neo(cs, sc_initial, &format!("step_{step_idx}_sc_initial_sum_binding"))?;
            helpers::enforce_k_eq(cs, &sc_initial_var, &t_var, &format!("step_{step_idx}_T_matches_sc0"));
        }

        let t_val = self.claimed_initial_sum_value_host(
            &proof.challenges_public.alpha,
            proof.challenges_public.gamma,
            me_inputs_vals,
        );
        tr.append_fields_vars(
            cs,
            b"sumcheck/initial_sum",
            &[t_var.c0, t_var.c1],
            &[
                helpers::neo_f_to_circuit(&t_val.as_coeffs()[0]),
                helpers::neo_f_to_circuit(&t_val.as_coeffs()[1]),
            ],
            &format!("step_{step_idx}_sumcheck_initial_sum"),
        )
        .map_err(SpartanBridgeError::BellpepperError)?;

        // 5) sample r_cycle (Route-A memory uses it; still sampled even when mem is empty).
        let (r_cycle_vars, r_cycle_vals) = self
            .sample_ext_point(
                cs,
                tr,
                b"route_a/r_cycle",
                b"route_a/cycle/0",
                b"route_a/cycle/1",
                self.ell_n,
                &format!("step_{step_idx}_r_cycle"),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;

        // 6) Route-A memory pre-time proofs (Shout/Twist addr-pre) advance the transcript
        // before the batched time sumcheck.
        let n_lut = step_public.lut_insts.len();
        let n_mem = step_public.mem_insts.len();
        let mut shout_claimed_sums_vars_opt: Option<Vec<KNumVar>> = None;
        let mut shout_finals_vars_opt: Option<Vec<KNumVar>> = None;
        let mut shout_r_addr_vars_opt: Option<Vec<KNumVar>> = None;
        let mut twist_r_addr_vars_by_mem: Vec<Vec<KNumVar>> = Vec::new();
        let mut twist_finals_vars_by_mem: Vec<Vec<KNumVar>> = Vec::new();
        if mem_enabled {
            // The proof vector is a concatenation of Shout proofs (one per LUT instance) followed by Twist proofs.
            if step_proof.mem.proofs.len() != n_lut + n_mem {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "step {step_idx}: mem.proofs.len() mismatch (got {}, expected n_lut+n_mem={})",
                    step_proof.mem.proofs.len(),
                    n_lut + n_mem
                )));
            }

            // Shout addr-pre (fixed-count profile).
            if n_lut == 0 {
                if !step_proof.mem.shout_addr_pre.claimed_sums.is_empty()
                    || !step_proof.mem.shout_addr_pre.round_polys.is_empty()
                    || !step_proof.mem.shout_addr_pre.r_addr.is_empty()
                {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "step {step_idx}: shout_addr_pre must be empty when there are no LUT instances"
                    )));
                }
            } else {
                // Require uniform ell_addr across LUTs (matches native verifier).
                let mut ell_addr: Option<usize> = None;
                for (idx, lut_inst) in step_public.lut_insts.iter().enumerate() {
                    let inst_ell_addr = lut_inst.d * lut_inst.ell;
                    if let Some(prev) = ell_addr {
                        if prev != inst_ell_addr {
                            return Err(SpartanBridgeError::InvalidInput(format!(
                                "step {step_idx}: shout addr-pre requires uniform ell_addr; got {prev} (lut_idx=0) vs {inst_ell_addr} (lut_idx={idx})"
                            )));
                        }
                    } else {
                        ell_addr = Some(inst_ell_addr);
                    }
                }
                let ell_addr = ell_addr.unwrap_or(0);

                let total_shout_lanes: usize = step_public
                    .lut_insts
                    .iter()
                    .map(|inst| inst.lanes.max(1))
                    .sum();
                let proof_ap = &step_proof.mem.shout_addr_pre;
                if proof_ap.claimed_sums.len() != total_shout_lanes {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "step {step_idx}: shout_addr_pre claimed_sums.len()={}, expected total_shout_lanes={total_shout_lanes}",
                        proof_ap.claimed_sums.len()
                    )));
                }
                if proof_ap.round_polys.len() != total_shout_lanes {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "step {step_idx}: shout_addr_pre round_polys.len()={}, expected total_shout_lanes={total_shout_lanes}",
                        proof_ap.round_polys.len()
                    )));
                }
                if proof_ap.r_addr.len() != ell_addr {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "step {step_idx}: shout_addr_pre r_addr.len()={}, expected ell_addr={ell_addr}",
                        proof_ap.r_addr.len()
                    )));
                }

                let labels_all: Vec<&[u8]> = vec![b"shout/addr_pre".as_slice(); total_shout_lanes];
                tr.append_message(
                    cs,
                    b"shout/addr_pre_time/step_idx",
                    &(step_idx as u64).to_le_bytes(),
                    &format!("step_{step_idx}_shout_addr_pre_step_idx"),
                )
                .map_err(SpartanBridgeError::BellpepperError)?;

                // bind_batched_claim_sums(tr, b"shout/addr_pre_time/claimed_sums", ...)
                tr.append_message(
                    cs,
                    b"shout/addr_pre_time/claimed_sums",
                    &(total_shout_lanes as u64).to_le_bytes(),
                    &format!("step_{step_idx}_shout_addr_pre_claimed_sums_len"),
                )
                .map_err(SpartanBridgeError::BellpepperError)?;

                let mut claimed_sums_vars: Vec<KNumVar> = Vec::with_capacity(total_shout_lanes);
                for (i, &sum) in proof_ap.claimed_sums.iter().enumerate() {
                    let var = helpers::alloc_k_from_neo(cs, sum, &format!("step_{step_idx}_shout_addr_pre_sum_{i}"))?;
                    tr.append_message(
                        cs,
                        b"addr_batch/label",
                        labels_all[i],
                        &format!("step_{step_idx}_shout_addr_pre_bind_{i}"),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;
                    tr.append_message(
                        cs,
                        b"addr_batch/idx",
                        &(i as u64).to_le_bytes(),
                        &format!("step_{step_idx}_shout_addr_pre_bind_{i}"),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;
                    let coeffs = sum.as_coeffs();
                    tr.append_fields_vars(
                        cs,
                        b"addr_batch/claimed_sum",
                        &[var.c0, var.c1],
                        &[
                            helpers::neo_f_to_circuit(&coeffs[0]),
                            helpers::neo_f_to_circuit(&coeffs[1]),
                        ],
                        &format!("step_{step_idx}_shout_addr_pre_bind_{i}_sum"),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;
                    claimed_sums_vars.push(var);
                }

                // Allocate per-LUT sumcheck rounds as K variables.
                let mut round_polys_vars: Vec<Vec<Vec<KNumVar>>> = Vec::with_capacity(total_shout_lanes);
                for (lane_idx, rounds) in proof_ap.round_polys.iter().enumerate() {
                    if rounds.len() != ell_addr {
                        return Err(SpartanBridgeError::InvalidInput(format!(
                            "step {step_idx}: shout_addr_pre round_polys[{lane_idx}].len()={}, expected ell_addr={ell_addr}",
                            rounds.len()
                        )));
                    }
                    let mut claim_vars = Vec::with_capacity(ell_addr);
                    for (round_idx, round_poly) in rounds.iter().enumerate() {
                        let mut coeff_vars = Vec::with_capacity(round_poly.len());
                        for (coeff_idx, &coeff) in round_poly.iter().enumerate() {
                            coeff_vars.push(helpers::alloc_k_from_neo(
                                cs,
                                coeff,
                                &format!("step_{step_idx}_shout_addr_pre_lane{lane_idx}_round{round_idx}_coeff{coeff_idx}"),
                            )?);
                        }
                        claim_vars.push(coeff_vars);
                    }
                    round_polys_vars.push(claim_vars);
                }

                let degree_bounds = vec![2usize; total_shout_lanes];
                let (r_addr, finals) = verify_batched_sumcheck_rounds_ds(
                    cs,
                    tr,
                    b"shout/addr_pre_time",
                    step_idx,
                    self.delta,
                    degree_bounds.as_slice(),
                    claimed_sums_vars.as_slice(),
                    labels_all.as_slice(),
                    round_polys_vars.as_slice(),
                    proof_ap.round_polys.as_slice(),
                    &format!("step_{step_idx}_shout_addr_pre"),
                )
                .map_err(SpartanBridgeError::BellpepperError)?;

                // Enforce r_addr matches proof.
                for (i, &want) in proof_ap.r_addr.iter().enumerate() {
                    let want_var = helpers::alloc_k_from_neo(cs, want, &format!("step_{step_idx}_shout_addr_pre_r_addr_{i}"))?;
                    helpers::enforce_k_eq(cs, &want_var, &r_addr[i], &format!("step_{step_idx}_shout_addr_pre_r_addr_fs_{i}"));
                }

                shout_claimed_sums_vars_opt = Some(claimed_sums_vars);
                shout_finals_vars_opt = Some(finals);
                shout_r_addr_vars_opt = Some(r_addr);
            }

            // Twist addr-pre (two-claim batch per mem instance).
            let proof_offset = n_lut;
            let zero = helpers::k_zero(cs, &format!("step_{step_idx}_twist_addr_pre_zero"))?;
            for mem_idx in 0..n_mem {
                let mem_inst = step_public
                    .mem_insts
                    .get(mem_idx)
                    .ok_or_else(|| SpartanBridgeError::InvalidInput(format!("step {step_idx}: missing mem_insts[{mem_idx}]")))?;
                let proof_twist = match step_proof.mem.proofs.get(proof_offset + mem_idx) {
                    Some(neo_fold::shard::MemOrLutProof::Twist(p)) => p,
                    _ => {
                        return Err(SpartanBridgeError::InvalidInput(format!(
                            "step {step_idx}: expected Twist proof at mem_proofs[{}]",
                            proof_offset + mem_idx
                        )))
                    }
                };

                if proof_twist.addr_pre.claimed_sums.len() != 2 || proof_twist.addr_pre.round_polys.len() != 2 {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "step {step_idx}: twist addr_pre malformed at mem_idx={mem_idx}"
                    )));
                }

                // Enforce claimed_sums == [0,0] in-circuit (mirrors native verifier).
                let mut claimed_sums_vars = Vec::with_capacity(2);
                for claim_idx in 0..2 {
                    let sum = proof_twist.addr_pre.claimed_sums[claim_idx];
                    let var = helpers::alloc_k_from_neo(cs, sum, &format!("step_{step_idx}_twist_addr_pre_sum_{mem_idx}_{claim_idx}"))?;
                    helpers::enforce_k_eq(
                        cs,
                        &var,
                        &zero,
                        &format!("step_{step_idx}_twist_addr_pre_sum_{mem_idx}_{claim_idx}_is_zero"),
                    );
                    claimed_sums_vars.push(var);
                }

                let labels: [&[u8]; 2] = [b"twist/read_addr_pre".as_slice(), b"twist/write_addr_pre".as_slice()];
                let degree_bounds = vec![2usize, 2usize];
                tr.append_message(
                    cs,
                    b"twist/addr_pre_time/claim_idx",
                    &(mem_idx as u64).to_le_bytes(),
                    &format!("step_{step_idx}_twist_addr_pre_claim_idx_{mem_idx}"),
                )
                .map_err(SpartanBridgeError::BellpepperError)?;

                // bind_batched_claim_sums(tr, b"twist/addr_pre_time/claimed_sums", ...)
                tr.append_message(
                    cs,
                    b"twist/addr_pre_time/claimed_sums",
                    &(2u64).to_le_bytes(),
                    &format!("step_{step_idx}_twist_addr_pre_claimed_sums_len_{mem_idx}"),
                )
                .map_err(SpartanBridgeError::BellpepperError)?;
                for i in 0..2 {
                    tr.append_message(
                        cs,
                        b"addr_batch/label",
                        labels[i],
                        &format!("step_{step_idx}_twist_addr_pre_bind_{mem_idx}_{i}"),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;
                    tr.append_message(
                        cs,
                        b"addr_batch/idx",
                        &(i as u64).to_le_bytes(),
                        &format!("step_{step_idx}_twist_addr_pre_bind_{mem_idx}_{i}"),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;
                    tr.append_fields_vars(
                        cs,
                        b"addr_batch/claimed_sum",
                        &[claimed_sums_vars[i].c0, claimed_sums_vars[i].c1],
                        &[CircuitF::from(0u64), CircuitF::from(0u64)],
                        &format!("step_{step_idx}_twist_addr_pre_bind_{mem_idx}_{i}_sum"),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;
                }

                // Allocate round polys (two claims).
                let mut round_polys_vars: Vec<Vec<Vec<KNumVar>>> = Vec::with_capacity(2);
                for claim_idx in 0..2 {
                    let rounds = &proof_twist.addr_pre.round_polys[claim_idx];
                    let mut claim_vars = Vec::with_capacity(rounds.len());
                    for (round_idx, round_poly) in rounds.iter().enumerate() {
                        let mut coeff_vars = Vec::with_capacity(round_poly.len());
                        for (coeff_idx, &coeff) in round_poly.iter().enumerate() {
                            coeff_vars.push(helpers::alloc_k_from_neo(
                                cs,
                                coeff,
                                &format!("step_{step_idx}_twist_addr_pre_mem{mem_idx}_claim{claim_idx}_round{round_idx}_coeff{coeff_idx}"),
                            )?);
                        }
                        claim_vars.push(coeff_vars);
                    }
                    round_polys_vars.push(claim_vars);
                }

                let (r_addr, finals) = verify_batched_sumcheck_rounds_ds(
                    cs,
                    tr,
                    b"twist/addr_pre_time",
                    mem_idx,
                    self.delta,
                    degree_bounds.as_slice(),
                    claimed_sums_vars.as_slice(),
                    &labels,
                    round_polys_vars.as_slice(),
                    proof_twist.addr_pre.round_polys.as_slice(),
                    &format!("step_{step_idx}_twist_addr_pre_{mem_idx}"),
                )
                .map_err(SpartanBridgeError::BellpepperError)?;

                // Enforce r_addr matches proof.
                if proof_twist.addr_pre.r_addr.len() != r_addr.len() {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "step {step_idx}: twist addr_pre r_addr.len() mismatch at mem_idx={mem_idx}"
                    )));
                }
                for (i, &want) in proof_twist.addr_pre.r_addr.iter().enumerate() {
                    let want_var = helpers::alloc_k_from_neo(
                        cs,
                        want,
                        &format!("step_{step_idx}_twist_addr_pre_r_addr_{mem_idx}_{i}"),
                    )?;
                    helpers::enforce_k_eq(
                        cs,
                        &want_var,
                        &r_addr[i],
                        &format!("step_{step_idx}_twist_addr_pre_r_addr_fs_{mem_idx}_{i}"),
                    );
                }

                // Host-side check: ensure the address point length matches the instance geometry.
                let ell_addr = mem_inst.d * mem_inst.ell;
                if proof_twist.addr_pre.r_addr.len() != ell_addr {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "step {step_idx}: twist addr_pre r_addr.len()={}, expected ell_addr={ell_addr} at mem_idx={mem_idx}",
                        proof_twist.addr_pre.r_addr.len()
                    )));
                }

                twist_r_addr_vars_by_mem.push(r_addr);
                twist_finals_vars_by_mem.push(finals);
            }
        }

        // 7) verify Route-A batched time (CCS/time + memory claims + optional output binding inc_total).
        let metas = neo_fold::memory_sidecar::claim_plan::RouteATimeClaimPlan::time_claim_metas_for_step(
            step_public,
            self.d_sc,
            ob_inc_total_degree_bound,
        );
        let labels: Vec<&'static [u8]> = metas.iter().map(|m| m.label).collect();
        let degree_bounds: Vec<usize> = metas.iter().map(|m| m.degree_bound).collect();
        let claim_is_dynamic: Vec<bool> = metas.iter().map(|m| m.is_dynamic).collect();
        let bt_out = verify_route_a_batched_time_step(
            cs,
            tr,
            step_idx,
            self.ell_n,
            self.delta,
            &t_var,
            t_val,
            &step_proof.batched_time,
            labels.as_slice(),
            degree_bounds.as_slice(),
            claim_is_dynamic.as_slice(),
            &format!("step_{step_idx}_route_a_batched_time"),
        )?;

        // 7) CCS structure consistency with batched time proof (time rounds + time challenges).
        if proof.sumcheck_rounds.len() < self.ell_n {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "step {step_idx}: Π-CCS sumcheck_rounds too short"
            )));
        }
        for round_idx in 0..self.ell_n {
            let round = proof.sumcheck_rounds[round_idx].as_slice();
            let want = step_proof
                .batched_time
                .round_polys
                .get(0)
                .and_then(|v| v.get(round_idx))
                .ok_or_else(|| SpartanBridgeError::InvalidInput(format!("step {step_idx}: missing batched_time round")))?;
            if round != want.as_slice() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "step {step_idx}: CCS time round poly mismatch at round {round_idx}"
                )));
            }
        }

        if proof.sumcheck_challenges.len() < self.ell_n {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "step {step_idx}: Π-CCS sumcheck_challenges too short"
            )));
        }
        for i in 0..self.ell_n {
            let want = helpers::alloc_k_from_neo(cs, proof.sumcheck_challenges[i], &format!("step_{step_idx}_r_time_{i}"))?;
            helpers::enforce_k_eq(cs, &want, &bt_out.r_time[i], &format!("step_{step_idx}_r_time_fs_{i}"));
        }

        // 8) CCS output r must equal r_time (Route-A shared challenges).
        if step.ccs_out.is_empty() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "step {step_idx}: Π-CCS output is empty"
            )));
        }
        if step.ccs_out[0].r.len() != self.ell_n {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "step {step_idx}: ccs_out[0].r length mismatch"
            )));
        }
        for i in 0..self.ell_n {
            let out_r = helpers::alloc_k_from_neo(cs, step.ccs_out[0].r[i], &format!("step_{step_idx}_ccs_out0_r_{i}"))?;
            helpers::enforce_k_eq(cs, &out_r, &bt_out.r_time[i], &format!("step_{step_idx}_ccs_out0_r_eq_r_time_{i}"));
        }

        // 9) Ajtai rounds (continuing transcript after batched time).
        //
        // Claim 0 is always `ccs/time` (see `RouteATimeClaimPlan`), so we use final_values[0]
        // as the claimed sum for the Ajtai-only continuation.
        if bt_out.final_values.is_empty() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "step {step_idx}: batched time produced no final values"
            )));
        }

        let mut after_time = t_val;
        for round_idx in 0..self.ell_n {
            after_time = Self::eval_poly_k(&step_proof.batched_time.round_polys[0][round_idx], proof.sumcheck_challenges[round_idx]);
        }

        let ajtai_rounds = &proof.sumcheck_rounds[self.ell_n..];
        let ajtai_rounds_vars: Vec<Vec<KNumVar>> = ajtai_rounds
            .iter()
            .enumerate()
            .map(|(round_idx, round_poly)| {
                round_poly
                    .iter()
                    .enumerate()
                    .map(|(coeff_idx, &coeff)| {
                        helpers::alloc_k_from_neo(cs, coeff, &format!("step_{step_idx}_ajtai_round_{round_idx}_coeff_{coeff_idx}"))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;

        let (ajtai_chals, running_sum) = verify_sumcheck_rounds_ds(
            cs,
            tr,
            b"ccs/ajtai",
            step_idx,
            self.d_sc,
            self.delta,
            &bt_out.final_values[0],
            after_time,
            &ajtai_rounds_vars,
            ajtai_rounds,
            &format!("step_{step_idx}_ccs_ajtai"),
        )
        .map_err(SpartanBridgeError::BellpepperError)?;

        // 10) Enforce stored challenges + final match transcript-derived.
        if ajtai_chals.len() != self.ell_d {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "step {step_idx}: ajtai_chals length mismatch"
            )));
        }
        if proof.sumcheck_challenges.len() != self.ell_n + self.ell_d {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "step {step_idx}: sumcheck_challenges.len() mismatch"
            )));
        }
        for i in 0..self.ell_d {
            let want = helpers::alloc_k_from_neo(
                cs,
                proof.sumcheck_challenges[self.ell_n + i],
                &format!("step_{step_idx}_ajtai_chal_{i}"),
            )?;
            helpers::enforce_k_eq(cs, &want, &ajtai_chals[i], &format!("step_{step_idx}_ajtai_chal_fs_{i}"));
        }
        let final_sum_expected =
            helpers::alloc_k_from_neo(cs, proof.sumcheck_final, &format!("step_{step_idx}_final_sum"))?;
        helpers::enforce_k_eq(cs, &running_sum, &final_sum_expected, &format!("step_{step_idx}_final_sum_matches_scalar"));

        // 11) Allocate CCS outputs and RLC parent once (shared across terminal identity + RLC/DEC).
        let mut ccs_out_vars: Vec<MeInstanceVars> = Vec::with_capacity(step.ccs_out.len());
        for (i, child) in step.ccs_out.iter().enumerate() {
            ccs_out_vars.push(self.alloc_me_instance_vars(cs, child, &format!("step_{step_idx}_ccs_out_{i}"))?);
        }
        let rlc_parent_vars = self.alloc_me_instance_vars(cs, &step.rlc_parent, &format!("step_{step_idx}_rlc_parent"))?;

        let ccs_out_y_vars: Vec<Vec<Vec<KNumVar>>> = ccs_out_vars.iter().map(|v| v.y.clone()).collect();
        self.verify_terminal_identity(
            cs,
            step_idx,
            step,
            proof,
            me_inputs_vals,
            me_inputs_vars,
            &running_sum,
            &ccs_out_y_vars,
            &alpha_vars,
            &beta_a_vars,
            &beta_r_vars,
            &gamma_var,
            &bt_out.r_time,
            &ajtai_chals,
        )?;

        // 12) Header digest binding (transcript-derived).
        if proof.header_digest.len() != 32 {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "step {step_idx}: header_digest must be 32 bytes"
            )));
        }
        let dig = tr
            .digest32(cs, &format!("step_{step_idx}_ccs_header_digest"))
            .map_err(SpartanBridgeError::BellpepperError)?;
        let header_digest_bytes = self.alloc_digest32_bytes_from_limbs(
            cs,
            &dig,
            Some(proof.header_digest.as_slice()),
            &format!("step_{step_idx}_header_digest"),
        )?;

        // --------------------------------------------------------------------
        // Phase 1 (mem-enabled): advance transcript through Twist val-eval batch.
        //
        // Native verifier does this inside `verify_route_a_memory_step` *before* Π_RLC/Π_DEC,
        // so ρ sampling depends on it.
        // --------------------------------------------------------------------
        let mut r_val_vars: Option<Vec<KNumVar>> = None;
        let mut val_eval_claimed_sums_vars: Option<Vec<KNumVar>> = None;
        let mut val_eval_claimed_sums_vals: Option<Vec<neo_math::K>> = None;
        let mut val_eval_finals_vars: Option<Vec<KNumVar>> = None;
        let mut cpu_me_digest_val_bytes: Option<Vec<AllocatedNum<CircuitF>>> = None;
        let mut cpu_me_claims_val_vars: Option<Vec<MeInstanceVars>> = None;

        if mem_enabled && !step_public.mem_insts.is_empty() {
            let n_mem = step_public.mem_insts.len();
            let n_lut = step_public.lut_insts.len();
            let proof_offset = n_lut;
            let has_prev = step_idx > 0;

            let plan = neo_fold::memory_sidecar::claim_plan::TwistValEvalClaimPlan::build(step_public.mem_insts.iter(), has_prev);
            let claim_count = plan.claim_count;

            let mut bind_kinds: Vec<u8> = Vec::with_capacity(claim_count);
            let mut claimed_sums_vals: Vec<neo_math::K> = Vec::with_capacity(claim_count);
            let mut claimed_sums_vars: Vec<KNumVar> = Vec::with_capacity(claim_count);
            let mut round_polys_vals: Vec<Vec<Vec<neo_math::K>>> = Vec::with_capacity(claim_count);
            let mut round_polys_vars: Vec<Vec<Vec<KNumVar>>> = Vec::with_capacity(claim_count);

            let mut claim_idx = 0usize;
            for mem_idx in 0..n_mem {
                let twist_proof = match step_proof.mem.proofs.get(proof_offset + mem_idx) {
                    Some(neo_fold::shard::MemOrLutProof::Twist(p)) => p,
                    _ => {
                        return Err(SpartanBridgeError::InvalidInput(format!(
                            "step {step_idx}: expected Twist proof at mem_proofs[{}] for val-eval",
                            proof_offset + mem_idx
                        )))
                    }
                };
                let val = twist_proof
                    .val_eval
                    .as_ref()
                    .ok_or_else(|| SpartanBridgeError::InvalidInput(format!("step {step_idx}: missing Twist val_eval at mem_idx={mem_idx}")))?;

                // LT claim
                bind_kinds.push(plan.bind_tags[claim_idx]);
                claimed_sums_vals.push(val.claimed_inc_sum_lt);
                claimed_sums_vars.push(helpers::alloc_k_from_neo(
                    cs,
                    val.claimed_inc_sum_lt,
                    &format!("step_{step_idx}_val_eval_claim_{claim_idx}_sum"),
                )?);
                round_polys_vals.push(val.rounds_lt.clone());
                round_polys_vars.push(
                    val.rounds_lt
                        .iter()
                        .enumerate()
                        .map(|(round_idx, round)| {
                            round
                                .iter()
                                .enumerate()
                                .map(|(coeff_idx, &coeff)| {
                                    helpers::alloc_k_from_neo(
                                        cs,
                                        coeff,
                                        &format!("step_{step_idx}_val_eval_claim_{claim_idx}_round_{round_idx}_coeff_{coeff_idx}"),
                                    )
                                })
                                .collect::<Result<Vec<_>>>()
                        })
                        .collect::<Result<Vec<_>>>()?,
                );
                claim_idx += 1;

                // TOTAL claim
                bind_kinds.push(plan.bind_tags[claim_idx]);
                claimed_sums_vals.push(val.claimed_inc_sum_total);
                claimed_sums_vars.push(helpers::alloc_k_from_neo(
                    cs,
                    val.claimed_inc_sum_total,
                    &format!("step_{step_idx}_val_eval_claim_{claim_idx}_sum"),
                )?);
                round_polys_vals.push(val.rounds_total.clone());
                round_polys_vars.push(
                    val.rounds_total
                        .iter()
                        .enumerate()
                        .map(|(round_idx, round)| {
                            round
                                .iter()
                                .enumerate()
                                .map(|(coeff_idx, &coeff)| {
                                    helpers::alloc_k_from_neo(
                                        cs,
                                        coeff,
                                        &format!("step_{step_idx}_val_eval_claim_{claim_idx}_round_{round_idx}_coeff_{coeff_idx}"),
                                    )
                                })
                                .collect::<Result<Vec<_>>>()
                        })
                        .collect::<Result<Vec<_>>>()?,
                );
                claim_idx += 1;

                // Optional rollover (prev-total) claim.
                if has_prev {
                    let prev_total = val
                        .claimed_prev_inc_sum_total
                        .ok_or_else(|| SpartanBridgeError::InvalidInput(format!(
                            "step {step_idx}: missing Twist claimed_prev_inc_sum_total at mem_idx={mem_idx}"
                        )))?;
                    let prev_rounds = val
                        .rounds_prev_total
                        .as_ref()
                        .ok_or_else(|| SpartanBridgeError::InvalidInput(format!(
                            "step {step_idx}: missing Twist rounds_prev_total at mem_idx={mem_idx}"
                        )))?;

                    bind_kinds.push(plan.bind_tags[claim_idx]);
                    claimed_sums_vals.push(prev_total);
                    claimed_sums_vars.push(helpers::alloc_k_from_neo(
                        cs,
                        prev_total,
                        &format!("step_{step_idx}_val_eval_claim_{claim_idx}_sum"),
                    )?);
                    round_polys_vals.push(prev_rounds.clone());
                    round_polys_vars.push(
                        prev_rounds
                            .iter()
                            .enumerate()
                            .map(|(round_idx, round)| {
                                round
                                    .iter()
                                    .enumerate()
                                    .map(|(coeff_idx, &coeff)| {
                                        helpers::alloc_k_from_neo(
                                            cs,
                                            coeff,
                                            &format!("step_{step_idx}_val_eval_claim_{claim_idx}_round_{round_idx}_coeff_{coeff_idx}"),
                                        )
                                    })
                                    .collect::<Result<Vec<_>>>()
                            })
                            .collect::<Result<Vec<_>>>()?,
                    );
                    claim_idx += 1;
                } else if val.claimed_prev_inc_sum_total.is_some() || val.rounds_prev_total.is_some() {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "step {step_idx}: rollover fields present but prev_step is None (mem_idx={mem_idx})"
                    )));
                }
            }

            if claim_idx != claim_count {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "step {step_idx}: val-eval claim count mismatch (built {claim_idx}, plan {claim_count})"
                )));
            }

            tr.append_message(
                cs,
                b"twist/val_eval/batch_start",
                &(n_mem as u64).to_le_bytes(),
                &format!("step_{step_idx}_val_eval_batch_start"),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;
            tr.append_message(
                cs,
                b"twist/val_eval/step_idx",
                &(step_idx as u64).to_le_bytes(),
                &format!("step_{step_idx}_val_eval_step_idx"),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;

            // bind_twist_val_eval_claim_sums(tr, &bind_claims)
            tr.append_message(
                cs,
                b"twist/val_eval/claimed_sums_len",
                &(claim_count as u64).to_le_bytes(),
                &format!("step_{step_idx}_val_eval_claims_len"),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;
            for i in 0..claim_count {
                tr.append_message(
                    cs,
                    b"twist/val_eval/claim_idx",
                    &(i as u64).to_le_bytes(),
                    &format!("step_{step_idx}_val_eval_claim_{i}"),
                )
                .map_err(SpartanBridgeError::BellpepperError)?;
                tr.append_message(
                    cs,
                    b"twist/val_eval/claim_kind",
                    &[bind_kinds[i]],
                    &format!("step_{step_idx}_val_eval_claim_{i}"),
                )
                .map_err(SpartanBridgeError::BellpepperError)?;

                let sum_val = claimed_sums_vals[i];
                let coeffs = sum_val.as_coeffs();
                tr.append_fields_vars(
                    cs,
                    b"twist/val_eval/claimed_sum",
                    &[claimed_sums_vars[i].c0, claimed_sums_vars[i].c1],
                    &[
                        helpers::neo_f_to_circuit(&coeffs[0]),
                        helpers::neo_f_to_circuit(&coeffs[1]),
                    ],
                    &format!("step_{step_idx}_val_eval_claim_{i}_sum"),
                )
                .map_err(SpartanBridgeError::BellpepperError)?;
            }

            let (r_val, finals) = verify_batched_sumcheck_rounds_ds(
                cs,
                tr,
                b"twist/val_eval_batch",
                step_idx,
                self.delta,
                plan.degree_bounds.as_slice(),
                claimed_sums_vars.as_slice(),
                plan.labels.as_slice(),
                round_polys_vars.as_slice(),
                round_polys_vals.as_slice(),
                &format!("step_{step_idx}_val_eval_batch"),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;
            if r_val.len() != self.ell_n {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "step {step_idx}: twist val-eval r_val.len()={}, expected ell_n={}",
                    r_val.len(),
                    self.ell_n
                )));
            }
            r_val_vars = Some(r_val);
            val_eval_claimed_sums_vars = Some(claimed_sums_vars);
            val_eval_claimed_sums_vals = Some(claimed_sums_vals);
            val_eval_finals_vars = Some(finals);

            tr.append_message(cs, b"twist/val_eval/batch_done", &[], &format!("step_{step_idx}_val_eval_batch_done"))
                .map_err(SpartanBridgeError::BellpepperError)?;

            // Compute the transcript-derived `fold_digest` used for CPU ME openings at r_val:
            // `tr.fork(b"cpu_bus/me_digest_val").digest32()`.
            let mut fork = tr
                .fork(cs, b"cpu_bus/me_digest_val", &format!("step_{step_idx}_cpu_bus_me_digest_val_fork"))
                .map_err(SpartanBridgeError::BellpepperError)?;
            let limbs = fork
                .digest32(cs, &format!("step_{step_idx}_cpu_bus_me_digest_val"))
                .map_err(SpartanBridgeError::BellpepperError)?;
            let hint_bytes = {
                let mut out = [0u8; 32];
                for i in 0..4 {
                    let limb = limbs[i].get_value().unwrap_or(CircuitF::from(0u64));
                    let u = limb.to_canonical_u64();
                    out[i * 8..(i + 1) * 8].copy_from_slice(&u.to_le_bytes());
                }
                out
            };
            let bytes = self.alloc_digest32_bytes_from_limbs(
                cs,
                &limbs,
                Some(&hint_bytes),
                &format!("step_{step_idx}_cpu_bus_me_digest_val_bytes"),
            )?;
            cpu_me_digest_val_bytes = Some(bytes);

            // Allocate CPU ME claims at r_val (cur + optional prev) once and reuse them in:
            // - Route-A memory terminal checks (Phase 1 completion)
            // - Val lane Π_RLC → Π_DEC
            let mut vars: Vec<MeInstanceVars> = Vec::with_capacity(step_proof.mem.cpu_me_claims_val.len());
            for (i, me) in step_proof.mem.cpu_me_claims_val.iter().enumerate() {
                vars.push(self.alloc_me_instance_vars(cs, me, &format!("step_{step_idx}_cpu_me_val_{i}"))?);
            }
            cpu_me_claims_val_vars = Some(vars);
        } else {
            // No mem instances => no val-eval transcript region.
            if !step_proof.mem.cpu_me_claims_val.is_empty() || step_proof.val_fold.is_some() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "step {step_idx}: unexpected val-lane artifacts with no mem instances"
                )));
            }
        }

        // --------------------------------------------------------------------
        // Phase 1 completion: enforce Route-A memory terminal checks (Twist/Shout).
        // --------------------------------------------------------------------
        if mem_enabled {
            // Build shout circuit data if LUT instances exist.
            let shout_pre = if n_lut == 0 {
                None
            } else {
                let claimed_sums_vars = shout_claimed_sums_vars_opt
                    .as_ref()
                    .ok_or_else(|| SpartanBridgeError::InvalidInput(format!("step {step_idx}: internal error: missing shout_claimed_sums_vars")))?;
                let finals_vars = shout_finals_vars_opt
                    .as_ref()
                    .ok_or_else(|| SpartanBridgeError::InvalidInput(format!("step {step_idx}: internal error: missing shout_finals_vars")))?;
                let r_addr_vars = shout_r_addr_vars_opt
                    .as_ref()
                    .ok_or_else(|| SpartanBridgeError::InvalidInput(format!("step {step_idx}: internal error: missing shout_r_addr_vars")))?;
                Some(ShoutAddrPreCircuitData {
                    claimed_sums_vars: claimed_sums_vars.as_slice(),
                    claimed_sums_vals: step_proof.mem.shout_addr_pre.claimed_sums.as_slice(),
                    finals_vars: finals_vars.as_slice(),
                    r_addr_vars: r_addr_vars.as_slice(),
                    r_addr_vals: step_proof.mem.shout_addr_pre.r_addr.as_slice(),
                })
            };

            // Build per-mem Twist addr-pre circuit data.
            if twist_r_addr_vars_by_mem.len() != n_mem || twist_finals_vars_by_mem.len() != n_mem {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "step {step_idx}: internal error: twist addr-pre vars len mismatch"
                )));
            }
            let mut twist_pre_data: Vec<TwistAddrPreCircuitData<'_>> = Vec::with_capacity(n_mem);
            for mem_idx in 0..n_mem {
                let proof_twist = match step_proof.mem.proofs.get(n_lut + mem_idx) {
                    Some(neo_fold::shard::MemOrLutProof::Twist(p)) => p,
                    _ => {
                        return Err(SpartanBridgeError::InvalidInput(format!(
                            "step {step_idx}: expected Twist proof at mem_idx={mem_idx}"
                        )));
                    }
                };
                twist_pre_data.push(TwistAddrPreCircuitData {
                    r_addr_vars: twist_r_addr_vars_by_mem[mem_idx].as_slice(),
                    r_addr_vals: proof_twist.addr_pre.r_addr.as_slice(),
                    finals_vars: twist_finals_vars_by_mem[mem_idx].as_slice(),
                });
            }

            // Build val-eval circuit data if mem instances exist.
            let val_eval = if n_mem == 0 {
                None
            } else {
                let r_val = r_val_vars.as_ref().ok_or_else(|| {
                    SpartanBridgeError::InvalidInput(format!("step {step_idx}: internal error: missing r_val_vars"))
                })?;
                let claimed_sums_vars = val_eval_claimed_sums_vars.as_ref().ok_or_else(|| {
                    SpartanBridgeError::InvalidInput(format!("step {step_idx}: internal error: missing val_eval_claimed_sums_vars"))
                })?;
                let claimed_sums_vals = val_eval_claimed_sums_vals.as_ref().ok_or_else(|| {
                    SpartanBridgeError::InvalidInput(format!("step {step_idx}: internal error: missing val_eval_claimed_sums_vals"))
                })?;
                let finals_vars = val_eval_finals_vars.as_ref().ok_or_else(|| {
                    SpartanBridgeError::InvalidInput(format!("step {step_idx}: internal error: missing val_eval_finals_vars"))
                })?;

                // r_val must match the CPU ME claim r vector.
                let r_val_vals = step_proof
                    .mem
                    .cpu_me_claims_val
                    .first()
                    .ok_or_else(|| SpartanBridgeError::InvalidInput(format!("step {step_idx}: missing cpu_me_claims_val for val-eval")))?;
                Some(TwistValEvalCircuitData {
                    r_val_vars: r_val.as_slice(),
                    r_val_vals: r_val_vals.r.as_slice(),
                    claimed_sums_vars: claimed_sums_vars.as_slice(),
                    claimed_sums_vals: claimed_sums_vals.as_slice(),
                    finals_vars: finals_vars.as_slice(),
                })
            };

            let cpu_me_vars_slice = cpu_me_claims_val_vars.as_ref().map(|v| v.as_slice());
            let twist_time_openings = route_a_memory_terminal::verify_route_a_memory_terminal_step(
                cs,
                self.ccs_m,
                self.ccs_t,
                self.base_b,
                self.delta,
                step_idx,
                step_public,
                prev_step_public,
                &mcs_vars,
                prev_mcs_vars.map(|p| p.c_data.as_slice()),
                &ccs_out_vars[0],
                &step.ccs_out[0],
                bt_out.r_time.as_slice(),
                step.ccs_out[0].r.as_slice(),
                r_cycle_vars.as_slice(),
                r_cycle_vals.as_slice(),
                bt_out.claimed_sums_vars.as_slice(),
                step_proof.batched_time.claimed_sums.as_slice(),
                bt_out.final_values.as_slice(),
                shout_pre,
                twist_pre_data.as_slice(),
                val_eval,
                cpu_me_vars_slice,
                step_proof.mem.cpu_me_claims_val.as_slice(),
            )?;

            // Output binding: enforce inc_total terminal and final output equation (last step only).
            if include_ob {
                let Some((cfg, _ob_proof, _program_io_claims)) = output_binding else {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "step {step_idx}: output binding enabled but missing witness config/proof"
                    )));
                };
                let ob = ob_state.take().ok_or_else(|| {
                    SpartanBridgeError::InvalidInput(format!("step {step_idx}: internal error: missing ob_state"))
                })?;

                let inc_idx = bt_out
                    .final_values
                    .len()
                    .checked_sub(1)
                    .ok_or_else(|| SpartanBridgeError::InvalidInput("output binding: missing inc_total claim".into()))?;
                if step_proof
                    .batched_time
                    .labels
                    .get(inc_idx)
                    .copied()
                    != Some(neo_fold::output_binding::OB_INC_TOTAL_LABEL)
                {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "step {step_idx}: output binding claim not last"
                    )));
                }
                let inc_total_claim_val = *step_proof
                    .batched_time
                    .claimed_sums
                    .get(inc_idx)
                    .ok_or_else(|| SpartanBridgeError::InvalidInput("output binding: missing inc_total claimed_sum".into()))?;
                let inc_total_claim_var = bt_out
                    .claimed_sums_vars
                    .get(inc_idx)
                    .ok_or_else(|| SpartanBridgeError::InvalidInput("output binding: missing inc_total claimed_sum var".into()))?;

                let twist_open = twist_time_openings.get(cfg.mem_idx).ok_or_else(|| {
                    SpartanBridgeError::InvalidInput(format!(
                        "step {step_idx}: missing twist_time_openings for mem_idx={}",
                        cfg.mem_idx
                    ))
                })?;

                // inc_terminal = Σ_lane has_write * inc_at_write_addr * eq_bits_prod(wa_bits, r')
                let mut inc_terminal_val = NeoK::ZERO;
                let mut inc_terminal = helpers::k_zero(cs, &format!("step_{step_idx}_ob_inc_acc0"))?;
                for (lane_idx, lane) in twist_open.lanes.iter().enumerate() {
                    let (eq_wa, eq_wa_val) = self.eq_points(
                        cs,
                        step_idx,
                        lane.wa_bits.as_slice(),
                        ob.r_prime_vars.as_slice(),
                        lane.wa_bits_vals.as_slice(),
                        ob.r_prime_vals.as_slice(),
                        &format!("step_{step_idx}_ob_inc_eq_wa_lane{lane_idx}"),
                    )?;
                    let (t0, t0_val) = helpers::k_mul_with_hint(
                        cs,
                        &lane.has_write,
                        lane.has_write_val,
                        &lane.inc_at_write_addr,
                        lane.inc_at_write_addr_val,
                        self.delta,
                        &format!("step_{step_idx}_ob_inc_t0_lane{lane_idx}"),
                    )?;
                    let (term, term_val) = helpers::k_mul_with_hint(
                        cs,
                        &t0,
                        t0_val,
                        &eq_wa,
                        eq_wa_val,
                        self.delta,
                        &format!("step_{step_idx}_ob_inc_term_lane{lane_idx}"),
                    )?;
                    inc_terminal_val += term_val;
                    inc_terminal = k_add_raw(
                        cs,
                        &inc_terminal,
                        &term,
                        Some(KNum::<CircuitF>::from_neo_k(inc_terminal_val)),
                        &format!("step_{step_idx}_ob_inc_acc_lane{lane_idx}"),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;
                }
                helpers::enforce_k_eq(
                    cs,
                    &inc_terminal,
                    bt_out
                        .final_values
                        .get(inc_idx)
                        .ok_or_else(|| SpartanBridgeError::InvalidInput("output binding: missing inc_total final_value var".into()))?,
                    &format!("step_{step_idx}_ob_inc_total_final_eq"),
                );

                // expected_out = eq_eval * io_mask_eval * (val_init(r') + inc_total_claim - val_io_eval)
                let mem_inst = step_public.mem_insts.get(cfg.mem_idx).ok_or_else(|| {
                    SpartanBridgeError::InvalidInput("output binding mem_idx out of range".into())
                })?;
                let (val_init, val_init_val) = route_a_memory_terminal::eval_mem_init_at_r_addr(
                    cs,
                    self.base_b,
                    self.delta,
                    &mem_inst.init,
                    mem_inst.k,
                    ob.r_prime_vars.as_slice(),
                    ob.r_prime_vals.as_slice(),
                    &format!("step_{step_idx}_ob_val_init"),
                )?;
                let val_final_val = val_init_val + inc_total_claim_val;
                let val_final = k_add_raw(
                    cs,
                    &val_init,
                    inc_total_claim_var,
                    Some(KNum::<CircuitF>::from_neo_k(val_final_val)),
                    &format!("step_{step_idx}_ob_val_final"),
                )
                .map_err(SpartanBridgeError::BellpepperError)?;

                // diff = val_final - val_io_eval (linear, allocate with hint and constrain diff + val_io = val_final)
                let diff_val = val_final_val - ob.val_io_eval_val;
                let diff = alloc_k(
                    cs,
                    Some(KNum::<CircuitF>::from_neo_k(diff_val)),
                    &format!("step_{step_idx}_ob_diff"),
                )
                .map_err(SpartanBridgeError::BellpepperError)?;
                cs.enforce(
                    || format!("step_{step_idx}_ob_diff_c0"),
                    |lc| lc + diff.c0 + ob.val_io_eval_var.c0,
                    |lc| lc + CS::one(),
                    |lc| lc + val_final.c0,
                );
                cs.enforce(
                    || format!("step_{step_idx}_ob_diff_c1"),
                    |lc| lc + diff.c1 + ob.val_io_eval_var.c1,
                    |lc| lc + CS::one(),
                    |lc| lc + val_final.c1,
                );

                let (t1, t1_val) = helpers::k_mul_with_hint(
                    cs,
                    &ob.eq_eval_var,
                    ob.eq_eval_val,
                    &ob.io_mask_eval_var,
                    ob.io_mask_eval_val,
                    self.delta,
                    &format!("step_{step_idx}_ob_t1"),
                )?;
                let (expected_out, _expected_out_val) = helpers::k_mul_with_hint(
                    cs,
                    &t1,
                    t1_val,
                    &diff,
                    diff_val,
                    self.delta,
                    &format!("step_{step_idx}_ob_expected"),
                )?;
                helpers::enforce_k_eq(
                    cs,
                    &expected_out,
                    &ob.output_final_var,
                    &format!("step_{step_idx}_ob_output_final_eq"),
                );
            }
        }

        // --------------------------------------------------------------------
        // Main lane Π_RLC → Π_DEC (includes transcript-derived ρ sampling).
        // --------------------------------------------------------------------
        let dec_children_vars = {
            let mut cs_main = cs.namespace(|| format!("step_{step_idx}_lane_main"));
            self.verify_rlc(
                &mut cs_main,
                tr,
                b"main",
                step_idx,
                &header_digest_bytes,
                &step.rlc_parent,
                &rlc_parent_vars,
                &step.ccs_out,
                &ccs_out_vars,
                &step.rlc_rhos,
            )?;

            let mut dec_children_vars = Vec::with_capacity(step.dec_children.len());
            for (i, child) in step.dec_children.iter().enumerate() {
                dec_children_vars.push(
                    self.alloc_me_instance_vars(&mut cs_main, child, &format!("step_{step_idx}_dec_child_{i}"))?,
                );
            }
            self.verify_dec(
                &mut cs_main,
                step_idx,
                &step.rlc_parent,
                &rlc_parent_vars,
                &step.dec_children,
                &dec_children_vars,
            )?;
            dec_children_vars
        };

        // --------------------------------------------------------------------
        // Val lane Π_RLC → Π_DEC (Twist r_val obligations).
        // --------------------------------------------------------------------
        let mut val_children_vars: Vec<MeInstanceVars> = Vec::new();
        match (
            step_proof.mem.cpu_me_claims_val.is_empty(),
            step_proof.val_fold.as_ref(),
        ) {
            (true, None) => {}
            (true, Some(_)) => {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "step {step_idx}: unexpected val_fold proof (no r_val ME claims)"
                )));
            }
            (false, None) => {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "step {step_idx}: missing val_fold proof (have r_val ME claims)"
                )));
            }
            (false, Some(val_fold)) => {
                let r_val = r_val_vars.as_ref().ok_or_else(|| {
                    SpartanBridgeError::InvalidInput(format!(
                        "step {step_idx}: internal error: missing r_val_vars for val lane"
                    ))
                })?;
                let fd_bytes = cpu_me_digest_val_bytes.as_ref().ok_or_else(|| {
                    SpartanBridgeError::InvalidInput(format!(
                        "step {step_idx}: internal error: missing cpu_me_digest_val_bytes for val lane"
                    ))
                })?;

                tr.append_message(
                    cs,
                    b"fold/val_lane_start",
                    &(step_idx as u64).to_le_bytes(),
                    &format!("step_{step_idx}_val_lane_start"),
                )
                .map_err(SpartanBridgeError::BellpepperError)?;

                // Allocate ME inputs for the val lane (CPU ME at r_val for cur + optional prev).
                let cpu_me_vars = cpu_me_claims_val_vars.as_ref().ok_or_else(|| {
                    SpartanBridgeError::InvalidInput(format!(
                        "step {step_idx}: internal error: missing cpu_me_claims_val_vars for val lane"
                    ))
                })?;

                // Enforce r == r_val for all CPU ME claims in this lane.
                for (me_idx, me_var) in cpu_me_vars.iter().enumerate() {
                    if me_var.r.len() != r_val.len() {
                        return Err(SpartanBridgeError::InvalidInput(format!(
                            "step {step_idx}: cpu_me_claims_val[{me_idx}] r.len() mismatch (vars={}, r_val={})",
                            me_var.r.len(),
                            r_val.len()
                        )));
                    }
                    for i in 0..r_val.len() {
                        helpers::enforce_k_eq(
                            cs,
                            &me_var.r[i],
                            &r_val[i],
                            &format!("step_{step_idx}_cpu_me_val_{me_idx}_r_eq_r_val_{i}"),
                        );
                    }
                }

                // Allocate parent + children and verify the val lane RLC/DEC proof.
                let val_parent_vars =
                    self.alloc_me_instance_vars(cs, &val_fold.rlc_parent, &format!("step_{step_idx}_val_rlc_parent"))?;

                let mut val_dec_children_vars = Vec::with_capacity(val_fold.dec_children.len());
                for (i, child) in val_fold.dec_children.iter().enumerate() {
                    val_dec_children_vars.push(self.alloc_me_instance_vars(cs, child, &format!("step_{step_idx}_val_dec_child_{i}"))?);
                }

                {
                    let mut cs_val = cs.namespace(|| format!("step_{step_idx}_lane_val"));
                    self.verify_rlc(
                        &mut cs_val,
                        tr,
                        b"val",
                        step_idx,
                        fd_bytes.as_slice(),
                        &val_fold.rlc_parent,
                        &val_parent_vars,
                        &step_proof.mem.cpu_me_claims_val,
                        cpu_me_vars,
                        &val_fold.rlc_rhos,
                    )?;
                    self.verify_dec(
                        &mut cs_val,
                        step_idx,
                        &val_fold.rlc_parent,
                        &val_parent_vars,
                        &val_fold.dec_children,
                        &val_dec_children_vars,
                    )?;
                }

                val_children_vars = val_dec_children_vars;
            }
        }

        tr.append_message(cs, b"fold/step_done", &(step_idx as u64).to_le_bytes(), &format!("step_{step_idx}_done"))
            .map_err(SpartanBridgeError::BellpepperError)?;

        Ok((dec_children_vars, mcs_vars, val_children_vars))
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
        me_inputs_vars: &[MeInstanceVars],
        me_inputs_vals: &[neo_ccs::MeInstance<neo_ajtai::Commitment, NeoF, neo_math::K>],
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
        if me_inputs_vars.len() != me_inputs_vals.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "claimed_initial_sum_gadget me_inputs length mismatch at step {}: vars={}, vals={}",
                step_idx,
                me_inputs_vars.len(),
                me_inputs_vals.len()
            )));
        }

        let k_total = 1 + me_inputs_vars.len(); // 1 MCS + |ME|
        if k_total < 2 {
            // No Eval block when k=1 → T = 0.
            return helpers::k_zero(cs, &format!("step_{}_T_zero", step_idx));
        }

        // Build χ_α over Ajtai domain.
        let d_sz = 1usize << alpha_vars.len();
        let mut chi_alpha_vars: Vec<KNumVar> = Vec::with_capacity(d_sz);
        let mut chi_alpha_vals: Vec<NeoK> = Vec::with_capacity(d_sz);

        // Canonical K-constant 1, shared across all χ entries.
        let one_const = helpers::k_one(cs, &format!("step_{}_chi_alpha_one", step_idx))?;

        // χ_α[ρ] = ∏_bit (α_bit if ρ_bit=1 else 1-α_bit)
        for rho in 0..d_sz {
            // Start with w = 1 in both the native and in-circuit representation.
            let mut w_val = NeoK::ONE;
            let mut w_var = one_const.clone();

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
                        &format!("step_{}_chi_alpha_{}_bit{}_factor", step_idx, rho, bit),
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
                        &format!("step_{}_chi_alpha_{}_bit{}_sum", step_idx, rho, bit),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;
                    helpers::enforce_k_eq(
                        cs,
                        &sum_var,
                        &one_const,
                        &format!("step_{}_chi_alpha_{}_bit{}_one_minus", step_idx, rho, bit),
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

        // γ^k_total (used in the weights) must be computed in-circuit.
        let mut gamma_to_k = helpers::k_one(cs, &format!("step_{}_T_gamma_to_k_init", step_idx))?;
        let mut gamma_to_k_val = NeoK::ONE;
        for pow_idx in 0..k_total {
            let (new_gamma_to_k, new_gamma_to_k_val) = helpers::k_mul_with_hint(
                cs,
                &gamma_to_k,
                gamma_to_k_val,
                gamma_var,
                gamma_val,
                self.delta,
                &format!("step_{}_T_gamma_to_k_step_{}", step_idx, pow_idx),
            )?;
            gamma_to_k = new_gamma_to_k;
            gamma_to_k_val = new_gamma_to_k_val;
        }

        // Inner weighted sum over (j, i>=2).
        let t = if me_inputs_vars.is_empty() {
            0
        } else {
            me_inputs_vars[0].y.len()
        };

        let mut inner_val = NeoK::ZERO;
        let mut inner = helpers::k_zero(cs, &format!("step_{}_T_inner_init", step_idx))?;

        for j in 0..t {
            // (γ^k_total)^j – shared across all i for this j, tracked natively and then lifted.
            let mut gamma_k_j_val = NeoK::ONE;
            let mut gamma_k_j = helpers::k_one(cs, &format!("step_{}_T_gamma_k_j_init_j{}", step_idx, j))?;
            for pow_idx in 0..j {
                let (new_gamma_k_j, new_gamma_k_j_val) = helpers::k_mul_with_hint(
                    cs,
                    &gamma_k_j,
                    gamma_k_j_val,
                    &gamma_to_k,
                    gamma_to_k_val,
                    self.delta,
                    &format!("step_{}_T_gamma_k_j_step_j{}_{}", step_idx, j, pow_idx),
                )?;
                gamma_k_j = new_gamma_k_j;
                gamma_k_j_val = new_gamma_k_j_val;
            }

            for (idx, (me_vars, me_vals)) in me_inputs_vars.iter().zip(me_inputs_vals.iter()).enumerate() {
                // me_inputs[idx] corresponds to instance i = idx + 2 in the paper.
                let i_abs = idx + 2;
                let row_vars = &me_vars.y[j];
                let row_vals = &me_vals.y[j];
                let limit = min(d_sz, min(row_vars.len(), row_vals.len()));

                // y_eval = ⟨ y_{(i,j)}, χ_α ⟩
                let mut y_eval_val = NeoK::ZERO;
                let mut y_eval = helpers::k_zero(cs, &format!("step_{}_T_y_eval_j{}_i{}", step_idx, j, i_abs))?;
                for rho in 0..limit {
                    let (prod, prod_val) = helpers::k_mul_with_hint(
                        cs,
                        &row_vars[rho],
                        row_vals[rho],
                        &chi_alpha_vars[rho],
                        chi_alpha_vals[rho],
                        self.delta,
                        &format!("step_{}_T_y_eval_j{}_i{}_rho{}", step_idx, j, i_abs, rho),
                    )?;

                    y_eval_val += prod_val;
                    let y_eval_hint = KNum::<CircuitF>::from_neo_k(y_eval_val);
                    y_eval = k_add_raw(
                        cs,
                        &y_eval,
                        &prod,
                        Some(y_eval_hint),
                        &format!("step_{}_T_y_eval_acc_j{}_i{}_rho{}", step_idx, j, i_abs, rho),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;
                }

                // γ^{i-1}
                let mut gamma_i_val = NeoK::ONE;
                let mut gamma_i = helpers::k_one(cs, &format!("step_{}_T_gamma_i_init_j{}_i{}", step_idx, j, i_abs))?;
                for pow_idx in 0..(i_abs - 1) {
                    let (new_gamma_i, new_gamma_i_val) = helpers::k_mul_with_hint(
                        cs,
                        &gamma_i,
                        gamma_i_val,
                        gamma_var,
                        gamma_val,
                        self.delta,
                        &format!("step_{}_T_gamma_i_step_j{}_i{}_{}", step_idx, j, i_abs, pow_idx),
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

        // T = γ^{k_total} * inner, matching the native `claimed_initial_sum_from_inputs`.
        let (t_var, _t_val) = helpers::k_mul_with_hint(
            cs,
            &gamma_to_k,
            gamma_to_k_val,
            &inner,
            inner_val,
            self.delta,
            &format!("step_{}_T_scale_by_gamma_k", step_idx),
        )?;

        Ok(t_var)
    }

    /// Equality polynomial eq_points over K, using the same formula as the
    /// native `eq_points`: ∏_i [(1-p_i)*(1-q_i) + p_i*q_i].
    ///
    /// This version constrains `eq` in terms of the K variables `p` and `q`,
    /// while using `p_vals`/`q_vals` as native hints for intermediate K ops.
    fn eq_points<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        p: &[KNumVar],
        q: &[KNumVar],
        p_vals: &[neo_math::K],
        q_vals: &[neo_math::K],
        label: &str,
    ) -> Result<(KNumVar, neo_math::K)> {
        use neo_math::K as NeoK;

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

        // eq over empty vectors is 1.
        if p.is_empty() {
            let one_var = helpers::k_one(cs, &format!("step_{}_{}_eq_one", step_idx, label))?;
            return Ok((one_var, NeoK::ONE));
        }

        // Canonical K-constant 1, shared across all coordinates.
        let one_var = helpers::k_one(cs, &format!("step_{}_{}_one_const", step_idx, label))?;

        // acc = 1 in K.
        let mut acc_var = one_var.clone();
        let mut acc_native = NeoK::ONE;

        for i in 0..p.len() {
            let pi_var = &p[i];
            let qi_var = &q[i];
            let pi_val = p_vals[i];
            let qi_val = q_vals[i];

            // 1 - p_i
            let one_minus_pi_val = NeoK::ONE - pi_val;
            let one_minus_pi_hint = KNum::<CircuitF>::from_neo_k(one_minus_pi_val);
            let one_minus_pi = alloc_k(
                cs,
                Some(one_minus_pi_hint),
                &format!("step_{}_{}_one_minus_p_{}", step_idx, label, i),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;

            // Enforce (1 - p_i) + p_i = 1 in K.
            let sum_p_val = one_minus_pi_val + pi_val; // native 1
            let sum_p_hint = KNum::<CircuitF>::from_neo_k(sum_p_val);
            let sum_p = k_add_raw(
                cs,
                &one_minus_pi,
                pi_var,
                Some(sum_p_hint),
                &format!("step_{}_{}_one_minus_p_sum_{}", step_idx, label, i),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;
            helpers::enforce_k_eq(
                cs,
                &sum_p,
                &one_var,
                &format!("step_{}_{}_one_minus_p_check_{}", step_idx, label, i),
            );

            // 1 - q_i
            let one_minus_qi_val = NeoK::ONE - qi_val;
            let one_minus_qi_hint = KNum::<CircuitF>::from_neo_k(one_minus_qi_val);
            let one_minus_qi = alloc_k(
                cs,
                Some(one_minus_qi_hint),
                &format!("step_{}_{}_one_minus_q_{}", step_idx, label, i),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;

            let sum_q_val = one_minus_qi_val + qi_val; // native 1
            let sum_q_hint = KNum::<CircuitF>::from_neo_k(sum_q_val);
            let sum_q = k_add_raw(
                cs,
                &one_minus_qi,
                qi_var,
                Some(sum_q_hint),
                &format!("step_{}_{}_one_minus_q_sum_{}", step_idx, label, i),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;
            helpers::enforce_k_eq(
                cs,
                &sum_q,
                &one_var,
                &format!("step_{}_{}_one_minus_q_check_{}", step_idx, label, i),
            );

            // (1 - p_i)*(1 - q_i)
            let (prod1_var, prod1_val) = helpers::k_mul_with_hint(
                cs,
                &one_minus_pi,
                one_minus_pi_val,
                &one_minus_qi,
                one_minus_qi_val,
                self.delta,
                &format!("step_{}_{}_prod1_{}", step_idx, label, i),
            )?;

            // p_i * q_i
            let (pq_var, pq_val) = helpers::k_mul_with_hint(
                cs,
                pi_var,
                pi_val,
                qi_var,
                qi_val,
                self.delta,
                &format!("step_{}_{}_pq_{}", step_idx, label, i),
            )?;

            // term_i = (1-p_i)*(1-q_i) + p_i*q_i
            let term_val = prod1_val + pq_val;
            let term_hint = KNum::<CircuitF>::from_neo_k(term_val);
            let term_var = k_add_raw(
                cs,
                &prod1_var,
                &pq_var,
                Some(term_hint),
                &format!("step_{}_{}_term_{}", step_idx, label, i),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;

            // acc *= term_i
            let (new_acc_var, new_acc_native) = helpers::k_mul_with_hint(
                cs,
                &acc_var,
                acc_native,
                &term_var,
                term_val,
                self.delta,
                &format!("step_{}_{}_eq_acc_step_{}", step_idx, label, i),
            )?;
            acc_var = new_acc_var;
            acc_native = new_acc_native;
        }

        Ok((acc_var, acc_native))
    }

    /// Recompose a single Ajtai y-row in base-b into a K element:
    /// m = Σ_{ℓ} b^ℓ · y[ℓ], using native K hints for the result and
    /// enforcing linear relations on the limbs.
    fn recompose_y_row_base_b<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        _j: usize,
        y_row_vars: &[KNumVar],
        y_row_vals: &[neo_math::K],
        label: &str,
    ) -> Result<(KNumVar, neo_math::K)> {
        use neo_math::{F as NeoF, K as NeoK};

        let len = core::cmp::min(y_row_vars.len(), y_row_vals.len());
        if len == 0 {
            let zero = helpers::k_zero(cs, &format!("step_{}_{}_empty_row", step_idx, label))?;
            return Ok((zero, NeoK::ZERO));
        }

        // Native recomposition in K: m_native = Σ b^ℓ * y[ℓ].
        let base_native_f: NeoF = NeoF::from_u64(self.base_b as u64);
        let base_native_k: NeoK = NeoK::from(base_native_f);

        let mut pow_k = NeoK::ONE;
        let mut m_native = NeoK::ZERO;
        for ell in 0..len {
            m_native += pow_k * y_row_vals[ell];
            pow_k *= base_native_k;
        }

        // Allocate m with the correct native value as a hint.
        let m_var = helpers::alloc_k_from_neo(cs, m_native, &format!("step_{}_{}_val", step_idx, label))?;

        // Enforce limb-wise base-b recomposition:
        // m.c0 = Σ b^ℓ * y[ℓ].c0,  m.c1 = Σ b^ℓ * y[ℓ].c1.
        let base_circ = CircuitF::from(self.base_b as u64);

        // c0 component
        cs.enforce(
            || format!("step_{}_{}_recompose_c0", step_idx, label),
            |lc| {
                let mut res = lc;
                // ℓ = 0 term has coefficient 1.
                res = res + (CircuitF::from(1u64), y_row_vars[0].c0);
                let mut pow = base_circ;
                for ell in 1..len {
                    res = res + (pow, y_row_vars[ell].c0);
                    pow *= base_circ;
                }
                res
            },
            |lc| lc + CS::one(),
            |lc| lc + m_var.c0,
        );

        // c1 component
        cs.enforce(
            || format!("step_{}_{}_recompose_c1", step_idx, label),
            |lc| {
                let mut res = lc;
                res = res + (CircuitF::from(1u64), y_row_vars[0].c1);
                let mut pow = base_circ;
                for ell in 1..len {
                    res = res + (pow, y_row_vars[ell].c1);
                    pow *= base_circ;
                }
                res
            },
            |lc| lc + CS::one(),
            |lc| lc + m_var.c1,
        );

        Ok((m_var, m_native))
    }

    /// Range product gadget: ∏_{t=-(b-1)}^{b-1} (val - t) over K, using native
    /// K hints and explicit limb-wise linear constraints for (val - t).
    fn range_product<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        val: &KNumVar,
        val_native: neo_math::K,
        label: &str,
    ) -> Result<(KNumVar, neo_math::K)> {
        use neo_math::{F as NeoF, K as NeoK};

        let mut acc_var = helpers::k_one(cs, &format!("step_{}_{}_range_init", step_idx, label))?;
        let mut acc_native = NeoK::ONE;

        let b = self.base_b as i32;
        let minus_one = CircuitF::from(0u64) - CircuitF::from(1u64);

        for t in (-(b - 1))..=(b - 1) {
            let abs = t.abs() as u64;
            let base_f: NeoF = NeoF::from_u64(abs);
            let t_native = if t >= 0 {
                NeoK::from(base_f)
            } else {
                NeoK::from(base_f) * NeoK::from(NeoF::from_u64(0u64)) - NeoK::from(base_f)
            };

            let diff_native = val_native - t_native;

            // Allocate diff with its native value.
            let diff_var = helpers::alloc_k_from_neo(
                cs,
                diff_native,
                &format!("step_{}_{}_val_minus_t_{}", step_idx, label, t),
            )?;

            // Enforce diff = val - t limb-wise in K.
            let t_var = helpers::k_const_from_neo(cs, t_native, &format!("step_{}_{}_t_{}", step_idx, label, t))?;

            // c0: diff.c0 = val.c0 - t.c0
            cs.enforce(
                || format!("step_{}_{}_range_diff_t{}_c0", step_idx, label, t),
                |lc| lc + val.c0 + (minus_one, t_var.c0),
                |lc| lc + CS::one(),
                |lc| lc + diff_var.c0,
            );
            // c1: diff.c1 = val.c1 - t.c1
            cs.enforce(
                || format!("step_{}_{}_range_diff_t{}_c1", step_idx, label, t),
                |lc| lc + val.c1 + (minus_one, t_var.c1),
                |lc| lc + CS::one(),
                |lc| lc + diff_var.c1,
            );

            // acc *= diff with native hints.
            let (new_acc_var, new_acc_native) = helpers::k_mul_with_hint(
                cs,
                &acc_var,
                acc_native,
                &diff_var,
                diff_native,
                self.delta,
                &format!("step_{}_{}_range_acc_{}", step_idx, label, t),
            )?;
            acc_var = new_acc_var;
            acc_native = new_acc_native;
        }

        Ok((acc_var, acc_native))
    }

    /// Evaluate the CCS polynomial f at the given m-values in K, using native K
    /// hints for all intermediate products to keep the witness consistent.
    fn eval_poly_f_in_k<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        m_vals: &[KNumVar],
        m_vals_native: &[neo_math::K],
    ) -> Result<(KNumVar, neo_math::K)> {
        use neo_math::K as NeoK;

        if m_vals.len() != m_vals_native.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "eval_poly_f_in_k length mismatch at step {}: m_vals={}, m_vals_native={}",
                step_idx,
                m_vals.len(),
                m_vals_native.len()
            )));
        }

        // Compute each monomial term_val = ∏_j m_j^{exp_j} with K hints.
        let mut term_vars: Vec<KNumVar> = Vec::with_capacity(self.poly_f.len());
        let mut term_natives: Vec<NeoK> = Vec::with_capacity(self.poly_f.len());

        for (term_idx, term) in self.poly_f.iter().enumerate() {
            let mut term_var = helpers::k_one(cs, &format!("step_{}_F_term{}_init", step_idx, term_idx))?;
            let mut term_native = NeoK::ONE;

            for (var_idx, &exp) in term.exps.iter().enumerate() {
                if exp == 0 {
                    continue;
                }
                let base_var = &m_vals[var_idx];
                let base_native = m_vals_native[var_idx];

                // pow = base^exp
                let mut pow_var = helpers::k_one(
                    cs,
                    &format!("step_{}_F_term{}_var{}_pow_init", step_idx, term_idx, var_idx),
                )?;
                let mut pow_native = NeoK::ONE;
                for e in 0..exp {
                    let (new_pow_var, new_pow_native) = helpers::k_mul_with_hint(
                        cs,
                        &pow_var,
                        pow_native,
                        base_var,
                        base_native,
                        self.delta,
                        &format!("step_{}_F_term{}_var{}_pow_mul{}", step_idx, term_idx, var_idx, e),
                    )?;
                    pow_var = new_pow_var;
                    pow_native = new_pow_native;
                }

                // term *= pow
                let (new_term_var, new_term_native) = helpers::k_mul_with_hint(
                    cs,
                    &term_var,
                    term_native,
                    &pow_var,
                    pow_native,
                    self.delta,
                    &format!("step_{}_F_term{}_var{}_mul", step_idx, term_idx, var_idx),
                )?;
                term_var = new_term_var;
                term_native = new_term_native;
            }

            term_vars.push(term_var);
            term_natives.push(term_native);
        }

        // Native F' value: F'(m) = Σ coeff * term_native.
        let mut F_prime_native = NeoK::ZERO;
        for (term_idx, term) in self.poly_f.iter().enumerate() {
            let coeff_k: NeoK = NeoK::from(term.coeff_native);
            F_prime_native += coeff_k * term_natives[term_idx];
        }

        let F_prime_var = helpers::alloc_k_from_neo(cs, F_prime_native, &format!("step_{}_F_prime", step_idx))?;

        // Enforce F' limb-wise as Σ coeff * term_j.
        cs.enforce(
            || format!("step_{}_F_prime_c0_check", step_idx),
            |lc| {
                let mut res = lc;
                for (term_idx, term) in self.poly_f.iter().enumerate() {
                    let coeff_circ = term.coeff;
                    res = res + (coeff_circ, term_vars[term_idx].c0);
                }
                res
            },
            |lc| lc + CS::one(),
            |lc| lc + F_prime_var.c0,
        );

        cs.enforce(
            || format!("step_{}_F_prime_c1_check", step_idx),
            |lc| {
                let mut res = lc;
                for (term_idx, term) in self.poly_f.iter().enumerate() {
                    let coeff_circ = term.coeff;
                    res = res + (coeff_circ, term_vars[term_idx].c1);
                }
                res
            },
            |lc| lc + CS::one(),
            |lc| lc + F_prime_var.c1,
        );

        Ok((F_prime_var, F_prime_native))
    }

    /// Verify Π-CCS terminal identity for a step:
    /// sumcheck_final == rhs_terminal_identity_paper_exact(α,β,γ,r',α', outputs, inputs.r).
    fn verify_terminal_identity<CS: ConstraintSystem<CircuitF>>(
        &self,
        cs: &mut CS,
        step_idx: usize,
        step: &FoldStep,
        proof: &neo_reductions::PiCcsProof,
        me_inputs_vals: &[neo_ccs::MeInstance<neo_ajtai::Commitment, NeoF, neo_math::K>],
        me_inputs_vars: &[MeInstanceVars],
        sumcheck_final: &KNumVar,
        out_y_vars: &[Vec<Vec<KNumVar>>],
        alpha_vars: &[KNumVar],
        beta_a_vars: &[KNumVar],
        beta_r_vars: &[KNumVar],
        gamma_var: &KNumVar,
        r_prime_vars: &[KNumVar],
        alpha_prime_vars: &[KNumVar],
    ) -> Result<()> {
        use neo_math::{D, K as NeoK};

        // Outputs y' for this step.
        let out_me = &step.ccs_out;
        if out_me.is_empty() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "Terminal identity at step {}: empty outputs",
                step_idx
            )));
        }

        let ell_n = proof.challenges_public.beta_r.len();
        let ell_d = proof.challenges_public.beta_a.len();
        if proof.sumcheck_challenges.len() != ell_n + ell_d {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "step {step_idx}: sumcheck_challenges.len()={}, expected ell_n+ell_d={}",
                proof.sumcheck_challenges.len(),
                ell_n + ell_d
            )));
        }
        let (r_prime_vals, alpha_prime_vals) = proof.sumcheck_challenges.split_at(ell_n);

        // --- Scalar equality polynomials eq((α',r'),β) and eq((α',r'),(α,r)) ---
        //
        // Computed in-circuit over K using eq_points, with native K values only
        // used as hints for k_mul_with_hint.

        // FS-derived variables must match proof-advertised lengths.
        if alpha_prime_vars.len() != alpha_prime_vals.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "step {step_idx}: alpha_prime_vars length mismatch (vars={}, vals={})",
                alpha_prime_vars.len(),
                alpha_prime_vals.len()
            )));
        }
        if r_prime_vars.len() != r_prime_vals.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "step {step_idx}: r_prime_vars length mismatch (vars={}, vals={})",
                r_prime_vars.len(),
                r_prime_vals.len()
            )));
        }
        if beta_a_vars.len() != proof.challenges_public.beta_a.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "step {step_idx}: beta_a_vars length mismatch (vars={}, vals={})",
                beta_a_vars.len(),
                proof.challenges_public.beta_a.len()
            )));
        }
        if beta_r_vars.len() != proof.challenges_public.beta_r.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "step {step_idx}: beta_r_vars length mismatch (vars={}, vals={})",
                beta_r_vars.len(),
                proof.challenges_public.beta_r.len()
            )));
        }
        if alpha_vars.len() != proof.challenges_public.alpha.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "step {step_idx}: alpha_vars length mismatch (vars={}, vals={})",
                alpha_vars.len(),
                proof.challenges_public.alpha.len()
            )));
        }

        // eq((α',r'), β) = eq(α', β_a) * eq(r', β_r) in K.
        let (eq_alpha_prime_beta_a, eq_alpha_prime_beta_a_native) = self.eq_points(
            cs,
            step_idx,
            alpha_prime_vars,
            beta_a_vars,
            alpha_prime_vals,
            &proof.challenges_public.beta_a,
            "eq_alpha_prime_beta_a",
        )?;

        let (eq_r_prime_beta_r, eq_r_prime_beta_r_native) = self.eq_points(
            cs,
            step_idx,
            r_prime_vars,
            beta_r_vars,
            r_prime_vals,
            &proof.challenges_public.beta_r,
            "eq_r_prime_beta_r",
        )?;

        let (eq_aprp_beta, eq_aprp_beta_native) = helpers::k_mul_with_hint(
            cs,
            &eq_alpha_prime_beta_a,
            eq_alpha_prime_beta_a_native,
            &eq_r_prime_beta_r,
            eq_r_prime_beta_r_native,
            self.delta,
            &format!("step_{}_eq_aprp_beta", step_idx),
        )?;

        // eq((α',r'),(α,r)) if we have ME inputs; else 0 (Eval' block vanishes).
        let (eq_aprp_ar, eq_aprp_ar_native) = if let Some(first_input) = me_inputs_vals.first() {
            let first_input_vars = me_inputs_vars.first().ok_or_else(|| {
                SpartanBridgeError::InvalidInput(format!(
                    "step {step_idx}: me_inputs_vars empty but me_inputs_vals non-empty"
                ))
            })?;

            let (eq_alpha_prime_alpha, eq_alpha_prime_alpha_native) = self.eq_points(
                cs,
                step_idx,
                alpha_prime_vars,
                alpha_vars,
                alpha_prime_vals,
                &proof.challenges_public.alpha,
                "eq_alpha_prime_alpha",
            )?;

            let (eq_r_prime_r, eq_r_prime_r_native) = self.eq_points(
                cs,
                step_idx,
                r_prime_vars,
                &first_input_vars.r,
                r_prime_vals,
                &first_input.r,
                "eq_r_prime_r",
            )?;

            helpers::k_mul_with_hint(
                cs,
                &eq_alpha_prime_alpha,
                eq_alpha_prime_alpha_native,
                &eq_r_prime_r,
                eq_r_prime_r_native,
                self.delta,
                &format!("step_{}_eq_aprp_ar", step_idx),
            )?
        } else {
            // No ME inputs ⇒ Eval' block vanishes; force eq((α',r'),(α,r)) = 0 so
            // that the whole Eval' contribution is zero.
            let zero_var = helpers::k_zero(cs, &format!("step_{}_eq_aprp_ar_zero", step_idx))?;
            (zero_var, NeoK::ZERO)
        };

        // --- Allocate γ and precompute γ^k_total in K ---
        let gamma_val = proof.challenges_public.gamma;

        let k_total = out_me.len();
        // Compute γ^k_total in-circuit (must not be a free witness).
        let mut gamma_k_total = helpers::k_one(cs, &format!("step_{}_gamma_k_total_init", step_idx))?;
        let mut gamma_k_total_val = neo_math::K::ONE;
        for pow_idx in 0..k_total {
            let (new_gamma_k_total, new_gamma_k_total_val) = helpers::k_mul_with_hint(
                cs,
                &gamma_k_total,
                gamma_k_total_val,
                gamma_var,
                gamma_val,
                self.delta,
                &format!("step_{}_gamma_k_total_mul_{}", step_idx, pow_idx),
            )?;
            gamma_k_total = new_gamma_k_total;
            gamma_k_total_val = new_gamma_k_total_val;
        }

        // --- F' from first output's y'[i=1] in-circuit ---
        //
        // Recompose m_j from Ajtai digits with base-b using only the first D
        // digits, then evaluate f via poly_f in K.
        let t = out_me[0].y.len();
        let (F_prime, F_prime_native) = if t == 0 {
            (
                helpers::k_zero(cs, &format!("step_{}_F_prime_zero", step_idx))?,
                NeoK::ZERO,
            )
        } else {
            // Use the shared y-table allocation for the first output.
            let first_out_y = &out_y_vars[0];
            let d_pad = first_out_y[0].len();
            let d_ring = D.min(d_pad);

            // Native y-table for the first output (for K hints).
            let first_out_y_vals = &out_me[0].y;

            let mut m_vals: Vec<KNumVar> = Vec::with_capacity(t);
            let mut m_vals_native: Vec<NeoK> = Vec::with_capacity(t);
            for j in 0..t {
                if first_out_y[j].len() != d_pad {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "Terminal identity at step {}: inconsistent y row length in output 0 (j={})",
                        step_idx, j
                    )));
                }
                if first_out_y_vals[j].len() != d_pad {
                    return Err(SpartanBridgeError::InvalidInput(format!(
                        "Terminal identity at step {}: inconsistent y row length in output 0 values (j={})",
                        step_idx, j
                    )));
                }
                let (m_j, m_j_native) = self.recompose_y_row_base_b(
                    cs,
                    step_idx,
                    j,
                    &first_out_y[j][..d_ring],
                    &first_out_y_vals[j][..d_ring],
                    &format!("step_{}_F_m_j{}", step_idx, j),
                )?;
                m_vals.push(m_j);
                m_vals_native.push(m_j_native);
            }

            self.eval_poly_f_in_k(cs, step_idx, &m_vals, &m_vals_native)?
        };

        // --- χ_{α'} table in K for Ajtai domain ---
        //
        // χ_{α'}[ρ] = ∏_bit (α'_bit if ρ_bit=1 else 1-α'_bit), with explicit
        // K equality constraints using native K hints (mirrors χ_α gadget).
        let d_sz = 1usize << alpha_prime_vals.len();
        let mut chi_alpha_prime: Vec<KNumVar> = Vec::with_capacity(d_sz);
        let mut chi_alpha_prime_vals: Vec<NeoK> = Vec::with_capacity(d_sz);

        let one_const = helpers::k_one(cs, &format!("step_{}_chi_alpha_prime_one", step_idx))?;
        for rho in 0..d_sz {
            let mut w_val = NeoK::ONE;
            let mut w_var = one_const.clone();

            for (bit, (a_var, &a_val)) in alpha_prime_vars
                .iter()
                .zip(alpha_prime_vals.iter())
                .enumerate()
            {
                let bit_is_one = ((rho >> bit) & 1) == 1;
                // Factor value in K.
                let (factor_var, factor_val) = if bit_is_one {
                    (a_var.clone(), a_val)
                } else {
                    // factor = 1 - α'_bit, enforced via factor + α'_bit = 1 in K.
                    let factor_val = NeoK::ONE - a_val;
                    let factor_hint = KNum::<CircuitF>::from_neo_k(factor_val);
                    let factor_var = alloc_k(
                        cs,
                        Some(factor_hint),
                        &format!("step_{}_chi_alpha_prime_{}_bit{}_factor", step_idx, rho, bit),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;

                    // sum = factor + α'_bit with native hint, enforce sum == 1.
                    let sum_val = factor_val + a_val;
                    let sum_hint = KNum::<CircuitF>::from_neo_k(sum_val);
                    let sum_var = k_add_raw(
                        cs,
                        &factor_var,
                        a_var,
                        Some(sum_hint),
                        &format!("step_{}_chi_alpha_prime_{}_bit{}_sum", step_idx, rho, bit),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;
                    helpers::enforce_k_eq(
                        cs,
                        &sum_var,
                        &one_const,
                        &format!("step_{}_chi_alpha_prime_{}_bit{}_one_minus", step_idx, rho, bit),
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
                    &format!("step_{}_chi_alpha_prime_{}_bit{}", step_idx, rho, bit),
                )?;
                w_var = new_w_var;
                w_val = new_w_val;
            }

            chi_alpha_prime.push(w_var);
            chi_alpha_prime_vals.push(w_val);
        }

        // --- Σ γ^i · N_i' over outputs, in K ---
        //
        // N_i' = ∏_{t} ( ẏ'_{(i,1)}(α') - t ), with ẏ' evaluated at α' as MLE.
        let mut nc_prime_sum = helpers::k_zero(cs, &format!("step_{}_N_prime_sum_init", step_idx))?;
        let mut nc_prime_sum_native = NeoK::ZERO;

        // g = γ^1
        let mut g = gamma_var.clone();
        let mut g_native = gamma_val;
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

            let mut y_eval = helpers::k_zero(cs, &format!("step_{}_N_y_eval_i{}", step_idx, i_idx))?;
            let mut y_eval_val = NeoK::ZERO;
            for rho in 0..limit {
                let y1_val = out_me[i_idx].y[0][rho];
                let chi_val = chi_alpha_prime_vals[rho];
                let (prod_var, prod_val) = helpers::k_mul_with_hint(
                    cs,
                    &y1[rho],
                    y1_val,
                    &chi_alpha_prime[rho],
                    chi_val,
                    self.delta,
                    &format!("step_{}_N_y_eval_i{}_rho{}", step_idx, i_idx, rho),
                )?;
                y_eval_val += prod_val;
                let hint = KNum::<CircuitF>::from_neo_k(y_eval_val);
                y_eval = k_add_raw(
                    cs,
                    &y_eval,
                    &prod_var,
                    Some(hint),
                    &format!("step_{}_N_y_eval_acc_i{}_rho{}", step_idx, i_idx, rho),
                )
                .map_err(SpartanBridgeError::BellpepperError)?;
            }

            let (Ni, Ni_native) = self.range_product(
                cs,
                step_idx,
                &y_eval,
                y_eval_val,
                &format!("step_{}_N_range_i{}", step_idx, i_idx),
            )?;

            let (gNi, gNi_native) = helpers::k_mul_with_hint(
                cs,
                &g,
                g_native,
                &Ni,
                Ni_native,
                self.delta,
                &format!("step_{}_N_weighted_i{}", step_idx, i_idx),
            )?;

            nc_prime_sum_native += gNi_native;
            let nc_hint = KNum::<CircuitF>::from_neo_k(nc_prime_sum_native);
            nc_prime_sum = k_add_raw(
                cs,
                &nc_prime_sum,
                &gNi,
                Some(nc_hint),
                &format!("step_{}_N_acc_i{}", step_idx, i_idx),
            )
            .map_err(SpartanBridgeError::BellpepperError)?;

            // g <- g * γ with hint
            let (new_g, new_g_native) = helpers::k_mul_with_hint(
                cs,
                &g,
                g_native,
                &gamma_var,
                gamma_val,
                self.delta,
                &format!("step_{}_N_gamma_step_i{}", step_idx, i_idx),
            )?;
            g = new_g;
            g_native = new_g_native;
        }

        // --- Eval' block in K ---
        //
        // γ^k · Σ_{j=1,i=2}^{t,k} γ^{i+(j-1)k-1} · E_{(i,j)} with
        // E_{(i,j)} = eq((α',r'),(α,r)) · ẏ'_{(i,j)}(α').
        let mut eval_sum = helpers::k_zero(cs, &format!("step_{}_Eval_sum_init", step_idx))?;
        let mut eval_sum_native = NeoK::ZERO;

        if !me_inputs_vals.is_empty() && k_total >= 2 {
            for j in 0..t {
                // Precompute (γ^k_total)^j once per j and reuse across outputs.
                let mut gamma_k_j_val = NeoK::ONE;
                let mut gamma_k_j = helpers::k_one(cs, &format!("step_{}_Eval_gamma_k_j_init_j{}", step_idx, j))?;
                for pow_idx in 0..j {
                    let (new_gamma_k_j, new_gamma_k_j_val) = helpers::k_mul_with_hint(
                        cs,
                        &gamma_k_j,
                        gamma_k_j_val,
                        &gamma_k_total,
                        gamma_k_total_val,
                        self.delta,
                        &format!("step_{}_Eval_gamma_k_j_step_j{}_{}", step_idx, j, pow_idx),
                    )?;
                    gamma_k_j = new_gamma_k_j;
                    gamma_k_j_val = new_gamma_k_j_val;
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

                    let mut y_eval = helpers::k_zero(cs, &format!("step_{}_Eval_y_eval_j{}_i{}", step_idx, j, i_abs))?;
                    let mut y_eval_val = NeoK::ZERO;
                    for rho in 0..limit {
                        let y_val = out_me[i_abs].y[j][rho];
                        let chi_val = chi_alpha_prime_vals[rho];
                        let (prod_var, prod_val) = helpers::k_mul_with_hint(
                            cs,
                            &row[rho],
                            y_val,
                            &chi_alpha_prime[rho],
                            chi_val,
                            self.delta,
                            &format!("step_{}_Eval_y_eval_j{}_i{}_rho{}", step_idx, j, i_abs, rho),
                        )?;
                        y_eval_val += prod_val;
                        let hint = KNum::<CircuitF>::from_neo_k(y_eval_val);
                        y_eval = k_add_raw(
                            cs,
                            &y_eval,
                            &prod_var,
                            Some(hint),
                            &format!("step_{}_Eval_y_eval_acc_j{}_i{}_rho{}", step_idx, j, i_abs, rho),
                        )
                        .map_err(SpartanBridgeError::BellpepperError)?;
                    }

                    // weight = γ^{i_abs} * (γ^k_total)^j  (0-based indices)
                    let mut gamma_i =
                        helpers::k_one(cs, &format!("step_{}_Eval_gamma_i_init_j{}_i{}", step_idx, j, i_abs))?;
                    let mut gamma_i_val = NeoK::ONE;
                    for pow_idx in 0..i_abs {
                        let (new_gamma_i, new_gamma_i_val) = helpers::k_mul_with_hint(
                            cs,
                            &gamma_i,
                            gamma_i_val,
                            &gamma_var,
                            gamma_val,
                            self.delta,
                            &format!("step_{}_Eval_gamma_i_step_j{}_i{}_{}", step_idx, j, i_abs, pow_idx),
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
                        &format!("step_{}_Eval_weight_j{}_i{}", step_idx, j, i_abs),
                    )?;

                    let (contrib, contrib_val) = helpers::k_mul_with_hint(
                        cs,
                        &weight,
                        weight_val,
                        &y_eval,
                        y_eval_val,
                        self.delta,
                        &format!("step_{}_Eval_contrib_j{}_i{}", step_idx, j, i_abs),
                    )?;

                    eval_sum_native += contrib_val;
                    let eval_hint = KNum::<CircuitF>::from_neo_k(eval_sum_native);
                    eval_sum = k_add_raw(
                        cs,
                        &eval_sum,
                        &contrib,
                        Some(eval_hint),
                        &format!("step_{}_Eval_acc_j{}_i{}", step_idx, j, i_abs),
                    )
                    .map_err(SpartanBridgeError::BellpepperError)?;
                }
            }
        }

        // Assemble RHS in K:
        // v = eq((α',r'), β)·(F' + Σ γ^i N_i') + γ^k · eq((α',r'), (α,r)) · Eval'.
        // F_plus_N = F' + Σ γ^i N'_i with native hint.
        let F_plus_N_native = F_prime_native + nc_prime_sum_native;
        let F_plus_N_hint = KNum::<CircuitF>::from_neo_k(F_plus_N_native);
        let F_plus_N = k_add_raw(
            cs,
            &F_prime,
            &nc_prime_sum,
            Some(F_plus_N_hint),
            &format!("step_{}_RHS_F_plus_N", step_idx),
        )
        .map_err(SpartanBridgeError::BellpepperError)?;

        let (left, left_native) = helpers::k_mul_with_hint(
            cs,
            &eq_aprp_beta,
            eq_aprp_beta_native,
            &F_plus_N,
            F_plus_N_native,
            self.delta,
            &format!("step_{}_RHS_left", step_idx),
        )?;

        // Eval' := γ^k · Σ_{j,i} γ^{i-1 + j·k} · ẏ'_{(i,j)}(α').
        let (eval_sum_scaled, eval_sum_scaled_native) = helpers::k_mul_with_hint(
            cs,
            &gamma_k_total,
            gamma_k_total_val,
            &eval_sum,
            eval_sum_native,
            self.delta,
            &format!("step_{}_Eval_sum_scaled", step_idx),
        )?;
        let (right, right_native) = helpers::k_mul_with_hint(
            cs,
            &eq_aprp_ar,
            eq_aprp_ar_native,
            &eval_sum_scaled,
            eval_sum_scaled_native,
            self.delta,
            &format!("step_{}_RHS_right", step_idx),
        )?;

        // rhs = left + right with native hint.
        let rhs_native = left_native + right_native;
        let rhs_hint = KNum::<CircuitF>::from_neo_k(rhs_native);
        let rhs = k_add_raw(
            cs,
            &left,
            &right,
            Some(rhs_hint),
            &format!("step_{}_RHS_total", step_idx),
        )
        .map_err(SpartanBridgeError::BellpepperError)?;

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
        tr: &mut Poseidon2TranscriptVar,
        lane: &'static [u8],
        step_idx: usize,
        fold_digest_bytes: &[AllocatedNum<CircuitF>],
        parent: &neo_ccs::MeInstance<neo_ajtai::Commitment, NeoF, neo_math::K>,
        parent_vars: &MeInstanceVars,
        children: &[neo_ccs::MeInstance<neo_ajtai::Commitment, NeoF, neo_math::K>],
        children_vars: &[MeInstanceVars],
        rhos: &[Mat<NeoF>],
    ) -> Result<()> {

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
        if children_vars.len() != children.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "RLC at step {}: children_vars/children length mismatch (vars={}, children={})",
                step_idx,
                children_vars.len(),
                children.len()
            )));
        }

        let d = parent.X.rows();
        let m_in = parent.m_in;
        let c_len = parent.c.data.len();

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
            if child.c.data.len() != c_len {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "RLC c.data length mismatch at step {}, child {}",
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

        // Sanity: parent/children variable shapes must match values (avoid unconstrained duplicates).
        if parent_vars.c_data.len() != parent.c.data.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "RLC at step {}: parent c_data vars length mismatch (vars={}, vals={})",
                step_idx,
                parent_vars.c_data.len(),
                parent.c.data.len()
            )));
        }
        if parent_vars.X.len() != d || (!parent_vars.X.is_empty() && parent_vars.X[0].len() != parent.X.cols()) {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "RLC at step {}: parent X vars shape mismatch",
                step_idx
            )));
        }
        for (i, child) in children.iter().enumerate() {
            if children_vars[i].c_data.len() != child.c.data.len() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "RLC at step {}: child {} c_data vars length mismatch (vars={}, vals={})",
                    step_idx,
                    i,
                    children_vars[i].c_data.len(),
                    child.c.data.len()
                )));
            }
            if children_vars[i].X.len() != d || (!children_vars[i].X.is_empty() && children_vars[i].X[0].len() != child.X.cols()) {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "RLC at step {}: child {} X vars shape mismatch",
                    step_idx, i
                )));
            }
        }

        // Allocate rho matrices as base-field variables so they cannot appear
        // as witness-dependent coefficients in constraints.
        let mut rho_vars = Vec::with_capacity(rhos.len());
        for (i, rho) in rhos.iter().enumerate() {
            rho_vars.push(helpers::alloc_matrix_from_neo(
                cs,
                rho,
                &format!("step_{}_rlc_rho_{}", step_idx, i),
            )?);
        }

        // Fiat–Shamir binding for Π_RLC ρ.
        self.bind_rlc_inputs_lane(cs, tr, lane, step_idx, children, children_vars, fold_digest_bytes)?;
        self.enforce_rot_rho_sampling_no_reject_mod5(cs, tr, lane, &rho_vars, step_idx)?;

        // Enforce c_parent = Σ_i ρ_i · c_i  (matrix multiply per commitment column).
        //
        // Commitment is stored as column-major flat data: data[col * d + row].
        let kappa = parent
            .c
            .kappa;
        if parent.c.d != d {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "RLC commitment d mismatch at step {} (c.d={}, X.rows()={})",
                step_idx, parent.c.d, d
            )));
        }
        for col in 0..kappa {
            for row in 0..d {
                let mut prod_vars: Vec<bellpepper_core::Variable> = Vec::with_capacity(children.len() * d);
                for i in 0..children.len() {
                    for k in 0..d {
                        let rho_val = helpers::neo_f_to_circuit(&rhos[i][(row, k)]);
                        let c_val = helpers::neo_f_to_circuit(&children[i].c.data[col * d + k]);
                        let prod_val = rho_val * c_val;
                        let prod = cs.alloc(
                            || format!("step_{}_rlc_c_prod_row{}_col{}_i{}_k{}", step_idx, row, col, i, k),
                            || Ok(prod_val),
                        )?;
                        cs.enforce(
                            || format!("step_{}_rlc_c_prod_constraint_row{}_col{}_i{}_k{}", step_idx, row, col, i, k),
                            |lc| lc + rho_vars[i][row][k],
                            |lc| lc + children_vars[i].c_data[col * d + k],
                            |lc| lc + prod,
                        );
                        prod_vars.push(prod);
                    }
                }
                cs.enforce(
                    || format!("step_{}_rlc_c_row{}_col{}", step_idx, row, col),
                    |lc| {
                        let mut res = lc;
                        for v in &prod_vars {
                            res = res + *v;
                        }
                        res
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + parent_vars.c_data[col * d + row],
                );
            }
        }

        // Enforce X_parent = Σ_i ρ_i · X_i
        for row in 0usize..d {
            for col in 0usize..m_in {
                let mut prod_vars: Vec<bellpepper_core::Variable> = Vec::with_capacity(children.len() * d);
                for i in 0..children.len() {
                    for k in 0..d {
                        let rho_val = helpers::neo_f_to_circuit(&rhos[i][(row, k)]);
                        let x_val = helpers::neo_f_to_circuit(&children[i].X[(k, col)]);
                        let prod_val = rho_val * x_val;
                        let prod = cs.alloc(
                            || format!("step_{}_rlc_X_prod_r{}_c{}_i{}_k{}", step_idx, row, col, i, k),
                            || Ok(prod_val),
                        )?;
                        cs.enforce(
                            || format!("step_{}_rlc_X_prod_constraint_r{}_c{}_i{}_k{}", step_idx, row, col, i, k),
                            |lc| lc + rho_vars[i][row][k],
                            |lc| lc + children_vars[i].X[k][col],
                            |lc| lc + prod,
                        );
                        prod_vars.push(prod);
                    }
                }
                cs.enforce(
                    || format!("step_{}_rlc_X_{}_{}", step_idx, row, col),
                    |lc| {
                        let mut res = lc;
                        for v in &prod_vars {
                            res = res + *v;
                        }
                        res
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + parent_vars.X[row][col],
                );
            }
        }

        // Enforce r preservation: parent.r == child.r for all inputs
        let r_len = parent.r.len();
        if parent_vars.r.len() != r_len {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "RLC at step {}: parent r length mismatch (vars={}, vals={})",
                step_idx,
                parent_vars.r.len(),
                r_len
            )));
        }
        for (i, child) in children.iter().enumerate() {
            if child.r.len() != r_len || children_vars[i].r.len() != r_len {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "RLC r length mismatch at step {}, child {}",
                    step_idx, i
                )));
            }
            for idx in 0..r_len {
                helpers::enforce_k_eq(
                    cs,
                    &parent_vars.r[idx],
                    &children_vars[i].r[idx],
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
        if parent_vars.y.len() != t {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "RLC at step {}: parent y vars length mismatch (vars={}, vals={})",
                step_idx,
                parent_vars.y.len(),
                t
            )));
        }
        for (i, child) in children.iter().enumerate() {
            if child.y.len() != t || children_vars[i].y.len() != t {
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
                let mut prod_c0_vars: Vec<bellpepper_core::Variable> = Vec::with_capacity(children.len() * d_ring);
                let mut prod_c1_vars: Vec<bellpepper_core::Variable> = Vec::with_capacity(children.len() * d_ring);

                for i in 0..children.len() {
                    for k in 0..d_ring {
                        let rho_val = helpers::neo_f_to_circuit(&rhos[i][(r_idx, k)]);

                        let child_y_val = children[i].y[j][k];
                        let child_y_coeffs = child_y_val.as_coeffs();
                        let child_y_c0_val = helpers::neo_f_to_circuit(&child_y_coeffs[0]);
                        let child_y_c1_val = helpers::neo_f_to_circuit(&child_y_coeffs[1]);

                        let prod_c0_val = rho_val * child_y_c0_val;
                        let prod_c1_val = rho_val * child_y_c1_val;

                        let prod_c0 = cs.alloc(
                            || {
                                format!(
                                    "step_{}_rlc_y_c0_prod_j{}_r{}_i{}_k{}",
                                    step_idx, j, r_idx, i, k
                                )
                            },
                            || Ok(prod_c0_val),
                        )?;
                        cs.enforce(
                            || {
                                format!(
                                    "step_{}_rlc_y_c0_prod_constraint_j{}_r{}_i{}_k{}",
                                    step_idx, j, r_idx, i, k
                                )
                            },
                            |lc| lc + rho_vars[i][r_idx][k],
                            |lc| lc + children_vars[i].y[j][k].c0,
                            |lc| lc + prod_c0,
                        );
                        prod_c0_vars.push(prod_c0);

                        let prod_c1 = cs.alloc(
                            || {
                                format!(
                                    "step_{}_rlc_y_c1_prod_j{}_r{}_i{}_k{}",
                                    step_idx, j, r_idx, i, k
                                )
                            },
                            || Ok(prod_c1_val),
                        )?;
                        cs.enforce(
                            || {
                                format!(
                                    "step_{}_rlc_y_c1_prod_constraint_j{}_r{}_i{}_k{}",
                                    step_idx, j, r_idx, i, k
                                )
                            },
                            |lc| lc + rho_vars[i][r_idx][k],
                            |lc| lc + children_vars[i].y[j][k].c1,
                            |lc| lc + prod_c1,
                        );
                        prod_c1_vars.push(prod_c1);
                    }
                }

                // c0 component
                cs.enforce(
                    || format!("step_{}_rlc_y_c0_j{}_r{}", step_idx, j, r_idx),
                    |lc| {
                        let mut res = lc;
                        for v in &prod_c0_vars {
                            res = res + *v;
                        }
                        res
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + parent_vars.y[j][r_idx].c0,
                );

                // c1 component
                cs.enforce(
                    || format!("step_{}_rlc_y_c1_j{}_r{}", step_idx, j, r_idx),
                    |lc| {
                        let mut res = lc;
                        for v in &prod_c1_vars {
                            res = res + *v;
                        }
                        res
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + parent_vars.y[j][r_idx].c1,
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
        parent: &neo_ccs::MeInstance<neo_ajtai::Commitment, NeoF, neo_math::K>,
        parent_vars: &MeInstanceVars,
        children: &[neo_ccs::MeInstance<neo_ajtai::Commitment, NeoF, neo_math::K>],
        children_vars: &[MeInstanceVars],
    ) -> Result<()> {

        if children.is_empty() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "DEC at step {} has no children",
                step_idx
            )));
        }

        let d = parent.X.rows();
        let m_in = parent.m_in;
        let c_len = parent.c.data.len();

        if parent_vars.c_data.len() != c_len {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "DEC commitment vars length mismatch at step {} (vars={}, vals={})",
                step_idx,
                parent_vars.c_data.len(),
                c_len
            )));
        }
        if parent_vars.X.len() != d || (!parent_vars.X.is_empty() && parent_vars.X[0].len() != parent.X.cols()) {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "DEC parent X vars shape mismatch at step {}",
                step_idx
            )));
        }

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
            if child.c.data.len() != c_len {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "DEC c.data length mismatch at step {}, child {}",
                    step_idx, i
                )));
            }
            if children_vars.get(i).is_none() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "DEC at step {}: missing children_vars[{}]",
                    step_idx, i
                )));
            }
        }

        // Commitment linear relation: c_parent = Σ b^i · c_child_i.
        if parent.c.d != d {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "DEC commitment d mismatch at step {} (c.d={}, X.rows()={})",
                step_idx, parent.c.d, d
            )));
        }
        let parent_kappa = parent.c.kappa;
        for (i, child) in children.iter().enumerate() {
            if child.c.d != d || child.c.kappa != parent_kappa {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "DEC commitment shape mismatch at step {}, child {}",
                    step_idx, i
                )));
            }
            if children_vars[i].c_data.len() != child.c.data.len() {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "DEC commitment vars length mismatch at step {}, child {}",
                    step_idx, i
                )));
            }
        }

        for idx in 0..c_len {
            cs.enforce(
                || format!("step_{}_dec_c_idx{}", step_idx, idx),
                |lc| {
                    let mut res = lc;
                    let mut pow = CircuitF::from(1u64);
                    for child_vars in children_vars.iter() {
                        res = res + (pow, child_vars.c_data[idx]);
                        pow *= CircuitF::from(self.base_b as u64);
                    }
                    res
                },
                |lc| lc + CS::one(),
                |lc| lc + parent_vars.c_data[idx],
            );
        }

        if children_vars.len() != children.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "DEC at step {}: children_vars/children length mismatch (vars={}, children={})",
                step_idx,
                children_vars.len(),
                children.len()
            )));
        }

        // X_parent = Σ b^i · X_i
        for row in 0usize..d {
            for col in 0usize..m_in {
                cs.enforce(
                    || format!("step_{}_dec_X_{}_{}", step_idx, row, col),
                    |lc| {
                        let mut res = lc;
                        let mut pow = CircuitF::from(1u64);
                        for child_vars in children_vars.iter() {
                            res = res + (pow, child_vars.X[row][col]);
                            pow *= CircuitF::from(self.base_b as u64);
                        }
                        res
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + parent_vars.X[row][col],
                );
            }
        }

        // r preservation: parent.r == child.r for all children
        let r_len = parent.r.len();
        if parent_vars.r.len() != r_len {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "DEC at step {}: parent r length mismatch (vars={}, vals={})",
                step_idx,
                parent_vars.r.len(),
                r_len
            )));
        }
        for (i, child) in children.iter().enumerate() {
            if child.r.len() != r_len {
                return Err(SpartanBridgeError::InvalidInput(format!(
                    "DEC r length mismatch at step {}, child {}",
                    step_idx, i
                )));
            }
            for idx in 0..r_len {
                helpers::enforce_k_eq(
                    cs,
                    &parent_vars.r[idx],
                    &children_vars[i].r[idx],
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

        if parent_vars.y.len() != parent.y.len() {
            return Err(SpartanBridgeError::InvalidInput(format!(
                "DEC at step {}: parent y length mismatch (vars={}, vals={})",
                step_idx,
                parent_vars.y.len(),
                parent.y.len()
            )));
        }

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
        }

        for j in 0..t {
            for r_idx in 0..d_pad {
                // c0 component
                cs.enforce(
                    || format!("step_{}_dec_y_c0_j{}_r{}", step_idx, j, r_idx),
                    |lc| {
                        let mut res = lc;
                        let mut pow = CircuitF::from(1u64);
                        for child_y in children_vars.iter() {
                            res = res + (pow, child_y.y[j][r_idx].c0);
                            pow *= CircuitF::from(self.base_b as u64);
                        }
                        res
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + parent_vars.y[j][r_idx].c0,
                );

                // c1 component
                cs.enforce(
                    || format!("step_{}_dec_y_c1_j{}_r{}", step_idx, j, r_idx),
                    |lc| {
                        let mut res = lc;
                        let mut pow = CircuitF::from(1u64);
                        for child_y in children_vars.iter() {
                            res = res + (pow, child_y.y[j][r_idx].c1);
                            pow *= CircuitF::from(self.base_b as u64);
                        }
                        res
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + parent_vars.y[j][r_idx].c1,
                );
            }
        }

        Ok(())
    }

}

/// Implement Spartan2's `SpartanCircuit` trait for `FoldRunCircuit` using the
/// Goldilocks + Hash-MLE PCS engine. This lets Spartan2 treat the FoldRun
/// circuit as an R1CS provider.
impl SpartanCircuitTrait<GoldilocksMerkleMleEngine> for FoldRunCircuit {
    fn public_values(&self) -> std::result::Result<Vec<CircuitF>, SynthesisError> {
        // Must mirror `allocate_public_inputs`.
        Ok(self.instance.public_io())
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
