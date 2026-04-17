//! Owns Spartan packaging for the compact RV64IM side-opening theorem.
//!
//! The circuit proves the selected-opening witness against the already-carried
//! compact side-opening relation statement. It does not own the Phase 0
//! public tuple binding or outer Nightstream linkage checks.

use bellpepper_core::{
    num::AllocatedNum, test_cs::TestConstraintSystem, Comparable, ConstraintSystem, Delta, SynthesisError,
};
use neo_math::F;
use neo_transcript::Transcript;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::{Deserialize, Serialize};
use spartan2::{
    provider::{goldi::F as SpartanF, GoldilocksP3MerkleMleEngine},
    spartan::R1CSSNARK,
    traits::{circuit::SpartanCircuit, snark::R1CSSNARKTrait},
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use super::side_claim_relation::single_step_packaged_statement_digest;
use super::side_claim_relation::{
    validate_rv64im_single_step_packaged_witness_shape, RV64IM_SINGLE_STEP_PACKAGED_FINAL_MAIN_CLAIM_COUNT,
};
use super::side_opening_relation::{
    build_rv64im_kernel_opening_claim_from_statement, validate_rv64im_side_opening_relation_statement,
    verify_rv64im_side_opening_relation, Rv64imSideOpeningRelationStatement, Rv64imSideOpeningRelationWitness,
    Rv64imStage1SelectedRowsWitness, Rv64imStage2SelectedEventsWitness, Rv64imStage3SelectedContinuityWitness,
};
use super::side_relation_circuit::digests::{
    continuity_event_digest, kernel_binding_opening_packaged_statement_digest,
    kernel_prepared_step_opening_packaged_statement_digest, ram_event_digest, register_read_event_digest,
    register_write_event_digest, stage1_opening_packaged_statement_digest, stage1_row_digest,
    stage2_opening_packaged_statement_digest, stage3_opening_packaged_statement_digest, twist_link_event_digest,
};
use crate::finalize::digest32_as_fields;
use crate::proof::PublicStep;
use crate::rv64im::kernel::{
    build_claim_packaged_public_step, build_kernel_binding_opening_public_step,
    build_kernel_prepared_step_opening_public_step, same_public_step, SimpleKernelError,
};
use crate::rv64im::main_relation_circuit::transcript::Poseidon2TranscriptCircuit;
use crate::rv64im::stage1::Stage1RowBinding;
use crate::rv64im::stage2::{
    RamAccessKind, RamEvent, RegisterReadEvent, RegisterReadRole, RegisterWriteEvent, TwistLinkEvent,
};
use crate::rv64im::stage3::ContinuityEvent;
use crate::rv64im::tables::Rv64FamilyTag;

pub type Rv64imSideOpeningSpartanEngine = GoldilocksP3MerkleMleEngine;
pub type Rv64imSideOpeningSpartanSnark = R1CSSNARK<Rv64imSideOpeningSpartanEngine>;
pub type Rv64imSideOpeningSpartanProverKey = spartan2::spartan::SpartanProverKey<Rv64imSideOpeningSpartanEngine>;
pub type Rv64imSideOpeningSpartanVerifierKey = spartan2::spartan::SpartanVerifierKey<Rv64imSideOpeningSpartanEngine>;
type Rv64imSideOpeningSpartanKeyPair = Arc<(Rv64imSideOpeningSpartanProverKey, Rv64imSideOpeningSpartanVerifierKey)>;

static RV64IM_SIDE_OPENING_SETUP_CACHE: OnceLock<Mutex<HashMap<[u8; 32], Rv64imSideOpeningSpartanKeyPair>>> =
    OnceLock::new();

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Rv64imSideOpeningSpartanProof {
    pub snark_data: Vec<u8>,
}

impl Rv64imSideOpeningSpartanProof {
    pub fn snark_bytes_len(&self) -> usize {
        self.snark_data.len()
    }
}

#[derive(Clone)]
struct Rv64imSideOpeningSpartanCircuit {
    statement: Rv64imSideOpeningRelationStatement,
    witness: Rv64imSideOpeningRelationWitness,
}

#[derive(Clone)]
struct DebugStage1PackagedOpeningDigestCircuit {
    claim: crate::rv64im::kernel::Stage1SelectedOpeningClaim,
    carried_statement_digest: [u8; 32],
    carried_packaged_digest: [u8; 32],
    witness: crate::nightstream::rv64im::side_claim_relation::Rv64imSingleStepPackagedProofWitness,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Rv64imSideOpeningSpartanCircuitShape {
    pub num_inputs: usize,
    pub num_aux: usize,
    pub num_constraints: usize,
    pub constraint_fingerprint: String,
}

impl Rv64imSideOpeningSpartanCircuit {
    fn expected_public_values(&self) -> Vec<SpartanF> {
        digest32_as_spartan_fields(self.statement.expected_digest()).to_vec()
    }
}

impl SpartanCircuit<Rv64imSideOpeningSpartanEngine> for Rv64imSideOpeningSpartanCircuit {
    fn public_values(&self) -> Result<Vec<SpartanF>, SynthesisError> {
        Ok(self.expected_public_values())
    }

    fn shared<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn precommitted<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
        _: &[AllocatedNum<SpartanF>],
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn num_challenges(&self) -> usize {
        0
    }

    fn synthesize<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        _: &[AllocatedNum<SpartanF>],
        _: &[AllocatedNum<SpartanF>],
        _: Option<&[SpartanF]>,
    ) -> Result<(), SynthesisError> {
        let public_inputs = self
            .expected_public_values()
            .into_iter()
            .enumerate()
            .map(|(idx, value)| AllocatedNum::alloc_input(cs.namespace(|| format!("public_input_{idx}")), || Ok(value)))
            .collect::<Result<Vec<_>, _>>()?;

        enforce_stage1_claim(
            &mut cs.namespace(|| "stage1_selected"),
            &self.statement.stage1.claim,
            &self.witness.stage1_selected_rows,
        )?;
        enforce_stage2_claim(
            &mut cs.namespace(|| "stage2_selected"),
            &self.statement.stage2.claim,
            &self.witness.stage2_selected_events,
        )?;
        enforce_stage3_claim(
            &mut cs.namespace(|| "stage3_selected"),
            &self.statement.stage3.claim,
            &self.witness.stage3_selected_continuity,
        )?;

        let stage1_statement_digest = stage1_opening_packaged_statement_digest(
            cs.namespace(|| "stage1_statement_digest"),
            &self.statement.stage1.claim,
            &self.witness.stage1_packaged.final_main_claim_digests,
            "stage1_statement_digest",
        )?;
        enforce_packaged_opening_digest(
            &mut cs.namespace(|| "stage1_packaged"),
            self.statement.stage1.claim.digest,
            self.statement.stage1.packaged_statement_digest,
            self.statement.stage1.packaged_digest,
            &self.witness.stage1_packaged,
            stage1_statement_digest,
        )?;
        let stage2_statement_digest = stage2_opening_packaged_statement_digest(
            cs.namespace(|| "stage2_statement_digest"),
            &self.statement.stage2.claim,
            &self.witness.stage2_packaged.final_main_claim_digests,
            "stage2_statement_digest",
        )?;
        enforce_packaged_opening_digest(
            &mut cs.namespace(|| "stage2_packaged"),
            self.statement.stage2.claim.digest,
            self.statement.stage2.packaged_statement_digest,
            self.statement.stage2.packaged_digest,
            &self.witness.stage2_packaged,
            stage2_statement_digest,
        )?;
        let stage3_statement_digest = stage3_opening_packaged_statement_digest(
            cs.namespace(|| "stage3_statement_digest"),
            &self.statement.stage3.claim,
            &self.witness.stage3_packaged.final_main_claim_digests,
            "stage3_statement_digest",
        )?;
        enforce_packaged_opening_digest(
            &mut cs.namespace(|| "stage3_packaged"),
            self.statement.stage3.claim.digest,
            self.statement.stage3.packaged_statement_digest,
            self.statement.stage3.packaged_digest,
            &self.witness.stage3_packaged,
            stage3_statement_digest,
        )?;
        let kernel_opening_claim = build_rv64im_kernel_opening_claim_from_statement(&self.statement)
            .map_err(|_| SynthesisError::Unsatisfiable)?;
        let bindings_statement_digest = kernel_binding_opening_packaged_statement_digest(
            cs.namespace(|| "bindings_statement_digest"),
            &kernel_opening_claim.bindings,
            &self.witness.bindings_packaged.final_main_claim_digests,
            "bindings_statement_digest",
        )?;
        enforce_packaged_opening_digest(
            &mut cs.namespace(|| "bindings_packaged"),
            kernel_opening_claim.bindings.digest,
            self.statement
                .kernel_opening_bridge
                .bindings_opening_statement_digest,
            self.statement.kernel_opening_bridge.bindings_opening_digest,
            &self.witness.bindings_packaged,
            bindings_statement_digest,
        )?;
        let prepared_steps_statement_digest = kernel_prepared_step_opening_packaged_statement_digest(
            cs.namespace(|| "prepared_steps_statement_digest"),
            &kernel_opening_claim.prepared_steps,
            &self
                .witness
                .prepared_steps_packaged
                .final_main_claim_digests,
            "prepared_steps_statement_digest",
        )?;
        enforce_packaged_opening_digest(
            &mut cs.namespace(|| "prepared_steps_packaged"),
            kernel_opening_claim.prepared_steps.digest,
            self.statement
                .kernel_opening_bridge
                .prepared_steps_opening_statement_digest,
            self.statement
                .kernel_opening_bridge
                .prepared_steps_opening_digest,
            &self.witness.prepared_steps_packaged,
            prepared_steps_statement_digest,
        )?;
        let statement_digest = alloc_digest32_const(
            &mut cs.namespace(|| "statement_digest_const"),
            self.statement.expected_digest(),
            "statement_digest_const",
        )?;
        enforce_digest_eq_public_inputs(
            &mut cs.namespace(|| "statement_digest"),
            &statement_digest,
            &public_inputs,
        )?;
        Ok(())
    }
}

impl SpartanCircuit<Rv64imSideOpeningSpartanEngine> for DebugStage1PackagedOpeningDigestCircuit {
    fn public_values(&self) -> Result<Vec<SpartanF>, SynthesisError> {
        Ok(Vec::new())
    }

    fn shared<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn precommitted<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
        _: &[AllocatedNum<SpartanF>],
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn num_challenges(&self) -> usize {
        0
    }

    fn synthesize<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        _: &[AllocatedNum<SpartanF>],
        _: &[AllocatedNum<SpartanF>],
        _: Option<&[SpartanF]>,
    ) -> Result<(), SynthesisError> {
        let statement_digest = stage1_opening_packaged_statement_digest(
            cs.namespace(|| "stage1_statement_digest"),
            &self.claim,
            &self.witness.final_main_claim_digests,
            "stage1_statement_digest",
        )?;
        enforce_packaged_opening_digest(
            &mut cs.namespace(|| "stage1_packaged"),
            self.claim.digest,
            self.carried_statement_digest,
            self.carried_packaged_digest,
            &self.witness,
            statement_digest,
        )
    }
}

pub fn setup_rv64im_side_opening_spartan(
    statement: &Rv64imSideOpeningRelationStatement,
    witness: &Rv64imSideOpeningRelationWitness,
) -> Result<(Rv64imSideOpeningSpartanProverKey, Rv64imSideOpeningSpartanVerifierKey), SimpleKernelError> {
    validate_rv64im_side_opening_packaged_witness_shapes(witness)?;
    let setup_witness = setup_rv64im_side_opening_witness(statement, witness)?;
    Rv64imSideOpeningSpartanSnark::setup(Rv64imSideOpeningSpartanCircuit {
        statement: statement.clone(),
        witness: setup_witness,
    })
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM side opening Spartan setup failed: {err}")))
}

pub fn debug_setup_rv64im_side_opening_spartan_without_packaged_final_main_claims(
    statement: &Rv64imSideOpeningRelationStatement,
    witness: &Rv64imSideOpeningRelationWitness,
) -> Result<(Rv64imSideOpeningSpartanProverKey, Rv64imSideOpeningSpartanVerifierKey), SimpleKernelError> {
    validate_rv64im_side_opening_packaged_witness_shapes(witness)?;
    let setup_witness = setup_rv64im_side_opening_witness_without_packaged_final_main_claims(statement, witness)?;
    Rv64imSideOpeningSpartanSnark::setup(Rv64imSideOpeningSpartanCircuit {
        statement: statement.clone(),
        witness: setup_witness,
    })
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM reduced side opening Spartan setup failed: {err}")))
}

pub fn debug_setup_rv64im_side_opening_spartan_without_stage1_packaged_final_main_claims(
    statement: &Rv64imSideOpeningRelationStatement,
    witness: &Rv64imSideOpeningRelationWitness,
) -> Result<(Rv64imSideOpeningSpartanProverKey, Rv64imSideOpeningSpartanVerifierKey), SimpleKernelError> {
    validate_rv64im_side_opening_packaged_witness_shapes(witness)?;
    let mut setup_witness = setup_rv64im_side_opening_witness(statement, witness)?;
    setup_witness.stage1_packaged.final_main_claim_digests =
        vec![[F::ZERO; 4]; RV64IM_SINGLE_STEP_PACKAGED_FINAL_MAIN_CLAIM_COUNT];
    Rv64imSideOpeningSpartanSnark::setup(Rv64imSideOpeningSpartanCircuit {
        statement: statement.clone(),
        witness: setup_witness,
    })
    .map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM reduced stage1 side opening Spartan setup failed: {err}"
        ))
    })
}

pub fn setup_rv64im_side_opening_spartan_cached(
    statement: &Rv64imSideOpeningRelationStatement,
    witness: &Rv64imSideOpeningRelationWitness,
) -> Result<Rv64imSideOpeningSpartanKeyPair, SimpleKernelError> {
    validate_rv64im_side_opening_packaged_witness_shapes(witness)?;
    let shape_digest = rv64im_side_opening_shape_digest(statement, witness);
    let cache = RV64IM_SIDE_OPENING_SETUP_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(keys) = cache
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("RV64IM side opening setup cache poisoned".into()))?
        .get(&shape_digest)
        .cloned()
    {
        return Ok(keys);
    }

    let keys = Arc::new(setup_rv64im_side_opening_spartan(statement, witness)?);
    cache
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("RV64IM side opening setup cache poisoned".into()))?
        .insert(shape_digest, keys.clone());
    Ok(keys)
}

pub fn prove_rv64im_side_opening_spartan(
    pk: &Rv64imSideOpeningSpartanProverKey,
    statement: &Rv64imSideOpeningRelationStatement,
    witness: &Rv64imSideOpeningRelationWitness,
) -> Result<Rv64imSideOpeningSpartanProof, SimpleKernelError> {
    verify_rv64im_side_opening_relation(statement, witness)?;
    let circuit = Rv64imSideOpeningSpartanCircuit {
        statement: statement.clone(),
        witness: witness.clone(),
    };
    let prep = Rv64imSideOpeningSpartanSnark::prep_prove(pk, circuit.clone(), true)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM side opening Spartan prepare failed: {err}")))?;
    let proof = Rv64imSideOpeningSpartanSnark::prove(pk, circuit, &prep, true)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM side opening Spartan prove failed: {err}")))?;
    let snark_data = bincode::serialize(&proof)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM side opening Spartan encode failed: {err}")))?;
    Ok(Rv64imSideOpeningSpartanProof { snark_data })
}

pub fn verify_rv64im_side_opening_spartan(
    vk: &Rv64imSideOpeningSpartanVerifierKey,
    statement: &Rv64imSideOpeningRelationStatement,
    proof: &Rv64imSideOpeningSpartanProof,
) -> Result<(), SimpleKernelError> {
    validate_side_opening_statement(statement)?;
    let snark: Rv64imSideOpeningSpartanSnark = bincode::deserialize(&proof.snark_data)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM side opening Spartan decode failed: {err}")))?;
    let public_values = snark
        .verify(vk)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM side opening Spartan verify failed: {err}")))?;
    let expected = digest32_as_spartan_fields(statement.expected_digest());
    if public_values != expected {
        return Err(SimpleKernelError::Bridge(
            "RV64IM side opening Spartan public IO mismatch".into(),
        ));
    }
    Ok(())
}

pub fn debug_check_rv64im_side_opening_spartan_circuit(
    statement: &Rv64imSideOpeningRelationStatement,
    witness: &Rv64imSideOpeningRelationWitness,
) -> Result<(), SimpleKernelError> {
    let circuit = Rv64imSideOpeningSpartanCircuit {
        statement: statement.clone(),
        witness: witness.clone(),
    };
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    circuit.synthesize(&mut cs, &[], &[], None).map_err(|err| {
        SimpleKernelError::Bridge(format!("RV64IM side opening Spartan debug synthesis failed: {err}"))
    })?;
    if !cs.is_satisfied() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV64IM side opening Spartan circuit unsatisfied: {}",
            cs.which_is_unsatisfied()
                .unwrap_or_else(|| "unknown".into())
        )));
    }
    Ok(())
}

pub fn debug_measure_rv64im_side_opening_spartan_circuit_shape(
    statement: &Rv64imSideOpeningRelationStatement,
    witness: &Rv64imSideOpeningRelationWitness,
) -> Result<Rv64imSideOpeningSpartanCircuitShape, SimpleKernelError> {
    let circuit = Rv64imSideOpeningSpartanCircuit {
        statement: statement.clone(),
        witness: witness.clone(),
    };
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    circuit.synthesize(&mut cs, &[], &[], None).map_err(|err| {
        SimpleKernelError::Bridge(format!("RV64IM side opening Spartan debug synthesis failed: {err}"))
    })?;
    Ok(Rv64imSideOpeningSpartanCircuitShape {
        num_inputs: cs.num_inputs(),
        num_aux: cs.scalar_aux().len(),
        num_constraints: cs.num_constraints(),
        constraint_fingerprint: format!(
            "inputs:{} aux:{} constraints:{}",
            cs.num_inputs(),
            cs.scalar_aux().len(),
            cs.num_constraints()
        ),
    })
}

pub fn debug_compare_rv64im_side_opening_spartan_statement_owned_shape(
    statement: &Rv64imSideOpeningRelationStatement,
    witness: &Rv64imSideOpeningRelationWitness,
) -> Result<Option<String>, SimpleKernelError> {
    let real_circuit = Rv64imSideOpeningSpartanCircuit {
        statement: statement.clone(),
        witness: witness.clone(),
    };
    let dummy_circuit = Rv64imSideOpeningSpartanCircuit {
        statement: statement.clone(),
        witness: dummy_rv64im_side_opening_witness(statement)?,
    };
    let mut real_cs = TestConstraintSystem::<SpartanF>::new();
    real_circuit
        .synthesize(&mut real_cs, &[], &[], None)
        .map_err(|err| {
            SimpleKernelError::Bridge(format!("RV64IM side opening Spartan real synthesis failed: {err}"))
        })?;
    let mut dummy_cs = TestConstraintSystem::<SpartanF>::new();
    dummy_circuit
        .synthesize(&mut dummy_cs, &[], &[], None)
        .map_err(|err| {
            SimpleKernelError::Bridge(format!("RV64IM side opening Spartan dummy synthesis failed: {err}"))
        })?;
    Ok(match real_cs.delta(&dummy_cs, false) {
        Delta::Equal => None,
        delta => Some(format!("{delta:?}")),
    })
}

pub fn debug_compare_rv64im_side_opening_spartan_setup_shape(
    statement: &Rv64imSideOpeningRelationStatement,
    witness: &Rv64imSideOpeningRelationWitness,
) -> Result<Option<String>, SimpleKernelError> {
    compare_side_opening_circuit_delta(
        statement,
        witness,
        &setup_rv64im_side_opening_witness(statement, witness)?,
    )
}

pub fn debug_compare_rv64im_side_opening_spartan_without_packaged_final_main_claims_shape(
    statement: &Rv64imSideOpeningRelationStatement,
    witness: &Rv64imSideOpeningRelationWitness,
) -> Result<Option<String>, SimpleKernelError> {
    compare_side_opening_circuit_delta(
        statement,
        witness,
        &setup_rv64im_side_opening_witness_without_packaged_final_main_claims(statement, witness)?,
    )
}

pub fn debug_compare_rv64im_stage1_packaged_opening_digest_without_packaged_final_main_claims_shape(
    statement: &Rv64imSideOpeningRelationStatement,
    witness: &Rv64imSideOpeningRelationWitness,
) -> Result<Option<String>, SimpleKernelError> {
    let reduced = setup_rv64im_side_opening_witness_without_packaged_final_main_claims(statement, witness)?;
    compare_stage_packaged_opening_digest_delta(statement, &witness.stage1_packaged, &reduced.stage1_packaged)
}

pub fn debug_compare_rv64im_stage1_packaged_opening_digest_zeroing_only_final_main_claims_shape(
    statement: &Rv64imSideOpeningRelationStatement,
    witness: &Rv64imSideOpeningRelationWitness,
) -> Result<Option<String>, SimpleKernelError> {
    let mut reduced = witness.stage1_packaged.clone();
    reduced.final_main_claim_digests = vec![[F::ZERO; 4]; RV64IM_SINGLE_STEP_PACKAGED_FINAL_MAIN_CLAIM_COUNT];
    compare_stage_packaged_opening_digest_delta(statement, &witness.stage1_packaged, &reduced)
}

pub fn debug_compare_rv64im_stage1_packaged_opening_digest_zeroing_final_main_claims_with_fixed_native_statement_shape(
    statement: &Rv64imSideOpeningRelationStatement,
    witness: &Rv64imSideOpeningRelationWitness,
) -> Result<Option<String>, SimpleKernelError> {
    let mut reduced = witness.stage1_packaged.clone();
    reduced.final_main_claim_digests = vec![[F::ZERO; 4]; RV64IM_SINGLE_STEP_PACKAGED_FINAL_MAIN_CLAIM_COUNT];
    let fixed_native_statement_digest = native_stage_packaged_statement_digest(
        "rv64im/stage1",
        &statement.stage1.claim.claim_words(),
        &witness.stage1_packaged,
    )
    .map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM stage1 packaged fixed native statement digest failed: {err}"
        ))
    })?;
    compare_stage_packaged_opening_digest_delta_with_fixed_native_statement(
        statement,
        &witness.stage1_packaged,
        &reduced,
        fixed_native_statement_digest,
    )
}

pub fn debug_native_stage1_packaged_statement_digest(
    statement: &Rv64imSideOpeningRelationStatement,
    witness: &crate::nightstream::rv64im::side_claim_relation::Rv64imSingleStepPackagedProofWitness,
) -> Result<[u8; 32], SimpleKernelError> {
    native_stage_packaged_statement_digest("rv64im/stage1", &statement.stage1.claim.claim_words(), witness)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM stage1 packaged native digest failed: {err}")))
}

pub fn debug_round_trip_rv64im_stage1_packaged_opening_digest_with_reduced_setup(
    statement: &Rv64imSideOpeningRelationStatement,
    witness: &Rv64imSideOpeningRelationWitness,
) -> Result<(), SimpleKernelError> {
    validate_rv64im_side_opening_packaged_witness_shapes(witness)?;
    let mut setup_stage1_witness = witness.stage1_packaged.clone();
    setup_stage1_witness.final_main_claim_digests =
        vec![[F::ZERO; 4]; RV64IM_SINGLE_STEP_PACKAGED_FINAL_MAIN_CLAIM_COUNT];
    let (pk, vk) = Rv64imSideOpeningSpartanSnark::setup(DebugStage1PackagedOpeningDigestCircuit {
        claim: statement.stage1.claim.clone(),
        carried_statement_digest: statement.stage1.packaged_statement_digest,
        carried_packaged_digest: statement.stage1.packaged_digest,
        witness: setup_stage1_witness,
    })
    .map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM reduced stage1 packaged-opening digest setup failed: {err}"
        ))
    })?;
    let prove_circuit = DebugStage1PackagedOpeningDigestCircuit {
        claim: statement.stage1.claim.clone(),
        carried_statement_digest: statement.stage1.packaged_statement_digest,
        carried_packaged_digest: statement.stage1.packaged_digest,
        witness: witness.stage1_packaged.clone(),
    };
    let prep = Rv64imSideOpeningSpartanSnark::prep_prove(&pk, prove_circuit.clone(), true).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM reduced stage1 packaged-opening digest prepare failed: {err}"
        ))
    })?;
    let proof = Rv64imSideOpeningSpartanSnark::prove(&pk, prove_circuit, &prep, true).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM reduced stage1 packaged-opening digest prove failed: {err}"
        ))
    })?;
    proof.verify(&vk).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV64IM reduced stage1 packaged-opening digest verify failed: {err}"
        ))
    })?;
    Ok(())
}

fn compare_side_opening_circuit_delta(
    statement: &Rv64imSideOpeningRelationStatement,
    lhs_witness: &Rv64imSideOpeningRelationWitness,
    rhs_witness: &Rv64imSideOpeningRelationWitness,
) -> Result<Option<String>, SimpleKernelError> {
    let lhs_circuit = Rv64imSideOpeningSpartanCircuit {
        statement: statement.clone(),
        witness: lhs_witness.clone(),
    };
    let rhs_circuit = Rv64imSideOpeningSpartanCircuit {
        statement: statement.clone(),
        witness: rhs_witness.clone(),
    };
    let mut lhs_cs = TestConstraintSystem::<SpartanF>::new();
    lhs_circuit
        .synthesize(&mut lhs_cs, &[], &[], None)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM side opening Spartan lhs synthesis failed: {err}")))?;
    let mut rhs_cs = TestConstraintSystem::<SpartanF>::new();
    rhs_circuit
        .synthesize(&mut rhs_cs, &[], &[], None)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM side opening Spartan rhs synthesis failed: {err}")))?;
    Ok(match lhs_cs.delta(&rhs_cs, false) {
        Delta::Equal => None,
        delta => Some(format!("{delta:?}")),
    })
}

fn compare_stage_packaged_opening_digest_delta(
    statement: &Rv64imSideOpeningRelationStatement,
    lhs_witness: &crate::nightstream::rv64im::side_claim_relation::Rv64imSingleStepPackagedProofWitness,
    rhs_witness: &crate::nightstream::rv64im::side_claim_relation::Rv64imSingleStepPackagedProofWitness,
) -> Result<Option<String>, SimpleKernelError> {
    compare_stage_packaged_opening_digest_delta_with_carried_statement(statement, lhs_witness, rhs_witness)
}

fn compare_stage_packaged_opening_digest_delta_with_fixed_native_statement(
    statement: &Rv64imSideOpeningRelationStatement,
    lhs_witness: &crate::nightstream::rv64im::side_claim_relation::Rv64imSingleStepPackagedProofWitness,
    rhs_witness: &crate::nightstream::rv64im::side_claim_relation::Rv64imSingleStepPackagedProofWitness,
    _: [u8; 32],
) -> Result<Option<String>, SimpleKernelError> {
    compare_stage_packaged_opening_digest_delta_with_carried_statement(statement, lhs_witness, rhs_witness)
}

fn compare_stage_packaged_opening_digest_delta_with_carried_statement(
    statement: &Rv64imSideOpeningRelationStatement,
    lhs_witness: &crate::nightstream::rv64im::side_claim_relation::Rv64imSingleStepPackagedProofWitness,
    rhs_witness: &crate::nightstream::rv64im::side_claim_relation::Rv64imSingleStepPackagedProofWitness,
) -> Result<Option<String>, SimpleKernelError> {
    let mut lhs_cs = TestConstraintSystem::<SpartanF>::new();
    let lhs_statement_digest = stage1_opening_packaged_statement_digest(
        lhs_cs.namespace(|| "stage1_statement_digest"),
        &statement.stage1.claim,
        &lhs_witness.final_main_claim_digests,
        "stage1_statement_digest",
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM stage1 packaged lhs digest synthesis failed: {err}")))?;
    enforce_packaged_opening_digest(
        &mut lhs_cs.namespace(|| "stage1_packaged"),
        statement.stage1.claim.digest,
        statement.stage1.packaged_statement_digest,
        statement.stage1.packaged_digest,
        lhs_witness,
        lhs_statement_digest,
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM stage1 packaged lhs enforce failed: {err}")))?;

    let mut rhs_cs = TestConstraintSystem::<SpartanF>::new();
    let rhs_statement_digest = stage1_opening_packaged_statement_digest(
        rhs_cs.namespace(|| "stage1_statement_digest"),
        &statement.stage1.claim,
        &rhs_witness.final_main_claim_digests,
        "stage1_statement_digest",
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM stage1 packaged rhs digest synthesis failed: {err}")))?;
    enforce_packaged_opening_digest(
        &mut rhs_cs.namespace(|| "stage1_packaged"),
        statement.stage1.claim.digest,
        statement.stage1.packaged_statement_digest,
        statement.stage1.packaged_digest,
        rhs_witness,
        rhs_statement_digest,
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV64IM stage1 packaged rhs enforce failed: {err}")))?;

    Ok(match lhs_cs.delta(&rhs_cs, false) {
        Delta::Equal => None,
        delta => Some(format!("{delta:?}")),
    })
}

fn validate_side_opening_statement(statement: &Rv64imSideOpeningRelationStatement) -> Result<(), SimpleKernelError> {
    validate_rv64im_side_opening_relation_statement(statement)
}

fn rv64im_side_opening_shape_digest(
    statement: &Rv64imSideOpeningRelationStatement,
    _: &Rv64imSideOpeningRelationWitness,
) -> [u8; 32] {
    let mut tr = neo_transcript::Poseidon2Transcript::new(b"neo.fold.next/nightstream/rv64im/side_opening_shape");
    tr.append_message(
        b"neo.fold.next/nightstream/rv64im/side_opening_shape/statement_digest",
        &statement.expected_digest(),
    );
    tr.digest32()
}

fn dummy_rv64im_side_opening_witness(
    statement: &Rv64imSideOpeningRelationStatement,
) -> Result<Rv64imSideOpeningRelationWitness, SimpleKernelError> {
    let kernel_opening_claim = build_rv64im_kernel_opening_claim_from_statement(statement)?;
    Ok(Rv64imSideOpeningRelationWitness {
        stage1_selected_rows: Rv64imStage1SelectedRowsWitness {
            first: dummy_stage1_row_binding(),
            effect_position: 0,
            effect: dummy_stage1_row_binding(),
            commit_position: 0,
            commit: dummy_stage1_row_binding(),
            last: dummy_stage1_row_binding(),
        },
        stage2_selected_events: Rv64imStage2SelectedEventsWitness {
            first_read: statement
                .stage2
                .claim
                .points
                .first_read
                .as_ref()
                .map(|_| dummy_register_read_event()),
            last_read: statement
                .stage2
                .claim
                .points
                .last_read
                .as_ref()
                .map(|_| dummy_register_read_event()),
            first_write: statement
                .stage2
                .claim
                .points
                .first_write
                .as_ref()
                .map(|_| dummy_register_write_event()),
            last_write: statement
                .stage2
                .claim
                .points
                .last_write
                .as_ref()
                .map(|_| dummy_register_write_event()),
            first_ram: statement
                .stage2
                .claim
                .points
                .first_ram
                .as_ref()
                .map(|_| dummy_ram_event()),
            last_ram: statement
                .stage2
                .claim
                .points
                .last_ram
                .as_ref()
                .map(|_| dummy_ram_event()),
            first_twist: statement
                .stage2
                .claim
                .points
                .first_twist
                .as_ref()
                .map(|_| dummy_twist_link_event()),
            last_twist: statement
                .stage2
                .claim
                .points
                .last_twist
                .as_ref()
                .map(|_| dummy_twist_link_event()),
        },
        stage3_selected_continuity: Rv64imStage3SelectedContinuityWitness {
            first_continuity: statement
                .stage3
                .claim
                .points
                .first_continuity
                .as_ref()
                .map(|_| dummy_continuity_event()),
            last_continuity: statement
                .stage3
                .claim
                .points
                .last_continuity
                .as_ref()
                .map(|_| dummy_continuity_event()),
        },
        stage1_packaged: dummy_single_step_packaged_witness(&build_claim_packaged_public_step(
            "rv64im/stage1",
            &statement.stage1.claim.claim_words(),
        )?),
        stage2_packaged: dummy_single_step_packaged_witness(&build_claim_packaged_public_step(
            "rv64im/stage2",
            &statement.stage2.claim.claim_words(),
        )?),
        stage3_packaged: dummy_single_step_packaged_witness(&build_claim_packaged_public_step(
            "rv64im/stage3",
            &statement.stage3.claim.claim_words(),
        )?),
        bindings_packaged: dummy_single_step_packaged_witness(&build_kernel_binding_opening_public_step(
            &kernel_opening_claim.bindings,
        )?),
        prepared_steps_packaged: dummy_single_step_packaged_witness(&build_kernel_prepared_step_opening_public_step(
            &kernel_opening_claim.prepared_steps,
        )?),
    })
}

fn setup_rv64im_side_opening_witness(
    statement: &Rv64imSideOpeningRelationStatement,
    _: &Rv64imSideOpeningRelationWitness,
) -> Result<Rv64imSideOpeningRelationWitness, SimpleKernelError> {
    dummy_rv64im_side_opening_witness(statement)
}

fn setup_rv64im_side_opening_witness_without_packaged_final_main_claims(
    statement: &Rv64imSideOpeningRelationStatement,
    witness: &Rv64imSideOpeningRelationWitness,
) -> Result<Rv64imSideOpeningRelationWitness, SimpleKernelError> {
    let mut setup_witness = setup_rv64im_side_opening_witness(statement, witness)?;
    for packaged in [
        &mut setup_witness.stage1_packaged,
        &mut setup_witness.stage2_packaged,
        &mut setup_witness.stage3_packaged,
        &mut setup_witness.bindings_packaged,
        &mut setup_witness.prepared_steps_packaged,
    ] {
        packaged.final_main_claim_digests = vec![[F::ZERO; 4]; RV64IM_SINGLE_STEP_PACKAGED_FINAL_MAIN_CLAIM_COUNT];
    }
    Ok(setup_witness)
}

fn dummy_single_step_packaged_witness(
    step: &PublicStep,
) -> super::side_claim_relation::Rv64imSingleStepPackagedProofWitness {
    super::side_claim_relation::Rv64imSingleStepPackagedProofWitness {
        step: step.clone(),
        final_main_claim_digests: vec![[F::ZERO; 4]; RV64IM_SINGLE_STEP_PACKAGED_FINAL_MAIN_CLAIM_COUNT],
        proof_digest: [0; 32],
    }
}

fn dummy_stage1_row_binding() -> Stage1RowBinding {
    Stage1RowBinding {
        trace_index: 0,
        step_index: 0,
        sequence_index: 0,
        fetch_pc: 0,
        fetched_word: 0,
        opcode: crate::rv64im::isa::Rv64Opcode::Addi,
        trace_opcode: None,
        trace_virtual_opcode: None,
        family: Rv64FamilyTag::NativeAlu,
        next_pc: 0,
        alu_result: 0,
        effective_addr: None,
        writes_rd: false,
        rd: 0,
        rd_after: 0,
        is_first_in_sequence: false,
        virtual_sequence_remaining: None,
        is_effect_row: false,
        is_commit_row: false,
        is_real: false,
        preserves_x0: true,
    }
}

fn dummy_register_read_event() -> RegisterReadEvent {
    RegisterReadEvent {
        trace_index: 0,
        step_index: 0,
        role: RegisterReadRole::Rs1,
        reg: 0,
        value: 0,
    }
}

fn dummy_register_write_event() -> RegisterWriteEvent {
    RegisterWriteEvent {
        trace_index: 0,
        step_index: 0,
        reg: 0,
        previous: 0,
        next: 0,
    }
}

fn dummy_ram_event() -> RamEvent {
    RamEvent {
        trace_index: 0,
        step_index: 0,
        kind: RamAccessKind::Read,
        addr: 0,
        previous: 0,
        next: 0,
    }
}

fn dummy_twist_link_event() -> TwistLinkEvent {
    TwistLinkEvent {
        trace_index: 0,
        step_index: 0,
        family: Rv64FamilyTag::NativeAlu,
        routed_write_value: None,
        routed_memory_before: None,
        routed_memory_after: None,
    }
}

fn dummy_continuity_event() -> ContinuityEvent {
    ContinuityEvent {
        step_index: 0,
        pc: 0,
        next_pc: 0,
        successor_pc: None,
        final_step: false,
        continuity_holds: false,
    }
}

fn enforce_stage1_claim<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &crate::rv64im::kernel::Stage1SelectedOpeningClaim,
    witness: &Rv64imStage1SelectedRowsWitness,
) -> Result<(), SynthesisError> {
    let first_digest = stage1_row_digest(cs.namespace(|| "first"), &witness.first, "first")?;
    let effect_digest = stage1_row_digest(cs.namespace(|| "effect"), &witness.effect, "effect")?;
    let commit_digest = stage1_row_digest(cs.namespace(|| "commit"), &witness.commit, "commit")?;
    let last_digest = stage1_row_digest(cs.namespace(|| "last"), &witness.last, "last")?;
    enforce_digest_eq(
        &mut cs.namespace(|| "first_digest"),
        &first_digest,
        claim.points.first.value_digest,
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "effect_digest"),
        &effect_digest,
        claim.points.effect.value_digest,
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "commit_digest"),
        &commit_digest,
        claim.points.commit.value_digest,
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| "last_digest"),
        &last_digest,
        claim.points.last.value_digest,
    )?;
    enforce_u64_eq(
        cs.namespace(|| "effect_position"),
        witness.effect_position,
        claim.points.effect.id.logical_index,
    )?;
    enforce_u64_eq(
        cs.namespace(|| "commit_position"),
        witness.commit_position,
        claim.points.commit.id.logical_index,
    )?;
    enforce_u64_eq(
        cs.namespace(|| "first_trace"),
        witness.first.trace_index as u64,
        claim.first_trace_index,
    )?;
    enforce_u64_eq(
        cs.namespace(|| "effect_trace"),
        witness.effect.trace_index as u64,
        claim.effect_trace_index,
    )?;
    enforce_u64_eq(
        cs.namespace(|| "commit_trace"),
        witness.commit.trace_index as u64,
        claim.commit_trace_index,
    )?;
    enforce_u64_eq(
        cs.namespace(|| "last_trace"),
        witness.last.trace_index as u64,
        claim.last_trace_index,
    )?;
    Ok(())
}

fn enforce_stage2_claim<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &crate::rv64im::kernel::Stage2SelectedOpeningClaim,
    witness: &Rv64imStage2SelectedEventsWitness,
) -> Result<(), SynthesisError> {
    match (witness.first_read.as_ref(), claim.points.first_read.as_ref()) {
        (Some(event), Some(reference)) => {
            let digest = register_read_event_digest(cs.namespace(|| "first_read"), event, "first_read")?;
            enforce_digest_eq(&mut cs.namespace(|| "first_read_eq"), &digest, reference.value_digest)?;
        }
        (None, None) => {}
        _ => return Err(SynthesisError::Unsatisfiable),
    }
    match (witness.last_read.as_ref(), claim.points.last_read.as_ref()) {
        (Some(event), Some(reference)) => {
            let digest = register_read_event_digest(cs.namespace(|| "last_read"), event, "last_read")?;
            enforce_digest_eq(&mut cs.namespace(|| "last_read_eq"), &digest, reference.value_digest)?;
        }
        (None, None) => {}
        _ => return Err(SynthesisError::Unsatisfiable),
    }
    match (witness.first_write.as_ref(), claim.points.first_write.as_ref()) {
        (Some(event), Some(reference)) => {
            let digest = register_write_event_digest(cs.namespace(|| "first_write"), event, "first_write")?;
            enforce_digest_eq(&mut cs.namespace(|| "first_write_eq"), &digest, reference.value_digest)?;
        }
        (None, None) => {}
        _ => return Err(SynthesisError::Unsatisfiable),
    }
    match (witness.last_write.as_ref(), claim.points.last_write.as_ref()) {
        (Some(event), Some(reference)) => {
            let digest = register_write_event_digest(cs.namespace(|| "last_write"), event, "last_write")?;
            enforce_digest_eq(&mut cs.namespace(|| "last_write_eq"), &digest, reference.value_digest)?;
        }
        (None, None) => {}
        _ => return Err(SynthesisError::Unsatisfiable),
    }
    match (witness.first_ram.as_ref(), claim.points.first_ram.as_ref()) {
        (Some(event), Some(reference)) => {
            let digest = ram_event_digest(cs.namespace(|| "first_ram"), event, "first_ram")?;
            enforce_digest_eq(&mut cs.namespace(|| "first_ram_eq"), &digest, reference.value_digest)?;
        }
        (None, None) => {}
        _ => return Err(SynthesisError::Unsatisfiable),
    }
    match (witness.last_ram.as_ref(), claim.points.last_ram.as_ref()) {
        (Some(event), Some(reference)) => {
            let digest = ram_event_digest(cs.namespace(|| "last_ram"), event, "last_ram")?;
            enforce_digest_eq(&mut cs.namespace(|| "last_ram_eq"), &digest, reference.value_digest)?;
        }
        (None, None) => {}
        _ => return Err(SynthesisError::Unsatisfiable),
    }
    match (witness.first_twist.as_ref(), claim.points.first_twist.as_ref()) {
        (Some(event), Some(reference)) => {
            let digest = twist_link_event_digest(cs.namespace(|| "first_twist"), event, "first_twist")?;
            enforce_digest_eq(&mut cs.namespace(|| "first_twist_eq"), &digest, reference.value_digest)?;
        }
        (None, None) => {}
        _ => return Err(SynthesisError::Unsatisfiable),
    }
    match (witness.last_twist.as_ref(), claim.points.last_twist.as_ref()) {
        (Some(event), Some(reference)) => {
            let digest = twist_link_event_digest(cs.namespace(|| "last_twist"), event, "last_twist")?;
            enforce_digest_eq(&mut cs.namespace(|| "last_twist_eq"), &digest, reference.value_digest)?;
        }
        (None, None) => {}
        _ => return Err(SynthesisError::Unsatisfiable),
    }
    Ok(())
}

fn enforce_stage3_claim<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim: &crate::rv64im::kernel::Stage3SelectedOpeningClaim,
    witness: &Rv64imStage3SelectedContinuityWitness,
) -> Result<(), SynthesisError> {
    match (
        witness.first_continuity.as_ref(),
        claim.points.first_continuity.as_ref(),
    ) {
        (Some(event), Some(reference)) => {
            let digest = continuity_event_digest(cs.namespace(|| "first_continuity"), event, "first_continuity")?;
            enforce_digest_eq(
                &mut cs.namespace(|| "first_continuity_eq"),
                &digest,
                reference.value_digest,
            )?;
        }
        (None, None) => {}
        _ => return Err(SynthesisError::Unsatisfiable),
    }
    match (witness.last_continuity.as_ref(), claim.points.last_continuity.as_ref()) {
        (Some(event), Some(reference)) => {
            let digest = continuity_event_digest(cs.namespace(|| "last_continuity"), event, "last_continuity")?;
            enforce_digest_eq(
                &mut cs.namespace(|| "last_continuity_eq"),
                &digest,
                reference.value_digest,
            )?;
        }
        (None, None) => {}
        _ => return Err(SynthesisError::Unsatisfiable),
    }
    Ok(())
}

fn validate_rv64im_side_opening_packaged_witness_shapes(
    witness: &Rv64imSideOpeningRelationWitness,
) -> Result<(), SimpleKernelError> {
    validate_rv64im_single_step_packaged_witness_shape("rv64im/stage1", &witness.stage1_packaged)?;
    validate_rv64im_single_step_packaged_witness_shape("rv64im/stage2", &witness.stage2_packaged)?;
    validate_rv64im_single_step_packaged_witness_shape("rv64im/stage3", &witness.stage3_packaged)?;
    validate_rv64im_single_step_packaged_witness_shape(
        "rv64im/kernel_opening_bundle/bindings",
        &witness.bindings_packaged,
    )?;
    validate_rv64im_single_step_packaged_witness_shape(
        "rv64im/kernel_opening_bundle/prepared_steps",
        &witness.prepared_steps_packaged,
    )?;
    Ok(())
}

fn enforce_packaged_opening_digest<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    claim_digest: [u8; 32],
    carried_statement_digest: [u8; 32],
    carried_packaged_digest: [u8; 32],
    witness: &crate::nightstream::rv64im::side_claim_relation::Rv64imSingleStepPackagedProofWitness,
    statement_digest: [AllocatedNum<SpartanF>; 4],
) -> Result<(), SynthesisError> {
    let claim_digest_vars = alloc_digest32_const(&mut cs.namespace(|| "claim_digest"), claim_digest, "claim_digest")?;
    let proof_digest_vars = alloc_digest32_witness(
        &mut cs.namespace(|| "proof_digest"),
        witness.proof_digest,
        "proof_digest",
    )?;
    let claim_digest_values = digest32_as_spartan_fields(claim_digest);
    let carried_statement_digest_values = digest32_as_spartan_fields(carried_statement_digest);
    let proof_digest_values = digest32_as_spartan_fields(witness.proof_digest);

    enforce_digest_eq(
        &mut cs.namespace(|| "statement_digest_eq"),
        &statement_digest,
        carried_statement_digest,
    )?;

    let mut tr = Poseidon2TranscriptCircuit::new(
        cs.namespace(|| "packaged_digest_tr"),
        b"neo.fold.next/rv64im/stage_packaged_opening_claim_proof",
    )?;
    tr.append_fields(
        cs.namespace(|| "claim_digest_append"),
        b"rv64im/stage_packaged_opening_claim_proof/claim_digest",
        &claim_digest_vars,
        &claim_digest_values,
    )?;
    tr.append_fields(
        cs.namespace(|| "statement_digest_append"),
        b"rv64im/stage_packaged_opening_claim_proof/statement_digest",
        &statement_digest,
        &carried_statement_digest_values,
    )?;
    tr.append_fields(
        cs.namespace(|| "proof_digest_append"),
        b"rv64im/stage_packaged_opening_claim_proof/proof_digest",
        &proof_digest_vars,
        &proof_digest_values,
    )?;
    let packaged_digest = tr.digest32(cs.namespace(|| "packaged_digest"))?;
    enforce_digest_eq(
        &mut cs.namespace(|| "packaged_digest_eq"),
        &packaged_digest,
        carried_packaged_digest,
    )
}

fn native_stage_packaged_statement_digest(
    label: &str,
    claim_words: &[u64],
    witness: &crate::nightstream::rv64im::side_claim_relation::Rv64imSingleStepPackagedProofWitness,
) -> Result<[u8; 32], SynthesisError> {
    let expected_step =
        build_claim_packaged_public_step(label, claim_words).map_err(|_| SynthesisError::Unsatisfiable)?;
    native_exact_packaged_statement_digest(&expected_step, witness)
}

fn native_exact_packaged_statement_digest(
    expected_step: &PublicStep,
    witness: &crate::nightstream::rv64im::side_claim_relation::Rv64imSingleStepPackagedProofWitness,
) -> Result<[u8; 32], SynthesisError> {
    if !same_public_step(&witness.step, expected_step) {
        return Err(SynthesisError::Unsatisfiable);
    }
    Ok(single_step_packaged_statement_digest(
        expected_step,
        &witness.final_main_claim_digests,
    ))
}

fn alloc_digest32_const<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    digest: [u8; 32],
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let values = digest32_as_spartan_fields(digest);
    Ok(core::array::from_fn(|idx| {
        alloc_constant(
            cs.namespace(|| format!("{label}_{idx}")),
            values[idx],
            &format!("{label}_{idx}"),
        )
        .expect("digest witness allocation must succeed")
    }))
}

fn alloc_digest32_witness<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    digest: [u8; 32],
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let values = digest32_as_spartan_fields(digest);
    Ok(core::array::from_fn(|idx| {
        AllocatedNum::alloc(cs.namespace(|| format!("{label}_{idx}")), || Ok(values[idx]))
            .expect("digest witness allocation must succeed")
    }))
}

fn enforce_digest_eq<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    actual: &[AllocatedNum<SpartanF>; 4],
    expected: [u8; 32],
) -> Result<(), SynthesisError> {
    let expected_vars = alloc_digest32_const(cs, expected, "expected_digest")?;
    for idx in 0..4 {
        cs.enforce(
            || format!("digest_eq_{idx}"),
            |lc| lc + actual[idx].get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + expected_vars[idx].get_variable(),
        );
    }
    Ok(())
}

fn enforce_digest_eq_public_inputs<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    actual: &[AllocatedNum<SpartanF>; 4],
    expected: &[AllocatedNum<SpartanF>],
) -> Result<(), SynthesisError> {
    if expected.len() != actual.len() {
        return Err(SynthesisError::Unsatisfiable);
    }
    for idx in 0..4 {
        cs.enforce(
            || format!("public_digest_eq_{idx}"),
            |lc| lc + actual[idx].get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + expected[idx].get_variable(),
        );
    }
    Ok(())
}

fn enforce_u64_eq<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    actual: u64,
    expected: u64,
) -> Result<(), SynthesisError> {
    let actual_var = AllocatedNum::alloc(cs.namespace(|| "actual"), || Ok(SpartanF::from_canonical_u64(actual)))?;
    let expected_var = alloc_constant(
        cs.namespace(|| "expected"),
        SpartanF::from_canonical_u64(expected),
        "expected",
    )?;
    cs.enforce(
        || "u64_eq",
        |lc| lc + actual_var.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + expected_var.get_variable(),
    );
    Ok(())
}

fn alloc_constant<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    value: SpartanF,
    label: &str,
) -> Result<AllocatedNum<SpartanF>, SynthesisError> {
    let out = AllocatedNum::alloc(cs.namespace(|| label.to_string()), || Ok(value))?;
    cs.enforce(
        || format!("{label}_constant"),
        |lc| lc + out.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + (value, CS::one()),
    );
    Ok(out)
}

fn digest32_as_spartan_fields(digest: [u8; 32]) -> [SpartanF; 4] {
    digest32_as_fields(digest).map(|value| SpartanF::from_canonical_u64(value.as_canonical_u64()))
}
