//! Owns debug and shape-inspection helpers for the side-opening Spartan shell.

use bellpepper_core::{test_cs::TestConstraintSystem, Comparable, ConstraintSystem, Delta};
use neo_math::F;
use spartan2::traits::{circuit::SpartanCircuit, snark::R1CSSNARKTrait};

use super::*;

pub fn debug_check_rv32im_side_opening_spartan_circuit(
    statement: &Rv32imSideOpeningRelationStatement,
    witness: &Rv32imSideOpeningRelationWitness,
) -> Result<(), SimpleKernelError> {
    let circuit = Rv32imSideOpeningSpartanCircuit {
        statement: statement.clone(),
        witness: witness.clone(),
    };
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    circuit.synthesize(&mut cs, &[], &[], None).map_err(|err| {
        SimpleKernelError::Bridge(format!("RV32IM side opening Spartan debug synthesis failed: {err}"))
    })?;
    if !cs.is_satisfied() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV32IM side opening Spartan circuit unsatisfied: {}",
            cs.which_is_unsatisfied()
                .unwrap_or_else(|| "unknown".into())
        )));
    }
    Ok(())
}

pub fn debug_measure_rv32im_side_opening_spartan_circuit_shape(
    statement: &Rv32imSideOpeningRelationStatement,
    witness: &Rv32imSideOpeningRelationWitness,
) -> Result<Rv32imSideOpeningSpartanCircuitShape, SimpleKernelError> {
    let circuit = Rv32imSideOpeningSpartanCircuit {
        statement: statement.clone(),
        witness: witness.clone(),
    };
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    circuit.synthesize(&mut cs, &[], &[], None).map_err(|err| {
        SimpleKernelError::Bridge(format!("RV32IM side opening Spartan debug synthesis failed: {err}"))
    })?;
    Ok(Rv32imSideOpeningSpartanCircuitShape {
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

pub fn debug_compare_rv32im_side_opening_spartan_statement_owned_shape(
    statement: &Rv32imSideOpeningRelationStatement,
    witness: &Rv32imSideOpeningRelationWitness,
) -> Result<Option<String>, SimpleKernelError> {
    let real_circuit = Rv32imSideOpeningSpartanCircuit {
        statement: statement.clone(),
        witness: witness.clone(),
    };
    let dummy_circuit = Rv32imSideOpeningSpartanCircuit {
        statement: statement.clone(),
        witness: dummy_rv32im_side_opening_witness(statement)?,
    };
    let mut real_cs = TestConstraintSystem::<SpartanF>::new();
    real_circuit
        .synthesize(&mut real_cs, &[], &[], None)
        .map_err(|err| {
            SimpleKernelError::Bridge(format!("RV32IM side opening Spartan real synthesis failed: {err}"))
        })?;
    let mut dummy_cs = TestConstraintSystem::<SpartanF>::new();
    dummy_circuit
        .synthesize(&mut dummy_cs, &[], &[], None)
        .map_err(|err| {
            SimpleKernelError::Bridge(format!("RV32IM side opening Spartan dummy synthesis failed: {err}"))
        })?;
    Ok(match real_cs.delta(&dummy_cs, false) {
        Delta::Equal => None,
        delta => Some(format!("{delta:?}")),
    })
}

pub fn debug_compare_rv32im_side_opening_spartan_setup_shape(
    statement: &Rv32imSideOpeningRelationStatement,
    witness: &Rv32imSideOpeningRelationWitness,
) -> Result<Option<String>, SimpleKernelError> {
    compare_side_opening_circuit_delta(
        statement,
        witness,
        &setup_rv32im_side_opening_witness(statement, witness)?,
    )
}

pub fn debug_compare_rv32im_side_opening_spartan_without_packaged_final_main_claims_shape(
    statement: &Rv32imSideOpeningRelationStatement,
    witness: &Rv32imSideOpeningRelationWitness,
) -> Result<Option<String>, SimpleKernelError> {
    compare_side_opening_circuit_delta(
        statement,
        witness,
        &setup_rv32im_side_opening_witness_without_packaged_final_main_claims(statement, witness)?,
    )
}

pub fn debug_compare_rv32im_stage1_packaged_opening_digest_without_packaged_final_main_claims_shape(
    statement: &Rv32imSideOpeningRelationStatement,
    witness: &Rv32imSideOpeningRelationWitness,
) -> Result<Option<String>, SimpleKernelError> {
    let reduced = setup_rv32im_side_opening_witness_without_packaged_final_main_claims(statement, witness)?;
    compare_stage_packaged_opening_digest_delta(statement, &witness.stage1_packaged, &reduced.stage1_packaged)
}

pub fn debug_compare_rv32im_stage1_packaged_opening_digest_zeroing_only_final_main_claims_shape(
    statement: &Rv32imSideOpeningRelationStatement,
    witness: &Rv32imSideOpeningRelationWitness,
) -> Result<Option<String>, SimpleKernelError> {
    let mut reduced = witness.stage1_packaged.clone();
    reduced.final_main_claim_digests = vec![[F::ZERO; 4]; RV32IM_SINGLE_STEP_PACKAGED_FINAL_MAIN_CLAIM_COUNT];
    compare_stage_packaged_opening_digest_delta(statement, &witness.stage1_packaged, &reduced)
}

pub fn debug_compare_rv32im_stage1_packaged_opening_digest_zeroing_final_main_claims_with_fixed_native_statement_shape(
    statement: &Rv32imSideOpeningRelationStatement,
    witness: &Rv32imSideOpeningRelationWitness,
) -> Result<Option<String>, SimpleKernelError> {
    let mut reduced = witness.stage1_packaged.clone();
    reduced.final_main_claim_digests = vec![[F::ZERO; 4]; RV32IM_SINGLE_STEP_PACKAGED_FINAL_MAIN_CLAIM_COUNT];
    let fixed_native_statement_digest = native_stage_packaged_statement_digest(
        "rv32im/stage1",
        &statement.stage1.claim.claim_words(),
        &witness.stage1_packaged,
    )
    .map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV32IM stage1 packaged fixed native statement digest failed: {err}"
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
    statement: &Rv32imSideOpeningRelationStatement,
    witness: &crate::public_proof::rv32im::side_claim_relation::Rv32imSingleStepPackagedProofWitness,
) -> Result<[u8; 32], SimpleKernelError> {
    native_stage_packaged_statement_digest("rv32im/stage1", &statement.stage1.claim.claim_words(), witness)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV32IM stage1 packaged native digest failed: {err}")))
}

pub fn debug_round_trip_rv32im_stage1_packaged_opening_digest_with_reduced_setup(
    statement: &Rv32imSideOpeningRelationStatement,
    witness: &Rv32imSideOpeningRelationWitness,
) -> Result<(), SimpleKernelError> {
    validate_rv32im_side_opening_packaged_witness_shapes(witness)?;
    let mut setup_stage1_witness = witness.stage1_packaged.clone();
    setup_stage1_witness.final_main_claim_digests =
        vec![[F::ZERO; 4]; RV32IM_SINGLE_STEP_PACKAGED_FINAL_MAIN_CLAIM_COUNT];
    let (pk, vk) = Rv32imSideOpeningSpartanSnark::setup(DebugStage1PackagedOpeningDigestCircuit {
        claim: statement.stage1.claim.clone(),
        carried_statement_digest: statement.stage1.packaged_statement_digest,
        carried_packaged_digest: statement.stage1.packaged_digest,
        witness: setup_stage1_witness,
    })
    .map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV32IM reduced stage1 packaged-opening digest setup failed: {err}"
        ))
    })?;
    let prove_circuit = DebugStage1PackagedOpeningDigestCircuit {
        claim: statement.stage1.claim.clone(),
        carried_statement_digest: statement.stage1.packaged_statement_digest,
        carried_packaged_digest: statement.stage1.packaged_digest,
        witness: witness.stage1_packaged.clone(),
    };
    let prep = Rv32imSideOpeningSpartanSnark::prep_prove(&pk, prove_circuit.clone(), true).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV32IM reduced stage1 packaged-opening digest prepare failed: {err}"
        ))
    })?;
    let proof = Rv32imSideOpeningSpartanSnark::prove(&pk, prove_circuit, &prep, true).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV32IM reduced stage1 packaged-opening digest prove failed: {err}"
        ))
    })?;
    proof.verify(&vk).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "RV32IM reduced stage1 packaged-opening digest verify failed: {err}"
        ))
    })?;
    Ok(())
}

fn compare_side_opening_circuit_delta(
    statement: &Rv32imSideOpeningRelationStatement,
    lhs_witness: &Rv32imSideOpeningRelationWitness,
    rhs_witness: &Rv32imSideOpeningRelationWitness,
) -> Result<Option<String>, SimpleKernelError> {
    let lhs_circuit = Rv32imSideOpeningSpartanCircuit {
        statement: statement.clone(),
        witness: lhs_witness.clone(),
    };
    let rhs_circuit = Rv32imSideOpeningSpartanCircuit {
        statement: statement.clone(),
        witness: rhs_witness.clone(),
    };
    let mut lhs_cs = TestConstraintSystem::<SpartanF>::new();
    lhs_circuit
        .synthesize(&mut lhs_cs, &[], &[], None)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV32IM side opening Spartan lhs synthesis failed: {err}")))?;
    let mut rhs_cs = TestConstraintSystem::<SpartanF>::new();
    rhs_circuit
        .synthesize(&mut rhs_cs, &[], &[], None)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV32IM side opening Spartan rhs synthesis failed: {err}")))?;
    Ok(match lhs_cs.delta(&rhs_cs, false) {
        Delta::Equal => None,
        delta => Some(format!("{delta:?}")),
    })
}

fn compare_stage_packaged_opening_digest_delta(
    statement: &Rv32imSideOpeningRelationStatement,
    lhs_witness: &crate::public_proof::rv32im::side_claim_relation::Rv32imSingleStepPackagedProofWitness,
    rhs_witness: &crate::public_proof::rv32im::side_claim_relation::Rv32imSingleStepPackagedProofWitness,
) -> Result<Option<String>, SimpleKernelError> {
    compare_stage_packaged_opening_digest_delta_with_carried_statement(statement, lhs_witness, rhs_witness)
}

fn compare_stage_packaged_opening_digest_delta_with_fixed_native_statement(
    statement: &Rv32imSideOpeningRelationStatement,
    lhs_witness: &crate::public_proof::rv32im::side_claim_relation::Rv32imSingleStepPackagedProofWitness,
    rhs_witness: &crate::public_proof::rv32im::side_claim_relation::Rv32imSingleStepPackagedProofWitness,
    _: [u8; 32],
) -> Result<Option<String>, SimpleKernelError> {
    compare_stage_packaged_opening_digest_delta_with_carried_statement(statement, lhs_witness, rhs_witness)
}

fn compare_stage_packaged_opening_digest_delta_with_carried_statement(
    statement: &Rv32imSideOpeningRelationStatement,
    lhs_witness: &crate::public_proof::rv32im::side_claim_relation::Rv32imSingleStepPackagedProofWitness,
    rhs_witness: &crate::public_proof::rv32im::side_claim_relation::Rv32imSingleStepPackagedProofWitness,
) -> Result<Option<String>, SimpleKernelError> {
    let mut lhs_cs = TestConstraintSystem::<SpartanF>::new();
    let lhs_statement_digest = stage1_opening_packaged_statement_digest(
        lhs_cs.namespace(|| "stage1_statement_digest"),
        &statement.stage1.claim,
        &lhs_witness.final_main_claim_digests,
        "stage1_statement_digest",
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV32IM stage1 packaged lhs digest synthesis failed: {err}")))?;
    enforce_packaged_opening_digest(
        &mut lhs_cs.namespace(|| "stage1_packaged"),
        statement.stage1.claim.digest,
        statement.stage1.packaged_statement_digest,
        statement.stage1.packaged_digest,
        lhs_witness,
        lhs_statement_digest,
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV32IM stage1 packaged lhs enforce failed: {err}")))?;

    let mut rhs_cs = TestConstraintSystem::<SpartanF>::new();
    let rhs_statement_digest = stage1_opening_packaged_statement_digest(
        rhs_cs.namespace(|| "stage1_statement_digest"),
        &statement.stage1.claim,
        &rhs_witness.final_main_claim_digests,
        "stage1_statement_digest",
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV32IM stage1 packaged rhs digest synthesis failed: {err}")))?;
    enforce_packaged_opening_digest(
        &mut rhs_cs.namespace(|| "stage1_packaged"),
        statement.stage1.claim.digest,
        statement.stage1.packaged_statement_digest,
        statement.stage1.packaged_digest,
        rhs_witness,
        rhs_statement_digest,
    )
    .map_err(|err| SimpleKernelError::Bridge(format!("RV32IM stage1 packaged rhs enforce failed: {err}")))?;

    Ok(match lhs_cs.delta(&rhs_cs, false) {
        Delta::Equal => None,
        delta => Some(format!("{delta:?}")),
    })
}
