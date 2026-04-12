//! Owns the authoritative RV64IM Nightstream side theorem witness relation.
//!
//! This relation is the minimal fixed theorem boundary the eventual side
//! Spartan circuit must prove. It is witness-backed and does not route theorem
//! meaning through the compact opening artifact.

use crate::nightstream::NightstreamStatement;
use crate::rv64im::kernel::{
    build_commitment_vector, Rv64imAcceptedProofArtifact, Rv64imProofStatement, SimpleKernelError,
};
use crate::rv64im::FamilyEvalClaimWitness;

use super::opening_artifact::build_rv64im_opening_artifact_from_accepted_artifact;
use super::opening_artifact::build_rv64im_opening_artifact_from_claim_witnesses_and_side_bundle;
use super::side_eval_claim_relation::{
    build_rv64im_phase0_opened_object_bundle_from_claim_witnesses, build_rv64im_side_eval_claim_relation_statement,
    build_rv64im_side_eval_claim_relation_witness_from_accepted_artifact_and_side_bundle,
    verify_rv64im_side_eval_claim_relation, Rv64imSideEvalClaimRelationWitness,
};
use super::side_opening_relation::{
    build_rv64im_side_selected_opening_witness_from_accepted_artifact,
    verify_rv64im_side_selected_opening_witness_against_compact_surfaces, Rv64imSideSelectedOpeningWitness,
};
use super::witness_backed_side_bridge::{
    build_rv64im_witness_backed_side_bridge_statement, Rv64imWitnessBackedSideBridgeStatement,
};
use super::{
    bind_rv64im_side_proof_bundle_to_statement_core,
    verify_rv64im_kernel_export_source_surface_against_compact_surfaces,
    verify_rv64im_root_execution_surface_against_compact_surfaces, Rv64imSideProofBundle,
};

#[derive(Clone, Debug)]
pub struct Rv64imDirectSideRelationWitness {
    pub public_statement: Rv64imProofStatement,
    pub side_bundle: Rv64imSideProofBundle,
    pub opening_witness: Rv64imSideSelectedOpeningWitness,
    pub phase0_claim_witnesses: Vec<FamilyEvalClaimWitness>,
}

pub fn build_rv64im_direct_side_relation_witness_from_accepted_artifact(
    side_bundle: &Rv64imSideProofBundle,
    accepted_artifact: &Rv64imAcceptedProofArtifact,
) -> Result<Rv64imDirectSideRelationWitness, SimpleKernelError> {
    let phase0_witness = build_rv64im_side_eval_claim_relation_witness_from_accepted_artifact_and_side_bundle(
        side_bundle,
        accepted_artifact,
    )?;
    Ok(Rv64imDirectSideRelationWitness {
        public_statement: accepted_artifact.statement.clone(),
        side_bundle: side_bundle.clone(),
        opening_witness: build_rv64im_side_selected_opening_witness_from_accepted_artifact(accepted_artifact),
        phase0_claim_witnesses: phase0_witness.claim_witnesses,
    })
}

pub fn build_rv64im_direct_side_relation_from_accepted_artifact(
    nightstream_statement: &NightstreamStatement,
    side_bundle: &Rv64imSideProofBundle,
    accepted_artifact: &Rv64imAcceptedProofArtifact,
) -> Result<(Rv64imWitnessBackedSideBridgeStatement, Rv64imDirectSideRelationWitness), SimpleKernelError> {
    let bound_side_bundle =
        bind_rv64im_side_proof_bundle_to_statement_core(side_bundle, nightstream_statement.core_digest())?;
    let opening_artifact = build_rv64im_opening_artifact_from_accepted_artifact(
        &accepted_artifact.statement,
        &bound_side_bundle,
        accepted_artifact,
    )?;
    let statement = build_rv64im_witness_backed_side_bridge_statement(nightstream_statement, opening_artifact.digest)?;
    let witness =
        build_rv64im_direct_side_relation_witness_from_accepted_artifact(&bound_side_bundle, accepted_artifact)?;
    Ok((statement, witness))
}

pub fn verify_rv64im_direct_side_relation(
    statement: &Rv64imWitnessBackedSideBridgeStatement,
    witness: &Rv64imDirectSideRelationWitness,
) -> Result<(), SimpleKernelError> {
    if witness.side_bundle.statement_core_digest != statement.nightstream_statement.core_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM direct side relation side bundle does not match the carried Nightstream statement core".into(),
        ));
    }
    if witness.public_statement.digest != witness.public_statement.expected_digest() {
        return Err(SimpleKernelError::Bridge(
            "RV64IM direct side relation public statement digest mismatch".into(),
        ));
    }
    if statement.nightstream_statement.public_io_digest != witness.public_statement.digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM direct side relation Nightstream public IO does not match the carried RV64IM statement".into(),
        ));
    }
    if statement.nightstream_statement.fold_schedule != witness.public_statement.fold_schedule {
        return Err(SimpleKernelError::Bridge(
            "RV64IM direct side relation fold schedule does not match the carried RV64IM statement".into(),
        ));
    }
    if statement.nightstream_statement.chunk_summaries.len() as u64 != witness.public_statement.chunk_count {
        return Err(SimpleKernelError::Bridge(
            "RV64IM direct side relation chunk count does not match the carried RV64IM statement".into(),
        ));
    }

    verify_rv64im_side_selected_opening_witness_against_compact_surfaces(
        &witness.public_statement,
        &witness.side_bundle,
        &witness.opening_witness,
    )?;
    verify_phase0_claim_witness_consistency(&witness.phase0_claim_witnesses)?;
    let opening_artifact = build_rv64im_opening_artifact_from_claim_witnesses_and_side_bundle(
        &witness.public_statement,
        &witness.side_bundle,
        &witness.phase0_claim_witnesses,
    )?;
    if statement.opening_artifact_digest != opening_artifact.digest {
        return Err(SimpleKernelError::Bridge(
            "RV64IM direct side relation opening artifact digest does not match the carried side witness".into(),
        ));
    }

    let phase0_opened_objects =
        build_rv64im_phase0_opened_object_bundle_from_claim_witnesses(&witness.phase0_claim_witnesses)?;
    let phase0_bundle =
        crate::rv64im::build_rv64im_eval_claim_bundle_from_claim_witnesses(&witness.phase0_claim_witnesses)?;
    let phase0_statement = build_rv64im_side_eval_claim_relation_statement(
        &witness.public_statement,
        &witness.side_bundle,
        &phase0_opened_objects,
        &phase0_bundle,
    )?;
    let phase0_witness = Rv64imSideEvalClaimRelationWitness {
        claim_witnesses: witness.phase0_claim_witnesses.clone(),
    };
    verify_rv64im_side_eval_claim_relation(&phase0_statement, &phase0_witness)?;

    verify_rv64im_root_execution_surface_against_compact_surfaces(
        &statement.nightstream_statement,
        &witness.side_bundle,
        &witness.public_statement,
    )?;
    verify_rv64im_kernel_export_source_surface_against_compact_surfaces(
        &witness.side_bundle,
        &witness.public_statement,
    )?;
    Ok(())
}

fn verify_phase0_claim_witness_consistency(
    claim_witnesses: &[FamilyEvalClaimWitness],
) -> Result<(), SimpleKernelError> {
    for claim_witness in claim_witnesses {
        FamilyEvalClaimWitness::new(claim_witness.claim.clone(), claim_witness.witness.clone()).map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV64IM direct side relation Phase 0 {:?} claim witness is internally inconsistent: {err}",
                claim_witness.claim.payload.schema
            ))
        })?;
        let rebuilt_commitment_vector = build_commitment_vector(
            claim_witness.claim.payload.schema,
            &claim_witness.witness.packed_columns,
        )
        .map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "RV64IM direct side relation Phase 0 {:?} could not rebuild the commitment vector: {err}",
                claim_witness.claim.payload.schema
            ))
        })?;
        if rebuilt_commitment_vector != claim_witness.witness.commitment_vector {
            return Err(SimpleKernelError::Bridge(format!(
                "RV64IM direct side relation Phase 0 {:?} packed columns do not match the carried commitment vector",
                claim_witness.claim.payload.schema
            )));
        }
    }
    Ok(())
}
