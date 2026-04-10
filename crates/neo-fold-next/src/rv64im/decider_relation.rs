//! Owns the compatibility adapter from the RV64IM main relation into the generic decider backend relation.

use crate::decider::spartan2::{validate_spartan2_decider_relation_surface, Spartan2DeciderRelation};
use crate::rv64im::final_relation::{Rv64imFinalProof, Rv64imFinalProofComponentDigests, Rv64imFinalStatement};
use crate::rv64im::kernel::{Rv64imKernelExportRelationResult, Rv64imProof, SimpleKernelError};
use crate::rv64im::main_relation::{
    build_rv64im_main_relation, build_rv64im_main_relation_backend_relation_from_artifact,
    build_rv64im_main_relation_backend_relation_from_verified_artifact_with_component_digests,
    build_rv64im_main_relation_from_final, build_rv64im_main_relation_from_verified_final_with_component_digests,
};

pub type Rv64imDeciderRelation = Spartan2DeciderRelation;

pub fn validate_rv64im_decider_relation_surface(relation: &Rv64imDeciderRelation) -> Result<(), SimpleKernelError> {
    validate_spartan2_decider_relation_surface(relation).map_err(|err| SimpleKernelError::Bridge(err.to_string()))
}

pub fn build_rv64im_decider_relation(proof: &Rv64imProof) -> Result<Rv64imDeciderRelation, SimpleKernelError> {
    let main_relation = build_rv64im_main_relation(proof)?;
    build_rv64im_main_relation_backend_relation_from_artifact(&main_relation)
}

pub fn verify_rv64im_decider_relation(
    relation: &Rv64imDeciderRelation,
    proof: &Rv64imProof,
) -> Result<(), SimpleKernelError> {
    let expected = build_rv64im_decider_relation(proof)?;
    if relation != &expected {
        return Err(SimpleKernelError::Bridge(
            "RV64IM decider relation does not match the carried public proof seam".into(),
        ));
    }
    Ok(())
}

pub fn build_rv64im_decider_relation_from_final(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalProof,
) -> Result<Rv64imDeciderRelation, SimpleKernelError> {
    let main_relation = build_rv64im_main_relation_from_final(statement, proof)?;
    build_rv64im_main_relation_backend_relation_from_artifact(&main_relation)
}

pub(crate) fn build_rv64im_decider_relation_from_verified_final_with_component_digests(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalProof,
    verified_kernel: &Rv64imKernelExportRelationResult,
    component_digests: &Rv64imFinalProofComponentDigests,
) -> Result<Rv64imDeciderRelation, SimpleKernelError> {
    let main_relation = build_rv64im_main_relation_from_verified_final_with_component_digests(
        statement,
        proof,
        verified_kernel,
        component_digests,
    )?;
    build_rv64im_main_relation_backend_relation_from_verified_artifact_with_component_digests(
        &main_relation,
        verified_kernel,
        component_digests,
    )
}
