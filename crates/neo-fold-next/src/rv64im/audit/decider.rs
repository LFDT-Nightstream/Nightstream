//! Owns audit helpers for the RV64IM main-recursion SNARK surface.

use crate::finalize::FixedShapeChunkSummary;
use crate::rv64im::final_relation::{Rv64imChunkTransitionWitness, Rv64imFinalBuildProof, Rv64imFinalStatement};
use crate::rv64im::kernel::Rv64imKernelExportProof;
use crate::rv64im::SimpleKernelError;

pub use crate::rv64im::decider::{
    build_rv64im_published_proof_seam, build_rv64im_published_proof_seam_with_perf,
    prove_rv64im_public_proof_and_published_seam_with_perf, Rv64imPublicProofAndSeamBuildPerf,
    Rv64imPublishedProofSeam, Rv64imPublishedProofSeamBuildPerf,
};
pub use crate::rv64im::ivc_snark::{Rv64imIvcRecursionSnarkSetupShape, Rv64imTerminalFPrimeCommittedStepShape};

pub fn build_rv64im_ivc_recursion_snark_setup_shape_from_components(
    statement: &Rv64imFinalStatement,
    proof_digest: [u8; 32],
    kernel_export: &Rv64imKernelExportProof,
    chunk_summaries: &[FixedShapeChunkSummary],
    steps: &[Rv64imChunkTransitionWitness],
) -> Result<Rv64imIvcRecursionSnarkSetupShape, SimpleKernelError> {
    crate::rv64im::ivc_snark::build_rv64im_ivc_recursion_snark_setup_shape_from_components(
        statement,
        proof_digest,
        kernel_export,
        chunk_summaries,
        steps,
    )
}

pub fn debug_check_rv64im_ivc_recursion_snark_circuit(
    statement: &Rv64imFinalStatement,
    proof: &Rv64imFinalBuildProof,
) -> Result<(), SimpleKernelError> {
    crate::rv64im::ivc_snark::debug_check_rv64im_ivc_recursion_snark_circuit(statement, proof)
}

pub fn debug_measure_rv64im_terminal_f_prime_committed_step_shape(
    spartan_shape: &crate::rv64im::main_relation_spartan::Rv64imMainRecursionStepSpartanShape,
    backend_relation: &crate::rv64im::main_relation_spartan::Rv64imMainRecursionFPrimeBackendRelation,
) -> Result<Rv64imTerminalFPrimeCommittedStepShape, SimpleKernelError> {
    crate::rv64im::ivc_snark::debug_measure_rv64im_terminal_f_prime_committed_step_shape(
        spartan_shape,
        backend_relation,
    )
}
