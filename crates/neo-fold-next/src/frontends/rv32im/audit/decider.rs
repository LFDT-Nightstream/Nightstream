//! Owns audit helpers for the RV32IM main-recursion SNARK surface.

use crate::finalize::FixedShapeChunkSummary;
use crate::rv32im::final_relation::{Rv32imChunkTransitionWitness, Rv32imFinalBuildProof, Rv32imFinalStatement};
use crate::rv32im::kernel::Rv32imKernelExportProof;
use crate::rv32im::SimpleKernelError;

pub use crate::rv32im::decider::{
    build_rv32im_published_proof_seam_with_perf, prove_rv32im_public_proof_and_published_seam_with_perf,
    Rv32imPublicProofAndSeamBuildPerf, Rv32imPublishedProofSeam, Rv32imPublishedProofSeamBuildPerf,
};
pub use crate::rv32im::ivc_snark::{Rv32imIvcRecursionSnarkSetupShape, Rv32imTerminalFPrimeCommittedStepShape};

pub fn build_rv32im_ivc_recursion_snark_setup_shape_from_components(
    statement: &Rv32imFinalStatement,
    proof_digest: [u8; 32],
    kernel_export: &Rv32imKernelExportProof,
    chunk_summaries: &[FixedShapeChunkSummary],
    steps: &[Rv32imChunkTransitionWitness],
) -> Result<Rv32imIvcRecursionSnarkSetupShape, SimpleKernelError> {
    crate::rv32im::ivc_snark::build_rv32im_ivc_recursion_snark_setup_shape_from_components(
        statement,
        proof_digest,
        kernel_export,
        chunk_summaries,
        steps,
    )
}

pub fn debug_check_rv32im_ivc_recursion_snark_circuit(
    statement: &Rv32imFinalStatement,
    proof: &Rv32imFinalBuildProof,
) -> Result<(), SimpleKernelError> {
    crate::rv32im::ivc_snark::debug_check_rv32im_ivc_recursion_snark_circuit(statement, proof)
}

pub fn debug_measure_rv32im_terminal_f_prime_committed_step_shape(
    spartan_shape: &crate::rv32im::main_relation_spartan::Rv32imMainRecursionStepSpartanShape,
    backend_relation: &crate::rv32im::main_relation_spartan::Rv32imMainRecursionFPrimeBackendRelation,
) -> Result<Rv32imTerminalFPrimeCommittedStepShape, SimpleKernelError> {
    crate::rv32im::ivc_snark::debug_measure_rv32im_terminal_f_prime_committed_step_shape(
        spartan_shape,
        backend_relation,
    )
}
