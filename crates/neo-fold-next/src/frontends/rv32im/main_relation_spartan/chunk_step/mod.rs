//! Owns fixed-shape chunk-step circuit helpers for RV32IM main recursion.

mod ivc;
mod recursive;

pub use ivc::{
    build_rv32im_chunk_step_ivc_recursive_step_cover_shape, build_rv32im_chunk_step_ivc_recursive_step_padding,
    build_rv32im_chunk_step_ivc_recursive_step_padding_from_shape, build_rv32im_chunk_step_ivc_shape,
    Rv32imChunkStepIvcRecursiveStepPadding, Rv32imChunkStepIvcShape, Rv32imChunkStepIvcSpartanError,
};
pub use recursive::{
    build_rv32im_main_recursion_f_prime_backend_relations,
    build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape,
    build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices,
    build_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices_and_perf,
    build_rv32im_main_recursion_f_prime_claim_cover, build_rv32im_main_recursion_f_prime_payload,
    build_rv32im_main_recursion_f_prime_payloads, build_rv32im_main_recursion_f_prime_payloads_with_spartan_shape,
    build_rv32im_main_recursion_step_spartan_shape,
    debug_check_rv32im_chunk_step_recursive_effective_chunk_trace_matches_native,
    debug_check_rv32im_main_recursion_f_prime_backend_relation_semantics,
    debug_trace_rv32im_main_recursion_f_prime_backend_relations_with_spartan_shape_from_advices, Rv32imCcsClaimShape,
    Rv32imCcsWitnessShape, Rv32imCeClaimDigestShape, Rv32imMainRecursionFPrimeBackendRelation,
    Rv32imMainRecursionFPrimeBackendRelationBuildPerf, Rv32imMainRecursionFPrimeClaimCover,
    Rv32imMainRecursionFPrimePayload, Rv32imMainRecursionStepSpartanShape,
};
pub(crate) use recursive::{
    build_rv32im_main_recursion_step_spartan_statement, rv32im_chunk_step_recursive_carry_state_digest,
};
