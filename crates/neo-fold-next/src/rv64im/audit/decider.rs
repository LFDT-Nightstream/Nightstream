//! Owns audit helpers for the direct main-relation Spartan compatibility surface.

pub use crate::rv64im::decider::{
    build_rv64im_published_proof_seam, build_rv64im_published_proof_seam_with_perf,
    prove_rv64im_public_proof_and_published_seam_with_options_and_perf,
    prove_rv64im_public_proof_and_published_seam_with_perf, prove_rv64im_spartan2_decider,
    prove_rv64im_spartan2_decider_cached, verify_rv64im_spartan2_decider, Rv64imPublicProofAndSeamBuildPerf,
    Rv64imPublishedProofSeam, Rv64imPublishedProofSeamBuildPerf,
};
pub use crate::rv64im::main_relation::{
    build_rv64im_decider_relation_from_final_surface, validate_rv64im_decider_relation_surface, Rv64imDeciderRelation,
};
pub use crate::rv64im::main_relation_spartan::{
    build_rv64im_spartan2_decider_setup_shape_from_components, debug_check_rv64im_spartan2_decider_circuit,
    inspect_rv64im_spartan2_decider_trace, measure_rv64im_spartan2_decider_circuit,
    setup_rv64im_spartan2_decider_cached_from_shape, setup_rv64im_spartan2_decider_from_shape,
    Rv64imMainRelationCircuitMetrics, Rv64imMainRelationCountBucket, Rv64imMainRelationHotspotDetail,
    Rv64imMainRelationPhaseBucket, Rv64imMainRelationSetupShape, Rv64imMainRelationSurfaceFamilyBucket,
    Rv64imMainRelationSurfaceMetrics, Rv64imMainRelationTraceStats, Rv64imSpartan2DeciderError,
    Rv64imSpartan2DeciderKeyPair, Rv64imSpartan2DeciderProof, Rv64imSpartan2DeciderProverKey,
    Rv64imSpartan2DeciderVerifierKey,
};
