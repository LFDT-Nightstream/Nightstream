//! Owns the RV32IM published Nightstream proof boundary above the current final/decider seam.

pub mod authoritative_side;
mod compact_surfaces;
mod flow;
pub mod opening_artifact;
pub mod proof;
pub mod side_bridges;
pub mod side_bundle;
pub mod side_claim_relation;
pub mod side_eval_claim_relation;
pub mod side_opening_relation;
pub mod side_opening_spartan;
pub mod side_proof;
pub mod side_relation_circuit;
pub mod side_relation_spartan;
mod side_runtime_binding;
pub mod statement;
pub mod surfaces;

pub use self::authoritative_side::{
    build_rv32im_side_binding_statement, validate_rv32im_side_opening_public, verify_rv32im_side_opening_native,
    Rv32imEvalPublic, Rv32imOpenedObjectPublic, Rv32imSideBindingStatement, Rv32imSideOpeningProof,
    Rv32imSideOpeningPublic, Rv32imSideSurfacePublic, Rv32imSideSurfaceTarget,
};
pub use self::flow::{
    build_rv32im_nightstream_from_public_proof_with_perf, build_rv32im_nightstream_from_published_proof_seam_with_perf,
    Rv32imNightstreamBuildPerf, Rv32imNightstreamSeamBuildPerf,
};
pub use self::flow::{verify_rv32im_nightstream_with_perf, Rv32imNightstreamVerifyPerf};
pub use self::proof::{rv32im_main_nightstream_proof_digest, Rv32imNightstreamProof, Rv32imSideProof};
pub use self::side_bundle::{
    build_rv32im_bound_side_opening_public_from_accepted_artifact,
    build_rv32im_bound_side_proof_bundle_from_accepted_artifact,
};
pub use self::side_opening_spartan::{Rv32imSideOpeningSpartanProof, Rv32imSideOpeningSpartanVerifierKey};
pub use self::side_proof::{build_rv32im_side_proof, verify_rv32im_side_proof};
pub use self::side_relation_spartan::{Rv32imSideBindingProof, Rv32imSideBindingVerifierKey};
pub use self::statement::rv32im_verifier_context_digest;
pub use crate::rv32im::Rv32imCompressedMainProof;
