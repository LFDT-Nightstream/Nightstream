//! Owns the generic Spartan2 decider target surface and its backend-binding contract.
//!
//! Ownership:
//! - reusable public/private target shapes for decider adapters
//! - Poseidon2-only target and witness digests
//! - canonical backend-visible public IO and witness layout
//! - Spartan2 shell proofs over public-target and backend-binding contracts
//! - end-to-end decider setup/prove/verify over that backend-binding seam
//! - does not own route-level theorem semantics or hidden-witness compression

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use neo_ccs::crypto::poseidon2_goldilocks::{poseidon2_hash, DIGEST_LEN as POSEIDON2_DIGEST_LEN};
use neo_math::F;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::{Deserialize, Serialize};
use spartan2::{
    bellpepper::poseidon2::hash_packed_goldilocks_fields,
    provider::{goldi::F as SpartanF, GoldilocksP3MerkleMleEngine},
    spartan::{SpartanProvePerf, R1CSSNARK},
    traits::circuit::SpartanCircuit,
    traits::snark::R1CSSNARKTrait,
};
use thiserror::Error;

use crate::finalize::{
    digest32_as_fields, digest_fields_as_digest32, digest_fixed_shape_final_proof,
    fixed_shape_terminal_handle_digest_fields, validate_fixed_shape_chunk_layout, FixedShapeChunkSummary,
    FIXED_SHAPE_DIGEST_FIELD_LEN,
};
use crate::proof::FoldSchedule;

mod backend_binding_shell;
mod decider;
mod packing;
mod public_relation_shell;
mod public_target_shell;
mod relation;
mod types;

pub use backend_binding_shell::{
    prove_spartan2_backend_binding_shell_with_perf, setup_spartan2_backend_binding_shell,
    verify_spartan2_backend_binding_shell,
};
pub use decider::{prove_spartan2_decider_with_perf, setup_spartan2_decider, verify_spartan2_decider};
pub use public_relation_shell::{
    prove_spartan2_public_relation_shell, setup_spartan2_public_relation_shell, verify_spartan2_public_relation_shell,
    Spartan2PublicRelationShellError, Spartan2PublicRelationShellProof, Spartan2PublicRelationShellProverKey,
    Spartan2PublicRelationShellSnark, Spartan2PublicRelationShellVerifierKey,
};
pub use public_target_shell::{
    prove_spartan2_public_target_shell_with_perf, setup_spartan2_public_target_shell,
    verify_spartan2_public_target_shell,
};
pub use relation::{
    build_spartan2_decider_relation, build_spartan2_self_bound_decider_relation,
    validate_spartan2_backend_relation_surface, validate_spartan2_decider_relation_surface,
};
pub use types::*;

use packing::{
    extend_packed_bytes_as_fields, extend_spartan2_chunk_summary_fields, packed_bytes_field_len,
    spartan2_chunk_summary_field_len, spartan2_chunk_summary_terminal_relation_digest_field_offset, spartan_inverse,
};
use relation::validate_spartan2_decider_target_surface;
