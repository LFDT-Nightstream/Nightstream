//! In-circuit Construction-2 accumulator update for direct F'.
//!
//! This module folds the prior compact F' accumulator inside the terminal
//! circuit. Synthesis and measurement are separate so the proof path is not
//! tangled with constraint accounting.

mod measurement;
mod synthesis;
mod types;

pub(crate) use measurement::measure_direct_construction2_fold;
pub(crate) use synthesis::synthesize_direct_construction2_fold;
pub(crate) use types::{
    DirectCcsConstruction2FoldBreakdown, DirectCcsConstruction2FoldContext, DirectCcsConstruction2FoldShapeDelta,
};

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CeClaim};
use neo_math::{F, K};
use neo_params::NeoParams;
use neo_reductions::engines::utils::Dims;

use super::super::state::{DirectCcsChunkCircuitSurface, DirectCcsFPrimeSnarkError};
use super::gadgets::{digest32_as_spartan_fields, direct_accumulator_digest_circuit_from_claims};
use super::initial_carry::{alloc_initial_claim_bundle, alloc_initial_transcript};
use super::public_io::{
    direct_terminal_construction2_accumulator_digest_range, enforce_digest_eq_constant, enforce_digest_fields_public_io,
};
use crate::ivc::SuperNeoIvcTranscriptSnapshot;
use crate::spartan_backend::{NeoFoldDeciderEngine, ShapeCS, SpartanF};
use crate::superneo_nifs_circuit::synthesize_superneo_nifs_chunk;
