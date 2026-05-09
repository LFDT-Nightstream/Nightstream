//! Owns the direct-CCS terminal Construction-2 committed-step proof.
//!
//! This is the non-VM analogue of the RV32IM terminal `F'` committed-step
//! boundary: the public output is `u_i = (C_i, x_i)`, where `C_i` opens to a
//! low-norm SuperNeo-packed source image linked to the latest direct-CCS F'
//! terminal circuit.

mod assignment;
mod circuit;
mod commitment;
mod measurement;
mod perf;
mod proof;
mod relation;
mod source_linking;
mod types;

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, LinearCombination, SynthesisError};
use neo_math::D;

use super::super::state::DirectCcsTerminalFPrimeCircuit;
use crate::construction2::terminal::{
    alloc_terminal_boundary_public_inputs, enforce_boolean_allocated, enforce_packed_padding_zero,
    enforce_public_commitment_shape, enforce_terminal_ajtai_commitment, enforce_terminal_boundary_digests,
    native_to_spartan, Construction2TerminalBoundaryInputs, TerminalPrivateColumnEncoding,
};
use crate::construction2::{
    Construction2Commitment, Construction2FreshInstance, Construction2PublicBoundary, CONSTRUCTION2_COMMITMENT_RAW_TAG,
    CONSTRUCTION2_ENC_INST_BITS, CONSTRUCTION2_PUBLIC_BOUNDARY_RAW_TAG,
};
use crate::spartan_backend::{NeoFoldDeciderEngine, ShapeCS, SpartanCircuit, SpartanF};
use crate::superneo_circuit::witness::{alloc_packed_mat_witness, PackedWitnessVar};
use crate::witness_layout::commit_cols_for_full_width;

use commitment::{direct_terminal_boundary_view, terminal_committed_boundary_public_values};
use perf::{shape_delta, shape_point};
pub(crate) use proof::{
    prove_direct_ccs_terminal_committed_relation, setup_direct_ccs_terminal_committed_relation_cached,
    verify_direct_ccs_terminal_committed_relation,
};
use source_linking::DirectSourceWitnessLinkingCs;
pub use types::DirectCcsTerminalCommittedConstraintBreakdown;
use types::{
    DirectCcsCommittedImageConstraintBreakdown, DirectCcsPublicBoundaryConstraintBreakdown,
    DirectCcsTerminalCommittedCircuit, DirectCcsTerminalR2Assignment, SimpleKernelError,
};
pub(crate) use types::{
    DirectCcsTerminalCommittedKeyPair, DirectCcsTerminalCommittedPerf, DirectCcsTerminalCommittedProof,
    DirectCcsTerminalCommittedRelation, DirectCcsTerminalError,
};
