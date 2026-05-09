//! Shared Construction-2 terminal committed-step circuit mechanics.
//!
//! This module owns only relation-neutral plumbing: public `u_i = (C_i, x_i)`
//! boundary allocation, Poseidon2 boundary digest checks, low-norm source-image
//! encoding helpers, and packed Ajtai commitment checks. It does not own any
//! RV32IM or direct-CCS F' semantics.

mod boundary;
mod commitment;
mod constraints;
mod labels;
mod low_norm;
mod types;

pub(crate) use boundary::{
    alloc_terminal_boundary_public_inputs, enforce_terminal_boundary_digests, terminal_boundary_public_values,
};
pub(crate) use commitment::{
    enforce_packed_padding_zero, enforce_public_commitment_shape, enforce_terminal_ajtai_commitment,
};
pub(crate) use constraints::{enforce_boolean_allocated, native_to_spartan};
pub(crate) use labels::{collect_private_witness_labels, padded_private_witness_labels};
pub(crate) use low_norm::{committed_nc_range_error, low_norm_encoded_values};
pub(crate) use types::{
    Construction2TerminalBoundaryInputs, Construction2TerminalBoundaryView, TerminalPrivateColumnEncoding,
};
