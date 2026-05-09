//! Shared circuit helpers for the direct CCS/R1CS terminal F' surface.

mod accumulator;
mod construction2;
mod fields;
mod final_ce;
mod transitions;

pub(crate) use accumulator::{
    direct_accumulator_digest_circuit_from_claims, direct_accumulator_digest_from_claims,
    direct_accumulator_digest_from_claims_with_base,
};
pub(crate) use construction2::enforce_direct_construction2_input_u_i;
pub(crate) use fields::{digest32_as_spartan_fields, field_to_spartan, u64_halves_as_spartan_fields};
pub(crate) use final_ce::enforce_direct_terminal_final_ce_consistency;
pub(crate) use transitions::{
    enforce_direct_current_boundary_transition, enforce_direct_public_trace_transition,
    enforce_direct_state_x_in_digest, enforce_direct_state_x_out_public_digest,
};
