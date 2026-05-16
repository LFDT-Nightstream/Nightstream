//! Owns reusable circuit helpers for the RV32IM Nightstream side relation.

pub mod digests;
pub mod exact_package;
pub mod phase0;
pub mod word;

pub use digests::{
    continuity_event_digest as circuit_continuity_event_digest, digest_u64_words as circuit_digest_u64_words,
    kernel_binding_opening_packaged_statement_digest as circuit_kernel_binding_opening_packaged_statement_digest,
    kernel_prepared_step_opening_packaged_statement_digest as circuit_kernel_prepared_step_opening_packaged_statement_digest,
    ram_event_digest as circuit_ram_event_digest, register_read_event_digest as circuit_register_read_event_digest,
    register_write_event_digest as circuit_register_write_event_digest,
    single_step_packaged_statement_digest as circuit_single_step_packaged_statement_digest,
    stage1_opening_packaged_statement_digest as circuit_stage1_opening_packaged_statement_digest,
    stage1_row_digest as circuit_stage1_row_digest,
    stage2_opening_packaged_statement_digest as circuit_stage2_opening_packaged_statement_digest,
    stage3_opening_packaged_statement_digest as circuit_stage3_opening_packaged_statement_digest,
    twist_link_event_digest as circuit_twist_link_event_digest,
};
pub use exact_package::{
    exact_vector_packaged_step_digest_from_native_words as circuit_exact_vector_packaged_step_digest_from_native_words,
    exact_vector_packaged_step_digest_from_words as circuit_exact_vector_packaged_step_digest_from_words,
};
pub use phase0::{
    derive_phase0_point as circuit_derive_phase0_point,
    enforce_commitment_root_and_opened_object_digest as circuit_enforce_phase0_commitment_root_and_opened_object_digest,
    enforce_payload_eq as circuit_enforce_phase0_payload_eq, enforce_point_eq as circuit_enforce_phase0_point_eq,
    evaluate_payload_from_packed_rows as circuit_evaluate_phase0_payload_from_packed_rows,
};
