//! R1CS circuit primitives for in-circuit verifier gadgets.
//!
//! This module owns the low-level constraint-emission machinery. Paper-level
//! reduction verifiers (Π_DEC.V, Π_RLC.V, Π_CCS.V) live in
//! `paper/reductions/*_circuit.rs` and consume these primitives.
//!
//! Auditor: nothing here is paper math. Files here are mechanical R1CS plumbing.

pub mod alphabet_sampling;
pub mod boolean;
pub mod builder;
pub mod encoding_trace;
pub mod field_ext;
pub mod mux;
mod pi_rlc_y_zcol_boundary;
pub mod poseidon2;
pub mod projection_identity_trace;
mod relation;
pub mod ring_action;
mod stage_provenance;
pub mod sumcheck;
pub mod transcript;
#[path = "u64.rs"]
pub mod u64_arith;

pub use builder::{Lc, R1csBuilder, Var};
pub use encoding_trace::{
    AcceptanceTraceEntry, AcceptanceTraceTestMutation, BalancedTernaryOpeningTraceEntry,
    BalancedTernaryTraceTestMutation, CanonicalU64TraceEntry, CanonicalU64TraceTestMutation,
    FirstAcceptedSelectionProducts, FirstAcceptedSelectionTraceEntry, KMulTraceEntry, Mod5TraceEntry,
    Mod5TraceTestMutation, PolynomialEvaluationTraceEntry, PolynomialEvaluationTraceTestMutation,
    PoseidonHashTraceEntry, PoseidonHashTraceTestMutation, PoseidonPermutationTraceEntry,
    PoseidonPermutationTraceTestMutation, ProjectionIdentityRole, ProjectionIdentityTraceEntry,
    ProjectionIdentityTraceTestMutation, ProjectionNebulaCoordinate, R1csEncodingTrace, R1csStageCheckpoint,
    RingMulToom3TraceEntry, Sbox7TraceEntry, Sbox7TraceTestMutation, Toom3ConvolutionTrace,
};
pub use field_ext::{alloc_klc, enforce_k_dot_product, enforce_k_mul, KLc, KVar};
pub use mux::{enforce_mux_var, enforce_mux_vec};
pub use pi_rlc_y_zcol_boundary::PiRlcYZcolBoundaryAudit;
pub use poseidon2::{enforce_poseidon2_hash, enforce_poseidon2_permutation, DIGEST_LEN};
pub use relation::{R1csRelation, R1csSnapshot};
pub use ring_action::{alloc_and_enforce_ring_mul, enforce_ring_mul};
pub(crate) use stage_provenance::finalize_physical_stages;
pub use stage_provenance::{PhysicalStageError, PhysicalStageRange};
pub use sumcheck::{
    enforce_chi_alpha, enforce_eq_k, enforce_gamma_indexed_sum, enforce_norm_check_b2, enforce_r1cs_f_term,
    enforce_sumcheck_round, enforce_sumcheck_rounds_engine, enforce_sumcheck_walk, gamma_powers, horner_eval_k,
};
pub use transcript::TranscriptGadget;
pub use u64_arith::{
    alloc_u64_bits, enforce_u64_add, enforce_u64_bitness, enforce_u64_constant, enforce_u64_equality,
    enforce_u64_increment,
};
