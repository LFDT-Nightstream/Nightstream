//! F' — the augmented function from Hypernova §6.3 Construction 2.
//!
//! ## Submodules
//!
//! - **`native`**: prover-/verifier-side native F' execution (NIFS step,
//!   state advance, `x_out` hash).
//! - **`digest_circuit`**: in-circuit Poseidon2 digest mirrors —
//!   canonical `state_x_out` plus helper mirrors such as
//!   `boundary_update`, `public_trace_update`, and the accumulator digest
//!   variants.
//!   Each is byte-for-byte parity-tested against the native `paper::digest::*`.
//! - **`r1cs`**: strict F' R1CS scaffold, split into base/recursive entry
//!   points. Composes the validated digest gadgets + transcript-driven
//!   NIFS.V to enforce one IVC step's constraints.
//! - **`source_image`**: bit-valued source coordinates for F' public
//!   *boundary* encodings — `enc_inst(x_out)` and selected u64 counters.
//!   Wired into `r1cs.rs` for the recursive public link only. It is
//!   **not** the production low-norm F' image encoder; that lives in
//!   the F' shell ([`crate::frontends::f_prime`]).
//! - **`poseidon_trace`** / **`ring_action_trace`**: F' trace
//!   primitives. They're paper-layer (no app knowledge) and the
//!   F' shell consumes them.
//!
//! ## Boundary rule
//!
//! `paper::f_prime` **does not know about any app frontend**. The
//! generic F' image, structure, and recursive-step plan live under
//! [`crate::frontends::f_prime`]. The R1CS and Nebula frontends build their
//! application relations on that shell. The dependency direction is one-way:
//! `frontends::* → paper::f_prime`.

pub mod digest_circuit;
pub mod native;
pub mod nebula_lane_circuit;
pub mod poseidon_trace;
pub mod projection_trace;
pub mod public_input_link;
// Only private diagnostic paths retain the old recursive R1CS emitter.
#[allow(dead_code, unused_imports)]
pub mod r1cs;
pub mod ring_action_trace;
pub mod source_image;
pub mod source_image_circuit;
pub mod stage;

// The lifecycle uses preparation to fold first and install the real encoded
// next instance after the frontend synthesizes it.
pub(crate) use native::{
    prepare_single_with_adapter_and_semantic_state, prepare_single_with_semantic_state, FreshClaimShape,
    PreparedFPrimeStep,
};
pub use native::{
    prove, prove_with_adapter_and_semantic_state, prove_with_semantic_state, verify, verify_with_execution_receipt,
    Error, NifsVerifyExecutionOutcome, VerifyStepDispatch, VerifyStepExecutionEvent, VerifyStepExecutionFoldProof,
    VerifyStepExecutionInput, VerifyStepExecutionProof, VerifyStepExecutionProofState, VerifyStepExecutionReceipt,
    VerifyStepExecutionStage, VerifyStepExecutionState,
};
