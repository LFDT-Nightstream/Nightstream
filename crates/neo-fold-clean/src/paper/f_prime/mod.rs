//! F' — the augmented function from Hypernova §6.3 Construction 2.
//!
//! ## Submodules
//!
//! - **`native`**: prover-/verifier-side native F' execution (NIFS step,
//!   state advance, `x_out` hash).
//! - **`digest_circuit`**: in-circuit Poseidon2 digest mirrors —
//!   `state_x_out`, `boundary_update`, `public_trace_update`, and the
//!   accumulator digest (both `from_parent` and `from_children` variants).
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
//! generic F' image / structure / encoder / recursive-step plan live
//! under [`crate::frontends::f_prime`] (the engine) and the app
//! adapters (`frontends::r1cs_f_prime` for production; the Fibonacci
//! test fixture lives under `tests/support/fibonacci_f_prime`) build
//! on top of that. The dependency direction is one-way:
//! `frontends::* → paper::f_prime`.

pub mod digest_circuit;
pub mod native;
pub mod poseidon_trace;
pub mod r1cs;
pub mod ring_action_trace;
pub mod source_image;
pub mod source_image_circuit;

// Public surface — paper-named entry points kept stable so call sites
// (mostly `paper::construction2`) don't churn when PR5 adds the R1CS
// siblings.
pub use native::{prove, verify, Error};
