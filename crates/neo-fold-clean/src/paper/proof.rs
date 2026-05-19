//! Wire-format index — every type that crosses the prover/verifier boundary.
//!
//! Open this file to see the entire proof structure in one place. There is no
//! protocol logic here, only re-exports and an ASCII tree of how the pieces
//! nest. Modeled on Jolt's `JoltProof`-as-single-struct convention.
//!
//! ## Wire-format tree
//!
//! ```text
//! lifecycle::Compressed
//!   ├─ proof:        decider::Proof              (Spartan2 SNARK; PR4 wires the bytes)
//!   ├─ vk:           decider::VerifierKeyDigest  (32-byte vk digest)
//!   └─ public_image: lifecycle::PublicImage      (i, z_0, z_i, pc, x_out, vk_fs)
//!
//! lifecycle::Uncompressed          (incremental prover state, before `compress`)
//!   ├─ state:      construction2::State          (i, z_0, z_i, U_i, pc)
//!   ├─ steps:      Vec<construction2::StepProof>
//!   └─ transcript: engine::transcript::Transcript
//!
//! construction2::StepProof         (one per IVC step)
//!   ├─ nifs:   nifs::NifsProof        (NIFS.V replay material; what F' re-runs in-circuit)
//!   │   ├─ pi_ccs:  pi_ccs::Proof     (sumcheck transcript + K+k output CE claims at r')
//!   │   │   ├─ sumcheck: SumcheckProof  (= neo_reductions::api::PiCcsProof; opaque)
//!   │   │   └─ outputs:  Vec<CeClaim>   (the K+k claims Π_RLC consumes next)
//!   │   ├─ pi_rlc:  pi_rlc::Proof     (empty — Π_RLC has no prover message)
//!   │   └─ pi_dec:  pi_dec::Proof     (k child CE claims of norm b)
//!   └─ x_out:  construction2::EncInst (paper §6.3 hash-chain output)
//! ```
//!
//! The Π_CCS sumcheck transcript lives in the engine's opaque wire format;
//! the audit-relevant surface (the K+k Π_CCS outputs and the k Π_DEC
//! children) is exposed as typed `CeClaim`s on the paper layer so the
//! verifier never has to peek through opaque handles to chain reductions.
//!
//! ## Why Π_RLC carries no bytes
//!
//! The Π_RLC combined CE claim `P = Σρ_i · u_i` is a deterministic public
//! function of (ρ_i, u_i) plus the homomorphic Ajtai mix `Σρ_i · c_i`.
//! ρ_i are public-coin (Construction 3, resampled from the transcript);
//! u_i are the `pi_ccs.outputs` already on the wire. The verifier
//! reproduces P entirely from public data and feeds it into Π_DEC. There
//! is no information the prover could put on the wire that wouldn't be
//! either redundant or a covert channel — `pi_rlc::Proof` is therefore
//! an empty marker struct. See `paper/pi_rlc.rs` for the long-form
//! soundness rationale.
//!
//! ## What's *not* in this file
//!
//! - The `Statement` type that the Spartan SNARK proves: that is in
//!   `decider::Statement`, alongside the verifier contract.
//! - Anything frontend-specific. Frontends translate to the paper types
//!   above before crossing the boundary.

pub use crate::paper::construction2::{EncInst, StepProof};
pub use crate::paper::decider::{Proof as DeciderProof, Statement, VerifierKeyDigest};
pub use crate::paper::pi_ccs::Proof as PiCcsProof;
pub use crate::paper::pi_dec::Proof as PiDecProof;
pub use crate::paper::pi_rlc::Proof as PiRlcProof;
