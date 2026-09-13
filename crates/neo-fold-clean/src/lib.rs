//! Nightstream Stage 1 over the selected Lean-emitted SuperNeo package.
//!
//! The selected lifecycle owns its relation, parameters and commitment key.
//! Verification takes the expected state from the caller's application.
//!
//! ```no_run
//! use neo_fold_clean::{Poseidon2HashChainV1Package, Stage1Envelope, Stage1State};
//! use neo_math::F;
//!
//! # fn example(package_bytes: &[u8], z0: [F; 4], messages: &[[F; 4]],
//! #     expected: Stage1State) -> Result<(), Box<dyn std::error::Error>> {
//! let package = Poseidon2HashChainV1Package::load(package_bytes)?;
//! let mut envelope = Stage1Envelope::initial(z0);
//! for &message in messages {
//!     envelope = package.extend(envelope, message)?;
//! }
//! package.verify(&expected, &envelope)?;
//! # Ok(())
//! # }
//! ```
//!
//! The first extension constructs the base assignment. Later extensions use
//! the actual NIFS prover and retain the returned child and fresh witnesses.
//! Sampler failure and a counter without a canonical successor return errors.
//! The terminal envelope contains complete openings and carries no proof history.
//!
//! The generic lifecycle and other frontends below have their own relation
//! and validation scopes. They do not supply the selected Stage 1 assurance
//! result. See `formal/nightstream-fprime/ASSURANCE_SURFACE.md` for the exact
//! checked scope and explicit assumptions.

pub mod config;
pub mod engine;
pub mod frontends;
mod heap;
pub mod lifecycle;
pub mod paper;
pub mod relation_artifact;
pub mod stage1;

// ── Public lifecycle re-exports. Keep this surface small. ─────────────────

// Generic relation lifecycle; the selected Stage 1 workflow uses the package methods.
pub use lifecycle::{
    extend, finish_uncompressed, preprocess, prove, verify_uncompressed, verify_uncompressed_with_opening_backend,
    Error, FinalWitnessOpeningBackend, Preprocessing, PublicImage, Uncompressed,
};

// Audit / decider path — chain-replay verifier, Spartan statement, diagnostic
// tests. See the crate-level docs for when each is appropriate; reach for
// these interfaces according to their separate relation and backend scope.
pub use lifecycle::{
    build_decider_statement, finish_uncompressed_with_audit, verify_uncompressed_audit, UncompressedAudit,
};

pub use frontends::r1cs_f_prime::{
    LeanNativeCcsManifest, LeanNebulaCombinedManifest, NebulaCombinedEmission, TerminalR1csError,
};
pub use lifecycle::{FoldSchedule, ScheduleError};
pub use paper::construction2::{
    FinalFoldProof, FoldProof, LatestInstance, ProofState, RunningInstance, State, StepProof, VerifierKey,
};
pub use paper::params::Params;
pub use paper::relations::{CcsInstance, CcsWitness, CeClaim, DecMixer, RlcMixer, Structure};
pub use relation_artifact::{RelationArtifactError, RelationArtifactReceipt, VerifierKeyRelationArtifact};
pub use stage1::{Poseidon2HashChainV1Package, Stage1Envelope, Stage1State};
