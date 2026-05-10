//! Spartan terminal-compression contract.
//!
//! Owns: the *contract* between the IVC layer and Spartan2. The contract is
//! deliberately small: a relation expressing "the latest F' step is satisfied
//! against the running U_i, and the public IO is `EncInst`."
//!
//! Does not own: Spartan2 itself; the actual SNARK lives in the `spartan2`
//! crate. The R1CS synthesis lives under `engine::decider` (not yet built).
//!
//! ## Why this is a separate module
//!
//! Because the auditor's question at this seam is narrow:
//! - What public IO does the SNARK bind?
//! - What relation does it prove?
//! - What digest of the verifier key is the user expected to compare?
//!
//! Those three questions are answered here.

use thiserror::Error;

use crate::paper::construction2::{EncInst, FinalFoldProof, State, VerifierKey};

#[derive(Debug, Error)]
pub enum Error {
    #[error("decider: Spartan terminal compression is not implemented yet")]
    Unsupported,
    #[error("decider: public-IO binding mismatch")]
    PublicIoMismatch,
    #[error("decider: Spartan verification failed")]
    SpartanFailed,
}

/// What the Spartan SNARK proves about the IVC state.
///
/// **Public inputs**:
///   - `vk_fs.digest`  (32 bytes)
///   - `state.i`       (u64)
///   - `state.z_0`     (32 bytes)
///   - `state.z_i`     (32 bytes)
///   - accumulator digest of `state.carry` (32 bytes)
///   - `state.pc`      (u64)
///   - `x_out`         (the EncInst — bit-decomposed)
///
/// **Private witness**:
///   - The CCS witness for the latest step's F' invocation, plus the
///     committed parts the recursive verifier consumed.
///
/// **Statement**:
///   - The R1CS shape encodes "F' ran on these inputs and produced this
///     `x_out`," with NIFS.V replicated in-circuit so the SNARK does not
///     trust an external folding oracle.
#[derive(Clone, Debug)]
pub struct Statement {
    pub vk: VerifierKey,
    pub state: State,
    /// Prover-side terminal fold witness. Present for `prove`, absent for
    /// public-only `verify` statement reconstruction.
    pub final_fold: Option<FinalFoldProof>,
    pub x_out: EncInst,
}

/// The compressed proof handed to the verifier.
///
/// PR5 will populate this with the Spartan SNARK bytes (and any auxiliary
/// public-IO fields the decider's R1CS exposes). Today it is a placeholder
/// type so the lifecycle wiring compiles end-to-end.
#[derive(Clone, Debug, Default)]
pub struct Proof;

/// Verifier key digest (32 bytes). Compared by the caller against an expected
/// value, never trusted as authority.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifierKeyDigest(pub [u8; 32]);

/// Run the Spartan terminal compression on the IVC state.
pub fn prove(_statement: &Statement) -> Result<(Proof, VerifierKeyDigest), Error> {
    Err(Error::Unsupported)
}

/// Verify a Spartan-compressed proof against the expected statement.
pub fn verify(_statement: &Statement, _vk_digest: &VerifierKeyDigest, _proof: &Proof) -> Result<(), Error> {
    Err(Error::Unsupported)
}
