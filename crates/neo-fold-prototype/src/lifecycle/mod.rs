//! Crate-root proof lifecycle entrypoints.
//!
//! This module owns the public proving vocabulary. Frontend-specific protocol
//! work stays in the frontend modules.

mod direct_ccs;
mod rv32im;
mod traits;

pub use direct_ccs::{
    extend_direct_ccs, finish_direct_ccs_with_spartan, preprocess_direct_ccs, prove_and_finish_direct_ccs_with_spartan,
    prove_direct_ccs, verify_direct_ccs, verify_finished_direct_ccs_with_spartan, DirectCcs, DirectCcsCommitmentOps,
    DirectCcsDecCommitmentMixer, DirectCcsFinishedProof, DirectCcsFinishedProofBundle, DirectCcsFinishedProofPerf,
    DirectCcsFinishedPublicImage, DirectCcsFinishedVerifierKey, DirectCcsProof, DirectCcsProofSummary,
    DirectCcsProverPreprocessing, DirectCcsRlcCommitmentMixer,
};
pub use rv32im::{prove_rv32im, verify_rv32im, Rv32im};
pub use traits::{IncrementalProofSystem, OneShotProofSystem, SpartanProofSystem};
