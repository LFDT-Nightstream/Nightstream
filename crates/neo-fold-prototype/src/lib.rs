//! `neo-fold-prototype` is the SuperNeo IVC integrator.
//!
//! Direct-CCS lifecycle:
//!   `preprocess_direct_ccs` → `prove_direct_ccs` → `extend_direct_ccs`* →
//!   `finish_direct_ccs_with_spartan`
//!   (or `prove_and_finish_direct_ccs_with_spartan` as a one-shot).
//! Verify with `verify_direct_ccs` (native, no Spartan) or
//! `verify_finished_direct_ccs_with_spartan` (after Spartan).
//!
//! RV32IM lifecycle: `prove_rv32im` → `verify_rv32im`.
//!
//! Protocol math lives in the sibling `neo_*` crates; this crate owns IVC
//! threading, Construction-2, the recursive verifier circuit, and frontend
//! lowering for `direct_ccs` and `rv32im`.

pub mod circuit;
pub mod core;
pub mod decider;
pub mod frontends;
pub mod lifecycle;
pub mod public_proof;
pub mod vm;

pub use self::direct_ccs::{DirectCcsFPrimeSnarkError, DirectCcsProgram, DirectCcsStep};
pub use self::frontends::{direct_ccs, rv32im};
pub use self::lifecycle::{
    extend_direct_ccs, finish_direct_ccs_with_spartan, preprocess_direct_ccs, prove_and_finish_direct_ccs_with_spartan,
    prove_direct_ccs, prove_rv32im, verify_direct_ccs, verify_finished_direct_ccs_with_spartan, verify_rv32im,
    DirectCcs, DirectCcsCommitmentOps, DirectCcsDecCommitmentMixer, DirectCcsFinishedProof,
    DirectCcsFinishedProofBundle, DirectCcsFinishedProofPerf, DirectCcsFinishedPublicImage,
    DirectCcsFinishedVerifierKey, DirectCcsProof, DirectCcsProofSummary, DirectCcsProverPreprocessing,
    DirectCcsRlcCommitmentMixer, IncrementalProofSystem, OneShotProofSystem, Rv32im, SpartanProofSystem,
};
pub use self::rv32im::{Rv32imProof, Rv32imProofInput, SimpleKernelError};

pub(crate) use self::circuit::{superneo as superneo_circuit, superneo_nifs as superneo_nifs_circuit};
pub(crate) use self::core::multilinear;
pub(crate) use self::core::{
    chunk_folding, construction2, finalize, ivc, opening, proof, prover, session, step_build, verifier, witness_layout,
};
pub(crate) use self::decider::spartan_backend;
