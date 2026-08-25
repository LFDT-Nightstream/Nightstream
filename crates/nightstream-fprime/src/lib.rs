//! Loader and execution boundary for the Lean-owned Nightstream F′ package.
//!
//! Owns strict decoding, identity binding, generic sparse-matrix expansion,
//! witness-program execution, and the direct production proof entrypoints.
//! It does not own the F′ relation or its circuit structure.

mod identity;
mod package;
mod proof;
mod sparse;
mod witness;

pub use package::{
    derive_pi_ccs_v1_1_transcript, load, load_file, CcsMatrixSource, LoadedPackage, PackageCcsRelation, PackageError,
    PackagePolynomialTerm, PackageProof, PackageProvingKey, PackageR1cs, PackageSparseMatrix, PackageVerifyingKey,
    PiCcsV1_1EncodedInputs, PiCcsV1_1OutputEvaluations, PiCcsV1_1PackageInputs, PiCcsV1_1Transcript,
    PI_CCS_V1_1_COEFFICIENT_COUNT, PI_CCS_V1_1_FRESH_COMMITMENT_WORDS, PI_CCS_V1_1_MATRIX_COUNT,
    PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS, PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT, PI_CCS_V1_1_ROUND_COUNT,
    PI_CCS_V1_1_SOURCE_COUNT, PI_CCS_V1_1_STATE_PREIMAGE_WORDS, PI_CCS_V1_1_VERIFIER_CONTEXT_WORDS,
};
pub use proof::{ProofRun, WitnessAssignment};
