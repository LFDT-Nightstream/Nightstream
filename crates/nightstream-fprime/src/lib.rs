//! Loader and execution boundary for the Lean-owned Nightstream F′ package.
//!
//! Owns strict decoding, identity binding, generic sparse-matrix expansion,
//! and witness-program execution. It does not own the F′ relation, its circuit
//! structure, or a proof backend.

#[cfg(test)]
extern crate self as nightstream_fprime;

mod identity;
mod package;
mod proof;
mod sparse;
mod witness;

pub use identity::{
    PiCcsV1_1VerifierContext, Stage1VerifierBinding, POSEIDON2_HASH_CHAIN_V1_PACKAGE_IDENTITY,
    POSEIDON2_HASH_CHAIN_V1_STRUCTURAL_IDENTIFIER, POSEIDON2_HASH_CHAIN_V1_VERIFICATION_KEY_DIGEST,
};
pub use package::{
    derive_pi_ccs_v1_1_transcript, load, load_file, load_per_application_package, load_poseidon2_hash_chain_v1_package,
    load_with_expanded_package, CcsMatrixSource, LoadedApplicationPlan, LoadedAssignmentPlan, LoadedPackage,
    LoadedPerApplicationPackage, LoadedTerminalLayout, LogicalAssignment, LogicalMatrixEntry, LogicalMatrixRow,
    PackageCcsRelation, PackageError, PackagePolynomialTerm, PackageR1cs, PackageSparseMatrix, PiCcsV1_1EncodedInputs,
    PiCcsV1_1OutputEvaluations, PiCcsV1_1PackageInputs, PiCcsV1_1Transcript, PiDecV1_1PackageInputs,
    PI_CCS_V1_1_COEFFICIENT_COUNT, PI_CCS_V1_1_FRESH_COMMITMENT_WORDS, PI_CCS_V1_1_MATRIX_COUNT,
    PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS, PI_CCS_V1_1_ROUND_COEFFICIENT_COUNT, PI_CCS_V1_1_ROUND_COUNT,
    PI_CCS_V1_1_SOURCE_COUNT, PI_CCS_V1_1_STATE_PREIMAGE_WORDS, PI_CCS_V1_1_VERIFIER_CONTEXT_WORDS,
    PI_DEC_V1_1_CHILD_COUNT, PI_DEC_V1_1_COMMITMENT_WORDS_PER_CHILD, PI_DEC_V1_1_EVAL_A_MATRICES_PER_CHILD,
    PI_DEC_V1_1_EVAL_K_VALUES_PER_CHILD, PI_DEC_V1_1_PUBLIC_INPUT_WORDS_PER_CHILD,
};
pub use proof::{ProofRun, WitnessAssignment};
