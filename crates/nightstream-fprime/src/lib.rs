//! Loader and execution boundary for the Lean-owned Nightstream F′ package.
//!
//! Owns strict decoding, identity binding, generic sparse-matrix expansion,
//! and witness-program execution. It does not own the F′ relation, its circuit
//! structure, or a proof backend.

#[cfg(test)]
extern crate self as nightstream_fprime;

mod application_records;
pub mod components;
mod identity;
mod package;
mod proof;
mod sparse;
mod witness;

pub use application_records::{
    ApplicationForm, ApplicationRecipeNode, ApplicationRecords, ApplicationRecordsWriter, ApplicationRowHeader,
    ApplicationTerm,
};

pub use identity::{
    PiCcsV1_2VerifierContext, Stage1VerifierBinding, POSEIDON2_HASH_CHAIN_V1_PACKAGE_IDENTITY,
    POSEIDON2_HASH_CHAIN_V1_STRUCTURAL_IDENTIFIER, POSEIDON2_HASH_CHAIN_V1_VERIFICATION_KEY_DIGEST,
};
pub use package::{
    derive_pi_ccs_v1_2_transcript, load_compiled_application_package, load_per_application_package,
    load_poseidon2_hash_chain_v1_package, load_prepared_application_records, load_prepared_application_value,
    CcsMatrixSource, LoadedApplicationPlan, LoadedAssignmentPlan, LoadedPackage, LoadedPerApplicationPackage,
    LoadedTerminalLayout, LogicalAssignment, LogicalMatrixEntry, LogicalMatrixRow, MatrixRun, PackageCcsRelation,
    PackageError, PackagePolynomialTerm, PackageR1cs, PackageSparseMatrix, PiCcsV1_2EncodedInputs,
    PiCcsV1_2OutputEvaluations, PiCcsV1_2PackageInputs, PiCcsV1_2Transcript, PiDecV1_2PackageInputs,
    PI_CCS_V1_2_COEFFICIENT_COUNT, PI_CCS_V1_2_FRESH_COMMITMENT_WORDS, PI_CCS_V1_2_MATRIX_COUNT,
    PI_CCS_V1_2_PRIOR_PUBLIC_INPUT_WORDS, PI_CCS_V1_2_ROUND_COEFFICIENT_COUNT, PI_CCS_V1_2_ROUND_COUNT,
    PI_CCS_V1_2_SOURCE_COUNT, PI_CCS_V1_2_STATE_PREIMAGE_WORDS, PI_CCS_V1_2_VERIFIER_CONTEXT_WORDS,
    PI_DEC_V1_2_CHILD_COUNT, PI_DEC_V1_2_COMMITMENT_WORDS_PER_CHILD, PI_DEC_V1_2_EVAL_A_MATRICES_PER_CHILD,
    PI_DEC_V1_2_EVAL_K_VALUES_PER_CHILD, PI_DEC_V1_2_PUBLIC_INPUT_WORDS_PER_CHILD,
};
pub use proof::{ProofRun, WitnessAssignment};
