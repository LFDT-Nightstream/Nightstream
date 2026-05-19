//! Owns Stage 1 row-binding summaries for the RV32IM parity slice.

mod proof;
mod semantic_inputs;
mod semantics;

pub use proof::{
    build_stage1_proof_bundle, build_stage1_summary, stage1_row_word_width, AluShoutProof, BranchShoutProof,
    BytecodeShoutProof, Stage1AddressCorrectnessProof, Stage1LinkageProof, Stage1ProofBundle, Stage1RowBinding,
    Stage1Summary,
};
pub(crate) use proof::{stage1_row_binding_from_row, stage1_row_digest, stage1_row_words};
pub use semantic_inputs::{build_sem_inputs, sem_in_digest, sem_in_from_row, sem_inputs_digest, SemIn};
pub use semantics::{
    build_stage1_semantics_proof, stage1_row_bindings_digest, verify_stage1_semantics, Stage1SemanticsProof,
};
