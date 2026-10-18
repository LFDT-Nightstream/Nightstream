#![forbid(unsafe_code)]
//! CCS frontend for Neo: structures, relations (CCS/CE), and row-wise checks.

pub mod crypto;
pub mod error;
pub mod geometric;
pub mod matrix;
pub mod poly;
pub mod r1cs;
pub mod relations;
pub mod sparse;
pub mod traits;
pub mod utils;

pub use error::{CcsError, DimMismatch, RelationError};
pub use geometric::GeometricRowRun;
pub use matrix::{CsrMatrix, Mat, MatRef};
pub use poly::{SparsePoly, Term};
pub use r1cs::{r1cs_to_ccs, sparse_r1cs_to_ccs, sparse_selected_r1cs_to_ccs};
pub use sparse::{CcsMatrix, CscMat};

pub use relations::{
    check_ccs_rowwise_relaxed, check_ccs_rowwise_zero, superneo_public_x_cols, CcsClaim, CcsStructure, CcsWitness,
    CeClaim, CeWitness, LaneCommitments, V1_2Evaluations,
};
pub use traits::SModuleHomomorphism;
pub use utils::{mat_vec_mul_ff, mat_vec_mul_fk, tensor_point, validate_power_of_two};
