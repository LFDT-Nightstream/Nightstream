//! Owns direct CCS/R1CS frontend adapters.
//!
//! This layer turns external R1CS/CCS shapes into the generic direct-CCS
//! program and step inputs. It does not own F' compression, terminal proving,
//! or recursive IVC state.

mod low_norm;
mod r1cs;
mod r1cs_export;

pub use low_norm::{
    lower_sparse_r1cs_export_to_low_norm, lower_sparse_r1cs_export_to_low_norm_program_and_step, DirectLowNormLaneKind,
    DirectR1csLowNormLayout,
};
pub use r1cs::{direct_ccs_program_from_sparse_r1cs, direct_ccs_program_from_sparse_r1cs_with_public_input_len};
pub use r1cs_export::{
    direct_sparse_r1cs_export_from_spartan_circuit, DirectSparseR1csExport, DirectSparseR1csLowNormReport,
    DirectSparseR1csLowNormViolation,
};
