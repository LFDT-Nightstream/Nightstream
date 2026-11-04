//! The inner errors of `Error`, so that a caller can name the type of each
//! value it matches.

pub use crate::assembly::AssemblyError;
pub use crate::engine::CommitError;
pub use crate::lifecycle::{CompressionError, ExtendError, ProofCodecError, VerifyError};
pub use neo_params::ParamsError;
pub use nightstream_fprime::PackageError;
