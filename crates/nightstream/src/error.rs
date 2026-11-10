//! The inner errors of `Error`, so that a caller can name the type of each
//! value it matches.

pub use crate::lifecycle::bounded_depth::{BoundedDepthError, RelationError};
pub use crate::lifecycle::ProofCodecError;
pub use neo_params::ParamsError;
pub use nightstream_fprime::PackageError;
