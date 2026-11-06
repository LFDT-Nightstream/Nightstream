//! The inner errors of `Error`, so that a caller can name the type of each
//! value it matches.

pub use crate::lifecycle::option_b::{OptionBError, RelationError};
pub use crate::lifecycle::ProofCodecError;
pub use neo_params::ParamsError;
pub use nightstream_fprime::PackageError;
