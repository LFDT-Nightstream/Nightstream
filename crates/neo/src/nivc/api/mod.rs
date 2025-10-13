//! Public API surface for NIVC

pub mod errors;
pub mod program;
pub mod state;
pub mod types;

// Re-export commonly used types
pub use errors::NivcError;
pub use program::{NivcProgram, NivcStepSpec};
pub use state::{NivcState, NivcAccumulators, LaneRunningState, NivcStepProof, NivcChainProof};
pub use types::{LaneId, StepIdx, Result};

