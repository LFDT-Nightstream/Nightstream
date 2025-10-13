//! Public types for NIVC API

/// Lane identifier (index into the program's step types)
pub type LaneId = usize;

/// Step index (sequential counter across all lanes)
pub type StepIdx = u64;

/// Result type alias for NIVC operations
pub type Result<T> = anyhow::Result<T>;

