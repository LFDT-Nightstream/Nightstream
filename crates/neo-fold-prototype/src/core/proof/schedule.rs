//! Owns generic SuperNeo proof chunk scheduling.

use neo_reductions::error::PiCcsError;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum FoldSchedule {
    WholeTrace,
    RowsPerChunk(usize),
}

impl FoldSchedule {
    pub fn validate(self) -> Result<(), PiCcsError> {
        match self {
            Self::WholeTrace => Ok(()),
            Self::RowsPerChunk(0) => Err(PiCcsError::InvalidInput(
                "RowsPerChunk(0) is not a valid fold schedule".into(),
            )),
            Self::RowsPerChunk(_) => Ok(()),
        }
    }

    pub fn meta_words(self) -> [u64; 2] {
        match self {
            Self::WholeTrace => [0, 0],
            Self::RowsPerChunk(rows) => [1, rows as u64],
        }
    }

    pub fn chunk_count(self, step_count: usize) -> Result<usize, PiCcsError> {
        self.validate()?;
        Ok(match self {
            Self::WholeTrace => usize::from(step_count != 0),
            Self::RowsPerChunk(rows) => {
                if step_count == 0 {
                    0
                } else {
                    step_count.div_ceil(rows)
                }
            }
        })
    }
}

impl Default for FoldSchedule {
    fn default() -> Self {
        Self::RowsPerChunk(1)
    }
}
