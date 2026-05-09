//! Owns proof-session chunk schedule validation.

use neo_reductions::error::PiCcsError;

use crate::proof::{FoldSchedule, PublicChunk};

pub(super) fn validate_chunk_layout(schedule: FoldSchedule, chunks: &[PublicChunk]) -> Result<(), PiCcsError> {
    schedule.validate()?;
    let mut next_start = 0usize;
    let row_cap = match schedule {
        FoldSchedule::WholeTrace => None,
        FoldSchedule::RowsPerChunk(rows) => Some(rows),
    };

    for (idx, chunk) in chunks.iter().enumerate() {
        if chunk.steps.is_empty() {
            return Err(PiCcsError::InvalidInput(format!("chunk[{idx}] is empty")));
        }
        if chunk.start_index != next_start {
            return Err(PiCcsError::InvalidInput(format!(
                "chunk[{idx}] starts at {}, expected {}",
                chunk.start_index, next_start
            )));
        }
        if let Some(rows_per_chunk) = row_cap {
            if chunk.steps.len() > rows_per_chunk {
                return Err(PiCcsError::InvalidInput(format!(
                    "chunk[{idx}] has {} steps, exceeds RowsPerChunk({rows_per_chunk})",
                    chunk.steps.len()
                )));
            }
            if idx + 1 != chunks.len() && chunk.steps.len() != rows_per_chunk {
                return Err(PiCcsError::InvalidInput(format!(
                    "chunk[{idx}] has {} steps, expected exactly {} before the final chunk",
                    chunk.steps.len(),
                    rows_per_chunk
                )));
            }
        }
        next_start += chunk.steps.len();
    }

    if matches!(schedule, FoldSchedule::WholeTrace) && chunks.len() > 1 {
        return Err(PiCcsError::InvalidInput(
            "WholeTrace schedule must carry exactly one chunk".into(),
        ));
    }
    if schedule.chunk_count(next_start)? != chunks.len() {
        return Err(PiCcsError::InvalidInput(format!(
            "chunk count {} does not match {:?} for {} steps",
            chunks.len(),
            schedule,
            next_start
        )));
    }
    Ok(())
}
