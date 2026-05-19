//! `FoldSchedule` — caller batching policy.
//!
//! How the caller's flat row sequence becomes a sequence of fold steps.
//! This is a UX/ergonomics concern (how does the API accept input?) — not
//! a Construction-2 primitive — so it lives in `lifecycle/`, not under
//! `paper/`.
//!
//! `partition<T>` is generic in the row type so each frontend uses
//! whatever shape suits it. The direct-CCS frontend partitions
//! `Vec<CcsInstance>` directly; another frontend could partition over
//! its own per-row type and let its own builder turn them into instances.

use thiserror::Error;

/// How to slice a flat row sequence into fold steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FoldSchedule {
    /// All rows fold in a single step (K = total).
    WholeRun,
    /// K rows per step.
    RowsPerStep(usize),
}

impl Default for FoldSchedule {
    fn default() -> Self {
        Self::RowsPerStep(1)
    }
}

#[derive(Debug, Error)]
pub enum ScheduleError {
    #[error("FoldSchedule::RowsPerStep(0) is not valid (would never advance)")]
    ZeroRowsPerStep,
}

impl FoldSchedule {
    pub fn validate(self) -> Result<(), ScheduleError> {
        match self {
            Self::WholeRun => Ok(()),
            Self::RowsPerStep(0) => Err(ScheduleError::ZeroRowsPerStep),
            Self::RowsPerStep(_) => Ok(()),
        }
    }

    /// Number of fold steps a row count produces under this schedule.
    pub fn step_count(self, row_count: usize) -> Result<usize, ScheduleError> {
        self.validate()?;
        Ok(match self {
            Self::WholeRun => usize::from(row_count != 0),
            Self::RowsPerStep(n) => {
                if row_count == 0 {
                    0
                } else {
                    row_count.div_ceil(n)
                }
            }
        })
    }

    /// Generic partition — slice a flat `Vec<T>` into batches sized by the
    /// schedule. Empty input yields no batches; otherwise every batch is
    /// non-empty. The frontend chooses `T`.
    pub fn partition<T>(self, rows: Vec<T>) -> Result<Vec<Vec<T>>, ScheduleError> {
        self.validate()?;
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        let chunk_len = match self {
            Self::WholeRun => rows.len(),
            Self::RowsPerStep(n) => n,
        };

        let mut batches = Vec::with_capacity(self.step_count(rows.len())?);
        let mut cursor = rows.into_iter();
        loop {
            let mut batch = Vec::with_capacity(chunk_len);
            for _ in 0..chunk_len {
                match cursor.next() {
                    Some(row) => batch.push(row),
                    None => break,
                }
            }
            if batch.is_empty() {
                break;
            }
            batches.push(batch);
        }
        Ok(batches)
    }
}
