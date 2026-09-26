//! Resolve checked coordinate maps before matrix expansion. Retained operands
//! must remain contiguous; source-row indices and source substitutions keep
//! their own index space. Missing reads fail before zero terms are removed.

use super::{checked_add, Block, Entry, PackageError, RetainedBlock, SourceProjection};

pub(super) struct ColumnProjection {
    source_width: usize,
    projection: SourceProjection,
}

impl ColumnProjection {
    pub(super) fn new(source_width: usize, projection: SourceProjection) -> Self {
        Self {
            source_width,
            projection,
        }
    }

    pub(super) fn column(&self, column: usize) -> Result<usize, PackageError> {
        if column >= self.source_width {
            return Err(PackageError::Invalid("mapped matrix source column"));
        }
        self.projection.column(column)
    }

    pub(super) fn entries(&self, entries: &mut [Entry]) -> Result<(), PackageError> {
        for entry in entries {
            entry.column = self.column(entry.column)?;
        }
        Ok(())
    }

    pub(super) fn retained(&self, block: &mut RetainedBlock) -> Result<(), PackageError> {
        if !block.fits(self.source_width)? {
            return Err(PackageError::Invalid("mapped retained source range"));
        }
        let count = block.coordinate_count()?;
        if count == 0 {
            return Ok(());
        }
        let start = block.start;
        let end = checked_add(start, count, "mapped retained range end")?;
        let mapped_start = self.column(start)?;
        let mut cuts = vec![start, end];
        if let SourceProjection::Mapped(ranges) = &self.projection {
            for range in ranges {
                // Only boundaries inside the source interval matter. A range
                // extending beyond usize cannot add a boundary inside it.
                for boundary in [range.package_start, range.package_start.saturating_add(range.count)] {
                    if start < boundary && boundary < end {
                        cuts.push(boundary);
                    }
                }
            }
        }
        cuts.sort_unstable();
        cuts.dedup();
        for interval in cuts.windows(2) {
            for column in [interval[0], interval[1] - 1] {
                let expected = checked_add(mapped_start, column - start, "mapped retained target range")?;
                if self.column(column)? != expected {
                    return Err(PackageError::Invalid("non-contiguous retained column projection"));
                }
            }
        }
        block.start = mapped_start;
        Ok(())
    }

    pub(super) fn apply(&self, block: &mut Block) -> Result<(), PackageError> {
        match block {
            Block::Ordinary(block) => {
                block.one_column = self.column(block.one_column)?;
                block.substitution.map_columns(self)
            }
            Block::OrdinaryTemplate(template) => {
                template.block.one_column = self.column(template.block.one_column)?;
                template.block.substitution.map_columns(self)
            }
            Block::Multiplication(block) => {
                block.one_column = self.column(block.one_column)?;
                block.left.map_columns(self)?;
                block.right.map_columns(self)?;
                block.output.map_columns(self)
            }
            Block::Phi81(block) => block.map_columns(self),
            Block::Pin(block) => {
                block.one_column = self.column(block.one_column)?;
                for form in &mut block.values {
                    self.entries(form)?;
                }
                Ok(())
            }
            Block::Poseidon(block) => block.map_columns(self),
        }
    }
}
