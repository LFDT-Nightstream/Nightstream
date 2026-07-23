//! Audit-only source-column decoder for bounded selective projections.
//!
//! Owns: the exact prepared-layout disposition of caller-selected source
//! fields. Does not own row semantics, protocol authority, or generated data.

use std::collections::BTreeSet;
use std::ops::Range;

use super::{trace_error, LowNormR1csError, SelectiveLayout};

/// Exact selective-layout disposition of one requested source field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectiveProjectedSourceResolution {
    ConstantOne,
    Direct {
        start: usize,
        width: usize,
        centered: bool,
    },
    DecompositionAlias {
        source: usize,
        digit: usize,
        start: usize,
        centered: bool,
    },
    EqualityAlias {
        source: usize,
        start: usize,
        width: usize,
        centered: bool,
    },
    LinearDefinition,
    TraceEliminated,
}

/// One source column and its exact prepared selective disposition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectiveProjectedSourceDecoder {
    column: usize,
    resolution: SelectiveProjectedSourceResolution,
}

impl SelectiveProjectedSourceDecoder {
    pub fn column(self) -> usize {
        self.column
    }

    pub fn resolution(self) -> SelectiveProjectedSourceResolution {
        self.resolution
    }
}

/// Decoder request kept separate from the established selected-row
/// provenance certificate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveProjectedDecoderProvenance {
    arm: usize,
    decoders: Vec<SelectiveProjectedSourceDecoder>,
}

/// One affine run in the complete source-to-final selective decoder.
///
/// Runs are an exact compression of consecutive source columns.  Every
/// numeric field is interpreted as `first + stride * offset`; singleton runs
/// use stride zero.  This keeps the complete production decoder auditable
/// without materializing one record per source column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectiveProjectedSourceResolutionRun {
    Direct {
        start: usize,
        start_stride: usize,
        width: usize,
        centered: bool,
    },
    DecompositionAlias {
        source: usize,
        source_stride: usize,
        digit: usize,
        digit_stride: usize,
        start: usize,
        start_stride: usize,
        centered: bool,
    },
    EqualityAlias {
        source: usize,
        source_stride: usize,
        start: usize,
        start_stride: usize,
        width: usize,
        centered: bool,
    },
    LinearDefinition,
    TraceEliminated,
}

/// One nonempty, consecutive source-column interval with one affine decoder
/// rule.  Source column zero is deliberately excluded from the complete
/// private decoder and therefore needs no `ConstantOne` variant here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectiveProjectedSourceDecoderRun {
    source_start: usize,
    length: usize,
    resolution: SelectiveProjectedSourceResolutionRun,
}

/// One nested source-allocation family intersected with the requested decoder
/// interval. Family names are diagnostic provenance; exact decoder rules,
/// not labels, remain the authority for generated templates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveProjectedSourceFamilyRange {
    path: &'static str,
    source_range: Range<usize>,
}

impl SelectiveProjectedSourceFamilyRange {
    pub fn path(&self) -> &'static str {
        self.path
    }

    pub fn source_range(&self) -> Range<usize> {
        self.source_range.clone()
    }
}

impl SelectiveProjectedSourceDecoderRun {
    pub fn source_start(self) -> usize {
        self.source_start
    }

    pub fn length(self) -> usize {
        self.length
    }

    pub fn source_end(self) -> usize {
        self.source_start + self.length
    }

    pub fn resolution(self) -> SelectiveProjectedSourceResolutionRun {
        self.resolution
    }

    /// Expand one owned source column back to the exact pointwise decoder.
    pub fn resolution_at(self, column: usize) -> Option<SelectiveProjectedSourceResolution> {
        let offset = column.checked_sub(self.source_start)?;
        if offset >= self.length {
            return None;
        }
        let affine = |first: usize, stride: usize| first.checked_add(stride.checked_mul(offset)?);
        Some(match self.resolution {
            SelectiveProjectedSourceResolutionRun::Direct {
                start,
                start_stride,
                width,
                centered,
            } => SelectiveProjectedSourceResolution::Direct {
                start: affine(start, start_stride)?,
                width,
                centered,
            },
            SelectiveProjectedSourceResolutionRun::DecompositionAlias {
                source,
                source_stride,
                digit,
                digit_stride,
                start,
                start_stride,
                centered,
            } => SelectiveProjectedSourceResolution::DecompositionAlias {
                source: affine(source, source_stride)?,
                digit: affine(digit, digit_stride)?,
                start: affine(start, start_stride)?,
                centered,
            },
            SelectiveProjectedSourceResolutionRun::EqualityAlias {
                source,
                source_stride,
                start,
                start_stride,
                width,
                centered,
            } => SelectiveProjectedSourceResolution::EqualityAlias {
                source: affine(source, source_stride)?,
                start: affine(start, start_stride)?,
                width,
                centered,
            },
            SelectiveProjectedSourceResolutionRun::LinearDefinition => {
                SelectiveProjectedSourceResolution::LinearDefinition
            }
            SelectiveProjectedSourceResolutionRun::TraceEliminated => {
                SelectiveProjectedSourceResolution::TraceEliminated
            }
        })
    }
}

/// Complete run-compressed decoder for one exact source interval.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveProjectedDecoderRunProvenance {
    arm: usize,
    source_range: Range<usize>,
    runs: Vec<SelectiveProjectedSourceDecoderRun>,
    source_families: Vec<SelectiveProjectedSourceFamilyRange>,
}

impl SelectiveProjectedDecoderRunProvenance {
    pub fn arm(&self) -> usize {
        self.arm
    }

    pub fn source_range(&self) -> Range<usize> {
        self.source_range.clone()
    }

    pub fn runs(&self) -> &[SelectiveProjectedSourceDecoderRun] {
        &self.runs
    }

    pub fn source_families(&self) -> &[SelectiveProjectedSourceFamilyRange] {
        &self.source_families
    }
}

impl SelectiveProjectedDecoderProvenance {
    pub fn arm(&self) -> usize {
        self.arm
    }

    pub fn decoders(&self) -> &[SelectiveProjectedSourceDecoder] {
        &self.decoders
    }
}

pub(super) fn decoder_provenance(
    layout: &SelectiveLayout,
    arm: usize,
    requested_source_columns: &[usize],
) -> Result<SelectiveProjectedDecoderProvenance, LowNormR1csError> {
    let Some(slots) = layout.slots.get(arm) else {
        return Err(trace_error("projected decoder-provenance arm is out of range"));
    };
    let requested = requested_source_columns
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if requested.len() != requested_source_columns.len() {
        return Err(trace_error(
            "projected decoder-provenance request repeats a source column",
        ));
    }
    if requested.iter().any(|&column| column >= slots.len()) {
        return Err(trace_error(
            "projected decoder-provenance column exceeds its source arm",
        ));
    }

    let mut decoders = Vec::with_capacity(requested.len());
    for column in requested {
        let resolution = source_resolution(layout, arm, column)?;
        decoders.push(SelectiveProjectedSourceDecoder { column, resolution });
    }
    Ok(SelectiveProjectedDecoderProvenance { arm, decoders })
}

fn source_resolution(
    layout: &SelectiveLayout,
    arm: usize,
    column: usize,
) -> Result<SelectiveProjectedSourceResolution, LowNormR1csError> {
    let Some(slots) = layout.slots.get(arm) else {
        return Err(trace_error("projected decoder-provenance arm is out of range"));
    };
    let plan = &layout.plans[arm];
    let aliases = &layout.aliases[arm];
    let equal_aliases = &layout.equal_aliases[arm];
    if column >= slots.len() {
        return Err(trace_error(
            "projected decoder-provenance column exceeds its source arm",
        ));
    }
    let resolution = if column == 0 {
        SelectiveProjectedSourceResolution::ConstantOne
    } else if plan.definitions.get(column).is_some() {
        if slots[column].is_some() {
            return Err(trace_error(
                "projected linear-definition decoder unexpectedly retains a final slot",
            ));
        }
        SelectiveProjectedSourceResolution::LinearDefinition
    } else if plan.widths[column] == 0 {
        if slots[column].is_some() {
            return Err(trace_error(
                "projected trace-eliminated decoder unexpectedly retains a final slot",
            ));
        }
        SelectiveProjectedSourceResolution::TraceEliminated
    } else if let Some((source, digit)) = aliases[column] {
        let (start, width) =
            slots[column].ok_or_else(|| trace_error("projected decomposition alias omitted its final slot"))?;
        let (source_start, source_width) =
            slots[source].ok_or_else(|| trace_error("projected decomposition alias source omitted its final slot"))?;
        if source >= column || width != 1 || digit >= source_width || start != source_start + digit {
            return Err(trace_error(
                "projected decomposition alias differs from its prepared source slot",
            ));
        }
        SelectiveProjectedSourceResolution::DecompositionAlias {
            source,
            digit,
            start,
            centered: plan.centered[column],
        }
    } else if let Some(source) = equal_aliases[column] {
        let (start, width) =
            slots[column].ok_or_else(|| trace_error("projected equality alias omitted its final slot"))?;
        if source >= column
            || slots[source] != Some((start, width))
            || plan.widths[source] != width
            || plan.centered[source] != plan.centered[column]
        {
            return Err(trace_error(
                "projected equality alias differs from its prepared source slot",
            ));
        }
        SelectiveProjectedSourceResolution::EqualityAlias {
            source,
            start,
            width,
            centered: plan.centered[column],
        }
    } else {
        let (start, width) =
            slots[column].ok_or_else(|| trace_error("projected direct decoder omitted its final slot"))?;
        if width != plan.widths[column] || width == 0 {
            return Err(trace_error(
                "projected direct decoder width differs from its prepared source width",
            ));
        }
        SelectiveProjectedSourceResolution::Direct {
            start,
            width,
            centered: plan.centered[column],
        }
    };
    Ok(resolution)
}

fn affine_step(first: usize, step: usize, length: usize, next: usize) -> Option<usize> {
    if length == 1 {
        next.checked_sub(first)
    } else {
        (first.checked_add(step.checked_mul(length)?)? == next).then_some(step)
    }
}

fn singleton_run_resolution(
    resolution: SelectiveProjectedSourceResolution,
) -> Option<SelectiveProjectedSourceResolutionRun> {
    Some(match resolution {
        SelectiveProjectedSourceResolution::ConstantOne => return None,
        SelectiveProjectedSourceResolution::Direct { start, width, centered } => {
            SelectiveProjectedSourceResolutionRun::Direct {
                start,
                start_stride: 0,
                width,
                centered,
            }
        }
        SelectiveProjectedSourceResolution::DecompositionAlias {
            source,
            digit,
            start,
            centered,
        } => SelectiveProjectedSourceResolutionRun::DecompositionAlias {
            source,
            source_stride: 0,
            digit,
            digit_stride: 0,
            start,
            start_stride: 0,
            centered,
        },
        SelectiveProjectedSourceResolution::EqualityAlias {
            source,
            start,
            width,
            centered,
        } => SelectiveProjectedSourceResolutionRun::EqualityAlias {
            source,
            source_stride: 0,
            start,
            start_stride: 0,
            width,
            centered,
        },
        SelectiveProjectedSourceResolution::LinearDefinition => SelectiveProjectedSourceResolutionRun::LinearDefinition,
        SelectiveProjectedSourceResolution::TraceEliminated => SelectiveProjectedSourceResolutionRun::TraceEliminated,
    })
}

fn extend_run_resolution(
    run: SelectiveProjectedSourceResolutionRun,
    length: usize,
    next: SelectiveProjectedSourceResolution,
) -> Option<SelectiveProjectedSourceResolutionRun> {
    Some(match (run, next) {
        (
            SelectiveProjectedSourceResolutionRun::Direct {
                start,
                start_stride,
                width,
                centered,
            },
            SelectiveProjectedSourceResolution::Direct {
                start: next_start,
                width: next_width,
                centered: next_centered,
            },
        ) if width == next_width && centered == next_centered => SelectiveProjectedSourceResolutionRun::Direct {
            start,
            start_stride: affine_step(start, start_stride, length, next_start)?,
            width,
            centered,
        },
        (
            SelectiveProjectedSourceResolutionRun::DecompositionAlias {
                source,
                source_stride,
                digit,
                digit_stride,
                start,
                start_stride,
                centered,
            },
            SelectiveProjectedSourceResolution::DecompositionAlias {
                source: next_source,
                digit: next_digit,
                start: next_start,
                centered: next_centered,
            },
        ) if centered == next_centered => SelectiveProjectedSourceResolutionRun::DecompositionAlias {
            source,
            source_stride: affine_step(source, source_stride, length, next_source)?,
            digit,
            digit_stride: affine_step(digit, digit_stride, length, next_digit)?,
            start,
            start_stride: affine_step(start, start_stride, length, next_start)?,
            centered,
        },
        (
            SelectiveProjectedSourceResolutionRun::EqualityAlias {
                source,
                source_stride,
                start,
                start_stride,
                width,
                centered,
            },
            SelectiveProjectedSourceResolution::EqualityAlias {
                source: next_source,
                start: next_start,
                width: next_width,
                centered: next_centered,
            },
        ) if width == next_width && centered == next_centered => SelectiveProjectedSourceResolutionRun::EqualityAlias {
            source,
            source_stride: affine_step(source, source_stride, length, next_source)?,
            start,
            start_stride: affine_step(start, start_stride, length, next_start)?,
            width,
            centered,
        },
        (
            SelectiveProjectedSourceResolutionRun::LinearDefinition,
            SelectiveProjectedSourceResolution::LinearDefinition,
        ) => SelectiveProjectedSourceResolutionRun::LinearDefinition,
        (
            SelectiveProjectedSourceResolutionRun::TraceEliminated,
            SelectiveProjectedSourceResolution::TraceEliminated,
        ) => SelectiveProjectedSourceResolutionRun::TraceEliminated,
        _ => return None,
    })
}

/// Compress the exact prepared decoder over one complete source interval.
/// The construction checks every source column against the same slots and
/// plans used by the production selective emitter; it never accepts census
/// totals as a substitute for pointwise data.
pub(super) fn decoder_run_provenance(
    layout: &SelectiveLayout,
    arm: usize,
    source_range: Range<usize>,
    column_families: &[crate::engine::r1cs_circuit::builder::ColumnFamilyRange],
) -> Result<SelectiveProjectedDecoderRunProvenance, LowNormR1csError> {
    let Some(slots) = layout.slots.get(arm) else {
        return Err(trace_error("complete decoder arm is out of range"));
    };
    if source_range.is_empty() || source_range.start == 0 || source_range.end > slots.len() {
        return Err(trace_error(
            "complete private decoder source interval is empty or out of range",
        ));
    }

    let first_resolution = source_resolution(layout, arm, source_range.start)?;
    let mut current = SelectiveProjectedSourceDecoderRun {
        source_start: source_range.start,
        length: 1,
        resolution: singleton_run_resolution(first_resolution)
            .ok_or_else(|| trace_error("complete private decoder unexpectedly contains constant one"))?,
    };
    let mut runs = Vec::new();
    for column in source_range.start + 1..source_range.end {
        let resolution = source_resolution(layout, arm, column)?;
        if let Some(extended) = extend_run_resolution(current.resolution, current.length, resolution) {
            current.length += 1;
            current.resolution = extended;
        } else {
            runs.push(current);
            current = SelectiveProjectedSourceDecoderRun {
                source_start: column,
                length: 1,
                resolution: singleton_run_resolution(resolution)
                    .ok_or_else(|| trace_error("complete private decoder unexpectedly contains constant one"))?,
            };
        }
    }
    runs.push(current);

    let mut cursor = source_range.start;
    for run in &runs {
        if run.source_start != cursor || run.length == 0 || run.source_end() > source_range.end {
            return Err(trace_error(
                "complete private decoder runs do not exactly partition the source interval",
            ));
        }
        let first = source_resolution(layout, arm, run.source_start)?;
        let last_column = run.source_end() - 1;
        let last = source_resolution(layout, arm, last_column)?;
        if run.resolution_at(run.source_start) != Some(first) || run.resolution_at(last_column) != Some(last) {
            return Err(trace_error(
                "complete private decoder affine run does not preserve endpoint resolutions",
            ));
        }
        cursor = run.source_end();
    }
    if cursor != source_range.end {
        return Err(trace_error(
            "complete private decoder leaves an uncovered source suffix",
        ));
    }

    let mut source_families = Vec::new();
    for family in column_families {
        if family.column_start > family.column_end || family.column_end > slots.len() {
            return Err(trace_error(
                "complete private decoder source-family interval is malformed",
            ));
        }
        let start = family.column_start.max(source_range.start);
        let end = family.column_end.min(source_range.end);
        if start < end {
            source_families.push(SelectiveProjectedSourceFamilyRange {
                path: family.name,
                source_range: start..end,
            });
        }
    }

    Ok(SelectiveProjectedDecoderRunProvenance {
        arm,
        source_range,
        runs,
        source_families,
    })
}
