//! Audit-only source-column decoder for bounded selective projections.
//!
//! Owns: the exact prepared-layout disposition of caller-selected source
//! fields. Does not own row semantics, protocol authority, or generated data.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use super::super::SparseR1cs;
use super::{trace_error, LowNormR1csError, SelectiveArmPlan, SelectiveLayout, SelectiveLayoutCore};

#[derive(Clone, Copy)]
struct SelectiveDecoderLayout<'a> {
    slots: &'a [Vec<Option<(usize, usize)>>],
    plans: &'a [SelectiveArmPlan],
    aliases: &'a [Vec<Option<(usize, usize)>>],
    equal_aliases: &'a [Vec<Option<usize>>],
    final_columns: usize,
}

impl<'a> SelectiveDecoderLayout<'a> {
    fn from_finished(layout: &'a SelectiveLayout) -> Self {
        Self {
            slots: &layout.slots,
            plans: &layout.plans,
            aliases: &layout.aliases,
            equal_aliases: &layout.equal_aliases,
            final_columns: layout.compiler_audit.layout().total_columns(),
        }
    }

    fn from_core(layout: &'a SelectiveLayoutCore) -> Self {
        Self {
            slots: &layout.slots,
            plans: &layout.plans,
            aliases: &layout.aliases,
            equal_aliases: &layout.equal_aliases,
            final_columns: layout.summary().columns,
        }
    }
}

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
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SelectiveProjectedSourceDecoderRun {
    source_start: usize,
    length: usize,
    resolution: SelectiveProjectedSourceResolutionRun,
}

/// One affine decoder rule over an arithmetic progression of source
/// columns. Numeric resolution strides are per selected source item, not per
/// unit source-column offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectiveProjectedSourceDecoderStridedRun {
    source_start: usize,
    count: usize,
    source_stride: usize,
    resolution: SelectiveProjectedSourceResolutionRun,
}

/// Affine repetition of one exact decoder template. The source and final
/// starts identify the first instance; both strides are per instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SelectiveProjectedSourceDecoderTemplateInstances {
    source_start: usize,
    count: usize,
    source_stride: usize,
    final_start: usize,
    final_stride: usize,
    reference_start: usize,
    reference_stride: usize,
    reference_final_start: usize,
    reference_final_stride: usize,
}

/// One repeated contiguous decoder template. Rule source starts and direct
/// final starts are relative to each concrete instance.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SelectiveProjectedSourceDecoderTemplate {
    source_width: usize,
    relative_runs: Vec<SelectiveProjectedSourceDecoderRun>,
    instances: Vec<SelectiveProjectedSourceDecoderTemplateInstances>,
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
        resolution_at_offset(self.resolution, offset)
    }
}

impl SelectiveProjectedSourceDecoderStridedRun {
    pub fn source_start(self) -> usize {
        self.source_start
    }

    pub fn count(self) -> usize {
        self.count
    }

    pub fn source_stride(self) -> usize {
        self.source_stride
    }

    pub fn source_column(self, index: usize) -> Option<usize> {
        (index < self.count).then(|| {
            self.source_start
                .checked_add(self.source_stride.checked_mul(index)?)
        })?
    }

    pub fn resolution(self) -> SelectiveProjectedSourceResolutionRun {
        self.resolution
    }

    /// Expand one owned source column back to the exact pointwise decoder.
    pub fn resolution_at(self, column: usize) -> Option<SelectiveProjectedSourceResolution> {
        let delta = column.checked_sub(self.source_start)?;
        if self.source_stride == 0 || delta % self.source_stride != 0 {
            return None;
        }
        let offset = delta / self.source_stride;
        if offset >= self.count {
            return None;
        }
        resolution_at_offset(self.resolution, offset)
    }
}

impl SelectiveProjectedSourceDecoderTemplateInstances {
    pub fn source_start(self) -> usize {
        self.source_start
    }

    pub fn count(self) -> usize {
        self.count
    }

    pub fn source_stride(self) -> usize {
        self.source_stride
    }

    pub fn final_start(self) -> usize {
        self.final_start
    }

    pub fn final_stride(self) -> usize {
        self.final_stride
    }

    pub fn reference_start(self) -> usize {
        self.reference_start
    }

    pub fn reference_stride(self) -> usize {
        self.reference_stride
    }

    pub fn reference_final_start(self) -> usize {
        self.reference_final_start
    }

    pub fn reference_final_stride(self) -> usize {
        self.reference_final_stride
    }

    pub fn instance(self, index: usize) -> Option<(usize, usize, usize, usize)> {
        if index >= self.count {
            return None;
        }
        Some((
            self.source_start
                .checked_add(self.source_stride.checked_mul(index)?)?,
            self.final_start
                .checked_add(self.final_stride.checked_mul(index)?)?,
            self.reference_start
                .checked_add(self.reference_stride.checked_mul(index)?)?,
            self.reference_final_start
                .checked_add(self.reference_final_stride.checked_mul(index)?)?,
        ))
    }
}

impl SelectiveProjectedSourceDecoderTemplate {
    pub fn source_width(&self) -> usize {
        self.source_width
    }

    pub fn relative_runs(&self) -> &[SelectiveProjectedSourceDecoderRun] {
        &self.relative_runs
    }

    pub fn instances(&self) -> &[SelectiveProjectedSourceDecoderTemplateInstances] {
        &self.instances
    }
}

fn resolution_at_offset(
    resolution: SelectiveProjectedSourceResolutionRun,
    offset: usize,
) -> Option<SelectiveProjectedSourceResolution> {
    let affine = |first: usize, stride: usize| first.checked_add(stride.checked_mul(offset)?);
    Some(match resolution {
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
        SelectiveProjectedSourceResolutionRun::LinearDefinition => SelectiveProjectedSourceResolution::LinearDefinition,
        SelectiveProjectedSourceResolutionRun::TraceEliminated => SelectiveProjectedSourceResolution::TraceEliminated,
    })
}

/// Complete run-compressed decoder for one exact source interval.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectiveProjectedDecoderRunProvenance {
    arm: usize,
    source_range: Range<usize>,
    final_columns: usize,
    runs: Vec<SelectiveProjectedSourceDecoderRun>,
    strided_runs: Vec<SelectiveProjectedSourceDecoderStridedRun>,
    repeated_templates: Vec<SelectiveProjectedSourceDecoderTemplate>,
    residual_strided_runs: Vec<SelectiveProjectedSourceDecoderStridedRun>,
    source_families: Vec<SelectiveProjectedSourceFamilyRange>,
}

impl SelectiveProjectedDecoderRunProvenance {
    pub fn arm(&self) -> usize {
        self.arm
    }

    pub fn source_range(&self) -> Range<usize> {
        self.source_range.clone()
    }

    pub fn final_columns(&self) -> usize {
        self.final_columns
    }

    pub fn runs(&self) -> &[SelectiveProjectedSourceDecoderRun] {
        &self.runs
    }

    /// Exact second-level compression of the same pointwise decoder. These
    /// rules can interleave, but they are checked to be disjoint and complete.
    pub fn strided_runs(&self) -> &[SelectiveProjectedSourceDecoderStridedRun] {
        &self.strided_runs
    }

    /// Repeated decoder templates extracted from exact Poseidon2 and
    /// canonical-opening allocation ranges and checked against the pointwise
    /// layout.
    pub fn repeated_templates(&self) -> &[SelectiveProjectedSourceDecoderTemplate] {
        &self.repeated_templates
    }

    /// Strided rules for every source column outside the repeated templates.
    pub fn residual_strided_runs(&self) -> &[SelectiveProjectedSourceDecoderStridedRun] {
        &self.residual_strided_runs
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
    decoder_provenance_from_layout(
        &SelectiveDecoderLayout::from_finished(layout),
        arm,
        requested_source_columns,
    )
}

fn decoder_provenance_from_layout(
    layout: &SelectiveDecoderLayout<'_>,
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
    layout: &SelectiveDecoderLayout<'_>,
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

fn compress_contiguous_runs(
    pointwise: &[(usize, SelectiveProjectedSourceResolution)],
) -> Result<Vec<SelectiveProjectedSourceDecoderRun>, LowNormR1csError> {
    let Some(&(source_start, first_resolution)) = pointwise.first() else {
        return Err(trace_error("complete decoder pointwise interval is empty"));
    };
    let mut current = SelectiveProjectedSourceDecoderRun {
        source_start,
        length: 1,
        resolution: singleton_run_resolution(first_resolution)
            .ok_or_else(|| trace_error("complete private decoder unexpectedly contains constant one"))?,
    };
    let mut runs = Vec::new();
    for &(column, resolution) in &pointwise[1..] {
        if column != current.source_end() {
            return Err(trace_error("complete decoder pointwise interval is not consecutive"));
        }
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
    Ok(runs)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ResolutionShape {
    Direct { width: usize, centered: bool },
    DecompositionAlias { centered: bool },
    EqualityAlias { width: usize, centered: bool },
    LinearDefinition,
    TraceEliminated,
}

fn resolution_shape(resolution: SelectiveProjectedSourceResolution) -> Option<ResolutionShape> {
    Some(match resolution {
        SelectiveProjectedSourceResolution::ConstantOne => return None,
        SelectiveProjectedSourceResolution::Direct { width, centered, .. } => {
            ResolutionShape::Direct { width, centered }
        }
        SelectiveProjectedSourceResolution::DecompositionAlias { centered, .. } => {
            ResolutionShape::DecompositionAlias { centered }
        }
        SelectiveProjectedSourceResolution::EqualityAlias { width, centered, .. } => {
            ResolutionShape::EqualityAlias { width, centered }
        }
        SelectiveProjectedSourceResolution::LinearDefinition => ResolutionShape::LinearDefinition,
        SelectiveProjectedSourceResolution::TraceEliminated => ResolutionShape::TraceEliminated,
    })
}

fn compress_strided_runs(
    pointwise: &[(usize, SelectiveProjectedSourceResolution)],
) -> Result<Vec<SelectiveProjectedSourceDecoderStridedRun>, LowNormR1csError> {
    let mut by_shape = BTreeMap::<ResolutionShape, Vec<(usize, SelectiveProjectedSourceResolution)>>::new();
    for &(column, resolution) in pointwise {
        let shape = resolution_shape(resolution)
            .ok_or_else(|| trace_error("complete private decoder unexpectedly contains constant one"))?;
        by_shape
            .entry(shape)
            .or_default()
            .push((column, resolution));
    }

    let mut compressed = Vec::new();
    for points in by_shape.values() {
        let mut cursor = 0;
        while cursor < points.len() {
            let (source_start, first) = points[cursor];
            let singleton = singleton_run_resolution(first)
                .ok_or_else(|| trace_error("strided decoder unexpectedly contains constant one"))?;
            if cursor + 1 == points.len() {
                compressed.push(SelectiveProjectedSourceDecoderStridedRun {
                    source_start,
                    count: 1,
                    source_stride: 1,
                    resolution: singleton,
                });
                break;
            }

            let (second_column, second) = points[cursor + 1];
            let Some(source_stride) = second_column
                .checked_sub(source_start)
                .filter(|&stride| stride > 0)
            else {
                return Err(trace_error(
                    "strided decoder source columns are not strictly increasing",
                ));
            };
            let Some(resolution) = extend_run_resolution(singleton, 1, second) else {
                compressed.push(SelectiveProjectedSourceDecoderStridedRun {
                    source_start,
                    count: 1,
                    source_stride: 1,
                    resolution: singleton,
                });
                cursor += 1;
                continue;
            };

            let mut count = 2;
            while cursor + count < points.len() {
                let (next_column, next) = points[cursor + count];
                let Some(expected_column) = source_start.checked_add(
                    source_stride
                        .checked_mul(count)
                        .ok_or_else(|| trace_error("strided decoder source stride overflow"))?,
                ) else {
                    return Err(trace_error("strided decoder source column overflow"));
                };
                if next_column != expected_column || extend_run_resolution(resolution, count, next) != Some(resolution)
                {
                    break;
                }
                count += 1;
            }
            compressed.push(SelectiveProjectedSourceDecoderStridedRun {
                source_start,
                count,
                source_stride,
                resolution,
            });
            cursor += count;
        }
    }
    compressed.sort_by_key(|run| run.source_start);
    Ok(compressed)
}

fn normalize_template_resolution(
    resolution: SelectiveProjectedSourceResolution,
    final_start: usize,
    reference_start: usize,
    reference_final_start: usize,
) -> Option<SelectiveProjectedSourceResolution> {
    Some(match resolution {
        SelectiveProjectedSourceResolution::Direct { start, width, centered } => {
            SelectiveProjectedSourceResolution::Direct {
                start: start.checked_sub(final_start)?,
                width,
                centered,
            }
        }
        SelectiveProjectedSourceResolution::DecompositionAlias {
            source,
            digit,
            start,
            centered,
        } => SelectiveProjectedSourceResolution::DecompositionAlias {
            source: source.checked_sub(reference_start)?,
            digit,
            start: start.checked_sub(reference_final_start)?,
            centered,
        },
        SelectiveProjectedSourceResolution::EqualityAlias {
            source,
            start,
            width,
            centered,
        } => SelectiveProjectedSourceResolution::EqualityAlias {
            source: source.checked_sub(reference_start)?,
            start: start.checked_sub(reference_final_start)?,
            width,
            centered,
        },
        SelectiveProjectedSourceResolution::LinearDefinition => SelectiveProjectedSourceResolution::LinearDefinition,
        SelectiveProjectedSourceResolution::TraceEliminated => SelectiveProjectedSourceResolution::TraceEliminated,
        SelectiveProjectedSourceResolution::ConstantOne => return None,
    })
}

fn concrete_template_resolution(
    resolution: SelectiveProjectedSourceResolution,
    final_start: usize,
    reference_start: usize,
    reference_final_start: usize,
) -> Option<SelectiveProjectedSourceResolution> {
    Some(match resolution {
        SelectiveProjectedSourceResolution::Direct { start, width, centered } => {
            SelectiveProjectedSourceResolution::Direct {
                start: final_start.checked_add(start)?,
                width,
                centered,
            }
        }
        SelectiveProjectedSourceResolution::DecompositionAlias {
            source,
            digit,
            start,
            centered,
        } => SelectiveProjectedSourceResolution::DecompositionAlias {
            source: reference_start.checked_add(source)?,
            digit,
            start: reference_final_start.checked_add(start)?,
            centered,
        },
        SelectiveProjectedSourceResolution::EqualityAlias {
            source,
            start,
            width,
            centered,
        } => SelectiveProjectedSourceResolution::EqualityAlias {
            source: reference_start.checked_add(source)?,
            start: reference_final_start.checked_add(start)?,
            width,
            centered,
        },
        SelectiveProjectedSourceResolution::LinearDefinition => SelectiveProjectedSourceResolution::LinearDefinition,
        SelectiveProjectedSourceResolution::TraceEliminated => SelectiveProjectedSourceResolution::TraceEliminated,
        SelectiveProjectedSourceResolution::ConstantOne => return None,
    })
}

fn compress_template_instances(
    instances: &[(usize, usize, usize, usize)],
) -> Result<Vec<SelectiveProjectedSourceDecoderTemplateInstances>, LowNormR1csError> {
    let mut compressed = Vec::new();
    let mut cursor = 0;
    while cursor < instances.len() {
        let (source_start, final_start, reference_start, reference_final_start) = instances[cursor];
        if cursor + 1 == instances.len() {
            compressed.push(SelectiveProjectedSourceDecoderTemplateInstances {
                source_start,
                count: 1,
                source_stride: 1,
                final_start,
                final_stride: 0,
                reference_start,
                reference_stride: 0,
                reference_final_start,
                reference_final_stride: 0,
            });
            break;
        }
        let (next_source, next_final, next_reference, next_reference_final) = instances[cursor + 1];
        let Some(source_stride) = next_source
            .checked_sub(source_start)
            .filter(|&stride| stride > 0)
        else {
            return Err(trace_error("decoder template sources are not strictly increasing"));
        };
        let Some(final_stride) = next_final.checked_sub(final_start) else {
            compressed.push(SelectiveProjectedSourceDecoderTemplateInstances {
                source_start,
                count: 1,
                source_stride: 1,
                final_start,
                final_stride: 0,
                reference_start,
                reference_stride: 0,
                reference_final_start,
                reference_final_stride: 0,
            });
            cursor += 1;
            continue;
        };
        let Some(reference_stride) = next_reference.checked_sub(reference_start) else {
            compressed.push(SelectiveProjectedSourceDecoderTemplateInstances {
                source_start,
                count: 1,
                source_stride: 1,
                final_start,
                final_stride: 0,
                reference_start,
                reference_stride: 0,
                reference_final_start,
                reference_final_stride: 0,
            });
            cursor += 1;
            continue;
        };
        let Some(reference_final_stride) = next_reference_final.checked_sub(reference_final_start) else {
            compressed.push(SelectiveProjectedSourceDecoderTemplateInstances {
                source_start,
                count: 1,
                source_stride: 1,
                final_start,
                final_stride: 0,
                reference_start,
                reference_stride: 0,
                reference_final_start,
                reference_final_stride: 0,
            });
            cursor += 1;
            continue;
        };
        let mut count = 2;
        while cursor + count < instances.len() {
            let (source, final_column, reference, reference_final) = instances[cursor + count];
            if source_start.checked_add(
                source_stride
                    .checked_mul(count)
                    .ok_or_else(|| trace_error("decoder template source stride overflow"))?,
            ) != Some(source)
                || final_start.checked_add(
                    final_stride
                        .checked_mul(count)
                        .ok_or_else(|| trace_error("decoder template final stride overflow"))?,
                ) != Some(final_column)
                || reference_start.checked_add(
                    reference_stride
                        .checked_mul(count)
                        .ok_or_else(|| trace_error("decoder template reference stride overflow"))?,
                ) != Some(reference)
                || reference_final_start.checked_add(
                    reference_final_stride
                        .checked_mul(count)
                        .ok_or_else(|| trace_error("decoder template reference-final stride overflow"))?,
                ) != Some(reference_final)
            {
                break;
            }
            count += 1;
        }
        compressed.push(SelectiveProjectedSourceDecoderTemplateInstances {
            source_start,
            count,
            source_stride,
            final_start,
            final_stride,
            reference_start,
            reference_stride,
            reference_final_start,
            reference_final_stride,
        });
        cursor += count;
    }
    Ok(compressed)
}

fn repeated_decoder_templates(
    layout: &SelectiveDecoderLayout<'_>,
    arm: usize,
    source_range: &Range<usize>,
    source_arm: &SparseR1cs,
    pointwise: &[(usize, SelectiveProjectedSourceResolution)],
) -> Result<
    (
        Vec<SelectiveProjectedSourceDecoderTemplate>,
        Vec<SelectiveProjectedSourceDecoderStridedRun>,
    ),
    LowNormR1csError,
> {
    let mut groups =
        BTreeMap::<(usize, Vec<SelectiveProjectedSourceDecoderRun>), Vec<(usize, usize, usize, usize)>>::new();
    let mut blocks = Vec::<(Range<usize>, usize)>::new();
    for trace in source_arm.poseidon2_traces() {
        let Some(&source_start) = trace.allocated_columns.first() else {
            return Err(trace_error("Poseidon2 decoder template has no allocated columns"));
        };
        let source_end = source_start
            .checked_add(trace.allocated_columns.len())
            .ok_or_else(|| trace_error("Poseidon2 decoder template source range overflow"))?;
        if trace
            .allocated_columns
            .iter()
            .copied()
            .ne(source_start..source_end)
        {
            return Err(trace_error("Poseidon2 decoder template allocation is not consecutive"));
        }
        blocks.push((source_start..source_end, 0));
    }
    for trace in source_arm.shifted_ternary_canonical_traces() {
        let digit_end = trace
            .digit_columns_start
            .checked_add(super::BALANCED_FIELD_WIDTH)
            .ok_or_else(|| trace_error("canonical decoder template digit range overflow"))?;
        let negative_end = trace
            .negative_columns_start
            .checked_add(super::BALANCED_FIELD_WIDTH)
            .ok_or_else(|| trace_error("canonical decoder template negative range overflow"))?;
        let source_end = trace
            .borrow_columns_start
            .checked_add(super::BALANCED_FIELD_WIDTH - 1)
            .ok_or_else(|| trace_error("canonical decoder template borrow range overflow"))?;
        if trace.negative_columns_start != digit_end || trace.borrow_columns_start != negative_end {
            return Err(trace_error(
                "canonical decoder template allocations are not consecutive",
            ));
        }
        blocks.push((trace.digit_columns_start..source_end, trace.field_column));
    }
    blocks.sort_by_key(|(range, _)| range.start);
    let mut template_columns = BTreeSet::new();
    for (source_block, reference_start) in blocks {
        let source_start = source_block.start;
        let source_end = source_block.end;
        let source_width = source_block.len();
        if source_end <= source_range.start || source_start >= source_range.end {
            continue;
        }
        if source_start < source_range.start || source_end > source_range.end {
            continue;
        }
        let block = &pointwise[source_start - source_range.start..source_end - source_range.start];
        let final_start = block
            .iter()
            .filter_map(|(_, resolution)| match resolution {
                SelectiveProjectedSourceResolution::Direct { start, .. } => Some(*start),
                _ => None,
            })
            .min()
            .ok_or_else(|| trace_error("Poseidon2 decoder template has no retained final coordinate"))?;
        let reference_final_start = block
            .iter()
            .filter_map(|(_, resolution)| match resolution {
                SelectiveProjectedSourceResolution::DecompositionAlias { start, .. }
                | SelectiveProjectedSourceResolution::EqualityAlias { start, .. } => Some(*start),
                _ => None,
            })
            .min()
            .unwrap_or(0);
        let mut normalized = Vec::with_capacity(source_width);
        for &(column, resolution) in block {
            let relative =
                normalize_template_resolution(resolution, final_start, reference_start, reference_final_start)
                    .ok_or_else(|| trace_error("decoder template contains an unsupported resolution"))?;
            normalized.push((column - source_start, relative));
            if !template_columns.insert(column) {
                return Err(trace_error("Poseidon2 decoder template source ranges overlap"));
            }
        }
        let relative_runs = compress_contiguous_runs(&normalized)?;
        groups
            .entry((source_width, relative_runs))
            .or_default()
            .push((source_start, final_start, reference_start, reference_final_start));
    }

    let mut repeated_templates = Vec::with_capacity(groups.len());
    for ((source_width, relative_runs), mut instances) in groups {
        instances.sort_unstable();
        repeated_templates.push(SelectiveProjectedSourceDecoderTemplate {
            source_width,
            relative_runs,
            instances: compress_template_instances(&instances)?,
        });
    }
    let residual = pointwise
        .iter()
        .copied()
        .filter(|(column, _)| !template_columns.contains(column))
        .collect::<Vec<_>>();
    let residual_strided_runs = compress_strided_runs(&residual)?;

    let mut owners = BTreeSet::new();
    for template in &repeated_templates {
        for batch in &template.instances {
            for instance_index in 0..batch.count {
                let (instance_source, instance_final, instance_reference, instance_reference_final) = batch
                    .instance(instance_index)
                    .ok_or_else(|| trace_error("decoder template instance overflow"))?;
                for relative_run in &template.relative_runs {
                    for relative_column in relative_run.source_start()..relative_run.source_end() {
                        let column = instance_source
                            .checked_add(relative_column)
                            .ok_or_else(|| trace_error("decoder template source column overflow"))?;
                        let relative_resolution = relative_run
                            .resolution_at(relative_column)
                            .ok_or_else(|| trace_error("decoder template relative rule has a gap"))?;
                        let resolution = concrete_template_resolution(
                            relative_resolution,
                            instance_final,
                            instance_reference,
                            instance_reference_final,
                        )
                        .ok_or_else(|| trace_error("decoder template relative rule is unsupported"))?;
                        if !owners.insert(column) || resolution != source_resolution(layout, arm, column)? {
                            return Err(trace_error(
                                "decoder template differs from the prepared source resolution",
                            ));
                        }
                    }
                }
            }
        }
    }
    for run in &residual_strided_runs {
        for index in 0..run.count {
            let column = run
                .source_column(index)
                .ok_or_else(|| trace_error("residual strided decoder source column overflow"))?;
            if !owners.insert(column) || run.resolution_at(column) != Some(source_resolution(layout, arm, column)?) {
                return Err(trace_error(
                    "residual strided decoder differs from the prepared source resolution",
                ));
            }
        }
    }
    if owners.len() != source_range.len()
        || owners.first().copied() != Some(source_range.start)
        || owners.last().copied() != Some(source_range.end - 1)
    {
        return Err(trace_error(
            "template decoder does not exactly cover the source interval",
        ));
    }
    Ok((repeated_templates, residual_strided_runs))
}

/// Compress the exact prepared decoder over one complete source interval.
/// The construction checks every source column against the same slots and
/// plans used by the production selective emitter; it never accepts census
/// totals as a substitute for pointwise data.
pub(super) fn decoder_run_provenance(
    layout: &SelectiveLayout,
    arm: usize,
    source_range: Range<usize>,
    source_arm: &SparseR1cs,
) -> Result<SelectiveProjectedDecoderRunProvenance, LowNormR1csError> {
    decoder_run_provenance_from_layout(
        &SelectiveDecoderLayout::from_finished(layout),
        arm,
        source_range,
        source_arm,
    )
}

pub(super) fn decoder_run_provenance_from_core(
    layout: &SelectiveLayoutCore,
    arm: usize,
    source_range: Range<usize>,
    source_arm: &SparseR1cs,
) -> Result<SelectiveProjectedDecoderRunProvenance, LowNormR1csError> {
    decoder_run_provenance_from_layout(
        &SelectiveDecoderLayout::from_core(layout),
        arm,
        source_range,
        source_arm,
    )
}

fn decoder_run_provenance_from_layout(
    layout: &SelectiveDecoderLayout<'_>,
    arm: usize,
    source_range: Range<usize>,
    source_arm: &SparseR1cs,
) -> Result<SelectiveProjectedDecoderRunProvenance, LowNormR1csError> {
    let Some(slots) = layout.slots.get(arm) else {
        return Err(trace_error("complete decoder arm is out of range"));
    };
    if source_range.is_empty() || source_range.start == 0 || source_range.end > slots.len() {
        return Err(trace_error(
            "complete private decoder source interval is empty or out of range",
        ));
    }

    let mut pointwise = Vec::with_capacity(source_range.len());
    for column in source_range.clone() {
        let resolution = source_resolution(layout, arm, column)?;
        pointwise.push((column, resolution));
    }
    let runs = compress_contiguous_runs(&pointwise)?;

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

    let strided_runs = compress_strided_runs(&pointwise)?;
    let mut strided_owners = BTreeSet::new();
    for run in &strided_runs {
        if run.count == 0 || run.source_stride == 0 {
            return Err(trace_error("strided decoder contains an empty or stationary rule"));
        }
        for index in 0..run.count {
            let column = run
                .source_column(index)
                .ok_or_else(|| trace_error("strided decoder source column overflow"))?;
            if !source_range.contains(&column) || !strided_owners.insert(column) {
                return Err(trace_error(
                    "strided decoder rules overlap or leave the source interval",
                ));
            }
            if run.resolution_at(column) != Some(source_resolution(layout, arm, column)?) {
                return Err(trace_error(
                    "strided decoder rule differs from the prepared source resolution",
                ));
            }
        }
    }
    if strided_owners.len() != source_range.len()
        || strided_owners.first().copied() != Some(source_range.start)
        || strided_owners.last().copied() != Some(source_range.end - 1)
    {
        return Err(trace_error(
            "strided decoder rules do not exactly cover the source interval",
        ));
    }
    let (repeated_templates, residual_strided_runs) =
        repeated_decoder_templates(layout, arm, &source_range, source_arm, &pointwise)?;

    let mut source_families = Vec::new();
    for family in source_arm.column_family_ranges() {
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
        final_columns: layout.final_columns,
        runs,
        strided_runs,
        repeated_templates,
        residual_strided_runs,
        source_families,
    })
}
