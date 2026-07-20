//! Audit-only source-column decoder for bounded selective projections.
//!
//! Owns: the exact prepared-layout disposition of caller-selected source
//! fields. Does not own row semantics, protocol authority, or generated data.

use std::collections::BTreeSet;

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
    let plan = &layout.plans[arm];
    let aliases = &layout.aliases[arm];
    let equal_aliases = &layout.equal_aliases[arm];
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
            let (source_start, source_width) = slots[source]
                .ok_or_else(|| trace_error("projected decomposition alias source omitted its final slot"))?;
            if width != 1 || digit >= source_width || start != source_start + digit {
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
            if slots[source] != Some((start, width))
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
        decoders.push(SelectiveProjectedSourceDecoder { column, resolution });
    }
    Ok(SelectiveProjectedDecoderProvenance { arm, decoders })
}
