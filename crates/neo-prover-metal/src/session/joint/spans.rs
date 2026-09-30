//! Host census of one matrix's geometric runs for the opening forms.
//!
//! Runs with the same start, length and ratio form one span. The openings add
//! a span's weighted references once and then expand the sum to the span's
//! columns, instead of expanding every reference. One layer keeps one span
//! shape per start column; a start used with another shape moves those runs
//! to the next layer. Field addition is exact, so the grouping cannot change
//! an opening. A form coordinate scans the spans that start less than the
//! longest span length before it, so long spans make that scan long.

use neo_math::D;
use neo_reductions::superneo_eval::SuperneoMatrixCache;

use crate::MetalError;

/// Spans with distinct start columns and their grouped references.
pub(super) struct SpanLayer {
    /// Bit `column` is set when a span starts at `column`.
    pub(super) starts: Vec<u64>,
    /// Number of spans that start before each 64-column word.
    pub(super) ranks: Vec<u32>,
    /// `[start | length << 32, ratio]` of each span.
    pub(super) shapes: Vec<[u64; 2]>,
    /// Offsets of each span's references.
    pub(super) offsets: Vec<u32>,
    /// Row of each reference, grouped by span.
    pub(super) rows: Vec<u32>,
    /// Canonical coefficient of each reference, grouped by span.
    pub(super) coefficients: Vec<u64>,
    /// The longest span, which bounds the spans that can cover one column.
    pub(super) longest: u64,
}

impl SpanLayer {
    pub(super) fn span_count(&self) -> usize {
        self.shapes.len()
    }

    pub(super) fn device_bytes(&self) -> usize {
        size_of_val(self.starts.as_slice())
            + size_of_val(self.ranks.as_slice())
            + size_of_val(self.shapes.as_slice())
            + size_of_val(self.offsets.as_slice())
            + size_of_val(self.rows.as_slice())
            + size_of_val(self.coefficients.as_slice())
    }
}

/// Mark the column blocks that the spans of `layers` cover.
pub(super) fn mark_blocks(layers: &[SpanLayer], active: &mut [bool]) {
    for layer in layers {
        for &[packed, _] in &layer.shapes {
            let start = (packed & u64::from(u32::MAX)) as usize;
            let end = start + (packed >> 32) as usize;
            active[start / D..end.div_ceil(D)].fill(true);
        }
    }
}

/// Group the runs of `matrix` into span layers over `columns` carrier columns.
pub(super) fn span_layers(matrix: &SuperneoMatrixCache, columns: usize) -> Result<Vec<SpanLayer>, MetalError> {
    let runs = matrix
        .compact_device_parts()
        .ok_or(MetalError::Shape("unfinished opening matrix"))?
        .geometric_runs;
    let mut invalid = false;
    let (layer, mut deferred) = span_layer(runs, columns, |visit| {
        matrix.for_each_compact_geometric_run(|index, row, _, _, _, _| {
            match (u32::try_from(row), u32::try_from(index)) {
                (Ok(row), Ok(index)) => visit(row, index),
                _ => invalid = true,
            }
        })
    })?;
    if invalid {
        return Err(MetalError::Shape(
            "one-joint geometric opening metadata exceeds device limits",
        ));
    }
    let mut layers = vec![layer];
    while !deferred.is_empty() {
        let (layer, next) = span_layer(runs, columns, |visit| {
            for &[row, index] in &deferred {
                visit(row, index);
            }
        })?;
        layers.push(layer);
        deferred = next;
    }
    Ok(layers)
}

/// Build one layer from the visited `(row, run)` references, and return the
/// references whose start already has a span of another shape.
fn span_layer(
    runs: &[[u64; 3]],
    columns: usize,
    mut references: impl FnMut(&mut dyn FnMut(u32, u32)),
) -> Result<(SpanLayer, Vec<[u32; 2]>), MetalError> {
    let words = columns / 64 + 1;
    let mut starts = vec![0u64; words];
    let mut longest = 0u64;
    let mut outside = false;
    references(&mut |_, index| {
        let packed = runs[index as usize][0];
        let (start, len) = (packed & u64::from(u32::MAX), packed >> 32);
        if start + len > columns as u64 {
            outside = true;
            return;
        }
        starts[start as usize / 64] |= 1 << (start % 64);
        longest = longest.max(len);
    });
    if outside {
        return Err(MetalError::Shape("opening run exceeds the carrier"));
    }
    let mut ranks = Vec::with_capacity(words);
    let mut spans = 0u32;
    for &word in &starts {
        ranks.push(spans);
        spans += word.count_ones();
    }
    let span_of = |packed: u64| {
        let start = (packed & u64::from(u32::MAX)) as usize;
        (ranks[start / 64] + (starts[start / 64] & ((1u64 << (start % 64)) - 1)).count_ones()) as usize
    };
    let same_shape = |left: [u64; 3], right: [u64; 3]| left[0] == right[0] && left[2] == right[2];

    let mut representatives = vec![u32::MAX; spans as usize];
    let mut counts = vec![0u32; spans as usize];
    let mut deferred = Vec::new();
    references(&mut |row, index| {
        let run = runs[index as usize];
        let span = span_of(run[0]);
        if representatives[span] == u32::MAX {
            representatives[span] = index;
        }
        if same_shape(runs[representatives[span] as usize], run) {
            counts[span] += 1;
        } else {
            deferred.push([row, index]);
        }
    });
    let mut offsets = Vec::with_capacity(counts.len() + 1);
    offsets.push(0u32);
    for &count in &counts {
        let next = offsets[offsets.len() - 1]
            .checked_add(count)
            .ok_or(MetalError::Shape(
                "one-joint geometric opening reference count exceeds u32",
            ))?;
        offsets.push(next);
    }
    let mut cursors = offsets[..counts.len()].to_vec();
    let total = offsets[counts.len()] as usize;
    let (mut rows, mut coefficients) = (vec![0u32; total], vec![0u64; total]);
    references(&mut |row, index| {
        let run = runs[index as usize];
        let span = span_of(run[0]);
        if same_shape(runs[representatives[span] as usize], run) {
            let cursor = cursors[span] as usize;
            (rows[cursor], coefficients[cursor]) = (row, run[1]);
            cursors[span] += 1;
        }
    });
    let shapes = representatives
        .iter()
        .map(|&run| [runs[run as usize][0], runs[run as usize][2]])
        .collect();
    Ok((
        SpanLayer {
            starts,
            ranks,
            shapes,
            offsets,
            rows,
            coefficients,
            longest,
        },
        deferred,
    ))
}
