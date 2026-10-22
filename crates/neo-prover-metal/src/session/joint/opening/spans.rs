//! Host census of one matrix's geometric runs for the opening forms.

use std::sync::atomic::{AtomicUsize, Ordering};

use neo_math::D;
use neo_reductions::superneo_eval::SuperneoMatrixCache;

use crate::MetalError;

pub(super) struct CensusBudget(AtomicUsize);

impl CensusBudget {
    pub(super) fn new(bytes: usize) -> Self {
        Self(AtomicUsize::new(bytes))
    }

    pub(super) fn left(&self) -> usize {
        self.0.load(Ordering::Relaxed)
    }

    fn reserve(&self, bytes: usize) -> bool {
        self.0
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |left| left.checked_sub(bytes))
            .is_ok()
    }
}

pub(super) struct SpanLayer {
    pub(super) starts: Vec<u64>,
    pub(super) ranks: Vec<u32>,
    pub(super) shapes: Vec<[u64; 2]>,
    pub(super) offsets: Vec<u32>,
    pub(super) rows: Vec<u32>,
    pub(super) coefficients: Vec<u64>,
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

pub(super) fn mark_blocks(layers: &[SpanLayer], active: &mut [bool]) {
    for layer in layers {
        for &[packed, _] in &layer.shapes {
            let start = (packed & u64::from(u32::MAX)) as usize;
            let end = start + (packed >> 32) as usize;
            active[start / D..end.div_ceil(D)].fill(true);
        }
    }
}

pub(super) fn span_layers(
    matrix: &SuperneoMatrixCache,
    columns: usize,
    budget: &CensusBudget,
) -> Result<Option<Vec<SpanLayer>>, MetalError> {
    let runs = matrix
        .compact_device_parts()
        .ok_or(MetalError::Shape("unfinished opening matrix"))?
        .geometric_runs;
    let mut invalid = false;
    let first = span_layer(runs, columns, budget, |visit| {
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
    let Some((layer, mut deferred)) = first else {
        return Ok(None);
    };
    let mut layers = vec![layer];
    while !deferred.is_empty() {
        let next = span_layer(runs, columns, budget, |visit| {
            for &[row, index] in &deferred {
                visit(row, index);
            }
        })?;
        let Some((layer, next)) = next else {
            return Ok(None);
        };
        layers.push(layer);
        deferred = next;
    }
    Ok(Some(layers))
}

fn span_layer(
    runs: &[[u64; 3]],
    columns: usize,
    budget: &CensusBudget,
    mut references: impl FnMut(&mut dyn FnMut(u32, u32)),
) -> Result<Option<(SpanLayer, Vec<[u32; 2]>)>, MetalError> {
    let words = columns / 64 + 1;
    if !budget.reserve(words * (size_of::<u64>() + size_of::<u32>())) {
        return Ok(None);
    }
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

    let spans = spans as usize;
    if !budget.reserve(spans * (4 * size_of::<u32>() + size_of::<[u64; 2]>()) + size_of::<u32>()) {
        return Ok(None);
    }
    let mut representatives = vec![u32::MAX; spans];
    let mut counts = vec![0u32; spans];
    let mut deferred_count = 0usize;
    references(&mut |_, index| {
        let run = runs[index as usize];
        let span = span_of(run[0]);
        if representatives[span] == u32::MAX {
            representatives[span] = index;
        }
        if same_shape(runs[representatives[span] as usize], run) {
            counts[span] += 1;
        } else {
            deferred_count += 1;
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
    if !budget.reserve(total * (size_of::<u32>() + size_of::<u64>()) + deferred_count * size_of::<[u32; 2]>()) {
        return Ok(None);
    }
    let (mut rows, mut coefficients) = (vec![0u32; total], vec![0u64; total]);
    let mut deferred = Vec::with_capacity(deferred_count);
    references(&mut |row, index| {
        let run = runs[index as usize];
        let span = span_of(run[0]);
        if same_shape(runs[representatives[span] as usize], run) {
            let cursor = cursors[span] as usize;
            (rows[cursor], coefficients[cursor]) = (row, run[1]);
            cursors[span] += 1;
        } else {
            deferred.push([row, index]);
        }
    });
    let shapes = representatives
        .iter()
        .map(|&run| [runs[run as usize][0], runs[run as usize][2]])
        .collect();
    Ok(Some((
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
    )))
}
