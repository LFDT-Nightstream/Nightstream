//! Scratch parts of one matrix with near-equal work, and for each part the
//! entries of each row that reach it. The work of a block is the number of
//! scratch coefficients that its entries update.

use core::cmp::{max, min};
use core::ops::Range;

use neo_math::D;

use super::{geometric::decode, SuperneoMatrixCache};

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

// More parts than threads let work stealing absorb the error of the estimate.
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
const PARTS_PER_THREAD: usize = 4;

#[derive(Clone, Debug)]
pub(super) struct PartPlan {
    bounds: Vec<usize>,
    row_offsets: Vec<usize>,
    rows: Vec<RowSpan>,
}

/// The entries of one row that a part visits: the compact entries and the
/// geometric runs in two index ranges of the matrix cache. Each range runs
/// from the first to the last entry of the row that reaches the part. The
/// part skips the other entries in the range.
#[derive(Clone, Copy, Debug)]
pub(super) struct RowSpan {
    pub(super) row: u32,
    compact: [u32; 2],
    runs: [u32; 2],
}

impl RowSpan {
    fn new(row: usize) -> Self {
        Self {
            row: u32::try_from(row).expect("SuperNeo row index exceeds u32"),
            compact: [u32::MAX, 0],
            runs: [u32::MAX, 0],
        }
    }

    pub(super) fn compact(self) -> Range<usize> {
        self.compact[0] as usize..self.compact[1] as usize
    }

    pub(super) fn runs(self) -> Range<usize> {
        self.runs[0] as usize..self.runs[1] as usize
    }
}

/// The span of `row` for `part` among the spans `out[first..]`, added if new.
/// Sorted entries reach the parts in order, so the search ends at once.
fn span(out: &mut Vec<(usize, RowSpan)>, first: usize, row: usize, part: usize) -> &mut RowSpan {
    let index = match out[first..].iter().rposition(|&(other, _)| other == part) {
        Some(index) => first + index,
        None => {
            out.push((part, RowSpan::new(row)));
            out.len() - 1
        }
    };
    &mut out[index].1
}

fn include(range: &mut [u32; 2], index: usize) {
    let index = u32::try_from(index).expect("SuperNeo entry index exceeds u32");
    range[0] = min(range[0], index);
    range[1] = max(range[1], index + 1);
}

fn close(range: &mut [u32; 2]) {
    if range[0] > range[1] {
        *range = [0, 0];
    }
}

impl PartPlan {
    /// Part `p` owns the blocks `bounds[p]..bounds[p + 1]`.
    pub(super) fn bounds(&self) -> &[usize] {
        &self.bounds
    }

    /// The rows with an entry in part `part`, in increasing row order.
    pub(super) fn rows(&self, part: usize) -> &[RowSpan] {
        &self.rows[self.row_offsets[part]..self.row_offsets[part + 1]]
    }
}

impl SuperneoMatrixCache {
    pub(super) fn part_plan(&self) -> &PartPlan {
        if let Some(plan) = self.part_plan.get() {
            return plan;
        }
        // The build waits for rayon tasks, so it does not run inside
        // `get_or_init`. A thread that waits there can run a task that asks
        // for this plan, and that task blocks on the cell. Two first callers
        // can each build the plan; the plans are equal.
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let threads = rayon::current_num_threads();
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let parts = threads * PARTS_PER_THREAD;
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        let (threads, parts) = (1, 1);
        let plan = self.build_part_plan(parts, threads);
        self.part_plan.get_or_init(|| plan)
    }

    /// All entries of `row`.
    pub(super) fn whole_row(&self, row: usize) -> RowSpan {
        let mut span = RowSpan::new(row);
        for (range, entries) in [
            (&mut span.compact, self.row_offsets.range(row)),
            (&mut span.runs, self.geometric_row_offsets.range(row)),
        ] {
            *range = [entries.start, entries.end].map(|index| index as u32);
        }
        span
    }

    /// Both passes read the rows in `threads` chunks, one task for each.
    fn build_part_plan(&self, parts: usize, threads: usize) -> PartPlan {
        let blocks = self.cols / D;
        let chunk_rows = self.rows.div_ceil(threads).max(1);
        let chunks = (0..self.rows)
            .step_by(chunk_rows)
            .map(|start| start..min(self.rows, start + chunk_rows))
            .collect::<Vec<_>>();
        let mut work = vec![0u64; blocks];
        for chunk in map_chunks(&chunks, |rows| {
            let mut work = vec![0u64; blocks];
            for row in rows {
                self.for_each_row_block(row, |block, coefficients| work[block] += coefficients as u64);
            }
            work
        }) {
            for (total, chunk) in work.iter_mut().zip(chunk) {
                *total += chunk;
            }
        }
        let total = work.iter().sum::<u64>();
        let mut bounds = vec![0];
        let mut done = 0u64;
        for (block, cost) in work.into_iter().enumerate() {
            done += cost;
            if bounds.len() < parts && done * parts as u64 >= total * bounds.len() as u64 {
                bounds.push(block + 1);
            }
        }
        bounds.push(blocks);
        bounds.dedup();

        let mut part_of = vec![0; blocks];
        for (part, range) in bounds.windows(2).enumerate() {
            part_of[range[0]..range[1]].fill(part);
        }
        let pairs = map_chunks(&chunks, |rows| {
            let mut pairs = Vec::new();
            for row in rows {
                self.push_row_spans(row, |block| part_of[block], &mut pairs);
            }
            pairs
        });
        let mut row_offsets = vec![0; bounds.len()];
        for &(part, _) in pairs.iter().flatten() {
            row_offsets[part + 1] += 1;
        }
        for part in 1..row_offsets.len() {
            row_offsets[part] += row_offsets[part - 1];
        }
        let mut next = row_offsets.clone();
        let mut rows = vec![RowSpan::new(0); row_offsets[bounds.len() - 1]];
        for (part, span) in pairs.into_iter().flatten() {
            rows[next[part]] = span;
            next[part] += 1;
        }
        PartPlan {
            bounds,
            row_offsets,
            rows,
        }
    }

    /// Pushes `(part, span)` for each part that `row` reaches, in increasing
    /// part order.
    fn push_row_spans(&self, row: usize, part_of: impl Fn(usize) -> usize, out: &mut Vec<(usize, RowSpan)>) {
        let first = out.len();
        if self.identity {
            span(out, first, row, part_of(row / D));
        }
        let entries = self.row_offsets.range(row);
        for (index, &compact) in self.row_blocks_for(row).iter().enumerate() {
            let part = part_of(self.row_block_index(compact));
            include(&mut span(out, first, row, part).compact, entries.start + index);
        }
        let runs = self.geometric_row_offsets.range(row);
        for (index, &run) in self.geometric_runs_for(row).iter().enumerate() {
            let (start, len, _, _) = decode(run);
            if len == 0 {
                continue;
            }
            for part in part_of(start / D)..=part_of((start + len - 1) / D) {
                include(&mut span(out, first, row, part).runs, runs.start + index);
            }
        }
        let spans = &mut out[first..];
        for (_, span) in spans.iter_mut() {
            close(&mut span.compact);
            close(&mut span.runs);
        }
        if !spans.is_sorted_by_key(|&(part, _)| part) {
            spans.sort_unstable_by_key(|&(part, _)| part);
        }
    }

    fn for_each_row_block(&self, row: usize, mut visit: impl FnMut(usize, usize)) {
        if self.identity {
            visit(row / D, 1);
        }
        for &compact in self.row_blocks_for(row) {
            let coefficients = match compact.dense_index() {
                Some(_) => self.dense_coefficient_count(self.dense_pattern_index(compact)),
                None => 1,
            };
            visit(self.row_block_index(compact), coefficients);
        }
        for &run in self.geometric_runs_for(row) {
            let (start, len, _, _) = decode(run);
            let end = start + len;
            for block in start / D..end.div_ceil(D) {
                visit(block, min(end, (block + 1) * D) - max(start, block * D));
            }
        }
    }
}

/// `f` of each chunk of rows, in chunk order.
fn map_chunks<T: Send>(chunks: &[Range<usize>], f: impl Fn(Range<usize>) -> T + Sync + Send) -> Vec<T> {
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let chunks = chunks.par_iter();
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    let chunks = chunks.iter();
    chunks.cloned().map(f).collect()
}
