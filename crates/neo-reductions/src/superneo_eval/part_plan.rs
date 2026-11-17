use core::cmp::{max, min};
use core::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use neo_math::D;

use super::{geometric::decode, row_block::CompactRowBlock, SuperneoMatrixCache};

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
const PARTS_PER_THREAD: usize = 4;

#[derive(Clone, Debug)]
pub(super) struct PartPlan {
    bounds: Vec<usize>,
    run_offsets: Vec<usize>,
    row_runs: Vec<[u32; 2]>,
    entries_in_block_order: bool,
    runs_in_column_order: bool,
}

impl PartPlan {
    pub(super) fn bounds(&self) -> &[usize] {
        &self.bounds
    }

    pub(super) fn rows(&self, part: usize) -> impl Iterator<Item = usize> + Clone + '_ {
        self.row_runs[self.run_offsets[part]..self.run_offsets[part + 1]]
            .iter()
            .flat_map(|&[start, end]| start as usize..end as usize)
    }
}

impl SuperneoMatrixCache {
    pub(super) fn part_plan(&self) -> &PartPlan {
        if let Some(plan) = self.part_plan.get() {
            return plan;
        }
        // A rayon thread that waits inside `get_or_init` can run a task that
        // reads this cell, so the build runs outside it.
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let threads = rayon::current_num_threads();
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let parts = threads * PARTS_PER_THREAD;
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        let (threads, parts) = (1, 1);
        let plan = self.build_part_plan(parts, threads);
        self.part_plan.get_or_init(|| plan)
    }

    pub(super) fn entry_range(&self, row: usize) -> Range<usize> {
        if self.identity {
            return 0..0;
        }
        self.row_offsets.range(row)
    }

    pub(super) fn run_range(&self, row: usize) -> Range<usize> {
        if self.identity {
            return 0..0;
        }
        self.geometric_row_offsets.range(row)
    }

    pub(super) fn part_entries(&self, plan: &PartPlan, part: usize, row: usize) -> Range<usize> {
        let entries = self.entry_range(row);
        if !plan.entries_in_block_order {
            return entries;
        }
        let (start, end) = (plan.bounds[part], plan.bounds[part + 1]);
        let below = |entry: &CompactRowBlock, bound: usize| self.row_block_index(*entry) < bound;
        let base = entries.start;
        narrow(
            base,
            &self.row_blocks[entries],
            |entry| below(entry, start),
            |entry| below(entry, end),
        )
    }

    pub(super) fn part_runs(&self, plan: &PartPlan, part: usize, row: usize) -> Range<usize> {
        let runs = self.run_range(row);
        if !plan.runs_in_column_order {
            return runs;
        }
        let (start, end) = (plan.bounds[part] * D, plan.bounds[part + 1] * D);
        let base = runs.start;
        narrow(
            base,
            &self.geometric_runs[runs],
            |run| run_columns(run).end <= start,
            |run| run_columns(run).start < end,
        )
    }

    fn build_part_plan(&self, parts: usize, threads: usize) -> PartPlan {
        let blocks = self.cols / D;
        let chunk_rows = self.rows.div_ceil(threads).max(1);
        let chunks = (0..self.rows)
            .step_by(chunk_rows)
            .map(|start| start..min(self.rows, start + chunk_rows))
            .collect::<Vec<_>>();
        let work = (0..blocks).map(|_| AtomicU64::new(0)).collect::<Vec<_>>();
        let [entries_in_block_order, runs_in_column_order] = map_chunks(&chunks, |rows| {
            let mut in_order = [true; 2];
            for row in rows {
                self.for_each_row_block(row, |block, coefficients| {
                    work[block].fetch_add(coefficients as u64, Relaxed);
                });
                in_order[0] &= self
                    .row_blocks_for(row)
                    .is_sorted_by_key(|&entry| self.row_block_index(entry));
                in_order[1] &= self
                    .geometric_runs_for(row)
                    .windows(2)
                    .all(|pair| run_columns(&pair[0]).end <= run_columns(&pair[1]).start);
            }
            in_order
        })
        .into_iter()
        .fold([true; 2], |all, chunk| [all[0] && chunk[0], all[1] && chunk[1]]);
        let total = work.iter().map(|cost| cost.load(Relaxed)).sum::<u64>();
        let mut bounds = vec![0];
        let mut done = 0u64;
        for (block, cost) in work.iter().enumerate() {
            done += cost.load(Relaxed);
            if bounds.len() < parts && done * parts as u64 >= total * bounds.len() as u64 {
                bounds.push(block + 1);
            }
        }
        drop(work);
        bounds.push(blocks);
        bounds.dedup();

        let part_count = bounds.len() - 1;
        let chunk_runs = map_chunks(&chunks, |rows| {
            let mut runs = vec![Vec::<[u32; 2]>::new(); part_count];
            let mut reached = Vec::new();
            for row in rows {
                self.row_parts(row, &bounds, &mut reached);
                let row = u32::try_from(row).expect("SuperNeo row index exceeds u32");
                for &part in &reached {
                    match runs[part].last_mut() {
                        Some(run) if run[1] == row => run[1] = row + 1,
                        _ => runs[part].push([row, row + 1]),
                    }
                }
            }
            runs
        });
        let mut run_offsets = Vec::with_capacity(part_count + 1);
        run_offsets.push(0);
        let mut row_runs = Vec::<[u32; 2]>::with_capacity(chunk_runs.iter().flatten().map(Vec::len).sum());
        for part in 0..part_count {
            let first = row_runs.len();
            for &run in chunk_runs.iter().flat_map(|runs| &runs[part]) {
                match row_runs[first..].last_mut() {
                    Some(last) if last[1] == run[0] => last[1] = run[1],
                    _ => row_runs.push(run),
                }
            }
            run_offsets.push(row_runs.len());
        }
        PartPlan {
            bounds,
            run_offsets,
            row_runs,
            entries_in_block_order,
            runs_in_column_order,
        }
    }

    fn row_parts(&self, row: usize, bounds: &[usize], parts: &mut Vec<usize>) {
        let part_of = |block: usize| bounds.partition_point(|&start| start <= block) - 1;
        parts.clear();
        if self.identity {
            parts.push(part_of(row / D));
        }
        for &entry in self.row_blocks_for(row) {
            push_part(parts, part_of(self.row_block_index(entry)));
        }
        for &run in self.geometric_runs_for(row) {
            let (start, len, _, _) = decode(run);
            if len != 0 {
                for part in part_of(start / D)..=part_of((start + len - 1) / D) {
                    push_part(parts, part);
                }
            }
        }
        if !parts.is_sorted() {
            parts.sort_unstable();
            parts.dedup();
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

fn narrow<T>(base: usize, items: &[T], before: impl Fn(&T) -> bool, inside: impl Fn(&T) -> bool) -> Range<usize> {
    let first = match items.first() {
        Some(item) if before(item) => items.partition_point(&before),
        _ => 0,
    };
    let end = match items.last() {
        Some(item) if !inside(item) => first + items[first..].partition_point(&inside),
        _ => items.len(),
    };
    base + first..base + end
}

fn run_columns(run: &[u64; 3]) -> Range<usize> {
    let start = run[0] as u32 as usize;
    start..start + (run[0] >> 32) as usize
}

fn push_part(parts: &mut Vec<usize>, part: usize) {
    if parts.last() != Some(&part) {
        parts.push(part);
    }
}

fn map_chunks<T: Send>(chunks: &[Range<usize>], f: impl Fn(Range<usize>) -> T + Sync + Send) -> Vec<T> {
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let chunks = chunks.par_iter();
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    let chunks = chunks.iter();
    chunks.cloned().map(f).collect()
}
