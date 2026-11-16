//! Scratch parts of one matrix with near-equal work, and the rows that reach
//! each part. The work of a block is the number of scratch coefficients that
//! its entries update.

use core::cmp::{max, min};

use neo_math::D;

use super::{geometric::decode, SuperneoMatrixCache};

// More parts than threads let work stealing absorb the error of the estimate.
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
const PARTS_PER_THREAD: usize = 4;

#[derive(Clone, Debug)]
pub(super) struct PartPlan {
    bounds: Vec<usize>,
    row_offsets: Vec<usize>,
    rows: Vec<u32>,
}

impl PartPlan {
    /// Part `p` owns the blocks `bounds[p]..bounds[p + 1]`.
    pub(super) fn bounds(&self) -> &[usize] {
        &self.bounds
    }

    /// The rows with an entry in part `part`, in increasing order.
    pub(super) fn rows(&self, part: usize) -> &[u32] {
        &self.rows[self.row_offsets[part]..self.row_offsets[part + 1]]
    }
}

impl SuperneoMatrixCache {
    pub(super) fn part_plan(&self) -> &PartPlan {
        self.part_plan.get_or_init(|| {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            let parts = rayon::current_num_threads() * PARTS_PER_THREAD;
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            let parts = 1;
            self.build_part_plan(parts)
        })
    }

    fn build_part_plan(&self, parts: usize) -> PartPlan {
        let blocks = self.cols / D;
        let mut work = vec![0u64; blocks];
        for row in 0..self.rows {
            self.for_each_row_block(row, |block, coefficients| work[block] += coefficients as u64);
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

        let mut pairs = Vec::new();
        let mut reached = Vec::new();
        for row in 0..self.rows {
            self.for_each_row_block(row, |block, _| {
                reached.push(bounds.partition_point(|&bound| bound <= block) - 1)
            });
            reached.sort_unstable();
            reached.dedup();
            let row = u32::try_from(row).expect("SuperNeo row index exceeds u32");
            pairs.extend(reached.drain(..).map(|part| (part, row)));
        }
        let mut row_offsets = vec![0; bounds.len()];
        for &(part, _) in &pairs {
            row_offsets[part + 1] += 1;
        }
        for part in 1..row_offsets.len() {
            row_offsets[part] += row_offsets[part - 1];
        }
        let mut next = row_offsets.clone();
        let mut rows = vec![0; pairs.len()];
        for (part, row) in pairs {
            rows[next[part]] = row;
            next[part] += 1;
        }
        PartPlan {
            bounds,
            row_offsets,
            rows,
        }
    }

    fn for_each_row_block(&self, row: usize, mut visit: impl FnMut(usize, usize)) {
        if self.identity {
            visit(row / D, 1);
        }
        for &compact in self.row_blocks_for(row) {
            let coefficients = if compact.single_parts().is_some() { 1 } else { D };
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
