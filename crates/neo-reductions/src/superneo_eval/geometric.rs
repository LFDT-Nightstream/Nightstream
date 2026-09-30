//! Exact evaluation of compact geometric matrix-row runs.

use super::scratch::ScratchPart;
use super::*;

#[inline]
fn decode(run: [u64; 3]) -> (usize, usize, F, F) {
    (
        run[0] as u32 as usize,
        (run[0] >> 32) as u32 as usize,
        F::from_u64(run[1]),
        F::from_u64(run[2]),
    )
}

impl SuperneoMatrixCache {
    #[inline]
    pub(super) fn geometric_runs_for(&self, row: usize) -> &[[u64; 3]] {
        if self.identity || row >= self.rows {
            return &[];
        }
        &self.geometric_runs[self.geometric_row_offsets.range(row)]
    }

    pub(super) fn append_geometric_row_blocks(&self, row: usize, out: &mut Vec<RowBlock>) {
        for &run in self.geometric_runs_for(row) {
            let (column_start, len, mut coefficient, ratio) = decode(run);
            let column_end = column_start + len;
            let mut column = column_start;
            while column < column_end {
                let block = column / D;
                let block_end = core::cmp::min(column_end, (block + 1) * D);
                let mut orig = [F::ZERO; D];
                while column < block_end {
                    orig[column % D] += coefficient;
                    coefficient *= ratio;
                    column += 1;
                }
                if orig.iter().any(|&value| value != F::ZERO) {
                    out.push(RowBlock {
                        blk: block,
                        bar: Rq(neo_math::superneo_bar_block(orig)),
                        orig: Rq(orig),
                    });
                }
            }
        }
    }

    #[inline]
    pub(super) fn geometric_dot_real(&self, row: usize, input: &SuperneoZBlocks) -> F {
        let mut out = F::ZERO;
        for &run in self.geometric_runs_for(row) {
            let (column_start, len, mut coefficient, ratio) = decode(run);
            for column in column_start..column_start + len {
                let block = column / D;
                if input.real_nonzero(block) {
                    out += coefficient * input.real_coefficient(block, column % D);
                }
                coefficient *= ratio;
            }
        }
        out
    }

    #[inline]
    pub(super) fn geometric_weighted_projection(&self, row: usize, projection: &[K]) -> K {
        let mut out = K::ZERO;
        for &run in self.geometric_runs_for(row) {
            let (column_start, len, mut coefficient, ratio) = decode(run);
            for value in &projection[column_start..column_start + len] {
                if coefficient == F::ONE {
                    out += *value;
                } else if coefficient != F::ZERO {
                    out += value.scale_base(coefficient);
                }
                coefficient *= ratio;
            }
        }
        out
    }

    /// The ratio whose runs are added as events: the first run's, if any.
    pub(super) fn event_ratio(&self) -> Option<F> {
        self.geometric_runs.first().map(|&run| decode(run).3)
    }

    /// Record this row's runs with `event_ratio` as a start event and, inside
    /// the part, an end event `-start * ratio^len`, and touch every covered
    /// block. A run that begins before the part starts at the part's first
    /// column with its advanced term.
    pub(super) fn add_geometric_events(
        &self,
        row: usize,
        weight_re: F,
        weight_im: F,
        event_ratio: F,
        part: &mut ScratchPart<'_>,
    ) {
        let blocks = part.blocks();
        let (part_start, part_end) = (blocks.start * D, blocks.end * D);
        let mut end_power = (0, F::ONE);
        for &run in self.geometric_runs_for(row) {
            let (column_start, len, coefficient, ratio) = decode(run);
            let column_end = column_start + len;
            let first = core::cmp::max(column_start, part_start);
            if ratio != event_ratio || first >= core::cmp::min(column_end, part_end) {
                continue;
            }
            let start = [weight_re * coefficient, weight_im * coefficient];
            let skip = ratio.exp_u64((first - column_start) as u64);
            part.add_event(first, [start[0] * skip, start[1] * skip]);
            for block in first / D + 1..=(core::cmp::min(column_end, part_end) - 1) / D {
                part.touch(block);
            }
            if column_end < part_end {
                if end_power.0 != len {
                    end_power = (len, ratio.exp_u64(len as u64));
                }
                part.add_event(column_end, [-start[0] * end_power.1, -start[1] * end_power.1]);
            }
        }
    }

    /// Add this row's runs directly, except those with `event_ratio`.
    pub(super) fn accumulate_geometric_ring_form_row(
        &self,
        row: usize,
        weight_re: F,
        weight_im: F,
        event_ratio: Option<F>,
        part: &mut ScratchPart<'_>,
    ) {
        let blocks = part.blocks();
        let (part_start, part_end) = (blocks.start * D, blocks.end * D);
        for &run in self.geometric_runs_for(row) {
            let (column_start, len, coefficient, ratio) = decode(run);
            if Some(ratio) == event_ratio {
                continue;
            }
            let column_end = core::cmp::min(column_start + len, part_end);
            let mut column = core::cmp::max(column_start, part_start);
            if column >= column_end {
                continue;
            }
            let skip = ratio.exp_u64((column - column_start) as u64);
            let mut term = [weight_re * coefficient * skip, weight_im * coefficient * skip];
            while column < column_end {
                let block = column / D;
                let block_end = core::cmp::min(column_end, (block + 1) * D);
                term = part.add_geometric(block, column % D..block_end - block * D, term, ratio);
                column = block_end;
            }
        }
    }
}
