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

    pub(super) fn accumulate_geometric_ring_form_row(
        &self,
        row: usize,
        weight_re: F,
        weight_im: F,
        part: &mut ScratchPart<'_>,
    ) {
        let blocks = part.blocks();
        let (part_start, part_end) = (blocks.start * D, blocks.end * D);
        for &run in self.geometric_runs_for(row) {
            let (column_start, len, coefficient, ratio) = decode(run);
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
