use std::borrow::Borrow;

use neo_ccs::Mat;
use neo_math::signed_sums::accelerate::{child_batch_workspace_bytes, ChildBatchKernel, MAX_BATCH_PLANES};
use p3_field::PrimeCharacteristicRing;
use rayon::prelude::*;

use super::{four_plane, Commitment, ExpandedKey, Fq, SignedBlock, D, MAX_MESSAGE_COLUMNS, PRODUCTION_VERIFIER_ROWS};

const ROWS: usize = PRODUCTION_VERIFIER_ROWS as usize;
const MAX_WORKSPACE_BYTES: usize = 24 << 20;
const APPLE_PAGE_BYTES: usize = 16 << 10;
type JobOutput = [[Fq; D]; MAX_BATCH_PLANES];

struct BatchLayout {
    kernel: ChildBatchKernel,
    jobs_per_task: usize,
    width: usize,
    workspace_bytes: usize,
}

impl BatchLayout {
    fn new(planes: usize, columns: usize) -> Self {
        let groups = planes.div_ceil(MAX_BATCH_PLANES);
        let width = planes.div_ceil(groups);
        let job_bytes = (0..planes)
            .step_by(width)
            .map(|first| child_batch_workspace_bytes(width.min(planes - first)))
            .sum::<usize>();
        let index_bytes = (columns * size_of::<u32>()).next_multiple_of(APPLE_PAGE_BYTES);
        let output_bytes = (ROWS * groups * size_of::<JobOutput>()).next_multiple_of(APPLE_PAGE_BYTES);
        let kernel = ChildBatchKernel::select();
        Self {
            kernel,
            jobs_per_task: kernel.jobs_per_task(ROWS * groups, width, ROWS * job_bytes),
            width,
            workspace_bytes: ROWS * job_bytes + groups * index_bytes + output_bytes + APPLE_PAGE_BYTES,
        }
    }

    fn fits(&self, blocks: usize, largest_plane: usize, four_plane: &four_plane::TableLayout) -> bool {
        let four_plane_bytes = blocks * size_of::<SignedBlock>() + four_plane.reserved_mask_bytes(largest_plane);
        self.workspace_bytes <= MAX_WORKSPACE_BYTES && self.workspace_bytes <= four_plane_bytes
    }
}

pub(super) fn commit<W: Borrow<Mat<Fq>>>(witnesses: &[W]) -> Option<Vec<Commitment>> {
    if !(3..=four_plane::MAX_PLANES).contains(&witnesses.len()) {
        return None;
    }
    let columns = witnesses[0].borrow().cols();
    if !(1..=MAX_MESSAGE_COLUMNS as usize).contains(&columns) {
        return None;
    }
    let key = ExpandedKey::shared()?;
    if key.blocks() < columns as u64 {
        return None;
    }
    let mut planes = [(&[][..], &[][..]); four_plane::MAX_PLANES];
    let mut slots = [0; four_plane::MAX_PLANES];
    let mut live = 0;
    let mut blocks = 0;
    let mut largest_plane = 0;
    for (slot, witness) in witnesses.iter().enumerate() {
        let witness = witness.borrow();
        if witness.rows() != D || witness.cols() != columns {
            return None;
        }
        if witness.virtual_constant_value() == Some(&Fq::ZERO) {
            continue;
        }
        let (positive, negative) = witness.packed_signed_unit_column_masks()?;
        let count = positive
            .iter()
            .zip(negative)
            .filter(|(p, n)| **p | **n != 0)
            .count();
        if count != 0 {
            planes[live] = (positive, negative);
            slots[live] = slot;
            live += 1;
            blocks += count;
            largest_plane = largest_plane.max(count);
        }
    }

    let four_plane = four_plane::TableLayout::new(live, columns)?;
    let layout = BatchLayout::new(live, columns);
    if !layout.fits(blocks, largest_plane, &four_plane) {
        return None;
    }
    let groups: Vec<_> = planes[..live]
        .chunks(layout.width)
        .map(|planes| {
            let mut indices = Vec::with_capacity(columns);
            indices.extend(
                (0..columns)
                    .filter(|&i| planes.iter().any(|(p, n)| p[i] | n[i] != 0))
                    .map(|i| i as u32),
            );
            (planes, indices)
        })
        .collect();
    let mut jobs = vec![[[Fq::ZERO; D]; MAX_BATCH_PLANES]; ROWS * groups.len()];
    let completed = jobs
        .par_chunks_mut(layout.jobs_per_task)
        .enumerate()
        .all(|(part, jobs)| {
            jobs.iter_mut().enumerate().all(|(index, out)| {
                let task = part * layout.jobs_per_task + index;
                let row = task / groups.len();
                let (planes, indices) = &groups[task % groups.len()];
                let offset = row * key.blocks() as usize * D;
                layout.kernel.try_dot(
                    &key.words()[offset..offset + columns * D],
                    indices,
                    planes,
                    &mut out[..planes.len()],
                )
            })
        });
    if !completed {
        return None;
    }
    let mut commitments = vec![Commitment::zeros(ROWS); witnesses.len()];
    for (row, results) in jobs.chunks_exact(groups.len()).enumerate() {
        for (result, slots) in results.iter().zip(slots[..live].chunks(layout.width)) {
            for (value, &slot) in result.iter().zip(slots) {
                commitments[slot].col_mut(row).copy_from_slice(value);
            }
        }
    }
    Some(commitments)
}

#[cfg(test)]
#[path = "../../tests/unit/child_batch.rs"]
mod tests;
