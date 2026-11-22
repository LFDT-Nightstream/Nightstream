use std::mem::size_of;

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

use super::{
    with_elements, Commitment, ExpandedKey, SignedBlock, SignedShiftSums, MAX_EXACT_TERMS, PRODUCTION_VERIFIER_ROWS,
};

const GROUP: usize = 4;
const PATTERNS: usize = 1 << GROUP;
const MAX_PLANES: usize = 14;
const SEGMENT_COLUMNS: usize = 256;
const SEGMENTS_PER_TASK: usize = 4;
const MASK_TABLE_BYTES: usize = 8 << 20;
const MAX_MASKS_PER_COLUMN: usize = MAX_PLANES / GROUP * (PATTERNS - 1) + (1 << (MAX_PLANES % GROUP)) - 1;
const _: () = assert!(SEGMENT_COLUMNS * MAX_MASKS_PER_COLUMN <= u16::MAX as usize);

const _: () = assert!(
    size_of::<MaskTable>() + SEGMENT_COLUMNS * (size_of::<u16>() + MAX_MASKS_PER_COLUMN * size_of::<Mask>())
        <= MASK_TABLE_BYTES
);

struct Mask {
    bucket: usize,
    positive: u64,
    negative: u64,
}

struct MaskTable<'a> {
    columns: &'a [u64],
    ends: Vec<u16>,
    masks: Vec<Mask>,
}

impl<'a> MaskTable<'a> {
    fn new(columns: &'a [u64], planes: &[(usize, &[SignedBlock])], masks_per_column: usize) -> Self {
        let mut ends = Vec::with_capacity(columns.len());
        let mut masks = Vec::with_capacity(columns.len() * masks_per_column);
        let mut positions = [0; MAX_PLANES];
        for (position, (_, blocks)) in positions.iter_mut().zip(planes) {
            *position = blocks.partition_point(|block| block.index < columns[0]);
        }
        for &column in columns {
            for (group, planes) in planes.chunks(GROUP).enumerate() {
                let mut positive = [0; PATTERNS];
                let mut negative = [0; PATTERNS];
                positive[0] = u64::MAX;
                negative[0] = u64::MAX;
                for (bit, &(_, blocks)) in planes.iter().enumerate() {
                    let position = &mut positions[group * GROUP + bit];
                    let Some(block) = blocks.get(*position).filter(|b| b.index == column) else {
                        continue;
                    };
                    *position += 1;
                    for pattern in 0..1 << bit {
                        let p = positive[pattern] & block.positive;
                        let n = negative[pattern] & block.negative;
                        positive[pattern] ^= p;
                        negative[pattern] ^= n;
                        positive[pattern | (1 << bit)] = p;
                        negative[pattern | (1 << bit)] = n;
                    }
                }
                for pattern in 1..1 << planes.len() {
                    if positive[pattern] | negative[pattern] != 0 {
                        masks.push(Mask {
                            bucket: group * PATTERNS + pattern,
                            positive: positive[pattern],
                            negative: negative[pattern],
                        });
                    }
                }
            }
            ends.push(masks.len() as u16);
        }
        Self { columns, ends, masks }
    }

    fn accumulate(&self, row: u32, key: Option<&ExpandedKey>, buckets: &mut [SignedShiftSums]) {
        let mut ends = self.ends.iter().copied();
        let mut start = 0;
        with_elements(
            row,
            self.columns,
            |&column| column,
            key,
            |_, element| {
                let end = usize::from(ends.next().expect("one mask range per column"));
                for mask in &self.masks[start..end] {
                    buckets[mask.bucket].add_signed_units(element, mask.positive, mask.negative);
                }
                start = end;
            },
        );
    }
}

pub(super) fn commit_if_supported(
    blocks: &[Vec<SignedBlock>],
    columns: &[u64],
    key: Option<&ExpandedKey>,
    commitments: &mut [Commitment],
) -> bool {
    let planes: Vec<_> = blocks
        .iter()
        .map(Vec::as_slice)
        .enumerate()
        .filter(|(_, plane)| !plane.is_empty())
        .take(MAX_PLANES + 1)
        .collect();
    if !(3..=MAX_PLANES).contains(&planes.len()) || columns.len() as u64 > MAX_EXACT_TERMS / (PATTERNS as u64 / 2) {
        return false;
    }
    let masks_per_column: usize = planes
        .chunks(GROUP)
        .map(|group| (1 << group.len()) - 1)
        .sum();
    let segment_bytes =
        size_of::<MaskTable>() + SEGMENT_COLUMNS * (size_of::<u16>() + masks_per_column * size_of::<Mask>());
    let segments_per_batch = MASK_TABLE_BYTES / segment_bytes;
    let rows = PRODUCTION_VERIFIER_ROWS as usize;
    let mut totals = vec![SignedShiftSums::zero(); rows * planes.len()];
    for batch in columns.chunks(segments_per_batch * SEGMENT_COLUMNS) {
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let segments = batch.par_chunks(SEGMENT_COLUMNS);
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        let segments = batch.chunks(SEGMENT_COLUMNS);
        let tables: Vec<_> = segments
            .map(|segment| MaskTable::new(segment, &planes, masks_per_column))
            .collect();
        let parts = tables.len().div_ceil(SEGMENTS_PER_TASK);
        let tasks = 0..rows * parts;
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let tasks = tasks.into_par_iter();
        let partial: Vec<_> = tasks
            .map(|task| {
                let row = (task / parts) as u32;
                let start = task % parts * SEGMENTS_PER_TASK;
                let mut buckets = vec![SignedShiftSums::zero(); PATTERNS * planes.len().div_ceil(GROUP)];
                for table in &tables[start..(start + SEGMENTS_PER_TASK).min(tables.len())] {
                    table.accumulate(row, key, &mut buckets);
                }
                let mut sums = vec![SignedShiftSums::zero(); planes.len()];
                for (group, sums) in sums.chunks_mut(GROUP).enumerate() {
                    for pattern in 1usize..1 << sums.len() {
                        let mut bits = pattern;
                        while bits != 0 {
                            sums[bits.trailing_zeros() as usize].add(&buckets[group * PATTERNS + pattern]);
                            bits &= bits - 1;
                        }
                    }
                }
                sums
            })
            .collect();
        for (row, parts) in totals.chunks_mut(planes.len()).zip(partial.chunks(parts)) {
            for part in parts {
                for (total, sum) in row.iter_mut().zip(part) {
                    total.add(sum);
                }
            }
        }
    }
    for (row, sums) in totals.chunks(planes.len()).enumerate() {
        for (&(plane, _), sum) in planes.iter().zip(sums) {
            commitments[plane]
                .col_mut(row)
                .copy_from_slice(&sum.reduce());
        }
    }
    true
}
