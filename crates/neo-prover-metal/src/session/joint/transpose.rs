//! Transpose compact row-block references without expanding their coefficients.

use neo_math::D;
use neo_reductions::superneo_eval::{SuperneoCompactRowOffsets, SuperneoMatrixCache};

use crate::MetalError;

const DENSE: u32 = 1 << 31;
const BLOCK_MASK: u32 = (1 << 24) - 1;

pub(super) struct BlockTranspose {
    pub offsets: Vec<u32>,
    pub entries: Vec<[u32; 2]>,
    pub local_masks: Vec<u64>,
    pub active: Vec<bool>,
    pub identity: bool,
}

impl BlockTranspose {
    pub fn new(
        matrix: &SuperneoMatrixCache,
        rows: usize,
        columns: usize,
        pattern_base: u32,
    ) -> Result<Self, MetalError> {
        let (actual_rows, actual_columns, identity) = matrix.compact_explicit_shape();
        if actual_rows != rows || actual_columns != columns || u32::try_from(rows).is_err() {
            return Err(MetalError::Shape("opening matrix dimensions differ"));
        }
        let blocks = columns.div_ceil(D);
        let parts = matrix
            .compact_device_parts()
            .ok_or(MetalError::Shape("opening needs compact rows"))?;
        let mut active = vec![false; blocks];
        for block in matrix.compact_seeded_column_blocks() {
            if block >= blocks {
                return Err(MetalError::Shape("seeded opening block is out of range"));
            }
            active[block] = true;
        }
        let mut invalid = false;
        matrix.for_each_compact_geometric_run(|_, _, start, len, _, _| {
            if let Some(end) = start.checked_add(len).filter(|&end| end <= columns) {
                active[start / D..end.div_ceil(D)].fill(true);
            } else {
                invalid = true;
            }
        });
        if invalid {
            return Err(MetalError::Shape("geometric opening range is invalid"));
        }
        if identity {
            active[..rows.min(columns).div_ceil(D)].fill(true);
            return Ok(Self {
                offsets: vec![0; blocks + 1],
                entries: Vec::new(),
                local_masks: vec![0; blocks],
                active,
                identity,
            });
        }
        let pattern_masks: Vec<_> = parts
            .dense_offsets
            .windows(2)
            .map(|range| {
                parts.dense_locals[range[0] as usize..range[1] as usize]
                    .iter()
                    .fold(0u64, |mask, &local| mask | (1u64 << local))
            })
            .collect();
        let decode = |reference: u32| -> Result<(usize, u32, u64), MetalError> {
            if reference & DENSE == 0 {
                let block = (reference & BLOCK_MASK) as usize;
                let local = (reference >> 24) & 0x3f;
                if block >= blocks || local >= D as u32 {
                    return Err(MetalError::Shape("opening row-block coordinate is invalid"));
                }
                Ok((block, reference, 1u64 << local))
            } else {
                let [block, pattern] = parts.dense_row_blocks[(reference & !DENSE) as usize];
                let global = pattern_base
                    .checked_add(pattern)
                    .filter(|&index| index < DENSE)
                    .ok_or(MetalError::Shape("opening pattern index exceeds u31"))?;
                if block as usize >= blocks {
                    return Err(MetalError::Shape("opening dense block is out of range"));
                }
                Ok((block as usize, DENSE | global, pattern_masks[pattern as usize]))
            }
        };
        let mut offsets = vec![0u32; blocks + 1];
        let mut local_masks = vec![0u64; blocks];
        for &reference in parts.row_blocks {
            let (block, _, mask) = decode(reference)?;
            active[block] = true;
            local_masks[block] |= mask;
            offsets[block + 1] = offsets[block + 1]
                .checked_add(1)
                .ok_or(MetalError::Shape("opening block entry count exceeds u32"))?;
        }
        for block in 0..blocks {
            offsets[block + 1] = offsets[block + 1]
                .checked_add(offsets[block])
                .ok_or(MetalError::Shape("opening matrix entry count exceeds u32"))?;
        }
        let mut next = offsets[..blocks].to_vec();
        let mut entries = vec![[0u32; 2]; parts.row_blocks.len()];
        for row in 0..rows {
            let range = row_offset(parts.row_offsets, row)..row_offset(parts.row_offsets, row + 1);
            for &reference in &parts.row_blocks[range] {
                let (block, pattern, _) = decode(reference)?;
                entries[next[block] as usize] = [row as u32, pattern];
                next[block] += 1;
            }
        }
        if next != offsets[1..] {
            return Err(MetalError::Shape("opening transpose coverage differs"));
        }
        Ok(Self {
            offsets,
            entries,
            local_masks,
            active,
            identity,
        })
    }
}

fn row_offset(offsets: SuperneoCompactRowOffsets<'_>, row: usize) -> usize {
    match offsets {
        SuperneoCompactRowOffsets::Empty => 0,
        SuperneoCompactRowOffsets::U16Chunked {
            chunk_offsets,
            local_offsets,
            chunk_rows,
        } => (chunk_offsets[row / chunk_rows] + u32::from(local_offsets[row])) as usize,
        SuperneoCompactRowOffsets::U24(values) => {
            let start = 3 * row;
            usize::from(values[start]) | (usize::from(values[start + 1]) << 8) | (usize::from(values[start + 2]) << 16)
        }
        SuperneoCompactRowOffsets::U32(values) => values[row] as usize,
    }
}
