use super::{
    as_base_field, is_superneo_compatible_shape, CompactRowBlock, DenseBlockStore, RowBlock, RowOffsetStore, Rq,
    SuperneoEvalCache, SuperneoMatrixCache,
};
use neo_ccs::{CcsMatrix, CcsStructure, GeometricRowRun};
use neo_math::{D, F, K};
use p3_field::{Field, PrimeCharacteristicRing};

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

fn build_matrix_cache<Ff>(mat: &CcsMatrix<Ff>) -> SuperneoMatrixCache
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    let rows = mat.rows();
    let cols = mat.cols().div_ceil(D) * D;
    let cache = match mat {
        CcsMatrix::Identity { .. } => SuperneoMatrixCache {
            rows,
            cols,
            row_offsets: RowOffsetStore::Empty,
            row_blocks: Vec::new(),
            dense_orig: DenseBlockStore::Building(Vec::new()),
            identity: true,
            seeded_phi81_blocks: Vec::new(),
        },
        CcsMatrix::Csc(csc) => build_csc_cache(csc, Vec::new()),
        CcsMatrix::CscWithSeededPhi81 {
            csc,
            blocks,
            geometric_runs,
        } => {
            let mut cache = build_csc_cache(csc, blocks.clone());
            merge_geometric_runs(&mut cache, geometric_runs);
            cache
        }
    };
    let mut cache = cache;
    cache.compact_dense_blocks();
    cache
}

fn build_csc_cache<Ff>(
    csc: &neo_ccs::CscMat<Ff>,
    seeded_phi81_blocks: Vec<neo_ccs::SeededPhi81LinearBlock>,
) -> SuperneoMatrixCache
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    let rows = csc.nrows;
    let cols = csc.ncols.div_ceil(D) * D;
    let mut counts = vec![0u32; rows];
    let mut last_blk_by_row = vec![u32::MAX; rows];
    for column in 0..csc.ncols {
        let block = (column / D) as u32;
        for index in csc.column_range(column) {
            let row = csc.row_index(index);
            if as_base_field(csc.vals[index]) != F::ZERO && last_blk_by_row[row] != block {
                counts[row] = counts[row]
                    .checked_add(1)
                    .expect("SuperNeo row block count exceeds u32");
                last_blk_by_row[row] = block;
            }
        }
    }

    let mut row_offsets = Vec::with_capacity(rows + 1);
    row_offsets.push(0u32);
    for count in counts {
        row_offsets.push(
            row_offsets
                .last()
                .copied()
                .expect("SuperNeo row offset")
                .checked_add(count)
                .expect("SuperNeo matrix block count exceeds u32"),
        );
    }
    let total_blocks = row_offsets.last().copied().unwrap_or(0) as usize;
    let mut row_blocks = vec![CompactRowBlock::default(); total_blocks];
    let mut dense_orig = Vec::new();
    let mut cursor = row_offsets[..rows].to_vec();
    last_blk_by_row.fill(u32::MAX);

    for column in 0..csc.ncols {
        let block = column / D;
        let local = column % D;
        for index in csc.column_range(column) {
            let row = csc.row_index(index);
            let value = as_base_field(csc.vals[index]);
            if value == F::ZERO {
                continue;
            }
            if last_blk_by_row[row] == block as u32 {
                let position = cursor[row] as usize - 1;
                add_compact_coefficient(&mut row_blocks[position], local, value, &mut dense_orig);
            } else {
                let position = cursor[row] as usize;
                cursor[row] += 1;
                last_blk_by_row[row] = block as u32;
                row_blocks[position] = compact_single_or_dense(block, local, value, &mut dense_orig);
            }
        }
    }

    SuperneoMatrixCache {
        rows,
        cols,
        row_offsets: RowOffsetStore::U32(row_offsets),
        row_blocks,
        dense_orig: DenseBlockStore::Building(dense_orig),
        identity: false,
        seeded_phi81_blocks,
    }
}

fn compact_single_or_dense(block: usize, local: usize, coefficient: F, dense_orig: &mut Vec<Rq>) -> CompactRowBlock {
    if coefficient == F::ONE || coefficient == F::ZERO - F::ONE {
        CompactRowBlock::single(block, local, coefficient)
    } else {
        let mut coefficients = [F::ZERO; D];
        coefficients[local] = coefficient;
        let index = dense_orig.len();
        dense_orig.push(Rq(coefficients));
        CompactRowBlock::dense(block, index)
    }
}

fn add_compact_coefficient(block: &mut CompactRowBlock, local: usize, value: F, dense_orig: &mut Vec<Rq>) {
    if let Some((old_local, old_value)) = block.single_parts() {
        let mut coefficients = [F::ZERO; D];
        coefficients[old_local] = old_value;
        coefficients[local] += value;
        let index = dense_orig.len();
        dense_orig.push(Rq(coefficients));
        *block = CompactRowBlock::dense(block.block(), index);
    } else {
        dense_orig[block.dense_index().expect("dense compact block")].0[local] += value;
    }
}

fn push_orig_block(row_blocks: &mut Vec<CompactRowBlock>, dense_orig: &mut Vec<Rq>, block: usize, orig: Rq) {
    let mut nonzero = orig
        .0
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, coefficient)| *coefficient != F::ZERO);
    let Some((local, coefficient)) = nonzero.next() else {
        return;
    };
    if nonzero.next().is_none() {
        row_blocks.push(compact_single_or_dense(block, local, coefficient, dense_orig));
    } else {
        let index = dense_orig.len();
        dense_orig.push(orig);
        row_blocks.push(CompactRowBlock::dense(block, index));
    }
}

fn merge_geometric_runs<Ff>(cache: &mut SuperneoMatrixCache, runs: &[GeometricRowRun<Ff>])
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    if runs.is_empty() {
        return;
    }

    let old_offsets = cache.row_offsets.take_dense(cache.rows + 1);
    let old_blocks = core::mem::take(&mut cache.row_blocks);
    let mut dense_orig = cache.dense_orig.take_building();
    let mut row_offsets = Vec::with_capacity(cache.rows + 1);
    let mut row_blocks = Vec::with_capacity(old_blocks.len() + 2 * runs.len());
    let mut run_cursor = 0usize;
    row_offsets.push(0u32);
    for row in 0..cache.rows {
        let existing = &old_blocks[old_offsets[row] as usize..old_offsets[row + 1] as usize];
        let run_start = run_cursor;
        while run_cursor < runs.len() && runs[run_cursor].row() == row {
            run_cursor += 1;
        }
        if run_start == run_cursor {
            row_blocks.extend_from_slice(existing);
            row_offsets.push(u32::try_from(row_blocks.len()).expect("SuperNeo matrix block count exceeds u32"));
            continue;
        }

        let mut merged = existing
            .iter()
            .copied()
            .map(|block| {
                let orig = if let Some((local, coefficient)) = block.single_parts() {
                    let mut coefficients = [F::ZERO; D];
                    coefficients[local] = coefficient;
                    Rq(coefficients)
                } else {
                    dense_orig[block.dense_index().expect("dense compact block")]
                };
                RowBlock {
                    blk: block.block(),
                    bar: Rq([F::ZERO; D]),
                    orig,
                }
            })
            .collect::<Vec<_>>();
        for run in &runs[run_start..run_cursor] {
            run.for_each_term(|_, column, value| {
                let block_index = column / D;
                let local = column % D;
                let position = match merged.binary_search_by_key(&block_index, |block| block.blk) {
                    Ok(position) => position,
                    Err(position) => {
                        merged.insert(
                            position,
                            RowBlock {
                                blk: block_index,
                                orig: Rq([F::ZERO; D]),
                                bar: Rq([F::ZERO; D]),
                            },
                        );
                        position
                    }
                };
                merged[position].orig.0[local] += as_base_field(value);
            });
        }
        merged.retain(|block| block.orig.0.iter().any(|&value| value != F::ZERO));
        for block in merged {
            push_orig_block(&mut row_blocks, &mut dense_orig, block.blk, block.orig);
        }
        row_offsets.push(u32::try_from(row_blocks.len()).expect("SuperNeo matrix block count exceeds u32"));
    }
    debug_assert_eq!(run_cursor, runs.len(), "geometric runs must be row-sorted and in range");
    cache.row_offsets = RowOffsetStore::U32(row_offsets);
    cache.row_blocks = row_blocks;
    cache.dense_orig.replace_building(dense_orig);
}

pub fn build_superneo_eval_cache<Ff>(structure: &CcsStructure<Ff>) -> Option<SuperneoEvalCache>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    if !is_superneo_compatible_shape(structure.m) {
        return None;
    }
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let mut mats: Vec<SuperneoMatrixCache> = structure
        .matrices
        .par_iter()
        .map(build_matrix_cache)
        .collect();
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    let mut mats: Vec<SuperneoMatrixCache> = structure.matrices.iter().map(build_matrix_cache).collect();

    for matrix in &mut mats {
        matrix.compact_row_offsets();
    }

    #[cfg(feature = "perf-timers")]
    {
        let explicit = mats
            .iter()
            .filter(|matrix| !matrix.identity)
            .flat_map(|matrix| &matrix.row_blocks);
        let (blocks, single, plus_one, minus_one) = explicit.fold(
            (0usize, 0usize, 0usize, 0usize),
            |(blocks, single, plus_one, minus_one), block| {
                let coefficient = block.single_parts().map(|(_, coefficient)| coefficient);
                (
                    blocks + 1,
                    single + usize::from(coefficient.is_some()),
                    plus_one + usize::from(coefficient == Some(F::ONE)),
                    minus_one + usize::from(coefficient == Some(-F::ONE)),
                )
            },
        );
        let compact_bytes = mats
            .iter()
            .map(|matrix| {
                matrix.row_offsets.compact_bytes()
                    + matrix.row_blocks.len() * core::mem::size_of::<CompactRowBlock>()
                    + matrix.dense_orig.compact_bytes()
            })
            .sum::<usize>();
        eprintln!(
            "SuperneoEvalCache::build: row blocks={blocks} single={single} ({:.1}%) +/-1={} ({:.1}% of singles) compact={:.2}GiB",
            if blocks == 0 {
                0.0
            } else {
                single as f64 * 100.0 / blocks as f64
            },
            plus_one + minus_one,
            if single == 0 {
                0.0
            } else {
                (plus_one + minus_one) as f64 * 100.0 / single as f64
            },
            compact_bytes as f64 / (1u64 << 30) as f64,
        );
    }

    let explicit_matrix_masks = if mats.len() <= u16::BITS as usize {
        let rows = mats.first().map_or(0, |matrix| matrix.rows);
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let masks = (0..rows)
            .into_par_iter()
            .map(|row| {
                mats.iter().enumerate().fold(0u16, |mask, (matrix, cache)| {
                    if !cache.identity && cache.row_offsets.get(row) != cache.row_offsets.get(row + 1) {
                        mask | (1u16 << matrix)
                    } else {
                        mask
                    }
                })
            })
            .collect();
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        let masks = (0..rows)
            .map(|row| {
                mats.iter().enumerate().fold(0u16, |mask, (matrix, cache)| {
                    if !cache.identity && cache.row_offsets.get(row) != cache.row_offsets.get(row + 1) {
                        mask | (1u16 << matrix)
                    } else {
                        mask
                    }
                })
            })
            .collect();
        Some(masks)
    } else {
        None
    };
    Some(SuperneoEvalCache {
        mats,
        explicit_matrix_masks,
    })
}
