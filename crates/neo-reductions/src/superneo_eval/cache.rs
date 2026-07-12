use super::{as_base_field, is_superneo_compatible_shape, RowBlock, Rq, SuperneoEvalCache, SuperneoMatrixCache};
use neo_ccs::{CcsMatrix, CcsStructure, GeometricRowRun};
use neo_math::{superneo_bar_block, D, F, K};
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
    match mat {
        CcsMatrix::Identity { .. } => {
            let mut basis_bar = [Rq([F::ZERO; D]); D];
            let mut basis_orig = [Rq([F::ZERO; D]); D];
            for (local, out) in basis_bar.iter_mut().enumerate() {
                let mut e = [F::ZERO; D];
                e[local] = F::ONE;
                basis_orig[local] = Rq(e);
                *out = Rq(superneo_bar_block(e));
            }
            let mut row_offsets = Vec::with_capacity(rows + 1);
            let mut row_blocks = Vec::with_capacity(rows);
            row_offsets.push(0);
            for row in 0..rows {
                let local = row % D;
                row_blocks.push(RowBlock {
                    blk: row / D,
                    bar: basis_bar[local],
                    orig: basis_orig[local],
                });
                row_offsets.push(row_blocks.len());
            }
            SuperneoMatrixCache {
                rows,
                cols,
                row_offsets,
                row_blocks,
                identity: true,
                seeded_phi81_blocks: Vec::new(),
            }
        }
        CcsMatrix::Csc(csc) => {
            let mut counts = vec![0usize; rows];
            let mut last_blk_by_row = vec![usize::MAX; rows];
            for c in 0..csc.ncols {
                let blk = c / D;
                for k in csc.col_ptr[c]..csc.col_ptr[c + 1] {
                    let row = csc.row_idx[k];
                    if as_base_field(csc.vals[k]) != F::ZERO && last_blk_by_row[row] != blk {
                        counts[row] += 1;
                        last_blk_by_row[row] = blk;
                    }
                }
            }

            let total_blocks: usize = counts.iter().sum();
            let mut row_offsets = Vec::with_capacity(rows + 1);
            row_offsets.push(0);
            for count in counts {
                row_offsets.push(row_offsets.last().copied().unwrap() + count);
            }

            let empty_block = RowBlock {
                blk: 0,
                bar: Rq([F::ZERO; D]),
                orig: Rq([F::ZERO; D]),
            };
            let mut row_blocks = vec![empty_block; total_blocks];
            let mut cursor = row_offsets[..rows].to_vec();
            last_blk_by_row.fill(usize::MAX);

            for c in 0..csc.ncols {
                let blk = c / D;
                let local = c % D;
                for k in csc.col_ptr[c]..csc.col_ptr[c + 1] {
                    let row = csc.row_idx[k];
                    let value = as_base_field(csc.vals[k]);
                    if value == F::ZERO {
                        continue;
                    }
                    if last_blk_by_row[row] == blk {
                        row_blocks[cursor[row] - 1].orig.0[local] += value;
                    } else {
                        let idx = cursor[row];
                        cursor[row] += 1;
                        last_blk_by_row[row] = blk;
                        let mut block = [F::ZERO; D];
                        block[local] = value;
                        row_blocks[idx] = RowBlock {
                            blk,
                            bar: Rq([F::ZERO; D]),
                            orig: Rq(block),
                        };
                    }
                }
            }

            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            row_blocks
                .par_iter_mut()
                .for_each(|block| block.bar.0 = superneo_bar_block(block.orig.0));
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            for block in &mut row_blocks {
                block.bar.0 = superneo_bar_block(block.orig.0);
            }

            SuperneoMatrixCache {
                rows,
                cols,
                row_offsets,
                row_blocks,
                identity: false,
                seeded_phi81_blocks: Vec::new(),
            }
        }
        CcsMatrix::CscWithSeededPhi81 {
            csc,
            blocks,
            geometric_runs,
        } => {
            let mut cache = build_matrix_cache(&CcsMatrix::Csc(csc.clone()));
            merge_geometric_runs(&mut cache, geometric_runs);
            cache.seeded_phi81_blocks = blocks.clone();
            cache
        }
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

    let mut row_offsets = Vec::with_capacity(cache.rows + 1);
    let mut row_blocks = Vec::with_capacity(cache.row_blocks.len() + 2 * runs.len());
    let mut run_cursor = 0usize;
    row_offsets.push(0);
    for row in 0..cache.rows {
        let existing = &cache.row_blocks[cache.row_offsets[row]..cache.row_offsets[row + 1]];
        let run_start = run_cursor;
        while run_cursor < runs.len() && runs[run_cursor].row() == row {
            run_cursor += 1;
        }
        if run_start == run_cursor {
            row_blocks.extend_from_slice(existing);
            row_offsets.push(row_blocks.len());
            continue;
        }

        let mut merged = existing.to_vec();
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
                                bar: Rq([F::ZERO; D]),
                                orig: Rq([F::ZERO; D]),
                            },
                        );
                        position
                    }
                };
                merged[position].orig.0[local] += as_base_field(value);
            });
        }
        merged.retain(|block| block.orig.0.iter().any(|&value| value != F::ZERO));
        for block in &mut merged {
            block.bar.0 = superneo_bar_block(block.orig.0);
        }
        row_blocks.extend(merged);
        row_offsets.push(row_blocks.len());
    }
    debug_assert_eq!(run_cursor, runs.len(), "geometric runs must be row-sorted and in range");
    cache.row_offsets = row_offsets;
    cache.row_blocks = row_blocks;
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
    let mats: Vec<SuperneoMatrixCache> = structure
        .matrices
        .par_iter()
        .map(build_matrix_cache)
        .collect();
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    let mats: Vec<SuperneoMatrixCache> = structure.matrices.iter().map(build_matrix_cache).collect();

    #[cfg(feature = "perf-timers")]
    {
        let explicit = mats
            .iter()
            .filter(|matrix| !matrix.identity)
            .flat_map(|matrix| &matrix.row_blocks);
        let (blocks, single) = explicit.fold((0usize, 0usize), |(blocks, single), block| {
            let nonzero = block
                .orig
                .0
                .iter()
                .filter(|&&coefficient| coefficient != F::ZERO)
                .count();
            (blocks + 1, single + usize::from(nonzero == 1))
        });
        eprintln!(
            "SuperneoEvalCache::build: explicit row blocks={blocks} single={single} ({:.1}%)",
            if blocks == 0 {
                0.0
            } else {
                single as f64 * 100.0 / blocks as f64
            }
        );
    }

    let explicit_matrix_masks = if mats.len() <= u16::BITS as usize {
        let rows = mats.first().map_or(0, |matrix| matrix.rows);
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let masks = (0..rows)
            .into_par_iter()
            .map(|row| {
                mats.iter().enumerate().fold(0u16, |mask, (matrix, cache)| {
                    if cache.row_offsets[row] != cache.row_offsets[row + 1] {
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
                    if cache.row_offsets[row] != cache.row_offsets[row + 1] {
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
