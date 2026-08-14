use super::{
    as_base_field, is_superneo_compatible_shape, CompactRowBlock, DenseBlockStore, DenseRowBlock, RowOffsetStore, Rq,
    SuperneoEvalCache, SuperneoMatrixCache, COMPACT_SINGLE_BLOCK_MASK,
};
use neo_ccs::{CcsMatrix, CcsStructure, GeometricRowRun};
use neo_math::{D, F, K};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};

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
            dense_row_blocks: Vec::new(),
            dense_orig: DenseBlockStore::Building(Vec::new()),
            geometric_row_offsets: RowOffsetStore::Empty,
            geometric_runs: Vec::new(),
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
            retain_geometric_runs(&mut cache, geometric_runs);
            cache
        }
        CcsMatrix::VerifierArtifact { .. } => {
            panic!("cache derivation is unavailable for verifier-artifact matrices")
        }
    };
    let mut cache = cache;
    deduplicate_dense_blocks(&mut cache);
    cache.compact_dense_blocks();
    cache
}

/// Intern equal ring-coefficient patterns before compact storage is emitted.
/// The row block still owns its column-block index; only the 54 coefficients
/// are shared. Fingerprints select comparison buckets, but exact field-array
/// equality decides every merge.
fn deduplicate_dense_blocks(cache: &mut SuperneoMatrixCache) {
    let DenseBlockStore::Building(patterns) = &mut cache.dense_orig else {
        return;
    };
    if patterns.len() < 2 {
        return;
    }

    let original = core::mem::take(patterns);
    let mut keyed = original
        .iter()
        .enumerate()
        .map(|(index, pattern)| {
            let [first, second] = dense_pattern_fingerprint(pattern);
            [first, second, index as u64]
        })
        .collect::<Vec<_>>();
    keyed.sort_unstable();

    let mut representative = vec![u32::MAX; original.len()];
    let mut start = 0usize;
    while start < keyed.len() {
        let mut end = start + 1;
        while end < keyed.len() && keyed[end][..2] == keyed[start][..2] {
            end += 1;
        }
        let mut exact_representatives = Vec::<usize>::new();
        for keyed_pattern in &keyed[start..end] {
            let index = keyed_pattern[2] as usize;
            let exact = exact_representatives
                .iter()
                .copied()
                .find(|&candidate| original[candidate].0 == original[index].0)
                .unwrap_or_else(|| {
                    exact_representatives.push(index);
                    index
                });
            representative[index] = exact as u32;
        }
        start = end;
    }

    let mut representative_to_compact = vec![u32::MAX; original.len()];
    let mut unique = Vec::new();
    for (index, pattern) in original.into_iter().enumerate() {
        if representative[index] as usize == index {
            representative_to_compact[index] = unique.len() as u32;
            unique.push(pattern);
        }
    }
    for block in &mut cache.dense_row_blocks {
        let old_index = block.pattern();
        let exact = representative[old_index] as usize;
        let compact = representative_to_compact[exact] as usize;
        block.set_pattern(compact);
    }
    *patterns = unique;
}

fn dense_pattern_fingerprint(pattern: &Rq) -> [u64; 2] {
    let mut first = 0xcbf2_9ce4_8422_2325u64;
    let mut second = 0x9e37_79b9_7f4a_7c15u64;
    for (index, coefficient) in pattern.0.iter().enumerate() {
        let value = coefficient.as_canonical_u64();
        first ^= value.wrapping_add(index as u64);
        first = first.wrapping_mul(0x0000_0100_0000_01b3);
        second ^= value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        second = second.rotate_left(27).wrapping_mul(0x94d0_49bb_1331_11eb);
    }
    [first, second]
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
    let mut dense_row_blocks = Vec::new();
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
                add_compact_coefficient(
                    &mut row_blocks[position],
                    local,
                    value,
                    &mut dense_row_blocks,
                    &mut dense_orig,
                );
            } else {
                let position = cursor[row] as usize;
                cursor[row] += 1;
                last_blk_by_row[row] = block as u32;
                row_blocks[position] =
                    compact_single_or_dense(block, local, value, &mut dense_row_blocks, &mut dense_orig);
            }
        }
    }

    SuperneoMatrixCache {
        rows,
        cols,
        row_offsets: RowOffsetStore::U32(row_offsets),
        row_blocks,
        dense_row_blocks,
        dense_orig: DenseBlockStore::Building(dense_orig),
        geometric_row_offsets: RowOffsetStore::Empty,
        geometric_runs: Vec::new(),
        identity: false,
        seeded_phi81_blocks,
    }
}

fn compact_single_or_dense(
    block: usize,
    local: usize,
    coefficient: F,
    dense_row_blocks: &mut Vec<DenseRowBlock>,
    dense_orig: &mut Vec<Rq>,
) -> CompactRowBlock {
    if coefficient == F::ONE || coefficient == F::ZERO - F::ONE {
        CompactRowBlock::single(block, local, coefficient)
    } else {
        let mut coefficients = [F::ZERO; D];
        coefficients[local] = coefficient;
        let pattern = dense_orig.len();
        dense_orig.push(Rq(coefficients));
        let index = dense_row_blocks.len();
        dense_row_blocks.push(DenseRowBlock::new(block, pattern));
        CompactRowBlock::dense(index)
    }
}

fn add_compact_coefficient(
    block: &mut CompactRowBlock,
    local: usize,
    value: F,
    dense_row_blocks: &mut Vec<DenseRowBlock>,
    dense_orig: &mut Vec<Rq>,
) {
    if let Some((old_block, old_local, old_value)) = block.single_parts() {
        let mut coefficients = [F::ZERO; D];
        coefficients[old_local] = old_value;
        coefficients[local] += value;
        let pattern = dense_orig.len();
        dense_orig.push(Rq(coefficients));
        let index = dense_row_blocks.len();
        dense_row_blocks.push(DenseRowBlock::new(old_block, pattern));
        *block = CompactRowBlock::dense(index);
    } else {
        let dense = dense_row_blocks[block.dense_index().expect("dense compact block")];
        dense_orig[dense.pattern()].0[local] += value;
    }
}

fn retain_geometric_runs<Ff>(cache: &mut SuperneoMatrixCache, runs: &[GeometricRowRun<Ff>])
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    if runs.is_empty() {
        return;
    }

    let mut offsets = vec![0u32; cache.rows + 1];
    for run in runs {
        offsets[run.row() + 1] = offsets[run.row() + 1]
            .checked_add(1)
            .expect("SuperNeo geometric run count exceeds u32");
    }
    for row in 0..cache.rows {
        offsets[row + 1] = offsets[row + 1]
            .checked_add(offsets[row])
            .expect("SuperNeo geometric run count exceeds u32");
    }
    cache.geometric_row_offsets = RowOffsetStore::from_dense(offsets);
    cache.geometric_runs = runs
        .iter()
        .map(|run| {
            let column_start = u32::try_from(run.column_start()).expect("SuperNeo geometric column exceeds u32");
            let len = u32::try_from(run.len()).expect("SuperNeo geometric length exceeds u32");
            [
                u64::from(column_start) | (u64::from(len) << 32),
                as_base_field(*run.initial()).as_canonical_u64(),
                as_base_field(*run.ratio()).as_canonical_u64(),
            ]
        })
        .collect();
}

pub fn build_superneo_eval_cache<Ff>(structure: &CcsStructure<Ff>) -> Option<SuperneoEvalCache>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    if structure.is_verifier_artifact_header() {
        return None;
    }
    if !is_superneo_compatible_shape(structure.m) {
        return None;
    }
    if structure
        .matrices
        .iter()
        .any(|matrix| matrix.cols().div_ceil(D) > COMPACT_SINGLE_BLOCK_MASK as usize + 1)
    {
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
                let coefficient = block.single_parts().map(|(_, _, coefficient)| coefficient);
                (
                    blocks + 1,
                    single + usize::from(coefficient.is_some()),
                    plus_one + usize::from(coefficient == Some(F::ONE)),
                    minus_one + usize::from(coefficient == Some(-F::ONE)),
                )
            },
        );
        let row_offset_bytes = mats
            .iter()
            .map(|matrix| matrix.row_offsets.compact_bytes())
            .sum::<usize>();
        let row_block_bytes = mats
            .iter()
            .map(|matrix| matrix.row_blocks.len() * core::mem::size_of::<CompactRowBlock>())
            .sum::<usize>();
        let dense_row_block_bytes = mats
            .iter()
            .map(|matrix| matrix.dense_row_blocks.len() * core::mem::size_of::<DenseRowBlock>())
            .sum::<usize>();
        let dense_bytes = mats
            .iter()
            .map(|matrix| matrix.dense_orig.compact_bytes())
            .sum::<usize>();
        let geometric_offset_bytes = mats
            .iter()
            .map(|matrix| matrix.geometric_row_offsets.compact_bytes())
            .sum::<usize>();
        let geometric_run_bytes = mats
            .iter()
            .map(|matrix| matrix.geometric_runs.len() * core::mem::size_of::<[u64; 3]>())
            .sum::<usize>();
        let compact_bytes = row_offset_bytes
            + row_block_bytes
            + dense_row_block_bytes
            + dense_bytes
            + geometric_offset_bytes
            + geometric_run_bytes;
        let geometric_runs = mats
            .iter()
            .map(|matrix| matrix.geometric_runs.len())
            .sum::<usize>();
        let (dense_blocks, signed_dense_blocks, dense_coefficients, signed_dense_coefficients) = mats.iter().fold(
            (0usize, 0usize, 0usize, 0usize),
            |(blocks, signed_blocks, coefficients, signed_coefficients), matrix| {
                let DenseBlockStore::Compact {
                    offsets,
                    coefficients: values,
                    ..
                } = &matrix.dense_orig
                else {
                    return (blocks, signed_blocks, coefficients, signed_coefficients);
                };
                let mut matrix_signed_blocks = 0usize;
                let mut matrix_signed_coefficients = 0usize;
                for dense in 0..offsets.len().saturating_sub(1) {
                    let range = offsets[dense] as usize..offsets[dense + 1] as usize;
                    if values[range.clone()]
                        .iter()
                        .all(|&value| value == F::ONE || value == -F::ONE)
                    {
                        matrix_signed_blocks += 1;
                        matrix_signed_coefficients += range.len();
                    }
                }
                (
                    blocks + offsets.len().saturating_sub(1),
                    signed_blocks + matrix_signed_blocks,
                    coefficients + values.len(),
                    signed_coefficients + matrix_signed_coefficients,
                )
            },
        );
        eprintln!(
            "SuperneoEvalCache::build: row blocks={blocks} geometric_runs={geometric_runs} single={single} ({:.1}%) +/-1={} ({:.1}% of singles) compact={:.2}GiB",
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
        eprintln!(
            "SuperneoEvalCache::storage: row_offsets={:.2}MiB row_refs={:.2}MiB dense_rows={:.2}MiB dense={:.2}MiB geometric_offsets={:.2}MiB geometric_runs={:.2}MiB",
            row_offset_bytes as f64 / (1u64 << 20) as f64,
            row_block_bytes as f64 / (1u64 << 20) as f64,
            dense_row_block_bytes as f64 / (1u64 << 20) as f64,
            dense_bytes as f64 / (1u64 << 20) as f64,
            geometric_offset_bytes as f64 / (1u64 << 20) as f64,
            geometric_run_bytes as f64 / (1u64 << 20) as f64,
        );
        eprintln!(
            "SuperneoEvalCache::dense-census: blocks={dense_blocks} signed_blocks={signed_dense_blocks} ({:.1}%) coefficients={dense_coefficients} signed_coefficients={signed_dense_coefficients} ({:.1}%)",
            if dense_blocks == 0 {
                0.0
            } else {
                signed_dense_blocks as f64 * 100.0 / dense_blocks as f64
            },
            if dense_coefficients == 0 {
                0.0
            } else {
                signed_dense_coefficients as f64 * 100.0 / dense_coefficients as f64
            },
        );
    }

    let explicit_matrix_masks = if mats.len() <= u16::BITS as usize {
        let rows = mats.first().map_or(0, |matrix| matrix.rows);
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let masks = (0..rows)
            .into_par_iter()
            .map(|row| {
                mats.iter().enumerate().fold(0u16, |mask, (matrix, cache)| {
                    if !cache.identity
                        && (cache.row_offsets.get(row) != cache.row_offsets.get(row + 1)
                            || cache.has_compact_geometric_row(row))
                    {
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
                    if !cache.identity
                        && (cache.row_offsets.get(row) != cache.row_offsets.get(row + 1)
                            || cache.has_compact_geometric_row(row))
                    {
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

impl SuperneoEvalCache {
    /// Common `(rows, columns, matrix count)` checked by cache construction
    /// and artifact decoding. An empty cache has no relation shape.
    pub fn relation_shape(&self) -> Option<(usize, usize, usize)> {
        self.mats
            .first()
            .map(|matrix| (matrix.rows, matrix.cols, self.mats.len()))
    }
}
