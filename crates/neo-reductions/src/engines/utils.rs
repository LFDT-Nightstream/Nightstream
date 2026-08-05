//! Shared PiCCS transcript binding, dimensions, and public-output checks.
//!
//! Contains only the essential functions needed by prove and verify.

#![allow(non_snake_case)]

use crate::error::PiCcsError;
use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsClaim, CcsMatrix, CcsStructure, CeClaim};
use neo_math::{D, K};
use neo_params::NeoParams;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_symmetric::Permutation;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

/// Validate that all ME inputs share the same evaluation point `r`.
///
/// Returns `None` when `me_inputs` is empty, otherwise returns a shared `r` slice.
pub fn shared_me_input_r<'a, C, Ff>(
    me_inputs: &'a [CeClaim<C, Ff, K>],
    ell_n: usize,
) -> Result<Option<&'a [K]>, PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    if me_inputs.is_empty() {
        return Ok(None);
    }

    let r0 = me_inputs[0].r.as_slice();
    if r0.len() != ell_n {
        return Err(PiCcsError::InvalidInput(format!(
            "ME input r length mismatch at accumulator #0: expected ell_n = {}, got {}",
            ell_n,
            r0.len()
        )));
    }

    for (idx, me) in me_inputs.iter().enumerate().skip(1) {
        if me.r.len() != ell_n {
            return Err(PiCcsError::InvalidInput(format!(
                "ME input r length mismatch at accumulator #{}: expected ell_n = {}, got {}",
                idx,
                ell_n,
                me.r.len()
            )));
        }
        if me.r.as_slice() != r0 {
            return Err(PiCcsError::InvalidInput(format!(
                "ME input r mismatch at accumulator #{}: all ME inputs must share the same r",
                idx
            )));
        }
    }

    Ok(Some(r0))
}

/// Validate MCS-output `X` content against public `x` under SuperNeo packed semantics.
///
/// CE carries projected input ring slots. Public field `x[c]` must occupy row
/// `c % D` in ring-slot column `c / D`; other lanes in those slots are owned by
/// `L_in(z)` and may be non-zero after ring-linear folding.
pub fn validate_mcs_output_x_recomposition<Ff>(
    _params: &NeoParams,
    ccs_m: usize,
    mcs_list: &[CcsClaim<Cmt, Ff>],
    me_outputs: &[CeClaim<Cmt, Ff, K>],
) -> Result<(), PiCcsError>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    if ccs_m == 0 {
        return Err(PiCcsError::InvalidInput("CCS width m must be > 0".into()));
    }
    for (idx, inst) in mcs_list.iter().enumerate() {
        let out = me_outputs.get(idx).ok_or_else(|| {
            PiCcsError::ProtocolError(format!(
                "missing me_outputs entry for mcs_list index {} (|me_outputs|={})",
                idx,
                me_outputs.len()
            ))
        })?;

        if inst.x.len() != inst.m_in {
            return Err(PiCcsError::InvalidInput(format!(
                "mcs_list[{idx}].x.len()={}, expected m_in={}",
                inst.x.len(),
                inst.m_in
            )));
        }
        if out.X.cols() != inst.m_in {
            return Err(PiCcsError::ProtocolError(format!(
                "me_outputs[{idx}].X cols mismatch (got {}, expected {})",
                out.X.cols(),
                inst.m_in
            )));
        }

        for c in 0..inst.m_in {
            let lane = c % D;
            let column = c / D;
            let got = out.X[(lane, column)];
            if got != inst.x[c] {
                return Err(PiCcsError::ProtocolError(format!(
                    "me_outputs[{idx}].X lane {lane} at ring column {column} does not match mcs_list[{idx}].x[{c}]"
                )));
            }
        }
    }

    Ok(())
}

pub fn digest_ccs_matrices<F: Field + PrimeField64>(s: &CcsStructure<F>) -> Vec<Goldilocks> {
    use rand_chacha_p3::{rand_core::SeedableRng, ChaCha8Rng};

    const CCS_DIGEST_SEED: u64 = 0x434353445F4D4154;
    let mut rng = ChaCha8Rng::seed_from_u64(CCS_DIGEST_SEED);
    let poseidon2 = Poseidon2Goldilocks::<16>::new_from_rng_128(&mut rng);

    let mut state = [Goldilocks::ZERO; 16];
    let mut absorbed = 0;

    const DOMAIN_STRING: &[u8] = b"neo/ccs/matrices/v1";
    for &byte in DOMAIN_STRING {
        if absorbed >= 15 {
            poseidon2.permute_mut(&mut state);
            absorbed = 0;
        }
        state[absorbed] = Goldilocks::from_u32(byte as u32);
        absorbed += 1;
    }

    if absorbed + 3 >= 16 {
        poseidon2.permute_mut(&mut state);
        absorbed = 0;
    }
    state[absorbed] = Goldilocks::from_u64(s.n as u64);
    state[absorbed + 1] = Goldilocks::from_u64(s.m as u64);
    state[absorbed + 2] = Goldilocks::from_u64(s.t() as u64);

    poseidon2.permute_mut(&mut state);

    for (j, matrix) in s.matrices.iter().enumerate() {
        absorbed = 0;
        state[absorbed] = Goldilocks::from_u64(j as u64);
        absorbed += 1;

        let mut emit = |row: usize, col: usize, val_u64: u64| {
            if absorbed + 3 > 15 {
                poseidon2.permute_mut(&mut state);
                absorbed = 0;
            }
            state[absorbed] = Goldilocks::from_u64(row as u64);
            state[absorbed + 1] = Goldilocks::from_u64(col as u64);
            state[absorbed + 2] = Goldilocks::from_u64(val_u64);
            absorbed += 3;
        };

        match matrix {
            CcsMatrix::Identity { n } => {
                debug_assert_eq!(*n, s.n);
                debug_assert_eq!(*n, s.m);
                let one_u = F::ONE.as_canonical_u64();
                for row in 0..s.n {
                    emit(row, row, one_u);
                }
            }
            CcsMatrix::Csc(csc) => {
                // Enumerate non-zeros in row-major order (matches dense scan) without allocating
                // a `Vec<Vec<_>>` of length `nrows` (which is massive for large circuits).
                //
                // Strategy: build CSR-style row segments in one contiguous allocation.
                let nrows = csc.nrows;
                let nnz = csc.vals.len();
                debug_assert_eq!(csc.row_idx.len(), nnz);

                // 1) Count entries per row.
                let mut row_counts = vec![0u32; nrows];
                for &r in csc.row_idx.iter() {
                    row_counts[r as usize] += 1;
                }

                // 2) Prefix sums to get row offsets.
                let mut row_offsets = vec![0usize; nrows + 1];
                for r in 0..nrows {
                    row_offsets[r + 1] = row_offsets[r] + (row_counts[r] as usize);
                }
                debug_assert_eq!(row_offsets[nrows], nnz);

                // 3) Fill (col,val) pairs into per-row segments while scanning columns in order.
                let mut write_pos = row_offsets[..nrows].to_vec();
                let mut entries = vec![(0usize, 0u64); nnz];

                for col in 0..csc.ncols {
                    for k in csc.column_range(col) {
                        let row = csc.row_index(k);
                        let idx = write_pos[row];
                        write_pos[row] = idx + 1;
                        entries[idx] = (col, csc.vals[k].as_canonical_u64());
                    }
                }

                // 4) Emit in row-major order.
                for row in 0..nrows {
                    let start = row_offsets[row];
                    let end = row_offsets[row + 1];
                    for &(col, val_u64) in &entries[start..end] {
                        emit(row, col, val_u64);
                    }
                }
            }
            CcsMatrix::CscWithSeededPhi81 {
                csc,
                blocks,
                geometric_runs,
            } => {
                let mut entries = Vec::with_capacity(csc.vals.len());
                for col in 0..csc.ncols {
                    for index in csc.column_range(col) {
                        entries.push((csc.row_index(index), col, csc.vals[index].as_canonical_u64()));
                    }
                }
                entries.sort_unstable_by_key(|&(row, col, _)| (row, col));
                for (row, col, value) in entries {
                    emit(row, col, value);
                }

                // Out-of-range row sentinels domain-separate compact block
                // descriptors from ordinary matrix entries.
                for (block_index, block) in blocks.iter().enumerate() {
                    emit(usize::MAX, block_index, 0x5048_4938_3153_4545);
                    emit(usize::MAX - 1, block.row_start(), block.kappa() as u64);
                    emit(usize::MAX - 2, block.message_cols(), block.chunk_size() as u64);
                    emit(
                        usize::MAX - 3,
                        block.word_starts().len(),
                        u64::from(block.has_superneo_transformed_columns()),
                    );
                    emit(usize::MAX - 4, usize::MAX, block.word_width() as u64);
                    for (word, &start) in block.word_starts().iter().enumerate() {
                        emit(usize::MAX - 4, word, start as u64);
                    }
                    for (seed_row, seeds) in block.chunk_seeds_by_row().iter().enumerate() {
                        for (chunk, seed) in seeds.iter().enumerate() {
                            for limb in 0..4 {
                                let value = u64::from_le_bytes(seed[limb * 8..(limb + 1) * 8].try_into().unwrap());
                                emit(usize::MAX - 5 - seed_row, chunk * 4 + limb, value);
                            }
                        }
                    }
                }
                for (run_index, run) in geometric_runs.iter().enumerate() {
                    let sentinel = usize::MAX / 2;
                    emit(sentinel, run_index, 0x4745_4f4d_5255_4e31);
                    emit(sentinel - 1, run.row(), run.column_start() as u64);
                    emit(sentinel - 2, run.len(), run.initial().as_canonical_u64());
                    emit(sentinel - 3, run_index, run.ratio().as_canonical_u64());
                }
            }
        }

        poseidon2.permute_mut(&mut state);
    }

    state[0..4].to_vec()
}

const CCS_DIGEST_SEED: u64 = 0x434353445F4D4154;
const CCS_DIGEST_CHUNK_WORDS: usize = 65_536;
const CCS_DIGEST_GEOMETRIC_RUNS_PER_CHUNK: usize = 8_192;

const CCS_MATRIX_LEAF_METADATA: u64 = 1;
const CCS_MATRIX_LEAF_IDENTITY: u64 = 2;
const CCS_MATRIX_LEAF_COL_PTR: u64 = 3;
const CCS_MATRIX_LEAF_ROW_IDX: u64 = 4;
const CCS_MATRIX_LEAF_VALS: u64 = 5;
const CCS_MATRIX_LEAF_SEEDED_PHI81: u64 = 6;
const CCS_MATRIX_LEAF_GEOMETRIC_RUN: u64 = 7;

enum CcsDigestLeaf<'a, Ff> {
    Identity {
        matrix: usize,
        n: usize,
    },
    Metadata {
        matrix: usize,
        nrows: usize,
        ncols: usize,
        col_ptr_len: usize,
        row_idx_len: usize,
        vals_len: usize,
    },
    IndexChunk {
        matrix: usize,
        segment: u64,
        chunk: usize,
        start: usize,
        values: &'a [u32],
    },
    FieldChunk {
        matrix: usize,
        chunk: usize,
        start: usize,
        values: &'a [Ff],
    },
    SeededPhi81 {
        matrix: usize,
        block: usize,
        value: &'a neo_ccs::SeededPhi81LinearBlock,
    },
    GeometricRunChunk {
        matrix: usize,
        chunk: usize,
        start: usize,
        values: &'a [neo_ccs::GeometricRowRun<Ff>],
    },
}

fn new_ccs_digest_poseidon2() -> Poseidon2Goldilocks<16> {
    use rand_chacha_p3::{rand_core::SeedableRng, ChaCha8Rng};

    let mut rng = ChaCha8Rng::seed_from_u64(CCS_DIGEST_SEED);
    Poseidon2Goldilocks::<16>::new_from_rng_128(&mut rng)
}

#[inline]
fn absorb_digest_limb(
    poseidon2: &Poseidon2Goldilocks<16>,
    state: &mut [Goldilocks; 16],
    absorbed: &mut usize,
    v: Goldilocks,
) {
    if *absorbed >= 15 {
        poseidon2.permute_mut(state);
        *absorbed = 0;
    }
    state[*absorbed] = v;
    *absorbed += 1;
}

#[inline]
fn absorb_digest_u64(poseidon2: &Poseidon2Goldilocks<16>, state: &mut [Goldilocks; 16], absorbed: &mut usize, v: u64) {
    absorb_digest_limb(poseidon2, state, absorbed, Goldilocks::from_u64(v & 0xffff_ffff));
    absorb_digest_limb(poseidon2, state, absorbed, Goldilocks::from_u64(v >> 32));
}

#[inline]
fn absorb_digest_bytes(
    poseidon2: &Poseidon2Goldilocks<16>,
    state: &mut [Goldilocks; 16],
    absorbed: &mut usize,
    bytes: &[u8],
) {
    for &byte in bytes {
        absorb_digest_u64(poseidon2, state, absorbed, byte as u64);
    }
}

fn digest_ccs_matrix_leaf<Ff: PrimeField64>(leaf: &CcsDigestLeaf<'_, Ff>) -> [Goldilocks; 4] {
    let poseidon2 = new_ccs_digest_poseidon2();
    let mut state = [Goldilocks::ZERO; 16];
    let mut absorbed = 0usize;

    absorb_digest_bytes(&poseidon2, &mut state, &mut absorbed, b"neo/ccs/matrix-leaf/v3");
    match leaf {
        CcsDigestLeaf::Identity { matrix, n } => {
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, *matrix as u64);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, CCS_MATRIX_LEAF_IDENTITY);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, 0);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, 0);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, 1);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, *n as u64);
        }
        CcsDigestLeaf::Metadata {
            matrix,
            nrows,
            ncols,
            col_ptr_len,
            row_idx_len,
            vals_len,
        } => {
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, *matrix as u64);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, CCS_MATRIX_LEAF_METADATA);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, 0);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, 0);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, 5);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, *nrows as u64);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, *ncols as u64);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, *col_ptr_len as u64);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, *row_idx_len as u64);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, *vals_len as u64);
        }
        CcsDigestLeaf::IndexChunk {
            matrix,
            segment,
            chunk,
            start,
            values,
        } => {
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, *matrix as u64);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, *segment);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, *chunk as u64);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, *start as u64);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, values.len() as u64);
            for &v in *values {
                absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, v as u64);
            }
        }
        CcsDigestLeaf::FieldChunk {
            matrix,
            chunk,
            start,
            values,
        } => {
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, *matrix as u64);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, CCS_MATRIX_LEAF_VALS);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, *chunk as u64);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, *start as u64);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, values.len() as u64);
            for &v in *values {
                absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, v.as_canonical_u64());
            }
        }
        CcsDigestLeaf::SeededPhi81 { matrix, block, value } => {
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, *matrix as u64);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, CCS_MATRIX_LEAF_SEEDED_PHI81);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, *block as u64);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, value.row_start() as u64);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, value.kappa() as u64);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, value.message_cols() as u64);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, value.chunk_size() as u64);
            absorb_digest_u64(
                &poseidon2,
                &mut state,
                &mut absorbed,
                value.has_superneo_transformed_columns() as u64,
            );
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, value.word_width() as u64);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, value.word_starts().len() as u64);
            for &start in value.word_starts() {
                absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, start as u64);
            }
            absorb_digest_u64(
                &poseidon2,
                &mut state,
                &mut absorbed,
                value.chunk_seeds_by_row().len() as u64,
            );
            for seeds in value.chunk_seeds_by_row() {
                absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, seeds.len() as u64);
                for seed in seeds {
                    absorb_digest_bytes(&poseidon2, &mut state, &mut absorbed, seed);
                }
            }
        }
        CcsDigestLeaf::GeometricRunChunk {
            matrix,
            chunk,
            start,
            values,
        } => {
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, *matrix as u64);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, CCS_MATRIX_LEAF_GEOMETRIC_RUN);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, *chunk as u64);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, *start as u64);
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, values.len() as u64);
            for value in *values {
                absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, value.row() as u64);
                absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, value.column_start() as u64);
                absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, value.len() as u64);
                absorb_digest_u64(
                    &poseidon2,
                    &mut state,
                    &mut absorbed,
                    value.initial().as_canonical_u64(),
                );
                absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, value.ratio().as_canonical_u64());
            }
        }
    }

    poseidon2.permute_mut(&mut state);
    [state[0], state[1], state[2], state[3]]
}

fn push_index_digest_chunks<'a, Ff>(
    leaves: &mut Vec<CcsDigestLeaf<'a, Ff>>,
    matrix: usize,
    segment: u64,
    values: &'a [u32],
) {
    for (chunk, slice) in values.chunks(CCS_DIGEST_CHUNK_WORDS).enumerate() {
        leaves.push(CcsDigestLeaf::IndexChunk {
            matrix,
            segment,
            chunk,
            start: chunk * CCS_DIGEST_CHUNK_WORDS,
            values: slice,
        });
    }
}

fn push_field_digest_chunks<'a, Ff>(leaves: &mut Vec<CcsDigestLeaf<'a, Ff>>, matrix: usize, values: &'a [Ff]) {
    for (chunk, slice) in values.chunks(CCS_DIGEST_CHUNK_WORDS).enumerate() {
        leaves.push(CcsDigestLeaf::FieldChunk {
            matrix,
            chunk,
            start: chunk * CCS_DIGEST_CHUNK_WORDS,
            values: slice,
        });
    }
}

/// Compute the CCS matrix digest, optionally using a prebuilt sparse cache.
///
/// This cache-aware variant binds a native CSC encoding under a Poseidon2 tree (`v3-tree`).
/// Matrix/segment leaves are hashed independently so preprocessing can use CPU parallelism,
/// then the root absorbs the ordered leaf digests. Prover/verifier soundness is preserved
/// because both sides bind the same domain, dimensions, matrix order, and full CSC content.
pub fn digest_ccs_matrices_with_sparse_cache<Ff: Field + PrimeField64 + Sync>(
    s: &CcsStructure<Ff>,
    sparse: Option<&crate::engines::optimized_engine::SparseCache<Ff>>,
) -> Vec<Goldilocks> {
    let mut leaves = Vec::new();

    for (j, matrix) in s.matrices.iter().enumerate() {
        match matrix {
            CcsMatrix::Identity { n } => {
                leaves.push(CcsDigestLeaf::Identity { matrix: j, n: *n });
            }
            CcsMatrix::Csc(csc_from_s) => {
                let cached_csc = sparse.and_then(|sp| sp.csc(j));
                #[cfg(debug_assertions)]
                if let Some(c) = cached_csc {
                    debug_assert_eq!(c.nrows, csc_from_s.nrows, "CSC cache nrows mismatch for matrix {j}");
                    debug_assert_eq!(c.ncols, csc_from_s.ncols, "CSC cache ncols mismatch for matrix {j}");
                    debug_assert_eq!(
                        c.col_ptr, csc_from_s.col_ptr,
                        "CSC cache col_ptr mismatch for matrix {j}"
                    );
                    debug_assert_eq!(
                        c.row_idx, csc_from_s.row_idx,
                        "CSC cache row_idx mismatch for matrix {j}"
                    );
                    debug_assert_eq!(c.vals, csc_from_s.vals, "CSC cache vals mismatch for matrix {j}");
                }
                let (nrows, ncols, col_ptr, row_idx, vals) = if let Some(c) = cached_csc {
                    (
                        c.nrows,
                        c.ncols,
                        c.col_ptr.as_slice(),
                        c.row_idx.as_slice(),
                        c.vals.as_slice(),
                    )
                } else {
                    (
                        csc_from_s.nrows,
                        csc_from_s.ncols,
                        csc_from_s.col_ptr.as_slice(),
                        csc_from_s.row_idx.as_slice(),
                        csc_from_s.vals.as_slice(),
                    )
                };

                leaves.push(CcsDigestLeaf::Metadata {
                    matrix: j,
                    nrows,
                    ncols,
                    col_ptr_len: col_ptr.len(),
                    row_idx_len: row_idx.len(),
                    vals_len: vals.len(),
                });
                push_index_digest_chunks(&mut leaves, j, CCS_MATRIX_LEAF_COL_PTR, col_ptr);
                push_index_digest_chunks(&mut leaves, j, CCS_MATRIX_LEAF_ROW_IDX, row_idx);
                push_field_digest_chunks(&mut leaves, j, vals);
            }
            CcsMatrix::CscWithSeededPhi81 {
                csc,
                blocks,
                geometric_runs,
            } => {
                leaves.push(CcsDigestLeaf::Metadata {
                    matrix: j,
                    nrows: csc.nrows,
                    ncols: csc.ncols,
                    col_ptr_len: csc.col_ptr.len(),
                    row_idx_len: csc.row_idx.len(),
                    vals_len: csc.vals.len(),
                });
                push_index_digest_chunks(&mut leaves, j, CCS_MATRIX_LEAF_COL_PTR, &csc.col_ptr);
                push_index_digest_chunks(&mut leaves, j, CCS_MATRIX_LEAF_ROW_IDX, &csc.row_idx);
                push_field_digest_chunks(&mut leaves, j, &csc.vals);
                for (block, value) in blocks.iter().enumerate() {
                    leaves.push(CcsDigestLeaf::SeededPhi81 {
                        matrix: j,
                        block,
                        value,
                    });
                }
                for (chunk, values) in geometric_runs
                    .chunks(CCS_DIGEST_GEOMETRIC_RUNS_PER_CHUNK)
                    .enumerate()
                {
                    leaves.push(CcsDigestLeaf::GeometricRunChunk {
                        matrix: j,
                        chunk,
                        start: chunk * CCS_DIGEST_GEOMETRIC_RUNS_PER_CHUNK,
                        values,
                    });
                }
            }
        }
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let leaf_digests: Vec<[Goldilocks; 4]> = leaves.par_iter().map(digest_ccs_matrix_leaf).collect();

    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    let leaf_digests: Vec<[Goldilocks; 4]> = leaves.iter().map(digest_ccs_matrix_leaf).collect();

    let poseidon2 = new_ccs_digest_poseidon2();
    let mut state = [Goldilocks::ZERO; 16];
    let mut absorbed = 0usize;
    absorb_digest_bytes(&poseidon2, &mut state, &mut absorbed, b"neo/ccs/matrices/v3-tree");
    absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, s.n as u64);
    absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, s.m as u64);
    absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, s.t() as u64);
    absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, leaf_digests.len() as u64);
    poseidon2.permute_mut(&mut state);
    absorbed = 0;

    for (idx, leaf) in leaf_digests.iter().enumerate() {
        absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, idx as u64);
        for &limb in leaf {
            absorb_digest_u64(&poseidon2, &mut state, &mut absorbed, limb.as_canonical_u64());
        }
    }
    poseidon2.permute_mut(&mut state);

    state[0..4].to_vec()
}
