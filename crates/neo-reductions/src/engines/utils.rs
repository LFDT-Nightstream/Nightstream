//! Shared PiCCS transcript binding, dimensions, and public-output checks.

#![allow(non_snake_case)]

use crate::error::PiCcsError;
use neo_ccs::crypto::poseidon2_goldilocks::{RATE, WIDTH};
use neo_ccs::{CcsMatrix, CcsStructure, CeClaim};
use neo_math::K;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_symmetric::Permutation;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

pub fn shared_me_input_r<C, Ff>(me_inputs: &[CeClaim<C, Ff, K>], ell_n: usize) -> Result<Option<&[K]>, PiCcsError>
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

pub fn digest_ccs_matrices<F: Field + PrimeField64 + Sync>(s: &CcsStructure<F>) -> Vec<Goldilocks> {
    digest_ccs_matrices_tree(s)
}

const CCS_DIGEST_SEED: u64 = 0x434353445F4D4154;
const CCS_DIGEST_RATE: usize = RATE;
const CCS_DIGEST_CHUNK_WORDS: usize = 65_536;
const CCS_DIGEST_GEOMETRIC_RUNS_PER_CHUNK: usize = 8_192;

const CCS_MATRIX_LEAF_METADATA: u64 = 1;
const CCS_MATRIX_LEAF_IDENTITY: u64 = 2;
const CCS_MATRIX_LEAF_COL_PTR: u64 = 3;
const CCS_MATRIX_LEAF_ROW_IDX: u64 = 4;
const CCS_MATRIX_LEAF_VALS: u64 = 5;
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
    GeometricRunChunk {
        matrix: usize,
        chunk: usize,
        start: usize,
        values: &'a [neo_ccs::GeometricRowRun<Ff>],
    },
}

fn new_ccs_digest_poseidon2() -> Poseidon2Goldilocks<WIDTH> {
    use rand_chacha_p3::{rand_core::SeedableRng, ChaCha8Rng};

    let mut rng = ChaCha8Rng::seed_from_u64(CCS_DIGEST_SEED);
    Poseidon2Goldilocks::<WIDTH>::new_from_rng_128(&mut rng)
}

#[inline]
fn absorb_digest_limb(
    poseidon2: &Poseidon2Goldilocks<WIDTH>,
    state: &mut [Goldilocks; WIDTH],
    absorbed: &mut usize,
    v: Goldilocks,
) {
    if *absorbed >= CCS_DIGEST_RATE {
        poseidon2.permute_mut(state);
        *absorbed = 0;
    }
    state[*absorbed] = v;
    *absorbed += 1;
}

#[inline]
fn absorb_digest_u64(
    poseidon2: &Poseidon2Goldilocks<WIDTH>,
    state: &mut [Goldilocks; WIDTH],
    absorbed: &mut usize,
    v: u64,
) {
    absorb_digest_limb(poseidon2, state, absorbed, Goldilocks::from_u64(v & 0xffff_ffff));
    absorb_digest_limb(poseidon2, state, absorbed, Goldilocks::from_u64(v >> 32));
}

#[inline]
fn absorb_digest_bytes(
    poseidon2: &Poseidon2Goldilocks<WIDTH>,
    state: &mut [Goldilocks; WIDTH],
    absorbed: &mut usize,
    bytes: &[u8],
) {
    for &byte in bytes {
        absorb_digest_u64(poseidon2, state, absorbed, byte as u64);
    }
}

fn digest_ccs_matrix_leaf<Ff: PrimeField64>(leaf: &CcsDigestLeaf<'_, Ff>) -> [Goldilocks; 4] {
    let poseidon2 = new_ccs_digest_poseidon2();
    let mut state = [Goldilocks::ZERO; WIDTH];
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

fn digest_ccs_matrices_tree<Ff: Field + PrimeField64 + Sync>(s: &CcsStructure<Ff>) -> Vec<Goldilocks> {
    let mut leaves = Vec::new();

    for (j, matrix) in s.matrices.iter().enumerate() {
        match matrix {
            CcsMatrix::Identity { n } => {
                leaves.push(CcsDigestLeaf::Identity { matrix: j, n: *n });
            }
            CcsMatrix::Csc(csc_from_s) => {
                let (nrows, ncols) = (csc_from_s.nrows, csc_from_s.ncols);
                let (col_ptr, row_idx, vals) = (
                    csc_from_s.col_ptr.as_slice(),
                    csc_from_s.row_idx.as_slice(),
                    csc_from_s.vals.as_slice(),
                );

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
            CcsMatrix::CscWithGeometricRuns { csc, geometric_runs } => {
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
            CcsMatrix::VerifierArtifact { .. } => {
                panic!("raw CCS digest is unavailable for verifier-artifact matrices")
            }
        }
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let leaf_digests: Vec<[Goldilocks; 4]> = leaves.par_iter().map(digest_ccs_matrix_leaf).collect();

    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    let leaf_digests: Vec<[Goldilocks; 4]> = leaves.iter().map(digest_ccs_matrix_leaf).collect();

    let poseidon2 = new_ccs_digest_poseidon2();
    let mut state = [Goldilocks::ZERO; WIDTH];
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
