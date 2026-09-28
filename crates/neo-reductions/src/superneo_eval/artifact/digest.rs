//! Poseidon2 binding for one compact cache representation.

use neo_ccs::crypto::poseidon2_goldilocks::{permute_state, DIGEST_LEN, RATE, WIDTH};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

use super::{
    digest_leaves, DenseBlockStore, DigestLeaf, RowOffsetStore, SuperneoEvalCache, OFFSET_EMPTY, OFFSET_U16_CHUNKED,
    OFFSET_U24, OFFSET_U32, SCHEMA_VERSION, SECTION_DENSE_COEFFICIENTS, SECTION_DENSE_ROW_BLOCKS,
    SECTION_EXPLICIT_MASKS, SECTION_GEOMETRIC_RUNS, SECTION_ROW_BLOCKS,
};

pub(super) fn cache_digest(
    cache: &SuperneoEvalCache,
    matrix_digest: &[F; DIGEST_LEN],
    artifact_bytes: u64,
) -> [F; DIGEST_LEN] {
    let leaves = digest_leaves(cache);
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    let digests = leaves.par_iter().map(digest_leaf).collect::<Vec<_>>();
    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
    let digests = leaves.iter().map(digest_leaf).collect::<Vec<_>>();

    let mut root = DigestState::default();
    root.bytes(b"nightstream/superneo-cache-artifact/v2");
    root.u64(u64::from(SCHEMA_VERSION));
    root.u64(artifact_bytes);
    root.u64(cache.mats.len() as u64);
    for &word in matrix_digest {
        root.field(word);
    }
    root.u64(digests.len() as u64);
    for (index, digest) in digests.iter().enumerate() {
        root.u64(index as u64);
        for &word in digest {
            root.field(word);
        }
    }
    root.finish()
}

fn digest_leaf(leaf: &DigestLeaf<'_>) -> [F; DIGEST_LEN] {
    let mut state = DigestState::default();
    state.bytes(b"nightstream/superneo-cache-leaf/v2");
    match leaf {
        DigestLeaf::MatrixMeta { matrix, value } => {
            state.u64(1);
            state.u64(*matrix as u64);
            state.u64(value.rows as u64);
            state.u64(value.cols as u64);
            state.u64(value.identity as u64);
            absorb_offset_meta(&mut state, &value.row_offsets);
            state.u64(value.row_blocks.len() as u64);
            state.u64(value.dense_row_blocks.len() as u64);
            let DenseBlockStore::Compact {
                offsets,
                locals,
                coefficients,
            } = &value.dense_orig
            else {
                unreachable!("cache validation rejects unfinished dense dictionaries")
            };
            state.u64(offsets.len() as u64);
            state.u64(locals.len() as u64);
            state.u64(coefficients.len() as u64);
            absorb_offset_meta(&mut state, &value.geometric_row_offsets);
            state.u64(value.geometric_runs.len() as u64);
            state.u64(value.seeded_phi81_blocks.len() as u64);
        }
        DigestLeaf::U32 {
            matrix,
            section,
            start,
            values,
        } => {
            absorb_leaf_header(&mut state, 2, *matrix, *section, *start, values.len());
            for &value in *values {
                state.u32(value);
            }
        }
        DigestLeaf::U16 {
            matrix,
            section,
            start,
            values,
        } => {
            absorb_leaf_header(&mut state, 3, *matrix, *section, *start, values.len());
            for &value in *values {
                state.u16(value);
            }
        }
        DigestLeaf::Bytes {
            matrix,
            section,
            start,
            values,
        } => {
            absorb_leaf_header(&mut state, 4, *matrix, *section, *start, values.len());
            state.bytes(values);
        }
        DigestLeaf::RowBlocks { matrix, start, values } => {
            absorb_leaf_header(&mut state, 5, *matrix, SECTION_ROW_BLOCKS, *start, values.len());
            for value in *values {
                state.u32(value.word());
            }
        }
        DigestLeaf::DenseRowBlocks { matrix, start, values } => {
            absorb_leaf_header(&mut state, 11, *matrix, SECTION_DENSE_ROW_BLOCKS, *start, values.len());
            for value in *values {
                let [block, pattern] = value.words();
                state.u32(block);
                state.u32(pattern);
            }
        }
        DigestLeaf::Fields { matrix, start, values } => {
            absorb_leaf_header(&mut state, 6, *matrix, SECTION_DENSE_COEFFICIENTS, *start, values.len());
            for &value in *values {
                state.u64(value.as_canonical_u64());
            }
        }
        DigestLeaf::Runs { matrix, start, values } => {
            absorb_leaf_header(&mut state, 7, *matrix, SECTION_GEOMETRIC_RUNS, *start, values.len());
            for value in *values {
                for &word in value {
                    state.u64(word);
                }
            }
        }
        DigestLeaf::Seeded { matrix, block, value } => {
            state.u64(8);
            state.u64(*matrix as u64);
            state.u64(*block as u64);
            state.u64(value.row_start() as u64);
            state.u64(value.word_width() as u64);
            state.u64(value.kappa() as u64);
            state.u64(value.message_cols() as u64);
            state.u64(value.chunk_size() as u64);
            state.u64(value.has_superneo_transformed_columns() as u64);
            state.u64(value.word_starts().len() as u64);
            for &start in value.word_starts() {
                state.u64(start as u64);
            }
            state.u64(value.chunk_seeds_by_row().len() as u64);
            for seeds in value.chunk_seeds_by_row() {
                state.u64(seeds.len() as u64);
                for seed in seeds {
                    state.bytes(seed);
                }
            }
        }
        DigestLeaf::MasksMeta { present, len } => {
            state.u64(9);
            state.u64(*present as u64);
            state.u64(*len as u64);
        }
        DigestLeaf::Masks { start, values } => {
            absorb_leaf_header(&mut state, 10, usize::MAX, SECTION_EXPLICIT_MASKS, *start, values.len());
            for &value in *values {
                state.u16(value);
            }
        }
    }
    state.finish()
}

fn absorb_leaf_header(state: &mut DigestState, kind: u64, matrix: usize, section: u64, start: usize, len: usize) {
    state.u64(kind);
    state.u64(matrix as u64);
    state.u64(section);
    state.u64(start as u64);
    state.u64(len as u64);
}

fn absorb_offset_meta(state: &mut DigestState, offsets: &RowOffsetStore) {
    match offsets {
        RowOffsetStore::Empty => {
            state.u64(u64::from(OFFSET_EMPTY));
            state.u64(0);
            state.u64(0);
        }
        RowOffsetStore::U16Chunked {
            chunk_offsets,
            local_offsets,
        } => {
            state.u64(u64::from(OFFSET_U16_CHUNKED));
            state.u64(chunk_offsets.len() as u64);
            state.u64(local_offsets.len() as u64);
        }
        RowOffsetStore::U24(bytes) => {
            state.u64(u64::from(OFFSET_U24));
            state.u64(bytes.len() as u64);
            state.u64(0);
        }
        RowOffsetStore::U32(offsets) => {
            state.u64(u64::from(OFFSET_U32));
            state.u64(offsets.len() as u64);
            state.u64(0);
        }
    }
}

#[derive(Clone)]
struct DigestState {
    state: [F; WIDTH],
    absorbed: usize,
}

impl Default for DigestState {
    fn default() -> Self {
        Self {
            state: [F::ZERO; WIDTH],
            absorbed: 0,
        }
    }
}

impl DigestState {
    fn field(&mut self, value: F) {
        if self.absorbed == RATE {
            self.state = permute_state(self.state);
            self.absorbed = 0;
        }
        self.state[self.absorbed] += value;
        self.absorbed += 1;
    }

    fn u16(&mut self, value: u16) {
        self.field(F::from_u64(u64::from(value)));
    }

    fn u32(&mut self, value: u32) {
        self.field(F::from_u64(u64::from(value)));
    }

    fn u64(&mut self, value: u64) {
        self.u32(value as u32);
        self.u32((value >> 32) as u32);
    }

    fn bytes(&mut self, bytes: &[u8]) {
        self.u64(bytes.len() as u64);
        for chunk in bytes.chunks(7) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.field(F::from_u64(u64::from_le_bytes(word)));
        }
    }

    fn finish(mut self) -> [F; DIGEST_LEN] {
        if self.absorbed != 0 {
            self.state = permute_state(self.state);
        }
        self.state[0] += F::ONE;
        self.state = permute_state(self.state);
        self.state[..DIGEST_LEN]
            .try_into()
            .expect("Poseidon2 digest width")
    }
}
