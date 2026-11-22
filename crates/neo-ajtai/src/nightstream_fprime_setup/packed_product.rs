use neo_ccs::Mat;
use neo_math::signed_sums::accelerate::{signed_ring_dot, ROW_WORKSPACE_WORDS};
use rayon::prelude::*;

use super::{Commitment, ExpandedKey, Fq, SignedBlock, D, MAX_MESSAGE_COLUMNS, PRODUCTION_VERIFIER_ROWS};

const WORKSPACE_WORDS: usize = PRODUCTION_VERIFIER_ROWS as usize * ROW_WORKSPACE_WORDS;
const MIN_ACTIVE_COLUMNS: usize = (2 * WORKSPACE_WORDS * size_of::<f64>()).div_ceil(size_of::<SignedBlock>());
const _: () = assert!(WORKSPACE_WORDS * size_of::<f64>() == 4_718_592);

pub(super) fn commit(witness: &Mat<Fq>) -> Option<Commitment> {
    let columns = witness.cols();
    if witness.rows() != D || !(MIN_ACTIVE_COLUMNS..=MAX_MESSAGE_COLUMNS as usize).contains(&columns) {
        return None;
    }
    let (positive, negative) = witness.packed_signed_unit_column_masks()?;
    let key = ExpandedKey::shared()?;
    if key.blocks() < columns as u64 {
        return None;
    }
    let active = positive
        .iter()
        .zip(negative)
        .filter(|(p, n)| **p | **n != 0)
        .count();
    if active < MIN_ACTIVE_COLUMNS || active < columns.div_ceil(2) {
        return None;
    }

    let mut output = Commitment::zeros(PRODUCTION_VERIFIER_ROWS as usize);
    let mut workspace = vec![0.0; WORKSPACE_WORDS];
    output
        .data_mut()
        .par_chunks_exact_mut(D)
        .zip(key.words().par_chunks_exact(key.blocks() as usize * D))
        .zip(workspace.par_chunks_exact_mut(ROW_WORKSPACE_WORDS))
        .for_each(|((row, key_row), scratch)| {
            row.copy_from_slice(&signed_ring_dot(&key_row[..columns * D], positive, negative, scratch));
        });
    Some(output)
}
