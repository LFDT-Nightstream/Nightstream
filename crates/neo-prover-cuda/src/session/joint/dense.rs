//! Running sources without norm terms (option B's light fold, Lean
//! `Spec/Folding/Light`): the accumulator is a Π_RLC output, so its entries
//! are not signed digits and have no mask. The device stores its field
//! values source by source, block by block and lane by lane.

use neo_math::D;
use neo_reductions::superneo_eval::SuperneoZBlocks;

use super::{Buffer, CudaSession, CudaWitnessMasks};
use crate::CudaError;

pub(crate) struct CudaDenseWitnesses {
    words: Buffer,
    count: usize,
    blocks: usize,
    active: Vec<u32>,
}

impl CudaDenseWitnesses {
    pub(super) fn words(&self) -> &Buffer {
        &self.words
    }

    pub(super) fn count(&self) -> usize {
        self.count
    }

    /// The nonzero sources.
    pub(super) fn active(&self) -> &[u32] {
        &self.active
    }
}

/// The sources of one opening or carried table: the masked sources first,
/// then the dense sources.
#[derive(Clone, Copy)]
pub(crate) struct JointSources<'a> {
    pub(super) masks: &'a CudaWitnessMasks,
    pub(super) dense: Option<&'a CudaDenseWitnesses>,
}

impl<'a> From<&'a CudaWitnessMasks> for JointSources<'a> {
    fn from(masks: &'a CudaWitnessMasks) -> Self {
        Self { masks, dense: None }
    }
}

impl JointSources<'_> {
    pub(super) fn witness_count(&self) -> usize {
        self.masks.witness_count() + self.dense.map_or(0, CudaDenseWitnesses::count)
    }

    pub(super) fn matches_joint(&self, witness_count: usize, blocks: usize) -> bool {
        self.witness_count() == witness_count
            && self.masks.blocks() == blocks
            && self.dense.is_none_or(|dense| dense.blocks == blocks)
    }

    /// The nonzero sources' count: masked, then dense.
    pub(super) fn active_count(&self) -> usize {
        self.masks.active_witnesses().len() + self.dense.map_or(0, |dense| dense.active.len())
    }
}

impl CudaSession {
    pub(super) fn prepare_dense_witnesses(
        &self,
        witnesses: &[SuperneoZBlocks],
    ) -> Result<CudaDenseWitnesses, CudaError> {
        let blocks = witnesses
            .first()
            .ok_or(CudaError::Shape("dense witnesses need sources"))?
            .block_len();
        if blocks == 0
            || witnesses
                .iter()
                .any(|witness| witness.block_len() != blocks || !witness.imag_all_zero())
        {
            return Err(CudaError::Shape("dense witnesses have inconsistent dimensions"));
        }
        let active = witnesses
            .iter()
            .enumerate()
            .filter(|(_, witness)| !witness.is_zero())
            .map(|(index, _)| u32::try_from(index).map_err(|_| CudaError::Shape("witness index exceeds u32")))
            .collect::<Result<Vec<_>, _>>()?;
        let words_per_source = blocks
            .checked_mul(D)
            .ok_or(CudaError::Shape("dense witness size overflow"))?;
        let word_count = words_per_source
            .checked_mul(witnesses.len())
            .ok_or(CudaError::Shape("dense witness size overflow"))?;
        let words = self.filled_buffer(word_count, |destination| {
            for (target, witness) in destination
                .chunks_exact_mut(words_per_source)
                .zip(witnesses)
            {
                if witness.is_zero() {
                    target.fill(0);
                } else {
                    target.copy_from_slice(&witness.re_plane_words());
                }
            }
            Ok(())
        })?;
        Ok(CudaDenseWitnesses {
            words,
            count: witnesses.len(),
            blocks,
            active,
        })
    }
}
