//! Packed row-block references and the dense-block side table.

use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

pub(super) const COMPACT_DENSE_TAG: u32 = 1 << 31;
pub(super) const COMPACT_NEGATIVE_TAG: u32 = 1 << 30;
pub(super) const COMPACT_SINGLE_LOCAL_SHIFT: u32 = 24;
pub(super) const COMPACT_SINGLE_LOCAL_MASK: u32 = 0x3f;
pub(super) const COMPACT_SINGLE_BLOCK_MASK: u32 = (1 << COMPACT_SINGLE_LOCAL_SHIFT) - 1;
pub(super) const COMPACT_DENSE_INDEX_MASK: u32 = COMPACT_DENSE_TAG - 1;

#[derive(Clone, Copy, Debug, Default)]
#[repr(transparent)]
pub(super) struct CompactRowBlock(u32);

// `compact_device_parts` exports these values as `u32`.
const _: () = assert!(
    core::mem::size_of::<CompactRowBlock>() == core::mem::size_of::<u32>()
        && core::mem::align_of::<CompactRowBlock>() == core::mem::align_of::<u32>()
);

impl CompactRowBlock {
    #[inline]
    pub(super) fn single(block: usize, local: usize, coefficient: F) -> Self {
        assert!(
            block <= COMPACT_SINGLE_BLOCK_MASK as usize,
            "SuperNeo block index exceeds packed cache"
        );
        debug_assert!(local < D);
        debug_assert!(coefficient == F::ONE || coefficient == F::ZERO - F::ONE);
        Self(
            block as u32
                | (local as u32) << COMPACT_SINGLE_LOCAL_SHIFT
                | (u32::from(coefficient == F::ZERO - F::ONE) * COMPACT_NEGATIVE_TAG),
        )
    }

    #[inline]
    pub(super) fn dense(index: usize) -> Self {
        assert!(
            index <= COMPACT_DENSE_INDEX_MASK as usize,
            "SuperNeo dense row-block cache exceeds u31"
        );
        Self(COMPACT_DENSE_TAG | index as u32)
    }

    #[inline]
    pub(super) fn single_parts(self) -> Option<(usize, usize, F)> {
        (self.0 & COMPACT_DENSE_TAG == 0).then(|| {
            let coefficient = if self.0 & COMPACT_NEGATIVE_TAG == 0 {
                F::ONE
            } else {
                F::ZERO - F::ONE
            };
            (
                (self.0 & COMPACT_SINGLE_BLOCK_MASK) as usize,
                ((self.0 >> COMPACT_SINGLE_LOCAL_SHIFT) & COMPACT_SINGLE_LOCAL_MASK) as usize,
                coefficient,
            )
        })
    }

    #[inline]
    pub(super) fn dense_index(self) -> Option<usize> {
        (self.0 & COMPACT_DENSE_TAG != 0).then_some((self.0 & COMPACT_DENSE_INDEX_MASK) as usize)
    }
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub(super) struct DenseRowBlock {
    block: u32,
    pattern: u32,
}

// `compact_device_parts` exports these values as `[block, pattern]`.
const _: () = assert!(
    core::mem::size_of::<DenseRowBlock>() == core::mem::size_of::<[u32; 2]>()
        && core::mem::align_of::<DenseRowBlock>() == core::mem::align_of::<[u32; 2]>()
);

impl DenseRowBlock {
    #[inline]
    pub(super) fn new(block: usize, pattern: usize) -> Self {
        Self {
            block: u32::try_from(block).expect("SuperNeo block index exceeds u32"),
            pattern: u32::try_from(pattern).expect("SuperNeo dense pattern index exceeds u32"),
        }
    }

    #[inline]
    pub(super) fn block(self) -> usize {
        self.block as usize
    }

    #[inline]
    pub(super) fn pattern(self) -> usize {
        self.pattern as usize
    }

    #[inline]
    pub(super) fn set_pattern(&mut self, pattern: usize) {
        self.pattern = u32::try_from(pattern).expect("SuperNeo dense pattern index exceeds u32");
    }
}
