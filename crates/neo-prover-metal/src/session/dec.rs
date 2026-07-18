//! Metal DEC ownership, split between form evaluation and child materialization.

use neo_math::D;
use objc2_metal::MTLBuffer;

use super::{Buffer, MetalSession};
use crate::MetalError;

mod forms;
mod split;

pub(crate) use forms::{MetalAjtaiRingForms, MetalDecFormPlan};

pub(super) const PRODUCT_COEFFICIENTS: usize = 2 * D - 1;
pub(super) const CHUNK_COLUMNS: usize = 512;

impl MetalSession {
    // Recycling is deliberately exact-size: aliasing a differently shaped
    // scratch allocation would make the shader ABI depend on stale metadata.
    fn take_recycled_buffer(
        &self,
        slot: &std::cell::RefCell<Option<Buffer>>,
        bytes: usize,
    ) -> Result<Buffer, MetalError> {
        let recycled = {
            let mut slot = slot.borrow_mut();
            slot.as_ref()
                .is_some_and(|buffer| buffer.length() as usize == bytes)
                .then(|| {
                    slot.take()
                        .expect("matching recycled Metal buffer exists above")
                })
        };
        recycled.map_or_else(|| self.buffer(bytes), Ok)
    }

    fn recycle_largest_buffer(slot: &std::cell::RefCell<Option<Buffer>>, buffer: Buffer) {
        // Keep one high-water allocation per scratch role rather than growing
        // an unbounded general-purpose pool.
        let bytes = buffer.length();
        let mut slot = slot.borrow_mut();
        if slot.as_ref().is_none_or(|cached| cached.length() <= bytes) {
            *slot = Some(buffer);
        }
    }
}

pub(super) fn checked_product(factors: &[usize], message: &'static str) -> Result<usize, MetalError> {
    factors
        .iter()
        .try_fold(1usize, |value, &factor| value.checked_mul(factor))
        .ok_or(MetalError::Shape(message))
}
