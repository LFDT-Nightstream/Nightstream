use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

use block2::RcBlock;
use neo_ajtai::nightstream_fprime_setup::{ExpandedKey, PRODUCTION_VERIFIER_ROWS};
use neo_math::D;
use objc2_foundation::NSPageSize;
use objc2_metal::{MTLDevice, MTLResourceOptions};

use super::{Buffer, MetalError, MetalSession};

impl MetalSession {
    pub(super) fn shared_key_rows(&self, key: &ExpandedKey, columns: usize) -> Result<Option<Vec<Buffer>>, MetalError> {
        let row_words = key.blocks() as usize * D;
        let page = NSPageSize();
        let bytes = (columns * D * size_of::<u64>()).next_multiple_of(page);
        let words = key.shared_words();
        if bytes > self.device.maxBufferLength()
            || bytes > row_words * size_of::<u64>()
            || !words.as_ptr().addr().is_multiple_of(page)
            || !(row_words * size_of::<u64>()).is_multiple_of(page)
        {
            return Ok(None);
        }
        let mut rows = Vec::with_capacity(PRODUCTION_VERIFIER_ROWS as usize);
        for row in 0..PRODUCTION_VERIFIER_ROWS as usize {
            let slice = &words[row * row_words..][..bytes / size_of::<u64>()];
            let pointer = NonNull::from(&slice[0]).cast::<c_void>();
            let owner = Arc::clone(&words);
            let release = RcBlock::new(move |_: NonNull<c_void>, _: usize| {
                let _ = &owner;
            });
            let _allocation = self.reserve_buffer(bytes)?;
            // SAFETY: The read-only key slice covers complete VM pages. Metal retains the block,
            // which owns the vector until the last Metal reference releases the buffer.
            let buffer = unsafe {
                self.device
                    .newBufferWithBytesNoCopy_length_options_deallocator(
                        pointer,
                        bytes,
                        MTLResourceOptions::StorageModeShared,
                        Some(&release),
                    )
            };
            let Some(buffer) = buffer else {
                return Ok(None);
            };
            rows.push(buffer);
        }
        Ok(Some(rows))
    }
}
