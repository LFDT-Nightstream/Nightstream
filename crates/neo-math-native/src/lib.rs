#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(nightstream_sme)]
mod sme {
    use std::sync::OnceLock;

    const TILE: usize = 320;
    const SCRATCH_WORDS: usize = TILE * (8 + 8 + 1);

    unsafe extern "C" {
        fn nightstream_has_signed_sme() -> bool;
        fn nightstream_signed_tile(key: *const [u64; 64], scratch: *mut u64, count: usize, sums: *mut i64) -> bool;
    }

    pub(super) fn sum<T>(key: &[[u64; 64]], blocks: &[T], mask: impl Fn(&T) -> [u64; 3]) -> Option<[i64; 128]> {
        static AVAILABLE: OnceLock<bool> = OnceLock::new();
        // SAFETY: The check takes no arguments and tests the ISA before entering streaming mode.
        if !*AVAILABLE.get_or_init(|| unsafe { nightstream_has_signed_sme() }) || blocks.len() > 4096 {
            return None;
        }
        let mut scratch = vec![0u64; SCRATCH_WORDS];
        let mut sums = [0; 128];
        for tile in blocks.chunks(TILE) {
            for (entry, block) in scratch.as_chunks_mut::<3>().0.iter_mut().zip(tile) {
                *entry = mask(block);
                if entry[0] >= key.len() as u64 {
                    return None;
                }
            }
            // SAFETY: Every index names a 64-word key element. The tile has at most 320 descriptors.
            // The kernel writes within the scratch and sum arrays. Each limb is bounded by 4096 * 64 * u32::MAX.
            let complete =
                unsafe { nightstream_signed_tile(key.as_ptr(), scratch.as_mut_ptr(), tile.len(), sums.as_mut_ptr()) };
            if !complete {
                return None;
            }
        }
        Some(sums)
    }
}

pub fn signed_ring_sum<T>(key: &[[u64; 64]], blocks: &[T], mask: impl Fn(&T) -> [u64; 3]) -> Option<[i64; 128]> {
    #[cfg(nightstream_sme)]
    return sme::sum(key, blocks, mask);
    #[cfg(not(nightstream_sme))]
    {
        let _ = (key, blocks, mask);
        None
    }
}
