#![no_std]
#![no_main]

extern crate alloc;

struct GuestBumpAlloc;

unsafe impl core::alloc::GlobalAlloc for GuestBumpAlloc {
    #[allow(static_mut_refs)]
    unsafe fn alloc(&self, layout: core::alloc::Layout) -> *mut u8 {
        static mut HEAP: [u8; 64 * 1024] = [0; 64 * 1024];
        static mut OFFSET: usize = 0;

        let align = layout.align();
        let size = layout.size();
        let mut off = OFFSET;
        off = (off + align - 1) & !(align - 1);
        if off.checked_add(size).is_none() || off + size > HEAP.len() {
            return core::ptr::null_mut();
        }
        OFFSET = off + size;
        HEAP.as_mut_ptr().add(off)
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: core::alloc::Layout) {}
}

#[global_allocator]
static GUEST_ALLOC: GuestBumpAlloc = GuestBumpAlloc;

#[derive(nightstream_sdk::NeoAbi)]
struct PoseidonInput {
    a: u64,
    b: u64,
    c: u64,
    d: u64,
}

#[derive(nightstream_sdk::NeoAbi)]
struct PoseidonDigest {
    d0: u64,
    d1: u64,
    d2: u64,
    d3: u64,
}

#[inline]
fn poseidon2_hash_soft(input: &[u64; 4]) -> [u64; 4] {
    let mut preimage = [0u8; 32];
    let mut i = 0usize;
    while i < 4 {
        let bytes = input[i].to_le_bytes();
        let start = i * 8;
        preimage[start..start + 8].copy_from_slice(&bytes);
        i += 1;
    }

    let digest = qp_poseidon_core::hash_variable_length_bytes(&preimage);
    [
        u64::from_le_bytes(digest[0..8].try_into().expect("digest split 0")),
        u64::from_le_bytes(digest[8..16].try_into().expect("digest split 1")),
        u64::from_le_bytes(digest[16..24].try_into().expect("digest split 2")),
        u64::from_le_bytes(digest[24..32].try_into().expect("digest split 3")),
    ]
}

#[nightstream_sdk::provable]
fn poseidon2_soft_example(input: PoseidonInput) -> PoseidonDigest {
    let digest = poseidon2_hash_soft(&[input.a, input.b, input.c, input.d]);
    PoseidonDigest {
        d0: digest[0],
        d1: digest[1],
        d2: digest[2],
        d3: digest[3],
    }
}
