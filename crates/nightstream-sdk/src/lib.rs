#![cfg_attr(not(feature = "std"), no_std)]

pub use nightstream_sdk_macros::{entry, provable, NeoAbi};

pub mod abi;
pub mod poseidon2;

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
mod riscv;
#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
mod stub;

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
pub use riscv::*;
#[cfg(not(any(target_arch = "riscv32", target_arch = "riscv64")))]
pub use stub::*;

#[cfg(all(feature = "alloc", not(feature = "std")))]
extern crate alloc;

#[cfg(all(
    feature = "alloc",
    not(feature = "std"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
mod riscv_alloc;

#[cfg(all(
    feature = "alloc",
    not(feature = "std"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
#[global_allocator]
static NEO_ALLOC: riscv_alloc::BumpAllocator = riscv_alloc::BumpAllocator;

#[cfg(all(
    feature = "alloc",
    not(feature = "std"),
    any(target_arch = "riscv32", target_arch = "riscv64")
))]
#[alloc_error_handler]
fn alloc_error(_layout: core::alloc::Layout) -> ! {
    halt()
}

#[cfg(all(not(feature = "std"), any(target_arch = "riscv32", target_arch = "riscv64")))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    halt()
}
