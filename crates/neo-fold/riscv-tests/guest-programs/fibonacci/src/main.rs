//! Fibonacci guest program for RISC-V proving.
//!
//! This program computes Fibonacci numbers and is designed to be
//! compiled to RISC-V and proven using Neo's folding scheme.

#![no_std]
#![no_main]

use core::panic::PanicInfo;

/// Entry point for the RISC-V guest program.
#[no_mangle]
pub extern "C" fn _start() -> ! {
    // Compute Fibonacci(10) = 55
    let result = fibonacci(10);
    
    // Store result in a fixed memory location for verification
    // Address 0x80001000 is our "output" register
    unsafe {
        core::ptr::write_volatile(0x80001000 as *mut u32, result);
    }
    
    // Halt the CPU (ECALL with a0 = 0)
    halt();
}

/// Compute the n-th Fibonacci number iteratively.
#[inline(never)]
fn fibonacci(n: u32) -> u32 {
    if n == 0 {
        return 0;
    }
    if n == 1 {
        return 1;
    }
    
    let mut a = 0u32;
    let mut b = 1u32;
    
    for _ in 2..=n {
        let temp = a.wrapping_add(b);
        a = b;
        b = temp;
    }
    
    b
}

/// Halt the CPU using ECALL.
#[inline(never)]
fn halt() -> ! {
    unsafe {
        // ECALL instruction
        core::arch::asm!("ecall", options(noreturn));
    }
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    halt();
}


