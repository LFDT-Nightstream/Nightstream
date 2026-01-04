#![no_std]
#![no_main]

use core::arch::global_asm;
use core::panic::PanicInfo;

global_asm!(
    r#"
    .section .neo_start,"ax",@progbits
    .globl _start
_start:
    lui x1, 1
    addi x1, x1, 5
    addi x2, x0, 7
    add x3, x1, x2
    sw x3, 0x100(x0)
    lw x4, 0x100(x0)
    auipc x5, 0
    ecall
"#
);

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}
