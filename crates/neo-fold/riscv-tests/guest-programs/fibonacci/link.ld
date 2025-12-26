/* RISC-V linker script for guest programs */

ENTRY(_start)

MEMORY
{
    /* Program memory starts at 0x80000000 */
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 64K
}

SECTIONS
{
    . = ORIGIN(RAM);
    
    .text : {
        *(.text._start)
        *(.text*)
    } > RAM
    
    .rodata : {
        *(.rodata*)
    } > RAM
    
    .data : {
        *(.data*)
    } > RAM
    
    .bss : {
        *(.bss*)
    } > RAM
    
    /* Output area at fixed address */
    .output 0x80001000 : {
        KEEP(*(.output))
    }
    
    /DISCARD/ : {
        *(.eh_frame*)
        *(.note*)
    }
}


