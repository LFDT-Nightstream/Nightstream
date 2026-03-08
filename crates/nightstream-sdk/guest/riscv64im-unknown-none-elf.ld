ENTRY(_start);

SECTIONS
{
  . = 0x0;

  /* Keep the guest ABI addresses stable while giving RV64 guests a much larger stack/heap. */
  PROVIDE(_STACK_PTR = 0x800000);
  PROVIDE(_NEO_OUTPUT_ADDR = 0x100);
  PROVIDE(_NEO_INPUT_ADDR  = 0x104);
  PROVIDE(_NEO_HEAP_START = 0x2000);
  PROVIDE(_NEO_HEAP_END   = 0x400000);

  .neo_start : ALIGN(8)
  {
    KEEP(*(.neo_start))
    *(.text*)
  }

  .rodata : ALIGN(8)
  {
    *(.rodata*)
    *(.srodata*)
  }

  .data : ALIGN(8)
  {
    *(.data*)
    *(.sdata*)
  }

  .bss (NOLOAD) : ALIGN(8)
  {
    *(.bss*)
    *(.sbss*)
    *(COMMON)
  }

  /DISCARD/ :
  {
    *(.eh_frame*)
    *(.comment*)
  }
}
