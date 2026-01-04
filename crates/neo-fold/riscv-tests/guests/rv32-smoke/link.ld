ENTRY(_start);

SECTIONS
{
  . = 0x0;

  .neo_start : ALIGN(4)
  {
    KEEP(*(.neo_start))
  }

  .text : ALIGN(4)
  {
    *(.text*)
  }

  /DISCARD/ :
  {
    *(.eh_frame*)
    *(.comment*)
  }
}
