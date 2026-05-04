# RV32IM Kernel Spec

This is the active RISC-V frontend contract for `neo-fold-next`.

The target is RV32IM: base 32-bit integer RISC-V plus the standard multiply and
divide extension. The project does not expose an RV64IM compatibility surface,
and proof-facing machine values are not represented as low/high limb pairs.

## Field Fit

The proving field is Goldilocks:

```text
p = 2^64 - 2^32 + 1
```

Every RV32 machine value is in `[0, 2^32)`, so each register value, program
counter, address, memory word, immediate word, and ALU result fits uniquely in a
single Goldilocks element. This is the reason the active target is RV32IM rather
than RV64IM: a 64-bit machine word would not be uniquely representable in one
Goldilocks element without an additional decomposition/range protocol.

`u64` remains valid only for non-machine metadata: counts, lengths, transcript
word serialization, digest limbs, field canonical values, and generic
SuperNeo/Spartan plumbing.

## Machine State

The architectural state is:

```text
pc      : u32
regs    : [u32; 32]
memory  : map<u32, u32>
halted  : bool
```

`x0` is hardwired to zero. Writes to `x0` are ignored architecturally.

Memory is byte-addressed but authenticated as aligned 4-byte words:

```text
RV32_WORD_BYTES = 4
backing_addr(addr) = addr & !0x3
```

Narrow loads and stores must remain inside one 4-byte backing word. Cross-word
`LB/LBU/LH/LHU/SB/SH` accesses are rejected.

## Supported ISA

The active decoder accepts the RV32IM opcodes below.

Arithmetic and logic:

```text
ADDI ADD SUB
ANDI AND ORI OR XORI XOR
SLTI SLT SLTIU SLTU
SLLI SLL SRLI SRL SRAI SRA
LUI AUIPC
```

Control:

```text
JAL JALR
BEQ BNE BLT BGE BLTU BGEU
ECALL FENCE
```

Memory:

```text
LB LBU LH LHU LW
SB SH SW
```

Multiply/divide:

```text
MUL MULH MULHSU MULHU
DIV DIVU REM REMU
```

The decoder rejects RV64-only opcodes, including:

```text
ADDIW ADDW SUBW
SLLIW SLLW SRLIW SRLW SRAIW SRAW
MULW DIVW DIVUW REMW REMUW
LWU LD SD
```

## Execution Semantics

All machine arithmetic wraps modulo `2^32`.

Signed operations interpret operands and results as two's-complement `i32`.
Unsigned operations interpret operands and results as `u32`.

Immediate fields are decoded according to the RV32 ISA and then stored in rows
as one 32-bit two's-complement word.

Shift amounts use RV32 masking:

```text
shamt = operand & 0x1f
```

RV64-only shift encodings are rejected by decode.

PC updates wrap modulo `2^32`:

```text
step_pc     = pc + 4
branch_pc   = pc + branch_imm
jal_pc      = pc + jal_imm
jalr_pc     = (rs1 + imm) & !1
```

All additions above are 32-bit wrapping additions.

## Memory Semantics

Loads:

```text
LB   sign_extend_8 (byte(addr))
LBU  zero_extend_8 (byte(addr))
LH   sign_extend_16(halfword(addr))
LHU  zero_extend_16(halfword(addr))
LW   word(backing_addr(addr))
```

Stores:

```text
SB   updates one byte in word(backing_addr(addr))
SH   updates two bytes in word(backing_addr(addr))
SW   writes word(backing_addr(addr))
```

`LH/LHU/SH` at byte offset `3` are invalid because they cross a 4-byte backing
word. `LW/SW` require byte offset `0`.

## Root Main-Lane Row

The root semantic row width is:

```text
RV32IM_ROOT_ROW_WIDTH = 27
```

The layout is one public `ONE` column, 11 single-field machine-value columns, 3
register index columns, and 12 boolean/control columns.

| Index | Column | Meaning |
| ---: | --- | --- |
| 0 | `ONE` | Constant one |
| 1 | `PC` | PC before the row |
| 2 | `PC_NEXT` | PC after the row |
| 3 | `RS1` | Source register 1 value |
| 4 | `RS2` | Source register 2 value |
| 5 | `RD_NEXT` | Destination value when the row writes `rd`, else zero |
| 6 | `IMM` | Decoded immediate as a 32-bit word |
| 7 | `ALU_OUT` | ALU/computation result |
| 8 | `STEP_PC` | `PC + 4` |
| 9 | `JUMP_TARGET` | Selected jump or taken-branch target, else zero |
| 10 | `MEM_ADDR` | Effective memory address for load/store rows, else zero |
| 11 | `MEM_VAL` | Loaded value or stored narrow/SW value, else zero |
| 12 | `RD_IDX` | Destination register index |
| 13 | `RS1_IDX` | Source register 1 index |
| 14 | `RS2_IDX` | Source register 2 index |
| 15 | `WritesAluToRd` | Row writes `ALU_OUT` to `rd` |
| 16 | `WritesMemToRd` | Row writes `MEM_VAL` to `rd` |
| 17 | `PreservesRd` | Row does not write `rd` |
| 18 | `IsJal` | Row is `JAL` |
| 19 | `IsJalr` | Row is `JALR` |
| 20 | `IsBranch` | Row is a branch |
| 21 | `BranchTaken` | Branch condition is true |
| 22 | `BranchTakenMux` | `IsBranch * BranchTaken` |
| 23 | `IsLoad` | Row is a load |
| 24 | `IsStore` | Row is a store |
| 25 | `UsesRs2` | Row consumes `rs2` |
| 26 | `AdvanceArchPc` | Commit row advances architectural PC |

There are no `*_LO` or `*_HI` columns in the root row.

## Root Row Constraints

The root row CCS enforces the row-local wiring that is common to all opcodes:

```text
boolean(control_columns)
BranchTakenMux = IsBranch * BranchTaken
WritesAluToRd + WritesMemToRd + PreservesRd = 1

WritesAluToRd * (RD_NEXT - ALU_OUT) = 0
WritesMemToRd * (RD_NEXT - MEM_VAL) = 0
PreservesRd   * RD_NEXT = 0

(IsJal + IsJalr + BranchTakenMux) * (PC_NEXT - JUMP_TARGET) = 0
(AdvanceArchPc - IsJal - IsJalr - BranchTakenMux) * (PC_NEXT - STEP_PC) = 0
(1 - AdvanceArchPc) * (PC_NEXT - PC) = 0

(1 - IsLoad - IsStore) * MEM_ADDR = 0
(1 - IsLoad - IsStore) * MEM_VAL  = 0
```

Opcode-specific arithmetic, memory extraction/blending, register timelines, RAM
timelines, and PC continuity are owned by the staged kernel surfaces below.

## Stage Ownership

Stage 1 owns decode and per-row semantic binding:

- instruction word, opcode, family, and row identity;
- selected register indices and register values;
- decoded immediate;
- ALU result, effective address, memory-before/after values;
- row flags such as `writes_rd`, `writes_ram`, `halted`, and commit/effect
  markers.

Stage 2 owns register and RAM timelines:

- register read events for `rs1` and `rs2`;
- register write events for architectural and virtual scratch registers;
- RAM read/write events over aligned 4-byte words;
- twist links that bind row-local Stage 1 values to timeline events.

Stage 3 owns PC adjacency and terminal control flow:

- first real row PC equals the public initial PC;
- each real row's `PC_NEXT` equals the next real row's `PC`;
- final row is a halted `ECALL`;
- final `PC_NEXT` equals the public final PC.

## Public Boundary

The public API is RV32IM-named. Examples:

```text
Rv32Opcode
Rv32State
prove_rv32im_public_proof
build_rv32im_nightstream_from_public_proof
nightstream::rv32im
```

No RV64IM compatibility aliases are part of the active API.

## Test Obligations

The test surface must cover:

- rejection of RV64-only decode paths (`ADDIW`, `LD`, `SD`, `LWU`, `MULW`);
- 32-bit arithmetic wraparound and RV32 shift masking;
- 4-byte backing-memory behavior for `SW/LW`;
- rejection of cross-word narrow memory access;
- root-row width `27` and absence of high-limb columns.
