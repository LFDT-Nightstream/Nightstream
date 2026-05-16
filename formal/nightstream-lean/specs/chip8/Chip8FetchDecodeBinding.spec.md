# Chip8FetchDecodeBinding Spec

## Purpose

- **What it is**: The theorem-facing Stage-1 binding contract for the final
  CHIP-8 kernel.
- **Key property**: `stage1Bound_of_fetchDecode`: a fixed ROM row and fetched
  opcode determine the exact authenticated Stage-1 decode tuple, ALU/branch
  lookup value, burst-equality output, and decode-handoff bits used by the
  main lane and Stage 2.
- **Protocol role**: This module owns the exact Stage-1 semantic objects that
  the kernel proves with the fetch, decode, ALU/branch, and Eq4 Shout channels,
  plus the explicit Stage-1 linkage batch back to `C_lane` and
  `C_decode_handoff`.

## Target Formulas

### Program and fetch surface

Let `Program` be the theorem-facing ROM object over the absolute CHIP-8
  11-bit word-address space.

Define:

$$
\mathrm{opcodeAt}(rom, pcWord)
$$

as the absolute ROM word fetched from the committed ROM table at the word
address `pcWord`.

This layer reasons over the final kernel address model:

- `PC` is an absolute CHIP-8 word address
- `NNN_ADDR` is the raw absolute CHIP-8 byte address
- `NNN_WORD` is the normalized absolute CHIP-8 word address

### Supported exact opcode families

The current kernel supports exactly:

- `LdImm`
- `AddImm`
- `Mov`
- `AddRegNoCarry`
- `SkipEqImm`
- `Jump`
- `LdI`
- `StoreRegs`
- `LoadRegs`

Every unsupported 16-bit word maps to `valid = 0`.

### Full decode tuple

The authenticated decode table outputs at least:

$$
\mathrm{DecodedStage1} :=
(
valid,
x_{dec}, y_{dec}, kk_{dec}, nnnAddr_{dec}, nnnWord_{dec},
writesLookupToX_{dec},
writesMemToX_{dec},
preservesX_{dec},
writesNnnToI_{dec},
isJump_{dec},
isBranch_{dec},
isMemOp_{dec},
isStore_{dec},
isLoad_{dec},
readsRam_{dec},
writesRam_{dec},
usesY_{dec},
lookupKind_{dec},
lhsSelector_{dec},
rhsSelector_{dec},
xBound_{dec}
).
$$

The Stage-1 owner must also expose the exact decode invariants:

- `valid = 1` on supported rows
- `(WritesLookupToX + WritesMemToX) * WritesNnnToI = 0`
- `IsJump * IsBranch = 0`
- `IsJump * IsMemOp = 0`
- `IsBranch * IsMemOp = 0`
- `isStore * isLoad = 0`
- `isStore + isLoad = isMemOp`
- `readsRam * writesRam = 0`
- `readsRam + writesRam = isMemOp`
- `writesRam = isStore`
- `readsRam = isLoad`
- `IsJump * (NNN_ADDR - NNN_WORD - NNN_WORD) = 0`

### Decode defaults and totality

The final kernel requires total decode outputs over the full 16-bit opcode
domain.

Key totality/default rules are:

- inactive decoded fields default to `0`
- if `lookupKind = NoLookup`, then:
  - `lhsSelector = ConstZero`
  - `rhsSelector = ConstZero`
  - `alu_lhs = 0`
  - `alu_rhs = 0`
  - `LOOKUP_OUTPUT = 0`
- if `IsMemOp = 0`, then:
  - `xBound = 0`
  - the Eq4 key is `(0, 0)`
  - `BURST_LAST = 0`
- if `usesY = 0`, then `Y_IDX = 0` and `REG_Y = 0`
- if `IsMemOp = 0`, then `X_IDX = x_dec`

### Stage-1 decoded row object

Define `DecodedRow` as the Stage-1 theorem-facing descriptor carrying:

- the exact supported opcode family
- the full decode tuple above
- the authenticated helper outputs:
  - `alu_lhs`
  - `alu_rhs`
  - `lookupValue = Val_alu(lookupKind, alu_lhs, alu_rhs)`
  - `burstEq = Eq4(X_IDX, xBound)`
- the lane-visible projections:
  - `KK`
  - `NNN_ADDR`
  - `NNN_WORD`
  - `WritesLookupToX`
  - `WritesMemToX`
  - `PreservesX`
  - `WritesNnnToI`
  - `IsJump`
  - `IsBranch`
  - `IsMemOp`
  - `X_IDX`
  - `Y_IDX`
  - `BURST_LAST`
- the handoff bits:
  - `handoff_uses_y`
  - `handoff_reads_ram`
  - `handoff_writes_ram`

### Fetch/decode binding

Define:

$$
\mathrm{FetchDecodeBound}(rom, pcWord, dec)
$$

to mean:

- `opcodeAt(rom, pcWord) = opcodeWord`
- the full decode table maps `opcodeWord` to the exact `DecodedRow dec`
- `valid = 1`

This is the theorem-facing statement that the fetched opcode determines the full
authenticated Stage-1 semantic object, not just a small parser record.

### Lookup and burst-equality binding

Define:

$$
\mathrm{AluLookupBound}(dec, regX, regY, lookupOut)
$$

to mean:

- `alu_lhs = Sel(lhsSelector; regX, regY, KK, 0)`
- `alu_rhs = Sel(rhsSelector; regX, regY, KK, 0)`
- `lookupOut = Val_alu(lookupKind, alu_lhs, alu_rhs)`

Define:

$$
\mathrm{BurstEqBound}(dec, burstLast)
$$

to mean:

- `burstEq = Eq4(X_IDX, xBound)`
- `burstLast = IsMemOp * burstEq`

### Decode-handoff binding

Define:

$$
\mathrm{DecodeHandoffBound}(dec, hUsesY, hReadsRam, hWritesRam)
$$

to mean:

- `hUsesY = usesY`
- `hReadsRam = readsRam`
- `hWritesRam = writesRam`

These are exactly the committed per-cycle `C_decode_handoff` columns consumed by
Stage 2.

### Stage-1 linkage batch

Define:

$$
\mathrm{Stage1LinkageBound}(dec, lane, handoff)
$$

to mean the explicit equality batch linking the authenticated Stage-1 objects
back to the lane and handoff columns at the Stage-1 cycle point:

- `KK = kk_dec`
- `NNN_ADDR = nnn_addr_dec`
- `NNN_WORD = nnn_word_dec`
- `WritesLookupToX = writes_lookup_to_x_dec`
- `WritesMemToX = writes_mem_to_x_dec`
- `PreservesX = preserves_x_dec`
- `WritesNnnToI = writes_nnn_to_i_dec`
- `IsJump = is_jump_dec`
- `IsBranch = is_branch_dec`
- `IsMemOp = is_memop_dec`
- `LOOKUP_OUTPUT = Val_alu(lookupKind_dec, alu_lhs, alu_rhs)`
- `BURST_LAST = IsMemOp * burstEq`
- `(1 - IsMemOp) * (X_IDX - x_dec) = 0`
- `usesY * (Y_IDX - y_dec) + (1 - usesY) * Y_IDX = 0`
- `handoff_uses_y = usesY`
- `handoff_reads_ram = readsRam`
- `handoff_writes_ram = writesRam`

### Exactness targets

The module must prove:

- fixed `(rom, pcWord)` determines at most one exact `DecodedRow`
- supported rows imply `valid = 1`
- unsupported rows imply decode failure or `valid = 0`
- `Jump` rows satisfy `NNN_ADDR = 2 * NNN_WORD`
- `NoLookup` rows force `LOOKUP_OUTPUT = 0`
- non-memory rows force `BURST_LAST = 0`
- the handoff bits are exactly the three committed Stage-1-to-Stage-2 columns

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/chip8-kernel.md`
- Anchors:
  - fetch channel
  - full-opcode decode channel
  - decode-handoff surface
  - ALU/branch lookup channel
  - burst-equality lookup channel
  - Stage-1 linkage batch

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/FetchDecodeBinding.lean` | Exact Stage-1 fetch/decode/lookup/handoff binding for the final CHIP-8 kernel |
| `Nightstream/Chip8/FetchDecodeBindingInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Semantic objects | `OpcodeId` | def | Definitional | Enumerates the exact supported opcode families |
| Semantic objects | `LookupKind` | def | Definitional | Enumerates `NoLookup`, `Identity`, `Equal8`, and `Add8Lo` |
| Semantic objects | `OperandSelector` | def | Definitional | Enumerates the Stage-1 ALU selector modes |
| Semantic objects | `DecodedStage1` | def | Definitional | Full authenticated decode-table tuple |
| Semantic objects | `DecodedRow` | def | Definitional | Full Stage-1 theorem-facing row descriptor |
| Semantic objects | `Program` | def | Definitional | Absolute ROM word-table object |
| Fetch | `opcodeAt` | def | Definitional | Fetches the absolute ROM word at `PC` |
| Binding | `FetchDecodeBound` | def | Definitional | ROM fetch determines the exact authenticated Stage-1 row descriptor |
| Binding | `AluLookupBound` | def | Definitional | Authenticated ALU/branch helper evaluation |
| Binding | `BurstEqBound` | def | Definitional | Authenticated Eq4 burst termination evaluation |
| Binding | `DecodeHandoffBound` | def | Definitional | Authenticated Stage-1-to-Stage-2 handoff bits |
| Binding | `Stage1LinkageBound` | def | Definitional | Explicit Stage-1 scalar linkage back to `C_lane` and `C_decode_handoff` |
| Theorem | `fetchDecodeBound_unique` | theorem | Theorem-Target | Fixed `(rom, pcWord)` determines at most one authenticated Stage-1 row |
| Theorem | `jumpAlignment_of_fetchDecodeBound` | theorem | Theorem-Target | `Jump` rows satisfy `NNN_ADDR = 2 * NNN_WORD` |
| Theorem | `noLookup_forces_zero_lookupOut` | theorem | Theorem-Target | `NoLookup` rows force `LOOKUP_OUTPUT = 0` |
| Theorem | `nonMemOp_forces_zero_burstLast` | theorem | Theorem-Target | Non-memory rows force `BURST_LAST = 0` |
| Theorem | `decodeHandoff_exact` | theorem | Theorem-Target | Stage-1 handoff bits are exactly `usesY`, `readsRam`, and `writesRam` |

## Proof Obligations

- The decode-table tuple must match the final kernel exactly, including
  `NNN_ADDR`, `NNN_WORD`, `lookupKind`, selectors, store/load bits, RAM bits,
  and `xBound`.
- `AddRegNoCarry` must be modeled as the actual supported exact family, not as
  full CHIP-8 `8xy4`.
- `FetchDecodeBound` must bind the exact Stage-1 row object to an actual ROM
  fetch at the absolute `PC` word address.
- The decode-handoff surface must be explicit; it may not be hidden inside a
  later Stage-2 assumption.
- The Stage-1 linkage batch must remain theorem-facing rather than prose-only.

## Assumption Ledger

- This module does not prove generic Shout address-correctness.
- This module does not prove generic Shout soundness.
- This module does not prove Stage-2 register/RAM history correctness.
- Public-table versus committed-table evaluator choices are owned by the opening
  boundary and evidence layers, not by this module.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/Routing.lean`
- **Downstream consumers**:
  - `Nightstream/Chip8/DecodeAddressBinding.lean`
  - `Nightstream/Chip8/WitnessMemoryBinding.lean`
  - `Nightstream/Chip8/EvidenceCoverage.lean`
  - `Nightstream/Chip8/StepComposition.lean`

## Implementation Plan

1. Define the exact Stage-1 semantic objects.
2. Define fetch, decode, helper-evaluation, and handoff bindings.
3. Define the explicit Stage-1 linkage batch surface.
4. Prove uniqueness and the exact kernel-specific invariants.

## Quality Expectations

- Keep this module exactly aligned with the final kernel spec.
- Keep Stage-1 decode, helper lookup, burst equality, and decode handoff in one
  owner.
- Make the linkage batch explicit instead of relying on informal “and they are
  equal” language.

## Acceptance Criteria

1. `lake build Nightstream.Chip8.FetchDecodeBinding` succeeds.
2. The theorem surface exposes the full decode tuple and the handoff bits.
3. The theorem surface uses `AddRegNoCarry`, not full CHIP-8 `8xy4`.
4. No `sorry`.

## Out of Scope

- generic Shout theorem proofs
- Stage-2 Twist memory proofs
- Stage-3 continuity or bridge binding
