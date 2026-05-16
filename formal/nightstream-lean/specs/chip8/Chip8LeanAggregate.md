> Convenience aggregate only.
> The canonical sources of truth are the individual `Chip8*.spec.md` files in
> this directory and the paired `Nightstream/Chip8/*Interface.lean` theorem
> boundaries.

# /Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/specs/chip8/Chip8Routing.spec.md
# Chip8Routing Spec

## Purpose

- **What it is**: The theorem-facing local contract for the final 24-coordinate
  CHIP-8 main-lane row relation.
- **Key property**: `chip8RowLocalSound_of_constraints`: if one semantic row
  satisfies the 19 row-local equations, then the local outputs
  `REG_X_NEXT`, `I_NEXT`, `PC_NEXT`, and `RAM_ADDR` are forced by the active
  control columns.
- **Protocol role**: This is the smallest owner for the root main-lane local
  CCS relation. It does not own fetch, decode, lookup, memory-history,
  continuity, or bridge binding.

## Target Formulas

### Semantic row layout

One semantic row is a vector \(z : \mathrm{Fin}\;24 \to \mathbb{F}\) with named
coordinates:

| Index | Name |
|---|---|
| 0 | `ONE` |
| 1 | `PC` |
| 2 | `PC_NEXT` |
| 3 | `REG_X` |
| 4 | `REG_Y` |
| 5 | `REG_X_NEXT` |
| 6 | `I_REG` |
| 7 | `I_NEXT` |
| 8 | `KK` |
| 9 | `NNN_ADDR` |
| 10 | `NNN_WORD` |
| 11 | `MEM_VALUE` |
| 12 | `LOOKUP_OUTPUT` |
| 13 | `WritesLookupToX` |
| 14 | `WritesMemToX` |
| 15 | `PreservesX` |
| 16 | `WritesNnnToI` |
| 17 | `IsJump` |
| 18 | `IsBranch` |
| 19 | `IsMemOp` |
| 20 | `X_IDX` |
| 21 | `Y_IDX` |
| 22 | `BURST_LAST` |
| 23 | `RAM_ADDR` |

Define:

- `wf z := z₀ = 1`
- `routingFlags z := (z₁₃, z₁₄, z₁₅, z₁₆, z₁₇, z₁₈, z₁₉)`
- `controlBits z := (z₁₃, z₁₄, z₁₅, z₁₆, z₁₇, z₁₈, z₁₉, z₂₂)`

### Imported boundary data

This module does not own the semantic origin of:

- `REG_X`, `REG_Y`, `I_REG`, `MEM_VALUE`
- `LOOKUP_OUTPUT`
- `NNN_ADDR`, `NNN_WORD`
- `X_IDX`, `Y_IDX`, `BURST_LAST`

Those meanings are imported from Stage 1 and Stage 2 theorem owners. This
module proves only the local algebra implied by the row-local relation once
those columns are fixed.

### Supported routing classes

The routing-relevant seven-bit image is ordered as:

$$
(\mathrm{WritesLookupToX}, \mathrm{WritesMemToX}, \mathrm{PreservesX},
\mathrm{WritesNnnToI}, \mathrm{IsJump}, \mathrm{IsBranch}, \mathrm{IsMemOp}).
$$

For the current kernel, the intended exact-family image is:

- `(1,0,0,0,0,0,0)` for `LdImm`, `AddImm`, `Mov`, `AddRegNoCarry`
- `(0,0,1,0,0,1,0)` for `SkipEqImm`
- `(0,0,1,0,1,0,0)` for `Jump`
- `(0,0,1,1,0,0,0)` for `LdI`
- `(0,0,1,0,0,0,1)` for `StoreRegs`
- `(0,1,0,0,0,0,1)` for `LoadRegs`

`BURST_LAST` is not part of the decode image. It is a separately authenticated
Stage-1 output derived from the burst-equality channel.

### Row-local constraint set

Define `chip8RowLocalConstraints z` as the conjunction of the canonical 19
row-local equations.

- Boolean rows:

$$
\forall b \in \{
\mathrm{WritesLookupToX}, \mathrm{WritesMemToX}, \mathrm{PreservesX},
\mathrm{WritesNnnToI}, \mathrm{IsJump}, \mathrm{IsBranch}, \mathrm{IsMemOp},
\mathrm{BURST_LAST}
\},\;
b \cdot (b - 1) = 0.
$$

- X-lane partition:

$$
\mathrm{WritesLookupToX}
+ \mathrm{WritesMemToX}
+ \mathrm{PreservesX}
- 1 = 0.
$$

- X-lane routing:

$$
\mathrm{WritesLookupToX} \cdot (\mathrm{REG\_X\_NEXT} - \mathrm{LOOKUP\_OUTPUT}) = 0
$$

$$
\mathrm{WritesMemToX} \cdot (\mathrm{REG\_X\_NEXT} - \mathrm{MEM\_VALUE}) = 0
$$

$$
\mathrm{PreservesX} \cdot (\mathrm{REG\_X\_NEXT} - \mathrm{REG\_X}) = 0
$$

- I-lane routing:

$$
\mathrm{WritesNnnToI} \cdot (\mathrm{NNN\_ADDR} - \mathrm{I\_REG})
=
\mathrm{I\_NEXT} - \mathrm{I\_REG}.
$$

- PC routing:

$$
\mathrm{IsJump} \cdot (\mathrm{PC\_NEXT} - \mathrm{NNN\_WORD}) = 0
$$

$$
\mathrm{IsBranch} \cdot (\mathrm{PC\_NEXT} - \mathrm{PC} - \mathrm{ONE} - \mathrm{LOOKUP\_OUTPUT}) = 0
$$

$$
\mathrm{IsMemOp} \cdot (\mathrm{PC\_NEXT} - \mathrm{PC} - \mathrm{BURST\_LAST}) = 0
$$

$$
(\mathrm{ONE} - \mathrm{IsJump} - \mathrm{IsBranch} - \mathrm{IsMemOp})
\cdot
(\mathrm{PC\_NEXT} - \mathrm{PC} - \mathrm{ONE}) = 0.
$$

- RAM-address routing:

$$
\mathrm{IsMemOp} \cdot (\mathrm{RAM\_ADDR} - \mathrm{I\_REG} - \mathrm{X\_IDX}) = 0
$$

$$
(\mathrm{ONE} - \mathrm{IsMemOp}) \cdot \mathrm{RAM\_ADDR} = 0.
$$

### Local soundness target

Define `chip8RowLocalSound z` as the conjunction of:

- `WritesLookupToX = 1 -> REG_X_NEXT = LOOKUP_OUTPUT`
- `WritesMemToX = 1 -> REG_X_NEXT = MEM_VALUE`
- `PreservesX = 1 -> REG_X_NEXT = REG_X`
- `WritesNnnToI = 1 -> I_NEXT = NNN_ADDR`
- `WritesNnnToI = 0 -> I_NEXT = I_REG`
- `IsJump = 1 -> PC_NEXT = NNN_WORD`
- `wf z ∧ IsBranch = 1 -> PC_NEXT = PC + 1 + LOOKUP_OUTPUT`
- `IsMemOp = 1 -> PC_NEXT = PC + BURST_LAST`
- `wf z ∧ IsJump = 0 ∧ IsBranch = 0 ∧ IsMemOp = 0 -> PC_NEXT = PC + 1`
- `IsMemOp = 1 -> RAM_ADDR = I_REG + X_IDX`
- `wf z ∧ IsMemOp = 0 -> RAM_ADDR = 0`

The main theorem target is:

$$
\mathrm{chip8RowLocalConstraints}(z)
\Longrightarrow
\mathrm{chip8RowLocalSound}(z).
$$

### Algebraic completeness

For every intended routing-class tuple
\(b \in \mathrm{decodeImage}\), every burst bit
\(last \in \{0,1\}\), and every assignment of the imported source columns

$$
pc, regX, regY, i, kk, nnnAddr, nnnWord, mem, lookup, xIdx, yIdx \in \mathbb{F},
$$

there exists a semantic row \(z : \mathrm{Fin}\;24 \to \mathbb{F}\) such that:

$$
\mathrm{wf}(z)
\land
\mathrm{chip8RowLocalConstraints}(z)
\land
\mathrm{routingFlags}(z) = b
\land
z_{22} = last,
$$

with all imported source columns fixed to those chosen values and with
`REG_X_NEXT`, `I_NEXT`, `PC_NEXT`, and `RAM_ADDR` set by the routing equations.

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/chip8-kernel.md`
- Anchors:
  - main-lane witness layout
  - row-local R1CS
  - root main lane does not prove decode/memory by itself

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/Stage1/Routing.lean` | Local row-local routing lemma over the 24-coordinate CHIP-8 main lane |
| `Nightstream/Chip8/Stage1/RoutingInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Witness | `wf` | def | Definitional | Pins the public `ONE` coordinate to `1` |
| Witness | `routingFlags` | def | Definitional | Extracts the seven decode-driven routing bits |
| Witness | `controlBits` | def | Definitional | Extracts the eight local boolean control bits, including `BURST_LAST` |
| Constraints | `chip8RowLocalConstraints` | def | Definitional | The 19 canonical row-local equations |
| Behavior classes | `decodeImage` | def | Definitional | Intended seven-bit routing-class image for the supported exact families |
| Soundness target | `chip8RowLocalSound` | def | Definitional | Local output equalities forced by one satisfying row |
| Theorem | `chip8RowLocalSound_of_constraints` | theorem | Theorem-Target | Satisfying the 19 row-local equations implies local routing soundness |
| Theorem | `xRouting_oneHot` | theorem | Theorem-Target | Booleanity plus partition imply exactly one active X-routing bit when `2 ≠ 0` |
| Theorem | `iRouting_forced` | theorem | Theorem-Target | `WritesNnnToI` forces `I_NEXT` |
| Theorem | `pcRouting_forced` | theorem | Theorem-Target | `IsJump`, `IsBranch`, and `IsMemOp` force `PC_NEXT` on their active cases |
| Theorem | `ramAddrRouting_forced` | theorem | Theorem-Target | `IsMemOp` forces `RAM_ADDR` and inactive rows force `RAM_ADDR = 0` |
| Theorem | `rowWitness_exists_of_decodeImage` | theorem | Theorem-Target | Every intended routing class admits satisfying rows |

## Proof Obligations

- `chip8RowLocalSound_of_constraints` must derive the local equalities from the
  19 equations only.
- `xRouting_oneHot` must use only the boolean rows, the X partition, and the
  field-side condition `2 ≠ 0`.
- `pcRouting_forced` must state every `+1` conclusion under the explicit `wf`
  premise.
- The module must not prove decode-table, lookup-table, register-history, RAM
  history, or continuity facts.

## Assumption Ledger

- This module introduces no new cryptographic or protocol-level assumptions.
- Exact X-routing one-hotness additionally requires `2 ≠ 0` in the witness
  field.
- The semantic origin of `LOOKUP_OUTPUT`, `MEM_VALUE`, `NNN_ADDR`, `NNN_WORD`,
  `X_IDX`, `Y_IDX`, and `BURST_LAST` is external.
- Membership of `routingFlags z` in `decodeImage` is external and belongs to
  Stage 1 fetch/decode binding.
- Continuity, row binding, and bridge correctness are external to this module.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - local main-lane row design from `chip8-kernel.md`
- **Downstream consumers**:
  - `Nightstream/Chip8/Execution/StepComposition.lean`
  - later Stage-3 continuity and bridge-binding theorems

## Implementation Plan

1. Define the 24-coordinate semantic row projections.
2. Define the 19 row-local equations.
3. Prove the local routing implications.
4. Prove algebraic witness existence for every intended routing class.

## Quality Expectations

- Keep this module strictly row-local.
- Use the exact 24-coordinate row from the kernel spec.
- Keep imported semantic sources explicit instead of smuggling them into the
  local algebra.

## Acceptance Criteria

1. `lake build Nightstream.Chip8.Stage1.Routing` succeeds.
2. The theorem surface matches the 24-coordinate main-lane row, not the older
   reduced witness.
3. The module proves only local row algebra and nothing stage-global.
4. No `sorry`.

## Out of Scope

- fetch correctness
- decode correctness
- lookup semantics
- Twist-backed memory consistency
- continuity
- bridge binding into `PreparedStep`


# /Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/specs/chip8/Chip8FetchDecodeBinding.spec.md
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
| `Nightstream/Chip8/Stage1/FetchDecodeBinding.lean` | Exact Stage-1 fetch/decode/lookup/handoff binding for the final CHIP-8 kernel |
| `Nightstream/Chip8/Stage1/FetchDecodeBindingInterface.lean` | Theorem-facing re-export surface |

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
  - `Nightstream/Chip8/Stage1/Routing.lean`
- **Downstream consumers**:
  - `Nightstream/Chip8/Stage1/DecodeAddressBinding.lean`
  - `Nightstream/Chip8/Stage2/WitnessMemoryBinding.lean`
  - `Nightstream/Chip8/Stage2/EvidenceCoverage.lean`
  - `Nightstream/Chip8/Execution/StepComposition.lean`

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

1. `lake build Nightstream.Chip8.Stage1.FetchDecodeBinding` succeeds.
2. The theorem surface exposes the full decode tuple and the handoff bits.
3. The theorem surface uses `AddRegNoCarry`, not full CHIP-8 `8xy4`.
4. No `sorry`.

## Out of Scope

- generic Shout theorem proofs
- Stage-2 Twist memory proofs
- Stage-3 continuity or bridge binding


# /Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/specs/chip8/Chip8DecodeAddressBinding.spec.md
# Chip8DecodeAddressBinding Spec

## Purpose

- **What it is**: The theorem-facing CHIP-8-local binding contract for the
  exact Stage-1 and Stage-2 address families used by the final kernel.
- **Key property**: `kernelAddressBound_iff_familyProjection`: every
  kernel-visible address is exactly one of the authenticated family-specific
  projected addresses determined by the decoded row and lane values.
- **Protocol role**: This module is the local owner that turns authenticated
  decode outputs, lane projections, and sink-routing conventions into the exact
  numeric addresses consumed by Shout and Twist.

## Target Formulas

### Decoded row and lane projections

This layer enriches the authenticated Stage-1 row object with the exact lane
projections that later families must reference:

- `PC`
- `opcodeWord`
- `lookupKind`
- `alu_lhs`
- `alu_rhs`
- `X_IDX`
- `Y_IDX`
- `xBound`
- `RAM_ADDR`
- `usesY`
- `readsRam`
- `writesRam`
- `WritesLookupToX`
- `WritesMemToX`
- `WritesNnnToI`

The theorem-facing owner is therefore one decoded-and-projected per-row object,
not just a small parser record.

### Address families

The final kernel uses the following exact address families:

$$
\mathrm{AddressFamily}
:=
\mathrm{fetch}
\mid
\mathrm{decode}
\mid
\mathrm{alu}
\mid
\mathrm{eq4}
\mid
\mathrm{regRaX}
\mid
\mathrm{regRaY}
\mid
\mathrm{regRaI}
\mid
\mathrm{regWa}
\mid
\mathrm{ramRa}
\mid
\mathrm{ramWa}.
$$

### Concrete key spaces

The local address formulas are:

$$
\mathrm{FetchAddr} = PC
$$

$$
\mathrm{DecodeAddr} = opcodeWord
$$

$$
\mathrm{AluKey} = 2^{16} \cdot lookupKind + 2^8 \cdot alu\_lhs + alu\_rhs
$$

$$
\mathrm{Eq4Key} = 16 \cdot X\_IDX + xBound
$$

Register sink address:

$$
\bot_{reg} = 17
$$

RAM sink address:

$$
\bot_{ram} = 4096
$$

### Stage-2 raw-address identities

The Stage-2 family projections must satisfy the exact raw-address identities:

$$
\Sigma_a RegRaX(a,j)\cdot a = X\_IDX
$$

$$
\Sigma_a RegRaY(a,j)\cdot a = usesY \cdot Y\_IDX + (1 - usesY)\cdot 17
$$

$$
\Sigma_a RegRaI(a,j)\cdot a = 16
$$

$$
\Sigma_a RegWa(a,j)\cdot a
=
(WritesLookupToX + WritesMemToX)\cdot X\_IDX
+ WritesNnnToI \cdot 16
+ (1 - WritesLookupToX - WritesMemToX - WritesNnnToI)\cdot 17
$$

$$
\Sigma_a RamRa(a,j)\cdot a
= readsRam \cdot RAM\_ADDR + (1 - readsRam)\cdot 4096
$$

$$
\Sigma_a RamWa(a,j)\cdot a
= writesRam \cdot RAM\_ADDR + (1 - writesRam)\cdot 4096
$$

These identities are in addition to any support/unmap relation. They are what
force inactive rows to the sink addresses rather than merely unmapping to zero.

### Family-specific projected addresses

Define `projectedAddressAt(dec, fam)` to be the exact numeric address or key
required by that family on the current row:

- `fetch -> PC`
- `decode -> opcodeWord`
- `alu -> AluKey`
- `eq4 -> Eq4Key`
- `regRaX -> X_IDX`
- `regRaY -> usesY * Y_IDX + (1 - usesY) * 17`
- `regRaI -> 16`
- `regWa -> (WritesLookupToX + WritesMemToX) * X_IDX + WritesNnnToI * 16 + (1 - WritesLookupToX - WritesMemToX - WritesNnnToI) * 17`
- `ramRa -> readsRam * RAM_ADDR + (1 - readsRam) * 4096`
- `ramWa -> writesRam * RAM_ADDR + (1 - writesRam) * 4096`

### Kernel address binding

Define:

$$
\mathrm{KernelAddressBoundAt}(dec, fam, addr)
\iff
\mathrm{projectedAddressAt}(dec, fam) = addr.
$$

$$
\mathrm{KernelAddressBound}(dec, addr)
\iff
\exists fam,\; \mathrm{KernelAddressBoundAt}(dec, fam, addr).
$$

The exactness target is:

$$
\mathrm{KernelAddressBound}(dec, addr)
\iff
\exists fam,\; \mathrm{projectedAddressAt}(dec, fam) = addr.
$$

### Role-exactness targets

The module must also prove the truthful role-exactness facts:

- `aluAddress_requires_lookup_family` under the explicit non-`NoLookup`
  hypothesis
- `ramReadAddress_requires_readsRam` for non-sink `RamRa` addresses
- `ramWriteAddress_requires_writesRam` for non-sink `RamWa` addresses
- `regYAddress_uses_sink_iff_not_usesY`
- `regWriteAddress_uses_sink_iff_no_lane_write` under the explicit active-index
  bound needed to rule out malformed burst rows hitting the sink numerically

These are the CHIP-8-local theorems that stop Shout/Twist address claims from
floating free of the intended row semantics.

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/chip8-kernel.md`
- Anchors:
  - Stage-1 address-correctness obligations
  - Stage-2 address-correctness obligations
  - sink-routing equations for inactive rows

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/Stage1/DecodeAddressBinding.lean` | Exact family-specific address projection and sink-routing theorems |
| `Nightstream/Chip8/Stage1/DecodeAddressBindingInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Semantic objects | `AddressFamily` | def | Definitional | Enumerates the exact Stage-1 and Stage-2 address families |
| Semantic objects | `DecodedRow` | def | Definitional | Authenticated decoded row together with lane projections needed for address binding |
| Shape | `projectedAddressAt` | def | Definitional | Returns the exact numeric address or key for one family |
| Binding | `KernelAddressBoundAt` | def | Definitional | Family-local CHIP-8 address binding |
| Binding | `KernelAddressBound` | def | Definitional | Family-erased CHIP-8 address binding used by later modules |
| Theorem | `kernelAddressBound_iff_familyProjection` | theorem | Theorem-Target | Kernel-bound addresses are exactly the family projections |
| Theorem | `aluAddress_requires_lookup_family` | theorem | Theorem-Target | Non-`NoLookup` ALU rows imply lookup-using opcode families |
| Theorem | `ramReadAddress_requires_readsRam` | theorem | Theorem-Target | Non-sink `RamRa` addresses imply authenticated RAM-read rows |
| Theorem | `ramWriteAddress_requires_writesRam` | theorem | Theorem-Target | Non-sink `RamWa` addresses imply authenticated RAM-write rows |
| Theorem | `regYAddress_uses_sink_iff_not_usesY` | theorem | Theorem-Target | `RegRaY` uses the sink iff `usesY = 0` |
| Theorem | `regWriteAddress_uses_sink_iff_no_lane_write` | theorem | Theorem-Target | `RegWa` uses the sink iff the row writes neither X nor I, assuming the active `X_IDX` stays in the real register range |

## Proof Obligations

- The address-family set must match the final kernel exactly.
- The ALU and Eq4 key-flattening formulas must match the kernel exactly.
- Sink-routing equations must remain explicit; support/unmap relations are not a
  substitute.
- `KernelAddressBound` must be defined only through exact family-specific
  projections.
- This module must not claim that Eq4 addresses alone imply `IsMemOp = 1`; the
  final kernel enforces inactivity only through `BURST_LAST = IsMemOp * burst_eq`.

## Assumption Ledger

- This module assumes the Stage-1 decoded row object is authenticated by
  `Chip8FetchDecodeBinding`.
- This module does not prove Booleanity or Hamming-weight of the one-hot
  families themselves; it proves the CHIP-8-local numeric address meaning once
  those validity facts are imported.
- This module does not prove Twist history soundness.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/Stage1/FetchDecodeBinding.lean`
- **Downstream consumers**:
  - `Nightstream/Chip8/Stage2/WitnessMemoryBinding.lean`
  - `Nightstream/Chip8/Stage2/EvidenceCoverage.lean`
  - `Nightstream/Chip8/Execution/StepComposition.lean`

## Implementation Plan

1. Define the exact address-family enumerator.
2. Define the concrete Stage-1 and Stage-2 projected-address formulas.
3. Define `KernelAddressBoundAt` and `KernelAddressBound`.
4. Prove the exact sink-routing and role-exactness theorems.

## Quality Expectations

- Keep the module exact and family-specific.
- Do not hide sink-routing behind abstract “inactive means harmless” prose.
- Make the numeric key/address formulas mechanically obvious.

## Acceptance Criteria

1. `lake build Nightstream.Chip8.Stage1.DecodeAddressBinding` succeeds.
2. The theorem surface covers all exact Stage-1 and Stage-2 families from the
   final kernel.
3. No unnamed address source remains.
4. No `sorry`.

## Out of Scope

- generic Shout proofs
- generic Twist proofs
- continuity
- bridge binding


# /Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/specs/chip8/Chip8WitnessMemoryBinding.spec.md
# Chip8WitnessMemoryBinding Spec

## Purpose

- **What it is**: The theorem-facing Stage-2 binding contract from semantic
  CHIP-8 state to the final kernel's lane values, register/RAM Twist ports,
  memory-transfer values, RAF support equations, and initial-state anchors.
- **Key property**: `memoryBound_memValue_total`: when the local Stage-2 memory
  binding holds, the lane value `MEM_VALUE` is total and equals exactly the
  semantic value required by the authenticated row kind.
- **Protocol role**: This is the local owner for the CHIP-8-specific semantic
  meaning of Stage-2 register and RAM claims. It does not re-prove generic
  Twist soundness.

## Target Formulas

### Semantic state

The local Stage-2 layer reasons about semantic state objects:

- `pre`: machine state before the current row
- `post`: machine state after the current row
- `init`: authenticated initial machine state for the chunk

For the current kernel, the state carrier must expose at least:

- `pc_word`
- `i`
- `V[0..15]`
- `RAM[0..4095]`

### Lane row binding

This layer reasons over the final 24-coordinate semantic row, not the older
reduced routing row.

Define:

$$
\mathrm{WitnessBinds}(pre, post, dec, z)
$$

to mean that the lane row `z` carries:

- `PC = pre.pc_word`
- `PC_NEXT = post.pc_word`
- `REG_X = primaryValue(pre, dec)`
- `REG_Y = secondaryValue(pre, dec)`
- `REG_X_NEXT = primaryValue(post, dec)`
- `I_REG = pre.i`
- `I_NEXT = post.i`
- `KK = dec.kk`
- `NNN_ADDR = dec.nnnAddr`
- `NNN_WORD = dec.nnnWord`
- `MEM_VALUE = memValueOf(pre, post, dec)`
- `X_IDX = xIndexOf(dec)`
- `Y_IDX = yIndexOf(dec)`
- `BURST_LAST = burstLastOf(dec)`
- `RAM_ADDR = ramAddrOf(pre, dec)`

The control columns and `LOOKUP_OUTPUT` are imported from Stage 1; this module
owns the memory-derived lane values.

### Memory-transfer value

Define the exact semantic transfer value:

$$
\mathrm{memValueOf}(pre, post, dec)
:=
\begin{cases}
REG\_X(pre,dec) & \text{if } writesRam(dec) = 1, \\
REG\_X(post,dec) & \text{if } readsRam(dec) = 1, \\
0 & \text{otherwise.}
\end{cases}
$$

The Stage-2 owner must prove the lane totality rule:

- if `writesRam = 1`, then `MEM_VALUE = REG_X`
- if `readsRam = 1`, then `MEM_VALUE` is the authenticated RAM-read value
- otherwise `MEM_VALUE = 0`

### Register subsystem

The register-file domain is:

- slots `0..15` for `V[0]..V[15]`
- slot `16` for `I`
- slot `17` for `⊥_reg`

Define the Stage-2 register objects:

- `RegInc`
- `RegRaX`
- `RegRaY`
- `RegRaI`
- `RegWa`
- virtual `RegVal`

The CHIP-8-local register-port semantics are:

- `RegRaX` reads slot `X_IDX`
- `RegRaY` reads slot `Y_IDX` when `usesY = 1`, else `⊥_reg`
- `RegRaI` reads slot `16`
- `RegWa` writes:
  - slot `X_IDX` when `WritesLookupToX + WritesMemToX = 1`
  - slot `16` when `WritesNnnToI = 1`
  - slot `⊥_reg` otherwise

Sink semantics:

$$
\mathrm{RegVal}(\bot_{reg},0) = 0
$$

$$
\mathrm{RegVal}(\bot_{reg},j+1) = \mathrm{RegVal}(\bot_{reg},j)
$$

$$
\mathrm{RegInc}(j) = 0 \text{ whenever } RegWa \text{ points to } \bot_{reg}
$$

### RAM subsystem

The RAM domain is:

- slots `0..4095` for RAM
- slot `4096` for `⊥_ram`

Define the Stage-2 RAM objects:

- `RamInc`
- `RamRa`
- `RamWa`
- virtual `RamVal`

The CHIP-8-local RAM-port semantics are:

- `RamRa` points to `RAM_ADDR` when `readsRam = 1`, else `⊥_ram`
- `RamWa` points to `RAM_ADDR` when `writesRam = 1`, else `⊥_ram`

Sink semantics:

$$
\mathrm{RamVal}(\bot_{ram},0) = 0
$$

$$
\mathrm{RamVal}(\bot_{ram},j+1) = \mathrm{RamVal}(\bot_{ram},j)
$$

$$
\mathrm{RamInc}(j) = 0 \text{ whenever } RamWa \text{ points to } \bot_{ram}
$$

### Stage-2 lane linkage

Define:

$$
\mathrm{Stage2LaneLinkBound}(pre, post, dec, z)
$$

to mean the exact scalar equalities consumed by the register/RAM subclaims:

- `rv_x = REG_X`
- `rv_y = REG_Y`
- `rv_i = I_REG`
- `wv_reg = (WritesLookupToX + WritesMemToX) * REG_X_NEXT + WritesNnnToI * I_NEXT`
- `rv_ram = readsRam * MEM_VALUE`
- `wv_ram = writesRam * MEM_VALUE`

This is the theorem-facing content of the Stage-2 linkage batch; it is not a
free consequence of generic Twist theory.

### RAM RAF support

Define:

$$
\mathrm{RamRafBound}(dec, z)
$$

to mean the exact support equations:

$$
\Sigma_a ra\_read(a)\cdot unmap\_{chip8}(a) = readsRam \cdot RAM\_ADDR
$$

$$
\Sigma_a ra\_write(a)\cdot unmap\_{chip8}(a) = writesRam \cdot RAM\_ADDR
$$

where the active RAM domain is exactly `0..4095` and `⊥_ram = 4096`.

### Initial-state authentication

Define:

$$
\mathrm{InitialStateBound}(init)
$$

to mean:

- `RegVal(a,0) = init_reg[a]` for `a in {0..16}`
- `RegVal(17,0) = 0`
- `RamVal(a,0) = init_ram[a]` for `a in {0..4095}`
- `RamVal(4096,0) = 0`

This is the final kernel's chosen non-zero-initialization route. It is not a
synthetic preload-write encoding.

### Memory bound

Define:

$$
\mathrm{MemoryBound}(pre, post, init, dec, z)
$$

to mean the conjunction of:

- `WitnessBinds(pre, post, dec, z)` for the memory-derived lane values
- register-port semantics
- RAM-port semantics
- `Stage2LaneLinkBound(pre, post, dec, z)`
- `RamRafBound(dec, z)`
- `InitialStateBound(init)`

This is the local theorem surface that later composition layers import as the
CHIP-8 meaning of the Stage-2 subsystem.

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/chip8-kernel.md`
- Anchors:
  - register-file domain and ports
  - register-file lane linkage
  - RAM domain and ports
  - RAM lane linkage
  - RAM RAF support relation
  - initial-state authentication

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/Stage2/WitnessMemoryBinding.lean` | CHIP-8-local semantic meaning of Stage-2 lane values, Twist ports, RAF, and initialization |
| `Nightstream/Chip8/Stage2/WitnessMemoryBindingInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Semantic objects | `MachineState` | def | Definitional | CHIP-8 semantic state carrier for Stage 2 |
| Semantic objects | `InitialState` | def | Definitional | Authenticated chunk-initial state carrier |
| Binding | `WitnessBinds` | def | Definitional | Binds memory-derived lane values to semantic state and decoded row |
| Binding | `memValueOf` | def | Definitional | Exact semantic source of `MEM_VALUE` |
| Register | `RegisterPortBound` | def | Definitional | Exact CHIP-8-local meaning of `RegRaX`, `RegRaY`, `RegRaI`, and `RegWa` |
| RAM | `RamPortBound` | def | Definitional | Exact CHIP-8-local meaning of `RamRa` and `RamWa` |
| Linkage | `Stage2LaneLinkBound` | def | Definitional | Exact scalar linkage between lane columns and Stage-2 read/write claims |
| RAF | `RamRafBound` | def | Definitional | Exact CHIP-8 RAM RAF support equations |
| Init | `InitialStateBound` | def | Definitional | Exact initial register/RAM base case for the virtual `Val` surfaces |
| Binding | `MemoryBound` | def | Definitional | Complete CHIP-8-local Stage-2 semantic binding bundle |
| Theorem | `memoryBound_memValue_total` | theorem | Theorem-Target | `MEM_VALUE` obeys the exact totality rule |
| Theorem | `registerPorts_exact` | theorem | Theorem-Target | Register ports realize exactly the intended CHIP-8 reads/writes/sink rows |
| Theorem | `ramPorts_exact` | theorem | Theorem-Target | RAM ports realize exactly the intended CHIP-8 reads/writes/sink rows |
| Theorem | `ramRaf_tracks_laneAddress` | theorem | Theorem-Target | RAM RAF support ties the committed address family back to `RAM_ADDR` |
| Theorem | `initialStateBound_exact` | theorem | Theorem-Target | The Stage-2 base case is the authenticated public initialization surface |

## Proof Obligations

- This module must target the final 24-coordinate lane row, not the older
  reduced routing witness.
- Sink semantics for `⊥_reg` and `⊥_ram` must remain explicit.
- `MemoryBound` must expose the lane-linkage equalities and the RAF equations as
  theorem-facing objects.
- Initial-state authentication must follow the final kernel's non-zero-init
  route directly, not a preload-write reduction.

## Assumption Ledger

- This module does not re-prove generic Twist read/write or `Val` soundness.
- This module does not re-prove generic address-correctness for the one-hot
  families.
- This module does not prove Stage-1 lookup correctness.
- This module does not prove Stage-3 continuity or bridge binding.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/Stage1/FetchDecodeBinding.lean`
  - `Nightstream/Chip8/Stage1/DecodeAddressBinding.lean`
  - `Nightstream/Chip8/Stage1/Routing.lean`
- **Downstream consumers**:
  - `Nightstream/Chip8/Stage2/EvidenceCoverage.lean`
  - `Nightstream/Chip8/Execution/StepComposition.lean`
  - later Rust-refinement theorems for Stage-2 proof objects

## Implementation Plan

1. Define the semantic lane-value and port objects.
2. Define `memValueOf`, register/RAM port semantics, RAF, and initialization.
3. Define the bundled `MemoryBound`.
4. Prove the exact totality, sink-routing, RAF, and initialization lemmas.

## Quality Expectations

- Keep Stage-2 ownership explicit and separate from generic Twist theorems.
- Use the exact sink-routing and initialization rules from the final kernel.
- Do not collapse Stage 2 into an opaque “memory passed” predicate.

## Acceptance Criteria

1. `lake build Nightstream.Chip8.Stage2.WitnessMemoryBinding` succeeds.
2. The theorem surface matches the final Stage-2 register/RAM design.
3. `MEM_VALUE` totality, RAF support, and initialization are explicit.
4. No `sorry`.

## Out of Scope

- generic Twist theorem proofs
- Stage-1 Shout proofs
- Stage-3 continuity and bridge binding


# /Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/specs/chip8/Chip8ExecutionSemantics.spec.md
# Chip8ExecutionSemantics Spec

## Purpose

- **What it is**: The shared semantic-state and authenticated-trace contract for
  the exact supported CHIP-8 kernel.
- **Key property**: it owns the definitions of row semantics,
  whole-instruction semantics, authenticated execution frames, and
  chunk-local execution traces that are shared by `Chip8StepComposition` and
  `Chip8BurstSession`.
- **Protocol role**: This module is the stable semantic owner below the
  composition theorem and below the decomposed burst theorem. It does not prove
  fetch/decode or memory soundness; it defines the semantic targets those
  modules discharge.

## Target Formulas

### Row semantics

Define:

$$
\mathrm{MicrostepCorrect}(rom,\sigma,dec,pre,post)
$$

for the exact supported 9-family kernel:

- `LdImm`
- `AddImm`
- `Mov`
- `AddRegNoCarry`
- `SkipEqImm`
- `Jump`
- `LdI`
- `StoreRegs`
- `LoadRegs`

This is the exact semantic target later proved from authenticated row-local
bounds by `Chip8StepComposition`.

### Whole-instruction semantics

Define:

$$
\mathrm{InstructionCorrect}(rom,\sigma,dec,pre,post)
$$

so that:

- non-burst instructions are definitionally the same as `MicrostepCorrect`
- `StoreRegs` / `LoadRegs` are macro-instruction semantics with final
  `PC_NEXT = PC + 1`

### Authenticated execution traces

Define an authenticated execution frame:

$$
\mathrm{ExecutionFrame} := (dec, pre, post, z)
$$

and the row-backed predicate:

$$
\mathrm{ExecutionFrameBound}(rom,\sigma,frame)
$$

meaning:

- the 24-coordinate row is well-formed
- the row binds to the semantic pre/post state and decoded row
- the row is semantically microstep-correct

Define:

$$
\mathrm{ExecutionCorrect}(rom,\sigma,init,trace)
$$

to mean:

- frames are locally chained
- every frame satisfies `ExecutionFrameBound`
- the trace satisfies the authenticated Stage-3 continuity relation
- the first frame matches the authenticated initial state and start-boundary law

### Prepared-step export

Define:

$$
\mathrm{PreparedStepTraceBound}(trace,preparedSteps)
$$

to mean the exported prepared steps are exactly the Stage-3 bridge images of
the authenticated row trace.

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/chip8-kernel.md`
- Anchors:
  - exact supported 9-family kernel semantics
  - chunk-local continuity / prepared-step export
  - macro-instruction semantics for `Fx55` / `Fx65`

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/Execution/ExecutionSemantics.lean` | Shared semantic objects and trace relations for the supported CHIP-8 kernel |
| `Nightstream/Chip8/Execution/ExecutionSemanticsInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Semantics | `MicrostepCorrect` | def | Definitional | Exact row semantics for the supported kernel |
| Semantics | `InstructionCorrect` | def | Definitional | Exact whole-instruction semantics |
| Trace | `ExecutionFrame` | def | Definitional | Row-backed semantic trace element |
| Trace | `ExecutionFrameBound` | def | Definitional | Authenticated row-backed microstep witness |
| Trace | `ExecutionCorrect` | def | Definitional | Authenticated chunk-local semantic execution trace |
| Bridge | `PreparedStepTraceBound` | def | Definitional | Prepared steps are exactly the Stage-3 images of the row trace |
| Theorem | `instructionCorrect_of_nonBurstMicrostep` | theorem | Theorem-Target | Non-burst microstep correctness implies whole-instruction correctness |
| Theorem | `executionCorrect_of_trace` | theorem | Theorem-Target | Chaining plus frame bounds plus continuity imply execution correctness |
| Theorem | `preparedStepTraceBound_of_execution` | theorem | Theorem-Target | Correct execution yields exact prepared-step export |

## Proof Obligations

- `MicrostepCorrect` must stay exact to the current supported kernel.
- `InstructionCorrect` must preserve the distinction between row-local memory
  prefix rows and whole `StoreRegs` / `LoadRegs` instructions.
- `ExecutionCorrect` must remain explicitly continuity-aware.

## Assumption Ledger

- Fetch/decode, lookup, and Stage-2 memory correctness are not proved here.
- Those semantic facts are proved in `Chip8StepComposition`.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/Stage2/WitnessMemoryBinding.lean`
  - `Nightstream/Chip8/Stage3/ContinuityBridge.lean`
- **Downstream consumers**:
  - `Nightstream/Chip8/Execution/StepComposition.lean`
  - `Nightstream/Chip8/Execution/BurstSession.lean`

## Implementation Plan

1. Define the shared semantic objects.
2. Define the authenticated execution-trace predicates.
3. Prove the generic non-burst, trace, and prepared-step export lemmas.

## Quality Expectations

- Keep this module semantic and shared.
- Do not move fetch/decode or memory-proof ownership here.
- Keep burst whole-instruction closure in `Chip8BurstSession`.

## Acceptance Criteria

1. `lake build Nightstream.Chip8.Execution.ExecutionSemantics` succeeds.
2. `Chip8StepComposition` and `Chip8BurstSession` consume the same semantic owner.
3. No `sorry`.

## Out of Scope

- proving row-local semantics from authenticated bounds
- proving burst-session schedule correctness
- proving claim coverage or PCS binding


# /Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/specs/chip8/Chip8RomScheduleBinding.spec.md
# Chip8KernelInputBinding Spec

## Purpose

- **What it is**: The theorem-facing authenticated public-input binding
  contract for the final CHIP-8 kernel.
- **Key property**: `kernelPublicInputsBound_of_authenticatedInputs`: the
  public metadata fixed in `meta_pub` determines the exact ROM table,
  program-shape metadata, padding row metadata, root-encoding parameters, and
  authenticated initial machine state used by the kernel.
- **Protocol role**: This is the layer that makes the kernel's public-input
  bundle theorem-facing instead of leaving it only inside Rust structs or
  transcript prose.

## Target Formulas

### Public kernel input bundle

The final kernel public boundary includes at least:

- `vm_spec`
- `public_program_image`
- `initial_state`
- `transcript_seed`
- the `meta_pub` fields absorbed into `root0`

The theorem-facing owner packages those into one authenticated public-input
object.

### Public metadata bindings

Let the public metadata package include at least:

- program image digest
- table digests
- trace length `N`
- `program_word_count`
- `program_base_addr`
- `pad_pc_word`
- initial-state digest(s)
- root-encoding / root-protocol version identifiers

Define:

$$
\mathrm{ProgramDigestBound}(meta, romTable)
$$

$$
\mathrm{ProgramShapeBound}(meta, romTable)
$$

$$
\mathrm{PadRowMetadataBound}(meta)
$$

$$
\mathrm{InitialStateDigestBound}(meta, init)
$$

$$
\mathrm{RootParamsBound}(meta, vmSpec)
$$

These express, respectively, that:

- the public program digest matches the exact committed absolute ROM table
- `program_word_count` and `program_base_addr` match the loaded absolute ROM
  interval
- `pad_pc_word` is the exact public self-loop padding address
- the public initial-state digest(s) match the exact authenticated initial
  register/RAM state
- `vm_spec` fixes the canonical root witness-encoding parameters used by
  `RootEncode`

### Authenticated public inputs

Define:

$$
\mathrm{AuthenticatedProgramImage}(publicInput, romTable)
$$

$$
\mathrm{AuthenticatedInitialState}(publicInput, init)
$$

$$
\mathrm{AuthenticatedKernelMeta}(publicInput, meta)
$$

to package the exact authenticated witnesses used to derive the bound
relations.

### Bundled public-input theorem

Define:

$$
\mathrm{KernelPublicInputsBound}(publicInput, meta, romTable, init)
$$

to mean the conjunction of:

- `ProgramDigestBound(meta, romTable)`
- `ProgramShapeBound(meta, romTable)`
- `PadRowMetadataBound(meta)`
- `InitialStateDigestBound(meta, init)`
- `RootParamsBound(meta, publicInput.vmSpec)`

The target theorem is:

$$
\mathrm{AuthenticatedProgramImage}
\land
\mathrm{AuthenticatedInitialState}
\land
\mathrm{AuthenticatedKernelMeta}
\Longrightarrow
\mathrm{KernelPublicInputsBound}.
$$

### Transport theorems

If the relevant digest/equality functions are injective, then shared public
metadata determines a unique ROM table or initial state.

These transport theorems let later Stage-1, Stage-2, and Stage-3 modules move
facts across a fixed public input bundle without re-proving the public bindings.

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/chip8-kernel.md`
- Anchors:
  - commitment bundle / `meta_pub`
  - fixed hypercube domains and absolute ROM addressing
  - public pad-row rule
  - Rust-facing kernel boundary
  - prepared-step construction / canonical root parameters

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/Kernel/RomScheduleBinding.lean` | Authenticated public-input binding theorems for the final CHIP-8 kernel |
| `Nightstream/Chip8/Kernel/RomScheduleBindingInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Public inputs | `KernelPublicInput` | def | Definitional | The theorem-facing public input bundle of the final kernel |
| Bindings | `ProgramDigestBound` | def | Definitional | Public metadata matches the exact committed ROM table |
| Bindings | `ProgramShapeBound` | def | Definitional | `program_word_count` and `program_base_addr` match the actual absolute ROM layout |
| Bindings | `PadRowMetadataBound` | def | Definitional | `pad_pc_word` and the public pad-row metadata are fixed exactly |
| Bindings | `InitialStateDigestBound` | def | Definitional | Public initial-state digest(s) match the exact initial register/RAM state |
| Bindings | `RootParamsBound` | def | Definitional | `vm_spec` fixes the canonical root encoding parameters |
| Evidence | `AuthenticatedProgramImage` | def | Definitional | Explicit authenticated ROM-table witness |
| Evidence | `AuthenticatedInitialState` | def | Definitional | Explicit authenticated initial-state witness |
| Evidence | `AuthenticatedKernelMeta` | def | Definitional | Explicit authenticated `meta_pub` witness |
| Bundle | `KernelPublicInputsBound` | def | Definitional | Exact bundled theorem surface for the public kernel inputs |
| Theorem | `kernelPublicInputsBound_of_authenticatedInputs` | theorem | Theorem-Target | Authenticated public inputs imply the bundled public-input bounds |
| Theorem | `romTable_eq_of_sharedMeta` | theorem | Theorem-Target | Shared authenticated public metadata determines one exact ROM table |
| Theorem | `initialState_eq_of_sharedMeta` | theorem | Theorem-Target | Shared authenticated public metadata determines one exact initial state |

## Proof Obligations

- The public-input layer must make `meta_pub` theorem-facing.
- The exact pad-row metadata and initial-state digest bindings must be explicit.
- The root witness-encoding parameters must be fixed by the public boundary, not
  left as hidden prover-chosen values.
- Shared-public-input transport theorems must require only the exact needed
  injectivity hypotheses.

## Assumption Ledger

- Poseidon2 security and collision resistance are external to this module.
- This module does not prove Stage-1, Stage-2, or Stage-3 semantic facts.
- This module binds public objects to semantic ones; it does not re-prove the
  PCS commitments themselves.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - final kernel public boundary from `chip8-kernel.md`
- **Downstream consumers**:
  - `Nightstream/Chip8/Stage1/FetchDecodeBinding.lean`
  - `Nightstream/Chip8/Stage2/WitnessMemoryBinding.lean`
  - `Nightstream/Chip8/Stage3/ContinuityBridge.lean`
  - `Nightstream/Chip8/Stage2/EvidenceCoverage.lean`

## Implementation Plan

1. Define the theorem-facing public input and metadata objects.
2. Define the exact bound relations for ROM shape, padding, initialization, and
   root parameters.
3. Define the authenticated witness structures.
4. Prove the bundled public-input theorem and the transport lemmas.

## Quality Expectations

- Keep the module small and ownership-specific.
- Make `meta_pub` explicit.
- Separate authenticated public inputs from later semantic extraction.

## Acceptance Criteria

1. `lake build Nightstream.Chip8.Kernel.RomScheduleBinding` succeeds.
2. The theorem surface makes `meta_pub`, pad-row metadata, and initial-state
   authentication explicit.
3. No `sorry`.

## Out of Scope

- Stage-1 proofs
- Stage-2 proofs
- Stage-3 proofs
- Poseidon2 security proofs


# /Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/specs/chip8/Chip8OpeningBoundary.spec.md
# Chip8OpeningBoundary Spec

## Purpose

- **What it is**: The theorem-facing opening-manifest contract for the final
  CHIP-8 kernel.
- **Key property**: `kernelOpeningBoundary_conforms`: every kernel-owned opening
  claim references exactly one commitment fixed in `root0`, uses the exact
  commitment-local polynomial registry, and appears in the correct kernel/root
  ownership bucket.
- **Protocol role**: This is the boundary that prevents the kernel from
  overclaiming direct opening access to undeclared commitments or from
  conflating kernel-owned openings with later root-prover openings.

This module classifies opening claims and manifests only. It does **not** own
the checked Shout/Twist proofs themselves, the continuity reduction proof, or
semantic extraction from those claims.

## Target Formulas

### Kernel commitment surface

The final kernel fixes the following opening commitments before any
stage-specific challenge is sampled:

- `C_lane`
- `C_fetch_ra`
- `C_decode_ra`
- `C_alu_ra`
- `C_eq4_ra`
- `C_decode_handoff`
- `C_reg`
- `C_ram`
- `C_rom_table`
- `C_decode_table`
- `C_alu_table`
- `C_eq4_table`

These are the only commitments that may appear in the kernel-owned opening
manifest.

The root prover later creates disjoint commitments, including the per-step
Ajtai commitments inside `PreparedStep`.

### Opening claim shape

Define the theorem-facing opening claim:

$$
\mathrm{OpeningClaim}
:=
(
\mathrm{source},
\mathrm{commitmentId},
\mathrm{point},
\mathrm{polynomialIds},
\mathrm{claimedValues},
\mathrm{digest}
).
$$

The commitment identifier is drawn from:

$$
\mathrm{CommitmentId}
:=
\mathrm{Lane}
\mid
\mathrm{FetchRa}
\mid
\mathrm{DecodeRa}
\mid
\mathrm{AluRa}
\mid
\mathrm{Eq4Ra}
\mid
\mathrm{DecodeHandoff}
\mid
\mathrm{RegTwist}
\mid
\mathrm{RamTwist}
\mid
\mathrm{RomTable}
\mid
\mathrm{DecodeTable}
\mid
\mathrm{AluTable}
\mid
\mathrm{Eq4Table}
\mid
\mathrm{RootProver}(\_).
$$

### Manifest ownership buckets

The opening boundary is split into two disjoint ownership buckets:

- `KernelOpeningManifest`
- `RootOpeningManifest`

The kernel manifest may reference only the commitments fixed in `root0`.
The root manifest may reference only commitments created after bridge
extraction.

### Grouping rule

Opening claims are grouped by:

$$
(\mathrm{commitmentId}, \mathrm{point}).
$$

This is the theorem-facing grouping rule consumed by the later PCS opening
verifier.

### Kernel manifest shape

At minimum, the kernel-owned manifest contains direct openings for:

- `C_lane @ r_lookup` for the Stage-1 lane columns
  `{PC, KK, NNN_ADDR, NNN_WORD, REG_X, REG_Y, LOOKUP_OUTPUT,
    WritesLookupToX, WritesMemToX, PreservesX, WritesNnnToI,
    IsJump, IsBranch, IsMemOp, X_IDX, Y_IDX, BURST_LAST}`
- `C_fetch_ra @ (r_fetch_addr, r_lookup)`
- `C_decode_ra @ (r_decode_addr, r_lookup)`
- `C_alu_ra @ (r_alu_addr, r_lookup)`
- `C_eq4_ra @ (r_eq4_addr, r_lookup)`
- `C_decode_handoff @ r_lookup` with exact polynomial subset
  `{uses_y_dec, reads_ram_dec, writes_ram_dec}`
- `C_rom_table @ r_fetch_addr`
- `C_decode_table @ r_decode_addr` with exact polynomial subset `0..21`
- `C_alu_table @ r_add8lo_addr`
- `C_eq4_table @ r_eq4_addr`
- `C_lane @ r_twist_cycle` for the Stage-2 lane columns
  `{REG_X, REG_Y, REG_X_NEXT, I_REG, I_NEXT, MEM_VALUE,
    WritesLookupToX, WritesMemToX, PreservesX, WritesNnnToI,
    IsMemOp, X_IDX, Y_IDX, RAM_ADDR}`
- `C_decode_handoff @ r_twist_cycle` with exact polynomial subset
  `{uses_y_dec, reads_ram_dec, writes_ram_dec}`
- `C_reg @ (r_addr_reg, r_twist_cycle)` with exact polynomial subset
  `{RegInc, RegRaX, RegRaY, RegRaI, RegWa}`
- `C_ram @ (r_addr_ram, r_twist_cycle)` with exact polynomial subset
  `{RamInc, RamRa, RamWa}`
- `C_lane @ r_shift` for `{PC, PC_NEXT, X_IDX, IsMemOp, BURST_LAST}`
- `C_lane @ j0_bits` for `{IsMemOp, X_IDX}`
- `C_lane @ j_last_bits` for `{IsMemOp, BURST_LAST}`
- `C_lane @ j_bits` for the exact 23 committed non-fixed lane coordinates of
  every exported semantic row, in canonical `C_lane` registry order:
  `{PC, PC_NEXT, REG_X, REG_Y, REG_X_NEXT, I_REG, I_NEXT, KK, NNN_ADDR,
    NNN_WORD, MEM_VALUE, LOOKUP_OUTPUT, WritesLookupToX, WritesMemToX,
    PreservesX, WritesNnnToI, IsJump, IsBranch, IsMemOp, X_IDX, Y_IDX,
    BURST_LAST, RAM_ADDR}`

### Exact exclusions

The opening boundary must also state the exact exclusions:

- `LaneShiftProof` is not an `OpeningClaim`
- `KernelOpeningManifest` may not reference any root-prover commitment
- `RootOpeningManifest` may not reference any `C_*` commitment fixed in `root0`
- later kernel stages do not reuse Stage-1 openings; each stage opens its own
  direct claims independently

### Canonical manifest ordering

Define the canonical commitment-id order:

$$
\mathrm{Lane}
\prec
\mathrm{FetchRa}
\prec
\mathrm{DecodeRa}
\prec
\mathrm{AluRa}
\prec
\mathrm{Eq4Ra}
\prec
\mathrm{DecodeHandoff}
\prec
\mathrm{RegTwist}
\prec
\mathrm{RamTwist}
\prec
\mathrm{RomTable}
\prec
\mathrm{DecodeTable}
\prec
\mathrm{AluTable}
\prec
\mathrm{Eq4Table}
\prec
\mathrm{RootProver}(\_).
$$

The canonical manifest sort key is:

$$
(\mathrm{commitmentIdOrder}, \mathrm{pointArity},
\mathrm{pointCoordinates}, \mathrm{polynomialIds}).
$$

The boundary must enforce:

- strictly increasing `polynomialIds` in the local registry order
- point coordinates ordered exactly by the commitment's domain convention
- no duplicate `(commitmentId, point, polynomialIds)` entries

### Kernel opening boundary

Define:

$$
\mathrm{KernelOpeningBoundary}(kernelManifest, rootManifest)
$$

to mean:

- every kernel claim references a kernel commitment fixed in `root0`
- every root claim references only root-prover commitments
- the kernel manifest satisfies the exact kernel-shape constraints above
- both manifests satisfy the canonical grouping and ordering rules

The main theorem target is:

$$
\mathrm{KernelOpeningBoundary}(kernelManifest, rootManifest)
\Longrightarrow
\text{no orphan or mis-owned opening claim exists}.
$$

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/chip8-kernel.md`
- Anchors:
  - commitment bundle
  - two commitment layers
  - opening boundary
  - canonical manifest ordering

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/Kernel/OpeningBoundary.lean` | Kernel/root opening-manifest ownership and conformance theorems |
| `Nightstream/Chip8/Kernel/OpeningBoundaryInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Commitments | `CommitmentId` | def | Definitional | Enumerates every opening commitment class relevant to the kernel/root boundary |
| Claims | `OpeningClaim` | def | Definitional | Theorem-facing direct opening claim object |
| Manifests | `KernelOpeningManifest` | def | Definitional | Kernel-owned direct opening manifest |
| Manifests | `RootOpeningManifest` | def | Definitional | Root-prover direct opening manifest |
| Boundary | `KernelManifestShape` | def | Definitional | Exact required kernel-owned claim families and points |
| Boundary | `CanonicalManifestOrder` | def | Definitional | Canonical ordering and grouping rule |
| Boundary | `KernelOpeningBoundary` | def | Definitional | Complete kernel/root ownership and conformance predicate |
| Theorem | `kernelOpeningBoundary_conforms` | theorem | Theorem-Target | A conforming manifest contains only legal, correctly owned opening claims |
| Theorem | `laneShift_not_openingClaim` | theorem | Theorem-Target | `LaneShiftProof` is not part of either opening manifest |
| Theorem | `kernel_root_commitments_disjoint` | theorem | Theorem-Target | Kernel and root opening commitments remain disjoint |

## Proof Obligations

- This module must model the final kernel/root manifest split explicitly.
- The theorem surface must enumerate the exact kernel opening commitments.
- Canonical manifest ordering and grouping must remain theorem-facing, not only
  implementation prose.
- `LaneShiftProof` must remain outside the direct opening-manifest type.

## Assumption Ledger

- This module does not re-prove PCS batching or final opening verification.
- This module does not prove Shout or Twist checked reductions.
- This module does not prove semantic extraction from opening claims.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - final kernel commitment surface from `chip8-kernel.md`
- **Downstream consumers**:
  - `Nightstream/Chip8/Stage2/EvidenceCoverage.lean`
  - later Rust-refinement theorems for `KernelOpeningManifest`

## Implementation Plan

1. Define the exact opening claim and manifest objects.
2. Define the kernel/root ownership split and manifest shape.
3. Define canonical ordering and grouping.
4. Prove manifest conformance and exact exclusions.

## Quality Expectations

- Keep this module about direct opening ownership only.
- Specialize to the final kernel commitment inventory.
- Do not smuggle checked Shout/Twist relations into the direct opening manifest.

## Acceptance Criteria

1. `lake build Nightstream.Chip8.Kernel.OpeningBoundary` succeeds.
2. The theorem surface matches the final kernel/root opening-manifest split.
3. Kernel and root commitment layers remain formally disjoint.
4. No `sorry`.

## Out of Scope

- checked Shout/Twist proof objects
- Stage-3 continuity reduction proofs
- semantic extraction
- final PCS opening verification


# /Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/specs/chip8/Chip8ContinuityBridge.spec.md
# Chip8ContinuityBridge Spec

## Purpose

- **What it is**: The theorem-facing Stage-3 contract for continuity and bridge
  binding in the final CHIP-8 kernel.
- **Key property**: `preparedStepBound_of_rowBinding`: if continuity is
  authenticated via `LaneShiftProof` and each exported row is bound back to
  `C_lane`, then every exported `PreparedStep` is the canonical root encoding of
  one authenticated semantic row.
- **Protocol role**: This is the Stage-3 owner for continuity support,
  start-boundary correctness, row-binding claims, and the exact handoff into
  the SuperNeo root main lane.

## Target Formulas

### Continuity support relation

Stage 3 owns:

- `PC(j+1) = PC_NEXT(j)` on real row pairs
- burst progression on intermediate memory-prefix rows
- burst reset on memory-prefix starts
- the explicit start-boundary rule `IsMemOp(0) * X_IDX(0) = 0`

The continuity relation is checked at one Stage-3 cycle point `r_shift`.

### Pair mask

Define the verifier-computable real-pair mask:

$$
\mathrm{PairMask}_N(j) =
\begin{cases}
1 & \text{if } 0 \le j < N-1, \\
0 & \text{otherwise.}
\end{cases}
$$

and let `PairMask_N(X)` be its multilinear extension over the cycle hypercube.

### LaneShift reduction

Stage 3 does not assume ad hoc access to `PC(j+1)`, `X_IDX(j+1)`, or
`IsMemOp(j+1)`. Instead it uses an explicit checked virtual reduction:

$$
\mathrm{Shift}[f](X) = \sum_{j=0}^{T-2} eq(X,j)\cdot f(j+1)
$$

for each `f` in `{PC, X_IDX, IsMemOp}`.

Define a theorem-facing `LaneShiftClaim` carrying:

- `source_commitment = C_lane`
- `source_point = r_shift`
- `source_columns = [PC, X_IDX, IsMemOp]`
- `shifted_columns = [Shift[PC], Shift[X_IDX], Shift[IsMemOp]]`
- `claimed_shift_values = [shift_pc, shift_x_idx, shift_is_memop]`

and a checked proof object:

$$
\mathrm{LaneShiftBound}(claim, proof).
$$

### Continuity check

After authenticating the shifted values, Stage 3 opens current-row lane columns
at `r_shift` and checks:

$$
\delta_{pc} = PairMask_N(r_{shift}) \cdot (shift\_pc - PC\_NEXT)
$$

$$
\delta_{burstStep}
=
PairMask_N(r_{shift}) \cdot IsMemOp \cdot (1 - BURST\_LAST)
\cdot (shift\_x\_idx - X\_IDX - 1)
$$

$$
\delta_{burstReset}
=
PairMask_N(r_{shift}) \cdot shift\_is\_memop
\cdot (1 - IsMemOp + BURST\_LAST)\cdot shift\_x\_idx
$$

and the batched identity:

$$
\delta_{pc} + \beta_1 \cdot \delta_{burstStep} + \beta_2 \cdot \delta_{burstReset} = 0.
$$

Define:

$$
\mathrm{ContinuityBound}(N, r_{shift}, claim, proof, currentRow).
$$

### Start-boundary rule

Let `j0_bits = 0^{CYCLE_BITS}`.

Stage 3 opens `IsMemOp` and `X_IDX` from `C_lane @ j0_bits` and proves:

$$
\mathrm{IsMemOp}(0)\cdot X\_IDX(0) = 0.
$$

Define:

$$
\mathrm{StartBoundaryBound}(j0Row).
$$

This is the theorem-facing statement that chunks begin on instruction
boundaries with no in-flight burst state.

### Row-binding claims

For each exported row `j`, the bridge owns:

$$
\mathrm{RowBindingClaim}_j = (C_{lane}, j_{bits},
[\mathrm{PC}, \mathrm{PC\_NEXT}, \mathrm{REG\_X}, \mathrm{REG\_Y}, \mathrm{REG\_X\_NEXT},
\mathrm{I\_REG}, \mathrm{I\_NEXT}, \mathrm{KK}, \mathrm{NNN\_ADDR},
\mathrm{NNN\_WORD}, \mathrm{MEM\_VALUE}, \mathrm{LOOKUP\_OUTPUT},
\mathrm{WritesLookupToX}, \mathrm{WritesMemToX}, \mathrm{PreservesX},
\mathrm{WritesNnnToI}, \mathrm{IsJump}, \mathrm{IsBranch}, \mathrm{IsMemOp},
\mathrm{X\_IDX}, \mathrm{Y\_IDX}, \mathrm{BURST\_LAST}, \mathrm{RAM\_ADDR}]).
$$

Define:

$$
\mathrm{RowBound}(claim_j, z_j)
$$

to mean:

- the claim opens exactly those 23 committed non-fixed lane coordinates of row
  `j` from `C_lane`, in that canonical registry order
- `z_j[0] = ONE = 1` is inserted as the fixed verifier-known coordinate
- the resulting semantic row is exactly `z_j`

### Root encoding and prepared-step binding

Let:

$$
\mathrm{RootEncode}(z_j) = (w_j, Z_j)
$$

be the canonical packed root witness encoding used by the root prover.

Normative meaning:

- `w_j` is coordinates `1..23` of the semantic row `z_j`;
- `Z_j` is obtained by padding, canonical reshaping, and the canonical
  Ajtai/root witness encoding determined only by the public root parameters;
- the imported root encoding and Ajtai commitment context is fixed by the
  public `root_params_id` / `vm_spec` boundary, not by hidden caller-chosen
  functions or witness-layout conventions.

Define:

$$
\mathrm{PreparedStepBound}(z_j, step_j)
$$

to mean:

- `step_j.witness.w = w_j`
- `step_j.witness.Z = Z_j`
- `step_j.mcs.c = Ajtai_commit(Z_j)`
- `step_j.mcs.x = [1]`
- `step_j.mcs.m_in = 1`

### Bundled Stage-3 theorem

Define:

$$
\mathrm{Stage3Bound}(N, shiftClaim, shiftProof, startRow, rowClaims, preparedSteps)
$$

to mean:

- `ContinuityBound(...)`
- `StartBoundaryBound(startRow)`
- every exported row claim satisfies `RowBound`
- every exported prepared step satisfies `PreparedStepBound`

The main theorem target is:

$$
\mathrm{Stage3Bound}(\dots)
\Longrightarrow
\text{every exported prepared step is bound to one authenticated row of } C_{lane}.
$$

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/chip8-kernel.md`
- Anchors:
  - continuity support relation
  - `LaneShiftProof`
  - `ContinuityCheck`
  - opening boundary row-binding openings
  - bridge binding mechanism
  - prepared-step construction

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/Stage3/ContinuityBridge.lean` | Stage-3 continuity and row-to-`PreparedStep` bridge-binding theorems |
| `Nightstream/Chip8/Stage3/ContinuityBridgeInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Continuity | `PairMaskN` | def | Definitional | Real-pair mask for continuity over the padded cycle domain |
| Continuity | `LaneShiftClaim` | def | Definitional | Exact Stage-3 shifted-column claim object |
| Continuity | `LaneShiftBound` | def | Definitional | Checked virtual reduction against `C_lane` |
| Continuity | `ContinuityBound` | def | Definitional | Exact Stage-3 continuity identity at `r_shift` |
| Boundary | `StartBoundaryBound` | def | Definitional | Exact start-of-chunk burst-boundary rule |
| Bridge | `RowBindingClaim` | def | Definitional | Exact per-row direct binding claim against `C_lane` |
| Bridge | `RowBound` | def | Definitional | One row-binding claim determines one semantic row |
| Bridge | `RootEncode` | def | Definitional | Canonical root witness encoding for one semantic row |
| Bridge | `PreparedStepBound` | def | Definitional | One prepared step is exactly the root encoding of one row |
| Bundle | `Stage3Bound` | def | Definitional | Complete Stage-3 continuity and bridge-binding bundle |
| Theorem | `continuityBound_of_laneShift` | theorem | Theorem-Target | Authenticated shifted values plus current-row openings imply continuity |
| Theorem | `preparedStepBound_of_rowBinding` | theorem | Theorem-Target | Row binding implies exact prepared-step binding |
| Theorem | `stage3Bound_exports_authenticatedRows` | theorem | Theorem-Target | Stage 3 exports only authenticated rows to the root main lane |

## Proof Obligations

- `LaneShiftProof` must remain a checked virtual reduction, not an opening
  claim.
- The continuity identity must use exactly the final kernel's
  `PairMask_N`, `β1`, and `β2` batching surface.
- Row binding must use explicit row openings against `C_lane`, not the rejected
  aggregate-identity shortcut.
- `RootEncode` must be the exact canonical root encoding used by the root
  prover.

## Assumption Ledger

- This module does not re-prove the root main-lane CCS proof.
- This module does not re-prove PCS opening verification.
- This module imports the canonical root encoding and Ajtai commitment surface.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/PCSOpeningSemantics.lean`
  - later root-encoding interface from the SuperNeo side
- **Downstream consumers**:
  - `Nightstream/Chip8/Stage2/EvidenceCoverage.lean`
  - later Rust-refinement theorems for `PreparedStep` export
  - generic `MainLaneBridge`

## Implementation Plan

1. Define the continuity mask and shift-claim surface.
2. Define the checked Stage-3 continuity predicate.
3. Define row-binding and prepared-step binding.
4. Prove the bundled Stage-3 bridge theorem.

## Quality Expectations

- Keep Stage 3 separate from Stage 1 and Stage 2.
- Keep row binding explicit and commitment-based.
- Make the handoff into the root main lane mechanically obvious.

## Acceptance Criteria

1. `lake build Nightstream.Chip8.Stage3.ContinuityBridge` succeeds.
2. The theorem surface covers `LaneShiftProof`, `ContinuityCheck`, and explicit
   row binding.
3. The bridge uses explicit row openings, not an aggregate shortcut.
4. No `sorry`.

## Out of Scope

- Stage-1 fetch/decode proofs
- Stage-2 Twist proofs
- the root main-lane CCS proof itself


# /Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/specs/chip8/Chip8EvidenceCoverage.spec.md
# Chip8EvidenceCoverage Spec

## Purpose

- **What it is**: The theorem-facing bridge from authenticated kernel proof
  objects to the semantic facts consumed by the CHIP-8 composition theorems.
- **Key property**: `semanticBounds_of_authenticatedEvidence`: if the final
  kernel proof exposes the exact public-input bindings, opening manifests,
  Stage-1 semantic objects, Stage-2 memory objects, Stage-3 continuity/bridge
  objects, and lower-layer opening refinements, then the semantic bounds
  consumed by `Chip8StepComposition` are derivable.
- **Protocol role**: This is the layer that closes the gap between the final
  kernel proof object and the semantic execution theorems.

## Target Formulas

### Evidence inputs

This layer reasons over:

- authenticated public inputs and `meta_pub`
- `KernelOpeningManifest` and `RootOpeningManifest`
- lower-layer PCS opening witnesses together with the refinement map from those
  witnesses to raw scalar opening claims
- Stage-1 authenticated fetch/decode/ALU/Eq4/handoff objects
- Stage-2 authenticated register/RAM/Twist/RAF objects
- Stage-3 authenticated `LaneShiftProof`, continuity object, and row-binding
  claims
- authenticated row/view objects used to bind opened scalars back to the
  semantic lane row

### Object-level provenance

Top-level opening-manifest conformance alone is not sufficient. This module
introduces theorem-facing provenance predicates for the internal objects
consumed by checked relations.

$$
\mathrm{TableProvenance}(table)
$$

states that a Stage-1 table object is either verifier-local evaluable or
backed by the exact committed table opening mandated by the kernel manifest.

$$
\mathrm{VirtualValProvenance}(val, session)
$$

states that a register or RAM virtual `Val` object is authenticated exactly by
the matching Twist read/write/Val proof chain for that session.

$$
\mathrm{AddressProvenance}(dec, fam, addr)
$$

states that the address or key is the exact Stage-1 or Stage-2 family
projection fixed by `Chip8DecodeAddressBinding`.

$$
\mathrm{HandoffProvenance}(dec, handoff)
$$

states that the Stage-2 `usesY`, `readsRam`, and `writesRam` bits come from the
exact committed `C_decode_handoff` surface and equal the authenticated Stage-1
decode outputs.

### Twist session closure

The final kernel keeps the register and RAM Twist chains explicit. Introduce:

$$
\mathrm{TwistSessionClosed}(stage2)
$$

meaning:

- register read/write batching, `Val`-from-`Inc`, and address-correctness refer
  to one closed authenticated register session
- RAM read/write batching, `Val`-from-`Inc`, RAF support, and
  address-correctness refer to one closed authenticated RAM session
- no Stage-2 semantic fact is extracted from a dangling or mismatched subclaim

### Row/view consistency

The semantic layer still needs an explicit positive row-binding theorem. Define:

$$
\mathrm{RowProjectionWitness}(kernelManifest, rowClaims, \rho)
$$

$$
\mathrm{RowConsistent}(\rho, z, dec, pre, post)
$$

where \(\rho\) is tied to the same authenticated `C_lane` row-binding claims
that later feed `PreparedStep`.

### PCS refinement and public-input authentication

Each directly opened scalar used by semantic extraction must carry an explicit
lower-layer refinement:

$$
\mathrm{OpeningRefinement}(\text{params}, \text{extract},
  \mathrm{RawScalarClaim}(b,p,v)).
$$

This layer also consumes:

$$
\mathrm{KernelPublicInputsBound}(publicInput, meta, romTable, init)
$$

from `Chip8KernelInputBinding`.

For every direct scalar consumed by this layer, accepted-opening provenance must
also enforce coordinatewise equality between the direct claim's
`claimedValues` and the exact-opening witness values in the same
`polynomialIds` order. The scalar theorem-facing carrier is
`AcceptedScalarOpening`, which is a scalar projection of the stronger
direct-claim object `AcceptedDirectOpening`.

### Evidence coverage of semantic facts

Define:

$$
\mathrm{SemanticEvidenceCovered}
(publicInput, meta, kernelManifest, rootManifest,
 stage1, stage2, stage3, romTable, init, pre, post, dec, z)
$$

to mean:

- `KernelPublicInputsBound(publicInput, meta, romTable, init)`
- `KernelOpeningBoundary(kernelManifest, rootManifest)`
- object-level provenance for every table, virtual `Val`, address family, and
  handoff object used by the current row
- `TwistSessionClosed(stage2)`
- a row/view witness tied to the authenticated `C_lane` row-binding claims
- lower-layer refinement, coordinatewise accepted-opening equality, and raw PCS
  opening separation for every direct scalar used by that row/view witness
- exact Stage-1, Stage-2, and Stage-3 CHIP-8 bindings for the current row

### Extraction theorems

The key extraction targets are:

$$
\mathrm{SemanticEvidenceCovered}(\dots)
\Longrightarrow
\mathrm{KernelPublicInputsBound}(publicInput, meta, romTable, init)
$$

$$
\mathrm{SemanticEvidenceCovered}(\dots)
\Longrightarrow
\mathrm{FetchDecodeBound}(romTable, pre.PC, dec)
$$

$$
\mathrm{SemanticEvidenceCovered}(\dots)
\Longrightarrow
\mathrm{LookupBound}(dec, pre, z)
$$

$$
\mathrm{SemanticEvidenceCovered}(\dots)
\Longrightarrow
\mathrm{WitnessBinds}(pre, post, dec, z)
$$

$$
\mathrm{SemanticEvidenceCovered}(\dots)
\Longrightarrow
\mathrm{MemoryBound}(pre, post, init, dec, z)
$$

$$
\mathrm{SemanticEvidenceCovered}(\dots)
\Longrightarrow
\mathrm{ContinuityRowBound}(stage3, z)
$$

Bundled together:

$$
\mathrm{SemanticEvidenceCovered}(\dots)
\Longrightarrow
\Big(
\mathrm{KernelPublicInputsBound}
\land
\mathrm{FetchDecodeBound}
\land
\mathrm{LookupBound}
\land
\mathrm{WitnessBinds}
\land
\mathrm{MemoryBound}
\land
\mathrm{ContinuityRowBound}
\Big).
$$

This is the exact bridge consumed by the semantic composition layer.

To keep the theorem-facing surface exact to the final kernel proof object,
the exported boundary may hide any internal stage-local claim lists used by the
Lean implementation. Define:

$$
\mathrm{ExactSemanticEvidenceCovered}(\dots)
$$

to mean that there exist whatever internal stage-local witnesses are needed to
establish `SemanticEvidenceCovered`, without exposing those internal lists as
top-level theorem parameters.

## Paper Anchors

- **Sources**:
  - `./docs/soundness-specs/twist-and-shout-requirements.md`
  - `./crates/neo-fold-prototype/specs/chip8-kernel.md`
- Anchors:
  - commitment-before-challenge discipline
  - Stage-1 linkage and handoff
  - Stage-2 explicit Twist chains
  - Stage-3 continuity and row binding
  - prohibition on treating sparse openings as unauthenticated full-row access

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/Stage2/EvidenceCoverage.lean` | Kernel-proof-to-semantics extraction theorems for the final CHIP-8 kernel |
| `Nightstream/Chip8/Stage2/EvidenceCoverageInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Provenance | `TableProvenance` | def | Definitional | Every Stage-1 table object has explicit committed or evaluator-local provenance |
| Provenance | `VirtualValProvenance` | def | Definitional | Every register/RAM virtual `Val` object is justified by the exact Stage-2 chain |
| Provenance | `AddressProvenance` | def | Definitional | Every Stage-1/Stage-2 address comes from the exact CHIP-8 family projection |
| Provenance | `HandoffProvenance` | def | Definitional | Every Stage-2 handoff bit comes from `C_decode_handoff` and equals the Stage-1 decode output |
| Closure | `TwistSessionClosed` | def | Definitional | Stage 2 contains closed register and RAM Twist sessions |
| Rows | `RowProjectionWitness` | def | Definitional | Explicit authenticated row/view witness tied to `C_lane` row-binding claims |
| Rows | `RowConsistent` | def | Definitional | The semantic row/view witness matches the consumed row objects |
| Inputs | `PCSContext` | def | Definitional | Lower-layer PCS parameters and scalar extractor for raw opening refinement |
| Coverage | `SemanticEvidenceCovered` | def | Definitional | Authenticated kernel evidence is sufficient to recover the semantic row facts |
| Coverage | `ExactSemanticEvidenceCovered` | def | Definitional | Final-kernel theorem-facing coverage predicate that hides internal stage-local witness lists |
| Theorem | `kernelPublicInputsBound_of_evidence` | theorem | Theorem-Target | Authenticated evidence yields the fixed public kernel input bounds |
| Theorem | `fetchDecodeBound_of_evidence` | theorem | Theorem-Target | Authenticated evidence yields Stage-1 fetch/decode binding |
| Theorem | `lookupBound_of_evidence` | theorem | Theorem-Target | Authenticated evidence yields Stage-1 helper-lookup semantics |
| Theorem | `witnessBinds_of_evidence` | theorem | Theorem-Target | Authenticated evidence yields semantic lane-row binding |
| Theorem | `memoryBound_of_evidence` | theorem | Theorem-Target | Authenticated evidence yields Stage-2 semantic memory binding |
| Theorem | `continuityRowBound_of_evidence` | theorem | Theorem-Target | Authenticated evidence yields the Stage-3 row-local continuity facts needed downstream |
| Theorem | `semanticBounds_of_authenticatedEvidence` | theorem | Theorem-Target | Authenticated evidence yields the full semantic fact bundle consumed by `Chip8StepComposition` |
| Theorem | `semanticBounds_of_exactAuthenticatedEvidence` | theorem | Theorem-Target | Exact final-kernel evidence yields the same semantic fact bundle without exposing internal stage claim lists |

## Proof Obligations

- The module must track the final kernel proof object structure, not the older
  abstract Stage-1/2/3 claim-multiset abstraction.
- Any internal stage-local claim lists used to witness extraction must remain
  hidden behind an exact theorem-facing coverage predicate.
- The module must keep Stage-2 Twist closure explicit.
- The row/view witness must be tied to explicit `C_lane` row-binding claims.
- Every direct scalar used in extraction must be backed by an explicit
  lower-layer opening refinement.
- No semantic fact may be extracted from a kernel proof object without
  explicit provenance.

## Assumption Ledger

- Generic Shout and Twist theorem statements are imported.
- PCS binding and Fiat-Shamir security remain external.
- This module owns only the CHIP-8 instantiation-level extraction story.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/Kernel/OpeningBoundary.lean`
  - `Nightstream/Chip8/Stage1/FetchDecodeBinding.lean`
  - `Nightstream/Chip8/Stage1/DecodeAddressBinding.lean`
  - `Nightstream/Chip8/Stage2/WitnessMemoryBinding.lean`
  - `Nightstream/Chip8/Kernel/RomScheduleBinding.lean`
  - `Nightstream/Chip8/Stage3/ContinuityBridge.lean`
  - `Nightstream/PCSOpeningSemantics.lean`
- **Downstream consumers**:
  - `Nightstream/Chip8/Kernel/StagedExecutionDigest.lean`
  - `Nightstream/Chip8/Kernel/ArtifactAudit.lean`
  - `Nightstream/Chip8/Execution/StepComposition.lean`
  - later Rust-refinement theorems about the final kernel proof object

## Implementation Plan

1. Define provenance, closure, and row-binding predicates over the final kernel
   proof object.
2. Define `SemanticEvidenceCovered`.
3. Prove the per-fact extraction theorems.
4. Prove the bundled semantic-coverage theorem.

## Quality Expectations

- Keep the module focused on authenticated extraction.
- Keep the final kernel's Stage-1, Stage-2, and Stage-3 ownership split
  explicit.
- Keep row-binding and PCS refinement explicit and separate.

## Acceptance Criteria

1. `lake build Nightstream.Chip8.Stage2.EvidenceCoverage` succeeds.
2. The theorem surface rules out unauthenticated extraction from internal proof
   objects.
3. The theorem surface rules out treating sparse openings as free full-row
   access.
4. No `sorry`.

## Out of Scope

- generic Shout theorem proofs
- generic Twist theorem proofs
- the root main-lane CCS proof itself
- final PCS opening verification


# /Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/specs/chip8/Chip8StagedExecutionDigest.spec.md
# Chip8StagedExecutionDigest Spec

## Purpose

- **What it is**: The theorem-facing normalized digest contract for one staged
  CHIP-8 kernel execution.
- **Key property**: `stagedExecutionDigest_of_exactEvidence`: exact
  authenticated public inputs, Stage-1/Stage-2/Stage-3 semantic evidence, and
  the resulting execution theorem determine one normalized staged execution
  digest.
- **Protocol role**: This is the single explicit comparison and audit boundary
  shared by Rust and Lean. Its shape is dictated by the theorem surfaces that
  the composition theorem actually consumes, not by implementation-local export
  convenience.

## Target Formulas

### Digest ownership

The digest owner packages the exact theorem-facing surfaces of the final kernel
into one explicit record.

The intended components are:

- one public-input surface
- one Stage-1 surface
- one Stage-2 surface
- one Stage-3 continuity / bridge surface
- one final semantic-result surface

This owner does not define new semantic facts. It normalizes the existing ones
into one comparison and audit contract.

### Public surface

Define:

$$
\mathrm{DigestPublicSurface}(d_{\mathrm{pub}}, publicInput, meta, romTable, init)
$$

to mean that the public portion of the digest determines exactly the bundled
public-input theorem surface:

$$
\mathrm{KernelPublicInputsBound}(publicInput, meta, romTable, init).
$$

### Stage-1 surface

Define:

$$
\mathrm{Stage1DigestSurface}(d_1, romTable, pre, dec, z)
$$

to mean that the Stage-1 digest component determines exactly the authenticated
Stage-1 theorem surfaces needed downstream:

$$
\mathrm{FetchDecodeBound}(romTable, pre.PC, dec)
$$

and

$$
\mathrm{LookupBound}(dec, pre, z).
$$

### Stage-2 surface

Define:

$$
\mathrm{Stage2DigestSurface}(d_2, pre, post, init, dec, z)
$$

to mean that the Stage-2 digest component determines exactly the row-binding
and memory-binding theorem surfaces:

$$
\mathrm{WitnessBinds}(pre, post, dec, z)
$$

and

$$
\mathrm{MemoryBound}(pre, post, init, dec, z).
$$

### Stage-3 surface

Define:

$$
\mathrm{Stage3DigestSurface}(d_3, z, prepared)
$$

to mean that the Stage-3 digest component determines exactly the continuity and
bridge theorem surfaces:

$$
\mathrm{ContinuityRowBound}(d_3, z)
$$

and, when the current row is exported,

$$
\mathrm{PreparedStepBound}(d_3, z, prepared).
$$

### Final semantic-result surface

Define:

$$
\mathrm{ExecutionResultSurface}(d_r, pre, post, dec, z)
$$

to mean that the digest result component determines the exact semantic theorem
surface for the current supported-kernel row or chunk-local execution object.

At minimum, this owner must support packaging the execution theorem surfaces
already owned by `Chip8ExecutionSemantics`, `Chip8BurstSession`, and
`Chip8StepComposition`.

### Bundled digest

Define the normalized digest object:

$$
\mathrm{StagedExecutionDigest}
(d_{\mathrm{pub}}, d_1, d_2, d_3, d_r).
$$

Define its exact realization predicate:

$$
\mathrm{StagedExecutionDigestBound}
(d, publicInput, meta, romTable, init, pre, post, dec, z, prepared)
$$

to mean the conjunction of:

- `DigestPublicSurface`
- `Stage1DigestSurface`
- `Stage2DigestSurface`
- `Stage3DigestSurface`
- `ExecutionResultSurface`

for one exact supported-kernel execution slice.

### Normalization theorem

The main normalization theorem is:

$$
\mathrm{ExactSemanticEvidenceCovered}(\dots)
\land
\mathrm{ExecutionCorrect}(\dots)
\Longrightarrow
\exists d,\ \mathrm{StagedExecutionDigestBound}(d,\dots).
$$

This is the theorem that makes the digest Lean-defined rather than Rust-defined.

### Projection theorems

From one realized digest, downstream users must be able to recover the exact
existing theorem surfaces:

$$
\mathrm{StagedExecutionDigestBound}(d,\dots)
\Longrightarrow
\mathrm{KernelPublicInputsBound}(publicInput, meta, romTable, init)
$$

$$
\mathrm{StagedExecutionDigestBound}(d,\dots)
\Longrightarrow
\mathrm{FetchDecodeBound}(romTable, pre.PC, dec)
$$

$$
\mathrm{StagedExecutionDigestBound}(d,\dots)
\Longrightarrow
\mathrm{LookupBound}(dec, pre, z)
$$

$$
\mathrm{StagedExecutionDigestBound}(d,\dots)
\Longrightarrow
\mathrm{WitnessBinds}(pre, post, dec, z)
$$

$$
\mathrm{StagedExecutionDigestBound}(d,\dots)
\Longrightarrow
\mathrm{MemoryBound}(pre, post, init, dec, z)
$$

$$
\mathrm{StagedExecutionDigestBound}(d,\dots)
\Longrightarrow
\mathrm{ContinuityRowBound}(d_3, z).
$$

This ensures the digest is a faithful normal form of the existing theorem
surfaces, not a parallel informal interface.

## Paper Anchors

- **Sources**:
  - `./docs/assurance-strategy.md`
  - `./docs/soundness-specs/twist-and-shout-requirements.md`
  - `./crates/neo-fold-prototype/specs/chip8-kernel.md`
  - `./docs/superneo-paper`
- Anchors:
  - staged proof composition
  - authenticated public-input binding
  - Stage-1 / Stage-2 / Stage-3 theorem ownership
  - bridge export into root prepared steps
  - one explicit digest/evidence contract for comparison and audit

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/Kernel/StagedExecutionDigest.lean` | Normalized digest contract and normalization/projection theorems for the final CHIP-8 kernel |
| `Nightstream/Chip8/Kernel/StagedExecutionDigestInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Digest | `DigestPublicSurface` | def | Definitional | The public digest component carries the exact bundled public-input theorem surface |
| Digest | `Stage1DigestSurface` | def | Definitional | The Stage-1 digest component carries the exact Stage-1 theorem surfaces |
| Digest | `Stage2DigestSurface` | def | Definitional | The Stage-2 digest component carries the exact Stage-2 theorem surfaces |
| Digest | `Stage3DigestSurface` | def | Definitional | The Stage-3 digest component carries the exact continuity / bridge theorem surfaces |
| Digest | `ExecutionResultSurface` | def | Definitional | The result digest component carries the exact supported-kernel semantic result surface |
| Bundle | `StagedExecutionDigest` | def | Definitional | One explicit normalized digest contract shared by Rust and Lean |
| Bundle | `StagedExecutionDigestBound` | def | Definitional | Exact theorem-facing realization predicate for one digest instance |
| Theorem | `stagedExecutionDigest_of_exactEvidence` | theorem | Theorem-Target | Exact authenticated evidence and execution semantics determine one realized digest |
| Theorem | `kernelPublicInputsBound_of_digest` | theorem | Theorem-Target | Realized digest recovers the exact public-input theorem surface |
| Theorem | `fetchDecodeBound_of_digest` | theorem | Theorem-Target | Realized digest recovers the exact Stage-1 theorem surface |
| Theorem | `memoryBound_of_digest` | theorem | Theorem-Target | Realized digest recovers the exact Stage-2 theorem surface |
| Theorem | `continuityRowBound_of_digest` | theorem | Theorem-Target | Realized digest recovers the exact Stage-3 theorem surface |

## Proof Obligations

- The digest shape must be dictated by theorem ownership, not by Rust export
  convenience.
- The digest must be a normalized boundary over exact public, Stage-1, Stage-2,
  Stage-3, and execution-result surfaces.
- Realized digests must project back to the exact existing theorem surfaces.
- The digest owner must not silently weaken or collapse the distinction between
  Stage-1, Stage-2, and Stage-3 responsibilities.

## Assumption Ledger

- Generic Shout, Twist, and lower-layer PCS theorems are imported.
- The digest owner does not prove cryptographic acceptance on its own.
- Serialization of digest instances between Rust and Lean is external to this
  module.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/Kernel/RomScheduleBinding.lean`
  - `Nightstream/Chip8/Stage2/EvidenceCoverage.lean`
  - `Nightstream/Chip8/Execution/ExecutionSemantics.lean`
  - `Nightstream/Chip8/Execution/BurstSession.lean`
  - `Nightstream/Chip8/Execution/StepComposition.lean`
- **Downstream consumers**:
  - `Nightstream/Chip8/Kernel/ArtifactAudit.lean`
  - Rust-vs-Lean differential testing over normalized digest instances
  - later Rust-refinement theorems over the final kernel proof object

## Implementation Plan

1. Define the normalized staged execution digest and its component surfaces.
2. Define `StagedExecutionDigestBound`.
3. Prove normalization from exact authenticated evidence to a realized digest.
4. Prove the projection theorems back to the exact theorem-owned surfaces.

## Quality Expectations

- Keep the digest owner normalized and ownership-specific.
- Keep the Stage-1 / Stage-2 / Stage-3 split explicit.
- Treat the digest as a theorem-owned contract, not a convenience log format.

## Acceptance Criteria

1. `lake build Nightstream.Chip8.Kernel.StagedExecutionDigest` succeeds.
2. The digest boundary is explicit and Lean-defined.
3. The normalization and projection theorems are explicit.
4. No `sorry`.

## Out of Scope

- Rust serializer implementations
- CI policy
- release policy
- cryptographic verifier acceptance


# /Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/specs/chip8/Chip8ArtifactAudit.spec.md
# Chip8ArtifactAudit Spec

## Purpose

- **What it is**: The Lean-defined audit-checker contract over normalized
  staged execution digests for the final CHIP-8 kernel.
- **Key property**: `artifactAuditSound`: if the audit checker accepts a
  staged execution digest, then that digest satisfies the exact theorem-facing
  realization predicate and therefore carries the semantic facts needed by the
  composition theorem.
- **Protocol role**: This is the theorem-facing audit layer that checks a
  concrete digest instance against the Lean-defined contract.

## Target Formulas

### Audit input

The audit checker consumes one normalized digest instance:

$$
d : \mathrm{StagedExecutionDigest}.
$$

It does not own cryptographic verification. It owns only the semantic audit of
the staged execution boundary after a digest has been produced.

### Executable checker surfaces

Define executable or decidable checks for the digest components:

$$
\mathrm{checkDigestPublicSurface}(d)
$$

$$
\mathrm{checkStage1Surface}(d)
$$

$$
\mathrm{checkStage2Surface}(d)
$$

$$
\mathrm{checkStage3Surface}(d)
$$

$$
\mathrm{checkExecutionResultSurface}(d)
$$

and the bundled checker:

$$
\mathrm{checkStagedExecutionDigest}(d).
$$

The intended meaning is:

- the public check validates the exact theorem-facing public-input boundary
- the Stage-1 check validates the exact Stage-1 digest surface
- the Stage-2 check validates the exact Stage-2 digest surface
- the Stage-3 check validates the exact continuity / bridge digest surface
- the result check validates the exact semantic-result digest surface

### Audit acceptance predicate

Define:

$$
\mathrm{ArtifactAuditAccepted}(d)
$$

to mean that the bundled checker accepts the digest.

Operational policy about when this predicate is enforced is outside this spec.
This owner defines only what checker acceptance means, not when a build system
must run it.

### Checker soundness

The primary theorem target is:

$$
\mathrm{ArtifactAuditAccepted}(d)
\Longrightarrow
\mathrm{StagedExecutionDigestBound}(d,\dots).
$$

This bridges executable checking to the theorem-facing digest realization
predicate.

### Semantic consequence

Because `StagedExecutionDigestBound` projects back to the exact semantic theorem
surfaces, the checker must also support the corollary:

$$
\mathrm{ArtifactAuditAccepted}(d)
\Longrightarrow
\mathrm{ExecutionCorrect}(\dots)
$$

for the exact supported-kernel execution object encoded by `d`, using the
already-owned normalization and projection theorems from
`Chip8StagedExecutionDigest` and `Chip8StepComposition`.

### Audit completeness boundary

The checker is intentionally not a second cryptographic verifier. It is complete
only with respect to the normalized digest contract:

$$
\mathrm{ArtifactAuditAccepted}(d)
$$

means:

- the staged digest is internally coherent,
- each stage surface matches the theorem-owned boundary,
- the stages compose into the semantic result surface.

It does not mean that low-level PCS or transcript verification has been
re-proved inside this module.

## Paper Anchors

- **Sources**:
  - `./docs/assurance-strategy.md`
  - `./docs/soundness-specs/twist-and-shout-requirements.md`
  - `./crates/neo-fold-prototype/specs/chip8-kernel.md`
  - `./docs/superneo-paper`
- Anchors:
  - staged proof composition
  - commitment-before-challenge discipline
  - Stage-1 / Stage-2 / Stage-3 ownership split
  - bridge export and prepared-step linkage
  - release-time audit over one explicit digest boundary

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/Kernel/ArtifactAudit.lean` | Executable/declarative audit checker and soundness theorems over staged execution digests |
| `Nightstream/Chip8/Kernel/ArtifactAuditInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Checker | `checkDigestPublicSurface` | def | Definitional | Checks the exact public digest surface |
| Checker | `checkStage1Surface` | def | Definitional | Checks the exact Stage-1 digest surface |
| Checker | `checkStage2Surface` | def | Definitional | Checks the exact Stage-2 digest surface |
| Checker | `checkStage3Surface` | def | Definitional | Checks the exact Stage-3 digest surface |
| Checker | `checkExecutionResultSurface` | def | Definitional | Checks the exact result digest surface |
| Checker | `checkStagedExecutionDigest` | def | Definitional | Bundled audit checker over one digest instance |
| Acceptance | `ArtifactAuditAccepted` | def | Definitional | Audit acceptance predicate over one digest instance |
| Theorem | `artifactAuditSound` | theorem | Theorem-Target | Accepted digest instances satisfy the exact theorem-facing digest realization predicate |
| Theorem | `artifactAuditImpliesExecutionCorrect` | theorem | Theorem-Target | Accepted digest instances imply the exact supported-kernel execution theorem surface |

## Proof Obligations

- The checker must be defined over the Lean-owned digest contract, not over a
  Rust-owned export format.
- Checker acceptance must imply the exact theorem-facing digest realization
  predicate.
- The checker must preserve the Stage-1 / Stage-2 / Stage-3 ownership split.
- The checker must not silently assume low-level cryptographic verification that
  is not already part of the imported boundary.

## Assumption Ledger

- Cryptographic verification and transcript checking are imported or external.
- Serialization from external artifact format into `StagedExecutionDigest` is
  external to this module.
- This owner checks the staged semantic boundary, not the whole prover stack.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/Kernel/StagedExecutionDigest.lean`
  - `Nightstream/Chip8/Execution/StepComposition.lean`
  - `Nightstream/Chip8/Execution/ExecutionSemantics.lean`
- **Downstream consumers**:
  - release-qualification artifact audits
  - later Rust-refinement theorems for accepted staged digests

## Implementation Plan

1. Define the checker surfaces over the normalized digest contract.
2. Define `ArtifactAuditAccepted`.
3. Prove checker soundness into `StagedExecutionDigestBound`.
4. Prove the semantic consequence theorem into the execution theorem surface.

## Quality Expectations

- Keep the checker contract narrow and theorem-owned.
- Keep executable checks aligned with the exact theorem-facing digest boundary.
- Separate digest checking from external serialization and cryptographic
  verification.

## Acceptance Criteria

1. `lake build Nightstream.Chip8.Kernel.ArtifactAudit` succeeds.
2. Accepted digest instances imply the exact theorem-facing digest realization
   predicate.
3. Accepted digest instances imply the exact supported-kernel semantic result.
4. No `sorry`.

## Out of Scope

- production runtime policy
- CI policy
- Rust feature policy
- low-level cryptographic verification


# /Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/specs/chip8/Chip8BurstSession.spec.md
# Chip8BurstSession Spec

## Purpose

- **What it is**: The theorem-facing session contract for the exact decomposed
  CHIP-8 families in the final kernel, namely `StoreRegs` (`Fx55`) and
  `LoadRegs` (`Fx65`).
- **Key property**: `instructionCorrect_of_burstSession`: if a burst session is
  correctly scheduled, anchored, authenticated, and continuity-aware, then the
  whole macro instruction is semantically correct.
- **Protocol role**: This is the layer that closes decomposed-instruction
  soundness using authenticated execution frames rather than a weak parallel
  list of descriptors and states.

## Target Formulas

### Burst session objects

For a macro decoded instruction `dec`, define a burst session as a finite
sequence of authenticated execution frames:

$$
\mathrm{BurstSession}(frames).
$$

Each frame carries:

- the microstep-local decoded row
- the semantic pre-state
- the semantic post-state
- the authenticated 24-coordinate lane row

The session models the exact decomposition of one macro CHIP-8 instruction into
its chunk-local memory-prefix microsteps.

### Anchoring and chaining

The burst session must be anchored to the macro pre/post states:

$$
\mathrm{BurstAnchored}(dec, pre, post, frames)
$$

meaning:

- the first frame pre-state is the macro pre-state
- the frame at local cursor `x` has post-state equal to the macro post-state

The session must also be chained:

$$
\mathrm{BurstChained}(frames)
\iff
\forall i < n-1,\; frames_i.post = frames_{i+1}.pre.
$$

### Exact schedule derivation

For decomposed families the frame descriptors must be derived from the macro
decoded instruction and the exact final-kernel mem-op handoff:

$$
\mathrm{BurstDerivedFrom}(dec, frames)
$$

meaning:

- every frame descriptor `frames_i.dec` has the same authenticated decoded core
  as `dec`
- `dec.family ∈ \{StoreRegs, LoadRegs\}`
- the microstep cursor is exactly the local list position `i`
- the microstep-local RAM address is exactly `frames_i.pre.I + i`
- the microstep-local mem-op handoff is exact (`IsMemOp = 1`,
  `X_BOUND = dec.x`)
- the final covered frame therefore has `BURST_LAST = 1`

### Coverage and cursor progression

For `Fx55` / `Fx65`-style prefix instructions:

$$
\mathrm{BurstCoversPrefix}(dec, frames)
$$

means the session covers exactly the intended cursor range `0..x`, neither
omitting nor duplicating microsteps.

$$
\mathrm{BurstCursorMonotone}(frames)
$$

means the cursor progresses exactly one step at a time.

### Frame conditions

The session must make explicit which parts of machine state are touched and
which are preserved:

$$
\mathrm{BurstFrameCorrect}(dec, pre, post)
$$

meaning:

- for `StoreRegs`, the RAM prefix `I..I+x` is written from the matching
  register prefix and the register file is otherwise preserved
- for `LoadRegs`, the register prefix `0..x` is written from the matching RAM
  prefix and RAM is otherwise preserved
- `I` and all unsupported machine components are preserved exactly as required
  by the current final kernel

### Authenticated frames and chunk-local continuity

The final kernel is chunk-local and exports row continuity through Stage 3.
The burst layer therefore assumes an imported authenticated frame-trace
boundary rather than proving cross-chunk linkage internally:

$$
\mathrm{BurstFramesBound}(rom,\sigma,frames)
$$

and

$$
\mathrm{BurstContinuityBound}(frames).
$$

This means:

- every frame in the session is authenticated and semantically row-correct
- adjacent microsteps in the same chunk satisfy the internal linking law
- chunk boundaries are linked only through authenticated Stage-3 continuity and
  row-binding claims
- the burst theorem never infers hidden future rows from sparse/random openings

### Whole-instruction theorem

Define:

$$
\mathrm{BurstScheduleCorrect}(rom,\sigma,dec,pre,post,frames)
$$

to mean:

- `BurstSession(frames)`
- `BurstAnchored(dec, pre, post, frames)`
- `BurstChained(frames)`
- `BurstDerivedFrom(dec, frames)`
- `BurstCoversPrefix(dec, frames)`
- `BurstCursorMonotone(frames)`
- `BurstFrameCorrect(dec, pre, post)`
- `BurstFramesBound(rom,\sigma,frames)`
- `BurstContinuityBound(frames)`

Then the target theorem is:

$$
\mathrm{BurstScheduleCorrect}(rom,\sigma,dec,pre,post,frames)
\Longrightarrow
\mathrm{InstructionCorrect}(rom,\sigma,dec,pre,post).
$$

## Paper Anchors

- **Sources**:
  - `./crates/neo-fold-prototype/specs/chip8-kernel.md`
  - `./docs/soundness-specs/twist-and-shout-requirements.md`
- Anchors:
  - decomposed instruction schedule for memory-prefix families
  - exact address/cursor binding for CHIP-8 burst instructions
  - chunk-local continuity exported through Stage 3

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/Execution/BurstSession.lean` | Burst-session correctness theorems for decomposed CHIP-8 instructions |
| `Nightstream/Chip8/Execution/BurstSessionInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Sessions | `BurstSession` | def | Definitional | Packages one authenticated burst-trace session |
| Sessions | `BurstAnchored` | def | Definitional | Connects macro pre/post state to the authenticated session endpoints |
| Sessions | `BurstChained` | def | Definitional | Enforces post-to-next-pre chaining |
| Sessions | `BurstDerivedFrom` | def | Definitional | Derives each frame descriptor from the macro decoded instruction |
| Sessions | `BurstCoversPrefix` | def | Definitional | Enforces exact microstep coverage |
| Sessions | `BurstCursorMonotone` | def | Definitional | Enforces exact cursor progression |
| Sessions | `BurstFrameCorrect` | def | Definitional | Enforces touched-address correctness and untouched-state preservation |
| Sessions | `BurstFramesBound` | def | Definitional | Every frame carries authenticated row-local semantic correctness |
| Sessions | `BurstContinuityBound` | def | Definitional | Imports the exact chunk-local continuity boundary used by the final kernel |
| Sessions | `BurstScheduleCorrect` | def | Definitional | Complete authenticated burst-session correctness condition |
| Theorem | `instructionCorrect_of_burstSession` | theorem | Theorem-Target | Correct authenticated burst session implies whole-instruction correctness |

## Proof Obligations

- `BurstScheduleCorrect` must be stronger than a bare list-equality or
  membership predicate.
- The theorem surface must make the state chaining explicit.
- The theorem surface must make the cursor/address derivation explicit.
- The theorem surface must make the authenticated frame evidence explicit.
- The theorem surface must respect the final kernel's chunk-local continuity
  boundary instead of implicitly assuming one global contiguous trace witness.

## Assumption Ledger

- `MicrostepCorrect`, `InstructionCorrect`, and `ExecutionFrameBound` are
  imported from `Chip8ExecutionSemantics`.
- The exact register/address cursor semantics are imported from the decode and
  address-binding layers.
- Continuity across chunk boundaries is imported from the Stage-3
  continuity/bridge layer.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/Execution/ExecutionSemantics.lean`
  - `Nightstream/Chip8/Stage1/DecodeAddressBinding.lean`
  - `Nightstream/Chip8/Stage3/ContinuityBridge.lean`
- **Downstream consumers**:
  - later ROM-specific whole-instruction and execution theorems

## Implementation Plan

1. Define the authenticated burst session predicates over execution frames.
2. State the exact strengthened `BurstScheduleCorrect`.
3. Prove `instructionCorrect_of_burstSession`.

## Quality Expectations

- Keep the burst layer explicit about chaining, authentication, and framing.
- Avoid weak schedule predicates that could overclaim instruction correctness.
- Keep the theorem exact for the supported `StoreRegs` / `LoadRegs` families.

## Acceptance Criteria

1. `lake build Nightstream.Chip8.Execution.BurstSession` succeeds.
2. The theorem surface explicitly closes the decomposed-instruction schedule
   hole for the selected CHIP-8 families.
3. No `sorry`.

## Out of Scope

- re-proving local microstep correctness
- proving authenticated claim coverage
- proving PCS binding or Fiat-Shamir security
- extending the theorem to unsupported CHIP-8 families such as `DRW`, stack
  operations, timers, or keypad-dependent rows


# /Users/nicolasarqueros/starstream/develop/nightstream-clean-up/formal/nightstream-lean/specs/chip8/Chip8StepComposition.spec.md
# Chip8StepComposition Spec

## Purpose

- **What it is**: The theorem-facing semantic composition contract for the
  final supported CHIP-8 kernel.
- **Key property**: `microstepCorrect_of_bounds`: if the local row relation,
  Stage-1 fetch/decode/lookup facts, Stage-2 memory facts, and authenticated
  lane-row bindings all hold, then the current supported CHIP-8 row is
  semantically correct.
- **Protocol role**: This is the layer where Nightstream proves the
  CHIP-8-specific glue from the final kernel's local and staged theorem
  surfaces to machine-level execution semantics.

This module is intentionally semantic. It consumes already-extracted semantic
facts. It does **not** derive those facts from opening manifests or proof
objects; that bridge belongs to `Chip8EvidenceCoverage`.

## Target Formulas

### Semantic objects

The composition layer reasons over:

- `DecodedRow`: the authenticated per-row supported-kernel descriptor
- `MachineState`: the semantic state for the supported kernel subset
  (`pc_word`, `i`, register file, RAM)
- `InitialState`: the authenticated chunk-initial state

The current supported subset is exactly:

- `LdImm`
- `AddImm`
- `Mov`
- `AddRegNoCarry`
- `SkipEqImm`
- `Jump`
- `LdI`
- `StoreRegs`
- `LoadRegs`

### Imported boundary relations

The composition theorem consumes the following imported or previously proved
relations.

$$
\mathrm{WitnessBinds}(pre, post, dec, z)
$$

This binds the 24-coordinate lane row to the semantic state objects and the
decoded row.

$$
\mathrm{FetchDecodeBound}(romTable, pre.PC, dec)
$$

This means the committed ROM row at the current absolute word address decodes
to the exact authenticated row descriptor.

$$
\mathrm{LookupBound}(dec, pre, z)
$$

This means `LOOKUP_OUTPUT` is the exact semantic helper result required by the
current row.

$$
\mathrm{MemoryBound}(pre, post, init, dec, z)
$$

This means the memory-derived lane values and Stage-2 register/RAM objects are
semantically correct for the current row.

$$
\mathrm{chip8RowLocalSound}(z)
$$

This is the local row-local consequence imported from `Chip8Routing`.

### Microstep correctness

The primary theorem target is:

$$
\mathrm{wf}(z)
\land
\mathrm{WitnessBinds}(pre, post, dec, z)
\land
\mathrm{FetchDecodeBound}(romTable, pre.PC, dec)
\land
\mathrm{LookupBound}(dec, pre, z)
\land
\mathrm{MemoryBound}(pre, post, init, dec, z)
\land
\mathrm{chip8RowLocalSound}(z)
\Longrightarrow
\mathrm{MicrostepCorrect}(romTable, init, dec, pre, post).
$$

For the supported exact families, `MicrostepCorrect` must express the exact
final-kernel row semantics:

- `LdImm`: `REG_X_NEXT = KK`, `PC_NEXT = PC + 1`
- `AddImm`: low-byte add to `V[x]`, `PC_NEXT = PC + 1`
- `Mov`: `REG_X_NEXT = REG_Y`, `PC_NEXT = PC + 1`
- `AddRegNoCarry`: low-byte add of `REG_X` and `REG_Y`, no `VF` side effect,
  `PC_NEXT = PC + 1`
- `SkipEqImm`: `PC_NEXT = PC + 1 + LOOKUP_OUTPUT`
- `Jump`: `PC_NEXT = NNN_WORD`
- `LdI`: `I_NEXT = NNN_ADDR`, `PC_NEXT = PC + 1`
- `StoreRegs` row: `RAM_ADDR = I_REG + X_IDX`, `REG_X_NEXT = REG_X`,
  `PC_NEXT = PC + BURST_LAST`
- `LoadRegs` row: `RAM_ADDR = I_REG + X_IDX`, `REG_X_NEXT = MEM_VALUE`,
  `PC_NEXT = PC + BURST_LAST`

### Whole-instruction correctness for decomposed instructions

For `Fx55` and `Fx65`, whole-instruction correctness is owned by
`Chip8BurstSession`, which consumes the shared execution semantics from
`Chip8ExecutionSemantics` together with authenticated frame/session bounds.
This module imports that result; it does not re-own the burst schedule
predicate locally.

### Chunk execution correctness

The final kernel is chunk-local and Stage 3 owns continuity. Define:

$$
\mathrm{ExecutionCorrect}(romTable, init, trace)
$$

to mean:

- every row in the trace satisfies `MicrostepCorrect`
- consecutive rows satisfy the authenticated Stage-3 continuity relation
- the first row agrees with the authenticated public initial state
- the chunk begins on an instruction boundary with no in-flight burst state

This is the supported-kernel execution theorem surface exported by the current
final kernel.

### Prepared-step correctness

Let `PreparedStepTraceBound(trace, preparedSteps)` mean that the exported
prepared steps are exactly the Stage-3 bridge images of the semantic rows in the
trace.

The final execution-to-bridge theorem target is:

$$
\mathrm{ExecutionCorrect}(romTable, init, trace)
\land
\mathrm{PreparedStepTraceBound}(trace, preparedSteps)
\Longrightarrow
\text{the root main lane receives the exact authenticated semantic rows}.
$$

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/chip8-kernel.md`
- Anchors:
  - supported opcode coverage
  - row-local routing relation
  - Stage-2 memory semantics
  - Stage-3 continuity and bridge binding

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/Execution/StepComposition.lean` | Composition theorems from final-kernel bounds to supported-subset CHIP-8 semantics |
| `Nightstream/Chip8/Execution/StepCompositionInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Semantic objects | `DecodedRow` | def | Definitional | Authenticated per-row supported-kernel descriptor |
| Semantic objects | `MachineState` | def | Definitional | Semantic state space for the supported CHIP-8 subset |
| Semantic objects | `InitialState` | def | Definitional | Authenticated chunk-initial state |
| Bindings | `WitnessBinds` | def | Definitional | Authenticated lane-row binding |
| Bindings | `FetchDecodeBound` | def | Definitional | Authenticated Stage-1 fetch/decode binding |
| Bindings | `LookupBound` | def | Definitional | Authenticated Stage-1 helper-lookup semantics |
| Bindings | `MemoryBound` | def | Definitional | Authenticated Stage-2 semantic memory binding |
| Semantics | `MicrostepCorrect` | def | Definitional | Semantic correctness of one supported-kernel row |
| Semantics | `InstructionCorrect` | def | Definitional | Semantic correctness of one whole supported instruction |
| Semantics | `BurstScheduleCorrect` | def | Imported-Definitional | Authenticated burst-session schedule/binding predicate imported from `Chip8BurstSession` |
| Semantics | `ExecutionCorrect` | def | Definitional | Semantic correctness of one chunk-local execution trace |
| Bridge | `PreparedStepTraceBound` | def | Definitional | Exported prepared steps are exactly the Stage-3 images of the semantic rows |
| Theorem | `microstepCorrect_of_bounds` | theorem | Theorem-Target | Final-kernel row bounds imply row semantics |
| Theorem | `instructionCorrect_of_burst` | theorem | Imported-Theorem | Authenticated burst-session bounds imply whole-instruction correctness for `StoreRegs` / `LoadRegs` |
| Theorem | `executionCorrect_of_trace` | theorem | Theorem-Target | Correct rows plus continuity imply chunk execution correctness |
| Theorem | `preparedStepTraceBound_of_execution` | theorem | Theorem-Target | Correct execution plus Stage-3 binding yields exact prepared-step export |

## Proof Obligations

- `MicrostepCorrect` must match the final supported-kernel semantics exactly,
  including `AddRegNoCarry`, `NNN_ADDR` vs `NNN_WORD`, and `PC_NEXT = PC + BURST_LAST`
  on memory-prefix rows.
- This module must target the final 24-coordinate row.
- `ExecutionCorrect` must use authenticated continuity and initial-state facts,
  not an ad hoc list-linking predicate alone.
- The composition theorem must remain exact to the currently supported subset; a
  future `DRW`/keypad/timer kernel is a separate extension.

## Assumption Ledger

- The semantic facts consumed here are expected to be supplied by
  `Chip8EvidenceCoverage`.
- The root main-lane CCS proof remains external to this module.
- Full-game semantics for unsupported opcodes are outside the current kernel.

## Dependency and Consumer Map

- **Upstream dependencies**:
  - `Nightstream/Chip8/Stage1/Routing.lean`
  - `Nightstream/Chip8/Stage1/FetchDecodeBinding.lean`
  - `Nightstream/Chip8/Stage2/WitnessMemoryBinding.lean`
  - `Nightstream/Chip8/Execution/ExecutionSemantics.lean`
  - `Nightstream/Chip8/Execution/BurstSession.lean`
  - `Nightstream/Chip8/Stage3/ContinuityBridge.lean`
  - `Nightstream/Chip8/Stage2/EvidenceCoverage.lean`
- **Downstream consumers**:
  - `Nightstream/Chip8/Kernel/StagedExecutionDigest.lean`
  - `Nightstream/Chip8/Kernel/ArtifactAudit.lean`
  - later Rust-refinement theorems for the final kernel proof object
  - later larger CHIP-8 kernels that add draw/input/timer families

## Implementation Plan

1. Define the exact row semantics for the supported subset.
2. Prove the row-level composition theorem.
3. Import and re-export the burst whole-instruction theorem from `Chip8BurstSession`.
4. Prove chunk execution correctness and prepared-step export correctness.

## Quality Expectations

- Keep the module exact to the final kernel.
- Keep unsupported game-oriented families out of the local theorem surface.
- Keep continuity and bridge facts imported explicitly rather than hidden.

## Acceptance Criteria

1. `lake build Nightstream.Chip8.Execution.StepComposition` succeeds.
2. The theorem surface matches the final supported kernel exactly.
3. The execution theorem is continuity-aware and chunk-local.
4. No `sorry`.

## Out of Scope

- full CHIP-8 ISA correctness
- `DRW`, keypad, timer, stack, and sound semantics
- the root main-lane CCS proof itself
