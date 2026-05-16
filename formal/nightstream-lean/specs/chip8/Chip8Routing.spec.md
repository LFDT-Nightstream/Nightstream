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
| `Nightstream/Chip8/Routing.lean` | Local row-local routing lemma over the 24-coordinate CHIP-8 main lane |
| `Nightstream/Chip8/RoutingInterface.lean` | Theorem-facing re-export surface |

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
  - `Nightstream/Chip8/StepComposition.lean`
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

1. `lake build Nightstream.Chip8.Routing` succeeds.
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
