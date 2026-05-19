# Rv64IMRamHistoryProjection Spec

## Purpose

- **What it is**: The theorem-facing Stage-2 owner for RV64IM RAM-history projection.
- **What it is not**: It is not the Stage-1 narrow-memory helper arithmetic and it does not own Stage-3 continuity.
- **Protocol role**: It fixes the merged RAM-history family, chunk-address flattening, row-local RAM read/write semantics, and the Jolt-style write-value virtualization shape.

## Target Formulas

Define the concrete family id:

$$
\mathrm{RamHistoryFamily} := \mathrm{ramHistory}.
$$

Define chunk-address flattening recursively:

$$
\mathrm{flattenRamAddr}([(b_0,a_0), \ldots, (b_{d-1}, a_{d-1})])
$$

as the canonical scalar RAM word index induced by the ordered chunk values.

Define:

$$
\mathrm{RamAddressVirtualizationBound}(\mathrm{chunkedAddr}, flatAddr)
\iff
flatAddr = \mathrm{flattenRamAddr}(\mathrm{chunkedAddr}).
$$

Define one row-local RAM-history record:

$$
\mathrm{RamHistoryRow}
$$

carrying at least:

- `activeMem`,
- `isLoad`,
- `isStore`,
- `addrChunks`,
- `readWord`,
- `writeWord`,
- `memVal`,
- `rs2Val`,
- `deltaWord`.

Define the row-local RAM boundary:

$$
\mathrm{RamHistoryRowBound}(0_{ram}, \mathrm{combine}, row)
$$

meaning:

- inactive rows force `deltaWord = 0_ram`,
- load rows satisfy `memVal = readWord`,
- store rows satisfy
  `memVal = rs2Val`,
  `writeWord = memVal`,
  and
  `writeWord = combine(readWord, deltaWord)`,
- non-store rows force `deltaWord = 0_ram`.

Define:

$$
\mathrm{RamHistoryProjection}(p)
:=
\mathrm{TwistValProjection}(\mathrm{RamHistoryFamily}, p).
$$

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/riscv-kernel.md`
- Anchors:
  - RAM domain and ports
  - merged RAM address family
  - RAM address support relation
  - Stage-2 linkage batch

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Rv64IM/Stage2/RamHistoryProjection.lean` | RAM-history projection owner |
| `Nightstream/Rv64IM/Stage2/RamHistoryProjectionInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Family id | `ramHistoryFamily` | def | Definitional | Fixes the concrete Stage-2 RAM-history family |
| Address | `flattenRamAddr` | def | Definitional | Canonical flattening map from chunk address to RAM word index |
| Address | `RamAddressVirtualizationBound` | def | Definitional | The chunked RAM address determines one scalar RAM word index |
| Row | `RamHistoryRow` | structure | Definitional | Packages one row-local RAM read/write surface |
| Boundary | `RamHistoryRowBound` | def | Definitional | Fixes load/store/no-op RAM semantics including virtualized write value |
| Bundle | `RamHistoryBundle` | structure | Definitional | Packages the RAM timeline and row-local Stage-2 RAM facts |
| Theorem | `ramHistoryRowBound_memVal_of_load` | theorem | Theorem-Target | Load rows expose the authenticated read word as `MEM_VAL` |
| Theorem | `ramHistoryRowBound_storePayload` | theorem | Theorem-Target | Store rows bind `MEM_VAL`, `RS2`, and the authenticated write word together |
| Theorem | `ramHistoryRowBound_zeroDelta_of_not_store` | theorem | Theorem-Target | Non-store rows have zero RAM delta |
| Projection | `ramHistoryProjection` | def | Definitional | Concrete Twist projection for the RAM-history family |

## Out of Scope

- Stage-1 extract/blend helper arithmetic
- Stage-3 continuity
- final ISA-equivalence
