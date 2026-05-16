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
| `Nightstream/Chip8/DecodeAddressBinding.lean` | Exact family-specific address projection and sink-routing theorems |
| `Nightstream/Chip8/DecodeAddressBindingInterface.lean` | Theorem-facing re-export surface |

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
  - `Nightstream/Chip8/FetchDecodeBinding.lean`
- **Downstream consumers**:
  - `Nightstream/Chip8/WitnessMemoryBinding.lean`
  - `Nightstream/Chip8/EvidenceCoverage.lean`
  - `Nightstream/Chip8/StepComposition.lean`

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

1. `lake build Nightstream.Chip8.DecodeAddressBinding` succeeds.
2. The theorem surface covers all exact Stage-1 and Stage-2 families from the
   final kernel.
3. No unnamed address source remains.
4. No `sorry`.

## Out of Scope

- generic Shout proofs
- generic Twist proofs
- continuity
- bridge binding
