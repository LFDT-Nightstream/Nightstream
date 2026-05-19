# Chip8ExtensionFamily Spec

## Purpose

- **What it is**: The exact theorem-facing family-id surface for CHIP-8
  release-path extension families.
- **What it is not**: It is not a proof object and it does not itself classify
  any family into the main lane or a separate fold.
- **Protocol role**: It fixes the family vocabulary shared by the generic
  Nightstream bridge and the CHIP-8 release-path family owners.

## Target Formulas

The family-id type is:

$$
\mathrm{ExtensionFamily} :=
\{
\mathrm{BytecodeFetch},
\mathrm{InstructionSemanticsLookup},
\mathrm{RegisterHistory},
\mathrm{RamHistory}
\}.
$$

These are the only release-path CHIP-8 extension families owned by this
surface.

The intended ownership split is:

- `BytecodeFetch`: readonly ROM fetch / bytecode verification family
- `InstructionSemanticsLookup`: Stage-1 instruction-lookup family
- `RegisterHistory`: register-side Twist history family
- `RamHistory`: RAM-side Twist history family

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Family ids | `ExtensionFamily` | def | Definitional | Enumerates the exact CHIP-8 release-path extension families |

## Dependency and Consumer Map

- **Upstream dependencies**: none
- **Downstream consumers**:
  - `Nightstream/Chip8/ReleaseBridge.lean`
  - `Nightstream/Chip8/Stage1/BytecodeFetchProjection.lean`
  - `Nightstream/Chip8/Stage1/InstructionSemanticsLookupProjection.lean`
  - `Nightstream/Chip8/Stage2/RegisterHistoryProjection.lean`
  - `Nightstream/Chip8/Stage2/RamHistoryProjection.lean`

## Proof Obligations

- The family list must stay aligned with the Rust release-path family boundary.
- New CHIP-8 extension families must be added here before any bridge owner may
  route them.

## Paper Anchors

- **Source**:
  - `./docs/assurance-strategy.md`
  - `./crates/neo-fold-prototype/src/proof.rs`

## Out of Scope

- main-lane admissibility
- separate-fold admissibility
- transcript or PCS binding
