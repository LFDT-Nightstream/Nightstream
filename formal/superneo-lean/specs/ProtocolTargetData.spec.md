# ProtocolTargetData Spec

## Purpose

- **What it is**: An explicit protocol-side owner for the Section 7.5 inputs on the paper-facing challenge-difference route.
- **Key property**: The actual theorem ingredients stay visible as fields: one Theorem-3 witness for `ctx.bar`, one arithmetic-obligation package, and one proof that `ctx.invDelta` is a nonzero difference of paper-carrier elements.
- **Protocol role**: This is the smallest protocol-side source from which the repo can canonically derive `protocolTargetProp ctx` without routing through a compatibility assumption bundle.

## Target Formulas

- `ProtocolTargetData.invDeltaInvertible : invertibleRq ctx.invDelta`
- `ProtocolTargetData.protocolTargetProp : protocolTargetProp ctx`
- `ProtocolTargetData.assumptions : ProtocolTargetAssumptions ctx`
- `ProtocolTargetData.ofPaperCarrierDiff : thm3CoreAssumption ctx.bar → ArithmeticObligations ... → samplingDiffSet paperCarrier ctx.invDelta → ctx.invDelta ≠ zeroRq → ProtocolTargetData ctx`
- `ProtocolTargetData.ofBasisKernelAssumption : thm3BasisKernelAssumption ctx.bar → ArithmeticObligations ... → samplingDiffSet paperCarrier ctx.invDelta → ctx.invDelta ≠ zeroRq → ProtocolTargetData ctx`
- `ProtocolTargetData.ofBasisKernelCheck : thm3BasisKernelCheck ctx.bar = true → ArithmeticObligations ... → samplingDiffSet paperCarrier ctx.invDelta → ctx.invDelta ≠ zeroRq → ProtocolTargetData ctx`
- `ProtocolTargetData.ofNativePaperCarrierDiff : ctx.bar = nativeBarMatrix → ArithmeticObligations ... → samplingDiffSet paperCarrier ctx.invDelta → ctx.invDelta ≠ zeroRq → ProtocolTargetData ctx`
- `protocolTargetProp_of_data : ProtocolTargetData ctx → protocolTargetProp ctx`

## Paper Anchors

- Source: `./formal/superneo-lean/SuperNeo.pdf.md`
- Section 7, lines 447-481: compact protocol relations and global reduction parameters
- Section 7.3-7.5, lines 481-596: protocol reductions consuming the compact target proposition

## Module Mapping

- Implementation: `SuperNeo.ProtocolTargetData`
- Interface: `SuperNeo.ProtocolTargetDataInterface`

## Contract Surface

| Group | Lean surface | Guarantee | Role |
|---|---|---|---|
| Context | `ProtocolTargetData` | Owns one compact target's theorem-side Section 7.5 inputs on the paper-facing challenge-difference route | Theorem-Target |
| Projection | `ProtocolTargetData.invDeltaInvertible` | Recover invertibility of `ctx.invDelta` from the carried paper-facing difference data | Theorem-Target |
| Projection | `ProtocolTargetData.protocolTargetProp` | Recover the compact protocol target proposition | Theorem-Target |
| Projection | `ProtocolTargetData.assumptions` | Recover the legacy compatibility assumption bundle | Theorem-Target |
| Constructor | `ProtocolTargetData.ofPaperCarrierDiff` | Build the protocol-side owner from the direct paper-facing route | Theorem-Target |
| Constructor | `ProtocolTargetData.ofBasisKernelAssumption` | Build the protocol-side owner from the finite basis-kernel Theorem-3 characterization | Theorem-Target |
| Constructor | `ProtocolTargetData.ofBasisKernelCheck` | Build the protocol-side owner from the executable finite basis-kernel checker | Theorem-Target |
| Constructor | `ProtocolTargetData.ofNativePaperCarrierDiff` | Build the protocol-side owner from the active native-bar route | Theorem-Target |
| Theorem | `protocolTargetProp_of_data` | One protocol-side target-data owner implies `protocolTargetProp` | Theorem-Target |

## Proof Obligations

- `ProtocolTargetData.protocolTargetProp` must derive `protocolTargetProp` without introducing any extra local assumptions.
- `ProtocolTargetData.invDeltaInvertible` must recover invertibility only from the carried paper-facing difference proof and nonzero proof.
- The native and basis-kernel constructors must reduce back to the same single protocol-side owner.

## Assumption Ledger

- This module introduces no new theorem-level assumptions.
- The carried Theorem-3 witness, arithmetic obligations, and paper-carrier difference proof are explicit fields of the protocol-side data object.
- Concrete construction of the data object from a specific protocol setup remains an upstream instantiation task.

## Dependency and Consumer Map

- Upstream dependencies:
  - `SuperNeo/ProtocolTarget.lean`
- Downstream consumers:
  - `SuperNeo/ProtocolRelations.lean`
  - `SuperNeo/PiCCS.lean`
  - `SuperNeo/PiRLC.lean`
  - `SuperNeo/PiDEC.lean`
  - `SuperNeo/InteractiveReductions.lean`

## Quality Expectations

- Keep the component thin and paper-facing.
- Keep the paper-facing challenge-difference inputs explicit instead of collapsing them into opaque compatibility bundles.
- Do not duplicate proof content already owned by `ProtocolTarget`.

## Acceptance Criteria

1. `lake build` succeeds.
2. `lake exe check` succeeds.
3. No `sorry`.
