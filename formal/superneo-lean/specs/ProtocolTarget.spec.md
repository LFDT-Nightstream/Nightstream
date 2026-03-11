# ProtocolTarget Spec

## Purpose

- **What it is**: A layer that binds Theorem 3 and arithmetic obligations into one target context (`ProtocolTargetContext`), then derives the core target proposition `protocolTargetProp` used by protocol relations.
- **Key property**: `protocolTargetProp ctx` is derivable both from the legacy compatibility bundles (`ProtocolTargetAssumptions ctx`, `ProtocolTargetNativeAssumptions ctx`) and directly from the active paper-facing inputs (`thm3`/native bar, arithmetic obligations, and the `paperCarrier` difference route for `ctx.invDelta`); the theorem-native single-owner version of those direct inputs lives in `ProtocolTargetData`.
- **Protocol role**: ProtocolRelations uses `protocolTargetProp` to define CCS/CE relations; PiCCS and downstream reductions depend on this target.

## Target Formulas (Paper → Lean)

- `protocolTargetProp ctx ↔ thm3CoreAssumption ctx.bar ∧ splitBase2TerminalZeroProp ctx.splitScalar ctx.kSplit ∧ evalHomAssumption ... ∧ vecModuleAssumption ... ∧ scalarModuleAssumption ... ∧ samplingExpansionProp ... ∧ qVals.size = 2^r.size ∧ mleEval qVals r = mleInnerProductForm qVals r ∧ interpolationProp ... ∧ invertibleRq ctx.invDelta`
- `ProtocolTargetAssumptions ctx → protocolTargetProp ctx`
- `ProtocolTargetNativeAssumptions ctx → protocolTargetProp ctx`
- `thm3BasisKernelAssumption ctx.bar → ArithmeticObligations ... → invertibleRq ctx.invDelta → protocolTargetProp ctx`
- `thm3BasisKernelCheck ctx.bar = true → ArithmeticObligations ... → invertibleRq ctx.invDelta → protocolTargetProp ctx`
- `thm3CoreAssumption ctx.bar → ArithmeticObligations ... → samplingDiffSet paperCarrier ctx.invDelta → ctx.invDelta ≠ 0 → protocolTargetProp ctx`
- `ctx.bar = nativeBarMatrix → ArithmeticObligations ... → samplingDiffSet paperCarrier ctx.invDelta → ctx.invDelta ≠ 0 → protocolTargetProp ctx`

## Paper Anchors

- Source: `./formal/superneo-lean/SuperNeo.pdf.md`
- Anchors:
  - Section 7 (Neo's folding scheme for CCS), lines 447–481: Relations (Definitions 11–13), Global Reduction Parameters (Definition 14)
  - Section 7.3 (Π_CCS), lines 481–547: Interactive reduction for CCS

## Module Mapping

- Implementation: `SuperNeo.ProtocolTarget`
- Interface: `SuperNeo.ProtocolTargetInterface`

## Contract Surface

| Contract group | Lean surface | Preconditions | Guarantee | Role | Used by |
|---|---|---|---|---|---|
| Context | `ProtocolTargetContext` | None | Bundles bar, m, r, rho1, rho2, hVec, hScal, splitScalar, kSplit, invDelta, cset, samples, xs, ys, qVals, coeffs, xEval, expectedEval | Definitional | ProtocolRelations, PiCCS |
| Assumptions | `ProtocolTargetAssumptions ctx` | None | Bundles thm3, arithmetic (ArithmeticObligations), direct witness `invertibleRq ctx.invDelta` | Definitional | — |
| Assumptions | `ProtocolTargetNativeAssumptions ctx` | `ctx.bar = nativeBarMatrix` | Bundles native bar equality, arithmetic (ArithmeticObligations), direct witness `invertibleRq ctx.invDelta` | Definitional | ProtocolRelations native path |
| Invertibility bridge | `strictInvertibilityWindowProp_five_of_paperCarrierDiff` | `samplingDiffSet paperCarrier δ`, `δ ≠ 0` | Strict paper-faithful window `< 5` | Theorem-Target | Protocol-facing invertibility assembly |
| Invertibility bridge | `invertibleRq_of_paperCarrierDiff` | `samplingDiffSet paperCarrier δ`, `δ ≠ 0` | `invertibleRq δ` | Theorem-Target | Protocol-facing invertibility assembly on the active Goldilocks path |
| Constructor | `ProtocolTargetAssumptions.ofPaperCarrierDiff` | thm3 + arithmetic + `samplingDiffSet paperCarrier ctx.invDelta` + `ctx.invDelta ≠ 0` | Canonical protocol-target bundle on the paper-facing challenge-difference path | Theorem-Target | ProtocolRelations / reductions |
| Constructor | `ProtocolTargetAssumptions.ofBasisKernelAssumption` | finite basis-kernel Thm-3 witness + arithmetic + `samplingDiffSet paperCarrier ctx.invDelta` + `ctx.invDelta ≠ 0` | Canonical protocol-target bundle on the paper-facing challenge-difference path without a raw `thm3CoreAssumption` | Theorem-Target | ProtocolRelations / reductions |
| Constructor | `ProtocolTargetAssumptions.ofBasisKernelCheck` | executable finite basis-kernel checker + arithmetic + `samplingDiffSet paperCarrier ctx.invDelta` + `ctx.invDelta ≠ 0` | Canonical protocol-target bundle on the paper-facing challenge-difference path from the checker surface | Theorem-Target | ProtocolRelations / reductions |
| Constructor | `ProtocolTargetAssumptions.ofLowNormAtLeastFive` | thm3 + arithmetic + `lowNormInvertibilityAssumption B` with `5 ≤ B` + `samplingDiffSet paperCarrier ctx.invDelta` + `ctx.invDelta ≠ 0` | Canonical protocol-target bundle through the stronger strict low-norm route | Theorem-Target | ProtocolRelations / reductions |
| Constructor | `ProtocolTargetNativeAssumptions.ofPaperCarrierDiff` | native bar + arithmetic + `samplingDiffSet paperCarrier ctx.invDelta` + `ctx.invDelta ≠ 0` | Native protocol-target bundle on the same path | Theorem-Target | ProtocolRelations / reductions |
| Constructor | `ProtocolTargetNativeAssumptions.ofLowNormAtLeastFive` | native bar + arithmetic + `lowNormInvertibilityAssumption B` with `5 ≤ B` + `samplingDiffSet paperCarrier ctx.invDelta` + `ctx.invDelta ≠ 0` | Native protocol-target bundle through the stronger strict low-norm route | Theorem-Target | ProtocolRelations / reductions |
| Target prop | `protocolTargetProp ctx` | None | Conjunction of all protocol-target predicates | Definitional | ProtocolRelations |
| Derivation | `protocolTargetProp_of_assumptions` | `ProtocolTargetAssumptions ctx` | `protocolTargetProp ctx` | Theorem-Target | ProtocolRelations |
| Derivation | `protocolTargetProp_of_basisKernelAssumption` | finite basis-kernel Thm-3 witness + arithmetic + `invertibleRq ctx.invDelta` | `protocolTargetProp ctx` | Theorem-Target | ProtocolRelations |
| Derivation | `protocolTargetProp_of_basisKernelCheck` | executable finite basis-kernel checker + arithmetic + `invertibleRq ctx.invDelta` | `protocolTargetProp ctx` | Theorem-Target | ProtocolRelations |
| Derivation | `protocolTargetProp_of_native_assumptions` | `ProtocolTargetNativeAssumptions ctx` | `protocolTargetProp ctx` | Theorem-Target | ProtocolRelations native path |
| Derivation | `protocolTargetProp_of_paperCarrierDiff` | thm3 + arithmetic + `samplingDiffSet paperCarrier ctx.invDelta` + `ctx.invDelta ≠ 0` | `protocolTargetProp ctx` directly on the active paper-facing route | Theorem-Target | ProtocolRelations / PiCCS / PiRLC / PiDEC |
| Derivation | `protocolTargetProp_of_native_paperCarrierDiff` | native bar + arithmetic + `samplingDiffSet paperCarrier ctx.invDelta` + `ctx.invDelta ≠ 0` | `protocolTargetProp ctx` directly on the active native paper-facing route | Theorem-Target | ProtocolRelations / PiCCS / PiRLC / PiDEC |

## Proof Obligations and Closure Plan

- `protocolTargetProp` must be derivable from the compatibility bundles, from direct active paper-facing/native paper-facing inputs, and from finite basis-kernel Theorem-3 providers.
- Canonical bundle constructors must internalize the active paper-facing invertibility bridge from `samplingDiffSet paperCarrier ctx.invDelta ∧ ctx.invDelta ≠ 0` without introducing an extra local invertibility boundary.
- The stronger strict low-norm theorem route `lowNormInvertibilityAssumption B` with `5 ≤ B` remains an alternative upstream provider surface.

## Assumption Ledger

- This module introduces no theorem-level boundary beyond `ProtocolTargetAssumptions` / `ProtocolTargetNativeAssumptions` and the upstream theorem providers they bundle.
- Native closure depends on upstream `thm3CoreAssumption_native` from `Thm3Core`.
- The concrete source of `ctx.invDelta` as a nonzero paper-carrier difference remains an upstream protocol fact, not something derived here.

## Dependency and Consumer Map

- Upstream dependencies:
  - `SuperNeo/Thm3Core.lean`: imports `thm3CoreAssumption`
  - `SuperNeo/ArithmeticObligations.lean`: uses `ArithmeticObligations` for arithmetic bundle
- Downstream consumers:
  - `SuperNeo/ProtocolTargetData.lean`: packages the paper-facing Section 7.5 inputs as one theorem-native owner
  - `SuperNeo/ProtocolRelations.lean`: uses `protocolTargetProp`, `protocolTargetProp_of_assumptions`, `ProtocolTargetContext` to define CCS/CE relations
  - `SuperNeo/PiCCS.lean`: depends on ProtocolRelations
  - `SuperNeo/ProtocolTheorem.lean`: uses `ProtocolTargetContext` for final theorem shape

## Implementation Plan

1. `ProtocolTargetContext` structure holds all protocol parameters.
2. `ProtocolTargetAssumptions` bundles thm3, arithmetic obligations, and a direct `invertibleRq` witness for `ctx.invDelta`.
3. `ProtocolTargetNativeAssumptions` bundles `ctx.bar = nativeBarMatrix`, arithmetic obligations, and a direct `invertibleRq` witness for `ctx.invDelta`.
4. `ProtocolTargetAssumptions.ofPaperCarrierDiff` and `...Native...` derive those direct witnesses from the active paper-facing `paperCarrier`-difference path using the proved Goldilocks theorem directly.
5. `protocolTargetProp` defined as conjunction of target predicates.
6. `protocolTargetProp_of_assumptions` and `protocolTargetProp_of_native_assumptions` proved by projection from the direct witness bundle.

## Quality Expectations

- No `sorry` in any theorem.
- All declarations proved natively.

## Acceptance Criteria

1. `lake build` succeeds.
2. `lake exe check` succeeds.
3. All surfaces exported through the interface.

## Out of Scope

- Concrete instantiation of `ProtocolTargetAssumptions` / `ProtocolTargetNativeAssumptions`; that belongs to protocol setup.
- `matrixTransformAssumption_of_thm3CoreAssumption` is re-exported from MatrixTransform for consumers; closure is in MatrixTransform.
