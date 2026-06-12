# ProtocolTarget Spec

## Purpose

- **What it is**: A layer that binds Theorem 3 and arithmetic obligations into one target context (`ProtocolTargetContext`), then derives the core target proposition `protocolTargetProp` used by protocol relations.
- **Key property**: `protocolTargetProp ctx` is derivable from `ProtocolTargetAssumptions ctx`; the active paper-facing route packages its evidence through `ProtocolTargetAssumptions.ofPaperCarrierDiff`, which internalizes the proved Goldilocks invertibility bridge for nonzero `paperCarrier` differences.
- **Protocol role**: ProtocolRelations uses `protocolTargetProp` to define CCS/CE relations; PiCCS and downstream reductions depend on this target.

## Target Formulas (Paper → Lean)

- `protocolTargetProp ctx ↔ thm3CoreAssumption ctx.bar ∧ splitBase2TerminalZeroProp ctx.splitScalar ctx.kSplit ∧ evalHomAssumption ... ∧ vecModuleAssumption ... ∧ scalarModuleAssumption ... ∧ samplingExpansionProp ... ∧ qVals.size = 2^r.size ∧ mleEval qVals r = mleInnerProductForm qVals r ∧ interpolationProp ... ∧ invertibleRq ctx.invDelta`
- `ProtocolTargetAssumptions ctx → protocolTargetProp ctx`
- `samplingDiffSet paperCarrier δ → δ ≠ 0 → invertibleRq δ`

## Paper Anchors

- Source: `./formal/superneo-lean/SuperNeo.pdf.md`
- Anchors:
  - Section 7 (Neo's folding scheme for CCS), lines 447–481: Relations (Definitions 11–13), Global Reduction Parameters (Definition 14)
  - Section 7.3 (Π_CCS), lines 481–547: Interactive reduction for CCS

## Module Mapping

- Implementation: `SuperNeo.FoldingProtocol.ProtocolTarget`
- Interface: `SuperNeo.FoldingProtocol.ProtocolTargetInterface`

## Contract Surface

| Contract group | Lean surface | Preconditions | Guarantee | Role | Used by |
|---|---|---|---|---|---|
| Context | `ProtocolTargetContext` | None | Bundles bar, m, r, rho1, rho2, hVec, hScal, splitScalar, kSplit, invDelta, cset, samples, xs, ys, qVals, coeffs, xEval, expectedEval | Definitional | ProtocolRelations, PiCCS |
| Assumptions | `ProtocolTargetAssumptions ctx` | None | Bundles thm3, arithmetic (ArithmeticObligations), direct witness `invertibleRq ctx.invDelta` | Definitional | InteractiveReductions, Pi* assumption routes |
| Invertibility bridge | `strictInvertibilityWindowProp_five_of_paperCarrierDiff` | `samplingDiffSet paperCarrier δ`, `δ ≠ 0` | Strict paper-faithful window `< 5` | Theorem-Target | Protocol-facing invertibility assembly |
| Invertibility bridge | `invertibleRq_of_paperCarrierDiff` | `samplingDiffSet paperCarrier δ`, `δ ≠ 0` | `invertibleRq δ` | Theorem-Target | Protocol-facing invertibility assembly on the active Goldilocks path |
| Constructor | `ProtocolTargetAssumptions.ofPaperCarrierDiff` | thm3 + arithmetic + `samplingDiffSet paperCarrier ctx.invDelta` + `ctx.invDelta ≠ 0` | Canonical protocol-target bundle on the paper-facing challenge-difference path | Theorem-Target | InteractiveReductions, ProtocolTheorem |
| Target prop | `protocolTargetProp ctx` | None | Conjunction of all protocol-target predicates | Definitional | ProtocolRelations |
| Derivation | `protocolTargetProp_of_components` | thm3 + arithmetic + `invertibleRq ctx.invDelta` | `protocolTargetProp ctx` | Theorem-Target | ProtocolRelations |
| Derivation | `protocolTargetProp_of_assumptions` | `ProtocolTargetAssumptions ctx` | `protocolTargetProp ctx` | Theorem-Target | ProtocolRelations, Pi* assumption routes |

Other Theorem-3 evidence forms convert upstream: `Thm3Core` owns
`thm3CoreAssumption_of_basisKernelAssumption` / `thm3BasisKernelAssumption_of_check`,
and the native bar discharges via `thm3CoreAssumption_native`; callers convert
their evidence and use the anonymous constructor or `ofPaperCarrierDiff`.

## Proof Obligations and Closure Plan

- `protocolTargetProp` must be derivable from explicit components and from the assumption bundle.
- `ofPaperCarrierDiff` must internalize the active paper-facing invertibility bridge from `samplingDiffSet paperCarrier ctx.invDelta ∧ ctx.invDelta ≠ 0` without introducing an extra local invertibility boundary.

## Assumption Ledger

- This module introduces no theorem-level boundary beyond `ProtocolTargetAssumptions` and the upstream theorem providers it bundles.
- The concrete source of `ctx.invDelta` as a nonzero paper-carrier difference remains an upstream protocol fact, not something derived here.

## Dependency and Consumer Map

- Upstream dependencies:
  - `SuperNeo/EmbeddingTheory/Thm3Core.lean`: imports `thm3CoreAssumption`
  - `SuperNeo/FoldingProtocol/ArithmeticObligations.lean`: uses `ArithmeticObligations` for arithmetic bundle
  - `SuperNeo/SecurityModel/InvertibilityGoldilocks.lean`, `SuperNeo/SecurityModel/SamplingSet.lean`: proved Goldilocks invertibility for paper-carrier differences
- Downstream consumers:
  - `SuperNeo/FoldingProtocol/ProtocolRelations.lean`: uses `protocolTargetProp`, `protocolTargetProp_of_assumptions`, `ProtocolTargetContext` to define CCS/CE relations
  - `SuperNeo/FoldingProtocol/PiCCS.lean` / `PiRLC.lean` / `PiDEC.lean`: assumption routes take `ProtocolTargetAssumptions`
  - `SuperNeo/SecurityModel/InteractiveReductions.lean`: embeds `ProtocolTargetAssumptions` in the composed reduction registry
  - `SuperNeo/FoldingProtocol/ProtocolTheorem.lean`: uses `ProtocolTargetContext` and `ofPaperCarrierDiff` on the active final route

## Implementation Plan

1. `ProtocolTargetContext` structure holds all protocol parameters.
2. `ProtocolTargetAssumptions` bundles thm3, arithmetic obligations, and a direct `invertibleRq` witness for `ctx.invDelta`.
3. `ProtocolTargetAssumptions.ofPaperCarrierDiff` derives the invertibility witness from the active paper-facing `paperCarrier`-difference path using the proved Goldilocks theorem directly.
4. `protocolTargetProp` defined as conjunction of target predicates.
5. `protocolTargetProp_of_components` / `protocolTargetProp_of_assumptions` proved by projection.

## Quality Expectations

- No `sorry` in any theorem.
- All declarations proved natively.

## Acceptance Criteria

1. `lake build` succeeds.
2. `lake exe check` succeeds.
3. All surfaces exported through the interface.

## Out of Scope

- Concrete instantiation of `ProtocolTargetAssumptions`; that belongs to protocol setup.
- `matrixTransformAssumption_of_thm3CoreAssumption` is re-exported from MatrixTransform for consumers; closure is in MatrixTransform.
