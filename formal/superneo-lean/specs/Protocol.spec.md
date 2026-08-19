# Protocol — Proof-System Capstone

## Purpose

- **What it is**: The proof-system-level re-export layer for `SuperNeo.ProtocolTheorem`.
- **Key property**: `hA : FinalTheoremAssumptions ctx -> FinalTheoremShape ctx hA`.
- **Protocol role**: Gives downstream proof-system consumers one typed entrypoint for the composed SuperNeo theorem.

## Target Formulas

- `FinalTheoremShape ctx hA` combines `FinalCompletenessStatement ctx hA` and `FinalKnowledgeSoundnessStatement ctx hA`.
- `FinalKnowledgeSoundnessStatement ctx hA` contains the witness-level SumCheck failure bound, the local Schwartz-Zippel bound, the MSIS/Ajtai advantage bounds, and total-error accounting.
- `epsTotal n = epsSumcheck n + epsMSIS n + epsSchwartzZippel n + epsBinding n + epsRelaxedBinding n`.
- `IsNegligible epsTotal`.

## Paper Anchors

- Source: `./formal/superneo-lean/SuperNeo.pdf.md`
- Sections 4–7 and Appendix C

## Module Mapping

| Lean module | Paper section |
|-------------|----------------|
| `SuperNeo.ProofSystem.Protocol` | Capstone re-export layer for the final protocol theorem |

## Contract Surface

| Group | Lean symbol | Kind | Guarantee |
|-------|-------------|------|-----------|
| Params | `LatticeParams` | abbrev | `= SuperNeo.ProofSystem.AjtaiParams` |
| Assumptions | `FinalTheoremAssumptions` | abbrev | Re-export of the final theorem assumption registry |
| Statements | `FinalCompletenessStatement`, `FinalKnowledgeSoundnessStatement`, `FinalTheoremShape` | abbrev | Re-export of the final theorem statement shapes |
| Error constructors | `finalErrorPackageOfComponentBoundaries`, `finalErrorPackageOfAlignedComponents`, `finalErrorPackageOfAlignedPaperCarrierFromThreeDLe`, `finalErrorPackageOfGoldilocksPaperCarrier` | def | Definitional re-exports of the canonical final-error constructors |
| Final constructors | `finalTheoremAssumptionsOfBoundaryPackages`, `finalTheoremAssumptionsOfAlignedPaperCarrierBoundaryPackages`, `finalTheoremAssumptionsOfAlignedPaperCarrierDiffBoundaryPackages`, `finalTheoremAssumptionsOfAlignedPaperCarrierLowNormBoundaryPackages`, `finalTheoremAssumptionsOfGoldilocksPaperCarrierBoundaryPackages`, `finalTheoremAssumptionsOfGoldilocksPaperCarrierDerivedSumcheck`, `finalTheoremAssumptionsOfGoldilocksPaperCarrierDiffBoundaryPackages`, `finalTheoremAssumptionsOfGoldilocksNativePaperCarrierDiffBoundaryPackages`, `finalTheoremAssumptionsOfGoldilocksPaperCarrierLowNormBoundaryPackages` | def | Definitional re-exports of the canonical final-theorem constructors |
| Final theorems | `finalTheoremShape_of_assumptions`, `finalTheoremShapeOfAlignedPaperCarrierBoundaryPackages`, `finalTheoremShapeOfAlignedPaperCarrierDiffBoundaryPackages`, `finalTheoremShapeOfAlignedPaperCarrierLowNormBoundaryPackages`, `finalTheoremShapeOfGoldilocksPaperCarrierBoundaryPackages`, `finalTheoremShapeOfGoldilocksPaperCarrierDiffBoundaryPackages`, `finalTheoremShapeOfGoldilocksNativePaperCarrierDiffBoundaryPackages`, `finalTheoremShapeOfGoldilocksPaperCarrierLowNormBoundaryPackages` | def/theorem | Exact forwarders of the `ProtocolTheorem` theorem surfaces |
| Accessors | `finalSumcheckErrorBoundary`, `finalSumcheckAdvantageBound`, `finalSchwartzZippelBoundaryPackage`, `finalSchwartzZippelErrorAligned`, `finalSchwartzZippelErrorNegligible`, `finalSchwartzZippelAdvantageBound`, `finalSumcheckErrorAligned`, `finalSumcheckErrorNegligible`, `finalMSISHardnessBoundary`, `finalMSISHardnessPackage`, `finalMSISToAjtaiReductions`, `finalMSISErrorAligned`, `finalMSISErrorNegligible`, `finalMSISAdvantageBound`, `finalBindingErrorAligned`, `finalBindingErrorNegligible`, `finalBindingAdvantageBound`, `finalRelaxedBindingErrorAligned`, `finalRelaxedBindingErrorNegligible`, `finalRelaxedBindingAdvantageBound`, `finalTotalErrorAligned`, `finalErrorTotalDecomp`, `finalTotalErrorNegligible`, `finalAjtaiBindingBoundaryPackage`, `finalAjtaiRelaxedBindingBoundaryPackage`, `finalAjtaiBindingBoundary`, `finalAjtaiRelaxedBindingBoundary` | def | Definitional accessors into the assembled final theorem package |

## Design Notes

This module adds no new proof content. Every symbol is a thin re-export of `SuperNeo.ProtocolTheorem`. On the native-bar Goldilocks route, the re-exported constructor derives the witness-level SumCheck, local Schwartz-Zippel, and internal MSIS boundary packages internally, so the only explicit theorem-level security assumption on that route is the MSIS hardness assumption.

## Assumption Ledger

No new assumptions are introduced here. All boundary assumptions live in `ProtocolTheorem`; this module only re-exports them.

## Dependency and Consumer Map

- Dependencies: `SuperNeo.ProtocolTheorem`, `SuperNeo.ProofSystem.Types`, `Security`, `Lattice`, `SumCheck`, `Folding`
- Consumers:
  - `SuperNeo.ProofSystem.ProtocolInterface`
  - downstream proof-system modules that want the final composed theorem

## Module Discipline

This module remains thin. All theorem logic lives in `ProtocolTheorem`; this layer forwards the proof-system surface without adding new assumptions or proof obligations.

## Quality Expectations

- Re-exports stay definitionally equal to the underlying `ProtocolTheorem` surfaces.
- No extra theorem logic is introduced here.

## Acceptance Criteria

1. `lake build` succeeds.
2. The re-export surface matches the implementation.
3. `finalTheoremShape_of_assumptions` remains a direct forwarder.

## Out of Scope

- New theorem development.
- Concrete protocol setup instantiation.
