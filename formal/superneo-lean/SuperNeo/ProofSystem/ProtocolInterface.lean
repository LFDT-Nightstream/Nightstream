import SuperNeo.ProofSystem.Protocol

/-!
Interface for `SuperNeo.ProofSystem.Protocol`.

Spec: `./formal/superneo-lean/specs/ProofSystem/Protocol.spec.md`
Paper: `./formal/superneo-lean/SuperNeo.pdf.md`
  - Sections 4–7, Appendix C: Definition 1 (lines 275–282), Definitions 2–14, Π_CCS (7.3), Π_RLC (7.4), Π_DEC (7.5), Theorem 6 (lines 438–445).
-/

namespace SuperNeo

namespace ProofSystem.ProtocolInterface

/-- Canonical implementation module name for this interface. -/
def implementationModule : String := "SuperNeo.ProofSystem.Protocol"

/-- Canonical paper source used for this module-level interface/spec pair. -/
def paperSource : String := "./formal/superneo-lean/SuperNeo.pdf.md"

/-- Paper sections used to ground this module boundary. -/
def paperAnchors : List String := ["§6 composition theorem context", "§7 protocol composition"]

/-- Public symbol inventory extracted from the implementation module. -/
def exportedSymbolNames : List String := ["LatticeParams", "FinalTheoremAssumptions", "FinalCompletenessStatement", "FinalKnowledgeSoundnessStatement", "FinalTheoremShape", "finalErrorPackageOfComponentBoundaries", "finalErrorPackageOfAlignedComponents", "finalErrorPackageOfAlignedPaperCarrierFromThreeDLe", "finalErrorPackageOfGoldilocksPaperCarrier", "finalTheoremAssumptionsOfBoundaryPackages", "finalTheoremAssumptionsOfAlignedPaperCarrierBoundaryPackages", "finalTheoremAssumptionsOfAlignedPaperCarrierDiffBoundaryPackages", "finalTheoremAssumptionsOfAlignedPaperCarrierLowNormBoundaryPackages", "finalTheoremAssumptionsOfGoldilocksPaperCarrierBoundaryPackages", "finalTheoremAssumptionsOfGoldilocksPaperCarrierDerivedSumcheck", "finalTheoremAssumptionsOfGoldilocksPaperCarrierDiffBoundaryPackages", "finalTheoremAssumptionsOfGoldilocksNativePaperCarrierDiffBoundaryPackages", "finalTheoremAssumptionsOfGoldilocksPaperCarrierLowNormBoundaryPackages", "finalTheoremShapeOfAlignedPaperCarrierBoundaryPackages", "finalTheoremShapeOfAlignedPaperCarrierDiffBoundaryPackages", "finalTheoremShapeOfAlignedPaperCarrierLowNormBoundaryPackages", "finalTheoremShapeOfGoldilocksPaperCarrierBoundaryPackages", "finalTheoremShapeOfGoldilocksPaperCarrierDiffBoundaryPackages", "finalTheoremShapeOfGoldilocksNativePaperCarrierDiffBoundaryPackages", "finalTheoremShapeOfGoldilocksPaperCarrierLowNormBoundaryPackages", "finalSumcheckErrorBoundary", "finalErrorModel", "finalSchwartzZippelBoundaryPackage", "finalSchwartzZippelErrorAligned", "finalSchwartzZippelErrorNegligible", "finalSchwartzZippelAdvantageBound", "finalSumcheckErrorAligned", "finalSumcheckAdvantageBound", "finalSumcheckErrorNegligible", "finalMSISHardnessBoundary", "finalMSISHardnessPackage", "finalMSISToAjtaiReductions", "finalMSISErrorAligned", "finalMSISErrorNegligible", "finalMSISAdvantageBound", "finalBindingErrorAligned", "finalBindingErrorNegligible", "finalBindingAdvantageBound", "finalRelaxedBindingErrorAligned", "finalRelaxedBindingErrorNegligible", "finalRelaxedBindingAdvantageBound", "finalTotalErrorAligned", "finalErrorTotalDecomp", "finalTotalErrorNegligible", "finalAjtaiBindingBoundaryPackage", "finalAjtaiRelaxedBindingBoundaryPackage", "finalAjtaiBindingBoundary", "finalAjtaiRelaxedBindingBoundary", "finalTheoremShape_of_assumptions"]

/-- Assumption/boundary-oriented symbols extracted by naming convention. -/
def boundarySymbolNames : List String := ["FinalTheoremAssumptions", "finalErrorPackageOfComponentBoundaries", "finalErrorPackageOfAlignedComponents", "finalErrorPackageOfAlignedPaperCarrierFromThreeDLe", "finalErrorPackageOfGoldilocksPaperCarrier", "finalTheoremAssumptionsOfBoundaryPackages", "finalTheoremAssumptionsOfAlignedPaperCarrierBoundaryPackages", "finalTheoremAssumptionsOfAlignedPaperCarrierDiffBoundaryPackages", "finalTheoremAssumptionsOfAlignedPaperCarrierLowNormBoundaryPackages", "finalTheoremAssumptionsOfGoldilocksPaperCarrierBoundaryPackages", "finalTheoremAssumptionsOfGoldilocksPaperCarrierDerivedSumcheck", "finalTheoremAssumptionsOfGoldilocksPaperCarrierDiffBoundaryPackages", "finalTheoremAssumptionsOfGoldilocksNativePaperCarrierDiffBoundaryPackages", "finalTheoremAssumptionsOfGoldilocksPaperCarrierLowNormBoundaryPackages", "finalTheoremShapeOfAlignedPaperCarrierBoundaryPackages", "finalTheoremShapeOfAlignedPaperCarrierDiffBoundaryPackages", "finalTheoremShapeOfAlignedPaperCarrierLowNormBoundaryPackages", "finalTheoremShapeOfGoldilocksPaperCarrierBoundaryPackages", "finalTheoremShapeOfGoldilocksPaperCarrierDiffBoundaryPackages", "finalTheoremShapeOfGoldilocksNativePaperCarrierDiffBoundaryPackages", "finalTheoremShapeOfGoldilocksPaperCarrierLowNormBoundaryPackages", "FinalCompletenessStatement", "FinalKnowledgeSoundnessStatement", "finalSumcheckErrorBoundary", "finalSumcheckAdvantageBound", "finalSchwartzZippelBoundaryPackage", "finalSchwartzZippelErrorNegligible", "finalSumcheckErrorNegligible", "finalMSISHardnessBoundary", "finalMSISErrorNegligible", "finalBindingErrorNegligible", "finalRelaxedBindingErrorNegligible", "finalTotalErrorNegligible", "finalAjtaiBindingBoundaryPackage", "finalAjtaiRelaxedBindingBoundaryPackage", "finalAjtaiBindingBoundary", "finalAjtaiRelaxedBindingBoundary", "finalTheoremShape_of_assumptions"]

/-- Interface inventory marker for the typed surface exposed here. -/
def interfaceScaffoldReady : Prop := True

theorem interfaceScaffoldReady_true : interfaceScaffoldReady := by
  trivial

end ProofSystem.ProtocolInterface

end SuperNeo
