import SuperNeo.ProtocolTheorem
import SuperNeo.ProofSystem.Types
import SuperNeo.ProofSystem.Security
import SuperNeo.ProofSystem.Lattice
import SuperNeo.ProofSystem.SumCheck
import SuperNeo.ProofSystem.Folding

namespace SuperNeo.ProofSystem

abbrev LatticeParams := SuperNeo.ProofSystem.AjtaiParams

abbrev FinalTheoremAssumptions (ctx : SuperNeo.ProtocolTargetContext) :=
  SuperNeo.FinalTheoremAssumptions ctx

abbrev FinalCompletenessStatement
  (ctx : SuperNeo.ProtocolTargetContext)
  (hA : FinalTheoremAssumptions ctx) :=
  SuperNeo.FinalCompletenessStatement ctx hA

abbrev FinalKnowledgeSoundnessStatement
  (ctx : SuperNeo.ProtocolTargetContext)
  (hA : FinalTheoremAssumptions ctx) :=
  SuperNeo.FinalKnowledgeSoundnessStatement ctx hA

abbrev FinalTheoremShape
  (ctx : SuperNeo.ProtocolTargetContext)
  (hA : FinalTheoremAssumptions ctx) :=
  SuperNeo.FinalTheoremShape ctx hA

/-- Canonical constructor for aligned final error packages. -/
def finalErrorPackageOfAlignedComponents :=
  @SuperNeo.FinalErrorPackage.ofAlignedComponents

/-- Canonical constructor for final error packages from component boundaries. -/
def finalErrorPackageOfComponentBoundaries :=
  @SuperNeo.FinalErrorPackage.ofComponentBoundaries

/-- Canonical constructor for aligned final error packages on the proved `paperCarrier` path, deriving Ajtai reduction data from the MSIS boundary. -/
def finalErrorPackageOfAlignedPaperCarrierFromThreeDLe :=
  @SuperNeo.FinalErrorPackage.ofAlignedPaperCarrierFromThreeDLe

/-- Canonical constructor for aligned final error packages on the Goldilocks Appendix B.2 paper-parameter family. -/
def finalErrorPackageOfGoldilocksPaperCarrier :=
  @SuperNeo.FinalErrorPackage.ofGoldilocksPaperCarrier

/-- Canonical constructor for final theorem assumptions from boundary packages. -/
def finalTheoremAssumptionsOfBoundaryPackages :=
  @SuperNeo.FinalTheoremAssumptions.ofBoundaryPackages

/-- Canonical constructor for final theorem assumptions on the proved `paperCarrier` path, deriving Ajtai reduction data from the MSIS boundary. -/
def finalTheoremAssumptionsOfAlignedPaperCarrierBoundaryPackages :=
  @SuperNeo.FinalTheoremAssumptions.ofAlignedPaperCarrierBoundaryPackages

/-- Canonical constructor for final theorem assumptions on the paper-facing `paperCarrier`-difference path, deriving Ajtai reduction data from the MSIS boundary. -/
def finalTheoremAssumptionsOfAlignedPaperCarrierDiffBoundaryPackages :=
  @SuperNeo.FinalTheoremAssumptions.ofAlignedPaperCarrierDiffBoundaryPackages

/-- Canonical constructor for final theorem assumptions on the proved `paperCarrier` path from a stronger strict low-norm invertibility theorem with threshold at least `5`. -/
def finalTheoremAssumptionsOfAlignedPaperCarrierLowNormBoundaryPackages :=
  @SuperNeo.FinalTheoremAssumptions.ofAlignedPaperCarrierLowNormBoundaryPackages

/-- Canonical constructor for final theorem assumptions on the Goldilocks Appendix B.2 paper-parameter family. -/
def finalTheoremAssumptionsOfGoldilocksPaperCarrierBoundaryPackages :=
  @SuperNeo.FinalTheoremAssumptions.ofGoldilocksPaperCarrierBoundaryPackages

/-- Canonical constructor for final theorem assumptions on the Goldilocks Appendix B.2 paper-parameter family, deriving the witness-level SumCheck and local Schwartz-Zippel boundaries from the carried transition witness and reconstructing the internal MSIS boundary from the theorem-level hardness assumption. -/
noncomputable def finalTheoremAssumptionsOfGoldilocksPaperCarrierDerivedSumcheck :=
  @SuperNeo.FinalTheoremAssumptions.ofGoldilocksPaperCarrierDerivedSumcheck

/-- Canonical constructor for final theorem assumptions on the Goldilocks Appendix B.2 paper-parameter family and active `paperCarrier`-difference path. -/
def finalTheoremAssumptionsOfGoldilocksPaperCarrierDiffBoundaryPackages :=
  @SuperNeo.FinalTheoremAssumptions.ofGoldilocksPaperCarrierDiffBoundaryPackages

/-- Canonical constructor for final theorem assumptions on the Goldilocks Appendix B.2 paper-parameter family and active native-bar `paperCarrier`-difference path, deriving all local theorem packages internally and keeping only the theorem-level MSIS hardness assumption explicit. -/
noncomputable def finalTheoremAssumptionsOfGoldilocksNativePaperCarrierDiffBoundaryPackages :=
  @SuperNeo.FinalTheoremAssumptions.ofGoldilocksNativePaperCarrierDiffBoundaryPackages

/-- Canonical constructor for final theorem assumptions on the Goldilocks Appendix B.2 paper-parameter family from a stronger strict low-norm invertibility theorem with threshold at least `5`. -/
def finalTheoremAssumptionsOfGoldilocksPaperCarrierLowNormBoundaryPackages :=
  @SuperNeo.FinalTheoremAssumptions.ofGoldilocksPaperCarrierLowNormBoundaryPackages

/-- Canonical final theorem specialized to the proved `paperCarrier` path. -/
def finalTheoremShapeOfAlignedPaperCarrierBoundaryPackages :=
  @SuperNeo.finalTheoremShape_of_alignedPaperCarrierBoundaryPackages

/-- Canonical final theorem specialized to the paper-facing `paperCarrier`-difference path. -/
def finalTheoremShapeOfAlignedPaperCarrierDiffBoundaryPackages :=
  @SuperNeo.finalTheoremShape_of_alignedPaperCarrierDiffBoundaryPackages

/-- Canonical final theorem specialized to the proved `paperCarrier` path from a stronger strict low-norm invertibility theorem with threshold at least `5`. -/
def finalTheoremShapeOfAlignedPaperCarrierLowNormBoundaryPackages :=
  @SuperNeo.finalTheoremShape_of_alignedPaperCarrierLowNormBoundaryPackages

/-- Canonical final theorem specialized to the Goldilocks Appendix B.2 paper-parameter family. -/
def finalTheoremShapeOfGoldilocksPaperCarrierBoundaryPackages :=
  @SuperNeo.finalTheoremShape_of_goldilocksPaperCarrierBoundaryPackages

/-- Canonical final theorem specialized to the Goldilocks Appendix B.2 paper-parameter family and active `paperCarrier`-difference path. -/
def finalTheoremShapeOfGoldilocksPaperCarrierDiffBoundaryPackages :=
  @SuperNeo.finalTheoremShape_of_goldilocksPaperCarrierDiffBoundaryPackages

/-- Canonical final theorem specialized to the Goldilocks Appendix B.2 paper-parameter family and active native-bar `paperCarrier`-difference path. -/
def finalTheoremShapeOfGoldilocksNativePaperCarrierDiffBoundaryPackages :=
  @SuperNeo.finalTheoremShape_of_goldilocksNativePaperCarrierDiffBoundaryPackages

/-- Canonical final theorem specialized to the Goldilocks Appendix B.2 paper-parameter family from a stronger strict low-norm invertibility theorem with threshold at least `5`. -/
def finalTheoremShapeOfGoldilocksPaperCarrierLowNormBoundaryPackages :=
  @SuperNeo.finalTheoremShape_of_goldilocksPaperCarrierLowNormBoundaryPackages

/-- Access canonical SumCheck soundness-error boundary from final assumptions. -/
def finalSumcheckErrorBoundary
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary :=
  hA.sumcheckErrorBoundary

/-- Access canonical final error-accounting model from final assumptions. -/
def finalErrorModel
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.ErrorModel :=
  hA.errorModel

/-- Access explicit Schwartz-Zippel boundary package from final assumptions. -/
def finalSchwartzZippelBoundaryPackage
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.SchwartzZippelBoundary ctx :=
  hA.schwartzZippelBoundary

/-- Access explicit alignment between error model and Schwartz-Zippel error term. -/
def finalSchwartzZippelErrorAligned
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  hA.errorModel.epsSchwartzZippel = hA.schwartzZippelBoundary.epsSchwartzZippel :=
  hA.schwartzZippelErrorAligned

/-- Access canonical Schwartz-Zippel error negligibility from final assumptions. -/
def finalSchwartzZippelErrorNegligible
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.IsNegligible hA.errorModel.epsSchwartzZippel :=
  hA.schwartzZippelErrorNegligible

/-- Access canonical Schwartz-Zippel advantage bound aligned to final error accounting. -/
def finalSchwartzZippelAdvantageBound
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.SchwartzZippelAdvantageBound ctx hA.errorModel.epsSchwartzZippel :=
  hA.schwartzZippelAdvantageBound

/-- Access explicit alignment between error model and SumCheck soundness error. -/
def finalSumcheckErrorAligned
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  hA.errorModel.epsSumcheck = hA.sumcheckError.epsSoundness :=
  hA.sumcheckErrorAligned

/-- Access canonical SumCheck soundness-failure advantage bound from final assumptions. -/
def finalSumcheckAdvantageBound
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.Sumcheck.SoundnessFailureAdvantageBound
      (SuperNeo.sumcheckInstanceOfContext ctx)
      hA.reduction.sumcheckTransitionWitness.transcript
      hA.errorModel.epsSumcheck :=
  hA.sumcheckAdvantageBound

/-- Access canonical SumCheck soundness-error negligibility from final assumptions. -/
def finalSumcheckErrorNegligible
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.IsNegligible hA.errorModel.epsSumcheck :=
  hA.sumcheckErrorNegligible

/-- Access explicit lattice MSIS hardness boundary from final assumptions. -/
def finalMSISHardnessBoundary
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.MSISHardnessAssumption hA.latticeParams :=
  hA.msisHardnessBoundary

/-- Access explicit lattice MSIS hardness package from final assumptions. -/
def finalMSISHardnessPackage
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.MSISHardnessBoundary hA.latticeParams :=
  hA.msisBoundary

/-- Access explicit reduction surface linking MSIS hardness to Ajtai boundaries. -/
def finalMSISToAjtaiReductions
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.MSISToAjtaiReductions hA.latticeParams :=
  hA.msisToAjtai

/-- Access explicit alignment between error model and MSIS hardness error term. -/
def finalMSISErrorAligned
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  hA.errorModel.epsMSIS = hA.msisBoundary.epsMSIS :=
  hA.msisErrorAligned

/-- Access canonical MSIS hardness-error negligibility from final assumptions. -/
def finalMSISErrorNegligible
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.IsNegligible hA.errorModel.epsMSIS :=
  hA.msisErrorNegligible

/-- Access canonical MSIS advantage bound aligned to final error accounting. -/
def finalMSISAdvantageBound
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.MSISAdvantageBound hA.latticeParams hA.errorModel.epsMSIS :=
  hA.msisAdvantageBound

/-- Access explicit alignment between error model and Ajtai binding error term. -/
def finalBindingErrorAligned
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  hA.errorModel.epsBinding = hA.msisToAjtai.epsBinding :=
  hA.bindingErrorAligned

/-- Access canonical Ajtai binding-error negligibility from final assumptions. -/
def finalBindingErrorNegligible
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.IsNegligible hA.errorModel.epsBinding :=
  hA.bindingErrorNegligible

/-- Access canonical Ajtai binding advantage bound aligned to final error accounting. -/
def finalBindingAdvantageBound
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.AjtaiBindingAdvantageBound hA.latticeParams hA.errorModel.epsBinding :=
  hA.bindingAdvantageBound

/-- Access explicit alignment between error model and Ajtai relaxed-binding error term. -/
def finalRelaxedBindingErrorAligned
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  hA.errorModel.epsRelaxedBinding = hA.msisToAjtai.epsRelaxedBinding :=
  hA.relaxedBindingErrorAligned

/-- Access canonical Ajtai relaxed-binding-error negligibility from final assumptions. -/
def finalRelaxedBindingErrorNegligible
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.IsNegligible hA.errorModel.epsRelaxedBinding :=
  hA.relaxedBindingErrorNegligible

/-- Access canonical Ajtai relaxed-binding advantage bound aligned to final error accounting. -/
def finalRelaxedBindingAdvantageBound
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.AjtaiRelaxedBindingAdvantageBound
      hA.latticeParams hA.msisToAjtai.laws.samplingCarrier hA.errorModel.epsRelaxedBinding :=
  hA.relaxedBindingAdvantageBound

/-- Access explicit alignment for aggregated total error accounting at final theorem level. -/
def finalTotalErrorAligned
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  ∀ n,
    hA.errorModel.epsTotal n =
      hA.sumcheckError.epsSoundness n +
        hA.msisBoundary.epsMSIS n +
          hA.schwartzZippelBoundary.epsSchwartzZippel n +
            hA.msisToAjtai.epsBinding n +
              hA.msisToAjtai.epsRelaxedBinding n :=
  hA.totalErrorDecompFromModel

/-- Access canonical total-error decomposition derived from model decomposition plus alignments. -/
def finalErrorTotalDecomp
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  ∀ n,
    hA.errorModel.epsTotal n =
      hA.sumcheckError.epsSoundness n +
        hA.msisBoundary.epsMSIS n +
          hA.schwartzZippelBoundary.epsSchwartzZippel n +
            hA.msisToAjtai.epsBinding n +
              hA.msisToAjtai.epsRelaxedBinding n :=
  hA.totalErrorDecompFromModel

/-- Access canonical total-error negligibility from final error accounting. -/
def finalTotalErrorNegligible
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.IsNegligible hA.errorModel.epsTotal :=
  hA.totalErrorNegligible

/-- Access canonical Ajtai binding boundary package derived from MSIS+reductions. -/
def finalAjtaiBindingBoundaryPackage
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.AjtaiBindingBoundary hA.latticeParams :=
  hA.ajtaiBindingBoundaryPackage

/-- Access canonical Ajtai relaxed-binding boundary package derived from MSIS+reductions. -/
def finalAjtaiRelaxedBindingBoundaryPackage
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.AjtaiRelaxedBindingBoundary
    hA.latticeParams hA.msisToAjtai.laws.samplingCarrier :=
  hA.ajtaiRelaxedBindingBoundaryPackage

/-- Access explicit lattice Ajtai binding boundary from final assumptions. -/
def finalAjtaiBindingBoundary
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.AjtaiBindingAssumption hA.latticeParams :=
  hA.ajtaiBindingBoundary

/-- Access explicit lattice Ajtai relaxed-binding boundary from final assumptions. -/
def finalAjtaiRelaxedBindingBoundary
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.AjtaiRelaxedBindingAssumption
    hA.latticeParams hA.msisToAjtai.laws.samplingCarrier :=
  hA.ajtaiRelaxedBindingBoundary

/-- Canonical proof-system final theorem shape constructor. -/
theorem finalTheoremShape_of_assumptions
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  FinalTheoremShape ctx hA := by
  exact SuperNeo.finalTheoremShape_of_assumptions hA

end SuperNeo.ProofSystem
