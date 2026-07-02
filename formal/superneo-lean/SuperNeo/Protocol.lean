import SuperNeo.FoldingProtocol.ProtocolTheorem
import SuperNeo.SecurityModel.Types
import SuperNeo.SecurityModel.Security
import SuperNeo.Commitment.Lattice
import SuperNeo.SumCheck

/-!
Proof-system facade for the final protocol theorem.

This module owns the client-facing entrypoint names; the theorem content lives
in `SuperNeo.FoldingProtocol.ProtocolTheorem`.
-/

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

/-- Canonical constructor for final error packages from component boundaries. -/
def finalErrorPackageOfComponentBoundaries :=
  @SuperNeo.FinalErrorPackage.ofComponentBoundaries

/-- Canonical constructor for final error packages on the Goldilocks Appendix B.2 paper-parameter family. -/
def finalErrorPackageOfGoldilocksPaperCarrier :=
  @SuperNeo.FinalErrorPackage.ofGoldilocksPaperCarrier

/-- Canonical constructor for final theorem assumptions from boundary packages. -/
def finalTheoremAssumptionsOfBoundaryPackages :=
  @SuperNeo.FinalTheoremAssumptions.ofBoundaryPackages

/-- Canonical Goldilocks final-theorem assumption constructor with internally derived SumCheck/Schwartz-Zippel/MSIS packaging. -/
noncomputable def finalTheoremAssumptionsOfGoldilocksPaperCarrierDerivedSumcheck :=
  @SuperNeo.FinalTheoremAssumptions.ofGoldilocksPaperCarrierDerivedSumcheck

/-- Active paper-faithful final-theorem assumption constructor: native bar, Goldilocks Appendix B.2 parameters, paper-carrier-difference invertibility, theorem-level MSIS hardness. -/
noncomputable def finalTheoremAssumptionsOfGoldilocksNativePaperCarrierDiffBoundaryPackages :=
  @SuperNeo.FinalTheoremAssumptions.ofGoldilocksNativePaperCarrierDiffBoundaryPackages

/-- Canonical proof-system final theorem shape constructor. -/
theorem finalTheoremShape_of_assumptions
  {ctx : SuperNeo.ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  FinalTheoremShape ctx hA := by
  exact SuperNeo.finalTheoremShape_of_assumptions hA

/-- Active paper-faithful final theorem on the native-bar Goldilocks paper-carrier-difference route. -/
theorem finalTheoremShape_of_goldilocksNativePaperCarrierDiffBoundaryPackages
  {ctx : SuperNeo.ProtocolTargetContext}
  (messageLength : Nat)
  (hBarNative : ctx.bar = SuperNeo.nativeBarMatrix)
  (hArithmetic : SuperNeo.ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hDiff : SuperNeo.samplingDiffSet SuperNeo.paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ SuperNeo.zeroRq)
  (hWitness : SuperNeo.SumCheckTransitionWitness ctx)
  (hMsis :
    SuperNeo.msisHardnessAssumption
      (SuperNeo.ProofSystem.goldilocksPaperAjtaiParams messageLength)) :
  FinalTheoremShape ctx
    (SuperNeo.FinalTheoremAssumptions.ofGoldilocksNativePaperCarrierDiffBoundaryPackages
      messageLength hBarNative hArithmetic hDiff hNe hWitness hMsis) := by
  exact SuperNeo.finalTheoremShape_of_goldilocksNativePaperCarrierDiffBoundaryPackages
    messageLength hBarNative hArithmetic hDiff hNe hWitness hMsis

end SuperNeo.ProofSystem
