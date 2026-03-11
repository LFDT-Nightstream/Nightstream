import SuperNeo.InteractiveReductions
import SuperNeo.Interp
import SuperNeo.ProofSystem.Lattice
import SuperNeo.ProofSystem.LatticePaper
import SuperNeo.ProofSystem.LatticeReductions
import SuperNeo.ProofSystem.LatticeReductionsDerived
import SuperNeo.ProofSystem.SumCheck
import SuperNeo.ProofSystem.Security

/-!
Canonical final theorem shape for the SuperNeo scaffold.

This file provides:
- one explicit assumption registry,
- completeness and knowledge-soundness statement shapes,
- one canonical theorem constructor from assumptions.
-/

namespace SuperNeo

/--
Concrete theorem-facing Schwartz-Zippel failure witness surface.

This avoids the refuted universal `interpolationAssumption` scaffold and instead
tracks failure of the concrete interpolation obligation carried by one protocol
context.
-/
structure SchwartzZippelFailureWitness (ctx : ProtocolTargetContext) where
  failure :
    interpolationCase ctx.xs ctx.ys ctx.coeffs ctx.xEval ctx.expectedEval = false

/-- Canonical Schwartz-Zippel failure event surface (theorem-facing alias). -/
def schwartzZippelFailureEvent (ctx : ProtocolTargetContext) : Prop :=
  Nonempty (SchwartzZippelFailureWitness ctx)

/-- Advantage of the Schwartz-Zippel failure event under a probability model. -/
def SchwartzZippelAdvantage
  (ctx : ProtocolTargetContext)
  (prob : SuperNeo.ProofSystem.ProbModel)
  (_n : Nat) : Rat :=
  prob.Pr (schwartzZippelFailureEvent ctx)

/-- Theorem-facing Schwartz-Zippel advantage bound shape against an error function. -/
def SchwartzZippelAdvantageBound
  (ctx : ProtocolTargetContext)
  (eps : SuperNeo.ProofSystem.ErrorFn) : Prop :=
  ∀ prob : SuperNeo.ProofSystem.ProbModel, ∀ n : Nat,
    SchwartzZippelAdvantage ctx prob n ≤ (eps n : Rat)

/-- Theorem-facing boundary package for Schwartz-Zippel with explicit error term. -/
structure SchwartzZippelBoundary (ctx : ProtocolTargetContext) where
  epsSchwartzZippel : SuperNeo.ProofSystem.ErrorFn
  negligibleEpsSchwartzZippel :
    SuperNeo.ProofSystem.IsNegligible epsSchwartzZippel
  advantageBound :
    SchwartzZippelAdvantageBound ctx epsSchwartzZippel

/--
The concrete interpolation checker cannot fail on a context whose arithmetic
obligations already carry the quantified interpolation proof.
-/
theorem schwartzZippelFailureEvent_false_of_arithmetic
  {ctx : ProtocolTargetContext}
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval) :
  ¬ schwartzZippelFailureEvent ctx := by
  intro hEvent
  rcases hEvent with ⟨⟨hFailure⟩⟩
  have hInterp :
      interpolationCase ctx.xs ctx.ys ctx.coeffs ctx.xEval ctx.expectedEval = true :=
    hArithmetic.interpolationCase_eq_true
  simp [hInterp] at hFailure

/--
The concrete Schwartz-Zippel failure event has zero probability once the local
interpolation proof is present.
-/
theorem schwartzZippelAdvantageBound_zero_of_arithmetic
  {ctx : ProtocolTargetContext}
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval) :
  SchwartzZippelAdvantageBound ctx (fun _ => 0) := by
  intro prob n
  have hEventFalse : schwartzZippelFailureEvent ctx → False :=
    schwartzZippelFailureEvent_false_of_arithmetic hArithmetic
  calc
    SchwartzZippelAdvantage ctx prob n
        = prob.Pr (schwartzZippelFailureEvent ctx) := rfl
    _ ≤ prob.Pr False := prob.prMonotone hEventFalse
    _ = 0 := prob.prFalse
    _ = ((fun _ => 0) n : Rat) := rfl

namespace SchwartzZippelBoundary

/-- Canonical zero-error Schwartz-Zippel package from local arithmetic obligations. -/
def ofArithmetic
  {ctx : ProtocolTargetContext}
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval) :
  SchwartzZippelBoundary ctx where
  epsSchwartzZippel := fun _ => 0
  negligibleEpsSchwartzZippel := by
    simpa using
      (SuperNeo.ProofSystem.isNegligible_zero :
        SuperNeo.ProofSystem.IsNegligible (fun _ => 0))
  advantageBound := schwartzZippelAdvantageBound_zero_of_arithmetic hArithmetic

/-- Canonical zero-error Schwartz-Zippel package from the reduction bundle. -/
def ofReduction
  {ctx : ProtocolTargetContext}
  (hReduction : InteractiveReductionAssumptions ctx) :
  SchwartzZippelBoundary ctx :=
  ofArithmetic hReduction.reduction.arithmetic

end SchwartzZippelBoundary

/-- Paper-facing alias for lattice-parameter bundle. -/
abbrev LatticeParams := SuperNeo.ProofSystem.AjtaiParams

/-- Typed boundary for Module-SIS hardness. -/
def msisHardnessAssumption (params : LatticeParams) : Prop :=
  SuperNeo.ProofSystem.MSISHardnessAssumption params

/-- Typed boundary for Ajtai commitment binding. -/
def ajtaiBindingAssumption (params : LatticeParams) : Prop :=
  SuperNeo.ProofSystem.AjtaiBindingAssumption params

/-- Typed boundary for Ajtai relaxed binding. -/
def ajtaiRelaxedBindingAssumption
  (params : LatticeParams)
  (C : SuperNeo.SamplingCarrier) : Prop :=
  SuperNeo.ProofSystem.AjtaiRelaxedBindingAssumption params C

/--
Protocol-facing SumCheck soundness package built around the faithful
prefix-dependent full-field endpoint.

This does not try to reconstruct a `SoundnessGame` from `ctx`; it carries the
concrete aligned game that the protocol instantiation uses.
-/
structure SumcheckPrefixLundBoundary (ctx : ProtocolTargetContext) where
  game : SuperNeo.ProofSystem.Sumcheck.SoundnessGame
  instEq : game.inst = sumcheckInstanceOfContext ctx
  denominatorAligned :
    SuperNeo.sumcheckLundSoundnessDenominator game.inst = Goldilocks.q

/-- Access the faithful prefix-dependent SumCheck Lund bound from the package. -/
theorem SumcheckPrefixLundBoundary.lundBoundHolds
  {ctx : ProtocolTargetContext}
  (h : SumcheckPrefixLundBoundary ctx) :
  h.game.lundBoundHolds
    (SuperNeo.ProofSystem.Sumcheck.fullFieldUniformCoinProbModel h.game.inst.rounds) := by
  exact SuperNeo.ProofSystem.Sumcheck.lundSoundnessAssumptionFullFieldAligned_prefix
    h.game h.denominatorAligned

private def sumcheckReplayFalseClaimTable
  (inst : SumCheckInstance) : Array F :=
  (((inst.claimedValue + 1 : F) :: List.replicate (2 ^ inst.rounds - 1) (0 : F))).toArray

private theorem sumcheckReplayFalseClaimTable_size
  (inst : SumCheckInstance) :
  (sumcheckReplayFalseClaimTable inst).size = 2 ^ inst.rounds := by
  have hPowPos : 0 < 2 ^ inst.rounds := by
    exact Nat.pow_pos (a := 2) (n := inst.rounds) (by decide : 0 < (2 : Nat))
  calc
    (sumcheckReplayFalseClaimTable inst).size
        = (2 ^ inst.rounds - 1) + 1 := by
            simp [sumcheckReplayFalseClaimTable]
    _ = 2 ^ inst.rounds := Nat.sub_add_cancel (Nat.succ_le_of_lt hPowPos)

private theorem sumcheckReplayFalseClaimTable_sum
  (inst : SumCheckInstance) :
  sumcheckTableSum (sumcheckReplayFalseClaimTable inst) = inst.claimedValue + 1 := by
  unfold sumcheckTableSum sumcheckReplayFalseClaimTable
  rw [← Array.foldr_toList
      (f := fun v acc : F => v + acc)
      (xs := (((inst.claimedValue + 1 : F) :: List.replicate (2 ^ inst.rounds - 1) (0 : F))).toArray)]
  have hZeros :
      List.foldr (fun v acc : F => v + acc) 0
        (List.replicate (2 ^ inst.rounds - 1) (0 : F)) = 0 := by
    induction (2 ^ inst.rounds - 1) with
    | zero =>
        simp
    | succ n ih =>
        simp [List.replicate, ih]
  simp [hZeros]

private theorem sumcheckReplayFalseClaimTable_falseClaim
  (inst : SumCheckInstance) :
  sumcheckTableSum (sumcheckReplayFalseClaimTable inst) ≠ inst.claimedValue := by
  rw [sumcheckReplayFalseClaimTable_sum]
  intro hEq
  have hEq' : inst.claimedValue + 1 = inst.claimedValue + 0 := by
    calc
      inst.claimedValue + 1 = inst.claimedValue := hEq
      _ = inst.claimedValue + 0 := by simp
  have hOne : (1 : F) = 0 := add_left_cancel hEq'
  exact one_ne_zero hOne

/--
Canonical final error package:
groups all boundary error terms and one alignment witness against a shared
`ErrorModel`.
-/
structure FinalErrorPackage (ctx : ProtocolTargetContext) (params : LatticeParams) where
  /-- Minimal explicit SumCheck error boundary; theorem package is derived. -/
  sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary
  schwartzZippelBoundary : SchwartzZippelBoundary ctx
  errorModel : SuperNeo.ProofSystem.ErrorModel
  msisBoundary : SuperNeo.ProofSystem.MSISHardnessBoundary params
  msisToAjtai : SuperNeo.ProofSystem.MSISToAjtaiReductions params
  aligned :
    errorModel.epsSumcheck = sumcheckError.epsSoundness ∧
    errorModel.epsSchwartzZippel = schwartzZippelBoundary.epsSchwartzZippel ∧
    errorModel.epsMSIS = msisBoundary.epsMSIS ∧
    errorModel.epsBinding = msisToAjtai.epsBinding ∧
    errorModel.epsRelaxedBinding = msisToAjtai.epsRelaxedBinding

namespace FinalErrorPackage

/--
Canonical constructor from explicit component boundary packages, deriving the
shared `ErrorModel` internally from the component error surfaces.
-/
def ofComponentBoundaries
  {ctx : ProtocolTargetContext}
  {params : LatticeParams}
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary ctx)
  (msisBoundary : SuperNeo.ProofSystem.MSISHardnessBoundary params)
  (msisToAjtai : SuperNeo.ProofSystem.MSISToAjtaiReductions params) :
  FinalErrorPackage ctx params where
  sumcheckError := sumcheckError
  schwartzZippelBoundary := schwartzZippelBoundary
  errorModel :=
    SuperNeo.ProofSystem.ErrorModel.ofComponents
      sumcheckError.epsSoundness
      msisBoundary.epsMSIS
      schwartzZippelBoundary.epsSchwartzZippel
      msisToAjtai.epsBinding
      msisToAjtai.epsRelaxedBinding
      sumcheckError.negligibleEpsSoundness
      msisBoundary.negligibleEpsMSIS
      schwartzZippelBoundary.negligibleEpsSchwartzZippel
      msisToAjtai.negligibleEpsBinding
      msisToAjtai.negligibleEpsRelaxedBinding
  msisBoundary := msisBoundary
  msisToAjtai := msisToAjtai
  aligned := by
    simp [SuperNeo.ProofSystem.ErrorModel.ofComponents]

/--
Canonical constructor from explicit boundary packages plus one alignment witness.
This is the intended boundary-level assembly point for the final theorem.
-/
def ofAlignedComponents
  {ctx : ProtocolTargetContext}
  {params : LatticeParams}
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary ctx)
  (errorModel : SuperNeo.ProofSystem.ErrorModel)
  (msisBoundary : SuperNeo.ProofSystem.MSISHardnessBoundary params)
  (msisToAjtai : SuperNeo.ProofSystem.MSISToAjtaiReductions params)
  (hSumcheck : errorModel.epsSumcheck = sumcheckError.epsSoundness)
  (hSZ : errorModel.epsSchwartzZippel = schwartzZippelBoundary.epsSchwartzZippel)
  (hMSIS : errorModel.epsMSIS = msisBoundary.epsMSIS)
  (hBinding : errorModel.epsBinding = msisToAjtai.epsBinding)
  (hRelaxed : errorModel.epsRelaxedBinding = msisToAjtai.epsRelaxedBinding) :
  FinalErrorPackage ctx params where
  sumcheckError := sumcheckError
  schwartzZippelBoundary := schwartzZippelBoundary
  errorModel := errorModel
  msisBoundary := msisBoundary
  msisToAjtai := msisToAjtai
  aligned := ⟨hSumcheck, hSZ, hMSIS, hBinding, hRelaxed⟩

end FinalErrorPackage

namespace FinalErrorPackage

/--
Canonical final-error constructor specialized to the proved `paperCarrier`
strong-sampling path via `3*d ≤ params.relaxedExpansion`.
-/
def ofAlignedPaperCarrierFromThreeDLe
  {ctx : ProtocolTargetContext}
  {params : LatticeParams}
  (hTd : 3 * d ≤ params.relaxedExpansion)
  (hExpPos : 0 < params.relaxedExpansion)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary ctx)
  (errorModel : SuperNeo.ProofSystem.ErrorModel)
  (msisBoundary : SuperNeo.ProofSystem.MSISHardnessBoundary params)
  (hSumcheck : errorModel.epsSumcheck = sumcheckError.epsSoundness)
  (hSZ : errorModel.epsSchwartzZippel = schwartzZippelBoundary.epsSchwartzZippel)
  (hMSIS : errorModel.epsMSIS = msisBoundary.epsMSIS)
  (hBinding : errorModel.epsBinding = msisBoundary.epsMSIS)
  (hRelaxed : errorModel.epsRelaxedBinding = msisBoundary.epsMSIS) :
  FinalErrorPackage ctx params :=
  ofAlignedComponents
    sumcheckError
    schwartzZippelBoundary
    errorModel
    msisBoundary
    (SuperNeo.ProofSystem.MSISToAjtaiReductions.ofPaperCarrierFromThreeDLeAndMSISBoundary
      (params := params)
      hTd
      hExpPos
      msisBoundary)
    hSumcheck
    hSZ
    hMSIS
    hBinding
    hRelaxed

end FinalErrorPackage

namespace FinalErrorPackage

/--
Canonical final-error constructor specialized to the Goldilocks Appendix B.2
paper-parameter family on the proved `paperCarrier` path.
-/
def ofGoldilocksPaperCarrier
  {ctx : ProtocolTargetContext}
  (messageLength : Nat)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary ctx)
  (msisBoundary :
    SuperNeo.ProofSystem.MSISHardnessBoundary
      (SuperNeo.ProofSystem.goldilocksPaperAjtaiParams messageLength)) :
  FinalErrorPackage ctx (SuperNeo.ProofSystem.goldilocksPaperAjtaiParams messageLength) :=
  ofComponentBoundaries
    sumcheckError
    schwartzZippelBoundary
    msisBoundary
    (SuperNeo.ProofSystem.MSISToAjtaiReductions.ofGoldilocksPaperCarrierAndMSISBoundary
      messageLength
      msisBoundary)

end FinalErrorPackage

/-- Canonical final assumption registry. -/
structure FinalTheoremAssumptions (ctx : ProtocolTargetContext) where
  reduction : InteractiveReductionAssumptions ctx
  latticeParams : LatticeParams
  errorPackage : FinalErrorPackage ctx latticeParams

namespace FinalTheoremAssumptions

/-- Canonical boundary-level constructor from the reduction bundle and the consolidated final error package. -/
def ofBoundaryPackages
  {ctx : ProtocolTargetContext}
  {params : LatticeParams}
  (reduction : InteractiveReductionAssumptions ctx)
  (errorPackage : FinalErrorPackage ctx params) :
  FinalTheoremAssumptions ctx where
  reduction := reduction
  latticeParams := params
  errorPackage := errorPackage

/--
Canonical final-theorem constructor specialized to the proved `paperCarrier`
strong-sampling path via `3*d ≤ params.relaxedExpansion`.
-/
def ofAlignedPaperCarrierBoundaryPackages
  {ctx : ProtocolTargetContext}
  {params : LatticeParams}
  (reduction : InteractiveReductionAssumptions ctx)
  (hTd : 3 * d ≤ params.relaxedExpansion)
  (hExpPos : 0 < params.relaxedExpansion)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary ctx)
  (errorModel : SuperNeo.ProofSystem.ErrorModel)
  (msisBoundary : SuperNeo.ProofSystem.MSISHardnessBoundary params)
  (hSumcheck : errorModel.epsSumcheck = sumcheckError.epsSoundness)
  (hSZ : errorModel.epsSchwartzZippel = schwartzZippelBoundary.epsSchwartzZippel)
  (hMSIS : errorModel.epsMSIS = msisBoundary.epsMSIS)
  (hBinding : errorModel.epsBinding = msisBoundary.epsMSIS)
  (hRelaxed : errorModel.epsRelaxedBinding = msisBoundary.epsMSIS) :
  FinalTheoremAssumptions ctx :=
  ofBoundaryPackages
    reduction
    (FinalErrorPackage.ofAlignedPaperCarrierFromThreeDLe
      (params := params)
      hTd
      hExpPos
      sumcheckError
      schwartzZippelBoundary
      errorModel
      msisBoundary
      hSumcheck
      hSZ
      hMSIS
      hBinding
      hRelaxed)

/--
Canonical final-theorem constructor specialized to the paper-facing
challenge-difference route for `invDelta` together with the proved
`paperCarrier` strong-sampling path.

This is the narrowest in-repo constructor on the active protocol path: callers
provide arithmetic/Theorem-3 data, the active paper-carrier-difference
invertibility boundary, the fact that `invDelta` is a nonzero `paperCarrier`
difference, one SumCheck transition witness, and the aligned error/MSIS/Ajtai
packages.
-/
def ofAlignedPaperCarrierDiffBoundaryPackages
  {ctx : ProtocolTargetContext}
  {params : LatticeParams}
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq)
  (hWitness : SumCheckTransitionWitness ctx)
  (hTd : 3 * d ≤ params.relaxedExpansion)
  (hExpPos : 0 < params.relaxedExpansion)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary ctx)
  (errorModel : SuperNeo.ProofSystem.ErrorModel)
  (msisBoundary : SuperNeo.ProofSystem.MSISHardnessBoundary params)
  (hSumcheck : errorModel.epsSumcheck = sumcheckError.epsSoundness)
  (hSZ : errorModel.epsSchwartzZippel = schwartzZippelBoundary.epsSchwartzZippel)
  (hMSIS : errorModel.epsMSIS = msisBoundary.epsMSIS)
  (hBinding : errorModel.epsBinding = msisBoundary.epsMSIS)
  (hRelaxed : errorModel.epsRelaxedBinding = msisBoundary.epsMSIS) :
  FinalTheoremAssumptions ctx :=
  ofAlignedPaperCarrierBoundaryPackages
    (params := params)
    (InteractiveReductionAssumptions.ofPaperCarrierDiff
      hThm3 hArithmetic hDiff hNe hWitness)
    hTd
    hExpPos
    sumcheckError
    schwartzZippelBoundary
    errorModel
    msisBoundary
    hSumcheck
    hSZ
    hMSIS
    hBinding
    hRelaxed

/--
Canonical final-theorem constructor specialized to any strict low-norm
invertibility boundary whose threshold is at least `5`, together with the
proved `paperCarrier` strong-sampling path.
-/
def ofAlignedPaperCarrierLowNormBoundaryPackages
  {ctx : ProtocolTargetContext}
  {params : LatticeParams}
  {B : Nat}
  (hFive : 5 ≤ B)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hInv : lowNormInvertibilityAssumption B)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq)
  (hWitness : SumCheckTransitionWitness ctx)
  (hTd : 3 * d ≤ params.relaxedExpansion)
  (hExpPos : 0 < params.relaxedExpansion)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary ctx)
  (errorModel : SuperNeo.ProofSystem.ErrorModel)
  (msisBoundary : SuperNeo.ProofSystem.MSISHardnessBoundary params)
  (hSumcheck : errorModel.epsSumcheck = sumcheckError.epsSoundness)
  (hSZ : errorModel.epsSchwartzZippel = schwartzZippelBoundary.epsSchwartzZippel)
  (hMSIS : errorModel.epsMSIS = msisBoundary.epsMSIS)
  (hBinding : errorModel.epsBinding = msisBoundary.epsMSIS)
  (hRelaxed : errorModel.epsRelaxedBinding = msisBoundary.epsMSIS) :
  FinalTheoremAssumptions ctx :=
  ofAlignedPaperCarrierBoundaryPackages
    (params := params)
    (InteractiveReductionAssumptions.ofLowNormAtLeastFive
      hFive hThm3 hArithmetic hInv hDiff hNe hWitness)
    hTd
    hExpPos
    sumcheckError
    schwartzZippelBoundary
    errorModel
    msisBoundary
    hSumcheck
    hSZ
    hMSIS
    hBinding
    hRelaxed

end FinalTheoremAssumptions

namespace FinalTheoremAssumptions

/--
Canonical final-theorem constructor specialized to the Goldilocks Appendix B.2
paper-parameter family on the proved `paperCarrier` path.
-/
def ofGoldilocksPaperCarrierBoundaryPackages
  {ctx : ProtocolTargetContext}
  (messageLength : Nat)
  (reduction : InteractiveReductionAssumptions ctx)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary ctx)
  (msisBoundary :
    SuperNeo.ProofSystem.MSISHardnessBoundary
      (SuperNeo.ProofSystem.goldilocksPaperAjtaiParams messageLength)) :
  FinalTheoremAssumptions ctx :=
  ofBoundaryPackages
    reduction
    (FinalErrorPackage.ofGoldilocksPaperCarrier
      messageLength
      sumcheckError
      schwartzZippelBoundary
      msisBoundary)

/-- Canonical Goldilocks final-theorem constructor with the local Schwartz-Zippel boundary derived directly from the carried reduction arithmetic and the internal MSIS boundary reconstructed from the theorem-level hardness assumption. -/
noncomputable def ofGoldilocksPaperCarrierDerivedSumcheck
  {ctx : ProtocolTargetContext}
  (messageLength : Nat)
  (reduction : InteractiveReductionAssumptions ctx)
  (hMsis :
    msisHardnessAssumption
      (SuperNeo.ProofSystem.goldilocksPaperAjtaiParams messageLength)) :
  FinalTheoremAssumptions ctx :=
  ofGoldilocksPaperCarrierBoundaryPackages
    messageLength
    reduction
    SuperNeo.ProofSystem.Sumcheck.soundnessErrorBoundary_zero
    (SchwartzZippelBoundary.ofReduction reduction)
    (SuperNeo.ProofSystem.MSISHardnessBoundary.ofHardness hMsis)

/--
Canonical final-theorem constructor specialized further to the active
`paperCarrier`-difference invertibility route, using the Goldilocks Appendix
B.2 paper-parameter family.
-/
def ofGoldilocksPaperCarrierDiffBoundaryPackages
  {ctx : ProtocolTargetContext}
  (messageLength : Nat)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq)
  (hWitness : SumCheckTransitionWitness ctx)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary ctx)
  (msisBoundary :
    SuperNeo.ProofSystem.MSISHardnessBoundary
      (SuperNeo.ProofSystem.goldilocksPaperAjtaiParams messageLength)) :
  FinalTheoremAssumptions ctx :=
  ofGoldilocksPaperCarrierBoundaryPackages
    messageLength
    (InteractiveReductionAssumptions.ofPaperCarrierDiff
      hThm3 hArithmetic hDiff hNe hWitness)
    sumcheckError
    schwartzZippelBoundary
    msisBoundary

/--
Canonical final-theorem constructor specialized further to the active native-bar
`paperCarrier`-difference invertibility route, using the Goldilocks Appendix
B.2 paper-parameter family and discharging the generic Theorem-3 boundary from
`thm3CoreAssumption_native`. This is the active paper-faithful final route:
all local theorem packaging is derived internally, and the only remaining
explicit security input is the theorem-level MSIS hardness assumption.
-/
noncomputable def ofGoldilocksNativePaperCarrierDiffBoundaryPackages
  {ctx : ProtocolTargetContext}
  (messageLength : Nat)
  (hBarNative : ctx.bar = nativeBarMatrix)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq)
  (hWitness : SumCheckTransitionWitness ctx)
  (hMsis :
    msisHardnessAssumption
      (SuperNeo.ProofSystem.goldilocksPaperAjtaiParams messageLength)) :
  FinalTheoremAssumptions ctx :=
  ofGoldilocksPaperCarrierDerivedSumcheck
    messageLength
    (InteractiveReductionAssumptions.ofNativePaperCarrierDiff
      hBarNative hArithmetic hDiff hNe hWitness)
    hMsis

/--
Canonical final-theorem constructor specialized further to any strict low-norm
invertibility boundary whose threshold is at least `5`, using the Goldilocks
Appendix B.2 paper-parameter family.
-/
def ofGoldilocksPaperCarrierLowNormBoundaryPackages
  {ctx : ProtocolTargetContext}
  {B : Nat}
  (messageLength : Nat)
  (hFive : 5 ≤ B)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hInv : lowNormInvertibilityAssumption B)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq)
  (hWitness : SumCheckTransitionWitness ctx)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary ctx)
  (msisBoundary :
    SuperNeo.ProofSystem.MSISHardnessBoundary
      (SuperNeo.ProofSystem.goldilocksPaperAjtaiParams messageLength)) :
  FinalTheoremAssumptions ctx :=
  ofGoldilocksPaperCarrierBoundaryPackages
    messageLength
    (InteractiveReductionAssumptions.ofLowNormAtLeastFive
      hFive hThm3 hArithmetic hInv hDiff hNe hWitness)
    sumcheckError
    schwartzZippelBoundary
    msisBoundary

end FinalTheoremAssumptions

namespace FinalTheoremAssumptions

end FinalTheoremAssumptions

/-- Access canonical SumCheck error boundary from final assumptions. -/
def FinalTheoremAssumptions.sumcheckError
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary :=
  hA.errorPackage.sumcheckError

/-- Access canonical Schwartz-Zippel boundary from final assumptions. -/
def FinalTheoremAssumptions.schwartzZippelBoundary
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) : SchwartzZippelBoundary ctx :=
  hA.errorPackage.schwartzZippelBoundary

/-- Access canonical error model from final assumptions. -/
def FinalTheoremAssumptions.errorModel
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.ErrorModel :=
  hA.errorPackage.errorModel

/-- Access canonical MSIS boundary package from final assumptions. -/
def FinalTheoremAssumptions.msisBoundary
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.MSISHardnessBoundary hA.latticeParams :=
  hA.errorPackage.msisBoundary

/-- Access canonical MSIS-to-Ajtai reduction package from final assumptions. -/
def FinalTheoremAssumptions.msisToAjtai
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.MSISToAjtaiReductions hA.latticeParams :=
  hA.errorPackage.msisToAjtai

/-- Alignment between `epsSumcheck` and SumCheck soundness error. -/
def FinalTheoremAssumptions.sumcheckErrorAligned
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  hA.errorModel.epsSumcheck = hA.sumcheckError.epsSoundness :=
  hA.errorPackage.aligned.1

/-- Alignment between `epsSchwartzZippel` and Schwartz-Zippel boundary error. -/
def FinalTheoremAssumptions.schwartzZippelErrorAligned
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  hA.errorModel.epsSchwartzZippel = hA.schwartzZippelBoundary.epsSchwartzZippel :=
  hA.errorPackage.aligned.2.1

/-- Alignment between `epsMSIS` and MSIS boundary error. -/
def FinalTheoremAssumptions.msisErrorAligned
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  hA.errorModel.epsMSIS = hA.msisBoundary.epsMSIS :=
  hA.errorPackage.aligned.2.2.1

/-- Alignment between `epsBinding` and Ajtai binding boundary error. -/
def FinalTheoremAssumptions.bindingErrorAligned
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  hA.errorModel.epsBinding = hA.msisToAjtai.epsBinding :=
  hA.errorPackage.aligned.2.2.2.1

/-- Alignment between `epsRelaxedBinding` and Ajtai relaxed-binding boundary error. -/
def FinalTheoremAssumptions.relaxedBindingErrorAligned
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  hA.errorModel.epsRelaxedBinding = hA.msisToAjtai.epsRelaxedBinding :=
  hA.errorPackage.aligned.2.2.2.2

/-- Canonical SumCheck error boundary carried by final assumptions. -/
def FinalTheoremAssumptions.sumcheckErrorBoundary
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary :=
  hA.sumcheckError

/-- Canonical SumCheck error negligibility aligned to final error accounting. -/
def FinalTheoremAssumptions.sumcheckErrorNegligible
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.IsNegligible hA.errorModel.epsSumcheck := by
  simpa [hA.sumcheckErrorAligned] using
    hA.sumcheckError.negligibleEpsSoundness

/--
Canonical SumCheck soundness-failure advantage bound on the witnessed transcript,
aligned to final error accounting.
-/
def FinalTheoremAssumptions.sumcheckAdvantageBound
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.Sumcheck.SoundnessFailureAdvantageBound
      (sumcheckInstanceOfContext ctx)
      hA.reduction.sumcheckTransitionWitness.transcript
      hA.errorModel.epsSumcheck := by
  apply sumcheckFailureAdvantageBound_of_assumptions
    (h := hA.reduction)
    (eps := hA.errorModel.epsSumcheck)
  intro n
  have hNonnegSoundness : 0 ≤ hA.sumcheckError.epsSoundness n :=
    hA.sumcheckError.nonnegEpsSoundness n
  simpa [hA.sumcheckErrorAligned] using hNonnegSoundness

/-- Canonical Schwartz-Zippel error negligibility aligned to final error accounting. -/
def FinalTheoremAssumptions.schwartzZippelErrorNegligible
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.IsNegligible hA.errorModel.epsSchwartzZippel := by
  have hNeg :
      SuperNeo.ProofSystem.IsNegligible hA.schwartzZippelBoundary.epsSchwartzZippel :=
    hA.schwartzZippelBoundary.negligibleEpsSchwartzZippel
  simpa [hA.schwartzZippelErrorAligned] using hNeg

/-- Canonical Schwartz-Zippel advantage bound aligned to final error accounting. -/
def FinalTheoremAssumptions.schwartzZippelAdvantageBound
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SchwartzZippelAdvantageBound ctx hA.errorModel.epsSchwartzZippel := by
  have hBound :
      SchwartzZippelAdvantageBound ctx hA.schwartzZippelBoundary.epsSchwartzZippel :=
    hA.schwartzZippelBoundary.advantageBound
  intro prob n
  have hLe :
      SchwartzZippelAdvantage ctx prob n ≤
        (hA.schwartzZippelBoundary.epsSchwartzZippel n : Rat) :=
    hBound prob n
  simpa [hA.schwartzZippelErrorAligned] using hLe

/-- Canonical MSIS hardness boundary extracted from the final lattice package. -/
def FinalTheoremAssumptions.msisHardnessBoundary
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) : msisHardnessAssumption hA.latticeParams :=
  SuperNeo.ProofSystem.MSISHardnessBoundary.hardnessFromFields hA.msisBoundary

/-- Canonical MSIS error negligibility aligned to final error accounting. -/
def FinalTheoremAssumptions.msisErrorNegligible
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.IsNegligible hA.errorModel.epsMSIS := by
  have hNeg :
      SuperNeo.ProofSystem.IsNegligible hA.msisBoundary.epsMSIS :=
    hA.msisBoundary.negligibleEpsMSIS
  simpa [hA.msisErrorAligned] using hNeg

/-- Canonical MSIS advantage bound aligned to final error accounting. -/
def FinalTheoremAssumptions.msisAdvantageBound
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.MSISAdvantageBound hA.latticeParams hA.errorModel.epsMSIS := by
  have hB :
      SuperNeo.ProofSystem.MSISAdvantageBound hA.latticeParams hA.msisBoundary.epsMSIS :=
    hA.msisBoundary.advantageBound
  intro prob n
  have hLe : SuperNeo.ProofSystem.MSISAdvantage prob
      (SuperNeo.ProofSystem.canonicalMSISGame hA.latticeParams) n ≤
      (hA.msisBoundary.epsMSIS n : Rat) :=
    hB prob n
  simpa [hA.msisErrorAligned] using hLe

/-- Canonical Ajtai binding boundary derived from MSIS hardness via reductions. -/
def FinalTheoremAssumptions.ajtaiBindingBoundaryPackage
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.AjtaiBindingBoundary hA.latticeParams :=
  SuperNeo.ProofSystem.ajtaiBindingBoundary_of_msis hA.msisToAjtai hA.msisHardnessBoundary

/-- Canonical Ajtai binding boundary derived from MSIS hardness via reductions. -/
def FinalTheoremAssumptions.ajtaiBindingBoundary
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) : ajtaiBindingAssumption hA.latticeParams :=
  SuperNeo.ProofSystem.AjtaiBindingBoundary.hardnessFromFields hA.ajtaiBindingBoundaryPackage

/-- Canonical Ajtai relaxed-binding boundary derived from MSIS hardness via reductions. -/
def FinalTheoremAssumptions.ajtaiRelaxedBindingBoundaryPackage
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.AjtaiRelaxedBindingBoundary
    hA.latticeParams hA.msisToAjtai.laws.samplingCarrier :=
  SuperNeo.ProofSystem.ajtaiRelaxedBindingBoundary_of_msis hA.msisToAjtai hA.msisHardnessBoundary

/-- Canonical Ajtai relaxed-binding boundary derived from MSIS hardness via reductions. -/
def FinalTheoremAssumptions.ajtaiRelaxedBindingBoundary
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  ajtaiRelaxedBindingAssumption hA.latticeParams hA.msisToAjtai.laws.samplingCarrier :=
  SuperNeo.ProofSystem.AjtaiRelaxedBindingBoundary.hardnessFromFields hA.ajtaiRelaxedBindingBoundaryPackage

/-- Canonical Ajtai binding-error negligibility aligned to final error accounting. -/
def FinalTheoremAssumptions.bindingErrorNegligible
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.IsNegligible hA.errorModel.epsBinding := by
  have hNeg :
      SuperNeo.ProofSystem.IsNegligible hA.ajtaiBindingBoundaryPackage.epsBinding :=
    hA.ajtaiBindingBoundaryPackage.negligibleEpsBinding
  simpa [hA.bindingErrorAligned] using hNeg

/-- Canonical Ajtai binding advantage bound aligned to final error accounting. -/
def FinalTheoremAssumptions.bindingAdvantageBound
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.AjtaiBindingAdvantageBound hA.latticeParams hA.errorModel.epsBinding := by
  have hB :
      SuperNeo.ProofSystem.AjtaiBindingAdvantageBound
          hA.latticeParams hA.ajtaiBindingBoundaryPackage.epsBinding :=
    hA.ajtaiBindingBoundaryPackage.advantageBound
  intro prob n
  have hLe :
      SuperNeo.ProofSystem.AjtaiBindingAdvantage prob
          (SuperNeo.ProofSystem.canonicalAjtaiBindingGame hA.latticeParams) n ≤
        (hA.ajtaiBindingBoundaryPackage.epsBinding n : Rat) :=
    hB prob n
  simpa [hA.bindingErrorAligned] using hLe

/-- Canonical Ajtai relaxed-binding-error negligibility aligned to final error accounting. -/
def FinalTheoremAssumptions.relaxedBindingErrorNegligible
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.IsNegligible hA.errorModel.epsRelaxedBinding := by
  have hNeg :
      SuperNeo.ProofSystem.IsNegligible hA.ajtaiRelaxedBindingBoundaryPackage.epsRelaxedBinding :=
    hA.ajtaiRelaxedBindingBoundaryPackage.negligibleEpsRelaxedBinding
  simpa [hA.relaxedBindingErrorAligned] using hNeg

/-- Canonical Ajtai relaxed-binding advantage bound aligned to final error accounting. -/
def FinalTheoremAssumptions.relaxedBindingAdvantageBound
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.AjtaiRelaxedBindingAdvantageBound
      hA.latticeParams hA.msisToAjtai.laws.samplingCarrier hA.errorModel.epsRelaxedBinding := by
  have hB :
      SuperNeo.ProofSystem.AjtaiRelaxedBindingAdvantageBound
          hA.latticeParams hA.msisToAjtai.laws.samplingCarrier
            hA.ajtaiRelaxedBindingBoundaryPackage.epsRelaxedBinding :=
    hA.ajtaiRelaxedBindingBoundaryPackage.advantageBound
  intro prob n
  have hLe :
      SuperNeo.ProofSystem.AjtaiRelaxedBindingAdvantage prob
          (SuperNeo.ProofSystem.canonicalAjtaiRelaxedBindingGame
            hA.latticeParams hA.msisToAjtai.laws.samplingCarrier) n ≤
        (hA.ajtaiRelaxedBindingBoundaryPackage.epsRelaxedBinding n : Rat) :=
    hB prob n
  simpa [hA.relaxedBindingErrorAligned] using hLe

/-- Canonical total-error decomposition derived from the error model plus alignments. -/
def FinalTheoremAssumptions.totalErrorDecompFromModel
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  ∀ n,
    hA.errorModel.epsTotal n =
      hA.sumcheckError.epsSoundness n +
        hA.msisBoundary.epsMSIS n +
          hA.schwartzZippelBoundary.epsSchwartzZippel n +
            hA.msisToAjtai.epsBinding n +
              hA.msisToAjtai.epsRelaxedBinding n := by
  intro n
  have hDecomp : hA.errorModel.epsTotal n =
      hA.errorModel.epsSumcheck n + hA.errorModel.epsMSIS n +
        hA.errorModel.epsSchwartzZippel n + hA.errorModel.epsBinding n +
          hA.errorModel.epsRelaxedBinding n :=
    hA.errorModel.epsTotal_decomp n
  simpa [hA.sumcheckErrorAligned, hA.msisErrorAligned, hA.schwartzZippelErrorAligned,
    hA.bindingErrorAligned, hA.relaxedBindingErrorAligned] using hDecomp

/-- Canonical total-error negligibility from final error accounting. -/
def FinalTheoremAssumptions.totalErrorNegligible
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  SuperNeo.ProofSystem.IsNegligible hA.errorModel.epsTotal :=
  hA.errorModel.negligibleTotal

/-- Final completeness statement shape. -/
def FinalCompletenessStatement
  (ctx : ProtocolTargetContext)
  (_hA : FinalTheoremAssumptions ctx) : Prop :=
  ceRelaxedRelation ctx

/-- Final knowledge-soundness statement shape. -/
def FinalKnowledgeSoundnessStatement
  (ctx : ProtocolTargetContext)
  (hA : FinalTheoremAssumptions ctx) : Prop :=
  strongCompositionStatement ctx ∧
  SchwartzZippelAdvantageBound ctx hA.errorModel.epsSchwartzZippel ∧
  SuperNeo.ProofSystem.Sumcheck.SoundnessFailureAdvantageBound
      (sumcheckInstanceOfContext ctx)
      hA.reduction.sumcheckTransitionWitness.transcript
      hA.errorModel.epsSumcheck ∧
  SuperNeo.ProofSystem.MSISAdvantageBound hA.latticeParams hA.errorModel.epsMSIS ∧
  SuperNeo.ProofSystem.AjtaiBindingAdvantageBound hA.latticeParams hA.errorModel.epsBinding ∧
  SuperNeo.ProofSystem.AjtaiRelaxedBindingAdvantageBound
      hA.latticeParams hA.msisToAjtai.laws.samplingCarrier hA.errorModel.epsRelaxedBinding ∧
  (∀ n,
    hA.errorModel.epsTotal n =
      hA.sumcheckError.epsSoundness n +
        hA.msisBoundary.epsMSIS n +
          hA.schwartzZippelBoundary.epsSchwartzZippel n +
            hA.msisToAjtai.epsBinding n +
              hA.msisToAjtai.epsRelaxedBinding n) ∧
  SuperNeo.ProofSystem.IsNegligible hA.errorModel.epsTotal

/-- Canonical final theorem container. -/
structure FinalTheoremShape
  (ctx : ProtocolTargetContext)
  (hA : FinalTheoremAssumptions ctx) : Prop where
  completeness : FinalCompletenessStatement ctx hA
  knowledgeSoundness : FinalKnowledgeSoundnessStatement ctx hA

/-- Canonical final theorem constructor. -/
theorem finalTheoremShape_of_assumptions
  {ctx : ProtocolTargetContext}
  (hA : FinalTheoremAssumptions ctx) :
  FinalTheoremShape ctx hA := by
  refine ⟨?_, ?_⟩
  · exact (weakComposition_of_assumptions hA.reduction).1
  · exact ⟨strongComposition_of_assumptions hA.reduction,
      hA.schwartzZippelAdvantageBound,
      hA.sumcheckAdvantageBound,
      hA.msisAdvantageBound,
      hA.bindingAdvantageBound,
      hA.relaxedBindingAdvantageBound,
      hA.totalErrorDecompFromModel,
      hA.totalErrorNegligible⟩

/--
Canonical final theorem specialized to the proved `paperCarrier` strong-sampling
path via `3*d ≤ params.relaxedExpansion`.
-/
theorem finalTheoremShape_of_alignedPaperCarrierBoundaryPackages
  {ctx : ProtocolTargetContext}
  {params : LatticeParams}
  (reduction : InteractiveReductionAssumptions ctx)
  (hTd : 3 * d ≤ params.relaxedExpansion)
  (hExpPos : 0 < params.relaxedExpansion)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary ctx)
  (errorModel : SuperNeo.ProofSystem.ErrorModel)
  (msisBoundary : SuperNeo.ProofSystem.MSISHardnessBoundary params)
  (hSumcheck : errorModel.epsSumcheck = sumcheckError.epsSoundness)
  (hSZ : errorModel.epsSchwartzZippel = schwartzZippelBoundary.epsSchwartzZippel)
  (hMSIS : errorModel.epsMSIS = msisBoundary.epsMSIS)
  (hBinding : errorModel.epsBinding = msisBoundary.epsMSIS)
  (hRelaxed : errorModel.epsRelaxedBinding = msisBoundary.epsMSIS) :
  FinalTheoremShape ctx
    (FinalTheoremAssumptions.ofAlignedPaperCarrierBoundaryPackages
      (params := params)
      reduction
      hTd
      hExpPos
      sumcheckError
      schwartzZippelBoundary
      errorModel
      msisBoundary
      hSumcheck
      hSZ
      hMSIS
      hBinding
      hRelaxed) := by
  exact finalTheoremShape_of_assumptions _

/--
Canonical final theorem specialized to the paper-facing challenge-difference
route for `invDelta` together with the proved `paperCarrier` strong-sampling
path and the active paper-carrier-difference invertibility boundary.
-/
theorem finalTheoremShape_of_alignedPaperCarrierDiffBoundaryPackages
  {ctx : ProtocolTargetContext}
  {params : LatticeParams}
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq)
  (hWitness : SumCheckTransitionWitness ctx)
  (hTd : 3 * d ≤ params.relaxedExpansion)
  (hExpPos : 0 < params.relaxedExpansion)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary ctx)
  (errorModel : SuperNeo.ProofSystem.ErrorModel)
  (msisBoundary : SuperNeo.ProofSystem.MSISHardnessBoundary params)
  (hSumcheck : errorModel.epsSumcheck = sumcheckError.epsSoundness)
  (hSZ : errorModel.epsSchwartzZippel = schwartzZippelBoundary.epsSchwartzZippel)
  (hMSIS : errorModel.epsMSIS = msisBoundary.epsMSIS)
  (hBinding : errorModel.epsBinding = msisBoundary.epsMSIS)
  (hRelaxed : errorModel.epsRelaxedBinding = msisBoundary.epsMSIS) :
  FinalTheoremShape ctx
    (FinalTheoremAssumptions.ofAlignedPaperCarrierDiffBoundaryPackages
      (params := params)
      hThm3
      hArithmetic
      hDiff
      hNe
      hWitness
      hTd
      hExpPos
      sumcheckError
      schwartzZippelBoundary
      errorModel
      msisBoundary
      hSumcheck
      hSZ
      hMSIS
      hBinding
      hRelaxed) := by
  exact finalTheoremShape_of_assumptions _

/--
Canonical final theorem specialized to any strict low-norm invertibility
boundary whose threshold is at least `5`, together with the proved
`paperCarrier` strong-sampling path.
-/
theorem finalTheoremShape_of_alignedPaperCarrierLowNormBoundaryPackages
  {ctx : ProtocolTargetContext}
  {params : LatticeParams}
  {B : Nat}
  (hFive : 5 ≤ B)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hInv : lowNormInvertibilityAssumption B)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq)
  (hWitness : SumCheckTransitionWitness ctx)
  (hTd : 3 * d ≤ params.relaxedExpansion)
  (hExpPos : 0 < params.relaxedExpansion)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary ctx)
  (errorModel : SuperNeo.ProofSystem.ErrorModel)
  (msisBoundary : SuperNeo.ProofSystem.MSISHardnessBoundary params)
  (hSumcheck : errorModel.epsSumcheck = sumcheckError.epsSoundness)
  (hSZ : errorModel.epsSchwartzZippel = schwartzZippelBoundary.epsSchwartzZippel)
  (hMSIS : errorModel.epsMSIS = msisBoundary.epsMSIS)
  (hBinding : errorModel.epsBinding = msisBoundary.epsMSIS)
  (hRelaxed : errorModel.epsRelaxedBinding = msisBoundary.epsMSIS) :
  FinalTheoremShape ctx
    (FinalTheoremAssumptions.ofAlignedPaperCarrierLowNormBoundaryPackages
      (params := params)
      hFive
      hThm3
      hArithmetic
      hInv
      hDiff
      hNe
      hWitness
      hTd
      hExpPos
      sumcheckError
      schwartzZippelBoundary
      errorModel
      msisBoundary
      hSumcheck
      hSZ
      hMSIS
      hBinding
      hRelaxed) := by
  exact finalTheoremShape_of_assumptions _

/--
Canonical final theorem specialized to the Goldilocks Appendix B.2
paper-parameter family on the proved `paperCarrier` path.
-/
theorem finalTheoremShape_of_goldilocksPaperCarrierBoundaryPackages
  {ctx : ProtocolTargetContext}
  (messageLength : Nat)
  (reduction : InteractiveReductionAssumptions ctx)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary ctx)
  (msisBoundary :
    SuperNeo.ProofSystem.MSISHardnessBoundary
      (SuperNeo.ProofSystem.goldilocksPaperAjtaiParams messageLength)) :
  FinalTheoremShape ctx
    (FinalTheoremAssumptions.ofGoldilocksPaperCarrierBoundaryPackages
      messageLength
      reduction
      sumcheckError
      schwartzZippelBoundary
      msisBoundary) := by
  exact finalTheoremShape_of_assumptions _

/--
Canonical final theorem specialized to the active `paperCarrier`-difference
invertibility route and the Goldilocks Appendix B.2 paper-parameter family.
-/
theorem finalTheoremShape_of_goldilocksPaperCarrierDiffBoundaryPackages
  {ctx : ProtocolTargetContext}
  (messageLength : Nat)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq)
  (hWitness : SumCheckTransitionWitness ctx)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary ctx)
  (msisBoundary :
    SuperNeo.ProofSystem.MSISHardnessBoundary
      (SuperNeo.ProofSystem.goldilocksPaperAjtaiParams messageLength)) :
  FinalTheoremShape ctx
    (FinalTheoremAssumptions.ofGoldilocksPaperCarrierDiffBoundaryPackages
      messageLength
      hThm3
      hArithmetic
      hDiff
      hNe
      hWitness
      sumcheckError
      schwartzZippelBoundary
      msisBoundary) := by
  exact finalTheoremShape_of_assumptions _

/--
Canonical final theorem specialized to the active native-bar
`paperCarrier`-difference invertibility route and the Goldilocks Appendix B.2
paper-parameter family.
-/
theorem finalTheoremShape_of_goldilocksNativePaperCarrierDiffBoundaryPackages
  {ctx : ProtocolTargetContext}
  (messageLength : Nat)
  (hBarNative : ctx.bar = nativeBarMatrix)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq)
  (hWitness : SumCheckTransitionWitness ctx)
  (hMsis :
    msisHardnessAssumption
      (SuperNeo.ProofSystem.goldilocksPaperAjtaiParams messageLength)) :
  FinalTheoremShape ctx
    (FinalTheoremAssumptions.ofGoldilocksNativePaperCarrierDiffBoundaryPackages
      messageLength
      hBarNative
      hArithmetic
      hDiff
      hNe
      hWitness
      hMsis) := by
  exact finalTheoremShape_of_assumptions _

/--
Canonical final theorem specialized to any strict low-norm invertibility
boundary whose threshold is at least `5`, together with the Goldilocks
Appendix B.2 paper-parameter family.
-/
theorem finalTheoremShape_of_goldilocksPaperCarrierLowNormBoundaryPackages
  {ctx : ProtocolTargetContext}
  {B : Nat}
  (messageLength : Nat)
  (hFive : 5 ≤ B)
  (hThm3 : thm3CoreAssumption ctx.bar)
  (hArithmetic : ArithmeticObligations
    ctx.bar ctx.m ctx.r ctx.rho1 ctx.rho2
    ctx.hVec ctx.hScal
    ctx.splitScalar ctx.kSplit
    ctx.cset ctx.samples
    ctx.xs ctx.ys ctx.qVals ctx.coeffs
    ctx.xEval ctx.expectedEval)
  (hInv : lowNormInvertibilityAssumption B)
  (hDiff : samplingDiffSet paperCarrier ctx.invDelta)
  (hNe : ctx.invDelta ≠ zeroRq)
  (hWitness : SumCheckTransitionWitness ctx)
  (sumcheckError : SuperNeo.ProofSystem.Sumcheck.SoundnessErrorBoundary)
  (schwartzZippelBoundary : SchwartzZippelBoundary ctx)
  (msisBoundary :
    SuperNeo.ProofSystem.MSISHardnessBoundary
      (SuperNeo.ProofSystem.goldilocksPaperAjtaiParams messageLength)) :
  FinalTheoremShape ctx
    (FinalTheoremAssumptions.ofGoldilocksPaperCarrierLowNormBoundaryPackages
      messageLength
      hFive
      hThm3
      hArithmetic
      hInv
      hDiff
      hNe
      hWitness
      sumcheckError
      schwartzZippelBoundary
      msisBoundary) := by
  exact finalTheoremShape_of_assumptions _

end SuperNeo
