import NightstreamFPrime.Export.Stage1.HyperNovaHistory
import NightstreamFPrime.Export.Stage1.HyperNovaFirstFailure
import NightstreamFPrime.Export.Stage1.NifsProviderLaw

/-!
The quantitative source-failure bound at the actual visited history laws.
The operational source uses one fixed total family of raw calls, tapes and
clocks. Each FS model is for the exact unconditional guarded real experiment
and its supported restriction of that same family. The transfer functions
are shared across the finite visit family; query counts may differ.

This module gives no new numerical cryptographic assumption, context sampler,
averaged-work bound, or whole-history runtime claim.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaVisitedSecurity

open scoped BigOperators ENNReal
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.Nifs
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.Nifs
open PiRLC.CoordinateForkLaw (Challenge)
open HyperNovaHistory (Statement)
open HyperNovaVisitedLaw (Visit goodActive visitedLaw guardedDraw)
open HyperNovaGuardedSourceLaw (inputs realLaw guardedPrefix)
open HyperNovaFirstFailure (MarkedSourceFailure)

variable (target : Wide.Target)

private theorem event_ne_top {Sample : Type*} (distribution : PMF Sample) (event : Set Sample) :
    distribution.toOuterMeasure event ≠ ∞ := by
  rw [PMF.toOuterMeasure_apply]
  exact distribution.tsum_coe_indicator_ne_top event

private theorem first_failure_real_bound
    (source : Statement → target.Payload → PMF (HyperNovaHistory.SourceResult target))
    (initial : PMF (Statement × target.Envelope)) (depth : Nat)
    (bounded : ∀ input ∈ initial.support, input.1.iteration ≤ depth) :
    (initial.toOuterMeasure {input |
      target.Holds input.1 input.2}).toReal ≤
      ((HyperNovaHistoryLaw.law target source initial).toOuterMeasure
        {sample | HyperNovaHistoryProbability.AdviceReturned target sample}).toReal +
        ∑ j : Fin depth,
          (((visitedLaw target source initial j.val).toOuterMeasure
              {visit | HyperNovaFirstFailure.MarkedHashCollision target visit}).toReal +
            (((visitedLaw target source initial j.val).bind (guardedDraw target source)).toOuterMeasure
              {draw | MarkedSourceFailure target draw}).toReal) := by
  have bound := HyperNovaFirstFailure.accepted_probability_le_first_failures target source initial
      depth bounded
  have finiteTerm (j : Fin depth) :
      (visitedLaw target source initial j.val).toOuterMeasure
          {visit | HyperNovaFirstFailure.MarkedHashCollision target visit} +
        ((visitedLaw target source initial j.val).bind (guardedDraw target source)).toOuterMeasure
          {draw | MarkedSourceFailure target draw} ≠ ∞ :=
    ENNReal.add_ne_top.mpr ⟨event_ne_top _ _, event_ne_top _ _⟩
  have finiteSum := ENNReal.sum_ne_top.mpr (fun j (_ : j ∈ Finset.univ) => finiteTerm j)
  have realBound := ENNReal.toReal_mono
    (ENNReal.add_ne_top.mpr ⟨event_ne_top _ _, finiteSum⟩) bound
  rw [ENNReal.toReal_add (event_ne_top _ _) finiteSum,
    ENNReal.toReal_sum (fun j (_ : j ∈ Finset.univ) => finiteTerm j)] at realBound
  have realTerm (j : Fin depth) := ENNReal.toReal_add
    (event_ne_top (visitedLaw target source initial j.val)
      {visit | HyperNovaFirstFailure.MarkedHashCollision target visit})
    (event_ne_top ((visitedLaw target source initial j.val).bind (guardedDraw target source))
      {draw | MarkedSourceFailure target draw})
  simpa only [realTerm] using realBound

variable {State Tape : Type*}
  (tapes : Visit target → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → PMF Tape)
  (rawCall : Visit target → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State →
      PaperWeakOracle.Call (Tape := Tape) (arity := PaperProfile.arity)
          (NifsExtractionProvider.rlc target.security))
  (checkClock : Visit target → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → NifsExtractionProvider.CheckClock
        target.security)
  (storageClock : Visit target → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → NifsExtractionProvider.StorageClock
        target.security)
  (parentClock : Visit target → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → NifsExtractionProvider.ParentClock
        target.security)
  (storageBound : Visit target → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → Nat)
  (storageBounded : ∀ visit coins output state assignments,
    storageClock visit coins output state assignments ≤ storageBound visit coins output state)
  (baseSummable : ∀ visit coins output state vector, Summable fun tape =>
    (tapes visit coins output state tape).toReal *
      (PaperWeakOracle.baseWork (NifsExtractionProvider.rlc target.security)
        (NifsExtractionProvider.suffixProgram target.security
            (NifsExtractionProvider.batchAt target.security (inputs target) visit coins output)
          (checkClock visit coins output state) (storageClock visit coins output state))
        (rawCall visit coins output state) vector tape : ℝ))
  [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key target.relation target.ajtai).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key target.relation target.ajtai).piRlcAlgebra)]

/-- The v1.2 per-visit source-failure bound uses the actual adaptive MSIS
reduction, including both acceptance gates. The visited law, guarded FS
models and source program are unchanged. This probability statement does
not assert a global work bound or discharge query applicability. -/
theorem source_failure_probability_linear_le
    (initial : PMF (Statement × target.Envelope)) (depth : Nat)
    (originalFirstPhase : Visit target → InteractivePrefix.Prover State productionShape 9)
    (abortTape : Tape) (g : Nat → ℝ → ℝ) (deltaFS : Nat → ℝ) (queries : Fin depth → Nat)
    (scalarSubClock : RingF → RingF → Nat) (inverseAdapterClock : RingF → Nat)
    (assignmentSubClock : PiRLCExtractionPrimitives.Assignment target.security →
        PiRLCExtractionPrimitives.Assignment target.security → Nat)
    (scalarActionClock : RingF → PiRLCExtractionPrimitives.Assignment target.security → Nat)
    (sourceCheckClock : Visit target → PiCCSStoredSourceProbability.CheckClock target.security)
    (accessClock : Visit target → PiCCSStoredSourceProbability.AccessClock target.security)
    (lowNorm : Phi81StrongSet.LowNormInvertibility)
    (bounds : PiRLC.PaperForkExtractionWork.PrimitiveBounds)
    (bounded : PiRLC.PaperForkExtractionWork.Bounded
      (PaperExtractionAlgebra.extractionAlgebra target.ajtai).ring
      (PiRLCExtractionPrimitives.program target.security scalarSubClock inverseAdapterClock
        assignmentSubClock scalarActionClock) bounds) :
    let continuation := NifsProviderLaw.continuation target.security (inputs target) tapes rawCall
        checkClock storageClock parentClock
      storageBound storageBounded baseSummable
    let program := PiRLCExtractionPrimitives.program target.security scalarSubClock inverseAdapterClock
      assignmentSubClock scalarActionClock
    let source := HyperNovaGuardedSourceLaw.source target originalFirstPhase continuation program
    let visits := fun j : Fin depth => visitedLaw target source initial j.val
    let running := fun visit => target.security.running (inputs target visit)
    let fresh := fun visit => target.security.fresh (inputs target visit)
    let firstPhase := guardedPrefix target originalFirstPhase
    let checked := InteractiveComposition.firstPhase firstPhase (SupportedExtraction.publicCheck running)
    let contexts := fun j : Fin depth => FiatShamirTransfer.contextLaw target.relation
        (realLaw target (visits j))
    let provider := fun j : Fin depth =>
      (NifsProviderLaw.supportedProvider target.security) (inputs target) tapes rawCall checkClock
          storageClock parentClock
        storageBound storageBounded baseSummable (contexts j) checked
    let extended := fun j : Fin depth =>
      SupportedContinuation.extension target.relation target.ajtai running fresh (contexts j) checked
        abortTape (provider j)
    (∀ j : Fin depth,
      WideFiatShamir.FiatShamirModel target.relation target.ajtai running fresh
        (realLaw target (visits j)) firstPhase abortTape (provider j) g deltaFS (queries j)) →
    ∀ j : Fin depth,
      (((visits j).bind (guardedDraw target source)).toOuterMeasure
        {draw | MarkedSourceFailure target draw}).toReal ≤
      ((visits j).toOuterMeasure {visit | goodActive target visit}).toReal -
        g (queries j) ((visits j).toOuterMeasure {visit | goodActive target visit}).toReal + deltaFS
            (queries j) +
        InteractiveComposition.weakLoss target.relation target.ajtai +
        IndependentExecution.testError productionShape 9 +
        AdaptiveBindingProbability.successProbability target.relation target.ajtai program running fresh
          firstPhase (SupportedExtraction.publicCheck running) (extended j)
          (fun visit => PiCCSStoredSourceProbability.sourceProgram target.security (inputs target visit)
            (sourceCheckClock visit) (accessClock visit)) (contexts j) * PaperProfile.arity.total := by
  dsimp only
  intro models j
  let continuation := NifsProviderLaw.continuation target.security (inputs target) tapes rawCall
      checkClock storageClock parentClock
    storageBound storageBounded baseSummable
  let program := PiRLCExtractionPrimitives.program target.security scalarSubClock inverseAdapterClock
    assignmentSubClock scalarActionClock
  let source := HyperNovaGuardedSourceLaw.source target originalFirstPhase continuation program
  let visits := visitedLaw target source initial j.val
  have extracted := NifsProviderLaw.source_probability_linear_bound target.security (inputs target)
      tapes rawCall checkClock storageClock
    parentClock storageBound storageBounded baseSummable (realLaw target visits)
        (guardedPrefix target originalFirstPhase)
    abortTape g deltaFS (queries j) (models j) scalarSubClock inverseAdapterClock
    assignmentSubClock scalarActionClock sourceCheckClock accessClock lowNorm bounds bounded
  dsimp only at extracted
  rw [HyperNovaVisitedAcceptance.realSuccessProbability_eq_goodActive] at extracted
  simp only [HyperNovaGuardedSourceLaw.realLaw_context_marginal] at extracted ⊢
  have sameLaw : HyperNovaSourceLaw.law target.security (inputs target) visits
      (guardedPrefix target originalFirstPhase) continuation program =
      visits.bind (guardedDraw target source) :=
    HyperNovaGuardedSourceLaw.law_eq_guardedDraw target originalFirstPhase continuation program visits
  rw [sameLaw] at extracted
  have partition := HyperNovaGuardedSourceLaw.first_source_failure_mass_eq target
    originalFirstPhase continuation program visits
  rw [HyperNovaGuardedSourceLaw.law_eq_guardedDraw] at partition
  change ((visits.bind (guardedDraw target source)).toOuterMeasure
      {draw | MarkedSourceFailure target draw}).toReal = _ at partition
  rw [partition]
  dsimp only [visits, source, continuation, program] at extracted ⊢
  linarith only [extracted]

/-- Final v1.2 history probability bound under the approved guarded FS
models. Each actual visit contributes additive test loss and the computed
adaptive MSIS success with its uniform source-coordinate loss. Depth and
queries remain parameters. Fixed-seed hardness, efficient translation,
global work and query applicability are separate explicit obligations. -/
theorem history_probability_linear_bound
    (initial : PMF (Statement × target.Envelope)) (depth : Nat)
    (depthBound : ∀ input ∈ initial.support, input.1.iteration ≤ depth)
    (originalFirstPhase : Visit target → InteractivePrefix.Prover State productionShape 9)
    (abortTape : Tape) (g : Nat → ℝ → ℝ) (deltaFS : Nat → ℝ) (queries : Fin depth → Nat)
    (scalarSubClock : RingF → RingF → Nat) (inverseAdapterClock : RingF → Nat)
    (assignmentSubClock : PiRLCExtractionPrimitives.Assignment target.security →
        PiRLCExtractionPrimitives.Assignment target.security → Nat)
    (scalarActionClock : RingF → PiRLCExtractionPrimitives.Assignment target.security → Nat)
    (sourceCheckClock : Visit target → PiCCSStoredSourceProbability.CheckClock target.security)
    (accessClock : Visit target → PiCCSStoredSourceProbability.AccessClock target.security)
    (lowNorm : Phi81StrongSet.LowNormInvertibility)
    (bounds : PiRLC.PaperForkExtractionWork.PrimitiveBounds)
    (bounded : PiRLC.PaperForkExtractionWork.Bounded
      (PaperExtractionAlgebra.extractionAlgebra target.ajtai).ring
      (PiRLCExtractionPrimitives.program target.security scalarSubClock inverseAdapterClock
        assignmentSubClock scalarActionClock) bounds) :
    let continuation := NifsProviderLaw.continuation target.security (inputs target) tapes rawCall
        checkClock storageClock parentClock
      storageBound storageBounded baseSummable
    let program := PiRLCExtractionPrimitives.program target.security scalarSubClock inverseAdapterClock
      assignmentSubClock scalarActionClock
    let source := HyperNovaGuardedSourceLaw.source target originalFirstPhase continuation program
    let visits := fun j : Fin depth => visitedLaw target source initial j.val
    let running := fun visit => target.security.running (inputs target visit)
    let fresh := fun visit => target.security.fresh (inputs target visit)
    let firstPhase := guardedPrefix target originalFirstPhase
    let checked := InteractiveComposition.firstPhase firstPhase (SupportedExtraction.publicCheck running)
    let contexts := fun j : Fin depth => FiatShamirTransfer.contextLaw target.relation
        (realLaw target (visits j))
    let provider := fun j : Fin depth =>
      (NifsProviderLaw.supportedProvider target.security) (inputs target) tapes rawCall checkClock
          storageClock parentClock
        storageBound storageBounded baseSummable (contexts j) checked
    let extended := fun j : Fin depth =>
      SupportedContinuation.extension target.relation target.ajtai running fresh (contexts j) checked
        abortTape (provider j)
    (∀ j : Fin depth,
      WideFiatShamir.FiatShamirModel target.relation target.ajtai running fresh
        (realLaw target (visits j)) firstPhase abortTape (provider j) g deltaFS (queries j)) →
    (initial.toOuterMeasure {input |
      target.Holds input.1 input.2}).toReal ≤
      ((HyperNovaHistoryLaw.law target source initial).toOuterMeasure
        {sample | HyperNovaHistoryProbability.AdviceReturned target sample}).toReal +
        ∑ j : Fin depth,
          (((visits j).toOuterMeasure
              {visit | HyperNovaFirstFailure.MarkedHashCollision target visit}).toReal +
            (((visits j).toOuterMeasure {visit | goodActive target visit}).toReal -
              g (queries j) ((visits j).toOuterMeasure {visit | goodActive target visit}).toReal +
              deltaFS (queries j) + InteractiveComposition.weakLoss target.relation target.ajtai +
              IndependentExecution.testError productionShape 9 +
              AdaptiveBindingProbability.successProbability target.relation target.ajtai program running fresh
                firstPhase (SupportedExtraction.publicCheck running) (extended j)
                (fun visit => PiCCSStoredSourceProbability.sourceProgram target.security (inputs target visit)
                  (sourceCheckClock visit) (accessClock visit)) (contexts j) * PaperProfile.arity.total)) := by
  dsimp only
  intro models
  let continuation := NifsProviderLaw.continuation target.security (inputs target) tapes rawCall
      checkClock storageClock parentClock
    storageBound storageBounded baseSummable
  let program := PiRLCExtractionPrimitives.program target.security scalarSubClock inverseAdapterClock
    assignmentSubClock scalarActionClock
  let source := HyperNovaGuardedSourceLaw.source target originalFirstPhase continuation program
  have first := first_failure_real_bound target source initial depth depthBound
  have each := source_failure_probability_linear_le target tapes rawCall checkClock storageClock parentClock
    storageBound storageBounded baseSummable initial depth originalFirstPhase abortTape g deltaFS queries
    scalarSubClock inverseAdapterClock assignmentSubClock scalarActionClock sourceCheckClock accessClock
    lowNorm bounds bounded models
  apply first.trans
  apply add_le_add_right
  apply Finset.sum_le_sum
  intro j _member
  exact add_le_add_right (each j) _

end NightstreamFPrime.Export.Stage1.HyperNovaVisitedSecurity
