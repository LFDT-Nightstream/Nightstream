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
open HyperNovaHistory (Statement Envelope)
open HyperNovaVisitedLaw (Visit goodActive visitedLaw guardedDraw)
open HyperNovaGuardedSourceLaw (inputs realLaw guardedPrefix)
open HyperNovaFirstFailure (MarkedSourceFailure)
open Poseidon2HashChainV1Setup (productionAjtaiKey)
open PiDECInputCheck (relation)

private theorem event_ne_top {Sample : Type*} (distribution : PMF Sample) (event : Set Sample) :
    distribution.toOuterMeasure event ≠ ∞ := by
  rw [PMF.toOuterMeasure_apply]
  exact distribution.tsum_coe_indicator_ne_top event

private theorem first_failure_real_bound
    (source : Statement → HyperNovaHistory.Payload → PMF HyperNovaHistory.SourceResult)
    (initial : PMF (Statement × Envelope)) (depth : Nat)
    (bounded : ∀ input ∈ initial.support, input.1.iteration ≤ depth) :
    (initial.toOuterMeasure {input |
      PerApplicationTerminal.Holds Poseidon2HashChainV1Package.application
        Poseidon2HashChainV1Package.fits Poseidon2HashChainV1Setup.productionSetup input.1 input.2}).toReal ≤
      ((HyperNovaHistoryLaw.law source initial).toOuterMeasure
        {sample | HyperNovaHistoryProbability.AdviceReturned sample}).toReal +
        ∑ j : Fin depth,
          (((visitedLaw source initial j.val).toOuterMeasure
              {visit | HyperNovaFirstFailure.MarkedHashCollision visit}).toReal +
            (((visitedLaw source initial j.val).bind (guardedDraw source)).toOuterMeasure
              {draw | MarkedSourceFailure draw}).toReal) := by
  have bound := HyperNovaFirstFailure.accepted_probability_le_first_failures source initial depth bounded
  have finiteTerm (j : Fin depth) :
      (visitedLaw source initial j.val).toOuterMeasure
          {visit | HyperNovaFirstFailure.MarkedHashCollision visit} +
        ((visitedLaw source initial j.val).bind (guardedDraw source)).toOuterMeasure
          {draw | MarkedSourceFailure draw} ≠ ∞ :=
    ENNReal.add_ne_top.mpr ⟨event_ne_top _ _, event_ne_top _ _⟩
  have finiteSum := ENNReal.sum_ne_top.mpr (fun j (_ : j ∈ Finset.univ) => finiteTerm j)
  have realBound := ENNReal.toReal_mono
    (ENNReal.add_ne_top.mpr ⟨event_ne_top _ _, finiteSum⟩) bound
  rw [ENNReal.toReal_add (event_ne_top _ _) finiteSum,
    ENNReal.toReal_sum (fun j (_ : j ∈ Finset.univ) => finiteTerm j)] at realBound
  have realTerm (j : Fin depth) := ENNReal.toReal_add
    (event_ne_top (visitedLaw source initial j.val)
      {visit | HyperNovaFirstFailure.MarkedHashCollision visit})
    (event_ne_top ((visitedLaw source initial j.val).bind (guardedDraw source))
      {draw | MarkedSourceFailure draw})
  simpa only [realTerm] using realBound

variable {State Tape : Type*}
  (tapes : Visit → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → PMF Tape)
  (rawCall : Visit → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State →
      PaperWeakOracle.Call (Tape := Tape) (arity := PaperProfile.arity) NifsExtractionProvider.rlc)
  (checkClock : Visit → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → NifsExtractionProvider.CheckClock)
  (storageClock : Visit → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → NifsExtractionProvider.StorageClock)
  (parentClock : Visit → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → NifsExtractionProvider.ParentClock)
  (storageBound : Visit → PublicCoins K productionShape →
    FullOutputCoordinates.FullOutput K productionShape → State → Nat)
  (storageBounded : ∀ visit coins output state assignments,
    storageClock visit coins output state assignments ≤ storageBound visit coins output state)
  (baseSummable : ∀ visit coins output state vector, Summable fun tape =>
    (tapes visit coins output state tape).toReal *
      (PaperWeakOracle.baseWork NifsExtractionProvider.rlc
        (NifsExtractionProvider.suffixProgram (NifsExtractionProvider.batchAt inputs visit coins output)
          (checkClock visit coins output state) (storageClock visit coins output state))
        (rawCall visit coins output state) vector tape : ℝ))
  [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key relation productionAjtaiKey).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation productionAjtaiKey).piRlcAlgebra)]

/-- Every observed source-failure mass is bounded using the same approved
transfer functions and the actual selected MSIS reduction at that visit.
The real success mass, context marginal, provider agreement and source law
are derived. The explicit hypotheses are the exact guarded FS models,
low-norm invertibility and the existing declared primitive-clock bounds. -/
theorem source_failure_probability_le
    (initial : PMF (Statement × Envelope)) (depth : Nat)
    (originalFirstPhase : Visit → InteractivePrefix.Prover State productionShape 9)
    (abortTape : Tape) (g : Nat → ℝ → ℝ) (deltaFS : Nat → ℝ) (queries : Fin depth → Nat)
    (scalarSubClock : RingF → RingF → Nat) (inverseAdapterClock : RingF → Nat)
    (assignmentSubClock : PiRLCExtractionPrimitives.Assignment → PiRLCExtractionPrimitives.Assignment → Nat)
    (scalarActionClock : RingF → PiRLCExtractionPrimitives.Assignment → Nat)
    (sourceCheckClock : Visit → PiCCSStoredSourceProbability.CheckClock)
    (accessClock : Visit → PiCCSStoredSourceProbability.AccessClock)
    (lowNorm : Phi81StrongSet.LowNormInvertibility)
    (bounds : PiRLC.PaperForkExtractionWork.PrimitiveBounds)
    (bounded : PiRLC.PaperForkExtractionWork.Bounded
      (PaperExtractionAlgebra.extractionAlgebra productionAjtaiKey).ring
      (PiRLCExtractionPrimitives.program scalarSubClock inverseAdapterClock
        assignmentSubClock scalarActionClock) bounds) :
    let continuation := NifsProviderLaw.continuation inputs tapes rawCall checkClock storageClock parentClock
      storageBound storageBounded baseSummable
    let program := PiRLCExtractionPrimitives.program scalarSubClock inverseAdapterClock
      assignmentSubClock scalarActionClock
    let source := HyperNovaGuardedSourceLaw.source originalFirstPhase continuation program
    let visits := fun j : Fin depth => visitedLaw source initial j.val
    let running := fun visit => PiCCSInputCheck.running (inputs visit)
    let fresh := fun visit => PiCCSInputCheck.fresh (inputs visit)
    let firstPhase := guardedPrefix originalFirstPhase
    let checked := InteractiveComposition.firstPhase firstPhase (SupportedExtraction.publicCheck running)
    let contexts := fun j : Fin depth => FiatShamirTransfer.contextLaw relation (realLaw (visits j))
    let provider := fun j : Fin depth =>
      NifsProviderLaw.supportedProvider inputs tapes rawCall checkClock storageClock parentClock
        storageBound storageBounded baseSummable (contexts j) checked
    let extended := fun j : Fin depth =>
      SupportedContinuation.extension relation productionAjtaiKey running fresh (contexts j) checked
        abortTape (provider j)
    (∀ j : Fin depth,
      FiatShamirTransfer.FiatShamirModel relation productionAjtaiKey running fresh
        (realLaw (visits j)) firstPhase abortTape (provider j) g deltaFS (queries j)) →
    ∀ j : Fin depth,
      (((visits j).bind (guardedDraw source)).toOuterMeasure
        {draw | MarkedSourceFailure draw}).toReal ≤
      ((visits j).toOuterMeasure {visit | goodActive visit}).toReal -
        g (queries j) ((visits j).toOuterMeasure {visit | goodActive visit}).toReal + deltaFS (queries j) +
        InteractiveComposition.weakLoss relation productionAjtaiKey +
        Real.sqrt (BindingProbability.successProbability productionAjtaiKey program relation running fresh
          firstPhase (SupportedExtraction.publicCheck running) (extended j)
          (fun visit => (PiCCSStoredSourceProbability.sourceProgram (inputs visit)
            (sourceCheckClock visit) (accessClock visit)).access) (contexts j) * PaperProfile.arity.total +
              IndependentExecution.testError productionShape 9) := by
  dsimp only
  intro models j
  let continuation := NifsProviderLaw.continuation inputs tapes rawCall checkClock storageClock parentClock
    storageBound storageBounded baseSummable
  let program := PiRLCExtractionPrimitives.program scalarSubClock inverseAdapterClock
    assignmentSubClock scalarActionClock
  let source := HyperNovaGuardedSourceLaw.source originalFirstPhase continuation program
  let visits := visitedLaw source initial j.val
  have extracted := NifsProviderLaw.source_probability_bound inputs tapes rawCall checkClock storageClock
    parentClock storageBound storageBounded baseSummable (realLaw visits) (guardedPrefix originalFirstPhase)
    abortTape g deltaFS (queries j) (models j) scalarSubClock inverseAdapterClock
    assignmentSubClock scalarActionClock sourceCheckClock accessClock lowNorm bounds bounded
  dsimp only at extracted
  rw [HyperNovaVisitedAcceptance.realSuccessProbability_eq_goodActive] at extracted
  simp only [HyperNovaGuardedSourceLaw.realLaw_context_marginal] at extracted ⊢
  have sameLaw : HyperNovaSourceLaw.law inputs visits (guardedPrefix originalFirstPhase) continuation program =
      visits.bind (guardedDraw source) :=
    HyperNovaGuardedSourceLaw.law_eq_guardedDraw originalFirstPhase continuation program visits
  rw [sameLaw] at extracted
  have partition := HyperNovaGuardedSourceLaw.first_source_failure_mass_eq
    originalFirstPhase continuation program visits
  rw [HyperNovaGuardedSourceLaw.law_eq_guardedDraw] at partition
  change ((visits.bind (guardedDraw source)).toOuterMeasure
      {draw | MarkedSourceFailure draw}).toReal = _ at partition
  rw [partition]
  dsimp only [visits, source, continuation, program] at extracted ⊢
  linarith only [extracted]

/-- Whole-history extraction probability under the exact finite family of
approved guarded FS models. One shared g and deltaFS applies to every visit.
The source law and every MSIS event come from the fixed selected continuation;
the hash terms are the first marked collisions in the same actual reverse run.
The initial counter bound is symbolic. This is a probability statement and
makes no averaged-work, efficient-translation, or native-conformance claim. -/
theorem history_probability_bound
    (initial : PMF (Statement × Envelope)) (depth : Nat)
    (depthBound : ∀ input ∈ initial.support, input.1.iteration ≤ depth)
    (originalFirstPhase : Visit → InteractivePrefix.Prover State productionShape 9)
    (abortTape : Tape) (g : Nat → ℝ → ℝ) (deltaFS : Nat → ℝ) (queries : Fin depth → Nat)
    (scalarSubClock : RingF → RingF → Nat) (inverseAdapterClock : RingF → Nat)
    (assignmentSubClock : PiRLCExtractionPrimitives.Assignment → PiRLCExtractionPrimitives.Assignment → Nat)
    (scalarActionClock : RingF → PiRLCExtractionPrimitives.Assignment → Nat)
    (sourceCheckClock : Visit → PiCCSStoredSourceProbability.CheckClock)
    (accessClock : Visit → PiCCSStoredSourceProbability.AccessClock)
    (lowNorm : Phi81StrongSet.LowNormInvertibility)
    (bounds : PiRLC.PaperForkExtractionWork.PrimitiveBounds)
    (bounded : PiRLC.PaperForkExtractionWork.Bounded
      (PaperExtractionAlgebra.extractionAlgebra productionAjtaiKey).ring
      (PiRLCExtractionPrimitives.program scalarSubClock inverseAdapterClock
        assignmentSubClock scalarActionClock) bounds) :
    let continuation := NifsProviderLaw.continuation inputs tapes rawCall checkClock storageClock parentClock
      storageBound storageBounded baseSummable
    let program := PiRLCExtractionPrimitives.program scalarSubClock inverseAdapterClock
      assignmentSubClock scalarActionClock
    let source := HyperNovaGuardedSourceLaw.source originalFirstPhase continuation program
    let visits := fun j : Fin depth => visitedLaw source initial j.val
    let running := fun visit => PiCCSInputCheck.running (inputs visit)
    let fresh := fun visit => PiCCSInputCheck.fresh (inputs visit)
    let firstPhase := guardedPrefix originalFirstPhase
    let checked := InteractiveComposition.firstPhase firstPhase (SupportedExtraction.publicCheck running)
    let contexts := fun j : Fin depth => FiatShamirTransfer.contextLaw relation (realLaw (visits j))
    let provider := fun j : Fin depth =>
      NifsProviderLaw.supportedProvider inputs tapes rawCall checkClock storageClock parentClock
        storageBound storageBounded baseSummable (contexts j) checked
    let extended := fun j : Fin depth =>
      SupportedContinuation.extension relation productionAjtaiKey running fresh (contexts j) checked
        abortTape (provider j)
    (∀ j : Fin depth,
      FiatShamirTransfer.FiatShamirModel relation productionAjtaiKey running fresh
        (realLaw (visits j)) firstPhase abortTape (provider j) g deltaFS (queries j)) →
    (initial.toOuterMeasure {input |
      PerApplicationTerminal.Holds Poseidon2HashChainV1Package.application
        Poseidon2HashChainV1Package.fits Poseidon2HashChainV1Setup.productionSetup input.1 input.2}).toReal ≤
      ((HyperNovaHistoryLaw.law source initial).toOuterMeasure
        {sample | HyperNovaHistoryProbability.AdviceReturned sample}).toReal +
        ∑ j : Fin depth,
          (((visits j).toOuterMeasure
              {visit | HyperNovaFirstFailure.MarkedHashCollision visit}).toReal +
            (((visits j).toOuterMeasure {visit | goodActive visit}).toReal -
              g (queries j) ((visits j).toOuterMeasure {visit | goodActive visit}).toReal +
              deltaFS (queries j) + InteractiveComposition.weakLoss relation productionAjtaiKey +
              Real.sqrt (BindingProbability.successProbability productionAjtaiKey program relation running fresh
                firstPhase (SupportedExtraction.publicCheck running) (extended j)
                (fun visit => (PiCCSStoredSourceProbability.sourceProgram (inputs visit)
                  (sourceCheckClock visit) (accessClock visit)).access) (contexts j) * PaperProfile.arity.total +
                    IndependentExecution.testError productionShape 9))) := by
  dsimp only
  intro models
  let continuation := NifsProviderLaw.continuation inputs tapes rawCall checkClock storageClock parentClock
    storageBound storageBounded baseSummable
  let program := PiRLCExtractionPrimitives.program scalarSubClock inverseAdapterClock
    assignmentSubClock scalarActionClock
  let source := HyperNovaGuardedSourceLaw.source originalFirstPhase continuation program
  have first := first_failure_real_bound source initial depth depthBound
  have each := source_failure_probability_le tapes rawCall checkClock storageClock parentClock
    storageBound storageBounded baseSummable initial depth originalFirstPhase abortTape g deltaFS queries
    scalarSubClock inverseAdapterClock assignmentSubClock scalarActionClock sourceCheckClock accessClock
    lowNorm bounds bounded models
  apply first.trans
  apply add_le_add_right
  apply Finset.sum_le_sum
  intro j _member
  exact add_le_add_right (each j) _

end NightstreamFPrime.Export.Stage1.HyperNovaVisitedSecurity
