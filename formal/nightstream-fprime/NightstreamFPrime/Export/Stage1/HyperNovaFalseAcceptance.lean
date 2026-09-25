import NightstreamFPrime.Export.Stage1.HyperNovaHistory
import NightstreamFPrime.Export.Stage1.HyperNovaVisitedSecurity

/-!
False acceptance at the selected full-opening terminal boundary means that
the verifier accepts but no advice list has the advertised length and forward
application result. This is exactly the history conclusion in AdviceReturned.
It is distinct from bare NIFS Boolean acceptance and from extractor failure.

The same original mixed terminal law and its actual visited laws are retained.
No conditioning on invalid inputs or new cryptographic premise is used. The
linear bound keeps the guarded FS models, symbolic depth/queries, declared
clock bounds and moments, low-norm invertibility, hash-collision mass and the
actual adaptive MSIS probability explicit. Honest rejection is separate.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.HyperNovaFalseAcceptance

open scoped BigOperators ENNReal
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.Nifs
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.Nifs
open PiRLC.CoordinateForkLaw (Challenge)
open HyperNovaHistory (Statement SourceResult)
open HyperNovaHistoryProbability (Sample Accepted AdviceReturned)
open HyperNovaVisitedLaw (Visit goodActive visitedLaw guardedDraw observedDraw)
open HyperNovaGuardedSourceLaw (inputs realLaw guardedPrefix)
open HyperNovaFirstFailure (MarkedHashCollision MarkedSourceFailure)

variable (target : Wide.Target)

attribute [local instance] Classical.propDecidable

/-- Acceptance with no history meeting the existing exact length and forward
evaluation contract. The initial law may mix valid and invalid statements. -/
def FalseAcceptance (input : Statement × target.Envelope) : Prop :=
  target.Holds input.1 input.2 ∧
    ¬ ∃ advice : List AppWitness,
      advice.length = input.1.iteration ∧
      advice.foldl target.program.step input.1.z0 = input.1.zi

/-- Invalid accepted statements cannot have a returned valid history. This
uses the event definitions only, with no output-soundness premise. -/
theorem falseAcceptance_not_adviceReturned (sample : Sample target)
    (invalid : FalseAcceptance target (sample.1, sample.2.1)) :
    ¬ AdviceReturned target sample := by
  rintro ⟨advice, _unused, _returned, length, evaluated⟩
  exact invalid.2 ⟨advice, length, evaluated⟩

private theorem marked_hash_mass
    (source : Statement → target.Payload → PMF (SourceResult target)) (contexts : PMF (Visit target)) :
    (contexts.bind (guardedDraw target source)).toOuterMeasure
      {draw | MarkedHashCollision target draw.1} =
      contexts.toOuterMeasure {visit | MarkedHashCollision target visit} := by
  have marginal : (contexts.bind (guardedDraw target source)).map Prod.fst = contexts := by
    rw [PMF.map_bind]
    have each (visit : Visit target) : (guardedDraw target source visit).map Prod.fst = PMF.pure visit := by
      by_cases good : goodActive target visit
      · rw [guardedDraw, if_pos good, PMF.map_comp]
        exact PMF.map_const _ _
      · rw [guardedDraw, if_neg good, PMF.pure_map]
    simp_rw [each]
    exact PMF.bind_pure _
  calc
    _ = ((contexts.bind (guardedDraw target source)).map Prod.fst).toOuterMeasure
        {visit | MarkedHashCollision target visit} := (PMF.toOuterMeasure_map_apply _ _ _).symm
    _ = _ := congrArg
      (fun distribution => distribution.toOuterMeasure {visit | MarkedHashCollision target visit})
      marginal

private theorem false_acceptance_mass_le_first_failures
    (source : Statement → target.Payload → PMF (SourceResult target))
    (initial : PMF (Statement × target.Envelope)) (depth : Nat)
    (bounded : ∀ input ∈ initial.support, input.1.iteration ≤ depth) :
    initial.toOuterMeasure {input | FalseAcceptance target input} ≤
      ∑ j : Fin depth,
        ((visitedLaw target source initial j.val).toOuterMeasure {visit | MarkedHashCollision target visit} +
          (((visitedLaw target source initial j.val).bind (guardedDraw target source)).toOuterMeasure
            {draw | MarkedSourceFailure target draw})) := by
  let distribution := HyperNovaHistoryLaw.law target source initial
  let event (j : Fin depth) : Set (Sample target) :=
    {sample | MarkedHashCollision target (observedDraw target j.val sample).1 ∨
      MarkedSourceFailure target (observedDraw target j.val sample)}
  have inclusion : {sample | FalseAcceptance target (sample.1, sample.2.1)} ∩
      distribution.support ⊆ ⋃ j : Fin depth, event j := by
    rintro sample ⟨invalid, supported⟩
    have supportedInitial : (sample.1, sample.2.1) ∈ initial.support := by
      rw [← HyperNovaHistoryLaw.initial_marginal target source initial]
      exact (PMF.mem_support_map_iff _ _ _).mpr ⟨sample, supported, rfl⟩
    rcases (HyperNovaFirstFailure.accepted_failure_exists_first target) depth sample
      (bounded (sample.1, sample.2.1) supportedInitial) invalid.1
      (falseAcceptance_not_adviceReturned target sample invalid) with ⟨j, below, failure⟩
    exact Set.mem_iUnion.mpr ⟨⟨j, below⟩, failure⟩
  have initialMass : initial.toOuterMeasure {input | FalseAcceptance target input} =
      distribution.toOuterMeasure {sample | FalseAcceptance target (sample.1, sample.2.1)} := by
    rw [← HyperNovaHistoryLaw.initial_marginal target source initial, PMF.toOuterMeasure_map_apply]
    rfl
  rw [initialMass]
  calc
    _ ≤ distribution.toOuterMeasure (⋃ j : Fin depth, event j) :=
      distribution.toOuterMeasure_mono inclusion
    _ ≤ ∑ j : Fin depth, distribution.toOuterMeasure (event j) :=
      MeasureTheory.measure_iUnion_fintype_le _ _
    _ ≤ _ := by
      apply Finset.sum_le_sum
      intro j _member
      have eventMass : distribution.toOuterMeasure (event j) =
          (((visitedLaw target source initial j.val).bind (guardedDraw target source)).toOuterMeasure
            {draw | MarkedHashCollision target draw.1 ∨ MarkedSourceFailure target draw}) := by
        rw [← HyperNovaVisitedLaw.visitedDraw_marginal target source initial j.val,
          PMF.toOuterMeasure_map_apply]
        rfl
      rw [eventMass]
      have unionBound := MeasureTheory.measure_union_le
        (μ := ((visitedLaw target source initial j.val).bind (guardedDraw target source)).toOuterMeasure)
        {draw | MarkedHashCollision target draw.1} {draw | MarkedSourceFailure target draw}
      rw [marked_hash_mass] at unionBound
      exact unionBound

private theorem event_ne_top {Sample : Type*} (distribution : PMF Sample) (event : Set Sample) :
    distribution.toOuterMeasure event ≠ ∞ := by
  rw [PMF.toOuterMeasure_apply]
  exact distribution.tsum_coe_indicator_ne_top event

/-- The actual false-acceptance mass is bounded by the first marked failures
under the same mixed law. Only the advertised iteration bound is assumed;
all finite-measure facts follow from the PMFs. No valid-source premise is used. -/
theorem probability_le_first_failures
    (source : Statement → target.Payload → PMF (SourceResult target))
    (initial : PMF (Statement × target.Envelope)) (depth : Nat)
    (bounded : ∀ input ∈ initial.support, input.1.iteration ≤ depth) :
    (initial.toOuterMeasure {input | FalseAcceptance target input}).toReal ≤
      ∑ j : Fin depth,
        (((visitedLaw target source initial j.val).toOuterMeasure
            {visit | MarkedHashCollision target visit}).toReal +
          (((visitedLaw target source initial j.val).bind (guardedDraw target source)).toOuterMeasure
            {draw | MarkedSourceFailure target draw}).toReal) := by
  have bound := false_acceptance_mass_le_first_failures target source initial depth bounded
  have finiteTerm (j : Fin depth) :
      (visitedLaw target source initial j.val).toOuterMeasure {visit | MarkedHashCollision target visit} +
        ((visitedLaw target source initial j.val).bind (guardedDraw target source)).toOuterMeasure
          {draw | MarkedSourceFailure target draw} ≠ ∞ :=
    ENNReal.add_ne_top.mpr ⟨event_ne_top _ _, event_ne_top _ _⟩
  have finiteSum := ENNReal.sum_ne_top.mpr (fun j (_ : j ∈ Finset.univ) => finiteTerm j)
  have realBound := ENNReal.toReal_mono finiteSum bound
  rw [ENNReal.toReal_sum (fun j (_ : j ∈ Finset.univ) => finiteTerm j)] at realBound
  have realTerm (j : Fin depth) := ENNReal.toReal_add
    (event_ne_top (visitedLaw target source initial j.val) {visit | MarkedHashCollision target visit})
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

/-- Symbolic false-acceptance bound for the selected terminal verifier, on
the original mixed law. Each actual visit contributes its marked hash mass,
good-active mass minus the shared FS transfer, FS error, weak reduction loss,
independent verifier test error and the actual adaptive MSIS probability.

The hypotheses are the support depth bound; the exact guarded FS models with
shared g/deltaFS and symbolic query counts; the existing raw-call/tape laws,
storage bound and base-work moments; low-norm invertibility; and the declared
primitive bounds. No invalid-source support, conditional experiment, honest
sampler success, output-soundness premise or numerical advantage is supplied.
Fixed-seed hardness, useful transfer bounds and total-query applicability
remain external. This is not a bound on bare NIFS Boolean acceptance. -/
theorem probability_linear_bound
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
    (initial.toOuterMeasure {input | FalseAcceptance target input}).toReal ≤
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
  have first := probability_le_first_failures target source initial depth depthBound
  have each := HyperNovaVisitedSecurity.source_failure_probability_linear_le target tapes rawCall
      checkClock storageClock parentClock
    storageBound storageBounded baseSummable initial depth originalFirstPhase abortTape g deltaFS queries
    scalarSubClock inverseAdapterClock assignmentSubClock scalarActionClock sourceCheckClock accessClock
    lowNorm bounds bounded models
  apply first.trans
  apply Finset.sum_le_sum
  intro j _member
  exact add_le_add (le_refl _) (each j)

end NightstreamFPrime.Export.Stage1.HyperNovaFalseAcceptance
