import NightstreamFPrime.Export.Stage1.NifsFiatShamir

/-!
The selected NIFS knowledge bound when every positive-mass input has no
source witness. The actual verifier-success event includes witnesses for
its exact sixteen children. Source return then has probability zero.

The FS transfer, supported continuation, low-norm property, primitive bounds
and exact MSIS success bound remain explicit. No conditioning on semantic
invalidity, adaptive union bound, model instance or clock refinement is used.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.NifsInvalidSource

open scoped BigOperators
attribute [local instance] Classical.propDecidable

open _root_.NightstreamFPrime.Spec
open _root_.NightstreamFPrime.Spec.Folding
open _root_.NightstreamFPrime.Spec.Folding.Nifs
open _root_.NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction ConcreteCarrier CheckedWitnessExtraction
open _root_.NightstreamFPrime.Lifecycle
open _root_.NightstreamFPrime.Lifecycle.Nifs
open PiRLC.PaperForkExtraction PiRLC.PaperForkExtractionWork
open PiRLC.CoordinateForkLaw
open Poseidon2HashChainV1Setup (productionAjtaiKey)
open PiDECInputCheck (relation)

/-- No full witness satisfies the selected fresh CCS and running CE source
relations, with their existing commitment and public-input semantics. -/
def SourceInvalid (input : PiCCSInputCheck.Input) : Prop :=
  ¬ ∃ witness : PiCCSStoredSourceProbability.FunctionalWitness,
    SourceHolds extensionOps K.embed
      (openingMaps PiCCSStoredWitnessCheck.commit) productionGlobalParams
      (PiCCSStoredWitnessCheck.statement input) witness

private theorem not_sourceReturned (input : PiCCSInputCheck.Input)
    (invalid : SourceInvalid input)
    (result : Option (WitnessProjection.SourceWitness productionShape
      PiCCSStoredWitnessCheck.carrier)) :
    ¬ SourceReturned PiCCSStoredWitnessCheck.commit productionGlobalParams
      (PiCCSStoredWitnessCheck.statement input) result := by
  rintro ⟨values, _returned, valid⟩
  exact invalid ⟨_, valid⟩

private theorem sequentialMean_zero {State Endpoint : Type*} [Fintype Endpoint]
    {shape : Shape} {columns width : Nat}
    (firstPhase : InteractivePrefix.Prover State shape width)
    (suffixLaw : PublicCoins K shape → FullOutputCoordinates.FullOutput K shape →
      State → PMF Endpoint)
    (consume : PublicCoins K shape → FullOutputCoordinates.FullOutput K shape →
      State → Endpoint → Option (OutputWitness shape columns)) :
    InteractiveDistribution.sequentialMean firstPhase suffixLaw consume (fun _ => 0) =
      fun _ _ _ => 0 := by
  funext alpha gamma point
  unfold InteractiveDistribution.sequentialMean
  split <;> simp only [mul_zero, Finset.sum_const_zero]

/-- The exact selected stored source-return event has zero probability.
Zero-mass contexts need no invalidity premise; all aborts remain in the law. -/
theorem source_event_probability_eq_zero {Context State Endpoint : Type*}
    [Fintype Endpoint]
    (inputs : Context → PiCCSInputCheck.Input) (contexts : PMF Context)
    (firstPhase : Context → InteractivePrefix.Prover State productionShape 9)
    (suffixLaw : Context → PublicCoins K productionShape →
      FullOutputCoordinates.FullOutput K productionShape → State → PMF Endpoint)
    (consume : Context → PublicCoins K productionShape →
      FullOutputCoordinates.FullOutput K productionShape → State → Endpoint →
        Option PiCCSStoredSourceProbability.FunctionalWitness)
    (invalid : ∀ context, 0 < (contexts context).toReal → SourceInvalid (inputs context)) :
    PaperCompositionProbability.eventProbability contexts firstPhase suffixLaw consume
      (fun context outcome => SourceReturned PiCCSStoredWitnessCheck.commit productionGlobalParams
        (PiCCSStoredWitnessCheck.statement (inputs context))
        (PiCCSStoredWitnessCheck.finishValue (inputs context)
          (PiCCSStoredSourceProbability.storeOutcome outcome))) = 0 := by
  unfold PaperCompositionProbability.eventProbability
  refine (tsum_congr fun context => ?_).trans tsum_zero
  by_cases positive : 0 < (contexts context).toReal
  · have indicatorZero :
        (fun outcome : Outcome productionShape PiCCSStoredWitnessCheck.carrier =>
          if SourceReturned PiCCSStoredWitnessCheck.commit productionGlobalParams
            (PiCCSStoredWitnessCheck.statement (inputs context))
            (PiCCSStoredWitnessCheck.finishValue (inputs context)
              (PiCCSStoredSourceProbability.storeOutcome outcome)) then (1 : ℝ) else 0) =
          fun _ => 0 := by
      funext outcome
      exact if_neg (not_sourceReturned (inputs context) (invalid context positive) _)
    rw [indicatorZero, sequentialMean_zero, StrongProbability.verifierMean_const, mul_zero]
  · have zero : (contexts context).toReal = 0 :=
      le_antisymm (le_of_not_gt positive) ENNReal.toReal_nonneg
    rw [zero, zero_mul]

variable {Context State Tape : Type*}
  (inputs : Context → PiCCSInputCheck.Input)
  [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key relation productionAjtaiKey).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation productionAjtaiKey).piRlcAlgebra)]
  (law : PMF (Context × Option (FiatShamirTransfer.RealOutput relation)))
  (originalFirstPhase : Context → InteractivePrefix.Prover State productionShape 9)
  (abortTape : Tape)
  (provider : SupportedContinuation.Provider Tape relation productionAjtaiKey
    (fun context => PiCCSInputCheck.running (inputs context))
    (fun context => PiCCSInputCheck.fresh (inputs context))
    (FiatShamirTransfer.contextLaw relation law)
    (InteractiveComposition.firstPhase originalFirstPhase
      (SupportedExtraction.publicCheck (fun context => PiCCSInputCheck.running (inputs context)))))
  (g : Nat → ℝ → ℝ) (deltaFS : Nat → ℝ) (Q : Nat)
  (model : FiatShamirTransfer.FiatShamirModel relation productionAjtaiKey
    (fun context => PiCCSInputCheck.running (inputs context))
    (fun context => PiCCSInputCheck.fresh (inputs context))
    law originalFirstPhase abortTape provider g deltaFS Q)
  (scalarSubClock : RingF → RingF → Nat)
  (inverseAdapterClock : RingF → Nat)
  (assignmentSubClock : PiRLCExtractionPrimitives.Assignment → PiRLCExtractionPrimitives.Assignment → Nat)
  (scalarActionClock : RingF → PiRLCExtractionPrimitives.Assignment → Nat)
  (checkClock : Context → PiCCSStoredSourceProbability.CheckClock)
  (accessClock : Context → PiCCSStoredSourceProbability.AccessClock)
  (lowNorm : Phi81StrongSet.LowNormInvertibility)
  (bounds : PrimitiveBounds)
  (bounded : Bounded (PaperExtractionAlgebra.extractionAlgebra productionAjtaiKey).ring
    (PiRLCExtractionPrimitives.program scalarSubClock inverseAdapterClock
      assignmentSubClock scalarActionClock) bounds)

include model lowNorm bounded in
/-- On an input law supported on invalid source instances, the existing
conditional knowledge reduction bounds actual witness-bearing NIFS success.
The factor seventeen is the selected arity, not a query or use budget. -/
theorem real_success_bound_of_invalid_source
    (invalid : ∀ context, 0 < (FiatShamirTransfer.contextLaw relation law context).toReal →
      SourceInvalid (inputs context))
    (epsilonMSIS : ℝ)
    (msisBound :
      let continuation := SupportedContinuation.extension relation productionAjtaiKey
        (fun context => PiCCSInputCheck.running (inputs context))
        (fun context => PiCCSInputCheck.fresh (inputs context))
        (FiatShamirTransfer.contextLaw relation law)
        (InteractiveComposition.firstPhase originalFirstPhase
          (SupportedExtraction.publicCheck (fun context => PiCCSInputCheck.running (inputs context))))
        abortTape provider
      BindingProbability.successProbability productionAjtaiKey
        (PiRLCExtractionPrimitives.program scalarSubClock inverseAdapterClock
          assignmentSubClock scalarActionClock) relation
        (fun context => PiCCSInputCheck.running (inputs context))
        (fun context => PiCCSInputCheck.fresh (inputs context))
        originalFirstPhase (SupportedExtraction.publicCheck
          (fun context => PiCCSInputCheck.running (inputs context))) continuation
        (fun context => (PiCCSStoredSourceProbability.sourceProgram
          (inputs context) (checkClock context) (accessClock context)).access)
        (FiatShamirTransfer.contextLaw relation law) ≤ epsilonMSIS) :
    g Q (FiatShamirTransfer.realSuccessProbability relation productionAjtaiKey
      (fun context => PiCCSInputCheck.running (inputs context))
      (fun context => PiCCSInputCheck.fresh (inputs context)) law) ≤
        deltaFS Q + InteractiveComposition.weakLoss relation productionAjtaiKey +
          Real.sqrt (epsilonMSIS * 17 + IndependentExecution.testError productionShape 9) := by
  have extracted := FiatShamirTransfer.returned_source_bound_of_msis
    relation productionAjtaiKey
    (fun context => PiCCSInputCheck.running (inputs context))
    (fun context => PiCCSInputCheck.fresh (inputs context))
    law originalFirstPhase abortTape provider g deltaFS Q model
    (PiRLCExtractionPrimitives.program scalarSubClock inverseAdapterClock
      assignmentSubClock scalarActionClock)
    (fun context => PiCCSStoredSourceProbability.sourceProgram
      (inputs context) (checkClock context) (accessClock context))
    lowNorm
    (PiRLCExtractionPrimitives.program_correct scalarSubClock inverseAdapterClock
      assignmentSubClock scalarActionClock)
    bounds bounded
    (fun context => PiCCSStoredSourceProbability.sourceProgram_correct
      (inputs context) (checkClock context) (accessClock context)) epsilonMSIS msisBound
  dsimp only at extracted
  rw [PiCCSStoredSourceProbability.returnedSourceProbability_eq_finishValue] at extracted
  have returnedZero := source_event_probability_eq_zero inputs
    (FiatShamirTransfer.contextLaw relation law)
    (InteractiveComposition.firstPhase originalFirstPhase
      (SupportedExtraction.publicCheck (fun context => PiCCSInputCheck.running (inputs context))))
    (InteractiveComposition.suffixLaw relation productionAjtaiKey
      (fun context => PiCCSInputCheck.running (inputs context))
      (fun context => PiCCSInputCheck.fresh (inputs context))
      (SupportedContinuation.extension relation productionAjtaiKey
        (fun context => PiCCSInputCheck.running (inputs context))
        (fun context => PiCCSInputCheck.fresh (inputs context))
        (FiatShamirTransfer.contextLaw relation law)
        (InteractiveComposition.firstPhase originalFirstPhase
          (SupportedExtraction.publicCheck (fun context => PiCCSInputCheck.running (inputs context))))
        abortTape provider))
    (InteractiveComposition.consume relation productionAjtaiKey
      (PiRLCExtractionPrimitives.program scalarSubClock inverseAdapterClock
        assignmentSubClock scalarActionClock)) invalid
  have errorBound := extracted.trans returnedZero.le
  simp only [PaperProfile.arity_total, Nat.cast_ofNat] at errorBound
  linarith only [errorBound]

end NightstreamFPrime.Export.Stage1.NifsInvalidSource
