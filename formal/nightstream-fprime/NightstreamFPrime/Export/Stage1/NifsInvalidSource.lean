import NightstreamFPrime.Export.Stage1.SecurityInstance
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

variable (inst : SecurityInstance)

/-- No full witness satisfies the selected fresh CCS and running CE source
relations, with their existing commitment and public-input semantics. -/
def SourceInvalid (input : PiCCSInputCheck.Input) : Prop :=
  ¬ ∃ witness : PiCCSStoredSourceProbability.FunctionalWitness inst,
    SourceHolds extensionOps K.embed
      (openingMaps (PiCCSStoredWitnessCheck.commit inst)) productionGlobalParams
      (PiCCSStoredWitnessCheck.statement inst input) witness

private theorem not_sourceReturned (input : PiCCSInputCheck.Input)
    (invalid : SourceInvalid inst input)
    (result : Option (WitnessProjection.SourceWitness productionShape
      (PiCCSStoredWitnessCheck.carrier inst))) :
    ¬ SourceReturned (PiCCSStoredWitnessCheck.commit inst) productionGlobalParams
      (PiCCSStoredWitnessCheck.statement inst input) result := by
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
        Option (PiCCSStoredSourceProbability.FunctionalWitness inst))
    (invalid : ∀ context, 0 < (contexts context).toReal → SourceInvalid inst (inputs context)) :
    PaperCompositionProbability.eventProbability contexts firstPhase suffixLaw consume
      (fun context outcome => SourceReturned (PiCCSStoredWitnessCheck.commit inst) productionGlobalParams
        (PiCCSStoredWitnessCheck.statement inst (inputs context))
        (PiCCSStoredWitnessCheck.finishValue inst (inputs context)
          (PiCCSStoredSourceProbability.storeOutcome inst outcome))) = 0 := by
  unfold PaperCompositionProbability.eventProbability
  refine (tsum_congr fun context => ?_).trans tsum_zero
  by_cases positive : 0 < (contexts context).toReal
  · have indicatorZero :
        (fun outcome : Outcome productionShape (PiCCSStoredWitnessCheck.carrier inst) =>
          if SourceReturned (PiCCSStoredWitnessCheck.commit inst) productionGlobalParams
            (PiCCSStoredWitnessCheck.statement inst (inputs context))
            (PiCCSStoredWitnessCheck.finishValue inst (inputs context)
              (PiCCSStoredSourceProbability.storeOutcome inst outcome)) then (1 : ℝ) else 0) =
          fun _ => 0 := by
      funext outcome
      exact if_neg (not_sourceReturned inst (inputs context) (invalid context positive) _)
    rw [indicatorZero, sequentialMean_zero, StrongProbability.verifierMean_const, mul_zero]
  · have zero : (contexts context).toReal = 0 :=
      le_antisymm (le_of_not_gt positive) ENNReal.toReal_nonneg
    rw [zero, zero_mul]

variable {Context State Tape : Type*}
  (inputs : Context → PiCCSInputCheck.Input)
  [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key inst.relation inst.ajtai).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key inst.relation inst.ajtai).piRlcAlgebra)]
  (law : PMF (Context × Option (FiatShamirTransfer.RealOutput inst.relation)))
  (originalFirstPhase : Context → InteractivePrefix.Prover State productionShape 9)
  (abortTape : Tape)
  (provider : SupportedContinuation.Provider Tape inst.relation inst.ajtai
    (fun context => inst.running (inputs context))
    (fun context => inst.fresh (inputs context))
    (FiatShamirTransfer.contextLaw inst.relation law)
    (InteractiveComposition.firstPhase originalFirstPhase
      (SupportedExtraction.publicCheck (fun context => inst.running (inputs context)))))
  (g : Nat → ℝ → ℝ) (deltaFS : Nat → ℝ) (Q : Nat)
  (model : WideFiatShamir.FiatShamirModel inst.relation inst.ajtai
    (fun context => inst.running (inputs context))
    (fun context => inst.fresh (inputs context))
    law originalFirstPhase abortTape provider g deltaFS Q)
  (scalarSubClock : RingF → RingF → Nat)
  (inverseAdapterClock : RingF → Nat)
  (assignmentSubClock : PiRLCExtractionPrimitives.Assignment inst →
      PiRLCExtractionPrimitives.Assignment inst → Nat)
  (scalarActionClock : RingF → PiRLCExtractionPrimitives.Assignment inst → Nat)
  (checkClock : Context → PiCCSStoredSourceProbability.CheckClock inst)
  (accessClock : Context → PiCCSStoredSourceProbability.AccessClock inst)
  (lowNorm : Phi81StrongSet.LowNormInvertibility)
  (bounds : PrimitiveBounds)
  (bounded : Bounded (PaperExtractionAlgebra.extractionAlgebra inst.ajtai).ring
    (PiRLCExtractionPrimitives.program inst scalarSubClock inverseAdapterClock
      assignmentSubClock scalarActionClock) bounds)

include model lowNorm bounded in
/-- On an input law supported on invalid source instances, the existing
conditional knowledge reduction bounds actual witness-bearing NIFS success.
The factor seventeen is the selected arity, not a query or use budget. -/
theorem real_success_bound_of_invalid_source
    (invalid : ∀ context, 0 < (FiatShamirTransfer.contextLaw inst.relation law context).toReal →
      (SourceInvalid inst) (inputs context))
    (epsilonMSIS : ℝ)
    (msisBound :
      let continuation := SupportedContinuation.extension inst.relation inst.ajtai
        (fun context => inst.running (inputs context))
        (fun context => inst.fresh (inputs context))
        (FiatShamirTransfer.contextLaw inst.relation law)
        (InteractiveComposition.firstPhase originalFirstPhase
          (SupportedExtraction.publicCheck (fun context => inst.running (inputs context))))
        abortTape provider
      BindingProbability.successProbability inst.ajtai
        (PiRLCExtractionPrimitives.program inst scalarSubClock inverseAdapterClock
          assignmentSubClock scalarActionClock) inst.relation
        (fun context => inst.running (inputs context))
        (fun context => inst.fresh (inputs context))
        originalFirstPhase (SupportedExtraction.publicCheck
          (fun context => inst.running (inputs context))) continuation
        (fun context => (PiCCSStoredSourceProbability.sourceProgram inst
          (inputs context) (checkClock context) (accessClock context)).access)
        (FiatShamirTransfer.contextLaw inst.relation law) ≤ epsilonMSIS) :
    g Q (WideFiatShamir.realSuccessProbability inst.relation inst.ajtai
      (fun context => inst.running (inputs context))
      (fun context => inst.fresh (inputs context)) law) ≤
        deltaFS Q + InteractiveComposition.weakLoss inst.relation inst.ajtai +
          Real.sqrt (epsilonMSIS * 17 + IndependentExecution.testError productionShape 9) := by
  have extracted := FiatShamirTransfer.returned_source_bound_of_msis
    inst.relation inst.ajtai
    (fun context => inst.running (inputs context))
    (fun context => inst.fresh (inputs context))
    law originalFirstPhase abortTape provider g deltaFS Q
    (WideFiatShamir.realSuccessProbability inst.relation inst.ajtai
      (fun context => inst.running (inputs context)) (fun context => inst.fresh (inputs context)) law)
    model.successTransfer
    (PiRLCExtractionPrimitives.program inst scalarSubClock inverseAdapterClock
      assignmentSubClock scalarActionClock)
    (fun context => PiCCSStoredSourceProbability.sourceProgram inst
      (inputs context) (checkClock context) (accessClock context))
    lowNorm
    (PiRLCExtractionPrimitives.program_correct inst scalarSubClock inverseAdapterClock
      assignmentSubClock scalarActionClock)
    bounds bounded
    (fun context => PiCCSStoredSourceProbability.sourceProgram_correct inst
      (inputs context) (checkClock context) (accessClock context)) epsilonMSIS msisBound
  dsimp only at extracted
  rw [PiCCSStoredSourceProbability.returnedSourceProbability_eq_finishValue] at extracted
  have returnedZero := source_event_probability_eq_zero inst inputs
    (FiatShamirTransfer.contextLaw inst.relation law)
    (InteractiveComposition.firstPhase originalFirstPhase
      (SupportedExtraction.publicCheck (fun context => inst.running (inputs context))))
    (InteractiveComposition.suffixLaw inst.relation inst.ajtai
      (fun context => inst.running (inputs context))
      (fun context => inst.fresh (inputs context))
      (SupportedContinuation.extension inst.relation inst.ajtai
        (fun context => inst.running (inputs context))
        (fun context => inst.fresh (inputs context))
        (FiatShamirTransfer.contextLaw inst.relation law)
        (InteractiveComposition.firstPhase originalFirstPhase
          (SupportedExtraction.publicCheck (fun context => inst.running (inputs context))))
        abortTape provider))
    (InteractiveComposition.consume inst.relation inst.ajtai
      (PiRLCExtractionPrimitives.program inst scalarSubClock inverseAdapterClock
        assignmentSubClock scalarActionClock)) invalid
  have errorBound := extracted.trans returnedZero.le
  simp only [PaperProfile.arity_total, Nat.cast_ofNat] at errorBound
  linarith only [errorBound]

end NightstreamFPrime.Export.Stage1.NifsInvalidSource
