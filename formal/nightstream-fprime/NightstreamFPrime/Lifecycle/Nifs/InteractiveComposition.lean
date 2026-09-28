import NightstreamFPrime.Lifecycle.Nifs.WeakExtraction
import NightstreamFPrime.Lifecycle.PaperExtractionAlgebra
import NightstreamFPrime.Spec.Folding.Nifs.PaperCompositionProbability

/-!
Interactive NIFS extraction for the selected Nightstream key. The public
PiCCS check runs before the captured PiRLC/PiDEC continuation. Each weak
query obtains its own final child witnesses. The exact returned endpoint
law feeds the causal strong extractor, with both probability losses kept.
Fiat–Shamir transfer and the probability of the binding event are separate.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.InteractiveComposition

open scoped BigOperators
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction ConcreteCarrier
open _root_.NightstreamFPrime.Spec.Folding.Nifs
open _root_.NightstreamFPrime.Lifecycle.PaperAlgebra
open PiRLC.PaperForkExtraction PiRLC.PaperForkExtractionWork
open PiRLC.CoordinateForkLaw

attribute [local instance] Classical.propDecidable

variable {Context State Tape : Type*} {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (running : Context → Lifecycle.Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  (fresh : Context → Lifecycle.Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
  [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]

abbrev Endpoint := PaperWeakLaw.Endpoint
  (Fin (ProductionKey.key relation ajtai).arity.total)
  (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)
  (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits))

variable
  (originalFirstPhase : Context → InteractivePrefix.Prover State productionShape 9)
  (publicCheck : Context → Probe K productionShape → Bool)
  (continuation : ∀ context (coins : PublicCoins K productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape), State →
      WeakExtraction.Continuation Tape relation ajtai (running context) (fresh context) coins output)
  (program : Primitives RingF
    (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)))

def firstPhase (context : Context) :=
  InteractivePrefix.checked (originalFirstPhase context) (publicCheck context)

noncomputable def suffixLaw (context : Context) (coins : PublicCoins K productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape) (state : State) :
    PMF (Endpoint relation ajtai) :=
  WeakExtraction.endpointLaw relation ajtai (running context) (fresh context)
    (continuation context coins output state)

noncomputable def consume (_context : Context) (_coins : PublicCoins K productionShape)
    (_output : FullOutputCoordinates.FullOutput K productionShape) (_state : State)
    (endpoint : Endpoint relation ajtai) :=
  WeakExtraction.consume relation ajtai program endpoint

/-- Original NIFS success: a checked C receipt followed by the original
continuation's accepted and valid final-output event. -/
noncomputable def originalSuccess (context : Context)
    (alpha : CubePoint K productionShape.cubeVariables) (gamma : K)
    (point : CubePoint K productionShape.cubeVariables) : ℝ :=
  match InteractivePrefix.run (firstPhase originalFirstPhase publicCheck context) alpha gamma point with
  | none => 0
  | some receipt =>
      (continuation context receipt.1.coins receipt.1.response.fullOutput receipt.2).successProbability

noncomputable def weakLoss : ℝ :=
  ((ProductionKey.key relation ajtai).arity.total : ℝ) /
    Fintype.card (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)

variable
  (checkCorrect : ∀ context probe, publicCheck context probe = true ↔
    probe.FixedWidthAccepted extensionOps K.embed
      ((ProductionKey.key relation ajtai).statement (running context) (fresh context)) 9)
  (laws : ExtractionAlgebra (ProductionKey.key relation ajtai).piRlcSemantics
    (ProductionKey.key relation ajtai).params (ProductionKey.key relation ajtai).piRlcAlgebra)
  (strongSet : StrongSetUnits laws.ring (ProductionKey.key relation ajtai).piRlcAlgebra.challengeValid)
  (correct : Correct laws.ring laws.assignmentModule program)
  (bounds : PrimitiveBounds) (bounded : Bounded laws.ring program bounds)

include checkCorrect strongSet correct bounded in
theorem local_weak_bound (context : Context)
    (alpha : CubePoint K productionShape.cubeVariables) (gamma : K)
    (point : CubePoint K productionShape.cubeVariables) :
    originalSuccess relation ajtai running fresh originalFirstPhase publicCheck continuation
      context alpha gamma point - weakLoss relation ajtai ≤
    InteractiveDistribution.sequentialMean
      (firstPhase originalFirstPhase publicCheck context)
      (suffixLaw relation ajtai running fresh continuation context)
      (consume relation ajtai program context)
      (fun outcome => if StrongProbability.RelaxedSuccess (width := 9)
        (PaperAlgebra.openingMaps ajtai) productionGlobalParams
        ((ProductionKey.key relation ajtai).statement (running context) (fresh context)) outcome
        then (1 : ℝ) else 0) alpha gamma point := by
  unfold originalSuccess InteractiveDistribution.sequentialMean
  cases returned : InteractivePrefix.run
      (firstPhase originalFirstPhase publicCheck context) alpha gamma point with
  | none =>
      have nonnegative : (0 : ℝ) ≤ weakLoss relation ajtai :=
        div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
      simpa [StrongProbability.RelaxedSuccess] using (sub_nonpos.mpr nonnegative)
  | some receipt =>
      have checked := (InteractivePrefix.checked_returns (originalFirstPhase context)
        (publicCheck context) alpha gamma point receipt returned).2
      have accepted := (checkCorrect context receipt.1).mp checked
      have lower := WeakExtraction.weak_relaxed_success_bound relation ajtai
        (running context) (fresh context)
        laws strongSet program correct bounds bounded receipt.1 accepted
        (continuation context receipt.1.coins receipt.1.response.fullOutput receipt.2)
      dsimp only [suffixLaw, consume, weakLoss]
      exact lower

include checkCorrect strongSet correct bounded in
/-- Selected one-step interactive extraction, with the original final-output
success probability and the actual two-execution disagreement event. -/
theorem source_success_bound (contexts : PMF Context) :
    StrongProbability.clockMean contexts
      (originalSuccess relation ajtai running fresh originalFirstPhase publicCheck continuation) -
      weakLoss relation ajtai -
      Real.sqrt (PaperCompositionProbability.disagreementProbability contexts
        (firstPhase originalFirstPhase publicCheck) (none : Endpoint relation ajtai)
        (suffixLaw relation ajtai running fresh continuation) (consume relation ajtai program)
        (fun _ => PaperAlgebra.openingMaps ajtai) productionGlobalParams
        (fun context => (ProductionKey.key relation ajtai).statement (running context) (fresh context)) +
          IndependentExecution.testError productionShape 9) ≤
      PaperCompositionProbability.sourceProbability contexts
        (firstPhase originalFirstPhase publicCheck)
        (suffixLaw relation ajtai running fresh continuation) (consume relation ajtai program)
        (fun _ => PaperAlgebra.openingMaps ajtai) productionGlobalParams
        (fun context => (ProductionKey.key relation ajtai).statement (running context) (fresh context)) := by
  apply PaperCompositionProbability.source_success_ge_from_weak contexts
    (firstPhase originalFirstPhase publicCheck) (none : Endpoint relation ajtai)
    (suffixLaw relation ajtai running fresh continuation) (consume relation ajtai program)
    (fun _ => PaperAlgebra.openingMaps ajtai) productionGlobalParams
    (fun context => (ProductionKey.key relation ajtai).statement (running context) (fresh context))
    (originalSuccess relation ajtai running fresh originalFirstPhase publicCheck continuation)
    (weakLoss relation ajtai)
  · intro context alpha gamma point
    unfold originalSuccess
    split
    · exact le_rfl
    · exact (WeakExtraction.continuation_success_range relation ajtai
        (running context) (fresh context) _).1
  · exact local_weak_bound relation ajtai running fresh originalFirstPhase publicCheck continuation
      program checkCorrect laws strongSet correct bounds bounded
  · rfl
  · intro context
    exact (ProductionKey.key relation ajtai).constantLaw
  · intro context
    exact (ProductionKey.key relation ajtai).statement_sumcheckDegreeBound_le
      (running context) (fresh context)

include checkCorrect strongSet correct bounded in
/-- The selected NIFS source output satisfies the v1.2 linear bound on the
same checked C receipt and actual weak continuation. The normalized retry
disagreement remains explicit until the stopped binding reduction supplies
its computed success and resource bounds. -/
theorem source_success_retry_bound (contexts : PMF Context) :
    StrongProbability.clockMean contexts
      (originalSuccess relation ajtai running fresh originalFirstPhase publicCheck continuation) -
      weakLoss relation ajtai - IndependentExecution.testError productionShape 9 -
      PaperCompositionProbability.retryDisagreementProbability contexts
        (firstPhase originalFirstPhase publicCheck) (none : Endpoint relation ajtai)
        (suffixLaw relation ajtai running fresh continuation) (consume relation ajtai program)
        (fun _ => PaperAlgebra.openingMaps ajtai) productionGlobalParams
        (fun context => (ProductionKey.key relation ajtai).statement (running context) (fresh context)) ≤
      PaperCompositionProbability.sourceProbability contexts
        (firstPhase originalFirstPhase publicCheck)
        (suffixLaw relation ajtai running fresh continuation) (consume relation ajtai program)
        (fun _ => PaperAlgebra.openingMaps ajtai) productionGlobalParams
        (fun context => (ProductionKey.key relation ajtai).statement (running context) (fresh context)) := by
  apply PaperCompositionProbability.source_success_ge_retry_from_weak contexts
    (firstPhase originalFirstPhase publicCheck) (none : Endpoint relation ajtai)
    (suffixLaw relation ajtai running fresh continuation) (consume relation ajtai program)
    (fun _ => PaperAlgebra.openingMaps ajtai) productionGlobalParams
    (fun context => (ProductionKey.key relation ajtai).statement (running context) (fresh context))
    (originalSuccess relation ajtai running fresh originalFirstPhase publicCheck continuation)
    (weakLoss relation ajtai)
  · intro context alpha gamma point
    unfold originalSuccess
    split
    · exact le_rfl
    · exact (WeakExtraction.continuation_success_range relation ajtai
        (running context) (fresh context) _).1
  · exact local_weak_bound relation ajtai running fresh originalFirstPhase publicCheck continuation
      program checkCorrect laws strongSet correct bounds bounded
  · rfl
  · intro context
    exact (ProductionKey.key relation ajtai).constantLaw
  · intro context
    exact (ProductionKey.key relation ajtai).statement_sumcheckDegreeBound_le
      (running context) (fresh context)

end NightstreamFPrime.Lifecycle.Nifs.InteractiveComposition
