import NightstreamFPrime.Lifecycle.Nifs.InteractiveComposition
import NightstreamFPrime.Lifecycle.Nifs.BindingBridge
import NightstreamFPrime.Spec.Folding.Nifs.PaperCompositionAgreement

/-!
The selected NIFS binding event uses the same checked PiCCS prefix and actual
weak continuation as InteractiveComposition. Two independent executions keep
their literal endpoint returns. The global probability bound retains the
original context law; no numerical hardness bound is inserted here.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.InteractiveAgreement

open scoped BigOperators
attribute [local instance] Classical.propDecidable
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open StrongReduction ConcreteCarrier
open _root_.NightstreamFPrime.Spec.Folding.Nifs
open _root_.NightstreamFPrime.Lifecycle.PaperAlgebra
open PiRLC.PaperForkExtraction PiRLC.PaperForkExtractionWork
open PiRLC.CoordinateForkLaw

variable {Context State Tape : Type*} {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
  (running : Context → Lifecycle.Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  (fresh : Context → Lifecycle.Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
  [DecidableEq RingF]
  [Fintype (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  (originalFirstPhase : Context → InteractivePrefix.Prover State productionShape 9)
  (publicCheck : Context → Probe K productionShape → Bool)
  (continuation : ∀ context (coins : PublicCoins K productionShape)
    (output : FullOutputCoordinates.FullOutput K productionShape), State →
      WeakExtraction.Continuation Tape relation ajtai (running context) (fresh context) coins output)
  (program : Primitives RingF
    (PaperAlgebra.Assignment (logicalWidth := logicalWidth) (publicFits := publicFits)))

/-- Actual receipt/endpoint pairs and their literal terminal lists identify
the binding event; arbitrary setup collisions are not counted. -/
def BindingEvent (context : Context) :
    PaperCompositionAgreement.Observation State (InteractiveComposition.Endpoint relation ajtai) productionShape →
    PaperCompositionAgreement.Observation State (InteractiveComposition.Endpoint relation ajtai) productionShape → Prop :=
  PaperCompositionAgreement.observationBindingEvent (ProductionKey.key relation ajtai)
    (running context) (fresh context) program
    (Phi81Relation.PiRLCAlgebra.Binding.relaxedOps
      (shape := FullShape logicalWidth publicFits) (rows := productionProfile.commitmentWidth))

/-- The event is measured under Main's exact independent pair experiment. -/
noncomputable def localBindingProbability (context : Context) : ℝ :=
  PaperCompositionAgreement.pairMean
    (InteractiveComposition.firstPhase originalFirstPhase publicCheck context)
    (InteractiveComposition.suffixLaw relation ajtai running fresh continuation context)
    (fun left right => if BindingEvent relation ajtai running fresh program context left right then 1 else 0)

/-- The original context law is retained when the two-execution event is averaged. -/
noncomputable def bindingProbability (contexts : PMF Context) : ℝ :=
  ∑' context, (contexts context).toReal *
    localBindingProbability relation ajtai running fresh originalFirstPhase publicCheck continuation program context

private noncomputable def localDisagreement (context : Context) : ℝ :=
  StrongProbability.disagreementProbability
    (InteractiveDistribution.tapes
      (InteractiveComposition.firstPhase originalFirstPhase publicCheck context)
      (none : InteractiveComposition.Endpoint relation ajtai)
      (InteractiveComposition.suffixLaw relation ajtai running fresh continuation context))
    (InteractiveDistribution.coupled
      (InteractiveComposition.firstPhase originalFirstPhase publicCheck context)
      (InteractiveComposition.consume relation ajtai program context))
    (PaperAlgebra.openingMaps ajtai) productionGlobalParams
    ((ProductionKey.key relation ajtai).statement (running context) (fresh context))

theorem localBindingProbability_range (context : Context) :
    0 ≤ localBindingProbability relation ajtai running fresh originalFirstPhase publicCheck continuation program context ∧
      localBindingProbability relation ajtai running fresh originalFirstPhase publicCheck continuation program context ≤ 1 := by
  apply PaperCompositionAgreement.pairMean_range
  intro left right
  split_ifs <;> norm_num

private theorem localDisagreement_nonnegative (context : Context) :
    0 ≤ localDisagreement relation ajtai running fresh originalFirstPhase publicCheck continuation program context := by
  unfold localDisagreement
  rw [PaperCompositionAgreement.disagreementProbability_eq_pairMean]
  exact (PaperCompositionAgreement.pairMean_range _ _ _ (by
    intro left right
    split_ifs <;> norm_num)).1

variable
  (strongSet : StrongSetUnits (PaperExtractionAlgebra.extractionAlgebra ajtai).ring
    (ProductionKey.key relation ajtai).piRlcAlgebra.challengeValid)
  (correct : Correct (PaperExtractionAlgebra.extractionAlgebra ajtai).ring
    (PaperExtractionAlgebra.extractionAlgebra ajtai).assignmentModule program)

include strongSet correct in
/-- Instantiate the checked pair bound with the selected continuation's own
charged oracle and exact parent checker. No per-context agreement bound is supplied. -/
theorem local_disagreement_le_binding (context : Context) :
    localDisagreement relation ajtai running fresh originalFirstPhase publicCheck continuation program context ≤
      localBindingProbability relation ajtai running fresh originalFirstPhase publicCheck continuation program context := by
  let oracle := fun coins output state => (continuation context coins output state).chargedOracle
  let checker := fun coins output state => (continuation context coins output state).parentChecker
  have checkSpec : ∀ (probe : Probe K productionShape) (state : State) response,
      (checker probe.coins probe.response.fullOutput state response).accepted = true ↔
        response.Success (ProductionKey.key relation ajtai).piRlcSemantics
          (ProductionKey.key relation ajtai).params (ProductionKey.key relation ajtai).piRlcAlgebra
          (PaperStrongInterface.piRlcBatchForProbe (ProductionKey.key relation ajtai)
            (running context) (fresh context) probe) := by
    intro probe state response
    rw [← WeakExtraction.batchForOutput_eq_probe relation ajtai (running context) (fresh context) probe]
    exact (continuation context probe.coins probe.response.fullOutput state).parentChecker_spec response
  have bound := PaperCompositionAgreement.disagreementProbability_le_bindingProbability
    (ProductionKey.key relation ajtai) (running context) (fresh context) oracle checker program
    (Phi81Relation.PiRLCAlgebra.Binding.relaxedOps
      (shape := FullShape logicalWidth publicFits) (rows := productionProfile.commitmentWidth))
    (PaperExtractionAlgebra.extractionAlgebra ajtai) strongSet correct
    (BindingBridge.compatible relation ajtai) checkSpec
    (InteractiveComposition.firstPhase originalFirstPhase publicCheck context)
    (PaperAlgebra.openingMaps ajtai) productionGlobalParams
    ((ProductionKey.key relation ajtai).statement (running context) (fresh context))
  simpa only [localDisagreement, localBindingProbability, BindingEvent,
    PaperCompositionAgreement.bindingProbability, PaperCompositionAgreement.weakSuffixLaw,
    PaperCompositionAgreement.weakConsume, oracle, checker,
    InteractiveComposition.suffixLaw, InteractiveComposition.consume,
    WeakExtraction.endpointLaw, WeakExtraction.consume, PaperWeakAlgorithm.Algorithm.check] using bound

include strongSet correct in
/-- The exact global disagreement term in Main is dominated by the tagged
binding event under the same context PMF. Bounded event mass supplies all
summability; no uniform runtime or numerical security premise is added. -/
theorem disagreement_le_bindingProbability (contexts : PMF Context) :
    PaperCompositionProbability.disagreementProbability contexts
      (InteractiveComposition.firstPhase originalFirstPhase publicCheck)
      (none : InteractiveComposition.Endpoint relation ajtai)
      (InteractiveComposition.suffixLaw relation ajtai running fresh continuation)
      (InteractiveComposition.consume relation ajtai program)
      (fun _ => PaperAlgebra.openingMaps ajtai) productionGlobalParams
      (fun context => (ProductionKey.key relation ajtai).statement (running context) (fresh context)) ≤
    bindingProbability relation ajtai running fresh originalFirstPhase publicCheck continuation program contexts := by
  let base := fun context (_ : CubePoint K productionShape.cubeVariables) (_ : K)
      (_ : CubePoint K productionShape.cubeVariables) =>
    localBindingProbability relation ajtai running fresh originalFirstPhase publicCheck continuation program context
  let total := fun context (_ : CubePoint K productionShape.cubeVariables) (_ : K)
      (_ : CubePoint K productionShape.cubeVariables) =>
    localDisagreement relation ajtai running fresh originalFirstPhase publicCheck continuation program context
  have baseSummable := StrongProbability.clockMean_summable_of_bounded (shape := productionShape) contexts base 1 (by
    intro context alpha gamma point
    exact localBindingProbability_range relation ajtai running fresh originalFirstPhase publicCheck
      continuation program context)
  have averaged := StrongProbability.clockMean_le_add_const (shape := productionShape) contexts base total 0
    (fun context _ _ _ => localDisagreement_nonnegative relation ajtai running fresh
      originalFirstPhase publicCheck continuation program context) baseSummable (by
        intro context alpha gamma point
        simpa only [base, total, add_zero] using local_disagreement_le_binding
          relation ajtai running fresh originalFirstPhase publicCheck continuation program
          strongSet correct context)
  simpa only [StrongProbability.clockMean, base, total, StrongProbability.verifierMean_const,
    add_zero, bindingProbability, PaperCompositionProbability.disagreementProbability,
    StrongProbability.globalDisagreementProbability, localDisagreement] using averaged.2

end NightstreamFPrime.Lifecycle.Nifs.InteractiveAgreement
