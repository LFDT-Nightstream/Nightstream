import NightstreamFPrime.Lifecycle.Nifs.InteractiveComposition
import NightstreamFPrime.Spec.Folding.Nifs.PaperCompositionWork

/-!
The composed extractor decodes its actual weak return, checks the PiCCS
witness and projects the source values. Its returned-source event is the
same event measured by the interactive composition bound.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.InteractiveOutput

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
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
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
  (sourceProgram : Context → CheckedWitnessExtraction.Program productionShape
    (FullShape logicalWidth publicFits))

/-- This probability counts actual returned values that satisfy the selected
source relation, including the verifier-owned public prefix. -/
noncomputable def returnedSourceProbability (contexts : PMF Context) : ℝ :=
  PaperCompositionProbability.eventProbability contexts
    (InteractiveComposition.firstPhase originalFirstPhase publicCheck)
    (InteractiveComposition.suffixLaw relation ajtai running fresh continuation)
    (InteractiveComposition.consume relation ajtai program)
    fun context outcome => CheckedWitnessExtraction.SourceReturned
      (PaperAlgebra.openingMaps ajtai).commit productionGlobalParams
      ((ProductionKey.key relation ajtai).statement (running context) (fresh context))
      (CheckedWitnessExtraction.finish (sourceProgram context) outcome).value

omit [Fintype (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)]
  [Nonempty (Challenge (ProductionKey.key relation ajtai).piRlcAlgebra)] in
/-- The costed terminal program returns the same value used in the measured
source event. No opening or source witness is selected by proof choice. -/
theorem finish_value_eq (context : Context) (probe : Probe K productionShape)
    (endpoint : InteractiveComposition.Endpoint relation ajtai) :
    (PaperCompositionWork.finish (sourceProgram context)
      (PaperWeakOutput.decode (ProductionKey.key relation ajtai)) probe
      (PaperWeakLaw.terminalValue program endpoint)).value =
    (CheckedWitnessExtraction.finish (sourceProgram context)
      ((WeakExtraction.consume relation ajtai program endpoint).map fun witness => (probe, witness))).value := rfl

variable
  (sourceCorrect : ∀ context, CheckedWitnessExtraction.Correct (width := 9)
    (sourceProgram context) (PaperAlgebra.openingMaps ajtai).commit productionGlobalParams
    ((ProductionKey.key relation ajtai).statement (running context) (fresh context)))

include sourceCorrect in
/-- The semantic success event is exactly the event of the actual checked
source result. The source program's own check and access implementations
must satisfy the selected relation. -/
theorem returnedSourceProbability_eq (contexts : PMF Context) :
    returnedSourceProbability relation ajtai running fresh originalFirstPhase publicCheck
      continuation program sourceProgram contexts =
    PaperCompositionProbability.sourceProbability contexts
      (InteractiveComposition.firstPhase originalFirstPhase publicCheck)
      (InteractiveComposition.suffixLaw relation ajtai running fresh continuation)
      (InteractiveComposition.consume relation ajtai program)
      (fun _ => PaperAlgebra.openingMaps ajtai) productionGlobalParams
      (fun context => (ProductionKey.key relation ajtai).statement (running context) (fresh context)) := by
  unfold returnedSourceProbability PaperCompositionProbability.sourceProbability
  congr 1
  funext context outcome
  exact propext (CheckedWitnessExtraction.finish_source_iff (sourceProgram context)
    (PaperAlgebra.openingMaps ajtai).commit productionGlobalParams
    ((ProductionKey.key relation ajtai).statement (running context) (fresh context))
    (sourceCorrect context) outcome)

end NightstreamFPrime.Lifecycle.Nifs.InteractiveOutput
