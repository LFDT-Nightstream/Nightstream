import NightstreamFPrime.Layout.OptionB.RefreshAssembler
import NightstreamFPrime.Lifecycle.OptionB.FinishStep

/-! The finish circuit at the refresh step's source layout (CW-6.3): the gadgets of
`FinishStep.main` at `sharedEnd` read the refresh step's shared words, so the refresh step's
input scoping (`RefreshStepBounds.assumptions`) gives every child's assumptions and no
input-scoping hypothesis remains. -/

namespace NightstreamFPrime.Layout.OptionB.FinishAssembler

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Gadgets.Poseidon2
open NightstreamFPrime.Spec.HyperNova.Construction2
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.OptionB
open NightstreamFPrime.Layout.OptionB.RefreshAssembler

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Soundness of the finish circuit at the source layout: rows that hold and shared words that
carry an active proof's input give `FinishStep.FinishHolds` at the production cap. -/
theorem rows_implies_finish (program : Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (base : Nat)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits) (circuit : Fin 2) (env : Env)
    (rows : holds env (Circuit.ops (FinishStep.main program relation circuit
      (interface base program.witnessWordCount)) (sharedEnd base program.witnessWordCount)))
    (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
    (proof : Spec.Folding.Refresh.Proof PaperAlgebra.Commitment productionShape 8)
    (represents : RefreshStepSoundness.StepRepresents program
      (interface base program.witnessWordCount) ajtai circuitKeys circuit input output proof env)
    (nextSlots : ∀ slot, Hash.evalList env ((interface (logicalWidth := logicalWidth)
      (publicFits := publicFits) base program.witnessWordCount).next slot).words =
        serializeSlot (output.runningNext slot))
    (nextDepth : ∀ slot, (output.runningNext slot).2 < goldilocksModulus)
    (active : input.iteration ≠ 0) :
    FinishStep.FinishHolds Folding.Schedule.productionLightFoldCap ajtai circuitKeys circuit input
      output :=
  FinishStep.rows_implies_finish program relation circuit (interface base program.witnessWordCount)
    (sharedEnd base program.witnessWordCount) ajtai circuitKeys env
    (RefreshStepBounds.assumptions program relation circuit
      (interface base program.witnessWordCount) (sharedEnd base program.witnessWordCount)
      (inputsBelow base program.witnessWordCount) env)
    rows input output proof represents nextSlots nextDepth active

end NightstreamFPrime.Layout.OptionB.FinishAssembler
