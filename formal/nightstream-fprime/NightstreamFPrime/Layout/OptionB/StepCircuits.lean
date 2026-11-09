import NightstreamFPrime.Layout.OptionB.LightAssemblerCircuit
import NightstreamFPrime.Layout.OptionB.RefreshAssemblerCircuit

/-! Option B's two step circuits (CW-4). Circuit 0 runs a light fold and circuit 1 a refresh fold
(`BoundedDepth.circuitStep`). `stepCircuit` picks the light or the refresh `FormalCircuit` by the
circuit index, `sharedEnd` picks where its gadgets start, and `StepRepresents` picks what its
shared words carry and that the output's next slots are the digest select's (`nextSlots`). `soundness` gives the option-B step relation of either circuit from its rows
and words; with `stepHolds_iff` (`OptionB.Step`) this is the per-circuit transition of HyperNova
Construction 2 that the bounded-depth lifecycle composes (`Construction2.BoundedDepth`).
Completeness is per kind: `LightAssemblerCompleteness.complete` and
`RefreshAssemblerCompleteness.complete`. -/

namespace NightstreamFPrime.Layout.OptionB.StepCircuits

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.OptionB

open NightstreamFPrime.Spec
open NightstreamFPrime.Circuit
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2

variable {logicalWidth : Nat}
  {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Where the gadgets of circuit `circuit` start. -/
def sharedEnd (base witnessWordCount : Nat) (circuit : Fin 2) : Nat :=
  if circuit.val = 0 then LightAssembler.sharedEnd base witnessWordCount
  else RefreshAssembler.sharedEnd base witnessWordCount

/-- The step circuit of circuit `circuit`: the light step for 0, the refresh step for 1. -/
noncomputable def stepCircuit (program : Stage1.Application.Program) (base : Nat)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (circuit : Fin 2)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    FormalCircuit :=
  if circuit.val = 0 then LightAssemblerCircuit.stepCircuit program base relation circuit ajtai
  else RefreshAssemblerCircuit.stepCircuit program base relation circuit ajtai

/-- What the shared words of circuit `circuit` carry, with the fold proof of its kind, and the
output's next slots: the new slot in the selected slot after the base case, as the digest select
fixes them (`DigestSelect.nextSlots`). -/
def StepRepresents (program : Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (base : Nat)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits) (circuit : Fin 2)
    (input : StepInput logicalWidth publicFits) (output : StepOutput logicalWidth publicFits)
    (env : Env) : Prop :=
  if circuit.val = 0 then
    (∃ proof, LightStepSoundness.StepRepresents program
      (LightAssembler.interface base program.witnessWordCount) ajtai circuitKeys circuit input
      output proof env) ∧
    output.runningNext = DigestSelect.nextSlots input.iteration input.running
      (DigestSelect.selected (LightStepLayout.selectInterface program
          (LightAssembler.interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
            program.witnessWordCount))
        (LightStepLayout.selectOffset program relation circuit
          (LightAssembler.interface base program.witnessWordCount)
          (LightAssembler.sharedEnd base program.witnessWordCount)) env)
      ((LightStepLayout.newSlot program relation circuit
        (LightAssembler.interface base program.witnessWordCount)
        (LightAssembler.sharedEnd base program.witnessWordCount)).value env)
  else
    (∃ proof, RefreshStepSoundness.StepRepresents program
      (RefreshAssembler.interface base program.witnessWordCount) ajtai circuitKeys circuit input
      output proof env) ∧
    output.runningNext = DigestSelect.nextSlots input.iteration input.running
      (DigestSelect.selected (RefreshStepLayout.selectInterface program
          (RefreshAssembler.interface (logicalWidth := logicalWidth) (publicFits := publicFits) base
            program.witnessWordCount))
        (RefreshStepLayout.selectOffset program relation circuit
          (RefreshAssembler.interface base program.witnessWordCount)
          (RefreshAssembler.sharedEnd base program.witnessWordCount)) env)
      ((RefreshStepLayout.newSlot program relation circuit
        (RefreshAssembler.interface base program.witnessWordCount)
        (RefreshAssembler.sharedEnd base program.witnessWordCount)).value env)

/-- Soundness of option B's step circuits: the rows of either circuit at its `sharedEnd`, with
shared words that carry the step and a fold proof of its kind, give that circuit's step relation
at the production cap. -/
theorem soundness (program : Stage1.Application.Program)
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) (base : Nat)
    (ajtai : PaperAlgebra.AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))
    (circuitKeys : Fin 2 → CircuitKey logicalWidth publicFits)
    (step : AppState → AppWitness → AppState) (control : AppState → AppWitness → Fin 2)
    (circuit : Fin 2) (stepEq : program.step = step)
    (validControl : ∀ state witness, program.valid state witness →
      control state witness = circuit)
    (env : Env) (input : StepInput logicalWidth publicFits)
    (output : StepOutput logicalWidth publicFits)
    (rows : (stepCircuit program base relation circuit ajtai).spec
      (sharedEnd base program.witnessWordCount circuit) env)
    (represents : StepRepresents program relation base ajtai circuitKeys circuit input output env) :
    StepHolds Folding.Schedule.productionLightFoldCap ajtai circuitKeys step control circuit input
      output := by
  rcases BoundedDepth.slot_cases circuit with rfl | rfl
  · obtain ⟨⟨proof, words⟩, next⟩ := represents
    exact LightAssembler.rows_implies_stepHolds program relation base ajtai circuitKeys step
      control 0 rfl stepEq validControl env rows input output proof words next
  · obtain ⟨⟨proof, words⟩, next⟩ := represents
    exact RefreshAssembler.rows_implies_stepHolds program relation base ajtai circuitKeys step
      control 1 rfl stepEq validControl env rows input output proof words next

end NightstreamFPrime.Layout.OptionB.StepCircuits
