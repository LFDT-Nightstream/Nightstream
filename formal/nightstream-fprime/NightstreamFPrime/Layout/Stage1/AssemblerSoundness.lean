import NightstreamFPrime.Layout.Stage1.AssemblerInputs
import NightstreamFPrime.Layout.Stage1.PiCCSRepresentation
import NightstreamFPrime.Lifecycle.Stage1.Accumulator

/-!
Owns deterministic semantic composition for the seven-child Stage 1 parent.

The representation record names the exact typed HyperNova input and output
carried by the symbolic wires. The theorem derives every fixed augmented-step
equation except the recursive SuperNeo accumulator graph, which remains one
explicit premise until compact PiCCS, PiRLC, and PiDEC phase results are
composed in this layout.

This file does not emit rows, select an application, close the recursive fixed
point, or include the outer terminal verifier.
-/

namespace NightstreamFPrime.Layout.Stage1.AssemblerSoundness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.Stage1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Spec.HyperNova.NonInteractiveMultiFold

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

def initialStateValue
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {program : Application.Program}
    (interface : Lifecycle.Stage1.Interface relation program)
    (offset : Nat) (env : Env) : AppState :=
  List.ofFn fun index =>
    (interface.running.initialState offset index).eval env

def currentStateValue
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {program : Application.Program}
    (interface : Lifecycle.Stage1.Interface relation program)
    (offset : Nat) (env : Env) : AppState :=
  List.ofFn fun index =>
    (interface.running.currentState offset index).eval env

def recursiveRunningValue
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {program : Application.Program}
    (interface : Lifecycle.Stage1.Interface relation program)
    (offset : Nat) (env : Env) :
    Running (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  PiCCS.v1_1.StatementAbsorption.evalRunning
    (interface.running.recursive offset) env

def outputRunningValue
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {program : Application.Program}
    (interface : Lifecycle.Stage1.Interface relation program)
    (offset : Nat) (env : Env) :
    Running (logicalWidth := logicalWidth) (publicFits := publicFits) :=
  PiCCS.v1_1.StatementAbsorption.evalRunning
    (interface.running.output offset) env

/-- Typed meaning of the external wires of one fixed Stage 1 parent. -/
structure Represents
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest) (program : Application.Program)
    (interface : Lifecycle.Stage1.Interface relation program)
    (template : Proof (ProductionKey.degreeBound relation))
    (offset : Nat) (env : Env)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (ProductionKey.degreeBound relation)) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      slotCount) : Prop where
  priorPreimage : PriorStateHash.RepresentsPreimage interface.pilot.prior
    (Lifecycle.Stage1.priorOffset offset) env
    (priorHashPreimage (setup relation ajtai vk) input)
  priorPublicInput : PriorStateHash.RepresentsPublicInput
    interface.pilot.prior (Lifecycle.Stage1.priorOffset offset) env
    ((machineFor publicFits program).freshPublic input.fresh)
  nextPreimage : OutputHash.RepresentsPreimage interface.pilot.output
    (Lifecycle.Stage1.outputHashOffset relation program interface offset) env
    (nextHashPreimage (setup relation ajtai vk) input output)
  nextDigest : OutputHash.RepresentsDigest interface.pilot.output
    (Lifecycle.Stage1.outputHashOffset relation program interface offset) env
    output.x
  applicationInput : Application.inputState interface.application
    (Lifecycle.Stage1.applicationOffset relation ajtai program interface
      template offset) env = input.zi
  applicationWitness : Application.witnessValue interface.application
    (Lifecycle.Stage1.applicationOffset relation ajtai program interface
      template offset) env = input.witness
  applicationOutput : Application.outputState interface.application
    (Lifecycle.Stage1.applicationOffset relation ajtai program interface
      template offset) env = output.zNext
  iterationZero :
    RunningTransition.iterationValue interface.running
        (Lifecycle.Stage1.runningOffset relation ajtai program interface
          template offset) env = 0 ↔
      input.iteration = 0
  initialState : initialStateValue interface
    (Lifecycle.Stage1.runningOffset relation ajtai program interface
      template offset) env = input.z0
  currentState : currentStateValue interface
    (Lifecycle.Stage1.runningOffset relation ajtai program interface
      template offset) env = input.zi
  runningOutput : outputRunningValue interface
    (Lifecycle.Stage1.runningOffset relation ajtai program interface
      template offset) env = output.runningNext functionIndex
  priorPc : input.priorPc = 1
  pcNext : output.pcNext = functionIndex

private theorem stateValues_eq
    {relation : ProductionKey.LogicalRelation logicalWidth publicFits}
    {program : Application.Program}
    (interface : Lifecycle.Stage1.Interface relation program)
    (offset : Nat) (env : Env)
    (equal : ∀ index,
      (interface.running.initialState offset index).eval env =
        (interface.running.currentState offset index).eval env) :
    initialStateValue interface offset env =
      currentStateValue interface offset env := by
  apply List.ext_get
  · simp [initialStateValue, currentStateValue]
  · intro index leftBound rightBound
    have bounded : index < RunningTransition.stateWordCount := by
      simpa [initialStateValue] using leftBound
    simpa [initialStateValue, currentStateValue] using
      equal ⟨index, bounded⟩

private theorem slot_eq_functionIndex (slot : Fin slotCount) :
    slot = functionIndex := by
  apply Fin.ext
  have bound := slot.isLt
  change slot.val < 1 at bound
  change slot.val = 0
  omega

private theorem application_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest)
    (program : Application.Program)
    (interface : Lifecycle.Stage1.Interface relation program)
    (template : Proof (ProductionKey.degreeBound relation))
    (offset : Nat) (env : Env)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (ProductionKey.degreeBound relation)) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      slotCount)
    (specification : Lifecycle.Stage1.SpecHolds relation ajtai program
      interface template offset env)
    (represents : Represents relation ajtai vk program interface template
      offset env input output) :
    output.zNext = program.step input.zi input.witness := by
  let applicationOffsetValue := Lifecycle.Stage1.applicationOffset relation
    ajtai program interface template offset
  have applicationSpec :
      (program.circuit interface.application).spec applicationOffsetValue env := by
    simpa [Lifecycle.Stage1.applicationChild, applicationOffsetValue] using
      specification.application
  have applicationHolds :
      Application.Holds program.step interface.application
        applicationOffsetValue env :=
    (program.spec_iff interface.application applicationOffsetValue env).mp
      applicationSpec
  unfold Application.Holds at applicationHolds
  calc
    output.zNext = Application.outputState interface.application
        applicationOffsetValue env :=
      represents.applicationOutput.symm
    _ = program.step
        (Application.inputState interface.application applicationOffsetValue env)
        (Application.witnessValue interface.application applicationOffsetValue
          env) :=
      applicationHolds
    _ = program.step input.zi input.witness := by
      rw [represents.applicationInput, represents.applicationWitness]

/-- Arbitrary satisfying child assignments imply the exact fixed HyperNova
step once their external wires are identified and the recursive SuperNeo
verifier graph is supplied. The accumulator premise is unused on the base
branch. -/
theorem spec_implies_stepHoldsFor
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (vk : KeyDigest) (program : Application.Program)
    (interface : Lifecycle.Stage1.Interface relation program)
    (template : Proof (ProductionKey.degreeBound relation))
    (offset : Nat) (env : Env)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
      (Proof (ProductionKey.degreeBound relation)) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := logicalWidth) (publicFits := publicFits))
      slotCount)
    (specification : Lifecycle.Stage1.SpecHolds relation ajtai program
      interface template offset env)
    (represents : Represents relation ajtai vk program interface template
      offset env input output)
    (accumulator : input.iteration ≠ 0 →
      Accumulator.Holds relation ajtai vk (input.running functionIndex)
        input.fresh input.nifsProof
        (recursiveRunningValue interface
          (Lifecycle.Stage1.runningOffset relation ajtai program interface
            template offset) env)) :
    StepHoldsFor relation ajtai vk program input output := by
  let runningAt := Lifecycle.Stage1.runningOffset relation ajtai program
    interface template offset
  have runningSpec : RunningTransition.SpecHolds interface.running
      runningAt env := by
    simpa [Lifecycle.Stage1.runningChild, runningAt] using
      specification.running
  have application : output.zNext = program.step input.zi input.witness :=
    application_eq relation ajtai vk program interface template offset env
      input output specification represents
  have outputHash : OutputHolds (setup relation ajtai vk)
      (machineFor publicFits program) input output := by
    have childSpec : OutputHash.SpecHolds interface.pilot.output
        (Lifecycle.Stage1.outputHashOffset relation program interface offset)
        env := by
      simpa [Lifecycle.Stage1.outputHashChild, Pilot.outputCircuit] using
        specification.outputHash
    simpa [machineFor] using
      OutputHash.builder_implies_output_slot interface.pilot.output
        (Lifecycle.Stage1.outputHashOffset relation program interface offset)
        env relation ajtai vk program.step input output childSpec
        represents.nextPreimage represents.nextDigest
  have priorPublicInput :
      (machineFor publicFits program).freshPublic input.fresh =
        (machineFor publicFits program).encodeInstance
          ((machineFor publicFits program).hash
            (priorHashPreimage (setup relation ajtai vk) input)) := by
    have childSpec : PriorStateHash.SpecHolds interface.pilot.prior
        (Lifecycle.Stage1.priorOffset offset) env := by
      simpa [Lifecycle.Stage1.priorChild, Pilot.priorCircuit] using
        specification.prior
    simpa [machineFor] using
      PriorStateHash.builder_implies_recursive_slot interface.pilot.prior
        (Lifecycle.Stage1.priorOffset offset) env relation ajtai vk
        program.step input childSpec represents.priorPreimage
        represents.priorPublicInput
  change FixedAugmentedTransition (setup relation ajtai vk)
    (machineFor publicFits program) functionIndex input output
  refine ⟨represents.pcNext, ?_, outputHash, ?_⟩
  · simpa [machineFor, machine] using application
  · rcases Nat.eq_zero_or_pos input.iteration with iterationZero |
      iterationPositive
    · have fieldZero : RunningTransition.iterationValue interface.running
          runningAt env = 0 :=
        represents.iterationZero.mpr iterationZero
      have initialState : input.z0 = input.zi := by
        calc
          input.z0 = initialStateValue interface runningAt env :=
            represents.initialState.symm
          _ = currentStateValue interface runningAt env :=
            stateValues_eq interface runningAt env
              (runningSpec.initialState fieldZero)
          _ = input.zi := represents.currentState
      have runningBase : outputRunningValue interface runningAt env =
          defaultRunning (logicalWidth := logicalWidth)
            (publicFits := publicFits) := by
        apply PiCCSRepresentation.serializeRunning_injective
        exact RunningTransition.spec_serialized_base runningSpec fieldZero
      have defaultOutput : output.runningNext =
          fun _ => (setup relation ajtai vk).defaultRunning := by
        funext slot
        have slotEq : slot = functionIndex := slot_eq_functionIndex slot
        subst slot
        calc
          output.runningNext functionIndex =
              outputRunningValue interface runningAt env :=
            represents.runningOutput.symm
          _ = defaultRunning (logicalWidth := logicalWidth)
              (publicFits := publicFits) := runningBase
          _ = (setup relation ajtai vk).defaultRunning := rfl
      exact Or.inl ⟨iterationZero, initialState, defaultOutput⟩
    · have iterationNonzero : input.iteration ≠ 0 :=
        Nat.ne_of_gt iterationPositive
      have fieldNonzero : RunningTransition.iterationValue interface.running
          runningAt env ≠ 0 := by
        intro fieldZero
        exact iterationNonzero (represents.iterationZero.mp fieldZero)
      have runningRecursive : outputRunningValue interface runningAt env =
          recursiveRunningValue interface runningAt env := by
        apply PiCCSRepresentation.serializeRunning_injective
        exact RunningTransition.spec_serialized_recursive runningSpec
          fieldNonzero
      have priorPcValid : InRange slotCount input.priorPc := by
        rw [represents.priorPc]
        norm_num [InRange, slotCount]
      have selectedEq : selectedIndex priorPcValid = functionIndex :=
        slot_eq_functionIndex _
      have selectedNifs : Accepts (setup relation ajtai vk).nifs
          ((setup relation ajtai vk).verifierKeys
            (selectedIndex priorPcValid))
          (input.running (selectedIndex priorPcValid)) input.fresh
          input.nifsProof
          (output.runningNext (selectedIndex priorPcValid)) := by
        have accepted := accumulator iterationNonzero
        rw [selectedEq]
        rw [← represents.runningOutput, runningRecursive]
        simpa only [Accepts, setup, nifsVerifier, Accumulator.Holds] using
          accepted
      have unchanged : ∀ slot, slot ≠ selectedIndex priorPcValid →
          output.runningNext slot = input.running slot := by
        intro slot notSelected
        exact False.elim (notSelected
          ((slot_eq_functionIndex slot).trans selectedEq.symm))
      exact Or.inr ⟨priorPcValid, iterationPositive, priorPublicInput,
        selectedNifs, unchanged⟩

end NightstreamFPrime.Layout.Stage1.AssemblerSoundness
