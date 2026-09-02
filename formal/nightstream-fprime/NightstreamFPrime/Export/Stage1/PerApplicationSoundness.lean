import NightstreamFPrime.Export.Stage1.AccumulatorPackage
import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPreservation
import NightstreamFPrime.Export.Stage1.RunningTransitionPackage
import NightstreamFPrime.Layout.Stage1.PiRLCInputBounds

/-!
Owns deterministic soundness of one generic per-application Stage 1 package.

The representation record identifies the package's canonical external columns
with one typed HyperNova input and output. Package rows then imply the exact
application transition, prior/output hash slots, SuperNeo accumulator update,
and base/recursive running-instance branch. This module adds no row or column.

The final relation plan, concrete application, static-key serializers,
recursive fixed point, and outer terminal verifier remain separate obligations.
-/

namespace NightstreamFPrime.Export.Stage1.PerApplicationSoundness

open NightstreamFPrime.Circuit
open NightstreamFPrime.Export.Package
open NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Lifecycle.Stage1
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Spec.HyperNova.NonInteractiveMultiFold

def sourceEnv (program : Application.Program) (env : Env) : Env :=
  Layout.Stage1.Spartan.pullback
    (PerApplicationPackage.baseEnv program env)

/-- Exact pilot-package environment after the per-application column shift. -/
def pilotEnv (program : Application.Program) (env : Env) : Env :=
  fun column => PerApplicationPackage.baseEnv program env
    (Layout.Stage1.Spartan.liftPilotColumn column)

private theorem slot_eq_functionIndex (slot : Fin slotCount) :
    slot = functionIndex := by
  apply Fin.ext
  have bound := slot.isLt
  change slot.val < 1 at bound
  change slot.val = 0
  omega

/-- Typed meaning of the canonical external package columns. Every field is
an ABI equality or a fixed-length condition; no field assumes an application
transition, hash result, NIFS acceptance, or terminal relation. -/
structure Represents
    (relation : ProductionKey.LogicalRelation Data.logicalWidth
      Data.publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (vk : KeyDigest) (program : Application.Program) (env : Env)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := Data.logicalWidth)
        (publicFits := Data.publicFits))
      (Fresh (logicalWidth := Data.logicalWidth)
        (publicFits := Data.publicFits))
      (Proof (ProductionKey.degreeBound relation)) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := Data.logicalWidth)
        (publicFits := Data.publicFits)) slotCount) : Prop where
  priorFixed : Layout.PilotProduction.FixedPreimage
    (priorHashPreimage (setup relation ajtai vk) input)
  outputFixed : Layout.PilotProduction.FixedPreimage
    (nextHashPreimage (setup relation ajtai vk) input output)
  digestFixed : output.x.length = Layout.PilotProduction.digestWords
  pilot : Layout.PilotProduction.AgreesBelow
    (Layout.PilotSpartan.pullback (pilotEnv program env))
    (Layout.PilotProduction.protocolEnv
      (priorHashPreimage (setup relation ajtai vk) input)
      ((machineFor Data.publicFits program).freshPublic input.fresh)
      (nextHashPreimage (setup relation ajtai vk) input output)
      output.x priorFixed outputFixed digestFixed)
    Layout.PilotProduction.witnessOffset
  applicationInput : Application.inputState
    (Layout.Stage1.ApplicationInputs.interface program)
    (Layout.Stage1.ApplicationInputs.localStart program) env = input.zi
  applicationWitness : Application.witnessValue
    (Layout.Stage1.ApplicationInputs.interface program)
    (Layout.Stage1.ApplicationInputs.localStart program) env = input.witness
  applicationOutput : Application.outputState
    (Layout.Stage1.ApplicationInputs.interface program)
    (Layout.Stage1.ApplicationInputs.localStart program) env = output.zNext
  iterationZero : RunningTransition.iterationValue
    (Layout.Stage1.RunningTransitionInputs.interface Data.logicalWidth
      Data.publicFits)
    Layout.Stage1.RunningTransitionInputs.phaseOffset (sourceEnv program env) =
      0 ↔ input.iteration = 0
  initialState : List.ofFn (fun index =>
    (Layout.Stage1.RunningTransitionInputs.initialStateExpr index).eval
      (sourceEnv program env)) = input.z0
  currentState : List.ofFn (fun index =>
    (Layout.Stage1.RunningTransitionInputs.currentStateExpr index).eval
      (sourceEnv program env)) = input.zi
  runningInput : Layout.Stage1.AccumulatorInputs.running Data.logicalWidth
    Data.publicFits (sourceEnv program env) = input.running functionIndex
  freshInput : Layout.Stage1.AccumulatorInputs.fresh Data.logicalWidth
    Data.publicFits (sourceEnv program env) = input.fresh
  proofInput : Layout.Stage1.AccumulatorInputs.proof relation
    (sourceEnv program env) = input.nifsProof
  runningOutput :
    PiCCS.v1_1.StatementAbsorption.evalRunning
        (Layout.Stage1.RunningTransitionInputs.outputRunningExpr
          Data.logicalWidth Data.publicFits)
        (sourceEnv program env) =
      output.runningNext functionIndex
  priorPc : input.priorPc = 1
  pcNext : output.pcNext = functionIndex

private theorem application_eq
    (relation : ProductionKey.LogicalRelation Data.logicalWidth
      Data.publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (vk : KeyDigest) (program : Application.Program) (env : Env)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := Data.logicalWidth)
        (publicFits := Data.publicFits))
      (Fresh (logicalWidth := Data.logicalWidth)
        (publicFits := Data.publicFits))
      (Proof (ProductionKey.degreeBound relation)) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := Data.logicalWidth)
        (publicFits := Data.publicFits)) slotCount)
    (represents : Represents relation ajtai vk program env input output)
    (rows : (PerApplicationPackage.package program).RowsHold env) :
    output.zNext = program.step input.zi input.witness := by
  have application :=
    PerApplicationPackage.packageRows_imply_applicationHolds program env rows
  unfold Application.Holds at application
  calc
    output.zNext = Application.outputState
        (Layout.Stage1.ApplicationInputs.interface program)
        (Layout.Stage1.ApplicationInputs.localStart program) env :=
      represents.applicationOutput.symm
    _ = program.step
        (Application.inputState
          (Layout.Stage1.ApplicationInputs.interface program)
          (Layout.Stage1.ApplicationInputs.localStart program) env)
        (Application.witnessValue
          (Layout.Stage1.ApplicationInputs.interface program)
          (Layout.Stage1.ApplicationInputs.localStart program) env) :=
      application
    _ = program.step input.zi input.witness := by
      rw [represents.applicationInput, represents.applicationWitness]

/-- Every satisfying final-package assignment that represents one typed ABI
satisfies the exact per-application HyperNova augmented-step relation. -/
theorem packageRows_imply_stepHoldsFor
    (relation : ProductionKey.LogicalRelation Data.logicalWidth
      Data.publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := Data.logicalWidth) (publicFits := Data.publicFits))
    (vk : KeyDigest) (program : Application.Program) (env : Env)
    (input : Input KeyDigest AppState AppWitness
      (Running (logicalWidth := Data.logicalWidth)
        (publicFits := Data.publicFits))
      (Fresh (logicalWidth := Data.logicalWidth)
        (publicFits := Data.publicFits))
      (Proof (ProductionKey.degreeBound relation)) slotCount)
    (output : Output Digest AppState
      (Running (logicalWidth := Data.logicalWidth)
        (publicFits := Data.publicFits)) slotCount)
    (represents : Represents relation ajtai vk program env input output)
    (rows : (PerApplicationPackage.package program).RowsHold env) :
    StepHoldsFor relation ajtai vk program input output := by
  have baseRows : PerApplicationPackage.basePackage.RowsHold
      (PerApplicationPackage.baseEnv program env) :=
    PerApplicationCanonicalPreservation.packageRows_imply_validatedPrefix
      program env rows
  have hashSlots := Package.circuitPackage_implies_recursive_hash_slots
    relation ajtai vk program.step input output represents.priorFixed
      represents.outputFixed represents.digestFixed
      (PerApplicationPackage.baseEnv program env) (by
        simpa [pilotEnv] using represents.pilot) baseRows
  have application : output.zNext = program.step input.zi input.witness :=
    application_eq relation ajtai vk program env input output represents rows
  have accumulator :=
    AccumulatorPackage.circuitPackage_implies_accumulatorHolds relation ajtai vk
      (PerApplicationPackage.baseEnv program env) baseRows
      (Layout.Stage1.PiRLCInputBounds.assumptions relation (sourceEnv program env))
  change Accumulator.Holds relation ajtai vk
    (Layout.Stage1.AccumulatorInputs.running Data.logicalWidth Data.publicFits
      (sourceEnv program env))
    (Layout.Stage1.AccumulatorInputs.fresh Data.logicalWidth Data.publicFits
      (sourceEnv program env))
    (Layout.Stage1.AccumulatorInputs.proof relation (sourceEnv program env))
    (Layout.Stage1.AccumulatorInputs.output relation (sourceEnv program env))
      at accumulator
  rw [represents.runningInput, represents.freshInput,
    represents.proofInput] at accumulator
  change FixedAugmentedTransition (setup relation ajtai vk)
    (machineFor Data.publicFits program) functionIndex input output
  refine ⟨represents.pcNext, application, hashSlots.2, ?_⟩
  rcases Nat.eq_zero_or_pos input.iteration with iterationZero |
      iterationPositive
  · have fieldZero : RunningTransition.iterationValue
        (Layout.Stage1.RunningTransitionInputs.interface Data.logicalWidth
          Data.publicFits)
        Layout.Stage1.RunningTransitionInputs.phaseOffset
          (sourceEnv program env) = 0 :=
      represents.iterationZero.mpr iterationZero
    have initialState : input.z0 = input.zi := by
      calc
        input.z0 = List.ofFn (fun index =>
            (Layout.Stage1.RunningTransitionInputs.initialStateExpr index).eval
              (sourceEnv program env)) := represents.initialState.symm
        _ = List.ofFn (fun index =>
            (Layout.Stage1.RunningTransitionInputs.currentStateExpr index).eval
              (sourceEnv program env)) := by
          apply List.ext_get
          · simp
          · intro index leftBound rightBound
            have bounded : index < RunningTransition.stateWordCount := by
              simpa using leftBound
            have runningSpec :=
              RunningTransitionPackage.circuitPackage_implies_specHolds relation
                (PerApplicationPackage.baseEnv program env) baseRows
            simpa [sourceEnv] using
              runningSpec.initialState fieldZero ⟨index, bounded⟩
        _ = input.zi := represents.currentState
    have runningBase :=
      RunningTransitionPackage.circuitPackage_implies_typed_base relation
        (PerApplicationPackage.baseEnv program env) baseRows fieldZero
    have defaultOutput : output.runningNext =
        fun _ => (setup relation ajtai vk).defaultRunning := by
      funext slot
      have slotEq : slot = functionIndex := slot_eq_functionIndex slot
      subst slot
      calc
        output.runningNext functionIndex =
            PiCCS.v1_1.StatementAbsorption.evalRunning
              (Layout.Stage1.RunningTransitionInputs.outputRunningExpr
                Data.logicalWidth Data.publicFits)
              (sourceEnv program env) := represents.runningOutput.symm
        _ = defaultRunning (logicalWidth := Data.logicalWidth)
              (publicFits := Data.publicFits) := by
            simpa [sourceEnv] using runningBase
        _ = (setup relation ajtai vk).defaultRunning := rfl
    exact Or.inl ⟨iterationZero, initialState, defaultOutput⟩
  · have iterationNonzero : input.iteration ≠ 0 :=
      Nat.ne_of_gt iterationPositive
    have fieldNonzero : RunningTransition.iterationValue
        (Layout.Stage1.RunningTransitionInputs.interface Data.logicalWidth
          Data.publicFits)
        Layout.Stage1.RunningTransitionInputs.phaseOffset
          (sourceEnv program env) ≠ 0 := by
      intro fieldZero
      exact iterationNonzero (represents.iterationZero.mp fieldZero)
    have runningRecursive :=
      RunningTransitionPackage.circuitPackage_implies_typed_recursive relation
        (PerApplicationPackage.baseEnv program env) baseRows fieldNonzero
    change PiCCS.v1_1.StatementAbsorption.evalRunning
        (Layout.Stage1.RunningTransitionInputs.outputRunningExpr
          Data.logicalWidth Data.publicFits)
        (sourceEnv program env) =
      Layout.Stage1.AccumulatorInputs.output relation (sourceEnv program env)
        at runningRecursive
    have accumulatorOutput : Accumulator.Holds relation ajtai vk
        (input.running functionIndex) input.fresh input.nifsProof
        (output.runningNext functionIndex) := by
      rw [← represents.runningOutput]
      rw [runningRecursive]
      exact accumulator
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
      rw [selectedEq]
      simpa only [Accepts, setup, nifsVerifier, Accumulator.Holds] using
        accumulatorOutput
    have unchanged : ∀ slot, slot ≠ selectedIndex priorPcValid →
        output.runningNext slot = input.running slot := by
      intro slot notSelected
      exact False.elim (notSelected
        ((slot_eq_functionIndex slot).trans selectedEq.symm))
    exact Or.inr ⟨priorPcValid, iterationPositive, hashSlots.1,
      selectedNifs, unchanged⟩

end NightstreamFPrime.Export.Stage1.PerApplicationSoundness
