import NightstreamFPrime.Layout.PilotProduction
import NightstreamFPrime.Layout.Stage1.ApplicationInputs
import NightstreamFPrime.Layout.Stage1.PiDECInputs
import NightstreamFPrime.Layout.Stage1.PiRLCInputs
import NightstreamFPrime.Layout.Stage1.RunningTransitionInputs
import NightstreamFPrime.Layout.Stage1.NextPreimageInputs
import NightstreamFPrime.Lifecycle.Stage1.Formal

/-!
Owns the compact logical wiring for the Stage 1 opaque-child assembler.

The existing package uses phase-local source offsets. This constructor instead
places the eight logical children in one adjacent suffix and wires each later
phase to the earlier child's generated expressions. A later lowering theorem
must map this compact logical suffix to the canonical physical package.

This file does not emit rows, select an application, or alter package identity.
-/

namespace NightstreamFPrime.Layout.Stage1.AssemblerInputs

open NightstreamFPrime.Circuit
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤
    Phi81CarrierLayout.carrierWidth logicalWidth}

/-- Application advice extends the canonical source layout before any compact
logical child allocation. -/
def applicationWitnessStart : Nat := Spartan.SourceColumnCount

def applicationWitnessColumn {count : Nat} (index : Fin count) : Nat :=
  applicationWitnessStart + index.val

def applicationLocalStart
    (program : Lifecycle.Stage1.Application.Program) : Nat :=
  applicationWitnessStart + program.witnessWordCount

def applicationInterface
    (program : Lifecycle.Stage1.Application.Program) :
    Lifecycle.Stage1.Application.Interface program.witnessWordCount where
  input := fun _ index => .var (ApplicationInputs.inputSourceColumn index)
  witness := fun _ index => .var (applicationWitnessColumn index)
  output := fun _ index => .var (ApplicationInputs.outputSourceColumn index)

/-- The compact logical suffix starts after every source-coordinate input. -/
def rootOffset (program : Lifecycle.Stage1.Application.Program) : Nat :=
  applicationLocalStart program

def priorOffset (program : Lifecycle.Stage1.Application.Program) : Nat :=
  rootOffset program

def outputHashOffset (program : Lifecycle.Stage1.Application.Program) : Nat :=
  priorOffset program + 7311464

def piCcsOffset (program : Lifecycle.Stage1.Application.Program) : Nat :=
  outputHashOffset program + 7311200

def piRlcOffset (program : Lifecycle.Stage1.Application.Program) : Nat :=
  piCcsOffset program + 4581414

def piDecOffset (program : Lifecycle.Stage1.Application.Program) : Nat :=
  piRlcOffset program + 315894

def runningOffset (program : Lifecycle.Stage1.Application.Program) : Nat :=
  piDecOffset program + 270

def applicationOffset (program : Lifecycle.Stage1.Application.Program) : Nat :=
  runningOffset program + 1

def piCcsInterface
    (program : Lifecycle.Stage1.Application.Program) :
    Lifecycle.PiCCS.v1_1.Formal.Interface logicalWidth 9 publicFits :=
  { PiCCSInputs.interface logicalWidth publicFits with
    baseOffset := piCcsOffset program }

def piCcsOutputState
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) :=
  Lifecycle.PiCCS.v1_1.Formal.outputBindingFinalState relation
    (piCcsInterface (logicalWidth := logicalWidth) (publicFits := publicFits)
      program)
    (piCcsOffset program)

def piCcsRoundPoint
    (program : Lifecycle.Stage1.Application.Program) :=
  Lifecycle.PiCCS.v1_1.Formal.roundPoint
    (piCcsInterface (logicalWidth := logicalWidth) (publicFits := publicFits)
      program)
    (piCcsOffset program)

theorem piCcsRoundPoint_eq_challenge
    (program : Lifecycle.Stage1.Application.Program)
    (coordinate : Fin productionShape.cubeVariables) :
    piCcsRoundPoint
        (logicalWidth := logicalWidth) (publicFits := publicFits) program
        coordinate =
      Lifecycle.PiCCS.v1_1.RoundTranscript.challenge
        (Lifecycle.PiCCS.v1_1.Formal.roundTranscriptInterface
          (Lifecycle.PiCCS.v1_1.Formal.atOffset
            (piCcsInterface
              (logicalWidth := logicalWidth) (publicFits := publicFits) program)
            (piCcsOffset program)))
        (Lifecycle.PiCCS.v1_1.Formal.roundTranscriptOffset
          (piCcsInterface
            (logicalWidth := logicalWidth) (publicFits := publicFits) program)
          (piCcsOffset program)) coordinate := by
  unfold piCcsRoundPoint Lifecycle.PiCCS.v1_1.Formal.roundPoint
  have interfaceEq :
      Lifecycle.PiCCS.v1_1.Formal.atOffset
          (piCcsInterface
            (logicalWidth := logicalWidth) (publicFits := publicFits) program)
          (piCcsOffset program) =
        piCcsInterface
          (logicalWidth := logicalWidth) (publicFits := publicFits) program := by
    rfl
  rw [← Lifecycle.PiCCS.v1_1.Formal.roundTranscriptStart_atOffset
    (piCcsInterface
      (logicalWidth := logicalWidth) (publicFits := publicFits) program)
    (piCcsOffset program), interfaceEq]

/-- PiRLC reads the exact compact PiCCS transcript outputs. Its statement
inputs remain the shared external PiCCS output fields. -/
def piRlcInterface
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) :
    Lifecycle.PiRLC.v1_1.Formal.Interface logicalWidth publicFits where
  baseOffset := piRlcOffset program
  initialState := fun _ => piCcsOutputState relation program
  point := fun _ => piCcsRoundPoint
    (logicalWidth := logicalWidth) (publicFits := publicFits) program
  input := fun _ => PiRLCInputs.sourceInput
    (logicalWidth := logicalWidth) (publicFits := publicFits)

def piRlcShared
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) :=
  Lifecycle.PiRLC.v1_1.Formal.atOffset
    (piRlcInterface relation program) (piRlcOffset program)

def piRlcOutputInterface
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) :=
  Lifecycle.PiRLC.v1_1.Formal.outputBindingInterface
    (piRlcShared relation program) (piRlcOffset program)

def piRlcOutputOffset (program : Lifecycle.Stage1.Application.Program) : Nat :=
  Lifecycle.PiRLC.v1_1.Formal.outputBindingOffset (piRlcOffset program)

/-- PiDEC's parent is exactly the compact PiRLC combined output. -/
def piDecParent
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) :
    Lifecycle.PiDEC.v1_1.InputBinding.ParentExpr logicalWidth publicFits :=
  let output := piRlcOutputInterface relation program
  let offset := piRlcOutputOffset program
  { commitment := output.commitment offset
    publicInput := output.publicInput offset
    evaluation := {
      eval_K := output.eval_K offset
      eval_A := output.eval_A offset } }

def piDecInterface
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) :
    Lifecycle.PiDEC.v1_1.Formal.Interface logicalWidth publicFits where
  parent := fun _ => piDecParent relation program
  point := fun _ =>
    (piRlcOutputInterface relation program).point (piRlcOutputOffset program)
  message := fun _ => PiDECInputs.message
  digit := fun _ child coordinate =>
    PiDECInputs.childPublicInput child
      (Fin.cast
        (Lifecycle.PiDEC.v1_1.PublicInputSplit.coordinateCount_eq
          logicalWidth publicFits)
        coordinate)

theorem runningCount_eq_childCount :
    productionShape.runningCount = productionGlobalParams.k := by
  decide

def childOfRunning
    (source : Fin productionShape.runningCount) : Radix.ChildIndex :=
  Fin.cast runningCount_eq_childCount source

def publicWidth_eq_coordinateCount :
    (FullShape logicalWidth publicFits).publicWidth =
      Lifecycle.PiDEC.v1_1.PublicInputSplit.coordinateCount
        logicalWidth publicFits := by
  rw [Lifecycle.PiDEC.v1_1.PublicInputSplit.coordinateCount_eq]
  rfl

def digitCoordinate
    (coordinate : Fin (FullShape logicalWidth publicFits).publicWidth) :
    Fin (Lifecycle.PiDEC.v1_1.PublicInputSplit.coordinateCount
      logicalWidth publicFits) :=
  Fin.cast publicWidth_eq_coordinateCount coordinate

/-- The recursive running value is exactly the compact PiDEC output family. -/
def recursiveRunningExpr
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) :
    Lifecycle.PiCCS.v1_1.StatementAbsorption.RunningExpr
      logicalWidth publicFits :=
  let piDec := piDecInterface relation program
  let offset := piDecOffset program
  { point := piDec.point offset
    commitment := fun source =>
      (piDec.message offset (childOfRunning source)).commitment
    publicInput := fun source coordinate =>
      piDec.digit offset (childOfRunning source) (digitCoordinate coordinate)
    evaluation := fun source =>
      (piDec.message offset (childOfRunning source)).evaluation }

def runningInterface
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) :
    Lifecycle.Stage1.RunningTransition.Interface logicalWidth publicFits where
  iteration := fun _ => RunningTransitionInputs.iterationExpr
  initialState := fun _ => RunningTransitionInputs.initialStateExpr
  currentState := fun _ => RunningTransitionInputs.currentStateExpr
  recursive := fun _ => recursiveRunningExpr relation program
  output := fun _ =>
    RunningTransitionInputs.outputRunningExpr logicalWidth publicFits

/-- The exact production child interfaces for the compact logical parent. -/
def interface
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) :
    Lifecycle.Stage1.Interface relation program where
  pilot := PilotProduction.interface
  piCcs := piCcsInterface program
  piRlc := piRlcInterface relation program
  piDec := piDecInterface relation program
  running := runningInterface relation program
  application := applicationInterface program
  nextPreimage := NextPreimageInputs.sourceInterface

theorem piRlc_initialState_wiring
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) :
    (interface relation program).piRlc.initialState (piRlcOffset program) =
      piCcsOutputState relation program := by
  change (piRlcInterface relation program).initialState (piRlcOffset program) =
    piCcsOutputState relation program
  simp only [piRlcInterface]

theorem piRlc_point_wiring
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) :
    (interface relation program).piRlc.point (piRlcOffset program) =
      piCcsRoundPoint (logicalWidth := logicalWidth)
        (publicFits := publicFits) program := by
  change (piRlcInterface relation program).point (piRlcOffset program) =
    piCcsRoundPoint (logicalWidth := logicalWidth)
      (publicFits := publicFits) program
  simp only [piRlcInterface]

/-- Evaluating the compact PiDEC parent is exactly evaluating the compact
PiRLC combined output; no copied digest or boundary value intervenes. -/
theorem piDecParent_eval_eq_piRlcOutput
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) (env : Env) :
    (Lifecycle.PiDEC.v1_1.Semantics.inputAttempt relation
      (piDecInterface relation program) (piDecOffset program) env).parent =
      Lifecycle.PiRLC.v1_1.Semantics.evalOutput relation
        (piRlcInterface relation program) (piRlcOffset program) env := by
  rfl

theorem piDec_parent_wiring
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) :
    (interface relation program).piDec.parent (piDecOffset program) =
      piDecParent relation program := by
  change (piDecInterface relation program).parent (piDecOffset program) =
    piDecParent relation program
  rfl

theorem running_recursive_wiring
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) :
    (interface relation program).running.recursive (runningOffset program) =
      recursiveRunningExpr relation program := by
  change (runningInterface relation program).recursive (runningOffset program) =
    recursiveRunningExpr relation program
  rfl

theorem parent_priorOffset_eq
    (_relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) :
    Lifecycle.Stage1.priorOffset (rootOffset program) = priorOffset program := by
  rfl

private theorem priorPrivateCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) :
    (Lifecycle.Stage1.priorChild relation program (interface relation program)
      ).privateCount (priorOffset program) = 7311464 := by
  change Lifecycle.PriorStateHash.logicalPrivateCount
    PilotProduction.priorInterface (priorOffset program) = 7311464
  unfold Lifecycle.PriorStateHash.logicalPrivateCount
    Lifecycle.PriorStateHash.hashLength
  rw [PilotProduction.priorInterface_preimage_apply,
    PilotProduction.priorPreimage_chunkCount]

theorem parent_outputHashOffset_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) :
    Lifecycle.Stage1.outputHashOffset relation program
        (interface relation program) (rootOffset program) =
      outputHashOffset program := by
  unfold Lifecycle.Stage1.outputHashOffset
  rw [parent_priorOffset_eq relation program, priorPrivateCount_eq]
  rfl

theorem pilot_outputOffset_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) :
    Lifecycle.Pilot.outputOffset (interface relation program).pilot
        (rootOffset program) = outputHashOffset program := by
  unfold Lifecycle.Pilot.outputOffset
  change priorOffset program +
      (Lifecycle.Stage1.priorChild relation program
        (interface relation program)).privateCount (priorOffset program) =
    outputHashOffset program
  rw [priorPrivateCount_eq]
  rfl

private theorem outputHashPrivateCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) :
    (Lifecycle.Stage1.outputHashChild relation program
      (interface relation program)).privateCount (outputHashOffset program) =
      7311200 := by
  change Lifecycle.OutputHash.hashLength PilotProduction.outputInterface
    (outputHashOffset program) = 7311200
  unfold Lifecycle.OutputHash.hashLength
  rw [PilotProduction.outputInterface_preimage_apply,
    PilotProduction.outputPreimage_chunkCount]

theorem parent_piCcsOffset_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) :
    Lifecycle.Stage1.piCcsOffset relation program
        (interface relation program) (rootOffset program) =
      piCcsOffset program := by
  unfold Lifecycle.Stage1.piCcsOffset
  rw [parent_outputHashOffset_eq relation program, outputHashPrivateCount_eq]
  rfl

private theorem piCcsPrivateCount_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Lifecycle.Stage1.Application.Program)
    (template : Proof (ProductionKey.degreeBound relation)) :
    (Lifecycle.Stage1.piCcsChild relation ajtai program
      (interface relation program) template).privateCount (piCcsOffset program) =
      4581414 := by
  change Lifecycle.PiCCS.v1_1.Formal.privateCount
    (ProductionKey.degreeBound relation) = 4581414
  exact Lifecycle.PiCCS.v1_1.Formal.privateCount_eq_of_degreeBound_eq_nine
    _ (ProductionKey.degreeBound_eq relation)

theorem parent_piRlcOffset_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Lifecycle.Stage1.Application.Program)
    (template : Proof (ProductionKey.degreeBound relation)) :
    Lifecycle.Stage1.piRlcOffset relation ajtai program
        (interface relation program) template (rootOffset program) =
      piRlcOffset program := by
  unfold Lifecycle.Stage1.piRlcOffset
  rw [parent_piCcsOffset_eq relation program,
    piCcsPrivateCount_eq relation ajtai program template]
  rfl

theorem parent_piDecOffset_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Lifecycle.Stage1.Application.Program)
    (template : Proof (ProductionKey.degreeBound relation)) :
    Lifecycle.Stage1.piDecOffset relation ajtai program
        (interface relation program) template (rootOffset program) =
      piDecOffset program := by
  unfold Lifecycle.Stage1.piDecOffset
  rw [parent_piRlcOffset_eq relation ajtai program template]
  change piRlcOffset program +
    Lifecycle.PiRLC.v1_1.Formal.logicalPrivateCount = piDecOffset program
  rfl

theorem parent_runningOffset_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Lifecycle.Stage1.Application.Program)
    (template : Proof (ProductionKey.degreeBound relation)) :
    Lifecycle.Stage1.runningOffset relation ajtai program
        (interface relation program) template (rootOffset program) =
      runningOffset program := by
  unfold Lifecycle.Stage1.runningOffset
  rw [parent_piDecOffset_eq relation ajtai program template]
  change piDecOffset program +
    Lifecycle.PiDEC.v1_1.Formal.logicalPrivateCount = runningOffset program
  rfl

theorem parent_applicationOffset_eq
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Lifecycle.Stage1.Application.Program)
    (template : Proof (ProductionKey.degreeBound relation)) :
    Lifecycle.Stage1.applicationOffset relation ajtai program
        (interface relation program) template (rootOffset program) =
      applicationOffset program := by
  unfold Lifecycle.Stage1.applicationOffset
  rw [parent_runningOffset_eq relation ajtai program template]
  change runningOffset program +
    Lifecycle.Stage1.RunningTransition.exactPrivateCount =
      applicationOffset program
  rfl

/-- The next-preimage child reuses the pilot preimage columns without a copy. -/
theorem nextPreimage_parent_wiring
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (program : Lifecycle.Stage1.Application.Program) :
    (interface relation program).nextPreimage =
      NextPreimageInputs.sourceInterface := by
  rfl

theorem rootOffset_le_finalOffset
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (ajtai : AjtaiKey
      (logicalWidth := logicalWidth) (publicFits := publicFits))
    (program : Lifecycle.Stage1.Application.Program)
    (template : Proof (ProductionKey.degreeBound relation)) :
    rootOffset program ≤ Lifecycle.Stage1.finalOffset relation ajtai program
      (interface relation program) template (rootOffset program) := by
  unfold Lifecycle.Stage1.finalOffset
  rw [parent_applicationOffset_eq relation ajtai program template]
  unfold applicationOffset runningOffset piDecOffset piRlcOffset piCcsOffset
    outputHashOffset priorOffset
  omega

end NightstreamFPrime.Layout.Stage1.AssemblerInputs
