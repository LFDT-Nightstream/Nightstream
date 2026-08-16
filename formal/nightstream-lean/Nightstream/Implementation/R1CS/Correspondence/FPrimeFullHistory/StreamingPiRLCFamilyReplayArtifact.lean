import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCAuthority
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyReplay
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.TranscriptMachineDuplex

/-!
Contract: exact semantic boundary for the production PiRLC family replays.

Assurance tier: Rust-conformant source-row correspondence.

Owns both cursor-parity shapes, the physical input and output call traces,
their exact column executions, and refinement to the overwrite-duplex
`absorbSlice` operation used by `FamilyPhaseRelation`.

Does not own PiRLC arithmetic, the input residual, challenge carry, the family
cursor, selective lowering, or recursive lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyReplayArtifact

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyReplay.Artifact
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyReplay
open Nightstream.SuperNeo.Concrete

inductive CursorParity where
  | even
  | odd
deriving DecidableEq, Repr

inductive ReplayKind where
  | input
  | output
deriving DecidableEq, Repr

def arm : CursorParity → RawArm
  | .even => evenArm
  | .odd => oddArm

def beforeAbsorbed : CursorParity → Fin (rate + 1)
  | .even => ⟨0, by decide⟩
  | .odd => ⟨2, by decide⟩

def afterAbsorbed : CursorParity → Fin (rate + 1)
  | .even => ⟨2, by decide⟩
  | .odd => ⟨0, by decide⟩

def replayColumns (parity : CursorParity) : ReplayKind → List Nat
  | .input => (arm parity).inputColumns
  | .output => (arm parity).outputColumns

def beforeColumns (parity : CursorParity) : ReplayKind → List Nat
  | .input => (arm parity).inputBeforeColumns
  | .output => (arm parity).outputBeforeColumns

def afterColumns (parity : CursorParity) : ReplayKind → List Nat
  | .input => (arm parity).inputAfterColumns
  | .output => (arm parity).outputAfterColumns

def trace (parity : CursorParity) : ReplayKind → TranscriptCertificate.Trace
  | .input => {
      pins := []
      calls := (arm parity).poseidon2Calls.take
        (arm parity).inputPoseidon2CallCount }
  | .output => {
      pins := []
      calls := (arm parity).poseidon2Calls.drop
        (arm parity).inputPoseidon2CallCount }

def operations (parity : CursorParity) (kind : ReplayKind) :
    List ColumnReplay.Operation :=
  (replayColumns parity kind).map ColumnReplay.Operation.external

def runFor
    (columns : List Nat) (absorbed : Fin (rate + 1)) : ColumnReplay.Run where
  cursor := {
    lanes := fun lane => columns.getD lane.val 0
    absorbed := absorbed
    nextPin := 0
    nextCall := 0 }
  digests := []

def startRun (parity : CursorParity) (kind : ReplayKind) : ColumnReplay.Run :=
  runFor (beforeColumns parity kind) (beforeAbsorbed parity)

def resultRun (parity : CursorParity) (kind : ReplayKind) : ColumnReplay.Run :=
  {
    cursor := {
      lanes := fun lane => (afterColumns parity kind).getD lane.val 0
      absorbed := afterAbsorbed parity
      nextPin := 0
      nextCall := (trace parity kind).calls.length }
    digests := [] }

private structure CursorView where
  lanes : List Nat
  absorbed : Nat
  nextPin : Nat
  nextCall : Nat
deriving DecidableEq

private def cursorView (cursor : ColumnReplay.Cursor) : CursorView where
  lanes := List.ofFn cursor.lanes
  absorbed := cursor.absorbed.val
  nextPin := cursor.nextPin
  nextCall := cursor.nextCall

private theorem cursorView_injective : Function.Injective cursorView := by
  intro left right equal
  cases left with
  | mk leftLanes leftAbsorbed leftPin leftCall =>
      cases right with
      | mk rightLanes rightAbsorbed rightPin rightCall =>
          have lanesEqual : leftLanes = rightLanes :=
            List.ofFn_injective (congrArg CursorView.lanes equal)
          have absorbedEqual : leftAbsorbed = rightAbsorbed :=
            Fin.ext (congrArg CursorView.absorbed equal)
          have pinEqual : leftPin = rightPin :=
            congrArg CursorView.nextPin equal
          have callEqual : leftCall = rightCall :=
            congrArg CursorView.nextCall equal
          subst rightLanes
          subst rightAbsorbed
          subst rightPin
          subst rightCall
          rfl

private structure RunView where
  cursor : CursorView
  digests : List (List Nat)
deriving DecidableEq

private def runView (run : ColumnReplay.Run) : RunView where
  cursor := cursorView run.cursor
  digests := run.digests.map List.ofFn

private theorem runView_injective : Function.Injective runView := by
  intro left right equal
  cases left with
  | mk leftCursor leftDigests =>
      cases right with
      | mk rightCursor rightDigests =>
          have cursorEqual : leftCursor = rightCursor :=
            cursorView_injective (congrArg RunView.cursor equal)
          have digestEqual : leftDigests = rightDigests := by
            apply (List.map_injective_iff.mpr fun first second valuesEqual =>
              List.ofFn_injective valuesEqual)
            exact congrArg RunView.digests equal
          subst rightCursor
          subst rightDigests
          rfl

private def executionMatches
    (result : Option ColumnReplay.Run) (expected : ColumnReplay.Run) : Bool :=
  match result with
  | none => false
  | some actual => decide (runView actual = runView expected)

private theorem executionMatches_sound
    {result : Option ColumnReplay.Run} {expected : ColumnReplay.Run}
    (checked : executionMatches result expected = true) :
    result = some expected := by
  cases result with
  | none => simp [executionMatches] at checked
  | some actual =>
      have viewsEqual : runView actual = runView expected := by
        exact of_decide_eq_true (by simpa [executionMatches] using checked)
      rw [runView_injective viewsEqual]

private theorem evenInputChecked :
    executionMatches
      (ColumnReplay.executeSlice (trace .even .input)
        (startRun .even .input) (operations .even .input))
      (resultRun .even .input) = true := by
  native_decide

private theorem evenOutputChecked :
    executionMatches
      (ColumnReplay.executeSlice (trace .even .output)
        (startRun .even .output) (operations .even .output))
      (resultRun .even .output) = true := by
  native_decide

private theorem oddInputChecked :
    executionMatches
      (ColumnReplay.executeSlice (trace .odd .input)
        (startRun .odd .input) (operations .odd .input))
      (resultRun .odd .input) = true := by
  native_decide

private theorem oddOutputChecked :
    executionMatches
      (ColumnReplay.executeSlice (trace .odd .output)
        (startRun .odd .output) (operations .odd .output))
      (resultRun .odd .output) = true := by
  native_decide

/-- Every generated physical trace consumes its exact input columns and ends
at the exact Rust-emitted lane columns and cursor. -/
theorem execution (parity : CursorParity) (kind : ReplayKind) :
    ColumnReplay.executeSlice (trace parity kind) (startRun parity kind)
        (operations parity kind) =
      some (resultRun parity kind) := by
  cases parity <;> cases kind
  · exact executionMatches_sound evenInputChecked
  · exact executionMatches_sound evenOutputChecked
  · exact executionMatches_sound oddInputChecked
  · exact executionMatches_sound oddOutputChecked

theorem trace_pins_canonical (parity : CursorParity) (kind : ReplayKind) :
    ConstantPins.ValuesCanonical (trace parity kind).pins := by
  cases kind <;> simp [trace, ConstantPins.ValuesCanonical]

private theorem trace_call_member
    (parity : CursorParity) (kind : ReplayKind) (call : Poseidon2Call.Call)
    (member : call ∈ (trace parity kind).calls) :
    call ∈ (arm parity).poseidon2Calls := by
  cases kind with
  | input =>
      apply List.mem_of_mem_take
      simpa [trace] using member
  | output =>
      apply List.mem_of_mem_drop
      simpa [trace] using member

/-- Satisfaction of all replay rows reconstructs independent acceptance of
either compact input or output trace. -/
theorem trace_accepted
    (parity : CursorParity) (kind : ReplayKind)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (arm parity).Satisfied assignment) :
    (trace parity kind).Accepted assignment := by
  constructor
  · cases kind <;> simp [trace]
  · intro call member
    apply Poseidon2PermutationSound.poseidon2Permutation_renamed_sound
      call.columnMap call.columnMap_zero canonical one
    exact satisfied call (trace_call_member parity kind call member)

/-- Accepted rows refine the complete value-level slice, including Rust's
eager final normalization. -/
theorem execution_refines
    (parity : CursorParity) (kind : ReplayKind)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (arm parity).Satisfied assignment) :
    ColumnReplay.semanticExecuteSlice assignment canonical
        (ColumnReplay.decodeRun assignment canonical (startRun parity kind))
        (operations parity kind) =
      ColumnReplay.decodeRun assignment canonical (resultRun parity kind) := by
  apply ColumnReplay.executeSlice_sound canonical
    (trace_pins_canonical parity kind) one
    (trace_accepted parity kind assignment canonical one satisfied)
  exact execution parity kind

/-- Assignment values in one exact generated replay agree with the
independent overwrite-duplex bulk operation. -/
theorem replay_eq_absorbSlice
    (parity : CursorParity) (kind : ReplayKind)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (arm parity).Satisfied assignment) :
    PiRlcChallenge.TranscriptMachineDuplex.toDuplex
        (ColumnReplay.decodeRun assignment canonical
          (resultRun parity kind)).state =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        ((replayColumns parity kind).map assignment)
        (PiRlcChallenge.TranscriptMachineDuplex.toDuplex
          (ColumnReplay.decodeRun assignment canonical
            (startRun parity kind)).state) := by
  calc
    PiRlcChallenge.TranscriptMachineDuplex.toDuplex
        (ColumnReplay.decodeRun assignment canonical
          (resultRun parity kind)).state =
        PiRlcChallenge.TranscriptMachineDuplex.toDuplex
          (ColumnReplay.semanticExecuteSlice assignment canonical
            (ColumnReplay.decodeRun assignment canonical
              (startRun parity kind)) (operations parity kind)).state :=
      congrArg (fun run =>
        PiRlcChallenge.TranscriptMachineDuplex.toDuplex run.state)
        (execution_refines parity kind assignment canonical one satisfied).symm
    _ = Poseidon2Duplex.absorbSlice
          Poseidon2CanonicalConstants.selected
          ((replayColumns parity kind).map assignment)
          (PiRlcChallenge.TranscriptMachineDuplex.toDuplex
            (ColumnReplay.decodeRun assignment canonical
              (startRun parity kind)).state) := by
      simpa [operations] using
        (PiRlcChallenge.TranscriptMachineDuplex.semanticExecuteSlice_external_toDuplex
          assignment canonical
          (ColumnReplay.decodeRun assignment canonical
            (startRun parity kind)) (replayColumns parity kind))
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          ((replayColumns parity kind).map assignment)
          (PiRlcChallenge.TranscriptMachineDuplex.toDuplex
            (ColumnReplay.decodeRun assignment canonical
              (startRun parity kind)).state) := by
      rfl

def stateAt
    (assignment : Nat → Nat) (columns : List Nat)
    (absorbed : Fin (rate + 1)) : BindingState where
  lanes := fun lane => assignment (columns.getD lane.val 0)
  absorbed := absorbed.val

@[simp] theorem decodedRun_eq_stateAt
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (columns : List Nat) (absorbed : Fin (rate + 1)) :
    PiRlcChallenge.TranscriptMachineDuplex.toDuplex
        (ColumnReplay.decodeRun assignment canonical
          (runFor columns absorbed)).state =
      stateAt assignment columns absorbed := by
  rfl

/-- Exact placement of both carried duplex states into the generated replay
columns. The cursor value is part of each equality. -/
structure FamilyStatesPlaced
    (parity : CursorParity) (assignment : Nat → Nat)
    (before after : FamilyState) : Prop where
  inputBefore : before.inputReplay =
    stateAt assignment (beforeColumns parity .input) (beforeAbsorbed parity)
  inputAfter : after.inputReplay =
    stateAt assignment (afterColumns parity .input) (afterAbsorbed parity)
  outputBefore : before.outputReplay =
    stateAt assignment (beforeColumns parity .output) (beforeAbsorbed parity)
  outputAfter : after.outputReplay =
    stateAt assignment (afterColumns parity .output) (afterAbsorbed parity)

/-- The algebraic values read by both replay traces. These equalities are
later derived from the fixed production PiRLC source layout. -/
structure ReplayValuesPlaced
    (parity : CursorParity) (assignment : Nat → Nat)
    (inputs : Source → RingF) (output : RingF) : Prop where
  input : (replayColumns parity .input).map assignment = phaseFields inputs
  output : (replayColumns parity .output).map assignment = ringFields output

/-- Exact input replay equality required by `FamilyTransition`. -/
theorem input_replay_exact
    (parity : CursorParity) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (before after : FamilyState)
    (states : FamilyStatesPlaced parity assignment before after)
    (satisfied : (arm parity).Satisfied assignment) :
    after.inputReplay =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        ((replayColumns parity .input).map assignment)
        before.inputReplay := by
  calc
    after.inputReplay =
        stateAt assignment (afterColumns parity .input)
          (afterAbsorbed parity) := states.inputAfter
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          ((replayColumns parity .input).map assignment)
          (stateAt assignment (beforeColumns parity .input)
            (beforeAbsorbed parity)) := by
      simpa using replay_eq_absorbSlice parity .input assignment canonical one
        satisfied
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          ((replayColumns parity .input).map assignment)
          before.inputReplay := by rw [states.inputBefore]

/-- Exact output replay equality required by `FamilyTransition`. -/
theorem output_replay_exact
    (parity : CursorParity) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (before after : FamilyState)
    (states : FamilyStatesPlaced parity assignment before after)
    (satisfied : (arm parity).Satisfied assignment) :
    after.outputReplay =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        ((replayColumns parity .output).map assignment)
        before.outputReplay := by
  calc
    after.outputReplay =
        stateAt assignment (afterColumns parity .output)
          (afterAbsorbed parity) := states.outputAfter
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          ((replayColumns parity .output).map assignment)
          (stateAt assignment (beforeColumns parity .output)
            (beforeAbsorbed parity)) := by
      simpa using replay_eq_absorbSlice parity .output assignment canonical one
        satisfied
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          ((replayColumns parity .output).map assignment)
          before.outputReplay := by rw [states.outputBefore]

/-- Both accepted replay traces give the two exact semantic equalities that
remain outside the 146,114 PiRLC source rows. -/
theorem family_replays_exact
    (parity : CursorParity) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (before after : FamilyState) (inputs : Source → RingF) (output : RingF)
    (states : FamilyStatesPlaced parity assignment before after)
    (values : ReplayValuesPlaced parity assignment inputs output)
    (satisfied : (arm parity).Satisfied assignment) :
    after.inputReplay =
        Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (phaseFields inputs) before.inputReplay ∧
      after.outputReplay =
        Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (ringFields output) before.outputReplay := by
  constructor
  · rw [← values.input]
    exact input_replay_exact parity assignment canonical one before after
      states satisfied
  · rw [← values.output]
    exact output_replay_exact parity assignment canonical one before after
      states satisfied

theorem artifact_valid : rawArtifact.Valid := rawArtifact_valid

theorem exact_shape :
    rawArtifact.sourceColumns = 146224 /\
      rawArtifact.even.rowCount = 129000 /\
      rawArtifact.even.columnCount = 275240 /\
      rawArtifact.odd.rowCount = 130200 /\
      rawArtifact.odd.columnCount = 276440 /\
      rawArtifact.even.poseidon2Calls.length = 215 /\
      rawArtifact.odd.poseidon2Calls.length = 217 := by
  native_decide

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyReplayArtifact
