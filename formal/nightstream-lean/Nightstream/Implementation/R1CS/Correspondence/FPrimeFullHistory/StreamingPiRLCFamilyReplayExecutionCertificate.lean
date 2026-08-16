import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyReplay
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.ColumnReplay

/-!
Contract: bounded structural execution certificate for the Rust-emitted
PiRLC family replay schedules.

Assurance tier: Rust-conformant replay geometry certificate.

Owns fixed-size execution leaves, explicit intermediate cursor states, and
structural composition of those leaves for both cursor parities.

Does not own transcript semantics, row satisfaction, PiRLC algebra, or
lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyReplayArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyReplay.Artifact
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyReplay

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

def checkpointRun
    (columns : List Nat) (absorbed : Fin (rate + 1))
    (nextCall : Nat) : ColumnReplay.Run where
  cursor := {
    lanes := fun lane => columns.getD lane.val 0
    absorbed := absorbed
    nextPin := 0
    nextCall := nextCall }
  digests := []

def runFor
    (columns : List Nat) (absorbed : Fin (rate + 1)) : ColumnReplay.Run :=
  checkpointRun columns absorbed 0

def startRun (parity : CursorParity) (kind : ReplayKind) : ColumnReplay.Run :=
  runFor (beforeColumns parity kind) (beforeAbsorbed parity)

def resultRun (parity : CursorParity) (kind : ReplayKind) : ColumnReplay.Run :=
  checkpointRun (afterColumns parity kind) (afterAbsorbed parity)
    (trace parity kind).calls.length

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

private def inputTail0 (parity : CursorParity) := operations parity .input
private def inputChunk0 (parity : CursorParity) := (inputTail0 parity).take 256
private def inputTail1 (parity : CursorParity) := (inputTail0 parity).drop 256
private def inputChunk1 (parity : CursorParity) := (inputTail1 parity).take 256
private def inputTail2 (parity : CursorParity) := (inputTail1 parity).drop 256
private def inputChunk2 (parity : CursorParity) := (inputTail2 parity).take 256
private def inputTail3 (parity : CursorParity) := (inputTail2 parity).drop 256

private def evenInputRun1 : ColumnReplay.Run :=
  checkpointRun [1063, 1064, 1065, 1066, 184036, 184037, 184038, 184039]
    ⟨4, by decide⟩ 63

private def evenInputRun2 : ColumnReplay.Run :=
  checkpointRun [1319, 1320, 1321, 1322, 222436, 222437, 222438, 222439]
    ⟨4, by decide⟩ 127

private def evenInputRun3 : ColumnReplay.Run :=
  checkpointRun [1575, 1576, 1577, 1578, 260836, 260837, 260838, 260839]
    ⟨4, by decide⟩ 191

private theorem evenInputChunk0_execution :
    ColumnReplay.execute (trace .even .input) (startRun .even .input)
        (inputChunk0 .even) =
      some evenInputRun1 := by
  apply executionMatches_sound
  rfl

private theorem evenInputChunk1_execution :
    ColumnReplay.execute (trace .even .input) evenInputRun1
        (inputChunk1 .even) =
      some evenInputRun2 := by
  apply executionMatches_sound
  rfl

private theorem evenInputChunk2_execution :
    ColumnReplay.execute (trace .even .input) evenInputRun2
        (inputChunk2 .even) =
      some evenInputRun3 := by
  apply executionMatches_sound
  rfl

private theorem evenInputTail3_execution :
    ColumnReplay.execute (trace .even .input) evenInputRun3
        (inputTail3 .even) =
      some (resultRun .even .input) := by
  apply executionMatches_sound
  rfl

private theorem evenInputTail2_execution :
    ColumnReplay.execute (trace .even .input) evenInputRun2
        (inputTail2 .even) =
      some (resultRun .even .input) := by
  rw [← List.take_append_drop 256 (inputTail2 .even)]
  exact ColumnReplay.execute_append evenInputChunk2_execution
    evenInputTail3_execution

private theorem evenInputTail1_execution :
    ColumnReplay.execute (trace .even .input) evenInputRun1
        (inputTail1 .even) =
      some (resultRun .even .input) := by
  rw [← List.take_append_drop 256 (inputTail1 .even)]
  exact ColumnReplay.execute_append evenInputChunk1_execution
    evenInputTail2_execution

private theorem evenInputRaw_execution :
    ColumnReplay.execute (trace .even .input) (startRun .even .input)
        (operations .even .input) =
      some (resultRun .even .input) := by
  change ColumnReplay.execute (trace .even .input) (startRun .even .input)
    (inputTail0 .even) = some (resultRun .even .input)
  rw [← List.take_append_drop 256 (inputTail0 .even)]
  exact ColumnReplay.execute_append evenInputChunk0_execution
    evenInputTail1_execution

private def oddInputRun1 : ColumnReplay.Run :=
  checkpointRun [1065, 1066, 184634, 184635, 184636, 184637, 184638, 184639]
    ⟨2, by decide⟩ 64

private def oddInputRun2 : ColumnReplay.Run :=
  checkpointRun [1321, 1322, 223034, 223035, 223036, 223037, 223038, 223039]
    ⟨2, by decide⟩ 128

private def oddInputRun3 : ColumnReplay.Run :=
  checkpointRun [1577, 1578, 261434, 261435, 261436, 261437, 261438, 261439]
    ⟨2, by decide⟩ 192

private def oddInputBeforeNormalize : ColumnReplay.Run :=
  checkpointRun [1617, 1618, 1619, 1620, 267436, 267437, 267438, 267439]
    ⟨4, by decide⟩ 202

private theorem oddInputChunk0_execution :
    ColumnReplay.execute (trace .odd .input) (startRun .odd .input)
        (inputChunk0 .odd) =
      some oddInputRun1 := by
  apply executionMatches_sound
  rfl

private theorem oddInputChunk1_execution :
    ColumnReplay.execute (trace .odd .input) oddInputRun1
        (inputChunk1 .odd) =
      some oddInputRun2 := by
  apply executionMatches_sound
  rfl

private theorem oddInputChunk2_execution :
    ColumnReplay.execute (trace .odd .input) oddInputRun2
        (inputChunk2 .odd) =
      some oddInputRun3 := by
  apply executionMatches_sound
  rfl

private theorem oddInputTail3_execution :
    ColumnReplay.execute (trace .odd .input) oddInputRun3
        (inputTail3 .odd) =
      some oddInputBeforeNormalize := by
  apply executionMatches_sound
  rfl

private theorem oddInputTail2_execution :
    ColumnReplay.execute (trace .odd .input) oddInputRun2
        (inputTail2 .odd) =
      some oddInputBeforeNormalize := by
  rw [← List.take_append_drop 256 (inputTail2 .odd)]
  exact ColumnReplay.execute_append oddInputChunk2_execution
    oddInputTail3_execution

private theorem oddInputTail1_execution :
    ColumnReplay.execute (trace .odd .input) oddInputRun1
        (inputTail1 .odd) =
      some oddInputBeforeNormalize := by
  rw [← List.take_append_drop 256 (inputTail1 .odd)]
  exact ColumnReplay.execute_append oddInputChunk1_execution
    oddInputTail2_execution

private theorem oddInputRaw_execution :
    ColumnReplay.execute (trace .odd .input) (startRun .odd .input)
        (operations .odd .input) =
      some oddInputBeforeNormalize := by
  change ColumnReplay.execute (trace .odd .input) (startRun .odd .input)
    (inputTail0 .odd) = some oddInputBeforeNormalize
  rw [← List.take_append_drop 256 (inputTail0 .odd)]
  exact ColumnReplay.execute_append oddInputChunk0_execution
    oddInputTail1_execution

private theorem evenOutputRaw_execution :
    ColumnReplay.execute (trace .even .output) (startRun .even .output)
        (operations .even .output) =
      some (resultRun .even .output) := by
  apply executionMatches_sound
  rfl

private def oddOutputBeforeNormalize : ColumnReplay.Run :=
  checkpointRun [1671, 1672, 1673, 1674, 275836, 275837, 275838, 275839]
    ⟨4, by decide⟩ 13

private theorem oddOutputRaw_execution :
    ColumnReplay.execute (trace .odd .output) (startRun .odd .output)
        (operations .odd .output) =
      some oddOutputBeforeNormalize := by
  apply executionMatches_sound
  rfl

private theorem evenInputNormalization :
    ColumnReplay.normalizeSlice (trace .even .input)
        (resultRun .even .input) =
      some (resultRun .even .input) := by
  apply executionMatches_sound
  rfl

private theorem evenOutputNormalization :
    ColumnReplay.normalizeSlice (trace .even .output)
        (resultRun .even .output) =
      some (resultRun .even .output) := by
  apply executionMatches_sound
  rfl

private theorem oddInputNormalization :
    ColumnReplay.normalizeSlice (trace .odd .input)
        oddInputBeforeNormalize =
      some (resultRun .odd .input) := by
  apply executionMatches_sound
  rfl

private theorem oddOutputNormalization :
    ColumnReplay.normalizeSlice (trace .odd .output)
        oddOutputBeforeNormalize =
      some (resultRun .odd .output) := by
  apply executionMatches_sound
  rfl

/-- Every generated physical trace consumes its exact input columns and ends
at the exact Rust-emitted lane columns and cursor. -/
theorem execution (parity : CursorParity) (kind : ReplayKind) :
    ColumnReplay.executeSlice (trace parity kind) (startRun parity kind)
        (operations parity kind) =
      some (resultRun parity kind) := by
  cases parity <;> cases kind
  · simp only [ColumnReplay.executeSlice, evenInputRaw_execution]
    exact evenInputNormalization
  · simp only [ColumnReplay.executeSlice, evenOutputRaw_execution]
    exact evenOutputNormalization
  · simp only [ColumnReplay.executeSlice, oddInputRaw_execution]
    exact oddInputNormalization
  · simp only [ColumnReplay.executeSlice, oddOutputRaw_execution]
    exact oddOutputNormalization

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyReplayArtifact
