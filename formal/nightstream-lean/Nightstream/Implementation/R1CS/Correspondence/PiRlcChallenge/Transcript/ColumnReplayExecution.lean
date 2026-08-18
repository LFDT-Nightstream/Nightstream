import Mathlib.Data.List.Basic
import Mathlib.Data.List.OfFn
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.ColumnReplay

/-!
Small result checker for bounded column-replay execution leaves.

Owns an injective finite view of replay cursors and runs. A concrete owner can
reduce one bounded execution leaf to this view without comparing functions.
It owns no generated schedule, row, protocol operation, or artifact validity.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.ColumnReplayExecution

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript

structure CursorView where
  lanes : List Nat
  absorbed : Nat
  nextPin : Nat
  nextCall : Nat
deriving DecidableEq

def cursorView (cursor : ColumnReplay.Cursor) : CursorView where
  lanes := List.ofFn cursor.lanes
  absorbed := cursor.absorbed.val
  nextPin := cursor.nextPin
  nextCall := cursor.nextCall

theorem cursorView_injective : Function.Injective cursorView := by
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

structure RunView where
  cursor : CursorView
  digests : List (List Nat)
deriving DecidableEq

def runView (run : ColumnReplay.Run) : RunView where
  cursor := cursorView run.cursor
  digests := run.digests.map List.ofFn

theorem runView_injective : Function.Injective runView := by
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

def executionMatches
    (result : Option ColumnReplay.Run) (expected : ColumnReplay.Run) : Bool :=
  match result with
  | none => false
  | some actual => decide (runView actual = runView expected)

theorem executionMatches_sound
    {result : Option ColumnReplay.Run} {expected : ColumnReplay.Run}
    (checked : executionMatches result expected = true) :
    result = some expected := by
  cases result with
  | none => simp [executionMatches] at checked
  | some actual =>
      have viewsEqual : runView actual = runView expected := by
        exact of_decide_eq_true (by simpa [executionMatches] using checked)
      rw [runView_injective viewsEqual]

/-- One exact call record is sufficient to prove one physical permutation.
The caller owns the small leaf proof that selects the call from its source
trace and that its input columns equal the current cursor lanes. -/
theorem permute_of_call
    (trace : TranscriptCertificate.Trace) (cursor : ColumnReplay.Cursor)
    (call : Poseidon2Call.Call)
    (bounded : cursor.nextCall < trace.calls.length)
    (callExact : trace.calls.get ⟨cursor.nextCall, bounded⟩ = call)
    (inputsExact : call.inputColumns = ColumnReplay.laneColumns cursor) :
    ColumnReplay.permute trace cursor = some {
      lanes := ColumnReplay.callOutputColumns call
      absorbed := ⟨0, by decide⟩
      nextPin := cursor.nextPin
      nextCall := cursor.nextCall + 1
    } := by
  unfold ColumnReplay.permute
  split
  case isFalse notBounded => exact (notBounded bounded).elim
  case isTrue actualBounded =>
    have indexExact :
        (⟨cursor.nextCall, actualBounded⟩ : Fin trace.calls.length) =
          ⟨cursor.nextCall, bounded⟩ := by
      rfl
    rw [indexExact, callExact]
    simp only [inputsExact, ↓reduceIte]

end Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript.ColumnReplayExecution
