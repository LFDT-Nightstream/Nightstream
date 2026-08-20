import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPublicHash
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.Artifact.RowAction
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonFirstLeafReconstruction

/-!
Contract: same-assignment final-row bridge for the first recursive-terminal
XOut Poseidon2 leaf.

Assurance tier: artifact-checked row interpretation.

Owns: the exact terminal column projection, bounds for every supported slot,
the exact 86-row final interval, and the implication from exact final matrix
actions to the 86 reconstructed source S-box equations.

Does not own: the complete terminal matrix artifact, satisfaction of that
artifact, later Poseidon2 leaves, lifecycle composition, or collision
resistance. `FinalRowSliceExact` is an explicit matrix-slice obligation;
placement metadata cannot prove it.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonFirstLeafFinalRowBridge

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.Interpreter
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.Artifact.RowAction
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafModel.Wire
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalXOutPublicHash.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalXOutPublicHash
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonFirstLeafCertificate
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonFirstLeafReconstruction

private abbrev LeafStepSboxHolds :=
  Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPoseidonLeafCertificate.StepSboxHolds

def placement : FirstLeafPlacement := rawArtifact.firstLeafPlacement

def finalColumns : Nat := placement.finalColumns

abbrev AbsoluteAssignment := Fin finalColumns → F

def externalSlotStart (lane : Fin 4) : Option Nat :=
  if bounded : lane.val < placement.externalSlotStarts.length then
    some (placement.externalSlotStarts.get ⟨lane.val, bounded⟩)
  else
    none

def digitColumn : Slot → Fin 41 → Option Nat
  | .externalA lane, digit =>
      (externalSlotStart lane).map (fun start => start + digit.val)
  | .externalB _, _ => none
  | .previousLocal _, _ => none
  | .local index, digit =>
      some (placement.localSlotStart +
        index.val * placement.slotWidth + digit.val)

def absoluteValue (assignment : AbsoluteAssignment) (column : Nat) : F :=
  if bounded : column < finalColumns then
    assignment ⟨column, bounded⟩
  else
    0

theorem absoluteValue_of_lt
    (assignment : AbsoluteAssignment) (column : Nat)
    (bounded : column < finalColumns) :
    absoluteValue assignment column = assignment ⟨column, bounded⟩ := by
  simp [absoluteValue, bounded]

theorem selectorColumn_lt : placement.selectorColumn < finalColumns := by
  have valid := firstLeafPlacement_valid
  unfold FirstLeafPlacement.Valid at valid
  rcases valid with ⟨_, _, _, _, bounded, _, _, _, _, _⟩
  exact bounded

theorem zeroColumn_lt : 0 < finalColumns := by
  have bounded := selectorColumn_lt
  omega

private theorem externalSlotStart_mem
    {lane : Fin 4} {start : Nat}
    (owned : externalSlotStart lane = some start) :
    start ∈ placement.externalSlotStarts := by
  unfold externalSlotStart at owned
  split at owned
  next bounded =>
    simp only [Option.some.injEq] at owned
    subst start
    exact List.get_mem _ ⟨lane.val, bounded⟩
  next => simp at owned

private theorem externalSlotStart_fits
    {lane : Fin 4} {start : Nat}
    (owned : externalSlotStart lane = some start) :
    start + placement.slotWidth ≤ finalColumns := by
  have valid := firstLeafPlacement_valid
  unfold FirstLeafPlacement.Valid at valid
  rcases valid with ⟨_, _, _, _, _, _, _, _, fits, _⟩
  exact fits start (externalSlotStart_mem owned)

theorem digitColumn_lt
    {slot : Slot} {digit : Fin 41} {column : Nat}
    (owned : digitColumn slot digit = some column) :
    column < finalColumns := by
  cases slot with
  | externalA lane =>
      cases startExact : externalSlotStart lane with
      | none => simp [digitColumn, startExact] at owned
      | some start =>
          simp [digitColumn, startExact] at owned
          subst column
          have fits := externalSlotStart_fits startExact
          have valid := firstLeafPlacement_valid
          unfold FirstLeafPlacement.Valid at valid
          rcases valid with ⟨_, _, _, _, _, _, width, _, _, _⟩
          unfold finalColumns placement at fits ⊢
          rw [width] at fits
          omega
  | externalB lane => simp [digitColumn] at owned
  | previousLocal index => simp [digitColumn] at owned
  | «local» index =>
      simp [digitColumn] at owned
      subst column
      have valid := firstLeafPlacement_valid
      unfold FirstLeafPlacement.Valid at valid
      rcases valid with
        ⟨_, _, _, _, _, _, width, count, _, fits⟩
      unfold finalColumns placement
      rw [width, count] at fits
      rw [width]
      omega

def projectFinalAssignment
    (assignment : AbsoluteAssignment) : FinalAssignment where
  explicit
    | .one => absoluteValue assignment 0
    | .selector => absoluteValue assignment placement.selectorColumn
  digit slot digit :=
    match digitColumn slot digit with
    | some column => absoluteValue assignment column
    | none => 0

@[simp] theorem projected_one (assignment : AbsoluteAssignment) :
    (projectFinalAssignment assignment).explicit .one =
      absoluteValue assignment 0 := by
  rfl

@[simp] theorem projected_selector (assignment : AbsoluteAssignment) :
    (projectFinalAssignment assignment).explicit .selector =
      absoluteValue assignment placement.selectorColumn := by
  rfl

theorem projected_digit_of_some
    (assignment : AbsoluteAssignment) (slot : Slot) (digit : Fin 41)
    (column : Nat) (owned : digitColumn slot digit = some column) :
    (projectFinalAssignment assignment).digit slot digit =
      assignment ⟨column, digitColumn_lt owned⟩ := by
  rw [show (projectFinalAssignment assignment).digit slot digit =
      absoluteValue assignment column by
    simp [projectFinalAssignment, owned]]
  exact absoluteValue_of_lt assignment column (digitColumn_lt owned)

def absolutePoint (assignment : AbsoluteAssignment)
    (row : Wire.Row) : Fin 13 → F :=
  point row (projectFinalAssignment assignment)

def absoluteResidual (assignment : AbsoluteAssignment)
    (row : Wire.Row) : F :=
  residual row (projectFinalAssignment assignment)

theorem final_row_range_exact :
    placement.finalRows.start + decodedRows.length =
      placement.finalRows.stop := by
  have valid := firstLeafPlacement_valid
  unfold FirstLeafPlacement.Valid at valid
  rcases valid with ⟨_, rangeValid, _, lengthExact, _, _, _, _, _, _⟩
  unfold placement at *
  unfold Range.Valid at rangeValid
  rw [decoded_rows_length]
  omega

def finalRowIndex {rows : Nat}
    (rowsFit : placement.finalRows.stop ≤ rows)
    (offset : Fin decodedRows.length) : Fin rows :=
  ⟨placement.finalRows.start + offset.val, by
    have offsetLt := offset.isLt
    have rangeExact := final_row_range_exact
    omega⟩

/-- Exact action of the complete final matrix on each row in this generated
slice. The complete terminal artifact must prove this field structurally. -/
structure FinalRowSliceExact
    {rows : Nat}
    (relation : InterpretedRelation rows finalColumns)
    (assignment : AbsoluteAssignment) : Prop where
  rowsFit : placement.finalRows.stop ≤ rows
  pointExact : ∀ offset : Fin decodedRows.length,
    rowPoint relation assignment (finalRowIndex rowsFit offset) =
      absolutePoint assignment (decodedRows.get offset)

def AllRowsSatisfied
    {rows : Nat}
    (relation : InterpretedRelation rows finalColumns)
    (assignment : AbsoluteAssignment) : Prop :=
  ∀ row, residualAt relation assignment row = 0

theorem final_rows_imply_decoded_rows
    {rows : Nat}
    {relation : InterpretedRelation rows finalColumns}
    {assignment : AbsoluteAssignment}
    (exact : FinalRowSliceExact relation assignment)
    (satisfied : AllRowsSatisfied relation assignment) :
    ∀ row ∈ decodedRows,
      residual row (projectFinalAssignment assignment) = 0 := by
  intro row member
  rcases List.mem_iff_get.mp member with ⟨offset, rowExact⟩
  subst row
  have finalZero := satisfied (finalRowIndex exact.rowsFit offset)
  rw [residualAt_eq_evaluate, exact.pointExact offset] at finalZero
  exact finalZero

/-- The exact final row slice and one satisfying terminal assignment imply
all 86 typed source S-box equations reconstructed from that same assignment. -/
theorem final_rows_imply_reconstructed_step_sboxes
    {rows : Nat}
    {relation : InterpretedRelation rows finalColumns}
    {assignment : AbsoluteAssignment}
    (exact : FinalRowSliceExact relation assignment)
    (satisfied : AllRowsSatisfied relation assignment)
    (one : absoluteValue assignment 0 = 1)
    (selectorOne :
      absoluteValue assignment placement.selectorColumn = 1) :
    ∀ step ∈ decodedSteps,
      LeafStepSboxHolds
        (reconstructedSource (projectFinalAssignment assignment)) step := by
  exact decoded_rows_imply_reconstructed_step_sboxes
    (projectFinalAssignment assignment) one selectorOne
    (final_rows_imply_decoded_rows exact satisfied)

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalXOutPoseidonFirstLeafFinalRowBridge
