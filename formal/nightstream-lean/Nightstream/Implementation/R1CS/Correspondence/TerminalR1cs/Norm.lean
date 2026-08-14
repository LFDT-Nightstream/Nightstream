import Nightstream.Implementation.R1CS.Correspondence.TerminalR1cs.Atoms
import Nightstream.SuperNeo.Concrete.Phi81Relation.Semantics

/-!
Contract: complete-witness strict `b = 2` norm rows for the terminal
Phi81 relation.

Assurance tier: model-level.

Owns: two rows and one square column for every coordinate of one complete
assignment, exact structural ownership, support, cost, soundness to
`assignmentNormBounded 2`, and honest completeness for prefilled squares.

Does not own: witness or public-column allocation, Ajtai commitments, matrix
evaluation, CCS satisfaction, terminal composition, Rust, or artifacts.

Emits constraints: `2 * shape.carrierWidth` rows and
`shape.carrierWidth` auxiliary columns.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.TerminalR1cs.Norm

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Typed
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- Physical coordinates for every checked assignment entry and square. -/
structure Frame (shape : Phi81Relation.Shape) where
  owner : PhysicalOwner
  firstOrdinal : Nat
  witness : Fin shape.carrierWidth → ColumnId
  square : Fin shape.carrierWidth → ColumnId

/-- The coordinate selected by one of the two row positions per entry. -/
def coordinateAt {shape : Phi81Relation.Shape}
    (position : Fin (2 * shape.carrierWidth)) :
    Fin shape.carrierWidth :=
  ⟨position.val / 2, by
    have below := position.isLt
    omega⟩

/-- Exact row at one physical position. Even positions write the square;
odd positions check the cubic root condition. -/
def rowAt {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (position : Fin (2 * shape.carrierWidth)) : OwnedRow :=
  let coordinate := coordinateAt position
  if position.val % 2 = 0 then
    ⟨⟨frame.owner, frame.firstOrdinal + position.val⟩,
      (CanonicalRow.product
        (frame.square coordinate)
        (frame.witness coordinate)
        (frame.witness coordinate)).row⟩
  else
    ⟨⟨frame.owner, frame.firstOrdinal + position.val⟩,
      (CanonicalRow.product
        (frame.witness coordinate)
        (frame.square coordinate)
        (frame.witness coordinate)).row⟩

/-- Ordered complete-witness norm program. -/
def rows {shape : Phi81Relation.Shape}
    (frame : Frame shape) : List OwnedRow :=
  List.ofFn (rowAt frame)

/-- Exact allocated square-column family. -/
def columns {shape : Phi81Relation.Shape}
    (frame : Frame shape) : List OwnedColumn :=
  List.ofFn fun coordinate =>
    ⟨frame.square coordinate, .auxiliaryColumn⟩

@[simp] theorem rows_length {shape : Phi81Relation.Shape}
    (frame : Frame shape) :
    (rows frame).length = 2 * shape.carrierWidth := by
  simp [rows]

@[simp] theorem columns_length {shape : Phi81Relation.Shape}
    (frame : Frame shape) :
    (columns frame).length = shape.carrierWidth := by
  simp [columns]

/-- Physical placement keeps all square columns distinct and outside the
caller-owned witness family. -/
structure Distinct {shape : Phi81Relation.Shape}
    (frame : Frame shape) : Prop where
  squareInjective : Function.Injective frame.square
  witnessNeSquare :
    ∀ witnessCoordinate squareCoordinate,
      frame.witness witnessCoordinate ≠ frame.square squareCoordinate

private theorem nodup_ofFn_of_injective
    {alpha : Type} :
    ∀ {count : Nat}
      (function : Fin count → alpha),
      Function.Injective function →
      (List.ofFn function).Nodup
  | 0, function, injective => by
      simp
  | _ + 1, function, injective => by
      rw [List.ofFn_succ, List.nodup_cons]
      constructor
      · intro member
        rcases List.mem_ofFn.mp member with ⟨index, equal⟩
        exact Fin.succ_ne_zero index (injective equal)
      · exact nodup_ofFn_of_injective
          (fun index => function index.succ)
          (fun first second equal =>
            Fin.succ_inj.mp (injective equal))

theorem columnIds_nodup {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (distinct : Distinct frame) :
    ((columns frame).map fun column => column.id).Nodup := by
  rw [columns, List.map_ofFn]
  exact nodup_ofFn_of_injective frame.square distinct.squareInjective

/-- Row occurrence IDs are positional and therefore duplicate-free. -/
theorem rowIds_nodup {shape : Phi81Relation.Shape}
    (frame : Frame shape) :
    ((rows frame).map fun owned => owned.id).Nodup := by
  rw [rows, List.map_ofFn]
  apply nodup_ofFn_of_injective
  intro first second equal
  apply Fin.ext
  have ordinalEqual :=
    congrArg (fun id : RowId => id.ordinal) equal
  simp only [Function.comp_apply, rowAt] at ordinalEqual
  split at ordinalEqual <;> split at ordinalEqual <;>
    exact Nat.add_left_cancel ordinalEqual

/-- Every norm row occurrence has the declared owner. -/
theorem rows_owned {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (owned : OwnedRow)
    (member : owned ∈ rows frame) :
    owned.id.owner = frame.owner := by
  rcases List.mem_ofFn.mp member with ⟨position, rfl⟩
  unfold rowAt
  split <;> rfl

/-- One norm row mentions only its selected witness coordinate and square. -/
theorem rowAt_supported {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (position : Fin (2 * shape.carrierWidth))
    (column : ColumnId)
    (mentioned : column ∈ (rowAt frame position).columnIds) :
    column = frame.witness (coordinateAt position) ∨
      column = frame.square (coordinateAt position) := by
  by_cases even : position.val % 2 = 0
  · simpa [rowAt, even, OwnedRow.columnIds,
      Nightstream.Implementation.Lowering.Goldilocks.Row.columnIds,
      CanonicalRow.row,
      Nightstream.Implementation.Lowering.Goldilocks.singleton,
      or_comm] using mentioned
  · simpa [rowAt, even, OwnedRow.columnIds,
      Nightstream.Implementation.Lowering.Goldilocks.Row.columnIds,
      CanonicalRow.row,
      Nightstream.Implementation.Lowering.Goldilocks.singleton,
      or_comm] using mentioned

/-- Satisfaction of an `ofFn` row family is pointwise satisfaction. -/
private theorem satisfies_ofFn_iff :
    ∀ {count : Nat}
      (function : Fin count → OwnedRow)
      (assignment : ColumnId → F),
      Nightstream.Implementation.Lowering.Goldilocks.Satisfies
        (List.ofFn function) assignment ↔
        ∀ position, (function position).row.Holds assignment
  | 0, function, assignment => by
      simp
  | _ + 1, function, assignment => by
      rw [List.ofFn_succ, satisfies_cons,
        satisfies_ofFn_iff (fun index => function index.succ) assignment]
      constructor
      · rintro ⟨head, tail⟩ position
        exact Fin.cases head tail position
      · intro every
        exact ⟨every 0, fun index => every index.succ⟩

private theorem satisfies_rows_iff {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (assignment : ColumnId → F) :
    Nightstream.Implementation.Lowering.Goldilocks.Satisfies
      (rows frame) assignment ↔
      ∀ position, (rowAt frame position).row.Holds assignment :=
  satisfies_ofFn_iff (rowAt frame) assignment

private def evenPosition {shape : Phi81Relation.Shape}
    (coordinate : Fin shape.carrierWidth) :
    Fin (2 * shape.carrierWidth) :=
  ⟨2 * coordinate.val, by
    have below := coordinate.isLt
    omega⟩

private def oddPosition {shape : Phi81Relation.Shape}
    (coordinate : Fin shape.carrierWidth) :
    Fin (2 * shape.carrierWidth) :=
  ⟨2 * coordinate.val + 1, by
    have below := coordinate.isLt
    omega⟩

@[simp] private theorem coordinateAt_even
    {shape : Phi81Relation.Shape}
    (coordinate : Fin shape.carrierWidth) :
    coordinateAt (evenPosition coordinate) = coordinate := by
  apply Fin.ext
  simp [coordinateAt, evenPosition]

@[simp] private theorem coordinateAt_odd
    {shape : Phi81Relation.Shape}
    (coordinate : Fin shape.carrierWidth) :
    coordinateAt (oddPosition coordinate) = coordinate := by
  apply Fin.ext
  change (2 * coordinate.val + 1) / 2 = coordinate.val
  omega

@[simp] private theorem rowAt_even
    {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (coordinate : Fin shape.carrierWidth) :
    (rowAt frame (evenPosition coordinate)).row =
      (CanonicalRow.product
        (frame.square coordinate)
        (frame.witness coordinate)
        (frame.witness coordinate)).row := by
  unfold rowAt
  rw [if_pos (by simp [evenPosition])]
  rw [coordinateAt_even]

@[simp] private theorem rowAt_odd
    {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (coordinate : Fin shape.carrierWidth) :
    (rowAt frame (oddPosition coordinate)).row =
      (CanonicalRow.product
        (frame.witness coordinate)
        (frame.square coordinate)
        (frame.witness coordinate)).row := by
  unfold rowAt
  rw [if_neg (by simp [oddPosition])]
  rw [coordinateAt_odd]

private theorem coordinate_satisfied
    {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (assignment : ColumnId → F)
    (satisfied : Nightstream.Implementation.Lowering.Goldilocks.Satisfies
      (rows frame) assignment)
    (coordinate : Fin shape.carrierWidth) :
    Nightstream.Implementation.Lowering.Goldilocks.Satisfies
      (Atoms.normRows frame.owner
        (frame.firstOrdinal + 2 * coordinate.val)
        (frame.witness coordinate)
        (frame.square coordinate))
      assignment := by
  have every := (satisfies_rows_iff frame assignment).mp satisfied
  have evenHolds := every (evenPosition coordinate)
  have oddHolds := every (oddPosition coordinate)
  change
    (CanonicalRow.product
      (frame.square coordinate)
      (frame.witness coordinate)
      (frame.witness coordinate)).row.Holds assignment ∧
    (CanonicalRow.product
      (frame.witness coordinate)
      (frame.square coordinate)
      (frame.witness coordinate)).row.Holds assignment ∧
    True
  simpa only [rowAt_even, rowAt_odd] using
    And.intro evenHolds (And.intro oddHolds True.intro)

/-- The physical rows imply the verifier-authoritative strict norm over every
coordinate of the complete assignment. -/
theorem rows_sound {shape : Phi81Relation.Shape}
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (frame : Frame shape)
    (assignment : ColumnId → F)
    (satisfied : Nightstream.Implementation.Lowering.Goldilocks.Satisfies
      (rows frame) assignment) :
    Phi81Relation.assignmentNormBounded 2
      (fun coordinate => assignment (frame.witness coordinate)) := by
  intro coordinate
  exact Atoms.normRows_strictNormTwo noZeroDivisors frame.owner
    (frame.firstOrdinal + 2 * coordinate.val)
    (frame.witness coordinate)
    (frame.square coordinate)
    assignment
    (coordinate_satisfied frame assignment satisfied coordinate)

/-- Prefilled square values and a bounded witness satisfy every norm row.
The global assembler owns the simultaneous assignment construction. -/
theorem rows_honest {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (assignment : ColumnId → F)
    (distinct : Distinct frame)
    (bounded :
      Phi81Relation.assignmentNormBounded 2
        (fun coordinate => assignment (frame.witness coordinate)))
    (squares :
      ∀ coordinate,
        assignment (frame.square coordinate) =
          assignment (frame.witness coordinate) *
            assignment (frame.witness coordinate)) :
    Nightstream.Implementation.Lowering.Goldilocks.Satisfies
      (rows frame) assignment := by
  apply (satisfies_rows_iff frame assignment).mpr
  intro position
  let coordinate := coordinateAt position
  have fresh :
      frame.witness coordinate ≠ frame.square coordinate :=
    distinct.witnessNeSquare coordinate coordinate
  have localSatisfaction :=
    Atoms.normRows_honest frame.owner
      (frame.firstOrdinal + 2 * coordinate.val)
      (frame.witness coordinate)
      (frame.square coordinate)
      assignment fresh (bounded coordinate)
  have witnessEq :
      Atoms.normWitness assignment
        (frame.witness coordinate)
        (frame.square coordinate) = assignment := by
    funext column
    by_cases equal : column = frame.square coordinate
    · subst column
      rw [Atoms.normWitness_square, squares coordinate]
    · exact Atoms.normWitness_off_square assignment
        (frame.witness coordinate)
        (frame.square coordinate) column equal
  rw [witnessEq] at localSatisfaction
  by_cases even : position.val % 2 = 0
  · simpa [rowAt, even, coordinate] using localSatisfaction.1
  · simpa [rowAt, even, coordinate] using localSatisfaction.2.1

/-- Exact complete-witness norm resource receipt. -/
def cost (shape : Phi81Relation.Shape) : Cost :=
  ⟨2 * shape.carrierWidth, 0, 0, shape.carrierWidth⟩

@[simp] theorem cost_rows (shape : Phi81Relation.Shape) :
    (cost shape).recurringRows = 2 * shape.carrierWidth :=
  rfl

@[simp] theorem cost_auxiliary (shape : Phi81Relation.Shape) :
    (cost shape).auxiliaryColumns = shape.carrierWidth :=
  rfl

end Nightstream.Implementation.R1CS.TerminalR1cs.Norm
