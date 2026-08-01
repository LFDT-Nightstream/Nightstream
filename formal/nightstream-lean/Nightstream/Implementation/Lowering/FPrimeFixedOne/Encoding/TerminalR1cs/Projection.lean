import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Atoms
import Nightstream.SuperNeo.Concrete.Phi81Relation.Types

/-!
Contract: Lean-owned public-projection rows for one Phi81 assignment.

Assurance tier: model-level.

Owns: the exact row family that binds every verifier-visible public coordinate
to the matching coordinate of one complete assignment, its structural owner,
cost, soundness, and honest completeness.

Does not own: allocation of the witness or public columns, commitments, norm
checks, matrix evaluation, CCS satisfaction, terminal assembly, or Rust.

Emits constraints: one linear equality row per public coordinate and no
auxiliary columns.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Projection

open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Typed
open Nightstream.SuperNeo.Concrete

/-- Physical locations read by one public-projection check. The caller owns
both source families; this slice allocates neither one. -/
structure Frame (shape : Phi81Relation.Shape) where
  owner : PhysicalOwner
  firstOrdinal : Nat
  one : ColumnId
  witness : Fin shape.carrierWidth → ColumnId
  publicColumn : Fin shape.publicWidth → ColumnId

/-- Exact row occurrence for one public coordinate. -/
def row {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (coordinate : Fin shape.publicWidth) : OwnedRow :=
  Atoms.linearCheckOwnedRow frame.owner
    (frame.firstOrdinal + coordinate.val) frame.one
    (Goldilocks.singleton
      (frame.witness (shape.publicColumn coordinate)) 1)
    (Goldilocks.singleton (frame.publicColumn coordinate) 1)

/-- Ordered public-projection program. -/
def rows {shape : Phi81Relation.Shape}
    (frame : Frame shape) : List OwnedRow :=
  List.ofFn (row frame)

@[simp] theorem rows_length {shape : Phi81Relation.Shape}
    (frame : Frame shape) :
    (rows frame).length = shape.publicWidth := by
  simp [rows]

/-- The projection slice allocates no columns. -/
def columns {shape : Phi81Relation.Shape}
    (_frame : Frame shape) : List OwnedColumn :=
  []

@[simp] theorem columns_length {shape : Phi81Relation.Shape}
    (frame : Frame shape) :
    (columns frame).length = 0 :=
  rfl

theorem columnIds_nodup {shape : Phi81Relation.Shape}
    (frame : Frame shape) :
    ((columns frame).map fun column => column.id).Nodup := by
  simp [columns]

private theorem nodup_ofFn_of_injective
    {alpha : Type} :
    ∀ {n : Nat}
      (function : Fin n → alpha),
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

/-- Every emitted row has a distinct physical occurrence ID. -/
theorem rowIds_nodup {shape : Phi81Relation.Shape}
    (frame : Frame shape) :
    ((rows frame).map fun owned => owned.id).Nodup := by
  rw [rows, List.map_ofFn]
  apply nodup_ofFn_of_injective
  intro first second equal
  apply Fin.ext
  have ordinalEqual :=
    congrArg (fun id : RowId => id.ordinal) equal
  exact Nat.add_left_cancel ordinalEqual

/-- Every projection occurrence has the declared structural owner. -/
theorem rows_owned {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (owned : OwnedRow)
    (member : owned ∈ rows frame) :
    owned.id.owner = frame.owner := by
  rcases List.mem_ofFn.mp member with ⟨coordinate, rfl⟩
  rfl

/-- A projection row can mention only the constant wire, the selected
witness coordinate, and the selected public coordinate. -/
theorem row_supported {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (coordinate : Fin shape.publicWidth)
    (column : ColumnId)
    (mentioned : column ∈ (row frame coordinate).columnIds) :
    column = frame.one ∨
      column = frame.witness (shape.publicColumn coordinate) ∨
        column = frame.publicColumn coordinate := by
  simpa [row, Atoms.linearCheckOwnedRow, Atoms.linearCheckRow,
    OwnedRow.columnIds, Row.columnIds, Goldilocks.singleton, or_assoc,
    or_left_comm, or_comm] using mentioned

/-- Satisfaction of an `ofFn` row family is pointwise satisfaction. -/
private theorem satisfies_ofFn_iff :
    ∀ {count : Nat}
      (function : Fin count → OwnedRow)
      (assignment : ColumnId → F),
      Satisfies (List.ofFn function) assignment ↔
        ∀ coordinate, (function coordinate).row.Holds assignment
  | 0, function, assignment => by
      simp
  | _ + 1, function, assignment => by
      rw [List.ofFn_succ, satisfies_cons,
        satisfies_ofFn_iff (fun index => function index.succ) assignment]
      constructor
      · rintro ⟨head, tail⟩ coordinate
        exact Fin.cases head tail coordinate
      · intro every
        exact ⟨every 0, fun index => every index.succ⟩

/-- A projection program is satisfied exactly when each generated row is
satisfied. -/
private theorem satisfies_rows_iff {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (assignment : ColumnId → F) :
    Satisfies (rows frame) assignment ↔
      ∀ coordinate,
        (row frame coordinate).row.Holds assignment := by
  exact satisfies_ofFn_iff (row frame) assignment

/-- Physical satisfaction binds all public columns to the authoritative
projection of the complete witness columns. -/
theorem rows_sound {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (assignment : ColumnId → F)
    (constantOne : assignment frame.one = 1)
    (satisfied : Satisfies (rows frame) assignment) :
    (fun coordinate => assignment (frame.publicColumn coordinate)) =
      Phi81Relation.projectPublicInput
        (fun coordinate => assignment (frame.witness coordinate)) := by
  funext coordinate
  have holds :=
    (satisfies_rows_iff frame assignment).mp satisfied coordinate
  have equality :=
    (Atoms.linearCheckRow_iff assignment frame.one
      (Goldilocks.singleton
        (frame.witness (shape.publicColumn coordinate)) 1)
      (Goldilocks.singleton (frame.publicColumn coordinate) 1)
      constantOne).mp holds
  simpa [Phi81Relation.projectPublicInput, Goldilocks.singleton,
    LinearCombination.eval, Fin.one_mul, Fin.add_zero] using equality.symm

/-- If the public columns contain the authoritative projection, every
projection row is satisfied. -/
theorem rows_honest {shape : Phi81Relation.Shape}
    (frame : Frame shape)
    (assignment : ColumnId → F)
    (constantOne : assignment frame.one = 1)
    (projectionMatches :
      (fun coordinate => assignment (frame.publicColumn coordinate)) =
        Phi81Relation.projectPublicInput
          (fun coordinate => assignment (frame.witness coordinate))) :
    Satisfies (rows frame) assignment := by
  apply (satisfies_rows_iff frame assignment).mpr
  intro coordinate
  apply
    (Atoms.linearCheckRow_iff assignment frame.one
      (Goldilocks.singleton
        (frame.witness (shape.publicColumn coordinate)) 1)
      (Goldilocks.singleton (frame.publicColumn coordinate) 1)
      constantOne).mpr
  have pointwise := congrFun projectionMatches coordinate
  simpa [Phi81Relation.projectPublicInput, Goldilocks.singleton,
    LinearCombination.eval, Fin.one_mul, Fin.add_zero] using pointwise.symm

/-- Exact local cost for one public-projection program. -/
def cost (shape : Phi81Relation.Shape) : Cost :=
  ⟨shape.publicWidth, 0, 0, 0⟩

@[simp] theorem cost_rows {shape : Phi81Relation.Shape}
    (frame : Frame shape) :
    (rows frame).length = (cost shape).recurringRows :=
  rows_length frame

@[simp] theorem cost_auxiliary (shape : Phi81Relation.Shape) :
    (cost shape).auxiliaryColumns = 0 :=
  rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.TerminalR1cs.Projection
