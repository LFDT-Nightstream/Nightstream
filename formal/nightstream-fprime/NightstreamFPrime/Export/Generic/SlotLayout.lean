import NightstreamFPrime.Export.Generic.LogicalPlan
import NightstreamFPrime.Layout.ProductionRelation.RetainedSlot

/-! The concrete layout of a step circuit's source columns (CW-5.11): the public columns first,
then one low-norm slot (`LowNormAssignment.Slot`) for each source column that has one, as today's
production assignment places them (`ProductionAssignment.logicalAssignment`). A slot's form
recomposes its digits (`RetainedSlot.form`). A source column without a slot, such as a dead
Poseidon2 local or a column that no row reads, maps to the empty form. The package lists the
slots as runs: a start column, a count and a kind (`runSlots`). -/

namespace NightstreamFPrime.Export.Generic.SlotLayout

open NightstreamFPrime.Spec
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-- The slot of source column `column`, if it has one. -/
def slotIndex {width : Nat} (slots : List (LowNormAssignment.Slot width)) (column : Fin width) :
    Option (Fin slots.length) :=
  (List.finRange slots.length).find? fun index => (slots.get index).source = column

theorem publicFits {width : Nat} (slots : List (LowNormAssignment.Slot width)) :
    LogicalPlan.publicCount ≤ ProductionAssignment.logicalWidth slots := by
  simp only [ProductionAssignment.logicalWidth]
  exact Nat.le_add_right _ _

def layout {width : Nat} (slots : List (LowNormAssignment.Slot width)) :
    LogicalPlan.Layout width (ProductionAssignment.logicalWidth slots) where
  map := ⟨fun column =>
    if bound : column.val < LogicalPlan.publicCount then
      SparseForm.singleton ⟨column.val, Nat.lt_of_lt_of_le bound (publicFits slots)⟩ 1
    else
      match slotIndex slots column with
      | some index => RetainedSlot.form slots index
      | none => .empty⟩
  publicFits := publicFits slots
  publicForm := fun column bound => by simp only [bound, dite_true]

/-- The kind of a run: 0 bit, 1 centered, 2 field. -/
def kindOf : Nat → LowNormSlot.Kind
  | 0 => .bit
  | 1 => .centered
  | _ => .field

def kindCode : LowNormSlot.Kind → Nat
  | .bit => 0
  | .centered => 1
  | .field => 2

theorem kindOf_kindCode (kind : LowNormSlot.Kind) : kindOf (kindCode kind) = kind := by
  cases kind <;> rfl

/-- The slots of runs `(start, count, kind)`; a column at or beyond `width` has no slot. -/
def runSlots (width : Nat) (runs : List (Nat × Nat × Nat)) : List (LowNormAssignment.Slot width) :=
  runs.flatMap fun (start, count, kind) =>
    (List.range count).filterMap fun offset =>
      if bound : start + offset < width then some ⟨⟨start + offset, bound⟩, kindOf kind⟩ else none

/-! ### What the rows read

A package gives a column a slot only if a row of its relation reads the column, and a bit slot
only if a row forces the column to 0 or 1 (`forcedColumn?`). -/

/-- The columns that `expression` reads, with repeats. -/
def exprColumns : Circuit.Expr → List Nat
  | .var column => [column]
  | .const _ => []
  | .add left right => exprColumns left ++ exprColumns right
  | .mul left right => exprColumns left ++ exprColumns right

/-- The columns of a row's terms, with repeats. -/
def rowColumns (row : R1CS.Row) : List Nat :=
  (row.a.terms ++ row.b.terms ++ row.c.terms).map Prod.fst

/-- The sum of the coefficients of the terms of `side` at `column`. -/
def coefficientAt (side : R1CS.LinearCombination) (column : Nat) : F :=
  ((side.terms.filter fun term => term.1 = column).map Prod.snd).sum

/-- The column that `row` forces to 0 or 1: the terms of the row read one column `c` and the
constant column 0. With `x` at `c` and 1 at column 0, the row is `(a₀ + a₁x)(b₀ + b₁x) = c₀ + c₁x`
with `a₁b₁ ≠ 0`, and it holds at `x = 0` and at `x = 1`. -/
def forcedColumn? (row : R1CS.Row) : Option Nat :=
  let terms := row.a.terms ++ row.b.terms ++ row.c.terms
  match terms.find? (fun term => term.1 != 0) with
  | none => none
  | some (column, _) =>
      let low := fun side : R1CS.LinearCombination => side.constant + coefficientAt side 0
      let high := fun side : R1CS.LinearCombination => coefficientAt side column
      if terms.all (fun term => term.1 == 0 || term.1 == column) = true ∧
          high row.a * high row.b ≠ 0 ∧ low row.a * low row.b = low row.c ∧
          (low row.a + high row.a) * (low row.b + high row.b) = low row.c + high row.c then
        some column
      else none

/-- `mask` with `true` at each of `columns` below its size. -/
def markColumns (mask : Array Bool) (columns : List Nat) : Array Bool :=
  columns.foldl (fun mask column => mask.set! column true) mask

end NightstreamFPrime.Export.Generic.SlotLayout
