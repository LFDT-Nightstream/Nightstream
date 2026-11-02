import NightstreamFPrime.Export.Generic.LogicalPlan
import NightstreamFPrime.Layout.ProductionRelation.RetainedSlot

/-! The concrete layout of a step circuit's source columns (CW-5.11): the public columns first,
then one low-norm slot (`LowNormAssignment.Slot`) for each live source column, as today's
production assignment places them (`ProductionAssignment.logicalAssignment`). A slot's form
recomposes its digits (`RetainedSlot.form`). A source column without a slot, such as a dead
Poseidon2 local, maps to the empty form. The package lists the slots as runs: a start column, a
count and a kind (`runSlots`). -/

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

end NightstreamFPrime.Export.Generic.SlotLayout
