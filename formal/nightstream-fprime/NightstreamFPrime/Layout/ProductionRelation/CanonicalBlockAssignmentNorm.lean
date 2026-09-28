import NightstreamFPrime.Layout.ProductionRelation.CanonicalBlockAssignment

/-!
Owns the coordinate norm of the existing compact block assignment. Valid
slot sources and bounded public cells suffice. The proof follows coordinate
lookup through the block schedule; it creates no expanded slot or value list.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Layout.ProductionRelation.CanonicalBlockAssignment

open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

private theorem slotCoordinate_norm (kind : LowNormSlot.Kind) (value : F)
    (valid : LowNormSlot.Valid kind value) (index : Fin kind.width) :
    centeredMagnitude (LowNormSlot.coordinate kind value index) < 2 := by
  apply LowNormSlot.encode_norm kind value valid
  unfold LowNormSlot.coordinate
  have bound : index.val < (LowNormSlot.encode kind value).length := by
    simpa only [LowNormSlot.encode_length] using index.isLt
  rw [List.getD_eq_getElem (l := LowNormSlot.encode kind value) (d := 0) bound]
  exact List.getElem_mem bound

private theorem blockCoordinate_norm (entry : BlockValue)
    (valid : ∀ slot, LowNormSlot.Valid entry.block.kind
      (entry.source (entry.block.source slot))) (index : Nat) :
    centeredMagnitude (entry.coordinateAt index) < 2 := by
  unfold BlockValue.coordinateAt
  split
  · exact slotCoordinate_norm _ _ (valid _) _
  · simp

private theorem coordinateAt_norm (schedule : Schedule)
    (valid : ∀ entry ∈ schedule, ∀ slot, LowNormSlot.Valid entry.block.kind
      (entry.source (entry.block.source slot))) (index : Nat) :
    centeredMagnitude (coordinateAt schedule index) < 2 := by
  induction schedule generalizing index with
  | nil => simp [coordinateAt]
  | cons entry rest induction =>
      unfold coordinateAt
      split
      · exact blockCoordinate_norm entry (valid entry (List.mem_cons_self)) index
      · exact induction (fun next member => valid next (List.mem_cons_of_mem _ member)) _

/-- The canonical compact assignment is pointwise bounded when its public
coordinates and every retained source slot satisfy their existing contracts.
Coordinates outside the schedule are zero, so no coverage premise is needed. -/
theorem assignment_norm {logicalWidth : Nat}
    (publicInput : Fin ProductionAssignment.publicWidth → F)
    (schedule : Schedule)
    (publicNorm : ∀ column, centeredMagnitude (publicInput column) < 2)
    (valid : ∀ entry ∈ schedule, ∀ slot, LowNormSlot.Valid entry.block.kind
      (entry.source (entry.block.source slot)))
    (column : Fin logicalWidth) :
    centeredMagnitude (assignment publicInput schedule column) < 2 := by
  unfold assignment
  split
  · exact publicNorm _
  · exact coordinateAt_norm schedule valid _

end NightstreamFPrime.Layout.ProductionRelation.CanonicalBlockAssignment
