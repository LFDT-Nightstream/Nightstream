import NightstreamFPrime.Export.Stage1.Wide.AssignmentProjection

/-! A proof view of a compact assignment at retained reference addresses.
Removed addresses have no map and cannot occur in a certified reused plan. -/

namespace NightstreamFPrime.Export.Stage1.Wide.AssignmentPullback

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

abbrev Program := RetainedLayout.Program

local instance (program : Program) (source : Nat) : Decidable (RetainedLayout.Live program source) := by
  unfold RetainedLayout.Live
  infer_instance

def assignment (program : Program) (after : Assignment F (RetainedLayout.logicalWidth program)) :
    Assignment F (PerApplicationFixedPoint.logicalWidth program) := fun source =>
  if live : RetainedLayout.Live program source.val then after (RetainedLayout.column program source live) else 0

theorem at_live (program : Program) (after : Assignment F (RetainedLayout.logicalWidth program))
    (source : Fin (PerApplicationFixedPoint.logicalWidth program)) (live : RetainedLayout.Live program source.val) :
    assignment program after source = after (RetainedLayout.column program source live) := by
  exact dif_pos live

theorem form_eval (program : Program) (after : Assignment F (RetainedLayout.logicalWidth program))
    (form : SparseForm (PerApplicationFixedPoint.logicalWidth program)) (supported : ReadSupport.Form program form) :
    (RetainedLayout.renameForm program form supported).eval after = form.eval (assignment program after) := by
  apply SparseForm.mapColumnsChecked_eval
  intro source live
  exact (at_live program after source live).symm

theorem rowsZero_iff (program : Program) (after : Assignment F (RetainedLayout.logicalWidth program))
    (plan : ProductionRelation.Plan (PerApplicationFixedPoint.logicalWidth program))
    (supported : ReadSupport.Plans program plan) :
    (Stage1Plan.rename program plan supported).RowsZero after ↔ plan.RowsZero (assignment program after) := by
  apply Plan.mapColumnsChecked_rowsZero_iff
  intro source live
  exact (at_live program after source live).symm

end NightstreamFPrime.Export.Stage1.Wide.AssignmentPullback
