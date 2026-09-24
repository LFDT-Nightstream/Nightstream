import NightstreamFPrime.Export.Stage1.Wide.CoordinateRecovery
import NightstreamFPrime.Export.Stage1.Wide.Stage1Witness
import NightstreamFPrime.Export.Stage1.Wide.PiRLCWitnessCongruence

/-! Read-only projection of the retained reference coordinates, followed by
the direct wide PiRLC constructor. No reference array is allocated. Source
completion for the reused phases remains a separate physical-layout proof. -/

namespace NightstreamFPrime.Export.Stage1.Wide.AssignmentProjection

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

abbrev Program := RetainedLayout.Program

/-- Only retained sources can be queried. Unmapped new coordinates start at
zero and are subsequently filled by the direct PiRLC constructor. -/
def project (program : Program) (before : Assignment F (PerApplicationFixedPoint.logicalWidth program)) :
    Assignment F (RetainedLayout.logicalWidth program) := fun target =>
  match found : CoordinateRecovery.source? program target.val with
  | some source => before ⟨source, CoordinateRecovery.source?_lt program target.val source found⟩
  | none => 0

theorem project_at (program : Program) (before : Assignment F (PerApplicationFixedPoint.logicalWidth program))
    (source : Fin (PerApplicationFixedPoint.logicalWidth program)) (live : RetainedLayout.Live program source.val) :
    project program before (RetainedLayout.column program source live) = before source := by
  have recovered := CoordinateRecovery.source?_column? program source.val _ (RetainedLayout.column_mapped program source live)
  unfold project
  split
  · rename_i result found
    apply congrArg before
    apply Fin.ext
    exact Option.some.inj (found.symm.trans recovered)
  · rename_i found
    rw [recovered] at found
    contradiction

/-- Changes to any removed coordinate are unobservable to this constructor. -/
theorem project_agree (program : Program)
    (before after : Assignment F (PerApplicationFixedPoint.logicalWidth program))
    (agrees : ∀ source, RetainedLayout.Live program source.val → before source = after source) :
    project program before = project program after := by
  funext target
  unfold project
  split
  · rename_i source found
    exact agrees _ (CoordinateRecovery.source?_live program target.val source found)
  · rfl

theorem project_form (program : Program)
    (before : Assignment F (PerApplicationFixedPoint.logicalWidth program))
    (form : SparseForm (PerApplicationFixedPoint.logicalWidth program)) (supported : ReadSupport.Form program form) :
    (RetainedLayout.renameForm program form supported).eval (project program before) = form.eval before :=
  SparseForm.mapColumnsChecked_eval _ _ _ before (project program before) (project_at program before)

/-- All six reused phases have this exact row-equivalence law. It applies to
any reference assignment, without an assumption that its rows are satisfied. -/
theorem project_rowsZero_iff (program : Program)
    (before : Assignment F (PerApplicationFixedPoint.logicalWidth program))
    (plan : ProductionRelation.Plan (PerApplicationFixedPoint.logicalWidth program))
    (supported : ReadSupport.Plans program plan) :
    (Stage1Plan.rename program plan supported).RowsZero (project program before) ↔ plan.RowsZero before :=
  Plan.mapColumnsChecked_rowsZero_iff _ _ _ before (project program before) (project_at program before)

theorem project_norm (program : Program)
    (before : Assignment F (PerApplicationFixedPoint.logicalWidth program))
    (bounded : ∀ source, RetainedLayout.Live program source.val → centeredMagnitude (before source) < 2) :
    ∀ target, centeredMagnitude (project program before target) < 2 := by
  intro target
  unfold project
  split
  · rename_i source found
    exact bounded _ (CoordinateRecovery.source?_live program target.val source found)
  · rw [Phi81Relation.PiRLCAlgebra.Norm.Centered.centeredMagnitude_zero]
    decide

/-- Copy the common prefix only. Old sampler, product-output and quotient
values are never queried when constructing the direct wide witness. -/
def seed (program : Program) (before : Assignment F (PerApplicationFixedPoint.logicalWidth program)) :
    Assignment F (RetainedLayout.logicalWidth program) := fun target =>
  if target.val < RetainedLayout.commonCount program then project program before target else 0

theorem seed_before (program : Program) (before : Assignment F (PerApplicationFixedPoint.logicalWidth program))
    (target : Fin (RetainedLayout.logicalWidth program)) (below : target.val < RetainedLayout.commonCount program) :
    seed program before target = project program before target := by
  exact if_pos below

theorem seed_agree (program : Program)
    (before after : Assignment F (PerApplicationFixedPoint.logicalWidth program))
    (agrees : ∀ source, CoordinateRecovery.CommonSource program source.val → before source = after source) :
    seed program before = seed program after := by
  funext target
  unfold seed
  split
  · rename_i below
    unfold project
    split
    · rename_i source found
      exact agrees _ (CoordinateRecovery.source?_common program target.val source below found)
    · rfl
  · rfl

theorem seed_norm (program : Program)
    (before : Assignment F (PerApplicationFixedPoint.logicalWidth program))
    (bounded : ∀ source, CoordinateRecovery.CommonSource program source.val → centeredMagnitude (before source) < 2) :
    ∀ target, centeredMagnitude (seed program before target) < 2 := by
  intro target
  unfold seed
  split
  · rename_i below
    unfold project
    split
    · rename_i source found
      exact bounded _ (CoordinateRecovery.source?_common program target.val source below found)
    · rw [Phi81Relation.PiRLCAlgebra.Norm.Centered.centeredMagnitude_zero]; decide
  · rw [Phi81Relation.PiRLCAlgebra.Norm.Centered.centeredMagnitude_zero]; decide

def assignment (program : Program) (before : Assignment F (PerApplicationFixedPoint.logicalWidth program)) :=
  Stage1Witness.assignment program (seed program before)

/-- Omitting all overwritten reference values gives exactly the same final
witness as projecting them. This is a value identity, not just row acceptance. -/
theorem assignment_eq_project (program : Program)
    (before : Assignment F (PerApplicationFixedPoint.logicalWidth program)) :
    assignment program before = Stage1Witness.assignment program (project program before) := by
  apply PiRLCWitness.assignment_eq_of_agrees_outside _ _ _ (InputSupport.inputsBefore program)
  intro target outside
  change target.val < RetainedLayout.commonCount program ∨ RetainedLayout.logicalWidth program ≤ target.val at outside
  exact seed_before program before target (outside.resolve_right (Nat.not_le.mpr target.isLt))

theorem piRlc_complete (program : Program) (compiled : PiRlcWideSampler.RangePlan.Compiled)
    (before : Assignment F (PerApplicationFixedPoint.logicalWidth program))
    (one : before (ApplicationRetainedGeometry.oneColumn (Stage1Plan.referenceGeometry program)) = 1) :
    (Stage1Plan.piRlc program compiled).RowsZero (assignment program before) := by
  apply Stage1Witness.complete
  calc
    seed program before (Stage1Plan.piRlcInterface program).oneColumn =
        project program before (Stage1Plan.piRlcInterface program).oneColumn :=
      seed_before program before _ (InputSupport.inputsBefore program).sampler.one
    _ = before (ApplicationRetainedGeometry.oneColumn (Stage1Plan.referenceGeometry program)) :=
      project_at program before _ (ReadSupport.one program _ rfl)
    _ = 1 := one

theorem assignment_agree (program : Program)
    (before after : Assignment F (PerApplicationFixedPoint.logicalWidth program))
    (agrees : ∀ source, CoordinateRecovery.CommonSource program source.val → before source = after source) :
    assignment program before = assignment program after :=
  congrArg (Stage1Witness.assignment program) (seed_agree program before after agrees)

theorem assignment_norm (program : Program)
    (before : Assignment F (PerApplicationFixedPoint.logicalWidth program))
    (bounded : ∀ source, CoordinateRecovery.CommonSource program source.val → centeredMagnitude (before source) < 2) :
    ∀ target, centeredMagnitude (assignment program before target) < 2 :=
  PiRLCWitness.preserves_norm _ _ (seed_norm program before bounded)

end NightstreamFPrime.Export.Stage1.Wide.AssignmentProjection
