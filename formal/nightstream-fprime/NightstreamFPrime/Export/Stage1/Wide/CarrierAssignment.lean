import NightstreamFPrime.Export.Stage1.Wide.AssignmentNorm
import NightstreamFPrime.Export.Stage1.Wide.FixedPoint

/-! Complete the constructed logical witness to the candidate's committed
carrier. Padding is zero, the norm is below two, and the public digest is exact. -/

namespace NightstreamFPrime.Export.Stage1.Wide.CarrierAssignment

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle NightstreamFPrime.Lifecycle.PaperAlgebra
open Spec.Folding.PiCCS.PaperJoint ProductionRelation

def values (program : RetainedLayout.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F) :
    Lifecycle.PaperAlgebra.Assignment
      (logicalWidth := RetainedLayout.logicalWidth program) (publicFits := FixedPoint.publicFits program) :=
  Phi81CarrierLayout.extendAssignment (0 : F) (SourceAssignment.assignment program env application)

theorem logical_value (program : RetainedLayout.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (column : Fin (RetainedLayout.logicalWidth program)) :
    values program env application (Phi81CarrierLayout.embedLogical column) =
      SourceAssignment.assignment program env application column :=
  Phi81CarrierLayout.extendAssignment_embedLogical _ _ column

theorem padding_zero (program : RetainedLayout.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (column : Fin (Phi81CarrierLayout.carrierWidth (RetainedLayout.logicalWidth program)))
    (padding : RetainedLayout.logicalWidth program ≤ column.val) :
    values program env application column = 0 :=
  Phi81CarrierLayout.extendAssignment_tail_zero _ _ column padding

private theorem extension_norm {logicalWidth : Nat}
    (assignment : Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra.Assignment F logicalWidth)
    (bounded : ∀ column, centeredMagnitude (assignment column) < 2) :
    ∀ column, centeredMagnitude (Phi81CarrierLayout.extendAssignment (0 : F) assignment column) < 2 := by
  intro column
  unfold Phi81CarrierLayout.extendAssignment
  cases found : Phi81CarrierLayout.logicalColumn? column with
  | some logical => exact bounded logical
  | none =>
    rw [Phi81Relation.PiRLCAlgebra.Norm.Centered.centeredMagnitude_zero]
    decide

theorem norm (program : RetainedLayout.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    {width : Nat} {fits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth width}
    (relation : ProductionKey.LogicalRelation width fits)
    (running : R1CS.RowsHold env (Layout.Stage1.Wide.RunningTransitionLayout.physicalRows width fits)) :
    ∀ column, centeredMagnitude (values program env application column) < 2 :=
  extension_norm _ (AssignmentNorm.assignment_norm program env application relation running)

/-- Every public coordinate is copied unchanged through projection and PiRLC. -/
theorem public_value (program : RetainedLayout.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F)
    (column : Fin ProductionAssignment.publicWidth) :
    (Phi81Relation.projectPublicInput (shape := FullShape (RetainedLayout.logicalWidth program) (FixedPoint.publicFits program)) (values program env application)) column =
      encodedHashCells (SourceAssignment.raw program env application).outputDigest column := by
  have before : column.val < RetainedLayout.logicalWidth program := by
    have bound : column.val < 270 := column.isLt
    rw [RetainedLayout.logicalWidth_eq]
    omega
  let source : Fin (PerApplicationFixedPoint.logicalWidth program) :=
    ⟨column.val, lt_of_lt_of_le column.isLt (PerApplicationCanonicalAssignment.publicFits (application := program))⟩
  have copied : CoordinateRecovery.CommonSource program source.val := by
    left
    rw [(RetainedLayout.boundaries program).1]
    have bound : column.val < 270 := column.isLt
    dsimp only [source]
    omega
  have same := AssignmentProjection.assignment_copied program
    (SourceAssignment.raw program env application).assignment source copied
  have columnEq : RetainedLayout.column program source
      (CoordinateRecovery.commonSource_live program source.val copied) = ⟨column.val, before⟩ := by
    apply Fin.ext
    exact RetainedLayout.column_of_some program source _ column.val
      (RetainedLayout.publicColumn program column.val column.isLt)
  rw [columnEq] at same
  have original := CanonicalBlockAssignment.assignment_publicColumn
    (encodedHashCells (SourceAssignment.raw program env application).outputDigest)
    (SourceAssignment.raw program env application).schedule (PerApplicationCanonicalAssignment.publicFits (application := program)) column
  change values program env application (Phi81CarrierLayout.embedLogical ⟨column.val, before⟩) = _
  rw [logical_value]
  exact same.trans original

theorem publicInput (program : RetainedLayout.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F) :
    Phi81Relation.projectPublicInput (shape := FullShape (RetainedLayout.logicalWidth program) (FixedPoint.publicFits program)) (values program env application) =
      encHash (publicFits := FixedPoint.publicFits program) (SourceAssignment.raw program env application).outputDigest := by
  funext column
  exact public_value program env application column

/-- The exposed public digest is the physical pilot output, not a separate input. -/
theorem publicOutput (program : RetainedLayout.Program) (env : Env)
    (application : Fin (PerApplicationPackage.addedPrivateColumnCount program) → F) :
    Phi81Relation.projectPublicInput (shape := FullShape (RetainedLayout.logicalWidth program) (FixedPoint.publicFits program))
        (values program env application) =
      encHash (publicFits := FixedPoint.publicFits program) (List.ofFn
        (fun lane : Fin Layout.PilotProduction.digestWords => env (Layout.PilotProduction.outputDigestStart + lane.val))) := by
  rw [publicInput, SourceAssignment.outputDigest]

end NightstreamFPrime.Export.Stage1.Wide.CarrierAssignment
