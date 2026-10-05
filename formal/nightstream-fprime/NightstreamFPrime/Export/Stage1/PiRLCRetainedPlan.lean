import NightstreamFPrime.Export.Stage1.PiRLCRetainedPreservation

/-! The direct Phi81 combination rows over the retained assignment.
The sampler's ordinary rows and Poseidon2 rows are owned by their own plans. -/

namespace NightstreamFPrime.Export.Stage1.PiRLCRetainedPlan

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open PiRLCRetainedGeometry PiRLCRetainedInputs PiRLCRetainedPreservation

def rowCount : Nat := 78948

@[simp] theorem rowCount_eq : rowCount = 78948 := rfl

def plan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (values : Values logicalWidth)
    (geometry : Geometry program logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  PiRLCProductPlan.plan (productInputs values geometry)

@[simp] theorem plan_rowCount {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (values : Values logicalWidth)
    (geometry : Geometry program logicalWidth) :
    (plan values geometry).rowCount = rowCount :=
  PiRLCProductPlan.plan_rowCount _

abbrev Semantics (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F) : Prop :=
  ∀ invocation,
    (PiRLCProductSchedule.descriptor invocation).sourceConstraint.eval
      (PiRLCProductPlan.baseEnv program base) = 0

theorem rowsZero_implies_semantics
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (values : Values logicalWidth)
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 1 → F)
    (one : assignment (oneColumn geometry) = 1)
    (valuePreserves : ∀ invocation,
      (values invocation).eval assignment =
        PiRLCProductPlan.baseEnv program base
          ((PiRLCProductSchedule.descriptor invocation).valueColumn
            (PiRLCProductSchedule.descriptor invocation).lane))
    (encodes : Encodes geometry assignment base groupValue)
    (rowsZero : (plan values geometry).RowsZero assignment) :
    Semantics program base := by
  exact PiRLCProductPlan.rowsZero_implies_sourceConstraint
    (productInputs values geometry) assignment base groupValue one
    (productInputs_preserves values geometry assignment base groupValue valuePreserves encodes)
    rowsZero

theorem semantics_implies_rowsZero
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (values : Values logicalWidth)
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (one : assignment (oneColumn geometry) = 1)
    (valuePreserves : ∀ invocation,
      (values invocation).eval assignment =
        PiRLCProductPlan.baseEnv program base
          ((PiRLCProductSchedule.descriptor invocation).valueColumn
            (PiRLCProductSchedule.descriptor invocation).lane))
    (encodes : Encodes geometry assignment base
      (PiRLCProductPlan.honestGroupValue (productInputs values geometry) assignment))
    (semantics : Semantics program base) :
    (plan values geometry).RowsZero assignment := by
  exact PiRLCProductPlan.sourceConstraints_imply_rowsZero
    (productInputs values geometry) assignment base one
    (productInputs_preserves values geometry assignment base _ valuePreserves encodes)
    semantics

theorem rowsZero_iff_semantics
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (values : Values logicalWidth)
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (one : assignment (oneColumn geometry) = 1)
    (valuePreserves : ∀ invocation,
      (values invocation).eval assignment =
        PiRLCProductPlan.baseEnv program base
          ((PiRLCProductSchedule.descriptor invocation).valueColumn
            (PiRLCProductSchedule.descriptor invocation).lane))
    (encodes : Encodes geometry assignment base
      (PiRLCProductPlan.honestGroupValue (productInputs values geometry) assignment)) :
    (plan values geometry).RowsZero assignment ↔ Semantics program base := by
  exact ⟨rowsZero_implies_semantics values geometry assignment base _ one valuePreserves encodes,
    semantics_implies_rowsZero values geometry assignment base one valuePreserves encodes⟩

end NightstreamFPrime.Export.Stage1.PiRLCRetainedPlan
