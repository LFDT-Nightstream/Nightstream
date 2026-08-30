import NightstreamFPrime.Export.Stage1.PiRLCFirst54DirectBridge
import NightstreamFPrime.Export.Stage1.PiRLCRetainedPreservation

/-!
Owns the ordered direct PiRLC plan over the canonical retained assignment.
Product-family rows precede First54 rows. The plan has exact soundness and
completeness against the existing PiRLC source constraints and sampler
semantics.

This module does not compose Poseidon2 rows or the other Stage 1 phases.
-/

namespace NightstreamFPrime.Export.Stage1.PiRLCRetainedPlan

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Gadgets.Sampling
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle.PiRLC.v1_1
open PiRLCRetainedGeometry
open PiRLCRetainedInputs
open PiRLCRetainedPreservation

def rowCount : Nat := 1654236 + 119697

@[simp] theorem rowCount_eq : rowCount = 1773933 := by
  rfl

theorem childRowCount_le {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    (PiRLCProductPlan.plan (productInputs geometry)).rowCount +
        (PiRLCFirst54DirectPlan.plan (first54Inputs geometry)).rowCount ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  rw [PiRLCProductPlan.plan_rowCount,
    PiRLCFirst54DirectPlan.plan_rowCount]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

/-- Product rows followed by First54 rows in one canonical direct plan. -/
def plan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.append
    (PiRLCProductPlan.plan (productInputs geometry))
    (PiRLCFirst54DirectPlan.plan (first54Inputs geometry))
    (childRowCount_le geometry)

@[simp] theorem plan_rowCount {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    (plan geometry).rowCount = 1773933 := by
  simp [plan]

/-- The combined plan vanishes exactly when both canonical child plans
vanish. -/
theorem rowsZero_iff {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) :
    (plan geometry).RowsZero assignment ↔
      (PiRLCProductPlan.plan (productInputs geometry)).RowsZero assignment ∧
        (PiRLCFirst54DirectPlan.plan
          (first54Inputs geometry)).RowsZero assignment := by
  exact ProductionRelation.Plan.append_rowsZero_iff _ _
    (childRowCount_le geometry) assignment

/-- Exact semantic content forced by the combined direct PiRLC plan. -/
structure Semantics (program : Lifecycle.Stage1.Application.Program)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F) : Prop where
  product : ∀ invocation,
    (PiRLCProductSchedule.descriptor invocation).sourceConstraint.eval
      (PiRLCProductPlan.baseEnv program base) = 0
  first54 : ∀ source,
    PiRLCFirst54DirectPlan.SourceHolds program base source

/-- Zero combined rows force every canonical PiRLC product constraint and
every First54 transition. -/
theorem rowsZero_implies_semantics
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (products : Fin PiRLCFirst54DirectSchedule.candidateCount → F)
    (one : assignment (oneColumn geometry) = 1)
    (encodes : Encodes geometry assignment base groupValue products)
    (rowsZero : (plan geometry).RowsZero assignment) :
    Semantics program base := by
  have children := (rowsZero_iff geometry assignment).mp rowsZero
  have productPreserves := productInputs_preserves geometry assignment base
    groupValue products encodes
  have first54Preserves := first54Inputs_preserves geometry assignment base
    groupValue products one encodes
  refine ⟨?_, ?_⟩
  · intro invocation
    exact PiRLCProductPlan.rowsZero_implies_sourceConstraint
      (productInputs geometry) assignment base groupValue one
        productPreserves children.1 invocation
  · intro source
    exact PiRLCFirst54DirectPlan.rowsZero_implies_sourceHolds
      (first54Inputs geometry) assignment base products one
        first54Preserves children.2 source

/-- The exact source semantics and honest retained intermediate values make
every combined direct PiRLC row vanish. -/
theorem semantics_implies_rowsZero
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (one : assignment (oneColumn geometry) = 1)
    (encodes : Encodes geometry assignment base
      (PiRLCProductPlan.honestGroupValue
        (productInputs geometry) assignment)
      (PiRLCFirst54DirectPlan.honestProducts program base))
    (semantics : Semantics program base) :
    (plan geometry).RowsZero assignment := by
  apply (rowsZero_iff geometry assignment).mpr
  constructor
  · exact PiRLCProductPlan.sourceConstraints_imply_rowsZero
      (productInputs geometry) assignment base one
        (productInputs_preserves geometry assignment base _ _ encodes)
        semantics.product
  · exact PiRLCFirst54DirectPlan.sourceHolds_imply_rowsZero
      (first54Inputs geometry) assignment base one
        (first54Inputs_preserves geometry assignment base _ _ one encodes)
        semantics.first54

/-- With honest retained intermediates, combined row acceptance is exactly
the direct PiRLC semantics. -/
theorem rowsZero_iff_semantics
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (one : assignment (oneColumn geometry) = 1)
    (encodes : Encodes geometry assignment base
      (PiRLCProductPlan.honestGroupValue
        (productInputs geometry) assignment)
      (PiRLCFirst54DirectPlan.honestProducts program base)) :
    (plan geometry).RowsZero assignment ↔ Semantics program base := by
  constructor
  · exact rowsZero_implies_semantics geometry assignment base _ _ one encodes
  · exact semantics_implies_rowsZero geometry assignment base one encodes

/-- Combined zero rows retain the established high-level First54 sampler
relation for every canonical source. -/
theorem rowsZero_implies_relationHolds
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (base : Fin (PiRLCProductPlan.baseSourceWidth program) → F)
    (groupValue : Fin PiRLCProductSchedule.invocationCount → Fin 33 → F)
    (interface : Sampler.Interface) (coordinate : Nat)
    (one : assignment (oneColumn geometry) = 1)
    (encodes : Encodes geometry assignment base groupValue
      (PiRLCFirst54DirectPlan.honestProducts program base))
    (rowsZero : (plan geometry).RowsZero assignment)
    (source : Fin PiRLCFirst54DirectSchedule.sourceCount)
    (assumptions : First54.Assumptions
      (Sampler.selectorInterface interface coordinate
        (PiRLCFirst54DirectBridge.samplerStart source))
      (PiRLCFirst54DirectBridge.selectorStart source)
      (PiRLCFirst54DirectPlan.baseEnv program base)) :
    First54.RelationHolds
      (Sampler.selectorInterface interface coordinate
        (PiRLCFirst54DirectBridge.samplerStart source))
      (PiRLCFirst54DirectBridge.selectorStart source)
      (PiRLCFirst54DirectPlan.baseEnv program base) := by
  have children := (rowsZero_iff geometry assignment).mp rowsZero
  exact PiRLCFirst54DirectBridge.rowsZero_implies_relationHolds
    (first54Inputs geometry) assignment base interface coordinate one
      (first54Inputs_preserves geometry assignment base groupValue _ one encodes)
      children.2 source assumptions

end NightstreamFPrime.Export.Stage1.PiRLCRetainedPlan
