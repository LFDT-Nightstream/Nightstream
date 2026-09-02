import NightstreamFPrime.Export.Stage1.RunningTransitionArithmetic
import NightstreamFPrime.Layout.ProductionRelation.OrdinarySourcePlan
import NightstreamFPrime.Layout.Stage1.RunningTransitionPreservation
import NightstreamFPrime.Layout.Stage1.SpartanBounds

/-!
Owns the indexed source-row program for the Stage 1 running transition. Its
rows are exactly the canonical Lean lowering after the established Spartan
permutation.

This module does not select final retained coordinates or compile sparse
forms over the final low-norm assignment.
-/

namespace NightstreamFPrime.Export.Stage1.RunningTransitionDirectSource

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

/-- Canonical running-transition source rows in the established Spartan
column order. -/
def sourceRows
    (logicalWidth : Nat)
    (publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth) : List R1CS.Row :=
  Spartan.remapRows
    (RunningTransitionLayout.physicalRows logicalWidth publicFits)

theorem sourceRows_length
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (sourceRows logicalWidth publicFits).length = 345495 := by
  rw [sourceRows, Spartan.remapRows, List.length_map,
    RunningTransitionLayout.physicalRows_length,
    RunningTransitionLayout.physicalRowCount_eq relation]

/-- Every remapped row is confined to the exact Spartan source domain. -/
theorem sourceRows_varsBelow
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    ∀ row ∈ sourceRows logicalWidth publicFits,
      row.VarsBelow Spartan.spartanColumnCount := by
  apply Spartan.remapRows_varsBelow
  intro row member
  have scope := RunningTransitionLayout.physicalRows_varsBelow relation
    row member
  rw [RunningTransitionLayout.physicalColumnCount_eq relation] at scope
  exact scope

theorem sourceRows_rowCount_le
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (sourceRows logicalWidth publicFits).length ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  rw [sourceRows_length relation]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

/-- Proof-oriented indexed access to the exact canonical source rows. -/
def program
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    OrdinarySourcePlan.Program Spartan.spartanColumnCount where
  rowCount := (sourceRows logicalWidth publicFits).length
  rowCount_le := sourceRows_rowCount_le relation
  row := fun index => (sourceRows logicalWidth publicFits).get index
  bounded := fun index =>
    sourceRows_varsBelow relation _ (List.get_mem _ index)

@[simp] theorem program_rowCount
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits) :
    (program relation).rowCount = 345495 := by
  exact sourceRows_length relation

/-- The indexed transition program depends on relation shape only. Matrix
entries are semantic inputs to the verifier, not inputs to row generation. -/
theorem program_eq_of_same_shape
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (left right : ProductionKey.LogicalRelation logicalWidth publicFits) :
    program left = program right := by
  rfl

/-- The indexed program is the same ordered row list as the existing
canonical package compiler. -/
theorem sourceRows_eq_canonicalRows
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth} :
    sourceRows logicalWidth publicFits =
      (RunningTransitionArithmetic.canonicalPlan logicalWidth publicFits).rows.map
        Rows.CompiledRow.toR1CS := by
  symm
  simpa [sourceRows, RunningTransitionArithmetic.canonicalLayoutPlan,
    RunningTransitionLayout.physicalRows] using
    (RunningTransitionArithmetic.Plan.rows_to_layout
      (RunningTransitionArithmetic.canonicalPlan logicalWidth publicFits)
      (RunningTransitionArithmetic.canonicalLayoutPlan logicalWidth publicFits)
      (RunningTransitionArithmetic.canonicalPlan_matches logicalWidth
        publicFits))

/-- Indexed source-row satisfaction is exactly list-based R1CS
satisfaction. -/
theorem program_holds_iff_rowsHold
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) :
    (program relation).Holds env ↔
      R1CS.RowsHold env (sourceRows logicalWidth publicFits) := by
  change (∀ index, ((sourceRows logicalWidth publicFits).get index).Holds env) ↔
    R1CS.RowsHold env (sourceRows logicalWidth publicFits)
  constructor
  · intro holds row member
    rcases List.mem_iff_get.mp member with ⟨index, rfl⟩
    exact holds index
  · intro holds index
    exact holds _ (List.get_mem _ index)

/-- The indexed source program accepts exactly the existing physical running
transition after the established Spartan pullback. -/
theorem program_holds_iff_physical
    {logicalWidth : Nat}
    {publicFits : ringDegree * publicRingColumns ≤
      Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (env : Env) :
    (program relation).Holds env ↔
      RunningTransitionLayout.PhysicalHolds logicalWidth publicFits
        (Spartan.pullback env) := by
  rw [program_holds_iff_rowsHold relation]
  simpa [sourceRows, RunningTransitionLayout.PhysicalHolds] using
    (Spartan.remapRows_hold env
      (RunningTransitionLayout.physicalRows logicalWidth publicFits))

end NightstreamFPrime.Export.Stage1.RunningTransitionDirectSource
