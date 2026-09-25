import NightstreamFPrime.Export.Stage1.PiRLCPoseidonGeometry
import NightstreamFPrime.Layout.ProductionRelation.PoseidonRetainedFamily

/-!
Owns the direct Poseidon2 plan for the two pilot hash chains. Each invocation
reads its retained preimage words and the prior invocation's closed-form
output. No recursive trace reconstruction or explicit invocation list is
used by the plan.

This module does not bind the final digests or select later phase
permutations.
-/

namespace NightstreamFPrime.Export.Stage1.PilotPoseidonPlan

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open PiRLCPoseidonGeometry

def invocationCount : Nat := PoseidonRetainedBlock.priorInvocationCount

@[simp] theorem invocationCount_eq : invocationCount = 12350 := by
  exact PoseidonRetainedBlock.priorInvocationCount_eq

def priorSchedule (program : Lifecycle.Stage1.Application.Program) :
    PoseidonRetainedFamily.Schedule (sourceWidth program) invocationCount where
  block := PiRLCRetainedGeometry.priorPoseidonBlock program
  slotCount_eq := by
    change
      (PiRLCRetainedGeometry.priorPoseidonBlock program).slotCount =
        invocationCount * PoseidonRetainedSlots.rows.length
    simp [PiRLCRetainedGeometry.priorPoseidonBlock,
      PoseidonRetainedBlock.priorBlock,
      Layout.ProductionRelation.PoseidonRetainedBlock.block_slotCount,
      invocationCount]

def outputSchedule (program : Lifecycle.Stage1.Application.Program) :
    PoseidonRetainedFamily.Schedule (sourceWidth program) invocationCount where
  block := PiRLCRetainedGeometry.outputPoseidonBlock program
  slotCount_eq := by
    change
      (PiRLCRetainedGeometry.outputPoseidonBlock program).slotCount =
        invocationCount * PoseidonRetainedSlots.rows.length
    simp [PiRLCRetainedGeometry.outputPoseidonBlock,
      PoseidonRetainedBlock.outputBlock,
      Layout.ProductionRelation.PoseidonRetainedBlock.block_slotCount,
      invocationCount, PoseidonRetainedBlock.priorInvocationCount,
      PoseidonRetainedBlock.outputInvocationCount]
    rfl

def previousOutput {sourceWidth invocationCount logicalWidth : Nat}
    (schedule : PoseidonRetainedFamily.Schedule sourceWidth invocationCount)
    (start : Nat)
    (fits : start + schedule.block.coordinateCount ≤ logicalWidth)
    (invocation : Fin invocationCount) : PoseidonSboxPlan.State logicalWidth :=
  if first : invocation.val = 0 then
    fun _ => .empty
  else
    PoseidonRetainedFamily.outputState schedule start fits
      ⟨invocation.val - 1, by
        have invocationBound := invocation.isLt
        omega⟩

def priorInputState {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth)
    (invocation : Fin invocationCount) :
    PoseidonSboxPlan.State logicalWidth :=
  fun lane =>
    let previous := previousOutput (priorSchedule program)
      (PiRLCRetainedGeometry.priorPoseidonStart program)
      (PiRLCRetainedGeometry.priorPoseidonFits (prefixGeometry geometry))
      invocation lane
    if invocation.val < Data.priorChain.absorbCount then
      if rateLane : lane.val < Spec.Poseidon2.rate then
        let offset := invocation.val * Spec.Poseidon2.rate + lane.val
        if present : offset < Data.priorChain.inputLength then
          SparseForm.add previous <|
            (priorInputBlock program).form
              (priorInputStart program) (priorInputFits geometry)
                ⟨offset, present⟩
        else
          previous
      else
        previous
    else if lane.val = 0 then
      SparseForm.add previous <|
        SparseForm.singleton (oneColumn geometry) 1
    else
      previous

def outputInputState {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth)
    (invocation : Fin invocationCount) :
    PoseidonSboxPlan.State logicalWidth :=
  fun lane =>
    let previous := previousOutput (outputSchedule program)
      (PiRLCRetainedGeometry.outputPoseidonStart program)
      (PiRLCRetainedGeometry.outputPoseidonFits (prefixGeometry geometry))
      invocation lane
    if invocation.val < Data.outputChain.absorbCount then
      if rateLane : lane.val < Spec.Poseidon2.rate then
        let offset := invocation.val * Spec.Poseidon2.rate + lane.val
        if present : offset < Data.outputChain.inputLength then
          SparseForm.add previous <|
            (outputInputBlock program).form
              (outputInputStart program) (outputInputFits geometry)
                ⟨offset, present⟩
        else
          previous
      else
        previous
    else if lane.val = 0 then
      SparseForm.add previous <|
        SparseForm.singleton (oneColumn geometry) 1
    else
      previous

def priorInterface {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    PoseidonSboxFamilyPlan.Interface logicalWidth invocationCount :=
  PoseidonRetainedFamily.familyInterface (priorSchedule program)
    (PiRLCRetainedGeometry.priorPoseidonStart program)
    (PiRLCRetainedGeometry.priorPoseidonFits (prefixGeometry geometry))
    (oneColumn geometry) (priorInputState geometry)

def outputInterface {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    PoseidonSboxFamilyPlan.Interface logicalWidth invocationCount :=
  PoseidonRetainedFamily.familyInterface (outputSchedule program)
    (PiRLCRetainedGeometry.outputPoseidonStart program)
    (PiRLCRetainedGeometry.outputPoseidonFits (prefixGeometry geometry))
    (oneColumn geometry) (outputInputState geometry)

theorem familyRowCount_le : invocationCount * 94 ≤
    2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  rw [invocationCount_eq]
  norm_num [NightstreamFPrime.Lifecycle.cubeVariables]

def priorPlan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  PoseidonSboxFamilyPlan.plan (priorInterface geometry) familyRowCount_le

def outputPlan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  PoseidonSboxFamilyPlan.plan (outputInterface geometry) familyRowCount_le

theorem combinedRowCount_le {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    (priorPlan geometry).rowCount + (outputPlan geometry).rowCount ≤
      2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  simp [priorPlan, outputPlan]
  norm_num [invocationCount_eq,
    NightstreamFPrime.Lifecycle.cubeVariables]

/-- Prior-hash invocations precede output-hash invocations. -/
def plan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  ProductionRelation.Plan.append (priorPlan geometry) (outputPlan geometry)
    (combinedRowCount_le geometry)

@[simp] theorem plan_rowCount {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth) :
    (plan geometry).rowCount = 2321800 := by
  simp [plan, priorPlan, outputPlan, invocationCount_eq]

theorem rowsZero_iff {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) :
    (plan geometry).RowsZero assignment ↔
      (priorPlan geometry).RowsZero assignment ∧
        (outputPlan geometry).RowsZero assignment := by
  exact ProductionRelation.Plan.append_rowsZero_iff _ _
    (combinedRowCount_le geometry) assignment

structure Semantics {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat} (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) : Prop where
  prior : ∀ invocation,
    List.ofFn (SparseLayer.evalState assignment
        ((priorInterface geometry).output invocation)) =
      Spec.Poseidon2.permute
        (List.ofFn (SparseLayer.evalState assignment
          ((priorInterface geometry).input invocation)))
  output : ∀ invocation,
    List.ofFn (SparseLayer.evalState assignment
        ((outputInterface geometry).output invocation)) =
      Spec.Poseidon2.permute
        (List.ofFn (SparseLayer.evalState assignment
          ((outputInterface geometry).input invocation)))

theorem rowsZero_implies_semantics
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (oneColumn geometry) = 1)
    (rowsZero : (plan geometry).RowsZero assignment) :
    Semantics geometry assignment := by
  have children := (rowsZero_iff geometry assignment).mp rowsZero
  refine ⟨?_, ?_⟩
  · intro invocation
    exact PoseidonSboxFamilyPlan.planRowsZero_implies_permute
      (priorInterface geometry) familyRowCount_le assignment one
        children.1 invocation
  · intro invocation
    exact PoseidonSboxFamilyPlan.planRowsZero_implies_permute
      (outputInterface geometry) familyRowCount_le assignment one
        children.2 invocation

/-- Honest retained S-box equations are sufficient for every pilot
Poseidon2 row. Output equations are automatic from the closed-form state. -/
theorem equations_imply_rowsZero
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (oneColumn geometry) = 1)
    (priorSboxes : ∀ invocation,
      PoseidonSboxPlan.SboxEquations
        (PoseidonSboxFamilyPlan.invocationInterface
          (priorInterface geometry) invocation) assignment)
    (outputSboxes : ∀ invocation,
      PoseidonSboxPlan.SboxEquations
        (PoseidonSboxFamilyPlan.invocationInterface
          (outputInterface geometry) invocation) assignment) :
    (plan geometry).RowsZero assignment := by
  apply (rowsZero_iff geometry assignment).mpr
  constructor
  · apply PoseidonSboxFamilyPlan.equations_imply_planRowsZero
      (priorInterface geometry) familyRowCount_le assignment one
    intro invocation
    refine ⟨priorSboxes invocation, ?_⟩
    exact PoseidonRetainedFamily.outputEquations
      (priorSchedule program)
      (PiRLCRetainedGeometry.priorPoseidonStart program)
      (PiRLCRetainedGeometry.priorPoseidonFits (prefixGeometry geometry))
      (oneColumn geometry) (priorInputState geometry) assignment invocation
  · apply PoseidonSboxFamilyPlan.equations_imply_planRowsZero
      (outputInterface geometry) familyRowCount_le assignment one
    intro invocation
    refine ⟨outputSboxes invocation, ?_⟩
    exact PoseidonRetainedFamily.outputEquations
      (outputSchedule program)
      (PiRLCRetainedGeometry.outputPoseidonStart program)
      (PiRLCRetainedGeometry.outputPoseidonFits (prefixGeometry geometry))
      (oneColumn geometry) (outputInputState geometry) assignment invocation

end NightstreamFPrime.Export.Stage1.PilotPoseidonPlan
