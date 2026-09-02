import NightstreamFPrime.Export.Stage1.PilotOrdinaryRetainedGeometry
import NightstreamFPrime.Export.Stage1.PilotPoseidonPlan
import NightstreamFPrime.Layout.ProductionRelation.PinFamilyPlan

/-!
Owns eight explicit custody rows between the direct pilot Poseidon2 outputs
and the legacy final-output columns consumed by the canonical pilot ordinary
rows. Four rows bind the prior digest and four bind the output digest.

This module does not retain or bind the unused capacity lanes.
-/

namespace NightstreamFPrime.Export.Stage1.PilotDigestBindingPlan

open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def chainCount : Nat := 2
def laneCount : Nat := 4
def rowCount : Nat := chainCount * laneCount

@[simp] theorem rowCount_eq : rowCount = 8 := by rfl

def poseidonGeometry {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    PiRLCPoseidonGeometry.Geometry program logicalWidth where
  pilotFits := by
    apply Nat.le_trans _ geometry.completeFits
    rw [PiRLCPoseidonGeometry.pilotLogicalWidth_eq,
      PilotOrdinaryRetainedGeometry.completeLogicalWidth_eq]
    omega

def oneColumn {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    Fin logicalWidth :=
  PilotOrdinaryRetainedGeometry.oneColumn geometry

def lastInvocation : Fin PilotPoseidonPlan.invocationCount :=
  ⟨12349, by
    rw [PilotPoseidonPlan.invocationCount_eq]
    omega⟩

def finalSlot (lane : Fin laneCount) : Fin 592 :=
  ⟨584 + lane.val, by
    have laneBound := lane.isLt
    change lane.val < 4 at laneBound
    omega⟩

def descriptor (row : Fin rowCount) : Fin chainCount × Fin laneCount :=
  Fin.decodeProd row

def priorChain : Fin chainCount := ⟨0, by norm_num [chainCount]⟩
def outputChain : Fin chainCount := ⟨1, by norm_num [chainCount]⟩

def priorRow (lane : Fin laneCount) : Fin rowCount :=
  Fin.encodeProd (priorChain, lane)

def outputRow (lane : Fin laneCount) : Fin rowCount :=
  Fin.encodeProd (outputChain, lane)

def digestLane (lane : Fin laneCount) : Fin 8 :=
  ⟨lane.val, by
    have laneBound := lane.isLt
    change lane.val < 4 at laneBound
    omega⟩

@[simp] theorem descriptor_priorRow (lane : Fin laneCount) :
    descriptor (priorRow lane) = (priorChain, lane) := by
  exact Fin.decodeProd_encodeProd _

@[simp] theorem descriptor_outputRow (lane : Fin laneCount) :
    descriptor (outputRow lane) = (outputChain, lane) := by
  exact Fin.decodeProd_encodeProd _

def derivedForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (row : Fin rowCount) : SparseForm logicalWidth :=
  let decoded := descriptor row
  let lane : Fin 8 := ⟨decoded.2.val, by
    have laneBound := decoded.2.isLt
    change decoded.2.val < 4 at laneBound
    omega⟩
  if decoded.1.val = 0 then
    (PilotPoseidonPlan.priorInterface (poseidonGeometry geometry)).output
      lastInvocation lane
  else
    (PilotPoseidonPlan.outputInterface (poseidonGeometry geometry)).output
      lastInvocation lane

def legacyForm {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (row : Fin rowCount) : SparseForm logicalWidth :=
  let decoded := descriptor row
  if decoded.1.val = 0 then
    (PiCCSOrdinaryRetainedBlocks.priorLastBlock program).form
      (PiCCSOrdinaryRetainedGeometry.priorLastStart program)
      (PiCCSOrdinaryRetainedGeometry.priorLastFits
        (PilotOrdinaryRetainedGeometry.prefixGeometry geometry))
      (finalSlot decoded.2)
  else
    (PiCCSOrdinaryRetainedBlocks.outputLastBlock program).form
      (PiCCSOrdinaryRetainedGeometry.outputLastStart program)
      (PiCCSOrdinaryRetainedGeometry.outputLastFits
        (PilotOrdinaryRetainedGeometry.prefixGeometry geometry))
      (finalSlot decoded.2)

theorem legacyForm_priorRow
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (lane : Fin laneCount) :
    legacyForm geometry (priorRow lane) =
      (PiCCSOrdinaryRetainedBlocks.priorLastBlock program).form
        (PiCCSOrdinaryRetainedGeometry.priorLastStart program)
        (PiCCSOrdinaryRetainedGeometry.priorLastFits
          (PilotOrdinaryRetainedGeometry.prefixGeometry geometry))
        (finalSlot lane) := by
  unfold legacyForm
  rw [descriptor_priorRow]
  rfl

theorem legacyForm_outputRow
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (lane : Fin laneCount) :
    legacyForm geometry (outputRow lane) =
      (PiCCSOrdinaryRetainedBlocks.outputLastBlock program).form
        (PiCCSOrdinaryRetainedGeometry.outputLastStart program)
        (PiCCSOrdinaryRetainedGeometry.outputLastFits
          (PilotOrdinaryRetainedGeometry.prefixGeometry geometry))
        (finalSlot lane) := by
  unfold legacyForm
  rw [descriptor_outputRow]
  rfl

theorem derivedForm_priorRow
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (lane : Fin laneCount) :
    derivedForm geometry (priorRow lane) =
      (PilotPoseidonPlan.priorInterface (poseidonGeometry geometry)).output
        lastInvocation (digestLane lane) := by
  simp [derivedForm, descriptor_priorRow, priorChain, digestLane]

theorem derivedForm_outputRow
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (lane : Fin laneCount) :
    derivedForm geometry (outputRow lane) =
      (PilotPoseidonPlan.outputInterface (poseidonGeometry geometry)).output
        lastInvocation (digestLane lane) := by
  simp [derivedForm, descriptor_outputRow, outputChain, digestLane]

def difference {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (row : Fin rowCount) : SparseForm logicalWidth :=
  SparseForm.add (legacyForm geometry row)
    (SparseForm.scale (-1) (derivedForm geometry row))

def interface {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    PinFamilyPlan.Interface logicalWidth rowCount where
  oneColumn := oneColumn geometry
  value := difference geometry

theorem rowCount_le : rowCount ≤
    2 ^ NightstreamFPrime.Lifecycle.cubeVariables := by
  norm_num [rowCount, chainCount, laneCount,
    NightstreamFPrime.Lifecycle.cubeVariables]

def plan {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    ProductionRelation.Plan logicalWidth :=
  PinFamilyPlan.plan (interface geometry) rowCount_le

@[simp] theorem plan_rowCount {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth) :
    (plan geometry).rowCount = 8 := by
  rfl

def Matches {program : Lifecycle.Stage1.Application.Program}
    {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth) : Prop :=
  ∀ row, (legacyForm geometry row).eval assignment =
    (derivedForm geometry row).eval assignment

/-- The eight bridge rows are exactly the eight digest-lane equalities. -/
theorem rowsZero_iff_matches
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    (geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth)
    (assignment : Assignment F logicalWidth)
    (one : assignment (oneColumn geometry) = 1) :
    (plan geometry).RowsZero assignment ↔ Matches geometry assignment := by
  rw [plan, PinFamilyPlan.planRowsZero_iff
    (interface geometry) rowCount_le assignment one]
  constructor
  · intro zeros row
    have zero := zeros row
    change (difference geometry row).eval assignment = 0 at zero
    unfold difference at zero
    rw [SparseForm.add_eval, SparseForm.scale_eval] at zero
    have subZero :
        (legacyForm geometry row).eval assignment -
          (derivedForm geometry row).eval assignment = 0 := by
      simpa [sub_eq_add_neg] using zero
    exact sub_eq_zero.mp subZero
  · intro matching row
    change (difference geometry row).eval assignment = 0
    unfold difference
    rw [SparseForm.add_eval, SparseForm.scale_eval, matching row]
    simp

theorem Matches.prior
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    {geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth}
    {assignment : Assignment F logicalWidth}
    (matching : Matches geometry assignment) (lane : Fin laneCount) :
    (legacyForm geometry (priorRow lane)).eval assignment =
      (derivedForm geometry (priorRow lane)).eval assignment :=
  matching (priorRow lane)

theorem Matches.output
    {program : Lifecycle.Stage1.Application.Program} {logicalWidth : Nat}
    {geometry : PilotOrdinaryRetainedGeometry.Geometry program logicalWidth}
    {assignment : Assignment F logicalWidth}
    (matching : Matches geometry assignment) (lane : Fin laneCount) :
    (legacyForm geometry (outputRow lane)).eval assignment =
      (derivedForm geometry (outputRow lane)).eval assignment :=
  matching (outputRow lane)

end NightstreamFPrime.Export.Stage1.PilotDigestBindingPlan
