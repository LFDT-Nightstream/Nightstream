import NightstreamFPrime.Layout.PiRlcWideSampler.Retained
import NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxFamilyPlan

/-! Candidate CCS sampler: 34 Poseidon2 permutations and 17 checked wide
reductions. The first permutation of each pair absorbs [4,i]; its four rate
lanes feed the range gadget. The second permutation advances the transcript.
Only S-box outputs and checked range values have committed coordinates. -/

namespace NightstreamFPrime.Layout.PiRlcWideSampler.BatchPlan

open NightstreamFPrime.Spec NightstreamFPrime.Circuit
open ProductionRelation
open Spec.Folding.PiCCS.PaperJoint
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

def coordinateCount : Nat := 34 * 86 * 41 + 17 * PiRlcWideSampler.coordinateCount

theorem coordinateCount_eq : coordinateCount = 135813 := by
  rw [coordinateCount, PiRlcWideSampler.coordinateCount_eq]

def poseidonBlock : LowNormBlock.Block (34 * 86) where
  kind := .field
  slotCount := 34 * 86
  source := id

structure Interface (columns : Nat) where
  oneColumn : Fin columns
  initialState : PoseidonSboxPlan.State columns
  start : Nat
  fits : start + coordinateCount ≤ columns

private theorem poseidonFits {columns : Nat} (interface : Interface columns) :
    interface.start + poseidonBlock.coordinateCount ≤ columns := by
  have fits := interface.fits
  rw [coordinateCount_eq] at fits
  change interface.start + 119884 ≤ columns
  omega

def sbox {columns : Nat} (interface : Interface columns) (invocation : Fin 34)
    (row : Fin PoseidonRetainedSlots.rows.length) : SparseForm columns :=
  poseidonBlock.form interface.start (poseidonFits interface)
    ⟨invocation.val * 86 + row.val, by
      have rowBound : row.val < 86 := by
        simpa only [PoseidonRetainedSlots.rows_length] using row.isLt
      change _ < 34 * 86
      omega⟩

def outputState {columns : Nat} (interface : Interface columns) (invocation : Fin 34) :
    PoseidonSboxPlan.State columns :=
  SparseLayer.external (fun lane => sbox interface invocation (PoseidonRetainedSlots.finalRow lane))

def priorState {columns : Nat} (interface : Interface columns) (invocation : Fin 34) :
    PoseidonSboxPlan.State columns :=
  if first : invocation.val = 0 then interface.initialState
  else outputState interface ⟨invocation.val - 1, by omega⟩

def entryWord (source : Nat) (lane : Fin 8) : F :=
  if lane.val = 0 then Poseidon2.ofNat 4
  else if lane.val = 1 then Poseidon2.ofNat source else 0

def poseidonInterface {columns : Nat} (interface : Interface columns) :
    PoseidonSboxFamilyPlan.Interface columns 34 where
  oneColumn := interface.oneColumn
  input := fun invocation lane =>
    if invocation.val % 2 = 0 then
      SparseLayer.addConstant interface.oneColumn (priorState interface invocation lane)
        (entryWord (invocation.val / 2) lane)
    else priorState interface invocation lane
  sboxOutput := sbox interface

def poseidonPlan {columns : Nat} (interface : Interface columns) : ProductionRelation.Plan columns :=
  PoseidonSboxFamilyPlan.plan (poseidonInterface interface) (by decide)

def rangeStart {columns : Nat} (interface : Interface columns) (source : Fin 17) : Nat :=
  interface.start + 119884 + source.val * 937

theorem rangeFits {columns : Nat} (interface : Interface columns) (source : Fin 17) :
    Retained.Fits columns (rangeStart interface source) := by
  have fits := interface.fits
  rw [coordinateCount_eq] at fits
  change interface.start + 119884 + source.val * 937 + 937 ≤ columns
  omega

def rangeInputs {columns : Nat} (compiled : RangePlan.Compiled) (interface : Interface columns)
    (source : Fin 17) : compiled.program.Inputs columns where
  oneColumn := interface.oneColumn
  sourceMap := fun _ => Retained.sourceMap (rangeStart interface source) (rangeFits interface source)
    (fun lane => outputState interface ⟨source.val * 2, by omega⟩ ⟨lane.val, by omega⟩)

def rangePlan {columns : Nat} (compiled : RangePlan.Compiled) (interface : Interface columns)
    (source : Fin 17) : ProductionRelation.Plan columns :=
  (compiled.program.compile (rangeInputs compiled interface source)).toPlan

def rangeFamily {columns : Nat} (compiled : RangePlan.Compiled) (interface : Interface columns) :
    ProductionRelation.Plan columns :=
  ProductionRelation.Plan.indexed (fun source : Fin 17 => (rangePlan compiled interface source).forms)
    (by rw [compiled.rowCount]; decide)

theorem rangeFamily_rowCount {columns : Nat} (compiled : RangePlan.Compiled) (interface : Interface columns) :
    (rangeFamily compiled interface).rowCount = 11577 := by
  change 17 * compiled.rows.length = _
  rw [compiled.rowCount]

def plan {columns : Nat} (compiled : RangePlan.Compiled) (interface : Interface columns) :
    ProductionRelation.Plan columns :=
  ProductionRelation.Plan.append (poseidonPlan interface) (rangeFamily compiled interface)
    (by rw [rangeFamily_rowCount]; change 34 * 86 + 11577 ≤ _; decide)

theorem rowCount_eq {columns : Nat} (compiled : RangePlan.Compiled) (interface : Interface columns) :
    (plan compiled interface).rowCount = 14501 := by
  change (poseidonPlan interface).rowCount + (rangeFamily compiled interface).rowCount = _
  rw [rangeFamily_rowCount]
  rfl

theorem rowsZero_iff {columns : Nat} (compiled : RangePlan.Compiled) (interface : Interface columns)
    (assignment : Assignment F columns) :
    (plan compiled interface).RowsZero assignment ↔
      (poseidonPlan interface).RowsZero assignment ∧ (rangeFamily compiled interface).RowsZero assignment :=
  ProductionRelation.Plan.append_rowsZero_iff _ _ _ _

end NightstreamFPrime.Layout.PiRlcWideSampler.BatchPlan
