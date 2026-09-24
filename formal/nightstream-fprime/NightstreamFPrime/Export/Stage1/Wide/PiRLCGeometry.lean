import NightstreamFPrime.Export.Stage1.Wide.PiRLCPlan
import NightstreamFPrime.Layout.PiRlcWideSampler.Norm

/-! Concrete retained layout of the candidate PiRLC phase. The sampler is
followed by ring outputs and quotient coefficients, each encoded as a field.
Temporary digit words and R1CS multiplication scratch have no block here. -/

namespace NightstreamFPrime.Export.Stage1.Wide.PiRLCGeometry

open NightstreamFPrime.Spec NightstreamFPrime.Layout
open ProductionRelation PiRlcWideSampler
open Spec.Folding.PiCCS.PaperJoint
open Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

abbrev RingIndex := PiRLCPlan.RingIndex
abbrev Descriptor := PiRLCProductRingSchedule.Descriptor

def outputCount : Nat := PiRLCProductRingSchedule.invocationCount * ringDegree

def fieldBlock : LowNormBlock.Block (2 * outputCount) where
  kind := .field
  slotCount := 2 * outputCount
  source := id

def coordinateCount : Nat := BatchPlan.coordinateCount + fieldBlock.coordinateCount

theorem outputCount_eq : outputCount = 52326 := rfl

theorem fieldBlock_coordinateCount : fieldBlock.coordinateCount = 4290732 := rfl

theorem coordinateCount_eq : coordinateCount = 4426545 := by
  rw [coordinateCount, BatchPlan.coordinateCount_eq, fieldBlock_coordinateCount]

structure Interface (columns : Nat) where
  oneColumn : Fin columns
  initialState : PoseidonSboxPlan.State columns
  value : RingIndex → Phi81ProductPlan.State columns
  start : Nat
  fits : start + coordinateCount ≤ columns

def sampler {columns : Nat} (interface : Interface columns) : BatchPlan.Interface columns where
  oneColumn := interface.oneColumn
  initialState := interface.initialState
  start := interface.start
  fits := by
    have fits := interface.fits
    rw [coordinateCount_eq] at fits
    rw [BatchPlan.coordinateCount_eq]
    omega

def fieldStart {columns : Nat} (interface : Interface columns) : Nat := interface.start + 135813

theorem fieldFits {columns : Nat} (interface : Interface columns) :
    fieldStart interface + fieldBlock.coordinateCount ≤ columns := by
  simpa only [coordinateCount, BatchPlan.coordinateCount_eq, fieldStart, Nat.add_assoc]
    using interface.fits

/-- Keep the reference family/source/block/lane/cell order. -/
def outputSlot (ring : RingIndex) (lane : Fin ringDegree) : Fin fieldBlock.slotCount :=
  let slot := PiRLCProductRingSchedule.laneInvocation ring lane
  ⟨slot.val, by
    have bound : slot.val < 52326 := slot.isLt
    change slot.val < 104652
    omega⟩

def quotientSlot (ring : RingIndex) (lane : Fin ringDegree) : Fin fieldBlock.slotCount :=
  let slot := PiRLCProductRingSchedule.laneInvocation ring lane
  ⟨outputCount + slot.val, by
    have bound : slot.val < 52326 := slot.isLt
    change 52326 + slot.val < 104652
    omega⟩

def output {columns : Nat} (interface : Interface columns) (ring : RingIndex) :
    Phi81ProductPlan.State columns :=
  fun lane => fieldBlock.form (fieldStart interface) (fieldFits interface) (outputSlot ring lane)

def quotient {columns : Nat} (interface : Interface columns) (ring : RingIndex) :
    Phi81ProductPlan.State columns :=
  fun lane => fieldBlock.form (fieldStart interface) (fieldFits interface) (quotientSlot ring lane)

def withSource (descriptor : Descriptor) (source : Fin 17) : Descriptor :=
  ⟨descriptor.family, source, descriptor.block, descriptor.cell⟩

theorem withSource_self (descriptor : Descriptor) : withSource descriptor descriptor.source = descriptor := by
  cases descriptor
  rfl

theorem withSource_withSource (descriptor : Descriptor) (left right : Fin 17) :
    withSource (withSource descriptor left) right = withSource descriptor right := rfl

def previous (descriptor : Descriptor) (_nonzero : descriptor.source.val ≠ 0) : Descriptor :=
  withSource descriptor ⟨descriptor.source.val - 1, by have h : descriptor.source.val < 17 := descriptor.source.isLt; omega⟩

def prior {columns : Nat} (interface : Interface columns) (ring : RingIndex) : Phi81ProductPlan.State columns :=
  let descriptor := PiRLCProductRingSchedule.descriptor ring
  if first : descriptor.source.val = 0 then fun _ => .empty
  else output interface (previous descriptor first).invocation

def planInterface {columns : Nat} (interface : Interface columns) : PiRLCPlan.Interface columns where
  sampler := sampler interface
  value := interface.value
  quotient := quotient interface
  prior := prior interface
  output := output interface

def plan {columns : Nat} (compiled : RangePlan.Compiled) (interface : Interface columns) : ProductionRelation.Plan columns :=
  PiRLCPlan.plan compiled (planInterface interface)

theorem rowCount_eq {columns : Nat} (compiled : RangePlan.Compiled) (interface : Interface columns) :
    (plan compiled interface).rowCount = 119153 := PiRLCPlan.plan_rows _ _

theorem soundness {columns : Nat} (compiled : RangePlan.Compiled) (interface : Interface columns)
    (assignment : Assignment F columns) (one : assignment interface.oneColumn = 1)
    (rows : (plan compiled interface).RowsZero assignment) (ring : RingIndex) :
    Phi81ProductPlan.evalState assignment (output interface ring) =
      ringFAdd (Phi81ProductPlan.evalState assignment (prior interface ring))
        (ringFMul (PiRLCPlan.challenge (planInterface interface) assignment ring)
          (Phi81ProductPlan.evalState assignment (interface.value ring))) :=
  PiRLCPlan.soundness compiled (planInterface interface) assignment one rows ring

end NightstreamFPrime.Export.Stage1.Wide.PiRLCGeometry
