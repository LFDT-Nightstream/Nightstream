import NightstreamFPrime.Export.Stage1.Wide.AssignmentTransport
import NightstreamFPrime.Export.Stage1.Wide.CarrierAssignment

/-! Pointwise execution of the schema-4 retained-assignment plan. Source runs
select physical values. The checked three-bit digits supply the challenges;
the only derived source suffix contains the Phi81 quotient coefficients. -/

namespace NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportExecution

open NightstreamFPrime.Circuit NightstreamFPrime.Spec NightstreamFPrime.Layout
open NightstreamFPrime.Lifecycle
open ProductionRelation
open Spec.Folding.PiCCS.PaperJoint Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open AssignmentTransport

deriving instance DecidableEq for Values
deriving instance DecidableEq for AssignmentTransport.Plan

def familyOrdinal : PiRLCProductSchedule.Family → Nat
  | .commitment => 0
  | .publicInput => 1
  | .evalK => 2
  | .evalA => 3

def familyShape (plan : Plan) (family : PiRLCProductSchedule.Family) :
    PerApplicationAssignmentTransport.Phi81FamilyShape :=
  plan.families.getD (familyOrdinal family) ⟨0, 0, 0⟩

def familyCount (plan : Plan) (family : PiRLCProductSchedule.Family) : Nat :=
  let shape := familyShape plan family
  shape.sourceCount * shape.blockCount * ringDegree * shape.cellCount

def familyOffset (plan : Plan) : PiRLCProductSchedule.Family → Nat
  | .commitment => 0
  | .publicInput => familyCount plan .commitment
  | .evalK => familyCount plan .commitment + familyCount plan .publicInput
  | .evalA => familyCount plan .commitment + familyCount plan .publicInput + familyCount plan .evalK

/-- The serialized family shape fixes source, block, lane, and cell order. -/
def invocationIndex (plan : Plan) (descriptor : PiRLCProductSchedule.Descriptor) : Nat :=
  let shape := familyShape plan descriptor.family
  familyOffset plan descriptor.family +
    descriptor.source.val * shape.blockCount * ringDegree * shape.cellCount +
    descriptor.block.val * ringDegree * shape.cellCount +
    descriptor.lane.val * shape.cellCount + descriptor.cell.val

def digitBit (plan : Plan) (physical : Env) (source : Fin 17)
    (lane : Fin ringDegree) (bit : Fin 3) : F :=
  physical (AffineRuns.sourceAt plan.challengeSources
    (source.val * (ringDegree * 3) + lane.val * 3 + bit.val))

def digit (plan : Plan) (physical : Env) (source : Fin 17) (lane : Fin ringDegree) : Nat :=
  (digitBit plan physical source lane ⟨0, by decide⟩).val +
    2 * (digitBit plan physical source lane ⟨1, by decide⟩).val +
    4 * (digitBit plan physical source lane ⟨2, by decide⟩).val

/-- This is the same fail-closed digit check used by the physical consumer. -/
def DigitsValid (plan : Plan) (physical : Env) : Prop :=
  (∀ source lane bit, digitBit plan physical source lane bit = 0 ∨
    digitBit plan physical source lane bit = 1) ∧
  ∀ source lane, digit plan physical source lane ≤ 4

instance (plan : Plan) (physical : Env) : Decidable (DigitsValid plan physical) :=
  inferInstanceAs (Decidable ((_ : Prop) ∧ (_ : Prop)))

def challengeRing (plan : Plan) (physical : Env) (source : Fin 17) : RingF :=
  fun lane => Gadgets.Sampling.WideReduction.fieldOfNat (digit plan physical source lane) - 2

def valueRing (plan : Plan) (physical : Env) (descriptor : PiRLCProductSchedule.Descriptor) : RingF :=
  fun lane => physical (AffineRuns.sourceAt plan.valueSources (invocationIndex plan (descriptor.withLane lane)))

/-- Quotients use the existing proved Phi81 arithmetic and the old lane order. -/
def quotientValue (plan : Plan) (physical : Env)
    (slot : Fin PiRLCProductSchedule.invocationCount) : F :=
  let descriptor := PiRLCProductSchedule.descriptor slot
  Phi81Relation.QuotientProduct.quotientCoeff
    (challengeRing plan physical descriptor.source) (valueRing plan physical descriptor) descriptor.lane

def sourceValue (plan : Plan) (physicalWidth : Nat) (physical : Env) (source : Nat) : F :=
  if source < physicalWidth then physical source
  else if derived : source - physicalWidth < PiRLCProductSchedule.invocationCount then
    quotientValue plan physical ⟨source - physicalWidth, derived⟩
  else 0

def identityBlock (block : Values) : LowNormBlock.Block block.count where
  kind := block.kind
  slotCount := block.count
  source := id

def blockValue (plan : Plan) (physicalWidth : Nat) (physical : Env) (block : Values) :
    CanonicalBlockAssignment.BlockValue :=
  CanonicalBlockAssignment.ofBlock (identityBlock block)
    (fun slot => sourceValue plan physicalWidth physical (AffineRuns.sourceAt block.sources slot.val))

def schedule (plan : Plan) (physicalWidth : Nat) (physical : Env) : CanonicalBlockAssignment.Schedule :=
  plan.blocks.map (blockValue plan physicalWidth physical)

def outputDigest (plan : Plan) (physical : Env) : Digest :=
  plan.outputDigestExpressions.map (fun expression => expression.eval physical)

/-- Execute by point lookup. No expanded coordinate array is needed. -/
def executeUnchecked (program : Program) (plan : Plan) (physicalWidth : Nat) (physical : Env) :
    Assignment F (RetainedLayout.logicalWidth program) :=
  CanonicalBlockAssignment.assignment (encodedHashCells (outputDigest plan physical))
    (schedule plan physicalWidth physical)

/-- Only the exact emitted schema-4 plan and valid checked digits execute. -/
def execute (program : Program) (plan : Plan) (physicalWidth : Nat) (physical : Env) :
    Option (Assignment F (RetainedLayout.logicalWidth program)) :=
  if AssignmentTransport.plan program physicalWidth = .ok plan ∧ DigitsValid plan physical then
    some (executeUnchecked program plan physicalWidth physical)
  else none

theorem execute_emitted (program : Program) (plan : Plan) (physicalWidth : Nat) (physical : Env)
    (emitted : AssignmentTransport.plan program physicalWidth = .ok plan)
    (digits : DigitsValid plan physical) :
    execute program plan physicalWidth physical = some (executeUnchecked program plan physicalWidth physical) := by
  exact if_pos ⟨emitted, digits⟩

theorem sourceValue_physical (plan : Plan) (physicalWidth : Nat) (physical : Env)
    (source : Nat) (bounded : source < physicalWidth) :
    sourceValue plan physicalWidth physical source = physical source := by
  exact if_pos bounded

theorem sourceValue_quotient (plan : Plan) (physicalWidth : Nat) (physical : Env)
    (slot : Fin PiRLCProductSchedule.invocationCount) :
    sourceValue plan physicalWidth physical (physicalWidth + slot.val) = quotientValue plan physical slot := by
  simp only [sourceValue, Nat.not_lt.mpr (Nat.le_add_right physicalWidth slot.val), if_false,
    Nat.add_sub_cancel_left, dif_pos slot.isLt]

theorem blockValue_at (plan : Plan) (physicalWidth : Nat) (physical : Env) (block : Values)
    (slot : Fin block.count) (coordinate : Fin block.kind.width) :
    (blockValue plan physicalWidth physical block).coordinateAt
        (slot.val * block.kind.width + coordinate.val) =
      LowNormSlot.coordinate block.kind
        (sourceValue plan physicalWidth physical (AffineRuns.sourceAt block.sources slot.val)) coordinate :=
  CanonicalBlockAssignment.BlockValue.coordinateAt_coordinateOffset
    (blockValue plan physicalWidth physical block) slot coordinate

end NightstreamFPrime.Export.Stage1.Wide.AssignmentTransportExecution
