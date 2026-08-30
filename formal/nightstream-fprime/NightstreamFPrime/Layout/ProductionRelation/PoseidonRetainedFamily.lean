import NightstreamFPrime.Layout.LowNormBlock
import NightstreamFPrime.Layout.ProductionRelation.PoseidonSboxFamilyPlan

/-!
Owns the retained-block adapter for one indexed Poseidon2 family. A schedule
maps invocation-major S-box rows to one low-norm block and derives each final
output state directly from rows 78 through 85.

This module does not select a concrete Stage 1 schedule or source columns.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.PoseidonRetainedFamily

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

structure Schedule (sourceWidth invocationCount : Nat) where
  block : LowNormBlock.Block sourceWidth
  slotCount_eq : block.slotCount =
    invocationCount * PoseidonRetainedSlots.rows.length

def slot {sourceWidth invocationCount : Nat}
    (schedule : Schedule sourceWidth invocationCount)
    (invocation : Fin invocationCount)
    (row : Fin PoseidonRetainedSlots.rows.length) :
    Fin schedule.block.slotCount :=
  Fin.cast schedule.slotCount_eq.symm (Fin.encodeProd (invocation, row))

def form {sourceWidth invocationCount logicalWidth : Nat}
    (schedule : Schedule sourceWidth invocationCount)
    (start : Nat)
    (fits : start + schedule.block.coordinateCount ≤ logicalWidth)
    (invocation : Fin invocationCount)
    (row : Fin PoseidonRetainedSlots.rows.length) : SparseForm logicalWidth :=
  schedule.block.form start fits (slot schedule invocation row)

theorem form_eval {sourceWidth invocationCount logicalWidth : Nat}
    (schedule : Schedule sourceWidth invocationCount)
    (start : Nat)
    (fits : start + schedule.block.coordinateCount ≤ logicalWidth)
    (assignment : Assignment F logicalWidth)
    (source : Fin sourceWidth → F)
    (encodes : schedule.block.EncodesAt start fits assignment source)
    (invocation : Fin invocationCount)
    (row : Fin PoseidonRetainedSlots.rows.length) :
    (form schedule start fits invocation row).eval assignment =
      source (schedule.block.source (slot schedule invocation row)) := by
  exact LowNormBlock.Block.form_eval schedule.block start fits assignment
    source encodes (slot schedule invocation row)

/-- Closed-form final output state from the final eight retained S-box rows. -/
def outputState {sourceWidth invocationCount logicalWidth : Nat}
    (schedule : Schedule sourceWidth invocationCount)
    (start : Nat)
    (fits : start + schedule.block.coordinateCount ≤ logicalWidth)
    (invocation : Fin invocationCount) : PoseidonSboxPlan.State logicalWidth :=
  SparseLayer.external fun lane =>
    form schedule start fits invocation (PoseidonRetainedSlots.finalRow lane)

/-- Evaluating a retained final state reconstructs the exact selected source
value in every lane. -/
theorem outputState_eval {sourceWidth invocationCount logicalWidth : Nat}
    (schedule : Schedule sourceWidth invocationCount)
    (start : Nat)
    (fits : start + schedule.block.coordinateCount ≤ logicalWidth)
    (assignment : Assignment F logicalWidth)
    (source : Fin sourceWidth → F)
    (encodes : schedule.block.EncodesAt start fits assignment source)
    (invocation : Fin invocationCount) :
    SparseLayer.evalState assignment
        (outputState schedule start fits invocation) =
      NightstreamFPrime.Gadgets.Poseidon2.Layer.externalF (fun lane =>
        source (schedule.block.source
          (slot schedule invocation (PoseidonRetainedSlots.finalRow lane)))) := by
  funext lane
  change (SparseLayer.external (fun row =>
    form schedule start fits invocation
      (PoseidonRetainedSlots.finalRow row)) lane).eval assignment = _
  rw [SparseLayer.eval_external]
  apply congrArg (fun state =>
    NightstreamFPrime.Gadgets.Poseidon2.Layer.externalF state lane)
  funext row
  exact form_eval schedule start fits assignment source encodes invocation
    (PoseidonRetainedSlots.finalRow row)

def invocationInterface {sourceWidth invocationCount logicalWidth : Nat}
    (schedule : Schedule sourceWidth invocationCount)
    (start : Nat)
    (fits : start + schedule.block.coordinateCount ≤ logicalWidth)
    (oneColumn : Fin logicalWidth)
    (input : Fin invocationCount → PoseidonSboxPlan.State logicalWidth)
    (invocation : Fin invocationCount) :
    PoseidonSboxPlan.Interface logicalWidth :=
  { oneColumn := oneColumn
    input := input invocation
    sboxOutput := form schedule start fits invocation
    output := outputState schedule start fits invocation }

def familyInterface {sourceWidth invocationCount logicalWidth : Nat}
    (schedule : Schedule sourceWidth invocationCount)
    (start : Nat)
    (fits : start + schedule.block.coordinateCount ≤ logicalWidth)
    (oneColumn : Fin logicalWidth)
    (input : Fin invocationCount → PoseidonSboxPlan.State logicalWidth) :
    PoseidonSboxFamilyPlan.Interface logicalWidth invocationCount :=
  { oneColumn := oneColumn
    input := input
    sboxOutput := form schedule start fits
    output := outputState schedule start fits }

@[simp] theorem family_invocationInterface
    {sourceWidth invocationCount logicalWidth : Nat}
    (schedule : Schedule sourceWidth invocationCount)
    (start : Nat)
    (fits : start + schedule.block.coordinateCount ≤ logicalWidth)
    (oneColumn : Fin logicalWidth)
    (input : Fin invocationCount → PoseidonSboxPlan.State logicalWidth)
    (invocation : Fin invocationCount) :
    PoseidonSboxFamilyPlan.invocationInterface
        (familyInterface schedule start fits oneColumn input) invocation =
      invocationInterface schedule start fits oneColumn input invocation := by
  rfl

/-- The generic trace output is the adapter's constant-time final state. -/
theorem trace_state_eq_outputState
    {sourceWidth invocationCount logicalWidth : Nat}
    (schedule : Schedule sourceWidth invocationCount)
    (start : Nat)
    (fits : start + schedule.block.coordinateCount ≤ logicalWidth)
    (oneColumn : Fin logicalWidth)
    (input : Fin invocationCount → PoseidonSboxPlan.State logicalWidth)
    (invocation : Fin invocationCount) :
    (PoseidonSboxPlan.trace
      (invocationInterface schedule start fits oneColumn input invocation)).state =
        outputState schedule start fits invocation := by
  rw [PoseidonSboxPlan.trace_state_eq_directOutput]
  unfold PoseidonSboxPlan.directOutput outputState
  apply congrArg SparseLayer.external
  funext lane
  unfold PoseidonSboxPlan.fullOutput PoseidonSboxPlan.sboxOutputAt
    invocationInterface
  have bounded : 78 + lane.val < PoseidonRetainedSlots.rows.length :=
    (PoseidonRetainedSlots.finalRow lane).isLt
  rw [dif_pos bounded]
  apply congrArg (form schedule start fits invocation)
  apply Fin.ext
  rfl

/-- Output pins are exact definitional custody checks for the derived final
state and require no extra witness values. -/
theorem outputEquations
    {sourceWidth invocationCount logicalWidth : Nat}
    (schedule : Schedule sourceWidth invocationCount)
    (start : Nat)
    (fits : start + schedule.block.coordinateCount ≤ logicalWidth)
    (oneColumn : Fin logicalWidth)
    (input : Fin invocationCount → PoseidonSboxPlan.State logicalWidth)
    (assignment : Assignment F logicalWidth)
    (invocation : Fin invocationCount) :
    PoseidonSboxPlan.OutputEquations
      (invocationInterface schedule start fits oneColumn input invocation)
      assignment := by
  intro lane
  rw [trace_state_eq_outputState]
  rfl

end NightstreamFPrime.Layout.ProductionRelation.PoseidonRetainedFamily
