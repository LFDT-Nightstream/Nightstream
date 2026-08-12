import Nightstream.Protocol.NebulaV2.Ports

/-!
Contract: sequential checked-step batching for successor production profiles.

One batch is one fresh-claim relation with a fixed ordered vector of checked
steps. It retains all normalized rows and fixed memory ports in row-major
order. A sequential witness contains one state at every boundary and links
each step output to the next step input.

This is not a backend batch of independent fresh claims. It does not own a
concrete application transition, memory carry, generated rows, or a selected
factor.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.NebulaV2.CheckedStepBatch

open Nightstream.Protocol.NebulaV2.Ports

/-- One fresh claim owns exactly `count` ordered checked steps. -/
structure Batch (count : Nat) where
  steps : Fin count -> CheckedStep

def Batch.stepList {count : Nat} (batch : Batch count) : List CheckedStep :=
  List.ofFn batch.steps

@[simp] theorem Batch.stepList_length
    {count : Nat} (batch : Batch count) :
    batch.stepList.length = count := by
  simp [Batch.stepList]

/-- All normalized rows in checked-step order, then row order. -/
def Batch.rowList {count : Nat} (batch : Batch count) : List NormalizedRow :=
  batch.stepList.flatMap CheckedStep.rowList

/-- The only batch memory-access list. It preserves step, row, and physical
port order while retaining inactive holes in the authority-bearing ports. -/
def Batch.accesses {count : Nat} (batch : Batch count) : List Access :=
  batch.stepList.flatMap CheckedStep.accesses

private theorem flat_rows_accesses (steps : List CheckedStep) :
    (steps.flatMap CheckedStep.rowList).flatMap NormalizedRow.accesses =
      steps.flatMap CheckedStep.accesses := by
  induction steps with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.flatMap_cons, List.flatMap_append]
      rw [CheckedStep.rowList_flatMap_accesses, inductionHypothesis]

/-- Flattening the normalized rows gives exactly the batch access list. -/
theorem Batch.rowList_flatMap_accesses
    {count : Nat} (batch : Batch count) :
    batch.rowList.flatMap NormalizedRow.accesses = batch.accesses := by
  exact flat_rows_accesses batch.stepList

@[simp] theorem Batch.rowList_length
    {count : Nat} (batch : Batch count) :
    batch.rowList.length = count * applicationRowsPerStep := by
  unfold Batch.rowList
  rw [List.length_flatMap]
  simp [CheckedStep.rowList_length, Batch.stepList_length]

/-- Independent witness shape for one batched relation. The same boundary
state is used as step `i` output and step `i+1` input. -/
structure Witness
    {State : Type} (transition : State -> CheckedStep -> State -> Prop)
    {count : Nat} (batch : Batch count) (before after : State) where
  states : Fin (count + 1) -> State
  initial : states ⟨0, Nat.zero_lt_succ count⟩ = before
  final : states ⟨count, Nat.lt_succ_self count⟩ = after
  step : forall index : Fin count,
    transition
      (states ⟨index.val, Nat.lt_succ_of_lt index.isLt⟩)
      (batch.steps index)
      (states ⟨index.val + 1, Nat.succ_lt_succ index.isLt⟩)

/-- The batched semantic relation is inhabited only by a complete adjacent
state witness. -/
def Sequential
    {State : Type} (transition : State -> CheckedStep -> State -> Prop)
    {count : Nat} (batch : Batch count) (before after : State) : Prop :=
  Nonempty (Witness transition batch before after)

/-- Each checked step is present and uses the exact adjacent states. -/
theorem Witness.step_exact
    {State : Type} {transition : State -> CheckedStep -> State -> Prop}
    {count : Nat} {batch : Batch count} {before after : State}
    (witness : Witness transition batch before after)
    (index : Fin count) :
    transition
      (witness.states ⟨index.val, Nat.lt_succ_of_lt index.isLt⟩)
      (batch.steps index)
      (witness.states
        ⟨index.val + 1, Nat.succ_lt_succ index.isLt⟩) :=
  witness.step index

end Nightstream.Protocol.NebulaV2.CheckedStepBatch
