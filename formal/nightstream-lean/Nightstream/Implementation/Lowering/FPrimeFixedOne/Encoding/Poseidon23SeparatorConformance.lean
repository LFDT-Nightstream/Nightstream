import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Poseidon23ApplicationProfile

/-!
Contract: where two selected preimages with the same payload differ when only
the hash mode changes.

Owns: the conformance bridge `POSEIDON2-HASH-SEPARATOR-APPLICATION` named as its
residue — the statement connecting the profile's `normalizedIteration` to the
slot a selected preimage carries it in.

Does not own: the sponge, the chunking, the digest, or the relation between the
actual F-prime prior and next calls. Those are
`Poseidon2HashRecipe` and below, and this module deliberately does not reach
them: the bridge is about *selection*, and selection is this layer's.

The actual calls use different current and running operands. The equal-tail
theorems in this file must not be used to link those two calls.

## Why the bridge is here and not in `Canonical`

`Poseidon2HashRecipe.separatedPreimage` supplies the *action* of a first-slot
separator and proves it non-degenerate for every residue.  It cannot say the
action is the profile's, because it cannot see an iteration —
`Encoding` imports `Canonical`, not the reverse.

This module can see both, so this is where the two are compared.

## The hypothesis is a plan property, and it is checkable

A `CoordinatePlan` is free to send slot zero anywhere.  The bridge therefore
takes `plan.preimage ⟨0, _⟩ = ⟨0, _⟩` — slot zero reads source coordinate zero,
which is where `sourceCoordinates` puts the normalized iteration.  A deployment
checks that of its own plan; nothing in the type enforces it, which is exactly
what `POSEIDON2-HASH-PROJECTION-INJECTIVITY` recorded about plans generally.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Poseidon23SeparatorConformance

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

/-- **Slot zero carries the normalized iteration**, when the plan sends it
there. -/
theorem selected_slot_zero
    {parameters : Parameters} (codecs : DataCodecs parameters)
    {alignmentWidth : Nat}
    (plan : Poseidon23Hash.CoordinatePlan
      (Poseidon23Hash.sourceWidth codecs) alignmentWidth)
    (slotZero : (plan.preimage ⟨0, by decide⟩).val = 0)
    (next : Bool) (iteration : Nat)
    (z0 current : parameters.State) (running : parameters.Running) :
    (Poseidon23Hash.select
        (Poseidon23Hash.sourceCoordinates codecs next iteration z0 current running)
        plan.preimage).getD 0 0
      = Poseidon23Hash.normalizedIteration next iteration := by
  simp only [Poseidon23Hash.select, List.getD_eq_getElem?_getD,
    List.getElem?_ofFn]
  rw [dif_pos (by decide : 0 < 23)]
  simp only [Option.getD_some, slotZero]
  rfl

/-- **The two calls differ at slot zero exactly as `normalizedIteration`
does.**

This is the bridge for fixed payload operands: whatever
`Poseidon2HashRecipe.separatedPreimage` does to slot zero, changing only the
mode gives `normalizedIteration true` against `normalizedIteration false`. -/
theorem prior_next_differ_at_slot_zero
    {parameters : Parameters} (codecs : DataCodecs parameters)
    {alignmentWidth : Nat}
    (plan : Poseidon23Hash.CoordinatePlan
      (Poseidon23Hash.sourceWidth codecs) alignmentWidth)
    (slotZero : (plan.preimage ⟨0, by decide⟩).val = 0)
    (iteration : Nat)
    (z0 current : parameters.State) (running : parameters.Running) :
    ((Poseidon23Hash.select
        (Poseidon23Hash.sourceCoordinates codecs true iteration z0 current running)
        plan.preimage).getD 0 0,
      (Poseidon23Hash.select
        (Poseidon23Hash.sourceCoordinates codecs false iteration z0 current running)
        plan.preimage).getD 0 0)
      = (Poseidon23Hash.normalizedIteration true iteration,
        Poseidon23Hash.normalizedIteration false iteration) := by
  rw [selected_slot_zero codecs plan slotZero true iteration z0 current running,
    selected_slot_zero codecs plan slotZero false iteration z0 current running]

/-- **Away from source coordinate zero the two fixed-payload selections agree.**

So the separation is confined to the slots the plan points at the iteration,
and every other slot carries identical data between the two calls. -/
theorem prior_next_agree_off_slot_zero
    {parameters : Parameters} (codecs : DataCodecs parameters)
    {alignmentWidth : Nat}
    (plan : Poseidon23Hash.CoordinatePlan
      (Poseidon23Hash.sourceWidth codecs) alignmentWidth)
    (slot : Fin 23) (away : (plan.preimage slot).val ≠ 0)
    (iteration : Nat)
    (z0 current : parameters.State) (running : parameters.Running) :
    (Poseidon23Hash.select
        (Poseidon23Hash.sourceCoordinates codecs true iteration z0 current running)
        plan.preimage).getD slot.val 0
      = (Poseidon23Hash.select
        (Poseidon23Hash.sourceCoordinates codecs false iteration z0 current running)
        plan.preimage).getD slot.val 0 := by
  have slotLt : slot.val < 23 := slot.isLt
  simp only [Poseidon23Hash.select, List.getD_eq_getElem?_getD,
    List.getElem?_ofFn, dif_pos slotLt, Option.getD_some]
  obtain ⟨previous, positive⟩ : ∃ previous, (plan.preimage slot).val = previous + 1 := by
    cases shape : (plan.preimage slot).val with
    | zero => exact absurd shape away
    | succ previous => exact ⟨previous, rfl⟩
  rw [positive]
  rfl

/-- **Any slot pointing at source coordinate zero carries the normalized
iteration.**

`selected_slot_zero` is this at `slot = 0`.  The general form is what a
composition needs: separation is located by *which source coordinate a slot
pulls from*, not by the slot's own index. -/
theorem selected_at_iteration_slot
    {parameters : Parameters} (codecs : DataCodecs parameters)
    {alignmentWidth : Nat}
    (plan : Poseidon23Hash.CoordinatePlan
      (Poseidon23Hash.sourceWidth codecs) alignmentWidth)
    (slot : Fin 23) (pointsAtZero : (plan.preimage slot).val = 0)
    (next : Bool) (iteration : Nat)
    (z0 current : parameters.State) (running : parameters.Running) :
    (Poseidon23Hash.select
        (Poseidon23Hash.sourceCoordinates codecs next iteration z0 current running)
        plan.preimage).getD slot.val 0
      = Poseidon23Hash.normalizedIteration next iteration := by
  simp only [Poseidon23Hash.select, List.getD_eq_getElem?_getD,
    List.getElem?_ofFn, dif_pos slot.isLt, Option.getD_some]
  rw [show (⟨slot.val, slot.isLt⟩ : Fin 23) = slot from Fin.ext rfl,
    pointsAtZero]
  rfl

/-! ## Carrying the hypothesis in a type

The three theorems above take `(plan.preimage ⟨0,_⟩).val = 0` as a loose side
condition.  A deployment that forgets to check it gets a plan that typechecks and
a separator that lands nowhere in particular.

`SeparatingPlan` bundles the plan with the property, so a deployment that
constructs one cannot omit the check.  The bridge restated over it carries no
side condition at all.

## Why not strengthen `CoordinatePlan` itself

Adding the field there would be the stronger fix and it is **not** taken
unilaterally.  `CoordinatePlan` is what a deployment must supply, so requiring a
new field changes what counts as a valid deployment — a change-control question
under spec §16, not a formalization one.  `POSEIDON2-HASH-PLAN-STRENGTHENING`
records it as the alternative rather than performing it.

A refinement is additive: existing plans still typecheck, and a deployment opts
in by proving one equation about its own. -/

/-- A coordinate plan that puts the iteration where the separator expects it. -/
structure SeparatingPlan
    {parameters : Parameters} (codecs : DataCodecs parameters)
    (alignmentWidth : Nat) where
  plan : Poseidon23Hash.CoordinatePlan
    (Poseidon23Hash.sourceWidth codecs) alignmentWidth
  slotZero : (plan.preimage ⟨0, by decide⟩).val = 0

/-- **Slot zero carries the normalized iteration**, with no side condition. -/
theorem SeparatingPlan.selected_slot_zero
    {parameters : Parameters} {codecs : DataCodecs parameters}
    {alignmentWidth : Nat} (separating : SeparatingPlan codecs alignmentWidth)
    (next : Bool) (iteration : Nat)
    (z0 current : parameters.State) (running : parameters.Running) :
    (Poseidon23Hash.select
        (Poseidon23Hash.sourceCoordinates codecs next iteration z0 current running)
        separating.plan.preimage).getD 0 0
      = Poseidon23Hash.normalizedIteration next iteration :=
  Poseidon23SeparatorConformance.selected_slot_zero codecs separating.plan
    separating.slotZero next iteration z0 current running

/-- **The two calls differ at slot zero**, with no side condition. -/
theorem SeparatingPlan.prior_next_differ
    {parameters : Parameters} {codecs : DataCodecs parameters}
    {alignmentWidth : Nat} (separating : SeparatingPlan codecs alignmentWidth)
    (iteration : Nat)
    (z0 current : parameters.State) (running : parameters.Running) :
    ((Poseidon23Hash.select
        (Poseidon23Hash.sourceCoordinates codecs true iteration z0 current running)
        separating.plan.preimage).getD 0 0,
      (Poseidon23Hash.select
        (Poseidon23Hash.sourceCoordinates codecs false iteration z0 current running)
        separating.plan.preimage).getD 0 0)
      = (Poseidon23Hash.normalizedIteration true iteration,
        Poseidon23Hash.normalizedIteration false iteration) :=
  Poseidon23SeparatorConformance.prior_next_differ_at_slot_zero codecs
    separating.plan separating.slotZero iteration z0 current running


/-- **Away from the iteration coordinate the two calls agree**, with no side
condition.

The third of the three bridges, restated over `SeparatingPlan` like the other
two.  It had been left out — an unevenness in a group of three that made the
composition below look unavailable. -/
theorem SeparatingPlan.prior_next_agree
    {parameters : Parameters} {codecs : DataCodecs parameters}
    {alignmentWidth : Nat} (separating : SeparatingPlan codecs alignmentWidth)
    (slot : Fin 23) (away : (separating.plan.preimage slot).val ≠ 0)
    (iteration : Nat)
    (z0 current : parameters.State) (running : parameters.Running) :
    (Poseidon23Hash.select
        (Poseidon23Hash.sourceCoordinates codecs true iteration z0 current running)
        separating.plan.preimage).getD slot.val 0
      = (Poseidon23Hash.select
        (Poseidon23Hash.sourceCoordinates codecs false iteration z0 current running)
        separating.plan.preimage).getD slot.val 0 :=
  Poseidon23SeparatorConformance.prior_next_agree_off_slot_zero codecs
    separating.plan slot away iteration z0 current running

/-- **With fixed payload operands, the next-mode preimage is the prior-mode
preimage with the iteration slots moved, and nothing else.**

This theorem is not the relation between the real F-prime hash calls. Those
calls can use different current and running operands.

**No change-control decision is involved.**  `POSEIDON2-HASH-PLAN-STRENGTHENING`
asks whether `CoordinatePlan` should carry the slot-zero field; this composition
needs only the hypothesis `SeparatingPlan` already carries, so the §16 question
is about where to store a hypothesis, not about whether the conformance
holds. -/
theorem SeparatingPlan.next_is_separated
    {parameters : Parameters} {codecs : DataCodecs parameters}
    {alignmentWidth : Nat} (separating : SeparatingPlan codecs alignmentWidth)
    (iteration : Nat)
    (z0 current : parameters.State) (running : parameters.Running)
    (slot : Fin 23) :
    (Poseidon23Hash.select
        (Poseidon23Hash.sourceCoordinates codecs true iteration z0 current running)
        separating.plan.preimage).getD slot.val 0
      = if (separating.plan.preimage slot).val = 0
          then Poseidon23Hash.normalizedIteration true iteration
          else (Poseidon23Hash.select
            (Poseidon23Hash.sourceCoordinates codecs false iteration z0 current
              running) separating.plan.preimage).getD slot.val 0 := by
  by_cases atIteration : (separating.plan.preimage slot).val = 0
  · rw [if_pos atIteration]
    exact Poseidon23SeparatorConformance.selected_at_iteration_slot codecs
      separating.plan slot atIteration true iteration z0 current running
  · rw [if_neg atIteration]
    exact separating.prior_next_agree slot atIteration iteration z0 current
      running

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Poseidon23SeparatorConformance
