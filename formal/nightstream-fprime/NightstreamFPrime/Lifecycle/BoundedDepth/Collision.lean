import NightstreamFPrime.Lifecycle.BoundedDepth.State

/-! The Poseidon2 collision that a finder outputs, and the input pair that a collision of two
bounded-depth states names (`StateCollision`). `collisionInputs` reads that pair from the two
states: the two state preimages, or the two slot preimages of one slot. It gives a pair for every
state collision (`collisionInputs_isSome`). The random-oracle finder
(`Lifecycle.BoundedDepth.RandomOracle.HashSolver`) and the IVC finder
(`Export.BoundedDepth.IvcSecurity.finder`) output these pairs. -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.BoundedDepth

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Lifecycle.PaperAlgebra

/-- A Poseidon2 collision: two different inputs of one length with one hash. The finder
outputs inputs of one length; the `10*` padding makes every collision of `Poseidon2.hash` a
sponge collision (`Spec.Sponge.collision_blocks`). -/
def PoseidonCollision : Type :=
  {pair : List F × List F // pair.1 ≠ pair.2 ∧ pair.1.length = pair.2.length ∧
    Poseidon2.hash pair.1 = Poseidon2.hash pair.2}

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ RingCarrierLayout.carrierWidth logicalWidth}

/-- The slot preimages of one slot of two states. -/
def slotInputs (left right : State logicalWidth publicFits) (slot : Fin 2) : List F × List F :=
  (slotDomainChunk ++ serializeSlot (left.slots slot),
    slotDomainChunk ++ serializeSlot (right.slots slot))

/-- Different slot serializations with one slot digest are a Poseidon2 collision on the slot
preimages. -/
theorem slotInputs_collide {left right : State logicalWidth publicFits} {slot : Fin 2}
    (collides : serializeSlot (left.slots slot) ≠ serializeSlot (right.slots slot) ∧
      slotDigest (left.slots slot) = slotDigest (right.slots slot)) :
    (slotInputs left right slot).1 ≠ (slotInputs left right slot).2 ∧
      (slotInputs left right slot).1.length = (slotInputs left right slot).2.length ∧
      Poseidon2.hash (slotInputs left right slot).1 =
        Poseidon2.hash (slotInputs left right slot).2 :=
  ⟨fun same => collides.1 (List.append_cancel_left same),
    by simp only [slotInputs, List.length_append, serializeSlot_length_eq], collides.2⟩

/-- The Poseidon2 input pair that a collision of two states names: the state preimages, else
the slot preimages of slot 0, else those of slot 1. -/
noncomputable def collisionInputs (left right : State logicalWidth publicFits) :
    Option PoseidonCollision :=
  if states : statePreimage left ≠ statePreimage right ∧
      (statePreimage left).length = (statePreimage right).length ∧
      stateHash left = stateHash right then
    some ⟨(statePreimage left, statePreimage right), states⟩
  else if first : serializeSlot (left.slots 0) ≠ serializeSlot (right.slots 0) ∧
      slotDigest (left.slots 0) = slotDigest (right.slots 0) then
    some ⟨slotInputs left right 0, slotInputs_collide first⟩
  else if second : serializeSlot (left.slots 1) ≠ serializeSlot (right.slots 1) ∧
      slotDigest (left.slots 1) = slotDigest (right.slots 1) then
    some ⟨slotInputs left right 1, slotInputs_collide second⟩
  else none

/-- Every collision of two states gives a Poseidon2 input pair. -/
theorem collisionInputs_isSome {left right : State logicalWidth publicFits}
    (collision : StateCollision left right) : (collisionInputs left right).isSome := by
  unfold collisionInputs
  by_cases states : statePreimage left ≠ statePreimage right ∧
      (statePreimage left).length = (statePreimage right).length ∧
      stateHash left = stateHash right
  · rw [dite_eq_left states]
    rfl
  · rw [dite_eq_right states]
    rcases collision with direct | ⟨slot, collides⟩
    · exact absurd direct states
    · by_cases first : serializeSlot (left.slots 0) ≠ serializeSlot (right.slots 0) ∧
          slotDigest (left.slots 0) = slotDigest (right.slots 0)
      · rw [dite_eq_left first]
        rfl
      · rw [dite_eq_right first]
        have second : serializeSlot (left.slots 1) ≠ serializeSlot (right.slots 1) ∧
            slotDigest (left.slots 1) = slotDigest (right.slots 1) := by
          fin_cases slot
          · exact absurd collides first
          · exact collides
        rw [dite_eq_left second]
        rfl

end NightstreamFPrime.Lifecycle.BoundedDepth
