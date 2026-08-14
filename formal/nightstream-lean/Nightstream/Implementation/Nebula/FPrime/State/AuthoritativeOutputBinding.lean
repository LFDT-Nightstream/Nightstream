import Nightstream.Implementation.Nebula.FPrime.State.AuthoritativeOutputRows
import Nightstream.Implementation.Nebula.FPrime.State.OutputPoseidonBinding

/-!
Contract: typed two-stage binding reduction for the complete local V2
state-output relation.

Assurance tier: implementation model and cryptographic boundary.

Owns reduction of equal outer state outputs to equality of all 26 typed
non-memory fields and the complete 3,433-bit carry, or to one exact inner or
outer Poseidon2 collision event. It also lifts that reduction to two
satisfying complete local row assignments.

Does not prove Poseidon2 collision resistance, the recursive transition, the
global generated relation manifest, terminal proof soundness, or Rust
conformance.

Emits constraints: no new rows. It composes exact emitted-row theorems.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.AuthoritativeStateOutputBinding

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.Nebula

abbrev Digest := Fin 4 → Nat

def typedFrame (payload : StateOutputAuthorityRows.Payload)
    (block : MemoryCarryParser.Block) : List Nat :=
  StateOutputAuthorityRows.fullFrame payload
    (MemoryCarryPoseidonRows.carryDigest block)

def typedDigest (payload : StateOutputAuthorityRows.Payload)
    (block : MemoryCarryParser.Block) : Digest :=
  StateOutputPoseidonBinding.outerHash (typedFrame payload block)

/-- Pure typed reduction. Canonicality constructs the exact collision-event
messages; it does not assume either collision event is absent. -/
theorem typed_authority_eq_or_two_stage_collision
    [DecidableEq MemoryCarryParser.Block]
    {leftPayload rightPayload : StateOutputAuthorityRows.Payload}
    {leftBlock rightBlock : MemoryCarryParser.Block}
    (leftLength : (typedFrame leftPayload leftBlock).length = 32)
    (rightLength : (typedFrame rightPayload rightBlock).length = 32)
    (leftCanonical : ∀ value ∈ typedFrame leftPayload leftBlock,
      value < goldilocksP)
    (rightCanonical : ∀ value ∈ typedFrame rightPayload rightBlock,
      value < goldilocksP)
    (equal : typedDigest leftPayload leftBlock =
      typedDigest rightPayload rightBlock) :
    (leftPayload = rightPayload ∧ leftBlock = rightBlock) ∨
      StateOutputPoseidonBinding.OuterCollision ∨
      MemoryCarryPoseidonBinding.PoseidonCollision := by
  let leftFrame : StateOutputPoseidonBinding.CanonicalFrame :=
    ⟨typedFrame leftPayload leftBlock, leftLength, leftCanonical⟩
  let rightFrame : StateOutputPoseidonBinding.CanonicalFrame :=
    ⟨typedFrame rightPayload rightBlock, rightLength, rightCanonical⟩
  have digestEqual :
      StateOutputPoseidonBinding.digest leftFrame =
        StateOutputPoseidonBinding.digest rightFrame := by
    simpa [StateOutputPoseidonBinding.digest,
      StateOutputPoseidonBinding.outerHash, typedDigest, leftFrame, rightFrame]
      using equal
  rcases StateOutputPoseidonBinding.frame_values_eq_or_outer_collision
      leftFrame rightFrame digestEqual with frameEqual | outerCollision
  · have recovered :=
      StateOutputAuthorityRows.payload_and_carry_eq_of_fullFrame_eq frameEqual
    rcases MemoryCarryPoseidonBinding.block_eq_or_poseidon_collision
        leftBlock rightBlock recovered.2 with blockEqual | innerCollision
    · exact Or.inl ⟨recovered.1, blockEqual⟩
    · exact Or.inr (Or.inr innerCollision)
  · exact Or.inr (Or.inl outerCollision)

private theorem hash_rows_hold
    {layout : AuthoritativeStateOutputRows.Layout}
    {assignment : Nat → Nat}
    (holds : Satisfies (AuthoritativeStateOutputRows.rows layout) assignment) :
    Satisfies (MemoryCarryStateOutputRows.rows layout.hash) assignment := by
  intro row member
  exact holds row (by simp [AuthoritativeStateOutputRows.rows, member])

private theorem carry_rows_hold
    {layout : AuthoritativeStateOutputRows.Layout}
    {assignment : Nat → Nat}
    (holds : Satisfies (AuthoritativeStateOutputRows.rows layout) assignment) :
    Satisfies (MemoryCarryPoseidonRows.rows layout.hash.carry) assignment := by
  intro row member
  exact holds row (by simp [AuthoritativeStateOutputRows.rows,
    MemoryCarryStateOutputRows.rows, member])

private theorem authority_rows_hold
    {layout : AuthoritativeStateOutputRows.Layout}
    {assignment : Nat → Nat}
    (holds : Satisfies (AuthoritativeStateOutputRows.rows layout) assignment) :
    Satisfies (StateOutputAuthorityRows.rows layout.authority) assignment := by
  intro row member
  exact holds row (by simp [AuthoritativeStateOutputRows.rows, member])

/-- Satisfying complete state-output rows make the normalized typed frame a
canonical Goldilocks message. This public form lets the lifetime owner compare
states produced by different generated-column layouts. -/
theorem typedFrame_canonical_of_rows
    {layout : AuthoritativeStateOutputRows.Layout} (valid : layout.Valid)
    {assignment : Nat → Nat} {block : MemoryCarryParser.Block}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed
      layout.hash.carry.frame.packing.publicBits assignment block)
    (holds : Satisfies (AuthoritativeStateOutputRows.rows layout) assignment) :
    ∀ value ∈
        typedFrame (StateOutputAuthorityRows.payload layout.authority assignment)
          block,
      value < goldilocksP := by
  have carryOutputs :=
    MemoryCarryPoseidonRows.output_columns_eq_carryDigest
      valid.hashValid.carryValid canonical one placed (carry_rows_hold holds)
  have carryCanonical : ∀ lane,
      MemoryCarryPoseidonRows.carryDigest block lane < goldilocksP := by
    intro lane
    rw [← carryOutputs lane]
    exact canonical _
  have sourceCanonical :=
    StateOutputFrameRows.sourceFrame_canonical
      layout.authority.frame assignment
      (MemoryCarryPoseidonRows.carryDigest block)
      canonical carryCanonical
  have sourceEqual :=
    StateOutputAuthorityRows.sourceFrame_eq_fullFrame
      valid.authorityValid canonical one (authority_rows_hold holds)
      (MemoryCarryPoseidonRows.carryDigest block)
  intro value member
  apply sourceCanonical value
  rw [sourceEqual]
  exact member

/-- Two satisfying complete local relations with equal public digest lanes
bind every typed recursive-state field and every carry bit, except for the
two explicitly named Poseidon2 collision events. -/
theorem satisfying_rows_bind_typed_authority_or_collision
    [DecidableEq MemoryCarryParser.Block]
    {layout : AuthoritativeStateOutputRows.Layout} (valid : layout.Valid)
    {leftAssignment rightAssignment : Nat → Nat}
    {leftBlock rightBlock : MemoryCarryParser.Block}
    (leftCanonical : ∀ column, leftAssignment column < goldilocksP)
    (rightCanonical : ∀ column, rightAssignment column < goldilocksP)
    (leftOne : leftAssignment 0 = 1)
    (rightOne : rightAssignment 0 = 1)
    (leftPlaced : PublicBitBlock.Placed
      layout.hash.carry.frame.packing.publicBits leftAssignment leftBlock)
    (rightPlaced : PublicBitBlock.Placed
      layout.hash.carry.frame.packing.publicBits rightAssignment rightBlock)
    (leftHolds : Satisfies
      (AuthoritativeStateOutputRows.rows layout) leftAssignment)
    (rightHolds : Satisfies
      (AuthoritativeStateOutputRows.rows layout) rightAssignment)
    (equalOutputs : ∀ lane : Fin 4,
      leftAssignment
          (layout.hash.stateOutput.trace.outputColumns.getD lane.val 0) =
        rightAssignment
          (layout.hash.stateOutput.trace.outputColumns.getD lane.val 0)) :
    (StateOutputAuthorityRows.payload layout.authority leftAssignment =
        StateOutputAuthorityRows.payload layout.authority rightAssignment ∧
      leftBlock = rightBlock) ∨
      StateOutputPoseidonBinding.OuterCollision ∨
      MemoryCarryPoseidonBinding.PoseidonCollision := by
  let leftPayload :=
    StateOutputAuthorityRows.payload layout.authority leftAssignment
  let rightPayload :=
    StateOutputAuthorityRows.payload layout.authority rightAssignment
  have leftOutput :=
    AuthoritativeStateOutputRows.output_columns_eq_typed_stateDigest
      valid leftCanonical leftOne leftPlaced leftHolds
  have rightOutput :=
    AuthoritativeStateOutputRows.output_columns_eq_typed_stateDigest
      valid rightCanonical rightOne rightPlaced rightHolds
  have digestEqual : typedDigest leftPayload leftBlock =
      typedDigest rightPayload rightBlock := by
    funext lane
    simp only [typedDigest, typedFrame, leftPayload, rightPayload]
    unfold StateOutputPoseidonBinding.outerHash
    rw [← leftOutput lane, ← rightOutput lane]
    exact equalOutputs lane
  exact typed_authority_eq_or_two_stage_collision
    (StateOutputAuthorityRows.fullFrame_length _ _)
    (StateOutputAuthorityRows.fullFrame_length _ _)
    (typedFrame_canonical_of_rows valid leftCanonical leftOne leftPlaced
      leftHolds)
    (typedFrame_canonical_of_rows valid rightCanonical rightOne rightPlaced
      rightHolds)
    digestEqual

end Nightstream.Implementation.Nebula.AuthoritativeStateOutputBinding
