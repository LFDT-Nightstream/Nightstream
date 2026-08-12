import Nightstream.Implementation.NebulaV2.FPrime.State.MemoryCarryOutputRows

/-!
Contract: two-stage collision reduction for the exact V2 carry and outer
state-output Poseidon2 sponges.

Assurance tier: implementation model and cryptographic boundary.

Owns a canonical 32-field outer-frame type, reduction of equal outer digests
to equal complete source frames or an explicit outer collision, recovery of
the complete carry block through the inner collision reduction, and a
row-level theorem for two satisfying local executions.

Does not prove either collision-resistance assumption, placement of the 26
non-memory payload fields into the complete recursive state, the global
recursive relation manifest, or Rust conformance.

Emits constraints: no new rows. It composes exact emitted-row theorems.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.StateOutputPoseidonBinding

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.NebulaV2.MemoryCarryPoseidonRows
open Nightstream.Implementation.NebulaV2.MemoryCarryPoseidonBinding
open Nightstream.Implementation.NebulaV2.MemoryCarryStateOutputRows
open Nightstream.Implementation.NebulaV2.StateOutputFrameRows
open Nightstream.Implementation.NebulaV2.StateOutputPoseidonRows

abbrev Digest := Fin 4 → Nat

def CanonicalFrame :=
  { values : List Nat //
    values.length = 32 ∧
      ∀ value ∈ values, value < goldilocksP }

def outerHash (values : List Nat) : Digest :=
  fun lane => StateOutputPoseidonRows.pureDigest values lane.val

def digest (frame : CanonicalFrame) : Digest :=
  outerHash frame.val

def OuterCollision : Prop :=
  ∃ left right : CanonicalFrame,
    left.val ≠ right.val ∧ digest left = digest right

theorem frame_values_eq_or_outer_collision
    (left right : CanonicalFrame)
    (equal : digest left = digest right) :
    left.val = right.val ∨ OuterCollision := by
  by_cases same : left.val = right.val
  · exact Or.inl same
  · exact Or.inr ⟨left, right, same, equal⟩

def stateFrame (layout : StateOutputFrameRows.Layout)
    (assignment : Nat → Nat) (block : MemoryCarryParser.Block) : List Nat :=
  StateOutputFrameRows.sourceFrame layout assignment
    (MemoryCarryPoseidonRows.carryDigest block)

def stateDigest (layout : StateOutputFrameRows.Layout)
    (assignment : Nat → Nat) (block : MemoryCarryParser.Block) : Digest :=
  outerHash (stateFrame layout assignment block)

theorem stateFrame_length
    (layout : StateOutputFrameRows.Layout) (assignment : Nat → Nat)
    (block : MemoryCarryParser.Block) :
    (stateFrame layout assignment block).length = 32 :=
  StateOutputFrameRows.sourceFrame_length _ _ _

theorem stateFrame_canonical
    (layout : StateOutputFrameRows.Layout) (assignment : Nat → Nat)
    (block : MemoryCarryParser.Block)
    (canonicalAssignment : ∀ column, assignment column < goldilocksP)
    (canonicalCarry : ∀ lane,
      MemoryCarryPoseidonRows.carryDigest block lane < goldilocksP) :
    ∀ value ∈ stateFrame layout assignment block,
      value < goldilocksP :=
  StateOutputFrameRows.sourceFrame_canonical layout assignment
    (MemoryCarryPoseidonRows.carryDigest block)
    canonicalAssignment canonicalCarry

/-- Equal complete state outputs recover both the full 32-field outer frame
and the complete carry block, unless one of the two exact Poseidon2 stages
collides. -/
theorem authority_eq_or_two_stage_collision
    [DecidableEq MemoryCarryParser.Block]
    {layout : StateOutputFrameRows.Layout}
    {leftAssignment rightAssignment : Nat → Nat}
    {leftBlock rightBlock : MemoryCarryParser.Block}
    (leftCanonical : ∀ column, leftAssignment column < goldilocksP)
    (rightCanonical : ∀ column, rightAssignment column < goldilocksP)
    (leftCarryCanonical : ∀ lane,
      MemoryCarryPoseidonRows.carryDigest leftBlock lane < goldilocksP)
    (rightCarryCanonical : ∀ lane,
      MemoryCarryPoseidonRows.carryDigest rightBlock lane < goldilocksP)
    (equal : stateDigest layout leftAssignment leftBlock =
      stateDigest layout rightAssignment rightBlock) :
    (stateFrame layout leftAssignment leftBlock =
        stateFrame layout rightAssignment rightBlock ∧
      leftBlock = rightBlock) ∨
      OuterCollision ∨ PoseidonCollision := by
  let leftFrame : CanonicalFrame :=
    ⟨stateFrame layout leftAssignment leftBlock,
      stateFrame_length layout leftAssignment leftBlock,
      stateFrame_canonical layout leftAssignment leftBlock
        leftCanonical leftCarryCanonical⟩
  let rightFrame : CanonicalFrame :=
    ⟨stateFrame layout rightAssignment rightBlock,
      stateFrame_length layout rightAssignment rightBlock,
      stateFrame_canonical layout rightAssignment rightBlock
        rightCanonical rightCarryCanonical⟩
  have framedEqual : digest leftFrame = digest rightFrame := by
    simpa [digest, outerHash, stateDigest, leftFrame, rightFrame] using equal
  rcases frame_values_eq_or_outer_collision leftFrame rightFrame framedEqual with
    frameEqual | outerCollision
  · have carryEqual :
        MemoryCarryPoseidonRows.carryDigest leftBlock =
          MemoryCarryPoseidonRows.carryDigest rightBlock :=
      StateOutputFrameRows.carryDigest_eq_of_sourceFrame_eq frameEqual
    rcases MemoryCarryPoseidonBinding.block_eq_or_poseidon_collision
        leftBlock rightBlock carryEqual with blockEqual | carryCollision
    · exact Or.inl ⟨frameEqual, blockEqual⟩
    · exact Or.inr (Or.inr carryCollision)
  · exact Or.inr (Or.inl outerCollision)

private theorem carry_rows_hold
    {layout : MemoryCarryStateOutputRows.Layout}
    {assignment : Nat → Nat}
    (holds : Satisfies (MemoryCarryStateOutputRows.rows layout) assignment) :
    Satisfies (MemoryCarryPoseidonRows.rows layout.carry) assignment := by
  intro row member
  exact holds row (by simp [MemoryCarryStateOutputRows.rows, member])

/-- Two satisfying exact local relations with equal public state-output lanes
have equal complete frames and carries, or expose one of the two exact
collision events. Canonical carry inputs are derived from circuit outputs. -/
theorem satisfying_rows_bind_authority_or_collision
    [DecidableEq MemoryCarryParser.Block]
    {layout : MemoryCarryStateOutputRows.Layout} (valid : layout.Valid)
    {leftAssignment rightAssignment : Nat → Nat}
    {leftBlock rightBlock : MemoryCarryParser.Block}
    (leftCanonical : ∀ column, leftAssignment column < goldilocksP)
    (rightCanonical : ∀ column, rightAssignment column < goldilocksP)
    (leftOne : leftAssignment 0 = 1)
    (rightOne : rightAssignment 0 = 1)
    (leftPlaced : PublicBitBlock.Placed
      layout.carry.frame.packing.publicBits leftAssignment leftBlock)
    (rightPlaced : PublicBitBlock.Placed
      layout.carry.frame.packing.publicBits rightAssignment rightBlock)
    (leftHolds : Satisfies
      (MemoryCarryStateOutputRows.rows layout) leftAssignment)
    (rightHolds : Satisfies
      (MemoryCarryStateOutputRows.rows layout) rightAssignment)
    (equalOutputs : ∀ lane : Fin 4,
      leftAssignment
          (layout.stateOutput.trace.outputColumns.getD lane.val 0) =
        rightAssignment
          (layout.stateOutput.trace.outputColumns.getD lane.val 0)) :
    (stateFrame layout.stateOutput.frame leftAssignment leftBlock =
        stateFrame layout.stateOutput.frame rightAssignment rightBlock ∧
      leftBlock = rightBlock) ∨
      OuterCollision ∨ PoseidonCollision := by
  have leftCarryOutputs :=
    MemoryCarryPoseidonRows.output_columns_eq_carryDigest
      valid.carryValid leftCanonical leftOne leftPlaced
      (carry_rows_hold leftHolds)
  have rightCarryOutputs :=
    MemoryCarryPoseidonRows.output_columns_eq_carryDigest
      valid.carryValid rightCanonical rightOne rightPlaced
      (carry_rows_hold rightHolds)
  have leftCarryCanonical : ∀ lane,
      MemoryCarryPoseidonRows.carryDigest leftBlock lane < goldilocksP := by
    intro lane
    rw [← leftCarryOutputs lane]
    exact leftCanonical _
  have rightCarryCanonical : ∀ lane,
      MemoryCarryPoseidonRows.carryDigest rightBlock lane < goldilocksP := by
    intro lane
    rw [← rightCarryOutputs lane]
    exact rightCanonical _
  have leftStateOutput :=
    MemoryCarryStateOutputRows.output_columns_eq_stateDigest valid
      leftCanonical leftOne leftPlaced leftHolds
  have rightStateOutput :=
    MemoryCarryStateOutputRows.output_columns_eq_stateDigest valid
      rightCanonical rightOne rightPlaced rightHolds
  have digestEqual :
      stateDigest layout.stateOutput.frame leftAssignment leftBlock =
        stateDigest layout.stateOutput.frame rightAssignment rightBlock := by
    funext lane
    simp only [stateDigest, outerHash, stateFrame]
    rw [← leftStateOutput lane, ← rightStateOutput lane]
    exact equalOutputs lane
  exact authority_eq_or_two_stage_collision leftCanonical rightCanonical
    leftCarryCanonical rightCarryCanonical digestEqual

end Nightstream.Implementation.NebulaV2.StateOutputPoseidonBinding
