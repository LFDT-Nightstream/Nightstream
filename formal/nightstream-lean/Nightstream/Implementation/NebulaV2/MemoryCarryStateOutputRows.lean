import Nightstream.Implementation.NebulaV2.MemoryCarryPoseidonBinding
import Nightstream.Implementation.NebulaV2.StateOutputPoseidonRows

/-!
Contract: exact composition of the V2 carry digest with the mandatory outer
stateful-with-Nebula state-output digest.

Assurance tier: implementation model and cryptographic primitive semantics.

Owns the structural four-lane wire identity between the carry Poseidon2
outputs and the outer frame links, composed row soundness, and honest row
completeness.

Does not own placement of non-memory recursive-state fields, either
collision-resistance assumption, the full recursive relation manifest, or
Rust conformance.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.MemoryCarryStateOutputRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.NebulaV2.MemoryCarryPoseidonRows
open Nightstream.Implementation.NebulaV2.StateOutputPoseidonRows

structure Layout where
  carry : MemoryCarryPoseidonRows.Layout
  stateOutput : StateOutputPoseidonRows.Layout

def rows (layout : Layout) : List Row :=
  MemoryCarryPoseidonRows.rows layout.carry ++
    StateOutputPoseidonRows.rows layout.stateOutput

structure Layout.Valid (layout : Layout) : Prop where
  carryValid : layout.carry.Valid
  stateOutputValid : layout.stateOutput.Valid
  exactCarryOutputColumns :
    layout.stateOutput.frame.carryDigestOutputColumn =
      fun lane => layout.carry.trace.outputColumns.getD lane.val 0

private theorem carry_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (MemoryCarryPoseidonRows.rows layout.carry) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem state_output_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (StateOutputPoseidonRows.rows layout.stateOutput) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

/-- End-to-end local state-output functionality from the exact accepted carry
bits through both fixed Poseidon2 sponges. -/
theorem output_columns_eq_stateDigest
    {layout : Layout} (valid : layout.Valid)
    {assignment : Nat → Nat} {block : MemoryCarryParser.Block}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed
      layout.carry.frame.packing.publicBits assignment block)
    (holds : Satisfies (rows layout) assignment) :
    ∀ lane : Fin 4,
      assignment
          (layout.stateOutput.trace.outputColumns.getD lane.val 0) =
        StateOutputPoseidonRows.pureDigest
          (StateOutputFrameRows.sourceFrame layout.stateOutput.frame assignment
            (MemoryCarryPoseidonRows.carryDigest block))
          lane.val := by
  have carryOutputs :=
    MemoryCarryPoseidonRows.output_columns_eq_carryDigest
      valid.carryValid canonical one placed (carry_rows_hold holds)
  have linkedCarryOutputs : ∀ lane,
      assignment
          (layout.stateOutput.frame.carryDigestOutputColumn lane) =
        MemoryCarryPoseidonRows.carryDigest block lane := by
    intro lane
    rw [valid.exactCarryOutputColumns]
    exact carryOutputs lane
  exact StateOutputPoseidonRows.output_columns_eq_pureDigest
    valid.stateOutputValid canonical one (state_output_rows_hold holds)
    (MemoryCarryPoseidonRows.carryDigest block) linkedCarryOutputs

structure Honest (layout : Layout) (assignment : Nat → Nat)
    (block : MemoryCarryParser.Block) : Prop where
  carry : MemoryCarryPoseidonRows.Honest layout.carry assignment block
  stateOutput : StateOutputPoseidonRows.Honest layout.stateOutput assignment

theorem rows_complete
    {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryCarryParser.Block}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (honest : Honest layout assignment block) :
    Satisfies (rows layout) assignment := by
  intro row member
  rw [rows, List.mem_append] at member
  rcases member with carryMember | stateMember
  · exact MemoryCarryPoseidonRows.rows_complete canonical one honest.carry
      row carryMember
  · exact StateOutputPoseidonRows.rows_complete canonical one
      honest.stateOutput row stateMember

end Nightstream.Implementation.NebulaV2.MemoryCarryStateOutputRows
