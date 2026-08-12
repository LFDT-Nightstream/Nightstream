import Nightstream.Implementation.NebulaV2.StateOutputAuthorityRows
import Nightstream.Implementation.NebulaV2.StateOutputRowCensus

/-!
Contract: complete local V2 state-output relation from one exact memory-carry
bit block and one exact typed non-memory recursive-state payload.

Assurance tier: implementation model and cryptographic primitive semantics.

Owns composition of the 3,433-bit carry digest, the 26-field authority block,
the mandatory stateful-with-Nebula outer frame, both Poseidon2 sponges, exact
functional soundness, honest completeness, and the local row census.

Does not own the transition that computes the recursive-state payload,
absolute generated-row placement, collision resistance, the global recursive
relation manifest, or Rust conformance.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.AuthoritativeStateOutputRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.NebulaV2

structure Layout where
  hash : MemoryCarryStateOutputRows.Layout
  authority : StateOutputAuthorityRows.Layout

def rows (layout : Layout) : List Row :=
  MemoryCarryStateOutputRows.rows layout.hash ++
    StateOutputAuthorityRows.rows layout.authority

/-- All fields are row, wire, schedule, or column-identity facts. -/
structure Layout.Valid (layout : Layout) : Prop where
  hashValid : layout.hash.Valid
  authorityValid : layout.authority.Valid
  exactOuterFrame :
    layout.authority.frame = layout.hash.stateOutput.frame

private theorem hash_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (MemoryCarryStateOutputRows.rows layout.hash) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem authority_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (StateOutputAuthorityRows.rows layout.authority) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

/-- The complete local relation hashes exactly one typed non-memory payload
and the carry digest recomputed from the same 3,433 parser-owned bits. -/
theorem output_columns_eq_typed_stateDigest
    {layout : Layout} (valid : layout.Valid)
    {assignment : Nat → Nat} {block : MemoryCarryParser.Block}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed
      layout.hash.carry.frame.packing.publicBits assignment block)
    (holds : Satisfies (rows layout) assignment) :
    ∀ lane : Fin 4,
      assignment
          (layout.hash.stateOutput.trace.outputColumns.getD lane.val 0) =
        StateOutputPoseidonRows.pureDigest
          (StateOutputAuthorityRows.fullFrame
            (StateOutputAuthorityRows.payload layout.authority assignment)
            (MemoryCarryPoseidonRows.carryDigest block))
          lane.val := by
  have hashOutput :=
    MemoryCarryStateOutputRows.output_columns_eq_stateDigest
      valid.hashValid canonical one placed (hash_rows_hold holds)
  have exactFrame :
      StateOutputFrameRows.sourceFrame layout.hash.stateOutput.frame assignment
          (MemoryCarryPoseidonRows.carryDigest block) =
        StateOutputAuthorityRows.fullFrame
          (StateOutputAuthorityRows.payload layout.authority assignment)
          (MemoryCarryPoseidonRows.carryDigest block) := by
    rw [← valid.exactOuterFrame]
    exact StateOutputAuthorityRows.sourceFrame_eq_fullFrame
      valid.authorityValid canonical one (authority_rows_hold holds) _
  intro lane
  rw [hashOutput lane, exactFrame]

theorem rows_length_exact
    {layout : Layout} (valid : layout.Valid) :
    (rows layout).length = 24497 := by
  rw [rows, List.length_append,
    StateOutputRowCensus.composed_rows_length valid.hashValid,
    StateOutputAuthorityRows.rows_length_exact]

structure Honest (layout : Layout) (assignment : Nat → Nat)
    (block : MemoryCarryParser.Block)
    (payload : StateOutputAuthorityRows.Payload) : Prop where
  hash : MemoryCarryStateOutputRows.Honest layout.hash assignment block
  authority : StateOutputAuthorityRows.Honest layout.authority assignment payload

theorem rows_complete
    {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryCarryParser.Block}
    {payload : StateOutputAuthorityRows.Payload}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (honest : Honest layout assignment block payload) :
    Satisfies (rows layout) assignment := by
  intro row member
  rw [rows, List.mem_append] at member
  rcases member with hashMember | authorityMember
  · exact MemoryCarryStateOutputRows.rows_complete canonical one honest.hash
      row hashMember
  · exact StateOutputAuthorityRows.rows_complete one honest.authority
      row authorityMember

end Nightstream.Implementation.NebulaV2.AuthoritativeStateOutputRows
