import Nightstream.Protocol.NebulaV2.CompactChain

set_option autoImplicit false

namespace Nightstream.Tests.NebulaV2CompactChain

open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.CompactChain
open Nightstream.Protocol.NebulaV2.CompactCommit
open Nightstream.Protocol.NebulaV2.Lifecycle
open Nightstream.Protocol.NebulaV2.SequenceBinding

example {Plan Seed Digest : Type}
    (hash : HashInput Plan Digest → Digest)
    (key : Key Plan Seed)
    (collision : RootCollision (chainRoot hash key)) :
    HashCollision hash ∨ TokenCollision key :=
  root_collision_implies_hash_or_token_collision hash key collision

example {Plan Seed Digest : Type}
    (hash : HashInput Plan Digest → Digest)
    (key : Key Plan Seed)
    (collision : RootCollision (chainRoot hash key)) :
    HashCollision hash ∨
      AnyPrimaryBindingFailure key ∨ AnyShortBindingFailure key :=
  root_collision_implies_hash_or_ajtai_failure hash key collision

def constantHash (_input : HashInput Unit Nat) : Nat := 0

/-- A constant typed hash is rejected by the exact chain model through a
named hash collision. Typed framing alone is not a collision-resistance
proof. -/
theorem constant_hash_is_a_named_collision : HashCollision constantHash := by
  let left : HashInput Unit Nat :=
    .header .operations Profile.v2 ()
  let right : HashInput Unit Nat :=
    .header .memory Profile.v2 ()
  refine ⟨left, right, ?_, rfl⟩
  intro equal
  have roleEqual : Role.operations = Role.memory := by
    injection equal
  exact roles_distinct roleEqual

end Nightstream.Tests.NebulaV2CompactChain
