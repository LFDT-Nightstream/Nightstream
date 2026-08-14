import Nightstream.Implementation.Nebula.Memory.Carry.HashFrame

/-!
Contract: collision reduction from the exact carry-hash frame to parsed V2
carry values.

Assurance tier: implementation model and cryptographic boundary.

Owns the reduction from equal recomputed carry digests to equal successfully
parsed typed carries or the explicit carry-hash collision event. It also keeps
a constant-hash countermodel to prevent frame injectivity from being confused
with collision resistance.

Does not own Poseidon2 security, permutation rows, or the outer state hash.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.MemoryCarryHashBinding

open Nightstream.Implementation.Nebula.MemoryCarryHashFrame
open Nightstream.Protocol.Nebula

private theorem parsed_values_eq_of_input_eq
    {Input Value : Type} (parse : Input → Option Value)
    {leftInput rightInput : Input} {leftValue rightValue : Value}
    (leftAccepted : parse leftInput = some leftValue)
    (rightAccepted : parse rightInput = some rightValue)
    (inputEqual : leftInput = rightInput) : leftValue = rightValue := by
  subst rightInput
  rw [leftAccepted] at rightAccepted
  exact Option.some.inj rightAccepted

/-- Equal carry digests bind the complete parsed typed carry unless the
concrete hash has a collision on two distinct canonical carry frames. -/
theorem parsed_value_eq_or_collision
    {HashDigest : Type} [DecidableEq MemoryCarryParser.Block]
    (hash : Hash HashDigest)
    {headers : FPrime.ChainHeaders Digest.Value}
    {leftBlock rightBlock : MemoryCarryParser.Block}
    {leftValue rightValue : MemoryCarryCodec.Value}
    (leftAccepted : MemoryCarryParser.parse headers leftBlock = some leftValue)
    (rightAccepted : MemoryCarryParser.parse headers rightBlock = some rightValue)
    (equal : digest hash leftBlock = digest hash rightBlock) :
    leftValue = rightValue ∨ Collision hash := by
  rcases block_eq_or_collision hash leftBlock rightBlock equal with
    blockEqual | collision
  · exact Or.inl (parsed_values_eq_of_input_eq
      (MemoryCarryParser.parse headers) leftAccepted rightAccepted blockEqual)
  · exact Or.inr collision

def constantHash {Digest : Type} (value : Digest) : Hash Digest :=
  fun _ => value

@[simp] theorem constantHash_apply
    {Digest : Type} (value : Digest) (input : List Nat) :
    constantHash value input = value := rfl

/-- Frame injectivity is not hash security. The constant-hash countermodel
returns one digest for every pair of complete carry frames. -/
theorem constant_hash_ignores_frames
    {Digest : Type} (value : Digest)
    (left right : MemoryCarryParser.Block) :
    digest (constantHash value) left = digest (constantHash value) right := by
  change
    constantHash value (frame left) = constantHash value (frame right)
  rw [constantHash_apply, constantHash_apply]

end Nightstream.Implementation.Nebula.MemoryCarryHashBinding
