import Nightstream.Implementation.NebulaV2.MemoryCarryPoseidonRows
import Nightstream.Implementation.NebulaV2.MemoryCarryHashBinding

/-!
Contract: collision reduction for the concrete V2 Poseidon2 carry digest.

Assurance tier: implementation model and cryptographic boundary.

Owns the exact specialization of the generic lossless-frame reduction to the
fixed 31-round Poseidon2 sponge. Equal concrete carry digests yield equal
accepted blocks and typed carries, or the exact named Poseidon2 collision
event.

Does not prove Poseidon2 collision resistance, outer state-hash security, or
Rust conformance.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.MemoryCarryPoseidonBinding

open Nightstream.Implementation.NebulaV2.MemoryCarryHashFrame
open Nightstream.Implementation.NebulaV2.MemoryCarryPoseidonRows
open Nightstream.Protocol.NebulaV2

abbrev Digest := Fin 4 → Nat

def poseidonHash : Hash Digest :=
  fun values lane => pureDigest values lane.val

theorem framed_digest_eq (block : MemoryCarryParser.Block) :
    MemoryCarryHashFrame.digest poseidonHash block = carryDigest block := rfl

abbrev PoseidonCollision : Prop :=
  MemoryCarryHashFrame.Collision poseidonHash

/-- Equal fixed Poseidon2 carry digests bind every authority-bearing carry
bit, unless the exact framed sponge collides. -/
theorem block_eq_or_poseidon_collision
    [DecidableEq MemoryCarryParser.Block]
    (left right : MemoryCarryParser.Block)
    (equal : carryDigest left = carryDigest right) :
    left = right ∨ PoseidonCollision := by
  apply MemoryCarryHashFrame.block_eq_or_collision poseidonHash left right
  simpa only [framed_digest_eq] using equal

/-- Parsed typed carry binding for the exact V2 Poseidon2 digest. -/
theorem parsed_value_eq_or_poseidon_collision
    [DecidableEq MemoryCarryParser.Block]
    {headers : FPrime.ChainHeaders Digest.Value}
    {leftBlock rightBlock : MemoryCarryParser.Block}
    {leftValue rightValue : MemoryCarryCodec.Value}
    (leftAccepted : MemoryCarryParser.parse headers leftBlock = some leftValue)
    (rightAccepted : MemoryCarryParser.parse headers rightBlock = some rightValue)
    (equal : carryDigest leftBlock = carryDigest rightBlock) :
    leftValue = rightValue ∨ PoseidonCollision := by
  apply MemoryCarryHashBinding.parsed_value_eq_or_collision poseidonHash
    leftAccepted rightAccepted
  simpa only [framed_digest_eq] using equal

end Nightstream.Implementation.NebulaV2.MemoryCarryPoseidonBinding
