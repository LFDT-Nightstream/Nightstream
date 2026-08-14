import Nightstream.Implementation.Nebula.Commitment.Compact.ConcreteChain

/-! Regression surface and hostile hash countermodel for the exact Nebula V2
compact-chain function. -/

set_option autoImplicit false

namespace tests.NebulaConcreteCompactChain

open Nightstream.Implementation.Nebula
open Nightstream.Protocol.Nebula

#check ConcreteCompactChain.toFrame
#check ConcreteCompactChain.toFrame_injective
#check ConcreteCompactChain.encodedFrame_injective
#check ConcreteCompactChain.hash
#check ConcreteCompactChain.hash_lane
#check ConcreteCompactChain.hash_lane_value
#check ConcreteCompactChain.injective_or_named_collision

def zeroDigest : Digest.Value where
  lanes := fun _ => ⟨0, by decide⟩

def constantHash :
    CompactChain.HashInput Digest.Value Digest.Value → Digest.Value :=
  fun _ => zeroDigest

/-! Exact framing alone does not imply collision resistance. A constant hash
has a named collision between the operations and memory header domains. -/
theorem constantHash_has_named_collision :
    CompactChain.HashCollision constantHash := by
  refine ⟨.header .operations Profile.v2 zeroDigest,
    .header .memory Profile.v2 zeroDigest, ?_, rfl⟩
  simp [CompactCommit.roles_distinct]

end tests.NebulaConcreteCompactChain
