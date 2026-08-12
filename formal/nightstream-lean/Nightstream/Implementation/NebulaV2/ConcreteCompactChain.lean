import Nightstream.Implementation.NebulaV2.CompactChainPoseidonRows

/-!
Contract: one exact protocol-level compact-chain hash for Nebula V2.

This module converts the typed `CompactChain.HashInput` into the canonical
field frame and evaluates the fixed Poseidon2 sponge. The result is a
canonical four-lane Goldilocks digest by construction.

It does not assume collision resistance. A collision in `hash` remains the
explicit `CompactChain.HashCollision hash` bad event.

Assurance tier: deterministic protocol-to-Poseidon2 bridge.

Emits constraints: no; the row relation is owned by
`CompactChainPoseidonRows`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.ConcreteCompactChain

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.NebulaV2.CompactChainHashFrame
open Nightstream.Implementation.NebulaV2.CompactChainPoseidonRows
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.CompactChain

/-- The protocol input and the numeric frame have the same typed fields and
constructor separation. -/
def toFrame : HashInput Digest.Value Digest.Value ->
    CompactChainHashFrame.Input
  | .header role profile plan => .header role profile plan
  | .leaf role profile plan token => .leaf role profile plan token
  | .link role index prior leaf => .link role index prior leaf

@[simp] theorem toFrame_header
    (role : CompactCommit.Role) (profile : Profile.Identity)
    (plan : Digest.Value) :
    toFrame (.header role profile plan) =
      CompactChainHashFrame.Input.header role profile plan := rfl

@[simp] theorem toFrame_leaf
    (role : CompactCommit.Role) (profile : Profile.Identity)
    (plan : Digest.Value) (token : CompactCommit.Token) :
    toFrame (.leaf role profile plan token) =
      CompactChainHashFrame.Input.leaf role profile plan token := rfl

@[simp] theorem toFrame_link
    (role : CompactCommit.Role)
    (index : Fin Lifecycle.claimsPerSegment)
    (prior leaf : Digest.Value) :
    toFrame (.link role index prior leaf) =
      CompactChainHashFrame.Input.link role index prior leaf := rfl

/-- No typed input distinction is lost before field framing. -/
theorem toFrame_injective : Function.Injective toFrame := by
  intro left right equal
  cases left <;> cases right <;> simp_all [toFrame]

/-- The complete numeric frame is injective on protocol hash inputs. -/
theorem encodedFrame_injective :
    Function.Injective (CompactChainHashFrame.encode ∘ toFrame) :=
  CompactChainHashFrame.encode_injective.comp toFrame_injective

/-- Exact canonical Poseidon2 digest used by `CompactChain.chainRoot`. -/
def hash (input : HashInput Digest.Value Digest.Value) : Digest.Value where
  lanes := fun lane =>
    ⟨pureHash (toFrame input) lane.val, by
      have canonical := runValueRounds_canonical
        (representativeRounds (toFrame input))
        (CompactChainHashFrame.encode (toFrame input))
        (fun _ => 0) (by
          intro stateLane
          norm_num [goldilocksP]) lane.val
      simpa [pureHash, goldilocksP,
        ShiftedTernary41V1.modulus] using canonical⟩

@[simp] theorem hash_lane
    (input : HashInput Digest.Value Digest.Value) (lane : Fin 4) :
    (hash input).lanes lane =
      ⟨pureHash (toFrame input) lane.val,
        by
          have canonical := runValueRounds_canonical
            (representativeRounds (toFrame input))
            (CompactChainHashFrame.encode (toFrame input))
            (fun _ => 0) (by
              intro stateLane
              norm_num [goldilocksP]) lane.val
          simpa [pureHash, goldilocksP,
            ShiftedTernary41V1.modulus] using canonical⟩ := rfl

@[simp] theorem hash_lane_value
    (input : HashInput Digest.Value Digest.Value) (lane : Fin 4) :
    ((hash input).lanes lane).val =
      pureHash (toFrame input) lane.val := rfl

/-- The concrete hash has no hidden injectivity premise. Either it is
injective on the typed domain or the standard named collision event occurs. -/
theorem injective_or_named_collision :
    Function.Injective hash ∨ HashCollision hash :=
  CompactChain.hash_injective_or_collision hash

end Nightstream.Implementation.NebulaV2.ConcreteCompactChain
