import Mathlib.Data.Fintype.Pi
import Nightstream.Protocol.NebulaV2.Encoding

/-!
Contract: exact protocol digest representation for Nebula V2.

Assurance tier: model-level.

Owns the four-lane Poseidon2 digest type, canonical Goldilocks lane domain,
lane order, and total serialized bit width.

Does not own Poseidon2 evaluation, collision resistance, byte framing,
generated rows, or Rust parsing.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Protocol.NebulaV2.Digest

open Nightstream.Protocol.NebulaV2.ShiftedTernary41V1

def laneCount : Nat := 4
def laneBitWidth : Nat := 64
def serializedBitCount : Nat := laneCount * laneBitWidth

/-- One Poseidon2 digest in lane order `0..3`. Canonicality is carried by the
lane subtype and cannot be omitted by a caller. -/
@[ext] structure Value where
  lanes : Fin laneCount → CanonicalGoldilocks

instance : DecidableEq Value := by
  intro left right
  letI : ∀ _ : Fin laneCount, DecidableEq CanonicalGoldilocks :=
    fun _ => inferInstance
  letI : DecidableEq (Fin laneCount → CanonicalGoldilocks) :=
    Fintype.decidablePiFintype
  exact decidable_of_iff (left.lanes = right.lanes)
    ⟨fun equal => Value.ext equal,
      fun equal => congrArg Value.lanes equal⟩

theorem serializedBitCount_eq : serializedBitCount = 256 := by
  decide

theorem lane_lt_modulus (digest : Value) (lane : Fin laneCount) :
    (digest.lanes lane).val < modulus :=
  (digest.lanes lane).property

theorem lanes_injective : Function.Injective Value.lanes := by
  intro left right equal
  exact Value.ext equal

end Nightstream.Protocol.NebulaV2.Digest
