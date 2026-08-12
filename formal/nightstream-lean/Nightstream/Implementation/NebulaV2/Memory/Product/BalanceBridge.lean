import Nightstream.Implementation.NebulaV2.Memory.Claim.ProductUpdate
import Nightstream.Implementation.NebulaV2.Memory.Carry.Codec
import Nightstream.Implementation.NebulaV2.Memory.Product.BalanceRows

/-!
Contract: exact balance bridge from executable SuperNeo products to the
independent mathematical challenge field.

Assurance tier: implementation-to-protocol bridge.

Owns the equivalence between the two concrete `K.mul` close equations and
`ProductState.Balanced` after the exact field isomorphism used by the product
update theorem.

Does not own balance rows, product-update rows, challenge derivation, or a
fingerprint probability bound.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.MemoryProductBalanceBridge

open Nightstream.Implementation.NebulaV2.ConcreteField
open Nightstream.Implementation.NebulaV2.MemoryClaimProductUpdate
open Nightstream.Implementation.NebulaV2.MemoryProductBalanceRows
open Nightstream.Protocol.NebulaV2.ProductState
open Nightstream.SuperNeo.Concrete

/-- Concrete close balance is exactly mathematical close balance after the
same field isomorphism used for every row-derived product update. -/
theorem concreteBalanced_iff_mapped
    (products : State K) :
    ConcreteBalanced products ↔ Balanced (mapState products) := by
  constructor
  · intro concrete repetition
    have mapped := congrArg superNeoEquiv (concrete repetition)
    simpa [Balanced, Four.Balanced, mapState, mapFour,
      superNeoEquiv_mul] using mapped
  · intro mapped repetition
    apply superNeoEquiv.injective
    have atRepetition := mapped repetition
    simpa [Balanced, Four.Balanced, mapState, mapFour,
      superNeoEquiv_mul] using atRepetition

@[simp]
theorem mapState_oneProductsK :
    mapState MemoryCarryCodec.oneProductsK =
      (Nightstream.Protocol.NebulaV2.ProductState.one :
        State ChallengeField) := by
  funext repetition
  apply Four.ext <;>
    change superNeoEquiv K.one = (1 : ChallengeField) <;>
    exact superNeoEquiv_one

end Nightstream.Implementation.NebulaV2.MemoryProductBalanceBridge
