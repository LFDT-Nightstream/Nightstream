import Nightstream.Protocol.NebulaV2.CarryEncoding

set_option autoImplicit false

namespace tests.NebulaV2CarryEncoding

open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.CarryEncoding
open Nightstream.Protocol.NebulaV2.FPrime

def headers : ChainHeaders Nat := ⟨11, 22⟩
def closed : ClosedCarry Nat := ⟨3, 17, 99⟩

def canonical : WireCarry Nat Rat := encodeClosed headers closed

def noncanonicalProducts : WireCarry Nat Rat :=
  { canonical with
    products := fun _ =>
      { initialSnapshot := 0
        writes := 1
        reads := 1
        finalSnapshot := 1 } }

def noncanonicalChallenge : WireCarry Nat Rat :=
  { canonical with
    challenges := fun _ => { gamma1 := 1, gamma2 := 0 } }

theorem canonical_closed_round_trip :
    Decodes headers canonical (.closed closed) :=
  decodes_encodeClosed headers closed

/-- Closed product accumulators cannot start at a prover-selected value. -/
theorem noncanonical_closed_products_rejected :
    ¬ Decodes headers noncanonicalProducts (.closed closed) := by
  intro decoded
  have exactWire := closed_decodes_exact decoded
  have productCoordinate := congrArg
    (fun wire => (wire.products (0 : Fin 2)).initialSnapshot) exactWire
  norm_num [noncanonicalProducts, canonical, encodeClosed,
    ProductState.one] at productCoordinate

/-- Closed challenge coordinates cannot carry data between segments. -/
theorem noncanonical_closed_challenge_rejected :
    ¬ Decodes headers noncanonicalChallenge (.closed closed) := by
  intro decoded
  have exactWire := closed_decodes_exact decoded
  have challengeCoordinate := congrArg
    (fun wire => (wire.challenges (0 : Fin 2)).gamma1) exactWire
  norm_num [noncanonicalChallenge, canonical, encodeClosed,
    zeroChallenges] at challengeCoordinate

end tests.NebulaV2CarryEncoding
