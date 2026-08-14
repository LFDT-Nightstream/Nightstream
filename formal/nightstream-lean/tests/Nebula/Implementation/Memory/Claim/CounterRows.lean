import Nightstream.Implementation.Nebula.Memory.Claim.CounterRows
import tests.Nebula.Implementation.Memory.Claim.Codec

set_option autoImplicit false

namespace tests.NebulaMemoryClaimCounterRows

open Nightstream.Implementation.Nebula.MemoryClaimCounterRows
open Nightstream.Implementation.R1CS

def layout : Layout :=
  { publicBitStart := 100
    valueColumn := fun counter =>
      match counter with
      | .segmentIndex => 1
      | .stepIndex => 2
      | .timestampIn => 3
      | .timestampOut => 4
      | .segmentStartTimestamp => 5
      | .segmentEndTimestamp => 6
      | .activeAccessCount => 7 }

theorem exact_counter_row_count : (rows layout).length = 123 :=
  rows_length_exact layout

/-- Counter bounds in the semantic claim follow from the row block and value
placement. The test does not pass `Claim.Canonical` as an assumption. -/
theorem rows_derive_claim_canonical
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : Placed layout assignment
      tests.NebulaMemoryClaimCodec.claim)
    (holds : Satisfies (rows layout) assignment) :
    tests.NebulaMemoryClaimCodec.claim.Canonical :=
  claim_canonical_of_rows canonical one placed holds

end tests.NebulaMemoryClaimCounterRows
