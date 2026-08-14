import Nightstream.Implementation.Nebula.Memory.Carry.Rows
import tests.Nebula.Implementation.Memory.Carry.Codec

set_option autoImplicit false

namespace tests.NebulaMemoryCarryRows

open Nightstream.Implementation.Nebula.MemoryCarryRows
open Nightstream.Implementation.Nebula.MemoryCarryCodec
open Nightstream.Implementation.R1CS

def layout : Layout :=
  { publicBitStart := 1000
    fieldColumn := fun tag => 10 + tag.bitWidth
    stepSlackColumn := 200
    stepSlackBitStart := 300
    zeroColumn := 400
    headerColumn := fun role lane =>
      500 +
        (match role with
         | .operations => 0
         | .initialSnapshot => 4
         | .finalSnapshot => 8) + lane.val }

theorem exact_carry_row_count : (rows layout).length = 178 :=
  rows_length_exact layout

/-- Closed canonicality is a row conclusion. It is not passed to the theorem
as an assumption. -/
theorem rows_derive_closed_canonicality
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : Placed layout assignment
      tests.NebulaMemoryCarryCodec.closedValue)
    (headersPlaced : HeadersPlaced layout assignment
      tests.NebulaMemoryCarryCodec.headers)
    (holds : Satisfies (rows layout) assignment) :
    tests.NebulaMemoryCarryCodec.closedValue.Canonical
      tests.NebulaMemoryCarryCodec.headers :=
  value_canonical_of_rows canonical one placed headersPlaced holds

end tests.NebulaMemoryCarryRows
