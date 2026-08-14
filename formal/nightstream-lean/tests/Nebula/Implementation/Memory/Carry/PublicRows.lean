import Nightstream.Implementation.Nebula.Memory.Carry.PublicRows

/-! Focused regressions for the complete 3,433-bit recursive-carry bridge. -/

namespace NightstreamTests.NebulaMemoryCarryPublicRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.MemoryCarryCodec
open Nightstream.Implementation.Nebula.MemoryCarryPublicRows
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.FPrime

example (layout : Layout) : (rows layout).length = 7094 :=
  rows_length_exact layout

example {layout : Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {block : MemoryCarryParser.Block} {value : Value}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed layout.publicBits assignment block)
    (headersPlaced : MemoryCarryRows.HeadersPlaced layout.carry assignment
      headers)
    (holds : Satisfies (rows layout) assignment)
    (accepted : MemoryCarryParser.parse headers block = some value) :
    ParsedColumnsMatch layout assignment headers value :=
  parsed_columns_match canonical one placed headersPlaced holds accepted

example {layout : Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {block : MemoryCarryParser.Block}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed layout.publicBits assignment block)
    (headersPlaced : MemoryCarryRows.HeadersPlaced layout.carry assignment
      headers)
    (holds : Satisfies (rows layout) assignment) :
    MemoryCarryParser.parse headers block =
      some (MemoryCarryParser.decodedValue block) :=
  rows_force_parse canonical one placed headersPlaced holds

example {layout : Layout} {assignment : Nat → Nat}
    {headers : ChainHeaders Digest.Value}
    {block : MemoryCarryParser.Block}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed layout.publicBits assignment block)
    (headersPlaced : MemoryCarryRows.HeadersPlaced layout.carry assignment
      headers)
    (holds : Satisfies (rows layout) assignment) :
    ParsedColumnsMatch layout assignment headers
      (MemoryCarryParser.decodedValue block) :=
  rows_force_parsed_columns_match canonical one placed headersPlaced holds

end NightstreamTests.NebulaMemoryCarryPublicRows
