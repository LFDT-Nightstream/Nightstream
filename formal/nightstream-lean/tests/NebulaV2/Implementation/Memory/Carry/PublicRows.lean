import Nightstream.Implementation.NebulaV2.Memory.Carry.PublicRows

/-! Focused regressions for the complete 3,433-bit recursive-carry bridge. -/

namespace NightstreamTests.NebulaV2MemoryCarryPublicRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.MemoryCarryCodec
open Nightstream.Implementation.NebulaV2.MemoryCarryPublicRows
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime

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

end NightstreamTests.NebulaV2MemoryCarryPublicRows
