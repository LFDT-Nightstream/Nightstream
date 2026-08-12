import Nightstream.Implementation.NebulaV2.MemoryClaimRows

/-! Focused regressions for the complete 4,980-bit fresh-claim row bridge. -/

namespace NightstreamTests.NebulaV2MemoryClaimRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.MemoryClaimRows
open Nightstream.Implementation.NebulaV2.MemoryClaimCodec

example (layout : Layout) : (rows layout).length = 10244 :=
  rows_length_exact layout

example {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryClaimParser.Block} {claim : Claim}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed layout.publicBits assignment block)
    (holds : Satisfies (rows layout) assignment)
    (accepted : MemoryClaimParser.parse block = some claim) :
    ParsedColumnsMatch layout assignment claim :=
  parsed_columns_match canonical one placed holds accepted

end NightstreamTests.NebulaV2MemoryClaimRows
