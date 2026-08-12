import Nightstream.Implementation.NebulaV2.Memory.Carry.HashPackingRows

/-! Focused regressions for the exact carry-to-hash-word row bridge. -/

namespace NightstreamTests.NebulaV2MemoryCarryHashPackingRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.MemoryCarryHashFrame
open Nightstream.Implementation.NebulaV2.MemoryCarryHashPackingRows

example (layout : Layout) : (rows layout).length = 131 :=
  rows_length_exact layout

example {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryCarryParser.Block}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed layout.publicBits assignment block)
    (holds : Satisfies (rows layout) assignment) :
    packedColumnValues layout assignment = encodePacked block :=
  packed_columns_eq_encodePacked canonical one placed holds

example {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryCarryParser.Block}
    (one : assignment 0 = 1)
    (honest : Honest layout assignment block) :
    Satisfies (rows layout) assignment :=
  rows_complete one honest

end NightstreamTests.NebulaV2MemoryCarryHashPackingRows
