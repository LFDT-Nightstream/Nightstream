import Nightstream.Implementation.NebulaV2.Memory.Claim.FieldRows

/-! Focused regressions for all 76 fresh-claim field limbs. -/

namespace NightstreamTests.NebulaV2MemoryClaimFieldRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.NebulaV2.MemoryClaimFieldRows

example : Slot.all.length = 76 := Slot.all_length_exact

example : (Slot.challenge 0 0 0).bitOffset = 116 := first_bit_exact

example :
    (Slot.root .seenAfter .finalSnapshot 3).bitOffset + 64 = 4980 :=
  last_bit_end_exact

example (layout : Layout) : (rows layout).length = 10108 :=
  rows_length_exact layout

example {layout : Layout} {assignment : Nat → Nat} {raw : RawWords}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment)
    (placed : Places layout assignment raw)
    (aliasEq : raw (.challenge 0 0 0) =
      Nightstream.Protocol.NebulaV2.CanonicalFieldBits.modulusWord) : False :=
  modulus_alias_impossible canonical one satisfies placed _ aliasEq

end NightstreamTests.NebulaV2MemoryClaimFieldRows
