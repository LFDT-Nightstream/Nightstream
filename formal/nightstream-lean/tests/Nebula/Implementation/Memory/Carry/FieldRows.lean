import Nightstream.Implementation.Nebula.Memory.Carry.FieldRows

/-! Focused regressions for all 52 recursive-carry field limbs. -/

namespace NightstreamTests.NebulaMemoryCarryFieldRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.Nebula.MemoryCarryFieldRows

example : Slot.all.length = 52 := Slot.all_length_exact

example : (Slot.challenge 0 0 0).bitOffset = 105 := first_bit_exact

example : (Slot.root .memory 3).bitOffset + 64 = 3433 :=
  last_bit_end_exact

example (layout : Layout) : (rows layout).length = 6916 :=
  rows_length_exact layout

example {layout : Layout} {assignment : Nat → Nat} {raw : RawWords}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies (rows layout) assignment)
    (placed : Places layout assignment raw)
    (aliasEq : raw (.root .memory 3) =
      Nightstream.Protocol.Nebula.CanonicalFieldBits.modulusWord) : False :=
  modulus_alias_impossible canonical one satisfies placed _ aliasEq

end NightstreamTests.NebulaMemoryCarryFieldRows
