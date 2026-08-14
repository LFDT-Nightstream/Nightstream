import Nightstream.Implementation.Nebula.Memory.Carry.InitialRows

/-! Compile gate for the authoritative Nebula V2 chain-start carry rows. -/

namespace tests.NebulaInitialMemoryCarryRows

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.R1CS

example (layout : InitialMemoryCarryRows.Layout) :
    (InitialMemoryCarryRows.rows layout).length = 7 :=
  InitialMemoryCarryRows.rows_length_exact layout

#check InitialMemoryCarryRows.expectedValue_canonical
#check InitialMemoryCarryRows.Exact.value_eq_expected
#check InitialMemoryCarryRows.sound_value_eq_expected

end tests.NebulaInitialMemoryCarryRows
