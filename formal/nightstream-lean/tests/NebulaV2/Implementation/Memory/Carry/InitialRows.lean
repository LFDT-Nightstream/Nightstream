import Nightstream.Implementation.NebulaV2.Memory.Carry.InitialRows

/-! Compile gate for the authoritative Nebula V2 chain-start carry rows. -/

namespace tests.NebulaV2InitialMemoryCarryRows

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.R1CS

example (layout : InitialMemoryCarryRows.Layout) :
    (InitialMemoryCarryRows.rows layout).length = 7 :=
  InitialMemoryCarryRows.rows_length_exact layout

#check InitialMemoryCarryRows.expectedValue_canonical
#check InitialMemoryCarryRows.Exact.value_eq_expected
#check InitialMemoryCarryRows.sound_value_eq_expected

end tests.NebulaV2InitialMemoryCarryRows
