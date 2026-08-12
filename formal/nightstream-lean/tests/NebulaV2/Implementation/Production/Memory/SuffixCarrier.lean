import Nightstream.Implementation.NebulaV2.Production.Memory.SuffixCarrier

/-! Regression surface for the mixed successor memory-suffix carrier. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionMemorySuffixCarrier

open Nightstream.Implementation.NebulaV2
open Nightstream.Implementation.NebulaV2.ProductionMemorySuffixCarrier

#check stepImage_injective_on_canonical
#check batchImage_injective_on_canonical
#check batch_coordinate_count

/-- Without the bounded-counter predicate, a seven-bit counter image aliases
zero and 128. -/
example : WasmStateCodec.encodeWord 7 0 = WasmStateCodec.encodeWord 7 128 /\
    (0 : Nat) ≠ 128 := by decide

end tests.NebulaV2ProductionMemorySuffixCarrier
