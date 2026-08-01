import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.Memory

set_option autoImplicit false

namespace tests.WasmBenchmark42x6Memory

open Nightstream.Protocol.Nebula.Fingerprint
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.Memory

theorem exact_trace_is_balanced (challengeValues : Challenges) :
    Nightstream.Protocol.Nebula.Memory.Balanced
      (Nightstream.Protocol.Nebula.Memory.products challengeValues
        initialSnapshot [access] finalSnapshot) :=
  balanced challengeValues

end tests.WasmBenchmark42x6Memory
