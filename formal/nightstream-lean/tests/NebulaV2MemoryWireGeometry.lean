import Nightstream.Protocol.NebulaV2.MemoryWireGeometry

set_option autoImplicit false

namespace tests.NebulaV2MemoryWireGeometry

open Nightstream.Protocol.NebulaV2.MemoryWireGeometry

example :
    stepPublicBits = 4980 ∧
    stepPublicRingWidth = 5022 ∧
    stepPublicRingPadding = 41 ∧
    carryBits = 3433 ∧
    mandatoryBundleBits = 248832 := by
  exact ⟨stepPublicBits_exact, stepPublicRingWidth_exact,
    stepPublicRingPadding_exact, carryBits_exact, mandatoryBundleBits_exact⟩

end tests.NebulaV2MemoryWireGeometry
