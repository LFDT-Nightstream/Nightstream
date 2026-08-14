import Nightstream.Protocol.Nebula.MemoryWireGeometry

set_option autoImplicit false

namespace tests.NebulaMemoryWireGeometry

open Nightstream.Protocol.Nebula.MemoryWireGeometry

example :
    stepPublicBits = 4980 ∧
    stepPublicRingWidth = 5022 ∧
    stepPublicRingPadding = 41 ∧
    carryBits = 3433 ∧
    mandatoryBundleBits = 248832 := by
  exact ⟨stepPublicBits_exact, stepPublicRingWidth_exact,
    stepPublicRingPadding_exact, carryBits_exact, mandatoryBundleBits_exact⟩

end tests.NebulaMemoryWireGeometry
