import Nightstream.Implementation.NebulaV2.ProductionMemoryBoundCcsPublic

/-! Regression surface for the successor batch CCS public carrier. -/

set_option autoImplicit false

namespace tests.NebulaV2ProductionMemoryBoundCcsPublic

open Nightstream.Implementation.NebulaV2.ProductionMemoryBoundCcsPublic
open Nightstream.Implementation.NebulaV2.MemoryBoundCcsPublic

#check word_memoryMatches
#check matched_batch_eq_or_collision
#check authority_eq_or_memory_collision

example : coordinateCount = 540 := by decide

end tests.NebulaV2ProductionMemoryBoundCcsPublic
