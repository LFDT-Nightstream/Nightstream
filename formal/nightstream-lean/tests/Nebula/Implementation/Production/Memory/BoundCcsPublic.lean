import Nightstream.Implementation.Nebula.Production.Memory.BoundCcsPublic

/-! Regression surface for the successor batch CCS public carrier. -/

set_option autoImplicit false

namespace tests.NebulaProductionMemoryBoundCcsPublic

open Nightstream.Implementation.Nebula.ProductionMemoryBoundCcsPublic
open Nightstream.Implementation.Nebula.MemoryBoundCcsPublic

#check word_memoryMatches
#check matched_batch_eq_or_collision
#check authority_eq_or_memory_collision

example : coordinateCount = 540 := by decide

end tests.NebulaProductionMemoryBoundCcsPublic
