import Nightstream.Implementation.NebulaV2.Memory.Claim.BoundCcsPublic

/-! Focused gates for the 540-coordinate state-and-memory carrier. -/

set_option autoImplicit false

namespace tests.NebulaV2MemoryBoundCcsPublic

open Nightstream.Implementation.NebulaV2
open Nightstream.Protocol.NebulaV2

example : MemoryBoundCcsPublic.coordinateCount = 540 := rfl
example : MemoryBoundCcsPublic.ringColumnCount = 10 := rfl
example : MemoryBoundCcsPublic.paddingBitCount = 27 := rfl

#check MemoryBoundCcsPublic.stateLaneWord_eq
#check MemoryBoundCcsPublic.memoryLaneWord_eq
#check MemoryBoundCcsPublic.encode_get_stateDigest
#check MemoryBoundCcsPublic.encode_get_memoryDigest
#check MemoryBoundCcsPublic.encode_get_padding
#check MemoryBoundCcsPublic.word_memoryMatches
#check MemoryBoundCcsPublic.matched_memory_eq_or_collision
#check MemoryBoundCcsPublic.authority_eq_or_memory_collision

end tests.NebulaV2MemoryBoundCcsPublic
