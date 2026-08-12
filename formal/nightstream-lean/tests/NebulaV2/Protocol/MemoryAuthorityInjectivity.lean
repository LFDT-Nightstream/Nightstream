import Nightstream.Implementation.NebulaV2.Memory.Transition.OpenSegmentSound

/-! Regression surface for lossless challenge-authority placement. -/

namespace tests.NebulaV2MemoryAuthorityInjectivity

open Nightstream.Implementation.NebulaV2

#check MemoryOpenSegment.Authority.digestFields_injective
#check MemoryOpenSegmentSound.AuthorityPlaced.unique

end tests.NebulaV2MemoryAuthorityInjectivity
